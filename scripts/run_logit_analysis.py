"""
Score-geometric analysis on PCA-reduced logit vectors.

Each invocation processes ONE (pe_type, seed) pair and writes:
    <out_dir>/<pe_type>_seed<seed>_logit_analysis.json

This is the logit-space analogue of run_positional_analysis.py.
Unlike hidden-state analysis (which loops over layers), logits are a single
projection — so there is one set of metrics per (pe_type, seed).
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.sbtg.estimators.localized_multiblock_trainer import (
    MinimalMLPScoreNet,
    extract_windows,
    train_score_model_layer,
    tune_score_model_hyperparams,
    tune_score_model_null_contrast,
)
from src.sbtg.estimators.localized_multiblock import LocalizedMultiBlockEstimator


# ---------------------------------------------------------------------------
# Helpers (same implementations as run_positional_analysis.py, inlined to
# avoid fragile cross-script imports on the cluster)
# ---------------------------------------------------------------------------

def fit_pca_layer(train_acts_l, pca_dim):
    """train_acts_l: (N, seq_len, D) -> (pca, mu_l)"""
    N, seq_len, D = train_acts_l.shape
    flat = train_acts_l.reshape(-1, D)
    mu_l = flat.mean(axis=0)
    pca  = PCA(n_components=pca_dim)
    pca.fit(flat)
    return pca, mu_l


def transform_pca_layer(pca, acts_l):
    """acts_l: (N, seq_len, D) -> (N, seq_len, pca_dim)"""
    N, seq_len, D = acts_l.shape
    flat = acts_l.reshape(-1, D)
    return pca.transform(flat).reshape(N, seq_len, -1)


def compute_linear_probe_accuracy(train_acts_l, test_acts_l, max_train_samples=40_000):
    """Linear probe to predict position from activation. Returns accuracy."""
    N_train, seq_len, D = train_acts_l.shape
    N_test = test_acts_l.shape[0]

    X_train = train_acts_l.reshape(-1, D)
    y_train = np.tile(np.arange(seq_len), N_train)
    X_test  = test_acts_l.reshape(-1, D)
    y_test  = np.tile(np.arange(seq_len), N_test)

    rng = np.random.default_rng(0)
    if len(X_train) > max_train_samples:
        idx     = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe.fit(X_train, y_train)
    return float(probe.score(X_test, y_test))


def compute_dsm_val_loss(score_model, val_windows_flat, sigma, device, batch_size=512):
    """Compute mean DSM loss on held-out validation windows."""
    score_model.eval()
    losses = []
    t = torch.tensor(val_windows_flat, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            z = t[i:i + batch_size].to(device)
            eps = torch.randn_like(z)
            z_noisy = z + sigma * eps
            s = score_model(z_noisy)
            target = -eps / sigma
            loss = ((s - target) ** 2).mean()
            losses.append(loss.item())
    return float(np.mean(losses))


def _compute_bootstrap_ci(score_model, test_windows, pca_dim, w, max_lag,
                           skip_edges, n_bootstrap, rng_seed, device,
                           point_estimates=None, batch_size=512):
    """Bootstrap 95% CIs for RDI, H_r, A_r, beta.

    Uses pivot (normal) CI when ``point_estimates`` is provided:
    point_estimate +/- 1.96 * bootstrap_SE.  This avoids the downward
    bias of percentile bootstrap on ratio statistics (like RDI) when
    n_test is small.  Falls back to percentile CI otherwise.
    """
    num_seqs, num_windows, mw = test_windows.shape

    score_model.eval()
    with torch.no_grad():
        flat = torch.tensor(test_windows.reshape(-1, mw), dtype=torch.float32)
        outs = []
        for i in range(0, len(flat), batch_size):
            outs.append(score_model(flat[i:i + batch_size].to(device)).cpu().numpy())
        scores_flat = np.concatenate(outs, axis=0)

    scores = scores_flat.reshape(num_seqs, num_windows, w, pca_dim)
    rng = np.random.default_rng(rng_seed)
    valid = slice(skip_edges, num_windows - skip_edges)

    rdi_vals, hr_vals, ar_vals, beta_vals = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, num_seqs, size=num_seqs)
        s = scores[idx]

        M_r_i = np.zeros((max_lag + 1, num_windows, pca_dim, pca_dim))
        for lag in range(max_lag + 1):
            if w - lag - 1 >= 0:
                s_cur = s[:, :, -1, :]
                s_lag = s[:, :, w - lag - 1, :]
                M_r_i[lag] = -np.einsum("nwi,nwj->wij", s_cur, s_lag) / num_seqs

        M_r_valid = M_r_i[:, valid, :, :]
        M_bar     = M_r_valid.mean(axis=1)
        Delta     = M_r_valid - M_bar[:, None, :, :]

        A = np.linalg.norm(M_bar, ord="fro", axis=(1, 2))
        H = (np.mean(np.linalg.norm(Delta, ord="fro", axis=(2, 3)) ** 2, axis=1)
             / (A ** 2 + 1e-8))

        num_r = np.sum(np.mean(np.linalg.norm(Delta, ord="fro", axis=(2, 3)) ** 2, axis=1))
        den_r = np.sum(np.mean(np.linalg.norm(M_r_valid, ord="fro", axis=(2, 3)) ** 2, axis=1)) + 1e-8
        rdi = 1.0 - num_r / den_r

        log_A = np.log(A[1:] + 1e-8)
        lags = np.arange(1, max_lag + 1)
        beta = float(np.polyfit(lags, log_A, 1)[0]) if max_lag > 0 else 0.0

        rdi_vals.append(rdi)
        hr_vals.append(H.tolist())
        ar_vals.append(A.tolist())
        beta_vals.append(beta)

    rdi_arr  = np.array(rdi_vals)
    hr_arr   = np.array(hr_vals)
    ar_arr   = np.array(ar_vals)
    beta_arr = np.array(beta_vals)

    if point_estimates is not None:
        pt_rdi  = point_estimates["RDI"]
        pt_hr   = np.array(point_estimates["H_r"])
        pt_ar   = np.array(point_estimates["A_r"])
        pt_beta = point_estimates["beta"]

        rdi_se  = float(np.std(rdi_arr))
        hr_se   = np.std(hr_arr, axis=0)
        ar_se   = np.std(ar_arr, axis=0)
        beta_se = float(np.std(beta_arr))

        return {
            "RDI_ci":  {"lower": float(pt_rdi - 1.96 * rdi_se),
                         "upper": float(pt_rdi + 1.96 * rdi_se),
                         "mean": float(pt_rdi), "se": rdi_se,
                         "method": "pivot"},
            "H_r_ci":  {"lower": (pt_hr - 1.96 * hr_se).tolist(),
                         "upper": (pt_hr + 1.96 * hr_se).tolist(),
                         "mean":  pt_hr.tolist(), "se": hr_se.tolist(),
                         "method": "pivot"},
            "A_r_ci":  {"lower": (pt_ar - 1.96 * ar_se).tolist(),
                         "upper": (pt_ar + 1.96 * ar_se).tolist(),
                         "mean":  pt_ar.tolist(), "se": ar_se.tolist(),
                         "method": "pivot"},
            "beta_ci": {"lower": float(pt_beta - 1.96 * beta_se),
                         "upper": float(pt_beta + 1.96 * beta_se),
                         "mean": float(pt_beta), "se": beta_se,
                         "method": "pivot"},
        }

    return {
        "RDI_ci":  {"lower": float(np.percentile(rdi_arr, 2.5)),
                     "upper": float(np.percentile(rdi_arr, 97.5)),
                     "mean":  float(rdi_arr.mean()),
                     "method": "percentile"},
        "H_r_ci":  {"lower": np.percentile(hr_arr, 2.5, axis=0).tolist(),
                     "upper": np.percentile(hr_arr, 97.5, axis=0).tolist(),
                     "mean":  hr_arr.mean(axis=0).tolist(),
                     "method": "percentile"},
        "A_r_ci":  {"lower": np.percentile(ar_arr, 2.5, axis=0).tolist(),
                     "upper": np.percentile(ar_arr, 97.5, axis=0).tolist(),
                     "mean":  ar_arr.mean(axis=0).tolist(),
                     "method": "percentile"},
        "beta_ci": {"lower": float(np.percentile(beta_arr, 2.5)),
                     "upper": float(np.percentile(beta_arr, 97.5)),
                     "mean":  float(beta_arr.mean()),
                     "method": "percentile"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run localized score-geometry analysis on logit vectors."
    )
    parser.add_argument("--models-dir", type=str, required=True,
                        help="Directory containing <pe_type>_seed<seed>/ with logit npy files")
    parser.add_argument("--out-dir",    type=str, required=True)
    parser.add_argument("--pe-types",   nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[0],
                        help="Seed values to process")
    parser.add_argument("--w",          type=int, default=16)
    parser.add_argument("--max-lag",    type=int, default=5)
    parser.add_argument("--pca-dim",    type=int, default=32,
                        help="Working PCA dim for score model (second-stage reduction)")
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Score model training epochs")
    parser.add_argument("--score-tuning-trials", type=int, default=60,
                        help="Optuna trials for score model HP search (0 = skip)")
    parser.add_argument("--tune-objective", type=str, default="null_contrast",
                        choices=["dsm", "null_contrast"],
                        help="Tuning objective: 'dsm' = val DSM loss, 'null_contrast' = NC ratio")
    parser.add_argument("--bootstrap-n", type=int, default=100,
                        help="Bootstrap iterations for 95%% CI (0 = skip)")
    parser.add_argument("--save-score-models", action="store_true",
                        help="Save fitted score model weights")
    parser.add_argument("--device",     type=str,
                        default="cuda:0" if torch.cuda.is_available() else
                                ("mps"   if torch.backends.mps.is_available() else "cpu"))

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_dim = args.pca_dim

    for pe_type in args.pe_types:
        for seed in args.seeds:
            print(f"\n{'='*55}")
            print(f"  LOGIT ANALYSIS: {pe_type.upper()} seed={seed}  pca_dim={pca_dim}")
            print(f"{'='*55}")

            seed_dir = models_dir / f"{pe_type}_seed{seed}"
            train_logits_path = seed_dir / "train_logits_pca.npy"
            test_logits_path  = seed_dir / "test_logits_pca.npy"

            if not train_logits_path.exists():
                print(f"  SKIP: {train_logits_path} not found")
                continue

            # Load PCA-reduced logits: (N, seq_len, logit_pca_dim)
            train_logits = np.load(train_logits_path)
            test_logits  = np.load(test_logits_path)

            n_samples, seq_len, logit_pca_dim = train_logits.shape
            print(f"  Loaded logits: train={train_logits.shape}, test={test_logits.shape}")

            # ----------------------------------------------------------------
            # Second-stage PCA: logit_pca_dim (256) -> working pca_dim (32)
            # ----------------------------------------------------------------
            pca, mu_l = fit_pca_layer(train_logits, pca_dim)
            train_pca = transform_pca_layer(pca, train_logits)
            test_pca  = transform_pca_layer(pca, test_logits)
            print(f"  Second-stage PCA: {logit_pca_dim} -> {pca_dim}  "
                  f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

            # ----------------------------------------------------------------
            # HP tuning
            # ----------------------------------------------------------------
            best_hp = None
            if args.score_tuning_trials > 0:
                print(f"\n  [HP tuning] ({args.score_tuning_trials} trials) ...")
                val_split = int(0.9 * n_samples)
                train_win = extract_windows(train_pca[:val_split], args.w)
                val_win   = extract_windows(train_pca[val_split:], args.w)

                if args.tune_objective == "null_contrast":
                    best_hp = tune_score_model_null_contrast(
                        train_win.reshape(-1, pca_dim * args.w),
                        val_win.reshape(-1,   pca_dim * args.w),
                        in_features  = pca_dim * args.w,
                        m            = pca_dim,
                        w            = args.w,
                        n_trials     = args.score_tuning_trials,
                        device       = args.device,
                        seed         = seed,
                    )
                else:
                    best_hp = tune_score_model_hyperparams(
                        train_win.reshape(-1, pca_dim * args.w),
                        val_win.reshape(-1,   pca_dim * args.w),
                        in_features  = pca_dim * args.w,
                        n_trials     = args.score_tuning_trials,
                        device       = args.device,
                        seed         = seed,
                    )
            else:
                best_hp = {"sigma": 0.3, "hidden_dim": 256, "lr": 1e-3}

            # ----------------------------------------------------------------
            # Score model training
            # ----------------------------------------------------------------
            val_split_idx = int(0.9 * n_samples)
            score_train   = train_pca[:val_split_idx]
            score_val     = train_pca[val_split_idx:]

            train_windows = extract_windows(score_train, args.w)
            val_windows   = extract_windows(score_val,   args.w)
            test_windows  = extract_windows(test_pca,    args.w)

            score_model = train_score_model_layer(
                train_windows, val_windows,
                m          = pca_dim,
                w          = args.w,
                epochs     = args.epochs,
                lr         = best_hp["lr"],
                sigma      = best_hp["sigma"],
                hidden_dim = best_hp["hidden_dim"],
                batch_size = 256,
                device     = args.device,
            )

            # DSM validation loss
            val_dsm_loss = compute_dsm_val_loss(
                score_model,
                val_windows.reshape(-1, pca_dim * args.w),
                sigma  = best_hp["sigma"],
                device = args.device,
            )
            print(f"    val DSM loss = {val_dsm_loss:.5f}")

            # Save score model weights if requested
            if args.save_score_models:
                sm_dir = out_dir / "score_models" / f"{pe_type}_seed{seed}"
                sm_dir.mkdir(parents=True, exist_ok=True)
                torch.save(score_model.state_dict(), sm_dir / "logit_score.pt")
                with open(sm_dir / "logit_score_hp.json", "w") as f:
                    json.dump({
                        "best_hp":     best_hp,
                        "pca_dim":     pca_dim,
                        "w":           args.w,
                        "in_features": pca_dim * args.w,
                    }, f)

            # ----------------------------------------------------------------
            # Score-geometric estimation
            # ----------------------------------------------------------------
            estimator = LocalizedMultiBlockEstimator(
                m=pca_dim, w=args.w, max_lag=args.max_lag
            )
            stats = estimator.estimate(score_model, test_windows, device=args.device)

            print(f"    RDI  = {stats['RDI']:.4f}")
            print(f"    H_r  = {[f'{h:.4f}' for h in stats['H_r']]}")
            print(f"    beta = {stats['beta']:.4f}")

            # Bootstrap CIs
            bootstrap_ci = None
            if args.bootstrap_n > 0:
                print(f"    bootstrap CIs (n={args.bootstrap_n}) ...")
                bootstrap_ci = _compute_bootstrap_ci(
                    score_model  = score_model,
                    test_windows = test_windows,
                    pca_dim      = pca_dim,
                    w            = args.w,
                    max_lag      = args.max_lag,
                    skip_edges   = estimator.skip_edges,
                    n_bootstrap  = args.bootstrap_n,
                    rng_seed     = seed * 1000,
                    device       = args.device,
                    point_estimates={
                        "RDI": float(stats["RDI"]),
                        "H_r": stats["H_r"].tolist(),
                        "A_r": stats["A_r"].tolist(),
                        "beta": float(stats["beta"]),
                    },
                )

            # Linear position probe on logits
            print(f"    fitting linear probe on logit space ...")
            probe_acc = compute_linear_probe_accuracy(train_pca, test_pca)
            print(f"    probe acc = {probe_acc:.4f}")

            # ----------------------------------------------------------------
            # Assemble result
            # ----------------------------------------------------------------
            result = {
                "pe_type":      pe_type,
                "seed":         seed,
                "space":        "logit",
                "tune_objective": args.tune_objective,
                "pca_dim":      pca_dim,
                "logit_pca_dim": int(logit_pca_dim),
                "w":            args.w,
                "max_lag":      args.max_lag,
                "best_hp":      best_hp,
                "A_r":          stats["A_r"].tolist(),
                "H_r":          stats["H_r"].tolist(),
                "RDI":          float(stats["RDI"]),
                "beta":         float(stats["beta"]),
                "SCR_r":        stats["SCR_r"].tolist(),
                "M_bar_r":      stats["M_bar_r"].tolist(),
                "delta_frob":   stats["delta_frob"].tolist(),
                "M_frob":       stats["M_frob"].tolist(),
                "probe_accuracy":  probe_acc,
                "val_dsm_loss":    val_dsm_loss,
                "pca_explained_var_ratio": pca.explained_variance_ratio_.tolist(),
            }
            if bootstrap_ci is not None:
                result["bootstrap_ci"] = bootstrap_ci

            out_path = out_dir / f"{pe_type}_seed{seed}_logit_analysis.json"
            with open(out_path, "w") as f:
                def _default(obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, np.generic): return obj.item()
                    raise TypeError(type(obj))
                json.dump(result, f, default=_default, indent=2)

            print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
