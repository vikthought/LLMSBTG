"""
Phase-2 analysis script: per-(pe_type, seed) score-geometry + baselines.

Each invocation processes ONE (pe_type, seed) pair and writes:
    <out_dir>/<pe_type>_seed<seed>_analysis.json

Optionally saves score models (needed for null-permutation tests):
    <out_dir>/score_models/<pe_type>_seed<seed>/layer_<l>.pt

The cluster script calls this in parallel (one process per GPU), then
scripts/aggregate_positional_results.py merges everything.
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
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca_layer(train_acts_l, pca_dim):
    """train_acts_l: (N, seq_len, hidden_size)  →  (pca, mu_l)"""
    N, seq_len, hidden_size = train_acts_l.shape
    flat   = train_acts_l.reshape(-1, hidden_size)
    mu_l   = flat.mean(axis=0)
    pca    = PCA(n_components=pca_dim)
    pca.fit(flat)
    return pca, mu_l


def transform_pca_layer(pca, acts_l):
    """acts_l: (N, seq_len, hidden_size)  →  (N, seq_len, pca_dim)"""
    N, seq_len, hidden_size = acts_l.shape
    flat = acts_l.reshape(-1, hidden_size)
    return pca.transform(flat).reshape(N, seq_len, -1)


# ---------------------------------------------------------------------------
# Linear position probe
# ---------------------------------------------------------------------------

def compute_linear_probe_accuracy(
    train_acts_l: np.ndarray,
    test_acts_l:  np.ndarray,
    max_train_samples: int = 40_000,
) -> float:
    """
    Train a linear classifier to predict token position from the activation
    at that position.  Returns top-1 accuracy on the test set.

    Acts are (N, seq_len, hidden_size).  Position labels are 0 … seq_len-1.
    """
    N_train, seq_len, D = train_acts_l.shape
    N_test               = test_acts_l.shape[0]

    X_train = train_acts_l.reshape(-1, D)
    y_train = np.tile(np.arange(seq_len), N_train)

    X_test  = test_acts_l.reshape(-1, D)
    y_test  = np.tile(np.arange(seq_len), N_test)

    # Sub-sample for speed
    rng = np.random.default_rng(0)
    if len(X_train) > max_train_samples:
        idx     = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    probe = LogisticRegression(
        max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1
    )
    probe.fit(X_train, y_train)
    return float(probe.score(X_test, y_test))


# ---------------------------------------------------------------------------
# DSM validation loss helper
# ---------------------------------------------------------------------------

def compute_dsm_val_loss(score_model, val_windows_flat: np.ndarray, sigma: float,
                         device: str, batch_size: int = 512) -> float:
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


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals  (B4)
# ---------------------------------------------------------------------------

def _compute_bootstrap_ci(
    score_model,
    test_windows: np.ndarray,   # (num_seqs, num_windows, m*w)
    pca_dim: int,
    w: int,
    max_lag: int,
    skip_edges: int,
    n_bootstrap: int,
    rng_seed: int,
    device: str,
    point_estimates: dict = None,
    batch_size: int = 512,
) -> dict:
    """Bootstrap 95% CIs for RDI, H_r, A_r, beta.

    Uses pivot (normal) CI when ``point_estimates`` is provided:
    point_estimate +/- 1.96 * bootstrap_SE.  This avoids the downward
    bias of percentile bootstrap on ratio statistics (like RDI) when
    n_test is small.  Falls back to percentile CI otherwise.
    """
    num_seqs, num_windows, mw = test_windows.shape

    # Run score model once on all test windows
    score_model.eval()
    with torch.no_grad():
        flat = torch.tensor(test_windows.reshape(-1, mw), dtype=torch.float32)
        outs = []
        for i in range(0, len(flat), batch_size):
            outs.append(score_model(flat[i:i + batch_size].to(device)).cpu().numpy())
        scores_flat = np.concatenate(outs, axis=0)  # (N*W, m*w)

    scores = scores_flat.reshape(num_seqs, num_windows, w, pca_dim)  # (N, W, w, m)

    rng = np.random.default_rng(rng_seed)
    valid = slice(skip_edges, num_windows - skip_edges)

    rdi_vals  = []
    hr_vals   = []
    ar_vals   = []
    beta_vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, num_seqs, size=num_seqs)
        s   = scores[idx]   # (N, W, w, m)

        M_r_i = np.zeros((max_lag + 1, num_windows, pca_dim, pca_dim))
        for lag in range(max_lag + 1):
            if w - lag - 1 >= 0:
                s_cur = s[:, :, -1, :]         # (N, W, m)
                s_lag = s[:, :, w - lag - 1, :] # (N, W, m)
                M_r_i[lag] = -np.einsum("nwi,nwj->wij", s_cur, s_lag) / num_seqs

        M_r_valid = M_r_i[:, valid, :, :]
        M_bar     = M_r_valid.mean(axis=1)                      # (max_lag+1, m, m)
        Delta     = M_r_valid - M_bar[:, None, :, :]

        A     = np.linalg.norm(M_bar, ord="fro", axis=(1, 2))
        H     = (np.mean(np.linalg.norm(Delta, ord="fro", axis=(2, 3)) ** 2, axis=1)
                 / (A ** 2 + 1e-8))

        num_r = np.sum(np.mean(np.linalg.norm(Delta, ord="fro", axis=(2, 3)) ** 2, axis=1))
        den_r = np.sum(np.mean(np.linalg.norm(M_r_valid, ord="fro", axis=(2, 3)) ** 2, axis=1)) + 1e-8
        rdi   = 1.0 - num_r / den_r

        log_A = np.log(A[1:] + 1e-8)
        lags  = np.arange(1, max_lag + 1)
        beta  = float(np.polyfit(lags, log_A, 1)[0]) if max_lag > 0 else 0.0

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
        "RDI_ci": {
            "lower": float(np.percentile(rdi_arr, 2.5)),
            "upper": float(np.percentile(rdi_arr, 97.5)),
            "mean":  float(rdi_arr.mean()),
            "method": "percentile",
        },
        "H_r_ci": {
            "lower": np.percentile(hr_arr, 2.5, axis=0).tolist(),
            "upper": np.percentile(hr_arr, 97.5, axis=0).tolist(),
            "mean":  hr_arr.mean(axis=0).tolist(),
            "method": "percentile",
        },
        "A_r_ci": {
            "lower": np.percentile(ar_arr, 2.5, axis=0).tolist(),
            "upper": np.percentile(ar_arr, 97.5, axis=0).tolist(),
            "mean":  ar_arr.mean(axis=0).tolist(),
            "method": "percentile",
        },
        "beta_ci": {
            "lower": float(np.percentile(beta_arr, 2.5)),
            "upper": float(np.percentile(beta_arr, 97.5)),
            "mean":  float(beta_arr.mean()),
            "method": "percentile",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run localized score-geometry analysis for one (pe_type, seed)."
    )
    parser.add_argument("--data-dir",   type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--out-dir",    type=str, required=True)
    parser.add_argument("--pe-types",   nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2],
                        help="Specific seed values to process (e.g. --seeds 0 1 2)")
    parser.add_argument("--w",          type=int, default=16)
    parser.add_argument("--max-lag",    type=int, default=5)
    parser.add_argument("--pca-dim",    type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Score model training epochs per layer")
    parser.add_argument("--score-tuning-trials", type=int, default=150,
                        help="Optuna trials for score model HP search (0 = skip tuning)")
    parser.add_argument("--tune-objective", type=str, default="null_contrast",
                        choices=["dsm", "null_contrast"],
                        help="Tuning objective: 'dsm' = val DSM loss, 'null_contrast' = NC ratio")
    parser.add_argument("--bootstrap-n", type=int, default=500,
                        help="Bootstrap iterations for per-layer 95%% CI (0 = skip)")
    parser.add_argument("--save-score-models", action="store_true",
                        help="Save fitted score model weights (required for null-permutation tests)")
    parser.add_argument("--device",     type=str,
                        default="cuda:0" if torch.cuda.is_available() else
                                ("mps"   if torch.backends.mps.is_available() else "cpu"))

    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    pca_dim = args.pca_dim

    for pe_type in args.pe_types:
        for seed in args.seeds:
            print(f"\n{'='*55}")
            print(f"  EVALUATING {pe_type.upper()} seed={seed}  pca_dim={pca_dim}")
            print(f"{'='*55}")

            seed_dir = models_dir / f"{pe_type}_seed{seed}"
            if not seed_dir.exists():
                print(f"  WARNING: {seed_dir} not found — skipping.")
                continue

            # ----------------------------------------------------------------
            # Load raw activations
            # ----------------------------------------------------------------
            train_acts = np.load(seed_dir / "train_acts.npy")
            test_acts  = np.load(seed_dir / "test_acts.npy")
            test_family_labels = np.load(seed_dir / "test_family_labels.npy")

            n_samples, n_layers_plus_one, seq_len, hidden_size = train_acts.shape

            # ----------------------------------------------------------------
            # Load attention statistics if available
            # ----------------------------------------------------------------
            attn_stats_path = seed_dir / "attn_stats.json"
            attn_stats = None
            if attn_stats_path.exists():
                with open(attn_stats_path) as f:
                    attn_stats = json.load(f)

            # ----------------------------------------------------------------
            # HP tuning on a representative layer (layer index = middle block)
            # ----------------------------------------------------------------
            best_hp = None
            if args.score_tuning_trials > 0:
                tune_layer = max(1, n_layers_plus_one // 2)
                print(f"\n  [HP tuning] using layer {tune_layer}  "
                      f"({args.score_tuning_trials} trials) …")
                pca_tune, _ = fit_pca_layer(train_acts[:, tune_layer], pca_dim)
                t_pca = transform_pca_layer(pca_tune, train_acts[:, tune_layer])

                val_split     = int(0.9 * n_samples)
                train_win     = extract_windows(t_pca[:val_split], args.w)
                val_win       = extract_windows(t_pca[val_split:], args.w)

                if args.tune_objective == "null_contrast":
                    best_hp = tune_score_model_null_contrast(
                        train_win.reshape(-1, pca_dim * args.w),
                        val_win.reshape(-1,   pca_dim * args.w),
                        in_features   = pca_dim * args.w,
                        m             = pca_dim,
                        w             = args.w,
                        n_trials      = args.score_tuning_trials,
                        device        = args.device,
                        seed          = seed,
                    )
                else:
                    best_hp = tune_score_model_hyperparams(
                        train_win.reshape(-1, pca_dim * args.w),
                        val_win.reshape(-1,   pca_dim * args.w),
                        in_features   = pca_dim * args.w,
                        n_trials      = args.score_tuning_trials,
                        device        = args.device,
                        seed          = seed,
                    )
            else:
                best_hp = {"sigma": 0.3, "hidden_dim": 256, "lr": 1e-3}

            # Optional: directory for score model weights
            score_models_dir = None
            if args.save_score_models:
                score_models_dir = out_dir / "score_models" / f"{pe_type}_seed{seed}"
                score_models_dir.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------------
            # Per-layer analysis
            # ----------------------------------------------------------------
            layer_results = []

            for l in range(1, n_layers_plus_one):
                print(f"\n  --- Layer {l}/{n_layers_plus_one - 1} ---")

                pca, mu_l = fit_pca_layer(train_acts[:, l], pca_dim)

                # PCA variance diagnostic: how much variance does m=32 capture?
                hidden_size = train_acts.shape[-1]
                max_diag_dim = min(128, hidden_size)
                if max_diag_dim > pca_dim:
                    pca_diag = PCA(n_components=max_diag_dim)
                    flat_diag = train_acts[:, l].reshape(-1, hidden_size)
                    pca_diag.fit(flat_diag)
                    cumvar = np.cumsum(pca_diag.explained_variance_ratio_)
                    checkpoints = [pca_dim, 64, 128]
                    for cp in checkpoints:
                        if cp <= max_diag_dim:
                            print(f"    PCA cumulative variance at m={cp}: {cumvar[cp-1]:.4f}")

                train_pca_l = transform_pca_layer(pca, train_acts[:, l])
                test_pca_l  = transform_pca_layer(pca, test_acts[:, l])

                # Score model: train / val split
                val_split_idx  = int(0.9 * n_samples)
                score_train    = train_pca_l[:val_split_idx]
                score_val      = train_pca_l[val_split_idx:]

                train_windows = extract_windows(score_train, args.w)
                val_windows   = extract_windows(score_val,   args.w)
                test_windows  = extract_windows(test_pca_l,  args.w)

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

                # DSM validation loss (D5 — report score model quality)
                val_dsm_loss = compute_dsm_val_loss(
                    score_model,
                    val_windows.reshape(-1, pca_dim * args.w),
                    sigma  = best_hp["sigma"],
                    device = args.device,
                )
                print(f"    val DSM loss = {val_dsm_loss:.5f}")

                # Save score model weights if requested
                if score_models_dir is not None:
                    torch.save(score_model.state_dict(),
                               score_models_dir / f"layer_{l}.pt")
                    # Also save the HP so the loader knows the architecture
                    with open(score_models_dir / f"layer_{l}_hp.json", "w") as f:
                        json.dump({
                            "best_hp":   best_hp,
                            "pca_dim":   pca_dim,
                            "w":         args.w,
                            "in_features": pca_dim * args.w,
                        }, f)

                estimator = LocalizedMultiBlockEstimator(
                    m=pca_dim, w=args.w, max_lag=args.max_lag
                )
                stats = estimator.estimate(score_model, test_windows, device=args.device)

                # Bootstrap CIs (B4)
                bootstrap_ci = None
                if args.bootstrap_n > 0:
                    print(f"    bootstrap CIs (n={args.bootstrap_n}) …")
                    bootstrap_ci = _compute_bootstrap_ci(
                        score_model   = score_model,
                        test_windows  = test_windows,
                        pca_dim       = pca_dim,
                        w             = args.w,
                        max_lag       = args.max_lag,
                        skip_edges    = estimator.skip_edges,
                        n_bootstrap   = args.bootstrap_n,
                        rng_seed      = seed * 1000 + l,
                        device        = args.device,
                        point_estimates={
                            "RDI": float(stats["RDI"]),
                            "H_r": stats["H_r"].tolist(),
                            "A_r": stats["A_r"].tolist(),
                            "beta": float(stats["beta"]),
                        },
                    )

                # ------------------------------------------------------------
                # Linear position probe (on raw PCA activations)
                # ------------------------------------------------------------
                print(f"    fitting linear probe …")
                probe_acc = compute_linear_probe_accuracy(
                    train_pca_l, test_pca_l
                )
                print(f"    probe acc = {probe_acc:.4f}")

                layer_entry = {
                    "layer": l,
                    "A_r":         stats["A_r"].tolist(),
                    "H_r":         stats["H_r"].tolist(),
                    "RDI":         float(stats["RDI"]),
                    "beta":        float(stats["beta"]),
                    "SCR_r":       stats["SCR_r"].tolist(),   # (max_lag+1, pca_dim)
                    "M_bar_r":     stats["M_bar_r"].tolist(),
                    "delta_frob":  stats["delta_frob"].tolist(),
                    "M_frob":      stats["M_frob"].tolist(),
                    "probe_accuracy":   probe_acc,
                    "val_dsm_loss":     val_dsm_loss,
                    "pca_components":   pca.components_.tolist(),
                    "pca_explained_var_ratio": pca.explained_variance_ratio_.tolist(),
                    "pca_cumvar_at_dims": {
                        str(cp): float(cumvar[cp-1])
                        for cp in [pca_dim, 64, 128]
                        if cp <= max_diag_dim
                    } if max_diag_dim > pca_dim else {},
                    "mu_l":             mu_l.tolist(),
                }
                if bootstrap_ci is not None:
                    layer_entry["bootstrap_ci"] = bootstrap_ci

                layer_results.append(layer_entry)

            # ----------------------------------------------------------------
            # Assemble result for this (pe_type, seed)
            # ----------------------------------------------------------------
            result = {
                "pe_type":    pe_type,
                "seed":       seed,
                "tune_objective": args.tune_objective,
                "pca_dim":    pca_dim,
                "w":          args.w,
                "max_lag":    args.max_lag,
                "best_hp":    best_hp,
                "layer_stats":       layer_results,
                "test_labels":       test_family_labels.tolist(),
                "attn_stats":        attn_stats,   # None if not available
            }

            out_path = out_dir / f"{pe_type}_seed{seed}_analysis.json"
            with open(out_path, "w") as f:
                def _default(obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, np.generic):  return obj.item()
                    raise TypeError(type(obj))
                json.dump(result, f, default=_default, indent=2)

            print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
