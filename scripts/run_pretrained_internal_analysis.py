"""
Score-geometric analysis on extracted pretrained-model activations.

Reads the per-model directory produced by `extract_pretrained_internals.py`:

    <pretrained_dir>/<model_key>/
        metadata.json
        logit_pca.npy              optional (--mode in {logit, both})
        hidden_L<N>_pca.npy        optional (--mode in {internal, both})

For each (model, cell) where cell ∈ {logit} ∪ {hidden_L<N>}, runs the
matching pipeline used elsewhere in paper3:

    train/test split (default 80/20, deterministic)
    extract overlapping width-w windows
    Optuna null-contrast tuning of the score model (configurable trials)
    train DSM score model on training windows
    LocalizedMultiBlockEstimator → A_r, S_r, C_r, AS_r, SCR_r, M_bar_r, ...
    pivot-bootstrap CIs on RDI / A_r / H_r / beta
    linear position probe accuracy on the same PCA-reduced features

Writes one JSON per cell:

    <out_dir>/<model_key>/<cell>_metrics.json
        cell ∈ {"logit", "hidden_L1", "hidden_L12", ...}

The structure exactly mirrors run_pretrained_logit_analysis.py's output
schema with the additional fields {space, layer_index, layer_relative_depth,
family, training_corpus, tokenizer_family} for cross-model aggregation.

Usage
-----
    python scripts/run_pretrained_internal_analysis.py \\
        --pretrained-dir results/pretrained_size_sweep \\
        --models pythia-1.4b \\
        --modes logit hidden \\
        --out-dir results/pretrained_size_sweep_analysis \\
        --tuning-trials 60 --epochs 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.estimators.localized_multiblock import LocalizedMultiBlockEstimator
from src.sbtg.estimators.localized_multiblock_trainer import (
    extract_windows,
    train_score_model_layer,
    tune_score_model_null_contrast,
)


# ============================================================================
# Helpers
# ============================================================================


def _safe_key(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_key)


def fit_pca_on_train(train_acts: np.ndarray, pca_dim: int) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on flattened train activations (leak-free for analysis)."""
    N, seq_len, D = train_acts.shape
    flat = train_acts.reshape(-1, D)
    mu = flat.mean(axis=0)
    pca = PCA(n_components=pca_dim)
    pca.fit(flat)
    return pca, mu


def transform_pca(pca: PCA, acts: np.ndarray) -> np.ndarray:
    N, seq_len, D = acts.shape
    return pca.transform(acts.reshape(-1, D)).reshape(N, seq_len, -1)


def linear_position_probe(
    train_acts: np.ndarray,
    test_acts: np.ndarray,
    max_train: int = 40_000,
    rng_seed: int = 0,
) -> float:
    N_tr, seq_len, D = train_acts.shape
    N_te = test_acts.shape[0]
    Xtr = train_acts.reshape(-1, D)
    ytr = np.tile(np.arange(seq_len), N_tr)
    Xte = test_acts.reshape(-1, D)
    yte = np.tile(np.arange(seq_len), N_te)
    rng = np.random.default_rng(rng_seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    probe = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe.fit(Xtr, ytr)
    return float(probe.score(Xte, yte))


def bootstrap_pivot_cis(
    score_model,
    test_windows: np.ndarray,
    pca_dim: int,
    w: int,
    max_lag: int,
    skip_edges: int,
    n_bootstrap: int,
    point_estimates: Dict[str, Any],
    device: str,
    rng_seed: int = 0,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Pivot-bootstrap CIs (matches run_pretrained_logit_analysis convention)."""
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

    rdi_b, ar_b, beta_b = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, num_seqs, size=num_seqs)
        s = scores[idx]
        M_r_i = np.zeros((max_lag + 1, num_windows, pca_dim, pca_dim))
        for lag in range(max_lag + 1):
            if w - lag - 1 >= 0:
                s_cur = s[:, :, -1, :]
                s_lag = s[:, :, w - lag - 1, :]
                M_r_i[lag] = -np.einsum("nwi,nwj->wij", s_cur, s_lag) / num_seqs
        M_v = M_r_i[:, valid]
        M_bar = M_v.mean(axis=1)
        Delta = M_v - M_bar[:, None]
        A = np.linalg.norm(M_bar, ord="fro", axis=(1, 2))
        num_r = np.sum(np.mean(np.linalg.norm(Delta, ord="fro", axis=(2, 3)) ** 2, axis=1))
        den_r = np.sum(np.mean(np.linalg.norm(M_v, ord="fro", axis=(2, 3)) ** 2, axis=1)) + 1e-8
        rdi_b.append(1.0 - num_r / den_r)
        ar_b.append(A.tolist())
        log_A = np.log(A[1:] + 1e-8)
        lags = np.arange(1, max_lag + 1)
        beta_b.append(float(np.polyfit(lags, log_A, 1)[0]) if max_lag > 0 else 0.0)

    rdi_b = np.array(rdi_b); ar_b = np.array(ar_b); beta_b = np.array(beta_b)
    pt_rdi = point_estimates["RDI"]
    pt_ar = np.array(point_estimates["A_r"])
    pt_beta = point_estimates["beta"]

    rdi_se = float(np.std(rdi_b))
    ar_se = np.std(ar_b, axis=0)
    beta_se = float(np.std(beta_b))
    return {
        "RDI_ci": {"lower": float(pt_rdi - 1.96 * rdi_se),
                   "upper": float(pt_rdi + 1.96 * rdi_se),
                   "mean": float(pt_rdi), "se": rdi_se, "method": "pivot"},
        "A_r_ci": {"lower": (pt_ar - 1.96 * ar_se).tolist(),
                   "upper": (pt_ar + 1.96 * ar_se).tolist(),
                   "mean": pt_ar.tolist(), "se": ar_se.tolist(), "method": "pivot"},
        "beta_ci": {"lower": float(pt_beta - 1.96 * beta_se),
                    "upper": float(pt_beta + 1.96 * beta_se),
                    "mean": float(pt_beta), "se": beta_se, "method": "pivot"},
    }


# ============================================================================
# Per-cell analysis
# ============================================================================


def analyze_cell(
    activations: np.ndarray,             # (N, seq_len, D_in)
    cell_label: str,                     # "logit" or "hidden_L<N>"
    args: argparse.Namespace,
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """One score-geometric analysis run on a (N, seq_len, D_in) tensor.

    D_in is whatever the extractor saved (typically inner_pca_dim=32, but
    we re-fit a leak-free PCA on the train split so we don't depend on
    exact match)."""
    N, seq_len, D_in = activations.shape
    print(f"\n  [{cell_label}]  input shape: {activations.shape}")

    n_train = int(N * args.train_frac)
    train_a = activations[:n_train]
    test_a = activations[n_train:]
    print(f"    train/test split: {n_train} / {N - n_train}")

    pca, mu_l = fit_pca_on_train(train_a, args.pca_dim)
    train_pca = transform_pca(pca, train_a)
    test_pca = transform_pca(pca, test_a)
    pca_evr = float(pca.explained_variance_ratio_.sum())
    print(f"    PCA fit (train-only): {D_in} → {args.pca_dim}, evr={pca_evr:.3f}")

    # HP tuning
    val_split = int(0.9 * n_train)
    tune_train = train_pca[:val_split]
    tune_val = train_pca[val_split:]
    tune_train_w = extract_windows(tune_train, args.w)
    tune_val_w = extract_windows(tune_val, args.w)

    if args.tuning_trials > 0:
        print(f"    HP tuning (NC, {args.tuning_trials} trials) ...")
        best_hp = tune_score_model_null_contrast(
            tune_train_w.reshape(-1, args.pca_dim * args.w),
            tune_val_w.reshape(-1, args.pca_dim * args.w),
            in_features=args.pca_dim * args.w,
            m=args.pca_dim, w=args.w,
            n_trials=args.tuning_trials,
            device=args.device,
            seed=0,
        )
    else:
        best_hp = {"sigma": 0.3, "hidden_dim": 256, "lr": 1e-3}

    # Train final score model on train_pca (90% for fit, 10% for val DSM check)
    score_train = train_pca[:val_split]
    score_val = train_pca[val_split:]
    train_w = extract_windows(score_train, args.w)
    val_w = extract_windows(score_val, args.w)
    test_w = extract_windows(test_pca, args.w)

    score_model = train_score_model_layer(
        train_w, val_w,
        m=args.pca_dim, w=args.w,
        epochs=args.epochs,
        lr=best_hp["lr"], sigma=best_hp["sigma"],
        hidden_dim=best_hp["hidden_dim"],
        batch_size=256, device=args.device,
    )

    # Operator + diagnostics
    estimator = LocalizedMultiBlockEstimator(m=args.pca_dim, w=args.w, max_lag=args.max_lag)
    stats = estimator.estimate(score_model, test_w, device=args.device)
    print(f"    A_r[1] = {stats['A_r'][1]:.4f}   "
          f"SCR_5 = {stats['SCR_r'][1, 4]:.4f}   "
          f"RDI = {stats['SI']:.4f}")

    # Bootstrap
    bootstrap_ci = None
    if args.bootstrap_n > 0:
        print(f"    bootstrap (n={args.bootstrap_n}) ...")
        bootstrap_ci = bootstrap_pivot_cis(
            score_model=score_model, test_windows=test_w,
            pca_dim=args.pca_dim, w=args.w, max_lag=args.max_lag,
            skip_edges=estimator.skip_edges, n_bootstrap=args.bootstrap_n,
            point_estimates={
                "RDI": float(stats["SI"]),
                "A_r": stats["A_r"].tolist(),
                "beta": float(stats["beta"]),
            },
            device=args.device,
        )

    # Linear position probe (on the same PCA-reduced activations)
    probe_acc = linear_position_probe(train_pca, test_pca)
    print(f"    probe acc = {probe_acc:.4f}  (chance = {1.0/seq_len:.4f})")

    result = {
        "cell": cell_label,
        **extra_metadata,
        "pca_dim": args.pca_dim, "w": args.w, "max_lag": args.max_lag,
        "n_train": n_train, "n_test": N - n_train,
        "train_pca_evr": pca_evr,
        "best_hp": best_hp,
        "A_r": stats["A_r"].tolist(),
        "H_r": stats["H_r"].tolist(),
        "RDI": float(stats["SI"]),
        "beta": float(stats["beta"]),
        "SCR_r": stats["SCR_r"].tolist(),
        "M_bar_r": stats["M_bar_r"].tolist(),
        "delta_frob": stats["delta_frob"].tolist(),
        "M_frob": stats["M_frob"].tolist(),
        "probe_accuracy": probe_acc,
    }
    if bootstrap_ci is not None:
        result["bootstrap_ci"] = bootstrap_ci
    return result


# ============================================================================
# Per-model driver
# ============================================================================


def cells_for_model(
    model_dir: Path,
    metadata: Dict[str, Any],
    modes: List[str],
) -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Return [(cell_label, npy_path, extra_metadata), ...] for the chosen modes."""
    cells: List[Tuple[str, Path, Dict[str, Any]]] = []
    if "logit" in modes:
        p = model_dir / "logit_pca.npy"
        if p.exists():
            cells.append(("logit", p, {
                "space": "logit", "layer_index": None,
                "layer_relative_depth": None,
            }))
    if "hidden" in modes or "internal" in modes:
        n_layer = metadata.get("n_layer")
        for L in metadata.get("chosen_layers", []):
            p = model_dir / f"hidden_L{L}_pca.npy"
            if p.exists():
                rel = float(L / n_layer) if n_layer else None
                cells.append((f"hidden_L{L}", p, {
                    "space": "hidden",
                    "layer_index": int(L),
                    "layer_relative_depth": rel,
                }))
    return cells


def process_model(
    model_key: str,
    pretrained_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    safe = _safe_key(model_key)
    model_dir = pretrained_dir / safe
    if not (model_dir / "metadata.json").exists():
        print(f"  SKIP {model_key}: no metadata at {model_dir}")
        return

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS: {model_key}  (PE={meta['pe_type']}, "
          f"family={meta['family']}, tier={meta['tier']})")
    print(f"{'=' * 72}")

    cells = cells_for_model(model_dir, meta, args.modes)
    if not cells:
        print(f"  SKIP {model_key}: no cells matching modes={args.modes}")
        return

    out_model = out_dir / safe
    out_model.mkdir(parents=True, exist_ok=True)

    base_meta = {
        "model_key": model_key,
        "hf_id": meta["hf_id"],
        "pe_type": meta["pe_type"],
        "family": meta["family"],
        "tier": meta["tier"],
        "training_corpus": meta["training_corpus"],
        "tokenizer_family": meta["tokenizer_family"],
        "n_params": meta["n_params"],
        "n_layer": meta["n_layer"],
        "hidden_dim": meta["hidden_dim"],
    }

    for cell_label, npy_path, extra in cells:
        out_path = out_model / f"{cell_label}_metrics.json"
        if out_path.exists() and not args.force:
            print(f"  SKIP {cell_label}: {out_path.name} exists (use --force to redo)")
            continue
        activations = np.load(npy_path)
        cell_meta = {**base_meta, **extra}
        result = analyze_cell(activations, cell_label, args, cell_meta)

        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(type(o))

        with open(out_path, "w") as fp:
            json.dump(result, fp, default=_default, indent=2)
        print(f"  Wrote {out_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained-dir", type=str, required=True,
                   help="Directory produced by extract_pretrained_internals.py")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model keys to analyze.")
    p.add_argument("--modes", nargs="+", default=["logit", "hidden"],
                   help="Which cells to analyze. 'logit' = logit_pca.npy; "
                        "'hidden' = every hidden_L<N>_pca.npy listed in the model's "
                        "metadata.chosen_layers.")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--w", type=int, default=16)
    p.add_argument("--max-lag", type=int, default=14)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--tuning-trials", type=int, default=60)
    p.add_argument("--bootstrap-n", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    pretrained_dir = Path(args.pretrained_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        process_model(key, pretrained_dir, out_dir, args)

    print(f"\nAll models analyzed. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
