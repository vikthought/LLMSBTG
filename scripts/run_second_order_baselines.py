"""
Second-order and conditional baselines for the score-SVD ablation comparison.

Reviewer-requested baselines that compete with score-SVD at the same forward
hook, all extracting k=3 source-side directions in m=32 PCA space and
back-projecting through the same per-cell `pca_components`:

  score_lag1_k3              top-3 right SVs of M_bar[1] (current pipeline reference)
  cross_cov_svd_k3           top-3 right SVs of E[u_i u_{i-1}^T] on real PCA-reduced acts
  cca_k3                     top-3 source-side canonical directions (sklearn CCA(u_{i-1}, u_i))
  cond_regression_k3         top-3 right SVs of W from u_i = W u_{i-1} + b
  top_pca_k3                 first 3 PCA components (variance-only, no probe / no score)
  shuffled_score_k3          top-3 right SVs of M_bar[1] with each row's coefficients
                             permuted across PCs (same magnitude profile, scrambled directions)
  pair_probe_concat_k3       top-3 right SVs of source-side block of an LR weight matrix
                             trained on concat(u_i, u_{i-1}) → bucketed |i-j|
  pair_probe_difference_k3   top-3 right SVs of weights from LR trained on
                             (u_i - u_{i-1}) → bucketed |i-j|

All are evaluated through the same `_eval_with_dirs` forward hook used by
score-SVD and probe-hidden, so ablation deltas are apples-to-apples.

This is Sprint 1 of the reviewer-response plan; pair-shuffled score-model
retraining (Sprint 2) lives in run_pair_shuffled_score_ablation.py.

Output (one JSON per cell):
  results/<out-dir>/<pe>_seed<N>_L<layer>_second_order.json

Usage
-----
python scripts/run_second_order_baselines.py \\
    --summary-path results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
    --models-dir   results/transformer_pos_models_20260419_114958 \\
    --data-dir     data/transformer_pos_cluster \\
    --out-dir      results/lagpair_ablation_3seed/second_order_baselines \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 1 2 3 4 \\
    --top-k 3 \\
    --alpha 2.0 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.data.transformer_variants import create_transformer_variant
from scripts.run_ablation_study import (
    _load_test_families,
    _masked_ce,
    _eval_with_dirs,
)


# ---------------------------------------------------------------------------
# PCA-space pair sampling for the regression / CCA / probe builders
# ---------------------------------------------------------------------------

# Reuse the bucketing convention from run_probe_baselines.py for comparability
# with rel_dist / near_far in App F.
DIST_BINS = [1, 2, 3, 4, 8, 16]


def _bucket_dist(d: int) -> int:
    for i, edge in enumerate(DIST_BINS):
        if d <= edge:
            return i
    return len(DIST_BINS)


def _pca_project_layer(
    acts_l: np.ndarray, pca_components: np.ndarray, mu_l: np.ndarray
) -> np.ndarray:
    """acts_l: (N, seq_len, H) → (N, seq_len, m) using per-cell components / mean."""
    N, T, H = acts_l.shape
    flat = (acts_l.reshape(-1, H) - mu_l[None, :])
    return (flat @ pca_components.T).reshape(N, T, -1).astype(np.float32)


def _pair_at_lag(u: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """u: (N, T, m) → (N*(T-lag), m), (N*(T-lag), m).  Returns (u_i, u_{i-lag}) flat."""
    u_i      = u[:, lag:, :].reshape(-1, u.shape[-1])
    u_lagged = u[:, :-lag, :].reshape(-1, u.shape[-1])
    return u_i, u_lagged


def _multi_lag_pairs(
    u: np.ndarray, max_lag: int = 14, max_pairs: int = 200_000, rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (u_i, u_{i-r}) pairs across r ∈ {1..max_lag}.  Returns (u_i, u_lagged, lags)."""
    N, T, m = u.shape
    rng = np.random.default_rng(rng_seed)
    pairs_i, pairs_j = [], []
    for n in range(N):
        # Sample roughly uniform over (i, j) with j < i, j >= i - max_lag
        for r in range(1, max_lag + 1):
            for i in range(r, T):
                pairs_i.append((n, i)); pairs_j.append((n, i - r))
    pairs_i = np.array(pairs_i); pairs_j = np.array(pairs_j)
    if len(pairs_i) > max_pairs:
        idx = rng.choice(len(pairs_i), max_pairs, replace=False)
        pairs_i = pairs_i[idx]; pairs_j = pairs_j[idx]
    u_i      = u[pairs_i[:, 0], pairs_i[:, 1]]
    u_lagged = u[pairs_j[:, 0], pairs_j[:, 1]]
    lags     = pairs_i[:, 1] - pairs_j[:, 1]
    return u_i, u_lagged, lags


# ---------------------------------------------------------------------------
# Direction extractors (each returns (k, m) in PCA space)
# ---------------------------------------------------------------------------

def _dir_score_lag1(m_bar_r: np.ndarray, k: int, lag: int = 1) -> np.ndarray:
    _, _, Vh = np.linalg.svd(m_bar_r[lag])
    return Vh[:k, :].astype(np.float32)


def _dir_cross_cov_svd(u_i: np.ndarray, u_lag: np.ndarray, k: int) -> np.ndarray:
    """Σ_cross = E[u_i u_lag^T] / N.  Top-k right SVs (source side)."""
    n = u_i.shape[0]
    cov = (u_i.T @ u_lag) / max(1, n)  # (m, m)
    _, _, Vh = np.linalg.svd(cov)
    return Vh[:k, :].astype(np.float32)


def _dir_cca(u_i: np.ndarray, u_lag: np.ndarray, k: int) -> np.ndarray:
    """sklearn CCA(X=u_lag, Y=u_i).  Returns top-k source-side x_weights."""
    cca = CCA(n_components=k, max_iter=500)
    cca.fit(u_lag, u_i)
    # x_weights_ shape: (m, k).  Treat each column as a source-side direction.
    dirs = cca.x_weights_.T.astype(np.float32)  # (k, m)
    # Orthonormalize (CCA weights are not in general orthonormal in m-space)
    Q, _ = np.linalg.qr(dirs.T)
    return Q.T.astype(np.float32)


def _dir_cond_regression(u_i: np.ndarray, u_lag: np.ndarray, k: int) -> np.ndarray:
    """Fit u_i = W u_lag + b.  Top-k right SVs of W (sources of biggest variance)."""
    reg = LinearRegression(fit_intercept=True)
    reg.fit(u_lag, u_i)
    W = reg.coef_  # (m_target, m_source) = (m, m)
    _, _, Vh = np.linalg.svd(W)
    return Vh[:k, :].astype(np.float32)


def _dir_top_pca(m: int, k: int) -> np.ndarray:
    """First k PCs in PCA-coordinate basis = first k rows of identity in m-space.

    Back-projection through pca_components recovers the actual top-k PC vectors
    in hidden space (which is what we want — variance-only ablation).
    """
    eye = np.eye(m, dtype=np.float32)
    return eye[:k]


def _dir_shuffled_score(
    m_bar_r: np.ndarray, k: int, lag: int = 1, rng_seed: int = 0,
) -> np.ndarray:
    """Permute each row's coefficients across PCs, then re-orthonormalize.

    Preserves each direction's per-row magnitude profile (Σ d_i² = 1) but
    scrambles which PCs the energy lives on. A null for "does the structure
    of M_bar[1] matter, or just its row count and magnitude?"
    """
    _, _, Vh = np.linalg.svd(m_bar_r[lag])
    top = Vh[:k, :].astype(np.float32)
    rng = np.random.default_rng(rng_seed)
    out = np.empty_like(top)
    for r in range(k):
        perm = rng.permutation(top.shape[1])
        out[r] = top[r, perm]
    Q, _ = np.linalg.qr(out.T)
    return Q.T.astype(np.float32)


def _dir_pair_probe_concat(
    u: np.ndarray, k: int, max_pairs: int = 200_000, rng_seed: int = 0,
    max_lag: int = 14,
) -> np.ndarray:
    """LR on concat(u_i, u_{i-r}) → bucketed |i-j|.  Top-k right SVs of the
    SOURCE-side block of the weight matrix (the second m columns)."""
    u_i, u_lag, lags = _multi_lag_pairs(u, max_lag=max_lag, max_pairs=max_pairs, rng_seed=rng_seed)
    X = np.concatenate([u_i, u_lag], axis=1)  # (N, 2m)
    y = np.array([_bucket_dist(int(d)) for d in lags])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=400, multi_class="multinomial", solver="lbfgs")
    clf.fit(Xs, y)
    W = clf.coef_  # (n_classes, 2m)
    m = u.shape[-1]
    W_source = W[:, m:]  # (n_classes, m) — the u_{i-r} half (source side)
    _, _, Vh = np.linalg.svd(W_source)
    dirs = Vh[:k, :].astype(np.float32)
    return dirs


def _dir_pair_probe_difference(
    u: np.ndarray, k: int, max_pairs: int = 200_000, rng_seed: int = 0,
    max_lag: int = 14,
) -> np.ndarray:
    """LR on (u_i - u_{i-r}) → bucketed |i-j|.  Top-k right SVs of weights."""
    u_i, u_lag, lags = _multi_lag_pairs(u, max_lag=max_lag, max_pairs=max_pairs, rng_seed=rng_seed)
    X = u_i - u_lag                             # (N, m)
    y = np.array([_bucket_dist(int(d)) for d in lags])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=400, multi_class="multinomial", solver="lbfgs")
    clf.fit(Xs, y)
    W = clf.coef_                               # (n_classes, m)
    _, _, Vh = np.linalg.svd(W)
    return Vh[:k, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _to_hidden(dirs_m: np.ndarray, pca_components: np.ndarray) -> torch.Tensor:
    if dirs_m.size == 0:
        return torch.zeros((0, pca_components.shape[1]), dtype=torch.float32)
    dirs_h = dirs_m @ pca_components
    t = torch.tensor(dirs_h, dtype=torch.float32)
    Q, _ = torch.linalg.qr(t.T)
    return Q.T


def _baseline_alpha0_loss(model, loaders, device) -> Dict[str, float]:
    model.eval()
    fam: Dict[str, float] = {}
    with torch.no_grad():
        for f, loader in loaders.items():
            losses = []
            for seqs, masks in loader:
                seqs = seqs.to(device); masks = masks.to(device)
                out = model(input_ids=seqs)
                losses.append(_masked_ce(out.logits, seqs, masks))
            fam[f] = float(np.mean(losses))
    return fam


def _fam_avg(fam_losses: Dict[str, float]) -> float:
    return float(np.mean([v for k, v in fam_losses.items() if k != "iid_random"]))


def _eval_one(dirs_h, model, loaders, target_layer, mu_l, alpha, device, side, lag):
    if dirs_h.shape[0] == 0:
        return _baseline_alpha0_loss(model, loaders, device)
    out = _eval_with_dirs(
        model=model, loaders=loaders, target_layer=target_layer,
        dirs_h=dirs_h, mu_l=mu_l,
        alpha_values=[alpha], device=device, side=side, lag=lag,
    )
    return out[alpha]


def _variance_footprint(dirs_m: np.ndarray, evr: np.ndarray) -> float:
    return float(np.sum((dirs_m ** 2) * evr[None, :]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-path",     type=str, required=True)
    p.add_argument("--models-dir",       type=str, required=True)
    p.add_argument("--data-dir",         type=str, required=True)
    p.add_argument("--out-dir",          type=str, required=True)
    p.add_argument("--pe-types",         type=str, nargs="+", required=True)
    p.add_argument("--seeds",            type=int, nargs="+", required=True)
    p.add_argument("--layers",           type=int, nargs="+", required=True)
    p.add_argument("--lag",              type=int, default=1,
                   help="Single lag for the score-SVD baselines and the hook's mask.")
    p.add_argument("--top-k",            type=int, default=3)
    p.add_argument("--alpha",            type=float, default=2.0)
    p.add_argument("--side",             type=str, default="source",
                   choices=["source", "target"])
    p.add_argument("--max-pairs",        type=int, default=200_000,
                   help="Cap on (i, j) pairs sampled per cell for CCA / regression / probes.")
    p.add_argument("--max-lag-for-probes", type=int, default=14,
                   help="Pair-probe sampling spans lags 1..this.")
    p.add_argument("--device",           type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rng-seed-base",    type=int, default=20260430)
    args = p.parse_args()

    summary_path = Path(args.summary_path)
    models_dir   = Path(args.models_dir)
    data_dir     = Path(args.data_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    families = meta.get("families", [
        "variable_lag_copy", "absolute_anchor",
        "order_sensitive", "distance_bucket", "iid_random",
    ])
    config = GPT2Config(
        vocab_size=meta["vocab_size"], n_positions=meta["seq_len"],
        n_embd=256, n_layer=4, n_head=4, n_inner=1024,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    loaders = _load_test_families(data_dir, families)

    pe_to_idx = {"rope": 0, "alibi": 1, "absolute": 2}

    for pe_type in args.pe_types:
        for seed in args.seeds:
            seed_dir      = models_dir / f"{pe_type}_seed{seed}"
            model_pt      = seed_dir / "model.pt"
            test_acts_pt  = seed_dir / "test_acts.npy"
            analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
            if not (model_pt.exists() and test_acts_pt.exists() and analysis_file.exists()):
                print(f"  Skipping {pe_type} seed {seed}: model / acts / analysis missing.")
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)

            # Defer test-act loading to per-layer (different layer slice).
            test_acts_full = np.load(test_acts_pt, mmap_mode="r")  # (N, n_layers+1, T, H)

            model = create_transformer_variant(config, pe_type)
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(args.device)

            for layer in args.layers:
                out_path = out_dir / f"{pe_type}_seed{seed}_L{layer}_second_order.json"
                if out_path.exists():
                    print(f"  SKIP (exists): {out_path.name}")
                    continue

                t0 = time.time()
                layer_module = model.transformer.h[layer - 1]
                ld           = analysis["layer_stats"][layer - 1]

                pca_components = np.asarray(ld["pca_components"], dtype=np.float32)
                mu_l           = np.asarray(ld["mu_l"], dtype=np.float32)
                m_bar_r        = np.asarray(ld["M_bar_r"], dtype=np.float32)
                evr            = np.asarray(ld.get("pca_explained_var_ratio", []), dtype=np.float32)
                m, H = pca_components.shape
                if evr.size == 0:
                    evr = np.full(m, 1.0 / m, dtype=np.float32)

                # PCA-project this layer's test activations once
                acts_layer = np.asarray(test_acts_full[:, layer], dtype=np.float32)  # (N, T, H)
                u = _pca_project_layer(acts_layer, pca_components, mu_l)             # (N, T, m)

                # Lag-1 pairs for cross-cov / CCA / cond-regression
                u_i, u_lag1 = _pair_at_lag(u, args.lag)

                # Per-cell rng seed for the shuffled-score and pair-probe sampling
                rng_seed_cell = args.rng_seed_base + pe_to_idx.get(pe_type, 0) * 1_000_000 \
                                + int(seed) * 10_000 + int(layer) * 100

                # Build all direction sets in m-space
                dir_sets: Dict[str, np.ndarray] = {}
                t_build = time.time()
                dir_sets["score_lag1_k3"]            = _dir_score_lag1(m_bar_r, args.top_k, args.lag)
                dir_sets["cross_cov_svd_k3"]         = _dir_cross_cov_svd(u_i, u_lag1, args.top_k)
                dir_sets["cca_k3"]                   = _dir_cca(u_i, u_lag1, args.top_k)
                dir_sets["cond_regression_k3"]       = _dir_cond_regression(u_i, u_lag1, args.top_k)
                dir_sets["top_pca_k3"]               = _dir_top_pca(m, args.top_k)
                dir_sets["shuffled_score_k3"]        = _dir_shuffled_score(
                    m_bar_r, args.top_k, args.lag, rng_seed=rng_seed_cell)
                dir_sets["pair_probe_concat_k3"]     = _dir_pair_probe_concat(
                    u, args.top_k, max_pairs=args.max_pairs,
                    rng_seed=rng_seed_cell, max_lag=args.max_lag_for_probes)
                dir_sets["pair_probe_difference_k3"] = _dir_pair_probe_difference(
                    u, args.top_k, max_pairs=args.max_pairs,
                    rng_seed=rng_seed_cell + 1, max_lag=args.max_lag_for_probes)
                build_secs = time.time() - t_build

                # Free the layer-specific test acts before the GPU forward sweep
                del acts_layer, u, u_i, u_lag1

                # Shared α=0 baseline
                base_losses = _baseline_alpha0_loss(model, loaders, args.device)
                base_avg    = _fam_avg(base_losses)

                # Run each direction set through the same hook at α=2
                results: Dict[str, Dict] = {}
                for name, dirs_m in dir_sets.items():
                    dirs_h = _to_hidden(dirs_m, pca_components)
                    fam_losses = _eval_one(
                        dirs_h, model, loaders, layer_module, mu_l,
                        args.alpha, args.device, args.side, args.lag,
                    )
                    results[name] = {
                        "k":             int(dirs_m.shape[0]),
                        "delta":         _fam_avg(fam_losses) - base_avg,
                        "var_footprint": _variance_footprint(dirs_m, evr),
                        "fam_losses":    fam_losses,
                    }

                summary = {
                    "cell": {
                        "pe_type": pe_type, "seed": seed, "layer": layer,
                        "lag": args.lag, "side": args.side, "alpha": args.alpha,
                    },
                    "config": {
                        "m": int(m), "H": int(H), "top_k": int(args.top_k),
                        "max_pairs": int(args.max_pairs),
                        "max_lag_for_probes": int(args.max_lag_for_probes),
                        "rng_seed_cell":      int(rng_seed_cell),
                        "build_secs":         build_secs,
                    },
                    "baseline_alpha0_fam_losses": base_losses,
                    "baseline_alpha0_fam_avg":    base_avg,
                    "conditions":                 results,
                    "wall_time_sec":              time.time() - t0,
                }

                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                # Compact log: ranked by Δ
                ordered = sorted(results.items(), key=lambda kv: -kv[1]["delta"])
                tag = f"{pe_type} s={seed} L={layer}"
                bits = [f"{name}={d['delta']:+.3f}" for name, d in ordered]
                print(f"  {tag}  build={build_secs:.0f}s  fwd={summary['wall_time_sec']-build_secs:.0f}s  | "
                      + "  ".join(bits))

    print("Done.")


if __name__ == "__main__":
    main()
