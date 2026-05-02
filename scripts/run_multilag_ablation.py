"""
Multi-lag and higher-k score-SVD ablation comparison.

Diagnosis of the random-control failure (Study 1, results/lagpair_ablation_3seed/
random_pca_distribution/): the existing score-SVD extraction in
run_ablation_study.py / SingularAblationHook reads ``Vh`` from
``m_bar[lag=1]`` only — a single 32×32 matrix from the 14 lag operators
the score model produces. SBTG's framework is per-LAG; capping at lag-1
discards 13 of 14 slices.

This script tests whether multi-lag aggregation (using
``multilag_svd_directions`` from interventions.py, which is implemented
but unused in the ablation pipeline) and / or expanded k closes the gap
to PCA-random in the cells where it currently fails (RoPE deep layers,
Absolute deep layers).

Conditions per cell (all back-project through the same pca_components):

  score_lag1_k3              top-3 right SVs of M_bar[1]            — baseline
  score_multilag_uniform_k3  stack M_bar[1..max_lag] (uniform), SVD top-3
  score_multilag_amplitude_k3 amplitude-weighted stack, SVD top-3
  score_lag1_k5              top-5 from M_bar[1]
  score_multilag_uniform_k5  multi-lag stack, SVD top-5

  random_pca_k3              N=20 PCA-random orthonormal subspaces, k=3
  random_pca_k5              N=20 PCA-random orthonormal subspaces, k=5

Plus per-direction-set variance footprint for diagnostic correlation.

Output (one JSON per cell):
  results/<out-dir>/<pe>_seed<N>_L<layer>_multilag_ablation.json

Usage
-----
python scripts/run_multilag_ablation.py \\
    --summary-path results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
    --models-dir   results/transformer_pos_models_20260419_114958 \\
    --data-dir     data/transformer_pos_cluster \\
    --out-dir      results/lagpair_ablation_3seed/multilag_ablation \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 1 2 3 4 \\
    --num-random-draws 20 \\
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
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.data.transformer_variants import create_transformer_variant
from src.sbtg.evaluation.interventions import multilag_svd_directions
from scripts.run_ablation_study import (
    _load_test_families,
    _masked_ce,
    _eval_with_dirs,
)


# ---------------------------------------------------------------------------
# Direction extraction strategies
# ---------------------------------------------------------------------------

def _lag1_svd_dirs(m_bar_r: np.ndarray, k: int, lag: int = 1) -> np.ndarray:
    """Existing pipeline behavior: top-k right SVs of M_bar at a single lag."""
    U, S, Vh = np.linalg.svd(m_bar_r[lag])
    return Vh[:k, :].astype(np.float32)  # (k, m)


def _multilag_svd_dirs(
    m_bar_r: np.ndarray,
    k: int,
    weighting: str = "uniform",
    lag_min: int = 1,
) -> np.ndarray:
    """Stack M_bar across lags=lag_min..max_lag, SVD, return top-k right SVs.

    Defaults to lags 1..max_lag (skips lag-0 self-coupling, which is
    typically the largest A_r block and would dominate uniform stacking).
    """
    max_lag_p1 = m_bar_r.shape[0]
    lags = list(range(lag_min, max_lag_p1))
    dirs_m, _info = multilag_svd_directions(
        m_bar_r=m_bar_r, k=k, lags=lags, weighting=weighting, side="source",
    )
    return dirs_m.astype(np.float32)


def _random_pca_dirs(rng: np.random.Generator, k: int, m: int) -> np.ndarray:
    g = rng.standard_normal((k, m)).astype(np.float32)
    q, _ = np.linalg.qr(g.T)
    return q.T


def _to_hidden(dirs_m: np.ndarray, pca_components: np.ndarray) -> torch.Tensor:
    """(k, m) → (k, H), QR-orthonormalized in hidden space."""
    if dirs_m.size == 0 or dirs_m.shape[0] == 0:
        return torch.zeros((0, pca_components.shape[1]), dtype=torch.float32)
    dirs_h = dirs_m @ pca_components
    t = torch.tensor(dirs_h, dtype=torch.float32)
    Q, _ = torch.linalg.qr(t.T)
    return Q.T


def _variance_footprint(dirs_m: np.ndarray, evr: np.ndarray) -> float:
    return float(np.sum((dirs_m ** 2) * evr[None, :]))


# ---------------------------------------------------------------------------
# Eval helpers (mirror run_random_pca_distribution.py for consistency)
# ---------------------------------------------------------------------------

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
                   help="Single lag for the score_lag1_kX baselines and the hook's mask.")
    p.add_argument("--lag-min",          type=int, default=1,
                   help="Smallest lag to include in multi-lag stacks (skip 0 to avoid "
                        "lag-0 self-coupling dominating).")
    p.add_argument("--num-random-draws", type=int, default=20)
    p.add_argument("--alpha",            type=float, default=2.0)
    p.add_argument("--side",             type=str, default="source",
                   choices=["source", "target"])
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
        vocab_size  = meta["vocab_size"],
        n_positions = meta["seq_len"],
        n_embd=256, n_layer=4, n_head=4, n_inner=1024,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    loaders = _load_test_families(data_dir, families)

    pe_to_idx = {"rope": 0, "alibi": 1, "absolute": 2}

    # Direction-set spec: (name, k, builder)
    # `builder(m_bar_r, k, rng)` -> dirs_m (k, m). rng only used for random.
    score_specs = [
        ("score_lag1_k3",                3, lambda M, k, rng: _lag1_svd_dirs(M, k, args.lag)),
        ("score_multilag_uniform_k3",    3, lambda M, k, rng: _multilag_svd_dirs(M, k, "uniform",   args.lag_min)),
        ("score_multilag_amplitude_k3",  3, lambda M, k, rng: _multilag_svd_dirs(M, k, "amplitude", args.lag_min)),
        ("score_lag1_k5",                5, lambda M, k, rng: _lag1_svd_dirs(M, k, args.lag)),
        ("score_multilag_uniform_k5",    5, lambda M, k, rng: _multilag_svd_dirs(M, k, "uniform",   args.lag_min)),
    ]
    rand_ks = [3, 5]

    for pe_type in args.pe_types:
        for seed in args.seeds:
            seed_dir      = models_dir / f"{pe_type}_seed{seed}"
            model_pt      = seed_dir / "model.pt"
            analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
            if not model_pt.exists() or not analysis_file.exists():
                print(f"  Skipping {pe_type} seed {seed}: model or analysis JSON missing.")
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)

            model = create_transformer_variant(config, pe_type)
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(args.device)

            for layer in args.layers:
                out_path = out_dir / f"{pe_type}_seed{seed}_L{layer}_multilag_ablation.json"
                if out_path.exists():
                    print(f"  SKIP (exists): {out_path.name}")
                    continue

                t0 = time.time()
                layer_module = model.transformer.h[layer - 1]
                ld           = analysis["layer_stats"][layer - 1]

                pca_components = np.asarray(ld["pca_components"], dtype=np.float32)  # (m, H)
                mu_l           = np.asarray(ld["mu_l"], dtype=np.float32)
                m_bar_r        = np.asarray(ld["M_bar_r"], dtype=np.float32)         # (max_lag+1, m, m)
                evr            = np.asarray(ld.get("pca_explained_var_ratio", []), dtype=np.float32)
                A_r            = np.asarray(ld.get("A_r", []), dtype=np.float32)
                m, H = pca_components.shape
                if evr.size == 0:
                    evr = np.full(m, 1.0 / m, dtype=np.float32)

                # ---- shared α=0 baseline ----
                base_losses = _baseline_alpha0_loss(model, loaders, args.device)
                base_avg = _fam_avg(base_losses)

                # ---- score-SVD conditions ----
                rng_seed_cell = args.rng_seed_base \
                                + pe_to_idx.get(pe_type, 0) * 1000000 \
                                + int(seed) * 10000 + int(layer) * 100
                score_results: Dict[str, Dict] = {}
                for name, k, build in score_specs:
                    dirs_m = build(m_bar_r, k, None)
                    dirs_h = _to_hidden(dirs_m, pca_components)
                    fam_losses = _eval_one(
                        dirs_h, model, loaders, layer_module, mu_l,
                        args.alpha, args.device, args.side, args.lag,
                    )
                    score_results[name] = {
                        "k":             k,
                        "delta":         _fam_avg(fam_losses) - base_avg,
                        "var_footprint": _variance_footprint(dirs_m, evr),
                        "fam_losses":    fam_losses,
                    }

                # ---- PCA-random null distributions, matched-k ----
                rand_results: Dict[str, Dict] = {}
                for k in rand_ks:
                    rng = np.random.default_rng(rng_seed_cell + k)
                    deltas = []
                    vfps = []
                    for d in range(args.num_random_draws):
                        dirs_m = _random_pca_dirs(rng, k, m)
                        dirs_h = _to_hidden(dirs_m, pca_components)
                        fam_losses = _eval_one(
                            dirs_h, model, loaders, layer_module, mu_l,
                            args.alpha, args.device, args.side, args.lag,
                        )
                        deltas.append(_fam_avg(fam_losses) - base_avg)
                        vfps.append(_variance_footprint(dirs_m, evr))
                    arr = np.array(deltas)
                    rand_results[f"random_pca_k{k}"] = {
                        "k":             k,
                        "deltas":        deltas,
                        "delta_mean":    float(arr.mean()),
                        "delta_std":     float(arr.std()),
                        "delta_median":  float(np.median(arr)),
                        "delta_min":     float(arr.min()),
                        "delta_max":     float(arr.max()),
                        "delta_q05":     float(np.quantile(arr, 0.05)),
                        "delta_q95":     float(np.quantile(arr, 0.95)),
                        "var_footprint_mean": float(np.mean(vfps)),
                    }

                # ---- Score-SVD percentile vs the matched-k random null ----
                percentiles: Dict[str, float] = {}
                for name, payload in score_results.items():
                    k = payload["k"]
                    rand_key = f"random_pca_k{k}"
                    if rand_key not in rand_results:
                        continue
                    rand_arr = np.array(rand_results[rand_key]["deltas"])
                    pct = float(100.0 * np.mean(rand_arr < payload["delta"]))
                    percentiles[name] = pct

                summary = {
                    "cell": {
                        "pe_type": pe_type, "seed": seed, "layer": layer,
                        "lag":     args.lag, "lag_min_multilag": args.lag_min,
                        "side":    args.side, "alpha": args.alpha,
                    },
                    "config": {
                        "m":     int(m), "H": int(H),
                        "num_random_draws": args.num_random_draws,
                        "rng_seed_cell":    int(rng_seed_cell),
                        "lags_in_multilag": list(range(args.lag_min, m_bar_r.shape[0])),
                        "A_r":              [float(x) for x in A_r],
                    },
                    "baseline_alpha0_fam_losses": base_losses,
                    "baseline_alpha0_fam_avg":    base_avg,
                    "score_conditions":           score_results,
                    "random_distributions":       rand_results,
                    "score_percentiles_vs_matched_k_random": percentiles,
                    "wall_time_sec": time.time() - t0,
                }

                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                # Compact log line
                bits = [f"{name}={d['delta']:+.3f} (pct={percentiles.get(name, float('nan')):.0f})"
                        for name, d in score_results.items()]
                print(f"  {pe_type} seed={seed} L={layer}  "
                      f"r3μ={rand_results['random_pca_k3']['delta_mean']:+.3f}  "
                      f"r5μ={rand_results['random_pca_k5']['delta_mean']:+.3f}  | "
                      + "  ".join(bits) + f"  t={summary['wall_time_sec']:.0f}s")

    print("Done.")


if __name__ == "__main__":
    main()
