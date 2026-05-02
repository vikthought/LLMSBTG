"""
PCA-random null distribution for the score-SVD ablation comparison.

Background: the existing `random_control` field in dose_response.json is a
*single* PCA-random draw per cell. That makes the stable/random ratio in
docs/ABLATION_PIPELINE.md noisy: 21 / 36 per-seed cells have ratio < 1
because a single random draw can land on high-variance directions that
matter more for the model's output than the score-SVD's coupling-aligned
directions in deep layers / non-rank-one mechanisms (RoPE L4, Absolute
L3-L4).

This script draws N random orthonormal 3-dim subspaces of the m=32 PCA
subspace per (pe, seed, layer) cell and runs the same forward-pass
ablation hook as `run_ablation_study.py`, producing a *distribution* of
random-Δ values. Score-SVD's Δ is then reported as a percentile against
that distribution. A score-SVD percentile of, e.g., 95 means random
exceeds score-SVD only 5% of the time — that's a robust win.

We also record each random draw's variance footprint
(``sum_i d_i^2 * evr_i``) so a follow-up analysis can correlate Δ with
how much PCA-variance the draw spans (the dominant confound: random
draws that happen to align with high-variance PCs will damage more).

Output (one JSON per cell):
  results/<out-dir>/<pe>_seed<N>_L<layer>_random_distribution.json

Usage
-----
python scripts/run_random_pca_distribution.py \\
    --summary-path  results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
    --models-dir    results/transformer_pos_models_20260419_114958 \\
    --data-dir      data/transformer_pos_cluster \\
    --out-dir       results/lagpair_ablation_3seed/random_pca_distribution \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 1 2 3 4 \\
    --num-random-draws 100 \\
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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.data.transformer_variants import create_transformer_variant
from scripts.run_ablation_study import (
    _load_test_families,
    _masked_ce,
    _eval_with_dirs,
)


def _orthonormal_random_pca_dirs(
    rng: np.random.Generator, top_k: int, m: int
) -> np.ndarray:
    """Draw top_k orthonormal vectors uniformly in the m-dim PCA subspace."""
    g = rng.standard_normal((top_k, m)).astype(np.float32)
    q, _ = np.linalg.qr(g.T)
    return q.T  # (top_k, m)


def _baseline_alpha0_loss(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
) -> Dict[str, float]:
    """One forward pass with no hook = α=0 baseline. Same for every direction set."""
    model.eval()
    fam_losses: Dict[str, float] = {}
    with torch.no_grad():
        for fam, loader in loaders.items():
            losses = []
            for seqs, masks in loader:
                seqs = seqs.to(device)
                masks = masks.to(device)
                out = model(input_ids=seqs)
                losses.append(_masked_ce(out.logits, seqs, masks))
            fam_losses[fam] = float(np.mean(losses))
    return fam_losses


def _fam_avg(fam_losses: Dict[str, float]) -> float:
    """Mean over non-iid families (matches run_ablation_study aggregation)."""
    return float(np.mean([v for k, v in fam_losses.items() if k != "iid_random"]))


def _variance_footprint(dirs_m: np.ndarray, evr: np.ndarray) -> float:
    """Σ_i d_i² · evr_i for each direction, then sum across the k directions.

    A direction lying entirely in PC0 has footprint = evr[0].
    A uniform draw in m=32 has expected footprint = top_k * sum(evr) / m.
    """
    return float(np.sum((dirs_m ** 2) * evr[None, :]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path",     type=str, required=True)
    parser.add_argument("--models-dir",       type=str, required=True)
    parser.add_argument("--data-dir",         type=str, required=True)
    parser.add_argument("--out-dir",          type=str, required=True)
    parser.add_argument("--pe-types",         type=str, nargs="+", required=True)
    parser.add_argument("--seeds",            type=int, nargs="+", required=True)
    parser.add_argument("--layers",           type=int, nargs="+", required=True)
    parser.add_argument("--lag",              type=int, default=1)
    parser.add_argument("--top-k",            type=int, default=3)
    parser.add_argument("--num-random-draws", type=int, default=100,
                        help="Number of PCA-random subspace draws per cell.")
    parser.add_argument("--alpha",            type=float, default=2.0,
                        help="Ablation strength for the random draws and score-SVD.")
    parser.add_argument("--side",             type=str, default="source",
                        choices=["source", "target"])
    parser.add_argument("--device",           type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rng-seed-base",    type=int, default=20260430,
                        help="Base seed for reproducibility; cell-unique offsets added.")
    args = parser.parse_args()

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
                out_path = out_dir / f"{pe_type}_seed{seed}_L{layer}_random_distribution.json"
                if out_path.exists():
                    print(f"  SKIP (exists): {out_path.name}")
                    continue

                t0 = time.time()
                layer_module = model.transformer.h[layer - 1]
                layer_data   = analysis["layer_stats"][layer - 1]

                pca_components = np.asarray(layer_data["pca_components"], dtype=np.float32)  # (m, H)
                mu_l           = np.asarray(layer_data["mu_l"], dtype=np.float32)            # (H,)
                m_bar_r        = np.asarray(layer_data["M_bar_r"], dtype=np.float32)         # (max_lag+1, m, m)
                evr            = np.asarray(layer_data.get("pca_explained_var_ratio", []),
                                            dtype=np.float32)
                m, H = pca_components.shape
                if evr.size == 0:
                    evr = np.full(m, 1.0 / m, dtype=np.float32)

                # ---- 1. α=0 baseline (shared across all direction sets) ----
                base_losses = _baseline_alpha0_loss(model, loaders, args.device)
                base_avg = _fam_avg(base_losses)

                # ---- 2. score-SVD top-k directions, α=2 ----
                U, S, Vh = np.linalg.svd(m_bar_r[args.lag])
                svd_dirs_m = Vh[: args.top_k, :].astype(np.float32)        # (k, m)
                svd_dirs_h = svd_dirs_m @ pca_components                    # (k, H)
                svd_dirs_h_t = torch.tensor(svd_dirs_h, dtype=torch.float32)
                Q, _ = torch.linalg.qr(svd_dirs_h_t.T)
                svd_dirs_h_orth = Q.T

                svd_eval = _eval_with_dirs(
                    model=model, loaders=loaders, target_layer=layer_module,
                    dirs_h=svd_dirs_h_orth, mu_l=mu_l,
                    alpha_values=[args.alpha], device=args.device,
                    side=args.side, lag=args.lag,
                )
                svd_alpha_losses = svd_eval[args.alpha]
                svd_delta = _fam_avg(svd_alpha_losses) - base_avg
                svd_var_fp = _variance_footprint(svd_dirs_m, evr)

                # ---- 3. N random PCA-subspace draws, α=2 ----
                rng_seed = args.rng_seed_base + pe_to_idx.get(pe_type, 0) * 10000 \
                           + int(seed) * 100 + int(layer)
                rng = np.random.default_rng(rng_seed)

                random_draws = []
                for d in range(args.num_random_draws):
                    rand_dirs_m = _orthonormal_random_pca_dirs(rng, args.top_k, m)  # (k, m)
                    rand_dirs_h = rand_dirs_m @ pca_components                       # (k, H)
                    rand_dirs_h_t = torch.tensor(rand_dirs_h, dtype=torch.float32)

                    rand_eval = _eval_with_dirs(
                        model=model, loaders=loaders, target_layer=layer_module,
                        dirs_h=rand_dirs_h_t, mu_l=mu_l,
                        alpha_values=[args.alpha], device=args.device,
                        side=args.side, lag=args.lag,
                    )
                    rand_alpha_losses = rand_eval[args.alpha]
                    rand_delta = _fam_avg(rand_alpha_losses) - base_avg
                    var_fp = _variance_footprint(rand_dirs_m, evr)

                    random_draws.append({
                        "draw_idx":     d,
                        "delta":        rand_delta,
                        "var_footprint": var_fp,
                        "fam_losses":   rand_alpha_losses,
                    })

                # Percentile = fraction of random draws with delta strictly less than score-SVD's
                rand_deltas = np.array([rd["delta"] for rd in random_draws])
                pct_strict = float(100.0 * np.mean(rand_deltas < svd_delta))
                pct_le     = float(100.0 * np.mean(rand_deltas <= svd_delta))

                summary = {
                    "cell": {
                        "pe_type": pe_type, "seed": seed, "layer": layer,
                        "lag": args.lag, "side": args.side, "alpha": args.alpha,
                    },
                    "config": {
                        "top_k": args.top_k,
                        "m":     int(m),
                        "H":     int(H),
                        "num_random_draws": args.num_random_draws,
                        "rng_seed":         int(rng_seed),
                    },
                    "baseline_alpha0_fam_losses": base_losses,
                    "baseline_alpha0_fam_avg":    base_avg,
                    "score_svd": {
                        "delta":          float(svd_delta),
                        "alpha_fam_losses": svd_alpha_losses,
                        "var_footprint":  float(svd_var_fp),
                        "singular_values": [float(x) for x in S[: args.top_k]],
                    },
                    "random_draws":         random_draws,
                    "random_summary": {
                        "delta_mean":   float(rand_deltas.mean()),
                        "delta_std":    float(rand_deltas.std()),
                        "delta_median": float(np.median(rand_deltas)),
                        "delta_min":    float(rand_deltas.min()),
                        "delta_max":    float(rand_deltas.max()),
                        "delta_q05":    float(np.quantile(rand_deltas, 0.05)),
                        "delta_q95":    float(np.quantile(rand_deltas, 0.95)),
                        "var_footprint_mean": float(np.mean([rd["var_footprint"] for rd in random_draws])),
                    },
                    "score_svd_percentile_strict": pct_strict,  # % of random draws < score-SVD
                    "score_svd_percentile_le":     pct_le,
                    "wall_time_sec": time.time() - t0,
                }
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"  Wrote {out_path.name}  "
                      f"(score Δ={svd_delta:+.4f}  rand μ={rand_deltas.mean():+.4f}±{rand_deltas.std():.4f}  "
                      f"pct={pct_strict:.0f}, t={summary['wall_time_sec']:.0f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
