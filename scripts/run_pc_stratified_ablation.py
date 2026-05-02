"""
PC-stratified ablation: separating coupling alignment from variance loading.

The single PCA-random control in `run_ablation_study.py` collapses two
distinct effects into one number:

  (A) variance loading — does the ablated subspace span high-variance PC
      directions the model relies on for its hidden-state arithmetic?
  (B) coupling alignment — does the ablated subspace span the joint
      cross-position structure SBTG was designed to identify?

If (A) and (B) coincide (ALiBi rank-one collapse, Absolute L1-L2
position-indexed vectors), score-SVD wins by a large margin. If they
diverge (RoPE deep layers, Absolute L3-L4 where coupling has bled into
low-variance modes), PCA-random can damage the model more than score-SVD
just by hitting variance-rich PCs.

This script disentangles the two by stratifying the m=32 PCA subspace
into a top-K_var "high variance" block and a bottom-(m-K_var)
"low variance" block, and running ablation in each block separately for
both random draws and score-SVD-restricted directions.

Conditions per cell:

  random_top_K   — k random orthonormal dirs in PCA[:K_var]
  random_bottom  — k random orthonormal dirs in PCA[K_var:]
  score_top_K    — score-SVD's top-k directions, projected onto PCA[:K_var]
                   then re-orthonormalized (drops their bottom-block mass)
  score_bottom   — score-SVD's top-k directions projected onto PCA[K_var:]

Plus analytics, computed without re-running ablation:

  score_var_footprint   — Σ_i (Vh_k_i)² · evr_i for the unrestricted top-k
  random_var_footprint  — single-draw observation; expected analytic value
  score_top_K_mass      — Σ_i (Vh_k_i)² over i ∈ [0, K_var)
                          (how much of score-SVD's mass lives in top-K_var)

Output (one JSON per cell):
  results/<out-dir>/<pe>_seed<N>_L<layer>_pc_stratified.json

Usage
-----
python scripts/run_pc_stratified_ablation.py \\
    --summary-path  results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
    --models-dir    results/transformer_pos_models_20260419_114958 \\
    --data-dir      data/transformer_pos_cluster \\
    --out-dir       results/lagpair_ablation_3seed/pc_stratified \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 1 2 3 4 \\
    --top-k 3 \\
    --top-k-var 8 \\
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
from scripts.run_ablation_study import (
    _load_test_families,
    _masked_ce,
    _eval_with_dirs,
)


def _baseline_alpha0_loss(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
) -> Dict[str, float]:
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
    return float(np.mean([v for k, v in fam_losses.items() if k != "iid_random"]))


def _random_subspace_in_block(
    rng: np.random.Generator, top_k: int, m: int, block: slice
) -> np.ndarray:
    """Top_k orthonormal dirs in PCA space, supported only on indices in `block`.

    Other dimensions are zero. Resulting matrix is (top_k, m).
    """
    dim = list(range(m))[block]  # the actual indices the block covers
    block_size = len(dim)
    if block_size < top_k:
        raise ValueError(f"block_size={block_size} < top_k={top_k}")
    g = rng.standard_normal((top_k, block_size)).astype(np.float32)
    q, _ = np.linalg.qr(g.T)
    block_dirs = q.T  # (top_k, block_size)
    out = np.zeros((top_k, m), dtype=np.float32)
    for col_in_block, col_in_full in enumerate(dim):
        out[:, col_in_full] = block_dirs[:, col_in_block]
    return out


def _restrict_to_block(dirs_m: np.ndarray, block: slice) -> np.ndarray:
    """Zero out dimensions outside `block`, then re-orthonormalize the rows.

    If the projected rows become rank-deficient (e.g. score-SVD has zero
    mass in the block), we orthonormalize what we can and return
    fewer-than-top_k rows. The caller treats num_rows < top_k as a
    near-degenerate case.
    """
    m = dirs_m.shape[1]
    mask = np.zeros(m, dtype=bool)
    mask[block] = True
    projected = dirs_m * mask[None, :]
    # Re-orthonormalize via SVD; zero-rank rows drop out.
    U, S, Vh = np.linalg.svd(projected, full_matrices=False)
    keep = S > 1e-8
    return Vh[keep]  # (rank, m)


def _to_hidden(dirs_m: np.ndarray, pca_components: np.ndarray) -> torch.Tensor:
    """(k, m) → (k, H), QR-orthonormalized in hidden space."""
    if dirs_m.size == 0 or dirs_m.shape[0] == 0:
        return torch.zeros((0, pca_components.shape[1]), dtype=torch.float32)
    dirs_h = dirs_m @ pca_components  # (k, H)
    t = torch.tensor(dirs_h, dtype=torch.float32)
    Q, _ = torch.linalg.qr(t.T)
    return Q.T


def _eval_or_zero(
    dirs_h: torch.Tensor, model, loaders, target_layer, mu_l, alpha, device, side, lag,
) -> Tuple[Dict[str, float], int]:
    """Run the ablation hook; if dirs_h is empty, return baseline and rank=0."""
    if dirs_h.shape[0] == 0:
        # No directions to ablate — return the baseline loss.
        return _baseline_alpha0_loss(model, loaders, device), 0
    eval_out = _eval_with_dirs(
        model=model, loaders=loaders, target_layer=target_layer,
        dirs_h=dirs_h, mu_l=mu_l,
        alpha_values=[alpha], device=device, side=side, lag=lag,
    )
    return eval_out[alpha], int(dirs_h.shape[0])


def _avg_random_in_block(
    rng: np.random.Generator,
    n_draws: int,
    top_k: int, m: int, block: slice,
    pca_components: np.ndarray, evr: np.ndarray,
    model, loaders, target_layer, mu_l, alpha, device, side, lag,
    base_avg: float,
) -> Dict:
    """Draw n_draws random subspaces in `block`, run ablation, summarize."""
    deltas, vfps, draws = [], [], []
    for d in range(n_draws):
        dirs_m = _random_subspace_in_block(rng, top_k, m, block)
        dirs_h = _to_hidden(dirs_m, pca_components)
        fam_losses, rank = _eval_or_zero(
            dirs_h, model, loaders, target_layer, mu_l, alpha, device, side, lag,
        )
        delta = _fam_avg(fam_losses) - base_avg
        vfp   = float(np.sum((dirs_m ** 2) * evr[None, :]))
        deltas.append(delta); vfps.append(vfp)
        draws.append({"draw_idx": d, "delta": delta, "var_footprint": vfp,
                      "rank": rank, "fam_losses": fam_losses})
    arr = np.array(deltas)
    return {
        "draws":      draws,
        "delta_mean": float(arr.mean()),
        "delta_std":  float(arr.std()),
        "delta_median": float(np.median(arr)),
        "var_footprint_mean": float(np.mean(vfps)),
    }


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
    parser.add_argument("--top-k",            type=int, default=3,
                        help="Subspace dimensionality (matches run_ablation_study.py).")
    parser.add_argument("--top-k-var",        type=int, default=8,
                        help="Number of leading PCs in the high-variance block.")
    parser.add_argument("--num-random-draws", type=int, default=20,
                        help="Random draws per condition for variance reduction.")
    parser.add_argument("--alpha",            type=float, default=2.0)
    parser.add_argument("--side",             type=str, default="source",
                        choices=["source", "target"])
    parser.add_argument("--device",           type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rng-seed-base",    type=int, default=20260430)
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
                out_path = out_dir / f"{pe_type}_seed{seed}_L{layer}_pc_stratified.json"
                if out_path.exists():
                    print(f"  SKIP (exists): {out_path.name}")
                    continue

                t0 = time.time()
                layer_module = model.transformer.h[layer - 1]
                layer_data   = analysis["layer_stats"][layer - 1]

                pca_components = np.asarray(layer_data["pca_components"], dtype=np.float32)
                mu_l           = np.asarray(layer_data["mu_l"], dtype=np.float32)
                m_bar_r        = np.asarray(layer_data["M_bar_r"], dtype=np.float32)
                evr            = np.asarray(layer_data.get("pca_explained_var_ratio", []),
                                            dtype=np.float32)
                m, H = pca_components.shape
                if evr.size == 0:
                    evr = np.full(m, 1.0 / m, dtype=np.float32)

                K = int(args.top_k_var)
                if K < args.top_k or (m - K) < args.top_k:
                    print(f"  WARNING: top-k-var={K} too small for top-k={args.top_k} in m={m}; "
                          f"clamping.")
                    K = max(args.top_k, min(K, m - args.top_k))
                top_block    = slice(0, K)
                bottom_block = slice(K, m)

                # ---------- Baseline α=0 ----------
                base_losses = _baseline_alpha0_loss(model, loaders, args.device)
                base_avg = _fam_avg(base_losses)

                # ---------- Score-SVD analytics + full ablation ----------
                U, S, Vh = np.linalg.svd(m_bar_r[args.lag])
                svd_dirs_m = Vh[: args.top_k, :].astype(np.float32)
                # variance footprint in full PCA basis
                svd_var_fp_full = float(np.sum((svd_dirs_m ** 2) * evr[None, :]))
                # mass in top-K block (squared coefficients summed over k and over the block)
                top_K_mass = float(np.sum(svd_dirs_m[:, top_block] ** 2)) / args.top_k

                # 1) score-SVD full
                score_full_h = _to_hidden(svd_dirs_m, pca_components)
                score_full_losses, _ = _eval_or_zero(
                    score_full_h, model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag,
                )
                score_full_delta = _fam_avg(score_full_losses) - base_avg

                # 2) score-SVD restricted to top-K block
                score_top_m = _restrict_to_block(svd_dirs_m, top_block)
                score_top_h = _to_hidden(score_top_m, pca_components)
                score_top_losses, score_top_rank = _eval_or_zero(
                    score_top_h, model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag,
                )
                score_top_delta = _fam_avg(score_top_losses) - base_avg

                # 3) score-SVD restricted to bottom block
                score_bot_m = _restrict_to_block(svd_dirs_m, bottom_block)
                score_bot_h = _to_hidden(score_bot_m, pca_components)
                score_bot_losses, score_bot_rank = _eval_or_zero(
                    score_bot_h, model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag,
                )
                score_bot_delta = _fam_avg(score_bot_losses) - base_avg

                # ---------- Random in each block ----------
                rng_seed = args.rng_seed_base + pe_to_idx.get(pe_type, 0) * 100000 \
                           + int(seed) * 1000 + int(layer)
                rng = np.random.default_rng(rng_seed)

                rand_top = _avg_random_in_block(
                    rng, args.num_random_draws, args.top_k, m, top_block,
                    pca_components, evr,
                    model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag, base_avg,
                )
                rand_bot = _avg_random_in_block(
                    rng, args.num_random_draws, args.top_k, m, bottom_block,
                    pca_components, evr,
                    model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag, base_avg,
                )
                # Also a full m-dim random for direct apples-to-apples with run_ablation_study
                rand_full = _avg_random_in_block(
                    rng, args.num_random_draws, args.top_k, m, slice(0, m),
                    pca_components, evr,
                    model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag, base_avg,
                )

                # ---------- Pack ----------
                summary = {
                    "cell": {
                        "pe_type": pe_type, "seed": seed, "layer": layer,
                        "lag": args.lag, "side": args.side, "alpha": args.alpha,
                    },
                    "config": {
                        "top_k":            args.top_k,
                        "top_k_var":        K,
                        "m":                int(m),
                        "H":                int(H),
                        "num_random_draws": args.num_random_draws,
                        "rng_seed":         int(rng_seed),
                    },
                    "baseline_alpha0_fam_losses": base_losses,
                    "baseline_alpha0_fam_avg":    base_avg,

                    "score_svd_full":           {"delta": float(score_full_delta),
                                                 "var_footprint": svd_var_fp_full,
                                                 "fam_losses": score_full_losses,
                                                 "singular_values": [float(x) for x in S[:args.top_k]]},
                    "score_svd_top_K":          {"delta": float(score_top_delta),
                                                 "rank":  int(score_top_rank),
                                                 "fam_losses": score_top_losses},
                    "score_svd_bottom":         {"delta": float(score_bot_delta),
                                                 "rank":  int(score_bot_rank),
                                                 "fam_losses": score_bot_losses},
                    "score_svd_top_K_mass":     top_K_mass,

                    "random_top_K":             rand_top,
                    "random_bottom":            rand_bot,
                    "random_full":              rand_full,

                    # Analytic expectation: uniform random in m-dim PCA space
                    # has E[var footprint] = top_k * sum(evr) / m.
                    "expected_random_full_var_footprint": float(args.top_k * float(evr.sum()) / m),

                    "wall_time_sec": time.time() - t0,
                }
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"  Wrote {out_path.name}  "
                      f"score_full Δ={score_full_delta:+.4f}  "
                      f"score_top Δ={score_top_delta:+.4f}  "
                      f"score_bot Δ={score_bot_delta:+.4f}  "
                      f"rand_top Δ={rand_top['delta_mean']:+.4f}  "
                      f"rand_bot Δ={rand_bot['delta_mean']:+.4f}  "
                      f"top_K_mass={top_K_mass:.3f}  "
                      f"t={summary['wall_time_sec']:.0f}s")

    print("Done.")


if __name__ == "__main__":
    main()
