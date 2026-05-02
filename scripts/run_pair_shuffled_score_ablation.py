"""
Pair-shuffled score-model ablation (Sprint 2 of the reviewer-response plan).

Reviewer-requested control on whether SBTG's signal requires joint
structure. The Optuna null-contrast objective in
src/sbtg/estimators/localized_multiblock_trainer.py tunes hyperparameters
*using* a real-vs-shuffled score-product ratio, but the trained score
model uses real (un-shuffled) data. The reviewer wants the score model
itself to be re-trained on data where the joint is destroyed, so any
directions it produces are by construction not coupling-aligned.

Experiment per (pe, seed, layer):

  1. Load train + test activations at this layer.
  2. PCA-project using the cell's existing pca_components / mu_l so the
     shuffled-trained model lives in the same coordinate system as the
     real model (apples-to-apples for ablation).
  3. Slice into width-w windows (w=16).
  4. Pair-shuffle the LAST position (block w-1) across windows in both
     train and test: window i's block w-1 is replaced by window perm[i]'s
     block w-1. This destroys the joint between block w-1 (the focal
     block from which M_r reads) and every earlier block, while keeping
     every per-position marginal intact.
  5. Train a fresh DSM score model on the shuffled training windows,
     reusing the cell's Optuna-tuned hyperparameters from the per-seed
     analysis JSON.
  6. Compute M_bar_r on the shuffled test windows. By construction these
     should be ≈ 0 off-diagonal at lag 1.
  7. Extract top-3 right SVs of M_bar[1], back-project to hidden through
     the same pca_components, run ablation on the REAL model through the
     same forward hook used by score-SVD / probe-hidden.

Compared against the cell's existing score_lag1_k3 ablation Δ (read from
the analysis JSON / dose_response.json):

  * If Δ_shuffled ≈ Δ_random (≈ 0.3–0.5 nats for variance-aligned random):
    the score model needs joint structure. SBTG's signal is real.
  * If Δ_shuffled ≈ Δ_score_real: the score model is learning marginal
    structure that produces ablation-loaded directions even without
    joint. Paper's score-vs-random framing needs pulling back.

Output (one JSON per cell):
  results/<out-dir>/<pe>_seed<N>_L<layer>_pair_shuffled.json

Usage
-----
python scripts/run_pair_shuffled_score_ablation.py \\
    --summary-path results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
    --models-dir   results/transformer_pos_models_20260419_114958 \\
    --data-dir     data/transformer_pos_cluster \\
    --out-dir      results/lagpair_ablation_3seed/pair_shuffled_score \\
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
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.data.transformer_variants import create_transformer_variant
from src.sbtg.estimators.localized_multiblock_trainer import (
    extract_windows,
    train_score_model_layer,
)
from src.sbtg.estimators.localized_multiblock import LocalizedMultiBlockEstimator
from scripts.run_ablation_study import (
    _load_test_families,
    _masked_ce,
    _eval_with_dirs,
)


# ---------------------------------------------------------------------------
# PCA projection (consistent with run_lagpair_analysis / run_positional_analysis)
# ---------------------------------------------------------------------------

def _pca_project_layer(
    acts_l: np.ndarray, pca_components: np.ndarray, mu_l: np.ndarray
) -> np.ndarray:
    """acts_l: (N, T, H) → (N, T, m) using per-cell components / mean."""
    N, T, H = acts_l.shape
    flat = (acts_l.reshape(-1, H) - mu_l[None, :])
    return (flat @ pca_components.T).reshape(N, T, -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Pair shuffle
# ---------------------------------------------------------------------------

def _pair_shuffle_block(
    windows_blk: np.ndarray, block_idx: int, rng: np.random.Generator,
) -> np.ndarray:
    """Replace `block_idx` block in each window with a random other window's.

    windows_blk: (N_windows, w, m)
    Returns: copy of same shape with block at `block_idx` permuted across windows.
    """
    out = windows_blk.copy()
    perm = rng.permutation(out.shape[0])
    out[:, block_idx, :] = windows_blk[perm, block_idx, :]
    return out


def _flatten_windows(windows_blk: np.ndarray) -> np.ndarray:
    """(N, w, m) → (N, w*m)."""
    N, w, m = windows_blk.shape
    return windows_blk.reshape(N, w * m).astype(np.float32)


# ---------------------------------------------------------------------------
# Eval helpers (mirror Sprint 1)
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
    p.add_argument("--lag",              type=int, default=1)
    p.add_argument("--top-k",            type=int, default=3)
    p.add_argument("--alpha",            type=float, default=2.0)
    p.add_argument("--side",             type=str, default="source",
                   choices=["source", "target"])
    p.add_argument("--score-epochs",     type=int, default=50,
                   help="DSM training epochs for the shuffled score model.")
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--w",                type=int, default=16,
                   help="Window width (must match the cell's analysis JSON).")
    p.add_argument("--max-lag",          type=int, default=14)
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
            train_acts_pt = seed_dir / "train_acts.npy"
            test_acts_pt  = seed_dir / "test_acts.npy"
            analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
            if not all(p.exists() for p in (model_pt, train_acts_pt, test_acts_pt, analysis_file)):
                print(f"  Skipping {pe_type} seed {seed}: required artifacts missing.")
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)
            best_hp = analysis.get("best_hp", {})

            train_full = np.load(train_acts_pt, mmap_mode="r")  # (N, n_layers+1, T, H)
            test_full  = np.load(test_acts_pt,  mmap_mode="r")

            real_model = create_transformer_variant(config, pe_type)
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            real_model.load_state_dict(state)
            real_model.to(args.device)

            for layer in args.layers:
                out_path = out_dir / f"{pe_type}_seed{seed}_L{layer}_pair_shuffled.json"
                if out_path.exists():
                    print(f"  SKIP (exists): {out_path.name}")
                    continue

                t0 = time.time()
                layer_module = real_model.transformer.h[layer - 1]
                ld           = analysis["layer_stats"][layer - 1]

                pca_components = np.asarray(ld["pca_components"], dtype=np.float32)  # (m, H)
                mu_l           = np.asarray(ld["mu_l"], dtype=np.float32)
                evr            = np.asarray(ld.get("pca_explained_var_ratio", []), dtype=np.float32)
                m, H = pca_components.shape
                if evr.size == 0:
                    evr = np.full(m, 1.0 / m, dtype=np.float32)

                # Per-cell HP (fall back to defaults if not present in best_hp)
                # best_hp may be a dict keyed by layer-string or a flat dict; we try both.
                hp_layer = (best_hp.get(str(layer)) or best_hp.get(layer)
                            or best_hp if isinstance(best_hp, dict) else {})
                if not isinstance(hp_layer, dict):
                    hp_layer = {}
                sigma      = float(hp_layer.get("sigma", 0.3))
                hidden_dim = int(hp_layer.get("hidden_dim", 256))
                lr         = float(hp_layer.get("lr", 1e-3))

                # Project this layer's acts (train + test) to PCA space
                t_proj = time.time()
                train_layer = np.asarray(train_full[:, layer], dtype=np.float32)
                test_layer  = np.asarray(test_full[:, layer],  dtype=np.float32)
                u_train = _pca_project_layer(train_layer, pca_components, mu_l)
                u_test  = _pca_project_layer(test_layer,  pca_components, mu_l)
                proj_secs = time.time() - t_proj

                # Window
                t_win = time.time()
                W_train = extract_windows(u_train, args.w)  # (N, num_w, w*m)
                W_test  = extract_windows(u_test,  args.w)
                # Reshape to (N*num_w, w, m) so we can shuffle a position
                Nt, NWt, _ = W_train.shape
                Ne, NWe, _ = W_test.shape
                W_train = W_train.reshape(Nt * NWt, args.w, m)
                W_test  = W_test.reshape(Ne * NWe, args.w, m)
                win_secs = time.time() - t_win

                rng_seed_cell = args.rng_seed_base + pe_to_idx.get(pe_type, 0) * 1_000_000 \
                                + int(seed) * 10_000 + int(layer) * 100
                rng = np.random.default_rng(rng_seed_cell)

                # Pair-shuffle the LAST position (block w-1) in train + test
                t_shuf = time.time()
                W_train_sh = _pair_shuffle_block(W_train, args.w - 1, rng)
                W_test_sh  = _pair_shuffle_block(W_test,  args.w - 1, rng)
                shuf_secs = time.time() - t_shuf

                # Train DSM score model on shuffled training data.
                # train_score_model_layer expects (N, num_w, m*w) shape — give it
                # the flattened windows reshaped back to (1, N_total, m*w) so its
                # internal reshape(-1, m*w) works correctly.
                W_train_flat = W_train_sh.reshape(-1, args.w * m).astype(np.float32)
                W_test_flat  = W_test_sh.reshape(-1, args.w * m).astype(np.float32)
                # Validation split: take last 10% of shuffled train as val
                n_val = max(1, len(W_train_flat) // 10)
                W_val_flat = W_train_flat[-n_val:]
                W_tr_flat  = W_train_flat[:-n_val]
                # Wrap in (1, N, m*w) shape since the trainer flattens via reshape(-1, m*w)
                t_train = time.time()
                shuffled_score_model = train_score_model_layer(
                    train_windows=W_tr_flat[None, ...],
                    val_windows=W_val_flat[None, ...],
                    m=m, w=args.w,
                    epochs=args.score_epochs,
                    lr=lr, sigma=sigma, hidden_dim=hidden_dim,
                    batch_size=args.score_batch_size,
                    device=args.device,
                )
                train_secs = time.time() - t_train

                # Compute M_bar_r on shuffled TEST data using shuffled-trained model
                t_est = time.time()
                estimator = LocalizedMultiBlockEstimator(
                    m=m, w=args.w, max_lag=args.max_lag, skip_edges=4,
                )
                # estimator expects (num_test_seqs, num_windows, m*w)
                # We reshape back to (Ne, NWe, w*m)
                test_windows_for_est = W_test_sh.reshape(Ne, NWe, args.w * m).astype(np.float32)
                stats = estimator.estimate(
                    shuffled_score_model, test_windows_for_est, device=args.device,
                )
                M_bar_r_shuffled = stats["M_bar_r"]  # (max_lag+1, m, m)
                A_r_shuffled     = stats["A_r"]
                est_secs = time.time() - t_est

                # Extract top-k right SVs of M_bar at lag=1 from shuffled model
                U, S, Vh = np.linalg.svd(M_bar_r_shuffled[args.lag])
                shuffled_dirs_m = Vh[: args.top_k, :].astype(np.float32)

                # Free the shuffled score model & windowing arrays before GPU ablation
                del shuffled_score_model, W_train, W_test, W_train_sh, W_test_sh
                del W_train_flat, W_test_flat, W_tr_flat, W_val_flat
                del u_train, u_test, train_layer, test_layer, test_windows_for_est
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Run ablation on the REAL transformer with these directions
                shuffled_dirs_h = _to_hidden(shuffled_dirs_m, pca_components)

                t_abl = time.time()
                base_losses = _baseline_alpha0_loss(real_model, loaders, args.device)
                base_avg    = _fam_avg(base_losses)
                shuffled_fam = _eval_one(
                    shuffled_dirs_h, real_model, loaders, layer_module, mu_l,
                    args.alpha, args.device, args.side, args.lag,
                )
                shuffled_delta = _fam_avg(shuffled_fam) - base_avg
                abl_secs = time.time() - t_abl

                summary = {
                    "cell": {
                        "pe_type": pe_type, "seed": seed, "layer": layer,
                        "lag": args.lag, "side": args.side, "alpha": args.alpha,
                    },
                    "config": {
                        "m": int(m), "H": int(H), "top_k": int(args.top_k),
                        "w": int(args.w), "max_lag": int(args.max_lag),
                        "score_epochs":     int(args.score_epochs),
                        "score_batch_size": int(args.score_batch_size),
                        "rng_seed_cell":    int(rng_seed_cell),
                        "shuffle":          "block_w_minus_1",
                        "score_hp_used":    {"sigma": sigma,
                                             "hidden_dim": hidden_dim, "lr": lr},
                    },
                    "baseline_alpha0_fam_losses": base_losses,
                    "baseline_alpha0_fam_avg":    base_avg,
                    "shuffled_score": {
                        "delta":         float(shuffled_delta),
                        "fam_losses":    shuffled_fam,
                        "var_footprint": _variance_footprint(shuffled_dirs_m, evr),
                        "singular_values_M_bar_lag1": [float(x) for x in S[: args.top_k]],
                        "A_r_shuffled":              [float(x) for x in A_r_shuffled],
                    },
                    "timings_sec": {
                        "proj":  proj_secs, "win": win_secs, "shuffle": shuf_secs,
                        "train": train_secs, "estimate": est_secs, "ablation": abl_secs,
                        "total": time.time() - t0,
                    },
                }
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"  {pe_type} s={seed} L={layer}  "
                      f"shuffled Δ={shuffled_delta:+.4f}  "
                      f"A_r[1]_shuf={A_r_shuffled[args.lag]:.4f}  "
                      f"(train={train_secs:.0f}s  est={est_secs:.0f}s  abl={abl_secs:.0f}s  "
                      f"total={summary['timings_sec']['total']:.0f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
