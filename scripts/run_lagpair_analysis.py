"""
Extended metric suite analysis using windowed score models.

Uses the existing PCA → window → single-score-model-per-layer pipeline, then
computes an extended metric suite from the per-position operators M_r(i):

  A_r  = ||M_bar_r||_F                      (lag amplitude — is there signal?)
  S_r  = 1 - E[||Delta_r||^2]/E[||M_r||^2]  (lag stationarity — shift-invariant?)
  C_r  = sigma_1^2 / ||M_bar_r||^2_F        (concentration — few dirs dominate?)
  AS_r = A_r * S_r                           (stable lag mass — strong AND stationary?)

One score model per (pe_type, seed, layer) — all lags extracted from the same
model via different block pairs in the score.  This is ~11x fewer score models
than the per-lag approach.

Subspace extraction:
  Stable subspace:    top-k left SVs of M_bar_r (in PCA space → back-project to hidden)
  Endpoint-specific:  top-k eigvecs of Gamma_r = E[Delta Delta^T] (same)

NOTE on ablation: directions extracted in PCA space and back-projected to hidden
space are constrained to the PCA subspace.  Random directions within the same
subspace cause comparable damage.  The metric suite (not ablation) is the
primary output of this script.

Usage:
  python scripts/run_lagpair_analysis.py \\
    --models-dir  transformer_pos_models_20260419_114958 \\
    --data-dir    data/transformer_pos_cluster \\
    --out-dir     lagpair_analysis \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 1 2 3 4 \\
    --max-lag 14 \\
    --top-k 3 \\
    --tuning-trials 150 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.estimators.localized_multiblock_trainer import (
    MinimalMLPScoreNet,
    extract_windows,
    train_score_model_layer,
    tune_score_model_null_contrast,
)
from src.sbtg.estimators.localized_multiblock import LocalizedMultiBlockEstimator
from src.sbtg.data.transformer_variants import (
    GPT2Config, create_transformer_variant,
)


# ---------------------------------------------------------------------------
# PCA helpers (same as run_positional_analysis.py)
# ---------------------------------------------------------------------------

def fit_pca_layer(train_acts_l, pca_dim):
    """train_acts_l: (N, seq_len, hidden_size)  →  (pca, mu_l)"""
    N, seq_len, hidden_size = train_acts_l.shape
    flat = train_acts_l.reshape(-1, hidden_size)
    mu_l = flat.mean(axis=0)
    pca = PCA(n_components=pca_dim)
    pca.fit(flat)
    return pca, mu_l


def transform_pca_layer(pca, acts_l):
    """acts_l: (N, seq_len, hidden_size)  →  (N, seq_len, pca_dim)"""
    N, seq_len, hidden_size = acts_l.shape
    flat = acts_l.reshape(-1, hidden_size)
    return pca.transform(flat).reshape(N, seq_len, -1)


# ---------------------------------------------------------------------------
# Extended metric suite from M_r_i
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    M_r_i: np.ndarray,
    M_bar_r: np.ndarray,
    skip_edges: int = 4,
    top_k: int = 3,
    side: str = "source",
) -> Dict[int, Dict[str, Any]]:
    """Compute S_r, C_r, AS_r and extract subspaces from M_r_i.

    Parameters
    ----------
    M_r_i : (max_lag+1, num_windows, m, m) — per-position operators.
        M_r is generally non-symmetric for r > 0 because it's
        ``-E[s^(w-1) s^(w-r-1)^T]`` — the row index is the query/target
        position, the column index is the lagged/source position.
    M_bar_r : (max_lag+1, m, m) — position-averaged operators
    skip_edges : positions to skip at boundaries
    top_k : number of subspace directions
    side : "source" or "target".  Determines which side of the SVD of
        ``M_bar_r`` is used as the ``stable_dirs``:
          - "source"  -> top-k right singular vectors V (col-space of M),
                         which span the source position's score space
                         (correct for ablating source positions).
          - "target"  -> top-k left singular vectors U (row-space of M),
                         which span the target/query position's score space
                         (correct for ablating target positions).
        Default "source" matches the principled convention in
        src/sbtg/evaluation/interventions.py::SingularAblationHook.

    Returns
    -------
    dict : lag -> {A_r, S_r, C_r, AS_r, stable_dirs, endpoint_dirs, ...}
    """
    if side not in ("source", "target"):
        raise ValueError(f"side must be 'source' or 'target', got {side!r}")

    n_lags, n_windows, m, _ = M_r_i.shape

    lo = skip_edges
    hi = n_windows - skip_edges
    if hi <= lo:
        lo, hi = 0, n_windows

    M_valid = M_r_i[:, lo:hi]  # (n_lags, n_valid, m, m)
    n_valid = M_valid.shape[1]

    results = {}

    for r in range(n_lags):
        X_bar = np.nan_to_num(M_bar_r[r], nan=0.0, posinf=0.0, neginf=0.0)
        M_r = np.nan_to_num(M_valid[r], nan=0.0, posinf=0.0, neginf=0.0)
        Delta = M_r - X_bar[None, :, :]  # (n_valid, m, m)

        # --- A_r ---
        A_r = float(np.linalg.norm(X_bar, 'fro'))

        # --- S_r ---
        num_S = np.mean(np.linalg.norm(Delta, ord='fro', axis=(1, 2)) ** 2)
        den_S = np.mean(np.linalg.norm(M_r, ord='fro', axis=(1, 2)) ** 2) + 1e-8
        S_r = float(1.0 - num_S / den_S)

        # --- C_r ---
        mat = X_bar
        if not np.all(np.isfinite(mat)):
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, sv, Vh = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            U = np.eye(m)[:, :min(m, m)]
            sv = np.zeros(min(m, m))
            Vh = np.eye(m)[:min(m, m), :]
        sv_sq_sum = float(np.sum(sv ** 2) + 1e-8)
        C_r = float(sv[0] ** 2 / sv_sq_sum)

        # --- AS_r ---
        AS_r = A_r * max(S_r, 0.0)  # clamp S_r at 0 for product

        # --- Stable subspace: side-aware SV selection ---
        # M = U Σ V^T.  For source-side ablation, ablate the column-space
        # (right SVs, rows of Vh).  For target-side, ablate the row-space
        # (left SVs, columns of U).  This matches interventions.py.
        if side == "source":
            stable_dirs = Vh[:top_k, :]  # (k, m) — right SVs in PCA space
        else:  # "target"
            stable_dirs = U[:, :top_k].T  # (k, m) — left SVs in PCA space

        # --- Endpoint-specific: top eigvecs of Gamma_r (left-side variant) ---
        # Gamma_r is built from Delta Delta^T (target-side covariance) when
        # side == "target", and from Delta^T Delta (source-side) when
        # side == "source", to align with stable_dirs' side.
        Gamma_r = np.zeros((m, m))
        if side == "source":
            for i in range(n_valid):
                Gamma_r += Delta[i].T @ Delta[i]
        else:
            for i in range(n_valid):
                Gamma_r += Delta[i] @ Delta[i].T
        Gamma_r /= n_valid

        eigvals, eigvecs = np.linalg.eigh(Gamma_r)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        endpoint_dirs = eigvecs[:, :top_k].T  # (k, m)

        results[r] = {
            "A_r": A_r,
            "S_r": S_r,
            "C_r": C_r,
            "AS_r": AS_r,
            "side": side,
            "singular_values": sv[:min(10, len(sv))].tolist(),
            "stable_dirs": stable_dirs.tolist(),
            "endpoint_dirs": endpoint_dirs.tolist(),
            "endpoint_eigvals": eigvals[:min(10, len(eigvals))].tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# RoPE theoretical profile
# ---------------------------------------------------------------------------

def compute_rope_theoretical_profile(head_dim: int, max_lag: int, base: float = 10000.0):
    """Theoretical RoPE coupling decay: mean cos(r * theta_k) across frequencies."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    periods = (2 * np.pi / inv_freq).tolist()
    lags_arr = np.arange(max_lag + 1)
    profile = np.array([np.mean(np.cos(r * inv_freq)) for r in lags_arr])
    return lags_arr, profile, periods


# ---------------------------------------------------------------------------
# Ablation evaluation
# ---------------------------------------------------------------------------

def _masked_ce(logits, targets, masks):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks = masks[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    return float((loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8))


def ablation_eval(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    target_layer: nn.Module,
    dirs_h: np.ndarray,
    mu_l: np.ndarray,
    alpha_values: List[float],
    device: str,
    side: str = "source",
    lag: int = 1,
) -> Dict[float, Dict[str, float]]:
    """Ablation sweep: project out directions from layer output, measure loss.

    Applies the projection at a position-mask matching ``side``:
      - "source" — mask positions ``i' = i - lag`` for valid queries i,
                   matching SingularAblationHook's source-side convention
                   in src/sbtg/evaluation/interventions.py.
      - "target" — mask positions i directly.
      - "all"    — legacy behaviour: ablate every position (the original
                   pre-2026-04-25 behaviour, retained for reproducing the
                   "Protocol A" numbers in paper2.tex Table 3).

    Parameters
    ----------
    dirs_h : (k, hidden_size) — directions in hidden space (back-projected from PCA)
    mu_l : (hidden_size,) — centering mean
    side : "source", "target", or "all".  Default "source" matches the
        principled convention.
    lag : int — used only when side == "source"; positions ``i - lag``
        are masked for each valid query position i.
    """
    if side not in ("source", "target", "all"):
        raise ValueError(f"side must be 'source', 'target', or 'all', got {side!r}")

    model.eval()
    model.to(device)

    dirs_t = torch.tensor(dirs_h, dtype=torch.float32).to(device)
    mu_t = torch.tensor(mu_l, dtype=torch.float32).to(device)

    results = {}
    for alpha in alpha_values:

        def _hook_fn(module, inp, out, _alpha=alpha, _dirs=dirs_t, _mu=mu_t,
                     _side=side, _lag=lag):
            h = out[0] if isinstance(out, tuple) else out
            b, seq_len, dim = h.shape
            h_c = h - _mu
            D = _dirs.T  # (hidden, k)
            proj = h_c @ D  # (b, seq, k)
            ablation = proj @ D.T  # (b, seq, hidden)

            if _side == "all":
                h_prime = h - _alpha * ablation
            else:
                mask = torch.zeros((1, seq_len, 1), device=h.device)
                for i in range(seq_len):
                    if _side == "source":
                        src = i - _lag
                        if 0 <= src < seq_len:
                            mask[:, src, :] = 1.0
                    else:  # "target"
                        mask[:, i, :] = 1.0
                h_prime = h - _alpha * ablation * mask

            return (h_prime,) + out[1:] if isinstance(out, tuple) else h_prime

        handle = target_layer.register_forward_hook(_hook_fn)
        fam_losses = {}
        with torch.no_grad():
            for fam, loader in loaders.items():
                losses = []
                for seqs, masks in loader:
                    seqs, masks = seqs.to(device), masks.to(device)
                    out = model(input_ids=seqs)
                    losses.append(_masked_ce(out.logits, seqs, masks))
                fam_losses[fam] = float(np.mean(losses))
        handle.remove()
        results[alpha] = fam_losses

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extended metric suite analysis using windowed score models"
    )
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="1-indexed layers")
    parser.add_argument("--w", type=int, default=16, help="Window size")
    parser.add_argument("--max-lag", type=int, default=14,
                        help="Maximum lag (must be < w)")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tuning-trials", type=int, default=150)
    parser.add_argument("--score-epochs", type=int, default=50)
    parser.add_argument("--max-windows-gb", type=float, default=20.0,
                        help="Cap on the materialized windows array per split "
                             "(train/val/test).  Subsamples sequences before "
                             "extract_windows if (num_windows × w × m × 4 bytes × N) "
                             "would exceed this cap.  At seq_len=64 the default "
                             "is non-binding; at seq_len=256 it triggers and prevents "
                             "the OOM that blew up the original rope_grid run.")
    parser.add_argument("--alpha-values", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation (measurement only)")
    parser.add_argument("--ablation-side", type=str, default="source",
                        choices=["source", "target", "all"],
                        help=("Ablation protocol: 'source' uses right SVs of "
                              "M_bar (col-space, source-position score) and "
                              "masks i'=i-r; 'target' uses left SVs (row-space, "
                              "query-position score) and masks i; 'all' is the "
                              "legacy pre-2026-04-25 behaviour (left SVs, no "
                              "position mask).  Default 'source' is the "
                              "principled choice; matches "
                              "src/sbtg/evaluation/interventions.py."))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    assert args.max_lag < args.w, f"max_lag={args.max_lag} must be < w={args.w}"

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    score_dir = out_dir / "score_models"
    score_dir.mkdir(exist_ok=True)

    pca_dim = args.pca_dim
    w = args.w

    # Load metadata for model config
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        config = GPT2Config(
            vocab_size=meta["vocab_size"],
            n_positions=meta["seq_len"],
            n_embd=256, n_layer=4, n_head=4, n_inner=1024,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        families = meta.get("families", [])
    else:
        config = None
        families = []

    # Load test data loaders for ablation
    loaders = {}
    if not args.skip_ablation and families:
        from torch.utils.data import TensorDataset, DataLoader
        for fam in families:
            sp = data_dir / f"{fam}_test.npy"
            mp = data_dir / f"{fam}_test_mask.npy"
            if sp.exists() and mp.exists():
                seqs = torch.tensor(np.load(sp), dtype=torch.long)
                masks = torch.tensor(np.load(mp), dtype=torch.long)
                loaders[fam] = DataLoader(
                    TensorDataset(seqs, masks), batch_size=128, shuffle=False)

    # ===================================================================
    # Main loop: per (pe_type, seed)
    # ===================================================================
    all_metrics = {}

    for pe_type in args.pe_types:
        for seed in args.seeds:
            seed_dir = models_dir / f"{pe_type}_seed{seed}"
            train_path = seed_dir / "train_acts.npy"
            test_path = seed_dir / "test_acts.npy"

            if not train_path.exists() or not test_path.exists():
                print(f"  Skip {pe_type} seed {seed}: no activations")
                continue

            train_acts = np.load(train_path)  # (N, n_layers+1, seq_len, d)
            test_acts = np.load(test_path)
            N_train, n_layers_p1, seq_len, d = train_acts.shape
            N_test = test_acts.shape[0]

            print(f"\n{'='*60}")
            print(f"  {pe_type.upper()} seed={seed}  "
                  f"N_train={N_train} N_test={N_test} seq_len={seq_len} d={d}")
            print(f"{'='*60}")

            # Load transformer model for ablation
            model = None
            if not args.skip_ablation and config is not None:
                model_pt = seed_dir / "model.pt"
                if model_pt.exists():
                    model = create_transformer_variant(config, pe_type)
                    state = torch.load(model_pt, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
                    model.to(args.device)
                    model.eval()

            for layer in args.layers:
                if layer >= n_layers_p1:
                    continue

                print(f"\n  --- Layer {layer} ---")

                # PCA
                pca, mu_l = fit_pca_layer(train_acts[:, layer], pca_dim)
                train_pca = transform_pca_layer(pca, train_acts[:, layer])
                test_pca = transform_pca_layer(pca, test_acts[:, layer])

                # PCA variance diagnostic
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                print(f"    PCA cumvar at m={pca_dim}: {cumvar[-1]:.4f}")

                # Windows (memory-capped: extract_windows materializes
                # a (N, num_windows, w*m) array which can exceed RAM at long
                # context.  At seq_len=64 this is ~9 GiB float32 — fine.
                # At seq_len=256 it's ~250 GiB at the same N — OOMs.
                # Subsample sequences before windowing to stay under
                # max_windows_gb; downstream score-model training and M_r
                # estimation are unaffected because they only need
                # (N × num_windows) total observations to be sufficient,
                # not full per-sequence coverage.)
                val_split = int(0.9 * N_train)
                _bytes_per_seq = (seq_len - w + 1) * w * pca_dim * 4  # float32
                _cap_gb = float(args.max_windows_gb)
                _max_N = max(1, int(_cap_gb * (1024 ** 3) / _bytes_per_seq))

                def _maybe_subsample(arr: np.ndarray, max_N: int, label: str) -> np.ndarray:
                    if len(arr) > max_N:
                        rng = np.random.default_rng(seed * 1000 + layer)
                        idx = rng.choice(len(arr), max_N, replace=False)
                        print(f"    [memory-cap] {label}: {len(arr)} → {max_N} seqs "
                              f"(per-seq window cost ≈ {_bytes_per_seq/1024**3:.2f} GiB, "
                              f"cap {_cap_gb:.1f} GiB)")
                        return arr[idx]
                    return arr

                train_pca_sub = _maybe_subsample(
                    train_pca[:val_split].astype(np.float32, copy=False),
                    _max_N, "train")
                val_pca_sub = _maybe_subsample(
                    train_pca[val_split:].astype(np.float32, copy=False),
                    max(1, _max_N // 9), "val")
                test_pca_sub = _maybe_subsample(
                    test_pca.astype(np.float32, copy=False),
                    _max_N, "test")
                train_windows = extract_windows(train_pca_sub, w).astype(np.float32, copy=False)
                val_windows   = extract_windows(val_pca_sub,   w).astype(np.float32, copy=False)
                test_windows  = extract_windows(test_pca_sub,  w).astype(np.float32, copy=False)

                mw = pca_dim * w

                # Check for saved score model
                sm_tag = f"{pe_type}_s{seed}_L{layer}"
                sm_path = score_dir / f"{sm_tag}.pt"
                hp_path = score_dir / f"{sm_tag}_hp.json"

                if sm_path.exists() and hp_path.exists():
                    with open(hp_path) as f:
                        best_hp = json.load(f)
                    score_model = MinimalMLPScoreNet(
                        mw, hidden_dim=best_hp["hidden_dim"])
                    score_model.load_state_dict(
                        torch.load(sm_path, map_location="cpu", weights_only=True))
                    score_model.to(args.device)
                    print(f"    Loaded saved score model")
                else:
                    # Null-contrast HP tuning
                    print(f"    NC tuning ({args.tuning_trials} trials)...")
                    best_hp = tune_score_model_null_contrast(
                        train_windows.reshape(-1, mw),
                        val_windows.reshape(-1, mw),
                        in_features=mw,
                        m=pca_dim,
                        w=w,
                        n_trials=args.tuning_trials,
                        device=args.device,
                        seed=seed * 100 + layer,
                    )

                    # Train full score model
                    print(f"    Training score model ({args.score_epochs} epochs)...")
                    score_model = train_score_model_layer(
                        train_windows, val_windows,
                        m=pca_dim, w=w,
                        epochs=args.score_epochs,
                        lr=best_hp["lr"],
                        sigma=best_hp["sigma"],
                        hidden_dim=best_hp["hidden_dim"],
                        device=args.device,
                    )

                    # Save
                    torch.save(score_model.state_dict(), sm_path)
                    with open(hp_path, "w") as f:
                        json.dump(best_hp, f)

                # Run estimator — gives M_r_i for all lags at once
                estimator = LocalizedMultiBlockEstimator(
                    m=pca_dim, w=w, max_lag=args.max_lag,
                    skip_edges=4,
                )
                stats = estimator.estimate(
                    score_model, test_windows, device=args.device)

                M_r_i = stats["M_r_i"]      # (max_lag+1, num_windows, m, m)
                M_bar_r = stats["M_bar_r"]   # (max_lag+1, m, m)

                # Compute extended metric suite
                ext = compute_extended_metrics(
                    M_r_i, M_bar_r,
                    skip_edges=estimator.skip_edges,
                    top_k=args.top_k,
                    side=args.ablation_side,
                )

                # Operator autocorrelation
                autocorr = LocalizedMultiBlockEstimator.compute_operator_autocorrelation(
                    M_r_i)

                # Print summary
                print(f"    {'lag':>4s}  {'A_r':>8s}  {'S_r':>8s}  {'C_r':>8s}  {'AS_r':>8s}")
                for r in range(min(args.max_lag + 1, len(ext))):
                    if r in ext:
                        e = ext[r]
                        print(f"    {r:4d}  {e['A_r']:8.4f}  {e['S_r']:8.4f}  "
                              f"{e['C_r']:8.3f}  {e['AS_r']:8.4f}")

                # ----- Ablation (lag=1, if model available) -----
                ablation_data = {}
                if model is not None and loaders and not args.skip_ablation and 1 in ext:
                    layer_module = model.transformer.h[layer - 1]

                    # Back-project PCA directions to hidden space
                    pca_components = pca.components_  # (pca_dim, hidden_size)

                    for label in ["stable", "endpoint"]:
                        dirs_key = f"{label}_dirs"
                        dirs_pca = np.array(ext[1][dirs_key])  # (k, m)
                        # Back-project: (k, m) @ (m, d) = (k, d)
                        dirs_hidden = (dirs_pca @ pca_components).astype(np.float32)
                        # Orthonormalize in hidden space
                        Q, _ = np.linalg.qr(dirs_hidden.T)
                        dirs_hidden = Q.T[:args.top_k]

                        abl_results = ablation_eval(
                            model, loaders, layer_module,
                            dirs_hidden, mu_l, args.alpha_values, args.device,
                            side=args.ablation_side, lag=1,
                        )
                        ablation_data[f"ablation_{label}"] = {
                            str(a): v for a, v in abl_results.items()
                        }

                    # Random control: drawn in the SAME m-dim PCA subspace
                    # where stable/endpoint live, then back-projected to hidden.
                    # This is the strict ("Protocol B") matched control.  The
                    # legacy hidden-space random was a too-lenient null because
                    # most random directions in R^d miss the PCA active subspace.
                    rng = np.random.default_rng(seed * 100 + layer)
                    rand_dirs_m = rng.standard_normal(
                        (args.top_k, args.pca_dim)
                    ).astype(np.float32)
                    q, _ = np.linalg.qr(rand_dirs_m.T)
                    rand_dirs_m = q.T  # (k, m) orthonormal in PCA space
                    rand_dirs = (rand_dirs_m @ pca_components).astype(np.float32)
                    Q_r, _ = np.linalg.qr(rand_dirs.T)
                    rand_dirs = Q_r.T[:args.top_k]  # re-orthonormalize in hidden

                    abl_rand = ablation_eval(
                        model, loaders, layer_module,
                        rand_dirs, mu_l, args.alpha_values, args.device,
                        side=args.ablation_side, lag=1,
                    )
                    ablation_data["ablation_random"] = {
                        str(a): v for a, v in abl_rand.items()
                    }

                # Save subspace directions as npy
                for r in ext:
                    r_tag = f"{sm_tag}_r{r}"
                    np.save(out_dir / f"{r_tag}_stable_dirs.npy",
                            np.array(ext[r]["stable_dirs"]))
                    np.save(out_dir / f"{r_tag}_endpoint_dirs.npy",
                            np.array(ext[r]["endpoint_dirs"]))

                # Build metrics dict for this (pe, seed, layer)
                lags_dict = {}
                for r in ext:
                    lag_entry = {
                        "A_r": ext[r]["A_r"],
                        "S_r": ext[r]["S_r"],
                        "C_r": ext[r]["C_r"],
                        "AS_r": ext[r]["AS_r"],
                        "singular_values": ext[r]["singular_values"],
                        "endpoint_eigvals": ext[r]["endpoint_eigvals"],
                        "autocorr": autocorr[r].tolist() if r < len(autocorr) else [],
                    }
                    # Add ablation data for lag 1 only
                    if r == 1:
                        lag_entry.update(ablation_data)
                    lags_dict[r] = lag_entry

                tag = f"{pe_type}_s{seed}_L{layer}"
                all_metrics[tag] = {
                    "pe_type": pe_type,
                    "seed": seed,
                    "layer": layer,
                    "pca_dim": pca_dim,
                    "w": w,
                    "max_lag": args.max_lag,
                    "d": d,
                    "seq_len": seq_len,
                    "pca_cumvar": float(cumvar[-1]),
                    "best_hp": best_hp,
                    "SI": float(stats["SI"]),
                    "RDI": float(stats["RDI"]),
                    "beta": float(stats["beta"]),
                    "A_r_original": stats["A_r"].tolist(),
                    "H_r_original": stats["H_r"].tolist(),
                    "lags": lags_dict,
                }

    # ===================================================================
    # Figures
    # ===================================================================

    PE_COLORS = {"rope": "#2CA02C", "alibi": "#FF7F0E", "absolute": "#1F77B4"}
    PE_LABELS = {"rope": "RoPE", "alibi": "ALiBi", "absolute": "Absolute"}
    all_lags = list(range(args.max_lag + 1))

    print("\n" + "=" * 80)
    print("  EXTENDED METRIC SUITE SUMMARY")
    print("=" * 80)

    for pe_type in args.pe_types:
        print(f"\n--- {PE_LABELS.get(pe_type, pe_type)} ---")
        for layer in args.layers:
            entries = []
            for seed in args.seeds:
                tag = f"{pe_type}_s{seed}_L{layer}"
                if tag in all_metrics and all_metrics[tag]["lags"]:
                    entries.append(all_metrics[tag])
            if not entries:
                continue

            common_lags = sorted(set.intersection(
                *[set(int(k) for k in e["lags"].keys()) for e in entries]))
            if not common_lags:
                continue

            print(f"\n  L{layer}:")
            print(f"    {'lag':>4s}  {'A_r':>8s}  {'S_r':>8s}  {'C_r':>8s}  {'AS_r':>8s}")
            for lag in common_lags:
                real_vals = []
                for e in entries:
                    for k in [lag, str(lag)]:
                        if k in e["lags"]:
                            real_vals.append(e["lags"][k])
                            break
                if not real_vals:
                    continue
                a_mean = np.mean([v["A_r"] for v in real_vals])
                s_mean = np.mean([v["S_r"] for v in real_vals])
                c_mean = np.mean([v["C_r"] for v in real_vals])
                as_mean = np.mean([v["AS_r"] for v in real_vals])
                print(f"    {lag:4d}  {a_mean:8.4f}  {s_mean:8.4f}  {c_mean:8.4f}  {as_mean:8.4f}")

    # Helper to collect metric across seeds for a given (pe, layer, metric_key)
    def collect_metric(pe_type, layer, metric_key):
        lags_out, means, stds = [], [], []
        for lag in all_lags:
            vals = []
            for seed in args.seeds:
                tag = f"{pe_type}_s{seed}_L{layer}"
                if tag in all_metrics:
                    for k in [lag, str(lag)]:
                        if k in all_metrics[tag]["lags"]:
                            v = all_metrics[tag]["lags"][k].get(metric_key)
                            if v is not None:
                                vals.append(v)
                            break
            if vals:
                lags_out.append(lag)
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
        return lags_out, means, stds

    def plot_metric(metric_key, ylabel, title, filename, ylim=None):
        n_layers = len(args.layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5),
                                 sharey=bool(ylim))
        if n_layers == 1:
            axes = [axes]
        for li, layer in enumerate(args.layers):
            ax = axes[li]
            for pe in args.pe_types:
                lgs, mn, sd = collect_metric(pe, layer, metric_key)
                if lgs:
                    ax.errorbar(lgs, mn, yerr=sd, fmt="o-",
                                color=PE_COLORS.get(pe, "gray"),
                                label=PE_LABELS.get(pe, pe),
                                capsize=3, linewidth=2, markersize=4)
            ax.set_xlabel("Lag $r$")
            ax.set_title(f"L{layer}")
            if li == 0:
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.2)
        fig.suptitle(title, y=1.02, fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_dir / filename}")

    # --- Figures ---
    plot_metric("A_r", "$A_r$", "Lag Amplitude", "LP_A_r_profile.pdf")
    plot_metric("S_r", "$S_r$", "Lag Stationarity", "LP_S_r_profile.pdf",
                ylim=(-0.05, 1.05))
    plot_metric("C_r", "$C_r$", "Concentration", "LP_C_r_profile.pdf",
                ylim=(-0.05, 1.05))
    plot_metric("AS_r", "$A_r S_r$", "Stable Lag Mass", "LP_AS_r_profile.pdf")

    # --- AS_r with RoPE theoretical overlay ---
    max_lag_val = args.max_lag
    lags_theo, rope_profile, rope_periods = compute_rope_theoretical_profile(
        head_dim=64, max_lag=max_lag_val, base=10000.0)

    n_layers_fig = len(args.layers)
    fig, axes = plt.subplots(1, n_layers_fig, figsize=(4 * n_layers_fig, 3.5),
                             sharey=False)
    if n_layers_fig == 1:
        axes = [axes]
    for li, layer in enumerate(args.layers):
        ax = axes[li]
        for pe in args.pe_types:
            lgs, mn, sd = collect_metric(pe, layer, "AS_r")
            if lgs and mn[0] > 1e-8:
                norm = mn[0]
                ax.errorbar(lgs, [v / norm for v in mn],
                            yerr=[v / norm for v in sd], fmt="o-",
                            color=PE_COLORS.get(pe, "gray"),
                            label=PE_LABELS.get(pe, pe),
                            capsize=3, linewidth=2, markersize=4)
        # RoPE theory
        rp = rope_profile[1:max_lag_val + 1]
        if len(rp) > 0 and abs(rp[0]) > 1e-8:
            ax.plot(range(1, len(rp) + 1), rp / rp[0],
                    "k--", linewidth=1.5, alpha=0.5, label="RoPE theory")
        ax.set_xlabel("Lag $r$")
        ax.set_title(f"L{layer}")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        if li == 0:
            ax.set_ylabel("$A_r S_r$ (normalized)")
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Stable Lag Mass with RoPE Theoretical Envelope", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "LP_AS_r_rope_overlay.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'LP_AS_r_rope_overlay.pdf'}")

    # --- Operator autocorrelation ---
    fig, axes = plt.subplots(len(args.pe_types), n_layers_fig,
                             figsize=(4 * n_layers_fig, 3 * len(args.pe_types)),
                             squeeze=False)
    for pi, pe in enumerate(args.pe_types):
        for li, layer in enumerate(args.layers):
            ax = axes[pi][li]
            for lag_show in [1, 2, 4]:
                ac_all = []
                for seed in args.seeds:
                    tag = f"{pe}_s{seed}_L{layer}"
                    if tag in all_metrics:
                        for k in [lag_show, str(lag_show)]:
                            if k in all_metrics[tag]["lags"]:
                                ac = all_metrics[tag]["lags"][k].get("autocorr", [])
                                if ac:
                                    ac_all.append(ac)
                                break
                if ac_all:
                    min_len = min(len(a) for a in ac_all)
                    ac_arr = np.array([a[:min_len] for a in ac_all])
                    ac_mean = ac_arr.mean(0)
                    ax.plot(np.arange(len(ac_mean)), ac_mean, linewidth=1.5,
                            label=f"lag {lag_show}", alpha=0.8)
            ax.set_xlabel("Shift $\\delta$")
            ax.set_title(f"{PE_LABELS.get(pe, pe)} L{layer}", fontsize=9)
            if li == 0:
                ax.set_ylabel("$C(r, \\delta)$")
            if pi == 0 and li == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Operator Autocorrelation", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "LP_operator_autocorrelation.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'LP_operator_autocorrelation.pdf'}")

    # --- Ablation figure ---
    if not args.skip_ablation and loaders:
        for pe in args.pe_types:
            for layer in args.layers:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for dir_type, color, marker in [
                    ("stable", "tab:blue", "o"),
                    ("endpoint", "tab:red", "s"),
                    ("random", "tab:gray", "^"),
                ]:
                    abl_key = f"ablation_{dir_type}"
                    all_deltas = {a: [] for a in args.alpha_values}

                    for seed in args.seeds:
                        tag = f"{pe}_s{seed}_L{layer}"
                        if tag not in all_metrics:
                            continue
                        lag_data = None
                        for k in [1, "1"]:
                            if k in all_metrics[tag]["lags"]:
                                lag_data = all_metrics[tag]["lags"][k]
                                break
                        if lag_data is None or abl_key not in lag_data:
                            continue
                        abl = lag_data[abl_key]
                        base_losses = abl.get("0.0", abl.get(0.0, {}))
                        for alpha in args.alpha_values:
                            a_losses = abl.get(str(alpha), abl.get(alpha, {}))
                            if a_losses and base_losses:
                                deltas = [a_losses[f] - base_losses[f]
                                          for f in a_losses
                                          if f != "iid_random" and f in base_losses]
                                if deltas:
                                    all_deltas[alpha].append(np.mean(deltas))

                    alphas_p, means_p, stds_p = [], [], []
                    for alpha in sorted(args.alpha_values):
                        if all_deltas[alpha]:
                            alphas_p.append(alpha)
                            means_p.append(np.mean(all_deltas[alpha]))
                            stds_p.append(np.std(all_deltas[alpha])
                                          if len(all_deltas[alpha]) > 1 else 0)
                    if alphas_p:
                        ax.errorbar(alphas_p, means_p, yerr=stds_p,
                                    fmt=f"{marker}-", color=color, label=dir_type,
                                    capsize=3, linewidth=2)

                ax.set_xlabel("Ablation strength $\\alpha$")
                ax.set_ylabel("Mean $\\Delta$ loss")
                ax.set_title(f"{PE_LABELS.get(pe, pe)} L{layer} lag=1")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color="gray", linewidth=0.5)
                fig.tight_layout()
                fig.savefig(fig_dir / f"LP_ablation_{pe}_L{layer}.pdf",
                            bbox_inches="tight")
                plt.close(fig)
        print(f"Saved ablation figures to {fig_dir}/")

    # Save metrics JSON
    json_out = out_dir / "lagpair_metrics.json"
    with open(json_out, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSaved: {json_out}")

    # Print RoPE frequency periods
    print(f"\nRoPE rotation periods (shortest 10): "
          f"{', '.join(f'{p:.1f}' for p in sorted(rope_periods)[:10])} tokens")

    # Score model count
    n_models = len(list(score_dir.glob("*.pt")))
    print(f"\nScore models: {n_models} (one per layer, reusable for all lags)")


if __name__ == "__main__":
    main()
