"""
Mechanistic validation via singular-direction ablation (paper.tex Section 8).

Generates:
  F6  dose_response.pdf     — task-loss vs ablation strength α for top-k SVD directions
  F7  source_vs_target.pdf  — comparison of source-side vs target-side ablation profiles
  F6b extended_ablation.pdf — B2: score-SVD vs probe-direction vs random (per layer)
  T6a ablation_controls.txt — B3: correct-layer vs wrong-layer vs wrong-position table

Also writes dose_response.json with all raw measurements.

New arguments for B2/B3
-----------------------
  --probe-baselines-path   path to probe_baselines_summary.json (B2)
  --wrong-layer            layer index whose SVD dirs are applied at --layers (B3)
  --wrong-position-offset  int; offset hook positions by this amount (B3)
  --ablation-alpha-b3      alpha value used for B3 controls table (default: 2.0)

Usage
-----
python scripts/run_ablation_study.py \\
    --summary-path  results/<run>/analysis_summary.json \\
    --models-dir    results/<run>/models \\
    --data-dir      data/transformer_pos_cluster \\
    --out-dir       results/<run>/ablation \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2 \\
    --layers 2 3 \\
    --lag 1 \\
    --top-k 3 \\
    --alpha-values 0.0 0.5 1.0 2.0 \\
    --probe-baselines-path results/<run>/probes/probe_baselines_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from transformers import GPT2Config

from src.sbtg.data.transformer_variants import create_transformer_variant
from src.sbtg.evaluation.interventions import SingularAblationHook

# Re-use colour / label conventions
PE_COLORS = {"rope": "#2CA02C", "alibi": "#FF7F0E", "absolute": "#1F77B4"}
PE_LABELS = {"rope": "RoPE",    "alibi": "ALiBi",   "absolute": "Absolute"}

FAMILY_COLORS = {
    "variable_lag_copy": "#1F77B4",
    "absolute_anchor":   "#FF7F0E",
    "order_sensitive":   "#2CA02C",
    "distance_bucket":   "#9467BD",
    "iid_random":        "#AAAAAA",
}
FAMILY_LABELS = {
    "variable_lag_copy": "Var-lag copy",
    "absolute_anchor":   "Abs. anchor",
    "order_sensitive":   "Order-sensitive",
    "distance_bucket":   "Dist. bucket",
    "iid_random":        "IID random",
}


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, masks):
        self.seqs  = torch.tensor(seqs,  dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.bool)

    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.masks[i]


def _load_test_families(
    data_dir: Path, families: List[str]
) -> Dict[str, torch.utils.data.DataLoader]:
    loaders = {}
    for fam in families:
        seqs  = np.load(data_dir / f"{fam}_test.npy")
        masks = np.load(data_dir / f"{fam}_test_mask.npy")
        ds    = _TokenDataset(seqs, masks)
        loaders[fam] = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    return loaders


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> float:
    """Masked cross-entropy loss — same formula as training."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks  = masks[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss    = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss    = loss.view(shift_labels.size())
    return float((loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8))


# ---------------------------------------------------------------------------
# Core ablation sweep  (testable, no disk I/O)
# ---------------------------------------------------------------------------

def _eval_with_dirs(
    model:        nn.Module,
    loaders:      Dict[str, torch.utils.data.DataLoader],
    target_layer: nn.Module,
    dirs_h:       torch.Tensor,   # (k, hidden_size) unit-norm directions
    mu_l:         np.ndarray,
    alpha_values: List[float],
    device:       str,
    side:         str = "source",
    lag:          int = 1,
    position_offset: int = 0,
) -> Dict[float, Dict[str, float]]:
    """
    Sweep alpha for a fixed set of ablation directions (arbitrary origin —
    score-SVD, probe-derived, random).  Returns {alpha: {fam: loss}}.
    Supports `position_offset` for wrong-position control (B3).
    """
    from src.sbtg.evaluation.interventions import SingularAblationHook

    # Build a dummy m_bar that will be ignored — we override dirs_h directly
    dummy_m_bar = np.zeros((1, 1, 1))  # will be replaced by hook patch

    results: Dict[float, Dict[str, float]] = {}
    model.eval()
    model.to(device)

    mu_tensor = torch.tensor(mu_l, dtype=torch.float32).to(device)
    dirs_dev  = dirs_h.to(device)

    for alpha in alpha_values:
        hook_handle = None

        def _hook_fn(module, inp, out, _alpha=alpha, _side=side, _lag=lag,
                     _dirs=dirs_dev, _mu=mu_tensor, _offset=position_offset):
            h = out[0] if isinstance(out, tuple) else out
            b, seq_len, dim = h.shape
            h_c = h - _mu
            D   = _dirs.T
            proj    = h_c @ D
            ablation = proj @ D.T
            mask = torch.zeros((1, seq_len, 1), device=h.device)
            for i in range(seq_len):
                src = i - _lag - _offset
                if _side == "source" and 0 <= src < seq_len:
                    mask[:, src, :] = 1.0
                elif _side == "target":
                    tgt = i - _offset
                    if 0 <= tgt < seq_len:
                        mask[:, tgt, :] = 1.0
            h_prime = h - _alpha * ablation * mask
            return (h_prime,) + out[1:] if isinstance(out, tuple) else h_prime

        hook_handle = target_layer.register_forward_hook(_hook_fn)
        fam_losses: Dict[str, float] = {}
        with torch.no_grad():
            for fam, loader in loaders.items():
                losses = []
                for seqs, masks in loader:
                    seqs  = seqs.to(device)
                    masks = masks.to(device)
                    out   = model(input_ids=seqs)
                    losses.append(_masked_ce(out.logits, seqs, masks))
                fam_losses[fam] = float(np.mean(losses))
        hook_handle.remove()
        results[alpha] = fam_losses

    return results


def ablation_sweep(
    model:           nn.Module,
    loaders:         Dict[str, torch.utils.data.DataLoader],
    target_layer:    nn.Module,
    m_bar_r:         np.ndarray,   # (max_lag+1, m, m)
    pca_components:  np.ndarray,   # (m, hidden_size)
    mu_l:            np.ndarray,   # (hidden_size,)
    lag:             int,
    top_k:           int,
    alpha_values:    List[float],
    device:          str,
    sides:           Tuple[str, ...] = ("source", "target"),
    rng_seed:        int = 0,       # cell-specific seed for random-direction control
) -> Dict[str, Any]:
    """
    Sweep ablation strength α for source-side and target-side ablations.

    Returns a dict keyed by side → α → family → mean_loss.
    """
    results: Dict[str, Dict] = {s: {} for s in sides}

    model.eval()
    model.to(device)

    for side in sides:
        hook = SingularAblationHook(
            target_layer   = target_layer,
            m_bar          = m_bar_r,
            pca_components = pca_components,
            mu_l           = mu_l,
            k              = top_k,
            lag            = lag,
            alpha          = 0.0,   # will update per alpha step
            side           = side,
        )

        for alpha in alpha_values:
            hook.alpha = alpha
            hook.register()

            fam_losses: Dict[str, float] = {}
            with torch.no_grad():
                for fam, loader in loaders.items():
                    losses = []
                    for seqs, masks in loader:
                        seqs  = seqs.to(device)
                        masks = masks.to(device)
                        out   = model(input_ids=seqs)
                        losses.append(_masked_ce(out.logits, seqs, masks))
                    fam_losses[fam] = float(np.mean(losses))

            hook.remove()
            results[side][alpha] = fam_losses

    # Random-direction control: same α sweep but ablate random directions
    # drawn in the PCA subspace (fair control at matched m-dim footing),
    # with a CELL-SPECIFIC seed so aggregated ± is across seeds AND random
    # realizations, not just seeds.  See docs/ABLATION_PIPELINE.md.
    results["random_control"] = {}
    rng = np.random.default_rng(rng_seed)
    rand_dirs_m = rng.standard_normal((top_k, pca_components.shape[0])).astype(np.float32)
    # Orthonormalise
    q, _ = np.linalg.qr(rand_dirs_m.T)
    rand_dirs_m = q.T
    # Map to hidden space
    rand_dirs_h = rand_dirs_m @ pca_components  # (top_k, hidden_size)
    # Build a fake m_bar_r whose right singular vectors are these random dirs
    # We'll directly patch the hook's dirs_h attribute
    hook_rand = SingularAblationHook(
        target_layer   = target_layer,
        m_bar          = m_bar_r,
        pca_components = pca_components,
        mu_l           = mu_l,
        k              = top_k,
        lag            = lag,
        alpha          = 0.0,
        side           = "source",
    )
    hook_rand.dirs_h = torch.tensor(rand_dirs_h, dtype=torch.float32)

    for alpha in alpha_values:
        hook_rand.alpha = alpha
        hook_rand.register()
        fam_losses = {}
        with torch.no_grad():
            for fam, loader in loaders.items():
                losses = []
                for seqs, masks in loader:
                    seqs  = seqs.to(device)
                    masks = masks.to(device)
                    out   = model(input_ids=seqs)
                    losses.append(_masked_ce(out.logits, seqs, masks))
                fam_losses[fam] = float(np.mean(losses))
        hook_rand.remove()
        results["random_control"][alpha] = fam_losses

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "lines.linewidth": 1.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def fig_dose_response(
    dose_results: Dict[str, Any],
    pe_type: str,
    seed: int,
    layer: int,
    lag: int,
    families_to_show: Optional[List[str]] = None,
) -> plt.Figure:
    """
    F6: per-family loss vs α for top-k SVD ablation vs random control.

    ``dose_results`` has structure:
        { "source": { alpha: {fam: loss}, ... },
          "target": { alpha: {fam: loss}, ... },
          "random_control": { alpha: {fam: loss}, ... } }
    """
    _apply_style()

    all_families = list(dose_results["source"][
        list(dose_results["source"].keys())[0]].keys())
    if families_to_show is None:
        families_to_show = [f for f in all_families if f != "iid_random"]

    n_fam = len(families_to_show)
    n_cols = min(n_fam, 3)
    n_rows = math.ceil(n_fam / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    alpha_vals  = sorted(dose_results["source"].keys(), key=float)
    alpha_arr   = np.array([float(a) for a in alpha_vals])

    for ax, fam in zip(axes_flat, families_to_show):
        for side, ls, label in [
            ("source",         "-",  "Source-side SVD"),
            ("target",         "--", "Target-side SVD"),
            ("random_control", ":",  "Random dir. ctrl"),
        ]:
            losses = np.array([dose_results[side][a][fam] for a in alpha_vals])
            ax.plot(alpha_arr, losses, linestyle=ls,
                    color=FAMILY_COLORS.get(fam, "#333"),
                    label=label)
        ax.set_title(FAMILY_LABELS.get(fam, fam))
        ax.set_xlabel("α (ablation strength)")
        ax.set_ylabel("Masked CE loss")

    # Hide unused axes
    for ax in axes_flat[n_fam:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.suptitle(
        f"Dose-response ablation — {PE_LABELS.get(pe_type, pe_type)} "
        f"seed {seed}  layer {layer}  lag {lag}  (F6)",
        y=1.02
    )
    fig.tight_layout()
    return fig


def fig_source_vs_target(
    all_results: Dict[str, Any],
) -> plt.Figure:
    """
    F7: Source-side vs target-side ablation profile across PE types.

    ``all_results`` has structure:
        { (pe_type, seed, layer, lag): dose_results_dict }

    Plots mean-family loss vs α for source vs target, one panel per
    (pe_type, layer, lag) combination (averaged over seeds and families).
    """
    _apply_style()

    keys = list(all_results.keys())
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ablation results available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # One subplot per unique (pe_type, layer, lag)
    combos = sorted({(k[0], k[2], k[3]) for k in keys})
    n_cols = min(len(combos), 3)
    n_rows = math.ceil(len(combos) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (pe, layer, lag) in zip(axes_flat, combos):
        # Collect all seeds for this combo
        seed_keys = [k for k in keys if k[0] == pe and k[2] == layer and k[3] == lag]
        alpha_vals = sorted(all_results[seed_keys[0]]["source"].keys(), key=float)
        alpha_arr  = np.array([float(a) for a in alpha_vals])

        for side, ls, label in [("source", "-", "Source"), ("target", "--", "Target")]:
            # Average over seeds and families (skip iid_random)
            per_seed = []
            for sk in seed_keys:
                dr = all_results[sk]
                per_alpha = []
                for a in alpha_vals:
                    fam_losses = [v for k, v in dr[side][a].items() if k != "iid_random"]
                    per_alpha.append(np.mean(fam_losses))
                per_seed.append(per_alpha)
            arr  = np.array(per_seed)  # (n_seeds, n_alpha)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(alpha_arr, mean, linestyle=ls,
                    color=PE_COLORS.get(pe, "#333"), label=label)
            ax.fill_between(alpha_arr, mean - std, mean + std,
                            color=PE_COLORS.get(pe, "#333"), alpha=0.18)

        ax.set_title(f"{PE_LABELS.get(pe, pe)}  L{layer}  r={lag}")
        ax.set_xlabel("α")
        ax.set_ylabel("Mean task loss")

    for ax in axes_flat[len(combos):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(frameon=False)

    fig.suptitle("Source vs target-side ablation comparison (F7)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F6b — extended ablation: score-SVD vs probe-direction vs random
# ---------------------------------------------------------------------------

def fig_extended_ablation(
    b2_results: Dict[Tuple, Any],
    dose_results: Dict[Tuple, Any],
    families_to_show: Optional[List[str]] = None,
) -> plt.Figure:
    """
    F6b: Compare ablation impact of score-SVD directions, probe-derived
    directions, and random directions — one panel per (pe, layer).

    Plots mean task loss (excl. iid_random) vs α for each direction type.
    Headline comparison is score_svd vs probe_hidden vs random_control — all
    top-3 in the same PCA subspace.
    ``b2_results`` keys: (pe, seed, layer, lag) → {"score_svd": ..., "probe_abs": ..., "probe_hidden": ...}
    ``dose_results`` keys: same → {"random_control": {alpha: {fam: loss}}}
    """
    _apply_style()

    # Gather unique (pe, layer) combos
    combos = sorted({(k[0], k[2]) for k in b2_results})
    if not combos:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No B2 data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_cols = min(len(combos), 3)
    n_rows = math.ceil(len(combos) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    # Headline comparison: score_svd (top-3 right SVs of M_bar) vs probe_hidden
    # (top-3 SVs of a linear position probe's hidden-space weight matrix) vs
    # PCA-subspace random control.  All three are k=3, matched subspace, so
    # apples-to-apples.  probe_abs (top-1 probe direction + 2 random pads) is
    # still saved to b2_probe_direction_ablation.json for transparency but is
    # not the headline comparison.
    dir_styles = {
        "score_svd":      ("-",  "Score-SVD (top-3)"),
        "probe_hidden":   ("--", "Linear probe (top-3)"),
        "random_control": (":",  "Random (PCA-subspace)"),
    }

    for ax, (pe, layer) in zip(axes_flat, combos):
        # Collect keys for this (pe, layer)
        b2_keys = [k for k in b2_results if k[0] == pe and k[2] == layer]
        if not b2_keys:
            continue

        # Get alpha values from the first key's score_svd result
        sample_res = b2_results[b2_keys[0]].get("score_svd", {})
        if not sample_res:
            continue
        alpha_vals = sorted(sample_res.keys(), key=float)
        alpha_arr = np.array([float(a) for a in alpha_vals])

        for dir_type, (ls, label) in dir_styles.items():
            per_seed = []
            for bk in b2_keys:
                if dir_type == "random_control":
                    # Pull random from dose_results
                    dr = dose_results.get(bk, {}).get("random_control", {})
                    if not dr:
                        continue
                    per_alpha = []
                    for a in alpha_vals:
                        fam_losses = [v for k, v in dr[a].items() if k != "iid_random"]
                        per_alpha.append(np.mean(fam_losses) if fam_losses else np.nan)
                    per_seed.append(per_alpha)
                else:
                    res = b2_results[bk].get(dir_type)
                    if res is None:
                        continue
                    per_alpha = []
                    for a in alpha_vals:
                        fam_losses = [v for k, v in res[a].items() if k != "iid_random"]
                        per_alpha.append(np.mean(fam_losses) if fam_losses else np.nan)
                    per_seed.append(per_alpha)

            if not per_seed:
                continue
            arr = np.array(per_seed)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            ax.plot(alpha_arr, mean, linestyle=ls,
                    color=PE_COLORS.get(pe, "#333"), label=label)
            ax.fill_between(alpha_arr, mean - std, mean + std,
                            color=PE_COLORS.get(pe, "#333"), alpha=0.12)

        ax.set_title(f"{PE_LABELS.get(pe, pe)}  L{layer}")
        ax.set_xlabel("α (ablation strength)")
        ax.set_ylabel("Mean task loss")

    for ax in axes_flat[len(combos):]:
        ax.set_visible(False)

    # Shared legend from first populated axes
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(frameon=False, fontsize=8)
            break

    fig.suptitle("Score-SVD vs probe-direction vs random ablation (F6b)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B3 table writer
# ---------------------------------------------------------------------------

def _write_b3_table(b3_results: Dict, path: Path, alpha: float) -> None:
    """Write T6a: correct vs wrong-layer vs wrong-position ablation comparison."""
    lines = [
        f"T6a — Ablation control comparison  (α = {alpha})",
        "=" * 80,
        f"{'PE/seed/layer':<22s}  {'condition':<20s}  {'mean_task_loss':>14s}",
        "-" * 80,
    ]
    for pe, keys in b3_results.items():
        for key, conds in keys.items():
            for cond_name in ("correct", "wrong_layer", "wrong_position"):
                if cond_name not in conds:
                    continue
                fam_losses = conds[cond_name]
                mean_loss  = float(np.mean([v for k, v in fam_losses.items()
                                            if k != "iid_random"]))
                lines.append(f"{key:<22s}  {cond_name:<20s}  {mean_loss:>14.4f}")
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run singular-direction ablation study and generate F6/F7."
    )
    parser.add_argument("--summary-path", type=str, required=True)
    parser.add_argument("--models-dir",   type=str, required=True)
    parser.add_argument("--data-dir",     type=str, required=True)
    parser.add_argument("--out-dir",      type=str, required=True)
    parser.add_argument("--pe-types",     nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds",        nargs="+", type=int, default=[0])
    parser.add_argument("--layers",       nargs="+", type=int, default=[2],
                        help="1-indexed transformer layers to ablate")
    parser.add_argument("--lag",          type=int, default=1)
    parser.add_argument("--top-k",        type=int, default=3)
    parser.add_argument("--alpha-values", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0])
    # B2: probe-direction comparison
    parser.add_argument("--probe-baselines-path", type=str, default=None,
                        help="Path to probe_baselines_summary.json (enables B2 comparison)")
    # B3: wrong-layer / wrong-position controls
    parser.add_argument("--wrong-layer",          type=int, default=None,
                        help="Layer whose SVD dirs are used but applied at --layers (B3)")
    parser.add_argument("--wrong-position-offset", type=int, default=10,
                        help="Position offset for wrong-position control (B3, default 10)")
    parser.add_argument("--ablation-alpha-b3",    type=float, default=2.0,
                        help="Alpha value used in B3 controls table (default: 2.0)")
    parser.add_argument("--device",       type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    models_dir   = Path(args.models_dir)
    data_dir     = Path(args.data_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path) as f:
        summary = json.load(f)

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

    all_dose_results: Dict[Tuple, Any] = {}

    for pe_type in args.pe_types:
        for seed in args.seeds:
            seed_dir  = models_dir / f"{pe_type}_seed{seed}"
            model_pt  = seed_dir / "model.pt"
            if not model_pt.exists():
                print(f"  Skipping {pe_type} seed {seed}: model.pt not found.")
                continue

            # Load per-seed analysis JSON
            analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
            if not analysis_file.exists():
                print(f"  Skipping {pe_type} seed {seed}: analysis JSON not found.")
                continue
            with open(analysis_file) as f:
                analysis = json.load(f)

            model = create_transformer_variant(config, pe_type)
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(args.device)

            for layer in args.layers:
                # layer is 1-indexed; model blocks are 0-indexed
                layer_module = model.transformer.h[layer - 1]
                layer_data   = analysis["layer_stats"][layer - 1]  # 0-indexed

                m_bar_r       = np.array(layer_data["M_bar_r"])
                pca_components = np.array(layer_data["pca_components"])
                mu_l           = np.array(layer_data["mu_l"])

                print(f"  Ablating {pe_type} seed={seed} layer={layer} lag={args.lag} …")
                # Deterministic cell-unique seed: pe_idx * 10000 + seed * 100 + layer.
                # Independent across (pe, seed, layer) while reproducible across runs.
                _pe_idx   = {"rope": 0, "alibi": 1, "absolute": 2}.get(pe_type, 0)
                _rng_seed = _pe_idx * 10000 + int(seed) * 100 + int(layer)
                dose = ablation_sweep(
                    model           = model,
                    loaders         = loaders,
                    target_layer    = layer_module,
                    m_bar_r         = m_bar_r,
                    pca_components  = pca_components,
                    mu_l            = mu_l,
                    lag             = args.lag,
                    top_k           = args.top_k,
                    alpha_values    = args.alpha_values,
                    device          = args.device,
                    rng_seed        = _rng_seed,
                )
                all_dose_results[(pe_type, seed, layer, args.lag)] = dose

                # F6 per (pe_type, seed, layer, lag)
                fig6 = fig_dose_response(dose, pe_type, seed, layer, args.lag)
                tag  = f"{pe_type}_s{seed}_l{layer}_r{args.lag}"
                fig6.savefig(out_dir / f"F6_dose_response_{tag}.pdf")
                plt.close(fig6)

    # -----------------------------------------------------------------------
    # B2: probe-direction vs score-direction comparison
    # -----------------------------------------------------------------------
    probe_summary = None
    if args.probe_baselines_path and Path(args.probe_baselines_path).exists():
        with open(args.probe_baselines_path) as f:
            probe_summary = json.load(f)
        print("\n  [B2] Running probe-direction ablation comparison …")

    b2_results: Dict[Tuple, Any] = {}

    if probe_summary is not None:
        for pe_type in args.pe_types:
            if pe_type not in probe_summary:
                continue
            probe_pe = probe_summary[pe_type]

            for seed in args.seeds:
                seed_dir  = models_dir / f"{pe_type}_seed{seed}"
                model_pt  = seed_dir / "model.pt"
                analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
                if not model_pt.exists() or not analysis_file.exists():
                    continue

                with open(analysis_file) as f:
                    analysis = json.load(f)

                model = create_transformer_variant(config, pe_type)
                state = torch.load(model_pt, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.to(args.device)

                for layer in args.layers:
                    layer_module = model.transformer.h[layer - 1]
                    layer_data   = analysis["layer_stats"][layer - 1]
                    pca_components = np.array(layer_data["pca_components"])
                    mu_l           = np.array(layer_data["mu_l"])
                    m_bar_r        = np.array(layer_data["M_bar_r"])

                    # Score-SVD source directions (top-k right singular vectors)
                    U, S, Vh = np.linalg.svd(m_bar_r[args.lag])
                    svd_dirs_m = Vh[:args.top_k, :]  # (k, m)
                    svd_dirs_h = torch.tensor(
                        svd_dirs_m @ pca_components, dtype=torch.float32)
                    Q, _ = torch.linalg.qr(svd_dirs_h.T)
                    svd_dirs_h = Q.T

                    # Probe direction (linear_abs, seed-matched if available)
                    probe_dirs_h = None
                    if "linear_abs" in probe_pe and "directions_by_seed" in probe_pe["linear_abs"]:
                        seed_dirs = probe_pe["linear_abs"]["directions_by_seed"]
                        seed_idx  = min(seed, len(seed_dirs) - 1)
                        layer_idx = layer - 1
                        if seed_idx < len(seed_dirs) and layer_idx < len(seed_dirs[seed_idx]):
                            dir_m = np.array(seed_dirs[seed_idx][layer_idx])  # (m,)
                            dir_h = dir_m @ pca_components                    # (H,)
                            dir_h = dir_h / (np.linalg.norm(dir_h) + 1e-12)
                            # Pad to top_k by adding orthogonal random directions
                            extra = np.random.randn(args.top_k - 1, dir_h.shape[0]).astype(np.float32)
                            probe_dirs_h = torch.tensor(
                                np.vstack([dir_h[None], extra]), dtype=torch.float32)
                            Q2, _ = torch.linalg.qr(probe_dirs_h.T)
                            probe_dirs_h = Q2.T

                    # Hidden-space probe directions (no PCA bottleneck)
                    probe_hidden_dirs_h = None
                    if "linear_hidden" in probe_pe and "directions_by_seed" in probe_pe["linear_hidden"]:
                        seed_dirs_h = probe_pe["linear_hidden"]["directions_by_seed"]
                        seed_idx = min(seed, len(seed_dirs_h) - 1)
                        layer_idx = layer - 1
                        if seed_idx < len(seed_dirs_h) and layer_idx < len(seed_dirs_h[seed_idx]):
                            dirs_raw = np.array(seed_dirs_h[seed_idx][layer_idx])  # (top_k, hidden_size)
                            if dirs_raw.ndim == 1:
                                dirs_raw = dirs_raw[None, :]
                            probe_hidden_dirs_h = torch.tensor(dirs_raw[:args.top_k], dtype=torch.float32)
                            Q3, _ = torch.linalg.qr(probe_hidden_dirs_h.T)
                            probe_hidden_dirs_h = Q3.T[:min(args.top_k, Q3.shape[1])]

                    key_b2 = (pe_type, seed, layer, args.lag)
                    b2_results[key_b2] = {}

                    for label, dirs_h in [("score_svd", svd_dirs_h),
                                          ("probe_abs", probe_dirs_h),
                                          ("probe_hidden", probe_hidden_dirs_h)]:
                        if dirs_h is None:
                            continue
                        res = _eval_with_dirs(
                            model, loaders, layer_module, dirs_h, mu_l,
                            alpha_values=args.alpha_values,
                            device=args.device,
                            side="source", lag=args.lag,
                        )
                        b2_results[key_b2][label] = res

                    print(f"    B2: {pe_type} seed={seed} layer={layer} "
                          f"keys={list(b2_results[key_b2].keys())}")

    # -----------------------------------------------------------------------
    # B3: wrong-layer and wrong-position controls
    # -----------------------------------------------------------------------
    b3_results: Dict[str, Any] = {}
    alpha_b3 = args.ablation_alpha_b3

    if args.wrong_layer is not None or True:  # always run wrong-position
        print(f"\n  [B3] Wrong-layer / wrong-position controls (α={alpha_b3}) …")

        for pe_type in args.pe_types:
            b3_results[pe_type] = {}
            for seed in args.seeds:
                seed_dir  = models_dir / f"{pe_type}_seed{seed}"
                model_pt  = seed_dir / "model.pt"
                analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"
                if not model_pt.exists() or not analysis_file.exists():
                    continue

                with open(analysis_file) as f:
                    analysis = json.load(f)

                model = create_transformer_variant(config, pe_type)
                state = torch.load(model_pt, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.to(args.device)

                for layer in args.layers:
                    layer_module = model.transformer.h[layer - 1]
                    layer_data   = analysis["layer_stats"][layer - 1]
                    pca_components = np.array(layer_data["pca_components"])
                    mu_l           = np.array(layer_data["mu_l"])
                    m_bar_r        = np.array(layer_data["M_bar_r"])

                    U, S, Vh = np.linalg.svd(m_bar_r[args.lag])
                    dirs_m   = Vh[:args.top_k, :]
                    dirs_h   = torch.tensor(dirs_m @ pca_components, dtype=torch.float32)
                    Q, _     = torch.linalg.qr(dirs_h.T);  dirs_h = Q.T

                    key = f"{pe_type}_seed{seed}_l{layer}"
                    b3_results[pe_type][key] = {}

                    # Correct-layer correct-position (reference)
                    ref = _eval_with_dirs(
                        model, loaders, layer_module, dirs_h, mu_l,
                        alpha_values=[alpha_b3], device=args.device,
                        side="source", lag=args.lag,
                    )
                    b3_results[pe_type][key]["correct"] = {
                        fam: ref[alpha_b3][fam] for fam in ref[alpha_b3]
                    }

                    # Wrong-layer: use SVD dirs from wrong_layer, apply at current layer
                    wl = args.wrong_layer
                    if wl is None:
                        # Use the other extreme layer
                        wl = 4 if layer <= 2 else 1
                    if 1 <= wl <= len(analysis["layer_stats"]) and wl != layer:
                        wl_data = analysis["layer_stats"][wl - 1]
                        wl_pca  = np.array(wl_data["pca_components"])
                        wl_mu   = np.array(wl_data["mu_l"])
                        wl_mbar = np.array(wl_data["M_bar_r"])
                        _, _, Vh_wl = np.linalg.svd(wl_mbar[args.lag])
                        dirs_wl = torch.tensor(
                            Vh_wl[:args.top_k] @ wl_pca, dtype=torch.float32)
                        Q2, _ = torch.linalg.qr(dirs_wl.T); dirs_wl = Q2.T
                        res_wl = _eval_with_dirs(
                            model, loaders, layer_module, dirs_wl, wl_mu,
                            alpha_values=[alpha_b3], device=args.device,
                            side="source", lag=args.lag,
                        )
                        b3_results[pe_type][key]["wrong_layer"] = {
                            fam: res_wl[alpha_b3][fam] for fam in res_wl[alpha_b3]
                        }
                        b3_results[pe_type][key]["wrong_layer_idx"] = wl

                    # Wrong-position: offset hook positions
                    offset = args.wrong_position_offset
                    res_wp = _eval_with_dirs(
                        model, loaders, layer_module, dirs_h, mu_l,
                        alpha_values=[alpha_b3], device=args.device,
                        side="source", lag=args.lag,
                        position_offset=offset,
                    )
                    b3_results[pe_type][key]["wrong_position"] = {
                        fam: res_wp[alpha_b3][fam] for fam in res_wp[alpha_b3]
                    }
                    b3_results[pe_type][key]["wrong_position_offset"] = offset

                    print(f"    B3: {pe_type} seed={seed} layer={layer} done")

    # Save all raw results
    def _serial(obj):
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)):     return int(obj)
        raise TypeError(type(obj))

    serialisable = {str(k): v for k, v in all_dose_results.items()}
    with open(out_dir / "dose_response.json", "w") as f:
        json.dump(serialisable, f, default=_serial, indent=2)

    if b2_results:
        b2_ser = {str(k): v for k, v in b2_results.items()}
        with open(out_dir / "b2_probe_direction_ablation.json", "w") as f:
            json.dump(b2_ser, f, default=_serial, indent=2)
        print(f"  Saved B2 → {out_dir / 'b2_probe_direction_ablation.json'}")

        # F6b — extended ablation comparison figure
        try:
            fig6b = fig_extended_ablation(b2_results, all_dose_results)
            fig6b.savefig(out_dir / "F6b_extended_ablation.pdf")
            plt.close(fig6b)
            print(f"  Saved F6b → {out_dir / 'F6b_extended_ablation.pdf'}")
        except Exception as e:
            print(f"  [F6b error]: {e}")

    if b3_results:
        with open(out_dir / "b3_controls.json", "w") as f:
            json.dump(b3_results, f, default=_serial, indent=2)
        print(f"  Saved B3 → {out_dir / 'b3_controls.json'}")

        # Write human-readable T6a table
        _write_b3_table(b3_results, out_dir / "T6a_ablation_controls.txt", alpha_b3)

    # F7 — source vs target comparison (if we have results)
    if all_dose_results:
        fig7 = fig_source_vs_target(all_dose_results)
        fig7.savefig(out_dir / "F7_source_vs_target.pdf")
        plt.close(fig7)
        print(f"  Saved F7 → {out_dir / 'F7_source_vs_target.pdf'}")

    print(f"\nAblation study complete.  Results in {out_dir}/")


if __name__ == "__main__":
    main()
