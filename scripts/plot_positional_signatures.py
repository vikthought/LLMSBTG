"""
Generate publication-quality figures from an analysis_summary.json produced
by scripts/aggregate_positional_results.py.

Figures produced (matching paper.tex Section 10):
  F1  training_parity.pdf        — per-epoch per-family val loss across PE types
  F2  lag_amplitude.pdf          — A_r^(ℓ) vs lag r for selected layers, all PE families
  F3  layerwise_rdi.pdf          — RDI^(ℓ) vs layer with ±1 SD band across seeds
  F4  heterogeneity_heatmap.pdf  — ||Δ_r^(ℓ)(i)||_F heatmap (lag × window-position)
  F5  spectral_concentration.pdf — SCR_r^(ℓ)(k) cumulative sv fraction
  F8  added_value_overview.pdf   — probe accuracy, attention entropy, RDI overlaid

Usage
-----
python scripts/plot_positional_signatures.py \\
    --summary-path results/<run>/analysis_summary.json \\
    --out-dir      results/<run>/figures

Optional for F1:
    --models-dir   results/<run>/models     # directory with pe_type_seedN/ subdirs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

import matplotlib
matplotlib.use("Agg")          # non-interactive; safe for cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# PE-type colour / label conventions
# ---------------------------------------------------------------------------
PE_COLORS: Dict[str, str] = {
    "rope":     "#2CA02C",
    "alibi":    "#FF7F0E",
    "absolute": "#1F77B4",
}
PE_LABELS: Dict[str, str] = {
    "rope":     "RoPE",
    "alibi":    "ALiBi",
    "absolute": "Absolute",
}
FAMILY_COLORS: Dict[str, str] = {
    "variable_lag_copy": "#1F77B4",
    "absolute_anchor":   "#FF7F0E",
    "order_sensitive":   "#2CA02C",
    "distance_bucket":   "#9467BD",
    "iid_random":        "#AAAAAA",
}
FAMILY_LABELS: Dict[str, str] = {
    "variable_lag_copy": "Var-lag copy",
    "absolute_anchor":   "Abs. anchor",
    "order_sensitive":   "Order-sensitive",
    "distance_bucket":   "Dist. bucket",
    "iid_random":        "IID random",
}

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def _pe_color(pe: str) -> str:
    return PE_COLORS.get(pe, "#333333")


def _pe_label(pe: str) -> str:
    return PE_LABELS.get(pe, pe)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(summary_path: Path) -> Dict[str, Any]:
    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# F1 — Training parity curves
# ---------------------------------------------------------------------------

def fig_training_parity(models_dir: Path, pe_types: Optional[List[str]] = None) -> plt.Figure:
    """
    Loads training_history.json from each models_dir/<pe_type>_seed<N>/ directory
    and plots per-family validation loss vs epoch for each PE type.

    One column per PE type; one coloured line per task family.  Thin lines =
    individual seeds; thick line = mean across seeds.
    """
    _apply_style()

    if pe_types is None:
        # Discover available PE types from directory listing
        pe_types = sorted({
            d.name.rsplit("_seed", 1)[0]
            for d in models_dir.iterdir()
            if d.is_dir() and "_seed" in d.name
        })

    if not pe_types:
        raise FileNotFoundError(f"No seed directories found in {models_dir}")

    # Discover all families from first available history
    families: List[str] = []
    for pe in pe_types:
        for d in sorted(models_dir.glob(f"{pe}_seed*")):
            hf = d / "training_history.json"
            if hf.exists():
                with open(hf) as f:
                    h = json.load(f)
                families = sorted(h["epochs"][0]["val_losses"].keys())
                break
        if families:
            break

    n_cols = len(pe_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, pe in zip(axes, pe_types):
        histories_by_family: Dict[str, List[List[float]]] = {f: [] for f in families}

        for seed_dir in sorted(models_dir.glob(f"{pe}_seed*")):
            hf = seed_dir / "training_history.json"
            if not hf.exists():
                continue
            with open(hf) as f:
                h = json.load(f)
            for fam in families:
                losses = [ep["val_losses"].get(fam, float("nan")) for ep in h["epochs"]]
                histories_by_family[fam].append(losses)
                # Thin seed line
                ax.plot(range(1, len(losses) + 1), losses,
                        color=FAMILY_COLORS.get(fam, "#888"),
                        alpha=0.3, linewidth=0.8)

        # Mean line
        for fam, all_losses in histories_by_family.items():
            if not all_losses:
                continue
            max_len = max(len(l) for l in all_losses)
            padded  = [l + [float("nan")] * (max_len - len(l)) for l in all_losses]
            mean    = np.nanmean(padded, axis=0)
            ax.plot(range(1, len(mean) + 1), mean,
                    color=FAMILY_COLORS.get(fam, "#888"),
                    label=FAMILY_LABELS.get(fam, fam), linewidth=2.0)

        ax.set_title(f"{_pe_label(pe)}", fontweight="bold")
        ax.set_xlabel("Epoch")
        if ax is axes[0]:
            ax.set_ylabel("Val loss (masked CE)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Shared legend below
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(families), 4), bbox_to_anchor=(0.5, -0.12),
                   frameon=False)
    fig.suptitle("Training parity: per-family validation loss (F1)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F2 — Lag amplitude curves A_r^(ℓ) vs r
# ---------------------------------------------------------------------------

def fig_lag_amplitude(
    summary: Dict[str, Any],
    layers_to_show: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Lag-amplitude curves A_r^(ℓ) vs lag r.

    One subplot per (selected) layer; one coloured line per PE family with ±1 SD band.
    """
    _apply_style()

    pe_types = list(summary.keys())
    if not pe_types:
        raise ValueError("summary is empty")

    n_layers = len(summary[pe_types[0]]["A_r_layerwise"]["mean"])
    if layers_to_show is None:
        step = max(1, n_layers // 4)
        layers_to_show = list(range(0, n_layers, step))

    n_cols = len(layers_to_show)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, l_idx in zip(axes, layers_to_show):
        max_lag = None
        for pe in pe_types:
            A_mean = np.array(summary[pe]["A_r_layerwise"]["mean"][l_idx])
            A_std  = np.array(summary[pe]["A_r_layerwise"]["std"][l_idx])
            if max_lag is None:
                max_lag = len(A_mean)
            r_vals = np.arange(len(A_mean))
            ax.plot(r_vals, A_mean, color=_pe_color(pe), label=_pe_label(pe))
            ax.fill_between(r_vals, A_mean - A_std, A_mean + A_std,
                            color=_pe_color(pe), alpha=0.18)

        ax.set_title(f"Layer {l_idx + 1}")
        ax.set_xlabel("Lag r")
        if ax is axes[0]:
            ax.set_ylabel(r"$A_r^{(\ell)}$ (Frobenius norm)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Lag amplitude curves (F2)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F3 — Layerwise RDI^(ℓ) comparison
# ---------------------------------------------------------------------------

def fig_layerwise_rdi(summary: Dict[str, Any]) -> plt.Figure:
    """
    Relative-position dominance index RDI^(ℓ) across layers.

    One line per PE family with ±1 SD shaded band.
    Large RDI → lag-structured (relative-position dominant).
    Small RDI → absolute-position dependent.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    for pe in summary:
        rdi_mean = np.array(summary[pe]["RDI_layerwise"]["mean"])
        rdi_std  = np.array(summary[pe]["RDI_layerwise"]["std"])
        layers   = np.arange(1, len(rdi_mean) + 1)
        ax.plot(layers, rdi_mean, color=_pe_color(pe), label=_pe_label(pe))
        ax.fill_between(layers, rdi_mean - rdi_std, rdi_mean + rdi_std,
                        color=_pe_color(pe), alpha=0.18)

    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.9,
               label="RDI = 0.5 (equal share)")
    ax.set_xlabel("Transformer layer ℓ")
    ax.set_ylabel(r"$\mathrm{RDI}^{(\ell)}$")
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.set_title("Layerwise relative-position dominance (F3)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4 — Heterogeneity heatmap ||Δ_r^(ℓ)(i)||_F
# ---------------------------------------------------------------------------

def fig_heterogeneity_heatmap(
    summary: Dict[str, Any],
    layer_idx: int = 1,
) -> plt.Figure:
    """
    Heatmap of ||Δ_r^(ℓ)(i)||_F over (lag r, window position i).

    Displays the mean over seeds.  One column per PE family.
    ``layer_idx`` is 0-based into the stored layer list (layer 1 = first block).
    """
    _apply_style()

    pe_types = list(summary.keys())
    n_pe = len(pe_types)
    fig, axes = plt.subplots(1, n_pe, figsize=(4.5 * n_pe, 3.8))
    if n_pe == 1:
        axes = [axes]

    vmax_global = 0.0
    mats = {}
    for pe in pe_types:
        df_mean = np.array(summary[pe]["delta_frob_layerwise"]["mean"][layer_idx])
        mats[pe] = df_mean   # shape (max_lag+1, num_windows)
        vmax_global = max(vmax_global, float(df_mean.max()))

    for ax, pe in zip(axes, pe_types):
        mat = mats[pe]
        n_lags, n_wins = mat.shape
        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       origin="lower", vmin=0, vmax=vmax_global,
                       cmap="viridis",
                       extent=[0, n_wins, 0, n_lags])
        ax.set_title(_pe_label(pe), fontweight="bold")
        ax.set_xlabel("Window position i")
        if ax is axes[0]:
            ax.set_ylabel("Lag r")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                 label=r"$\|\Delta_r^{(\ell)}(i)\|_F$")
    fig.suptitle(
        f"Position-heterogeneity heatmap  (layer {layer_idx + 1})  (F4)", y=1.02
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F5 — Spectral concentration ratio SCR_r^(ℓ)(k)
# ---------------------------------------------------------------------------

def fig_spectral_concentration(
    summary: Dict[str, Any],
    layer_idx: int = 1,
    lags_to_show: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Spectral concentration ratio SCR_r^(ℓ)(k) = sum(σ[:k]) / sum(σ).

    One subplot per lag r (or a selection); one line per PE family.
    High concentration in few dimensions → low-rank positional subspace.
    """
    _apply_style()

    pe_types = list(summary.keys())
    max_lag_plus1 = len(summary[pe_types[0]]["SCR_r_layerwise"]["mean"][layer_idx])
    if lags_to_show is None:
        lags_to_show = list(range(1, min(max_lag_plus1, 4)))  # lags 1, 2, 3

    n_cols = len(lags_to_show)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 3.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, r in zip(axes, lags_to_show):
        for pe in pe_types:
            scr_mean = np.array(summary[pe]["SCR_r_layerwise"]["mean"][layer_idx][r])
            scr_std  = np.array(summary[pe]["SCR_r_layerwise"]["std"][layer_idx][r])
            k_vals   = np.arange(1, len(scr_mean) + 1)
            ax.plot(k_vals, scr_mean, color=_pe_color(pe), label=_pe_label(pe))
            ax.fill_between(k_vals, scr_mean - scr_std, scr_mean + scr_std,
                            color=_pe_color(pe), alpha=0.18)
        ax.set_title(f"Lag r = {r}")
        ax.set_xlabel("Top-k singular directions")
        if ax is axes[0]:
            ax.set_ylabel(r"$\mathrm{SCR}_r^{(\ell)}(k)$")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.15),
                   frameon=False)
    fig.suptitle(f"Spectral concentration ratio  (layer {layer_idx + 1})  (F5)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F9 — CKA heatmaps  (A4)
# ---------------------------------------------------------------------------

def fig_cka_heatmaps(cka_data: Dict[str, Any]) -> plt.Figure:
    """
    F9: Cross-layer CKA heatmaps for each PE family + cross-architecture panel.

    cka_data is loaded from cka_summary.json produced by run_cka_baseline.py.

    Layout: one column per PE family (within-architecture cross-layer heatmap)
            plus a final column showing cross-architecture (rope vs alibi,
            rope vs absolute, alibi vs absolute) at each layer.
    """
    _apply_style()

    pe_types = list(cka_data.keys())
    n_pe     = len(pe_types)

    # Discover number of layers from key names like 'l0_l1'
    def _max_layer(cross_layer: dict) -> int:
        mx = 0
        for k in cross_layer:
            a, b = k.split("_")
            mx = max(mx, int(a[1:]), int(b[1:]))
        return mx

    n_layers = max(_max_layer(cka_data[pe]["cross_layer"]) for pe in pe_types) + 1

    # Build within-architecture matrices  (n_layers × n_layers symmetric)
    within_mats = {}
    for pe in pe_types:
        mat = np.eye(n_layers)
        for key, val in cka_data[pe]["cross_layer"].items():
            a, b = key.split("_")
            i, j = int(a[1:]), int(b[1:])
            v = val["mean"] if isinstance(val, dict) else float(val)
            mat[i, j] = v
            mat[j, i] = v
        within_mats[pe] = mat

    # Build cross-architecture per-layer vectors
    # Each cross_pe entry looks like {"rope_vs_alibi": {"l0": {...}, ...}, ...}
    # Collect all unique pairs
    cross_pairs: Dict[str, np.ndarray] = {}
    for pe in pe_types:
        for pair_key, layer_dict in cka_data[pe].get("cross_pe", {}).items():
            if pair_key in cross_pairs:
                continue
            arr = np.zeros(n_layers)
            for lk, val in layer_dict.items():
                l = int(lk[1:])
                arr[l] = val["mean"] if isinstance(val, dict) else float(val)
            cross_pairs[pair_key] = arr

    n_cols = n_pe + (1 if cross_pairs else 0)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(3.8 * n_cols, 3.8),
                             squeeze=False)
    axes = axes[0]  # shape (n_cols,)

    tick_labels = [f"L{i}" for i in range(n_layers)]

    for ax, pe in zip(axes[:n_pe], pe_types):
        mat = within_mats[pe]
        im  = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis",
                        aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n_layers)); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(range(n_layers)); ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(f"{_pe_label(pe)}\n(cross-layer CKA)", fontweight="bold")
        # Annotate cells
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if cross_pairs:
        ax = axes[n_pe]
        layers_x = np.arange(n_layers)
        for pair_key, arr in cross_pairs.items():
            # Build a friendly label: "rope vs alibi" → two PE colours blended
            parts = pair_key.replace("_vs_", " vs ").split(" vs ")
            label = " / ".join(PE_LABELS.get(p, p) for p in parts)
            ax.plot(layers_x, arr, marker="o", label=label)
        ax.set_xticks(layers_x)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Layer")
        ax.set_ylabel("CKA")
        ax.set_title("Cross-architecture CKA\n(at each layer)", fontweight="bold")
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle("CKA analysis (F9)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F8 — Added-value overview
# ---------------------------------------------------------------------------

def fig_added_value_overview(
    summary: Dict[str, Any],
    probe_baselines: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Multi-panel overlay showing, per transformer layer:
      (a) Linear abs-position probe accuracy (from analysis_summary)
      (b) Rel-distance + near/far probe accuracy (from probe_baselines_summary)
      (c) Mean attention entropy (from attn_stats)
      (d) RDI — the score-geometric summary

    The H7 argument: panels (a)-(c) may agree across PE families at some
    layers while (d) diverges — that is where score geometry adds insight.
    """
    _apply_style()

    has_probes = probe_baselines is not None
    n_panels = 4 if has_probes else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)

    # Build panel list dynamically
    panels: List = [
        ("probe_accuracy_layerwise", "Linear abs. probe acc."),
    ]
    if has_probes:
        panels.append(("_probe_baselines", "Distance probe acc."))
    panels.append((None, "Mean attention entropy"))
    panels.append(("RDI_layerwise", r"$\mathrm{RDI}^{(\ell)}$ (score geom.)"))

    PROBE_STYLES = {
        "rel_dist":  ("--", "Rel-dist"),
        "near_far":  (":",  "Near/far"),
    }

    for ax, (key, ylabel) in zip(axes, panels):
        for pe in summary:
            entry = summary[pe]
            n_layers = len(entry["RDI_layerwise"]["mean"])
            layers   = np.arange(1, n_layers + 1)

            if key == "_probe_baselines":
                # Panel (b): rel_dist and near_far from probe_baselines_summary
                pe_probes = probe_baselines.get(pe, {})
                for ptype, (ls, lbl) in PROBE_STYLES.items():
                    if ptype not in pe_probes:
                        continue
                    vals = np.array(pe_probes[ptype]["mean"])
                    ax.plot(layers, vals, color=_pe_color(pe), linestyle=ls,
                            label=f"{_pe_label(pe)} {lbl}")
            elif key is not None:
                vals_mean = np.array(entry[key]["mean"])
                vals_std  = np.array(entry[key]["std"])
                ax.plot(layers, vals_mean, color=_pe_color(pe), label=_pe_label(pe))
                ax.fill_between(layers, vals_mean - vals_std, vals_mean + vals_std,
                                color=_pe_color(pe), alpha=0.18)
            else:
                # Attention entropy: mean over heads, then mean & std over seeds
                attn_seeds = entry.get("attn_stats_seeds", [])
                if not attn_seeds:
                    continue
                entropies = []
                for a in attn_seeds:
                    ent_lh = np.array(a["mean_entropy"])   # (n_layers, n_heads)
                    entropies.append(ent_lh.mean(axis=-1)) # (n_layers,)
                entropies = np.array(entropies)            # (n_seeds, n_layers)
                vals_mean = entropies.mean(axis=0)
                vals_std  = entropies.std(axis=0)
                ax.plot(layers, vals_mean, color=_pe_color(pe), label=_pe_label(pe))
                ax.fill_between(layers, vals_mean - vals_std, vals_mean + vals_std,
                                color=_pe_color(pe), alpha=0.18)

        ax.set_xlabel("Transformer layer ℓ")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # y-limits for probe and RDI panels
    axes[0].set_ylim(0, 1.05)
    if has_probes:
        axes[1].set_ylim(0, 1.05)
    axes[-1].set_ylim(-0.05, 1.05)

    # Legends
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Added-value overview (F8)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F10 — Recency slope (beta) across layers
# ---------------------------------------------------------------------------

def fig_recency_slope(summary: Dict[str, Any]) -> plt.Figure:
    """
    Layerwise recency slope beta_ell for each PE family.
    More negative beta = stronger recency decay in lag amplitudes.
    Tests hypothesis H2: ALiBi should have the most negative beta.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    for pe in summary:
        beta_mean = np.array(summary[pe]["beta_layerwise"]["mean"])
        beta_std  = np.array(summary[pe]["beta_layerwise"]["std"])
        layers    = np.arange(1, len(beta_mean) + 1)
        ax.plot(layers, beta_mean, color=_pe_color(pe), marker="o",
                markersize=5, label=_pe_label(pe))
        ax.fill_between(layers, beta_mean - beta_std, beta_mean + beta_std,
                        color=_pe_color(pe), alpha=0.18)

    ax.axhline(0, color="#888", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Transformer layer ℓ")
    ax.set_ylabel(r"Recency slope $\beta_\ell$")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.set_title(r"Layerwise recency slope $\beta_\ell$ (F10)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F11 — Attention statistics (entropy, mean distance, recency fraction)
# ---------------------------------------------------------------------------

def fig_attention_stats(summary: Dict[str, Any]) -> plt.Figure:
    """
    3-panel figure showing per-layer attention-pattern statistics:
      (a) Mean attention entropy (nats)
      (b) Mean attended distance
      (c) Recency-5 fraction (mass on |i-j| <= 5)
    Averaged over heads; shaded bands = ±1 SD across seeds.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    stat_keys = [
        ("mean_entropy",           "Mean attention entropy (nats)"),
        ("mean_attended_distance", "Mean attended distance"),
        ("recency_k5_fraction",    "Recency-5 fraction"),
    ]

    for ax, (stat_key, ylabel) in zip(axes, stat_keys):
        for pe in summary:
            attn_seeds = summary[pe].get("attn_stats_seeds", [])
            if not attn_seeds:
                continue
            per_seed = []
            for a in attn_seeds:
                arr = np.array(a[stat_key])      # (n_layers, n_heads)
                per_seed.append(arr.mean(axis=-1))  # (n_layers,)
            per_seed = np.array(per_seed)          # (n_seeds, n_layers)
            vals_mean = per_seed.mean(axis=0)
            vals_std  = per_seed.std(axis=0)
            layers = np.arange(1, len(vals_mean) + 1)

            ax.plot(layers, vals_mean, color=_pe_color(pe), marker="o",
                    markersize=4, label=_pe_label(pe))
            ax.fill_between(layers, vals_mean - vals_std, vals_mean + vals_std,
                            color=_pe_color(pe), alpha=0.18)

        ax.set_xlabel("Transformer layer ℓ")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].legend(frameon=False)
    fig.suptitle("Attention-pattern statistics (F11)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate positional-signature figures from analysis_summary.json"
    )
    parser.add_argument("--summary-path", type=str, required=True,
                        help="Path to analysis_summary.json")
    parser.add_argument("--out-dir",      type=str, required=True,
                        help="Directory to write PDF figures into")
    parser.add_argument("--models-dir",   type=str, default=None,
                        help="Optional: models directory for F1 training-parity plot")
    parser.add_argument("--cka-path",     type=str, default=None,
                        help="Optional: path to cka_summary.json for F9 CKA heatmaps")
    parser.add_argument("--probe-baselines-path", type=str, default=None,
                        help="Optional: path to probe_baselines_summary.json for F8 expanded panels")
    parser.add_argument("--f4-layer",     type=int, default=1,
                        help="0-based layer index for F4/F5 heatmaps (default: 1)")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)

    probe_baselines = None
    if args.probe_baselines_path:
        pb_path = Path(args.probe_baselines_path)
        if pb_path.exists():
            with open(pb_path) as f:
                probe_baselines = json.load(f)
            print(f"  Loaded probe baselines from {pb_path}")
        else:
            print(f"  WARNING: probe baselines not found at {pb_path}")

    saved: List[str] = []

    # F1 — training parity (needs models dir)
    if args.models_dir:
        models_dir = Path(args.models_dir)
        try:
            f = fig_training_parity(models_dir)
            p = out_dir / "F1_training_parity.pdf"
            f.savefig(p)
            plt.close(f)
            saved.append(str(p))
        except Exception as e:
            print(f"  [F1 skipped]: {e}")

    # F2
    try:
        f = fig_lag_amplitude(summary)
        p = out_dir / "F2_lag_amplitude.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F2 error]: {e}")

    # F3
    try:
        f = fig_layerwise_rdi(summary)
        p = out_dir / "F3_layerwise_rdi.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F3 error]: {e}")

    # F4
    try:
        f = fig_heterogeneity_heatmap(summary, layer_idx=args.f4_layer)
        p = out_dir / "F4_heterogeneity_heatmap.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F4 error]: {e}")

    # F5
    try:
        f = fig_spectral_concentration(summary, layer_idx=args.f4_layer)
        p = out_dir / "F5_spectral_concentration.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F5 error]: {e}")

    # F8
    try:
        f = fig_added_value_overview(summary, probe_baselines=probe_baselines)
        p = out_dir / "F8_added_value_overview.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F8 error]: {e}")

    # F9 — CKA heatmaps (needs cka_summary.json)
    if args.cka_path:
        try:
            with open(args.cka_path) as f:
                cka_data = json.load(f)
            fig9 = fig_cka_heatmaps(cka_data)
            p = out_dir / "F9_cka_heatmaps.pdf"
            fig9.savefig(p); plt.close(fig9); saved.append(str(p))
        except Exception as e:
            print(f"  [F9 error]: {e}")

    # F10 — Recency slope (beta) across layers
    try:
        f = fig_recency_slope(summary)
        p = out_dir / "F10_recency_slope.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F10 error]: {e}")

    # F11 — Attention statistics (entropy, mean distance, recency fraction)
    try:
        f = fig_attention_stats(summary)
        p = out_dir / "F11_attention_stats.pdf"
        f.savefig(p); plt.close(f); saved.append(str(p))
    except Exception as e:
        print(f"  [F11 error]: {e}")

    print(f"\nSaved {len(saved)} figure(s) to {out_dir}:")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    main()
