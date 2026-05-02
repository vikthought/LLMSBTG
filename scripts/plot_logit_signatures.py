"""
Generate figures for logit-space score-geometric analysis.

Figures:
  FL1  logit_mbar_heatmaps.pdf    — M_bar_r heatmaps for each PE type (logit space)
  FL2  logit_vs_hidden_bars.pdf   — RDI / H_r / SCR bar chart: hidden L4 vs logit
  FL3  logit_probe_comparison.pdf — Position probe accuracy: logit vs hidden layers
  FL4  logit_dsm_loss.pdf         — DSM validation loss comparison

Usage
-----
python scripts/plot_logit_signatures.py \
    --logit-dir    results/<run>/logit_analysis/ \
    --hidden-summary results/<run>/analysis_summary.json \
    --out-dir      results/<run>/logit_figures/

Optional for pretrained:
    --pretrained-dir  results/<run>/pretrained_logit_analysis/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PE-type colour / label conventions (shared with plot_positional_signatures.py)
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


def _unwrap(val):
    """Extract the mean list from an analysis_summary value.

    analysis_summary.json stores layerwise metrics as either:
      - a plain list  (old format / single-seed)
      - a dict with {"mean": [...], "std": [...]}  (aggregated format)
    This helper normalises both to a plain list.
    """
    if isinstance(val, dict):
        return val.get("mean", [])
    return val


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
    })


def load_logit_results(logit_dir: Path, pe_types, seeds):
    """Load per-(pe, seed) logit analysis JSONs."""
    results = {}
    for pe in pe_types:
        pe_seeds = []
        for s in seeds:
            p = logit_dir / f"{pe}_seed{s}_logit_analysis.json"
            if p.exists():
                with open(p) as f:
                    pe_seeds.append(json.load(f))
        if pe_seeds:
            results[pe] = pe_seeds
    return results


def load_pretrained_logit_results(pretrained_dir: Path, model_keys):
    """Load pretrained model logit analysis JSONs, grouped by PE type."""
    results = {}  # keyed by pe_type, values are lists of result dicts
    for mk in model_keys:
        p = pretrained_dir / f"{mk}_logit_analysis.json"
        if not p.exists():
            continue
        with open(p) as f:
            r = json.load(f)
        pe = r.get("pe_type", "unknown")
        results.setdefault(pe, []).append(r)
    return results


# ---------------------------------------------------------------------------
# FL1 — M_bar_r heatmaps (logit space)
# ---------------------------------------------------------------------------

def fig_logit_mbar_heatmaps(logit_results: dict, max_lag: int = 5) -> plt.Figure:
    """Heatmaps of M_bar_r for each PE type in logit space."""
    _apply_style()
    pe_types = [pe for pe in ["rope", "alibi", "absolute"] if pe in logit_results]
    n_pe = len(pe_types)

    fig, axes = plt.subplots(1, n_pe, figsize=(4.5 * n_pe, 4))
    if n_pe == 1:
        axes = [axes]

    for ax, pe in zip(axes, pe_types):
        # Average M_bar_r across seeds
        mbar_list = [np.array(r["M_bar_r"]) for r in logit_results[pe]]
        mbar_avg = np.mean(mbar_list, axis=0)  # (max_lag+1, m, m)

        # Show lag=1 as representative
        lag = 1
        mat = mbar_avg[lag]
        vmax = np.abs(mat).max()
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{PE_LABELS.get(pe, pe)} — logit $\\bar{{M}}_{{r={lag}}}$")
        ax.set_xlabel("PCA dim $j$")
        ax.set_ylabel("PCA dim $i$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Logit-space score operator $\\bar{M}_r$ (lag 1)", y=1.02,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FL2 — RDI / H_r / SCR bars: hidden L4 vs logit
# ---------------------------------------------------------------------------

def fig_logit_vs_hidden_bars(logit_results: dict, hidden_summary: dict,
                              compare_layer_idx: int = -1) -> plt.Figure:
    """Bar chart comparing hidden-state vs logit metrics."""
    _apply_style()
    pe_types = [pe for pe in ["rope", "alibi", "absolute"] if pe in logit_results]

    # Determine comparison layer
    sample_pe = pe_types[0]
    rdi_lw = _unwrap(hidden_summary.get(sample_pe, {}).get("RDI_layerwise", []))
    n_layers = len(rdi_lw)
    c_idx = compare_layer_idx if compare_layer_idx >= 0 else (n_layers - 1)
    layer_name = f"L{c_idx + 1}"

    metrics = ["RDI", "H_r (mean)", "Probe acc."]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    x = np.arange(len(pe_types))
    width = 0.35

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        hidden_vals = []
        logit_vals = []

        for pe in pe_types:
            pe_data = hidden_summary.get(pe, {})
            logit_seeds = logit_results[pe]

            if metric == "RDI":
                rdi = _unwrap(pe_data.get("RDI_layerwise", [0]*(c_idx+1)))
                hidden_vals.append(rdi[c_idx])
                logit_vals.append(np.mean([r["RDI"] for r in logit_seeds]))
            elif metric == "H_r (mean)":
                hr_lw = _unwrap(pe_data.get("H_r_layerwise", [[0]]*(c_idx+1)))
                hidden_vals.append(np.mean(hr_lw[c_idx]))
                logit_vals.append(np.mean([np.mean(r["H_r"]) for r in logit_seeds]))
            elif metric == "Probe acc.":
                probe_lw = _unwrap(pe_data.get("probe_accuracy_layerwise", [0]*(c_idx+1)))
                hidden_vals.append(probe_lw[c_idx])
                logit_vals.append(np.mean([r["probe_accuracy"] for r in logit_seeds]))

        bars1 = ax.bar(x - width/2, hidden_vals, width, label=f"Hidden {layer_name}",
                        color="#4C72B0", alpha=0.8)
        bars2 = ax.bar(x + width/2, logit_vals, width, label="Logit",
                        color="#DD8452", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([PE_LABELS.get(pe, pe) for pe in pe_types])
        ax.set_title(metric)
        ax.legend(fontsize=8)

    fig.suptitle(f"Hidden {layer_name} vs Logit Space", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FL3 — Probe accuracy: logit vs all hidden layers
# ---------------------------------------------------------------------------

def fig_logit_probe_comparison(logit_results: dict, hidden_summary: dict) -> plt.Figure:
    """Position probe accuracy across hidden layers + logit space."""
    _apply_style()
    pe_types = [pe for pe in ["rope", "alibi", "absolute"] if pe in logit_results]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pe in pe_types:
        pe_data = hidden_summary.get(pe, {})
        probe_lw = _unwrap(pe_data.get("probe_accuracy_layerwise", []))
        n_layers = len(probe_lw)

        # Hidden layers
        layers = list(range(1, n_layers + 1))
        ax.plot(layers, probe_lw, 'o-', color=PE_COLORS.get(pe, "gray"),
                label=f"{PE_LABELS.get(pe, pe)} (hidden)")

        # Logit point
        logit_probe = np.mean([r["probe_accuracy"] for r in logit_results[pe]])
        ax.plot(n_layers + 1, logit_probe, 's', color=PE_COLORS.get(pe, "gray"),
                markersize=10, markeredgecolor="black", markeredgewidth=1.5)

    # Add "Logit" label on x-axis
    sample_n = len(_unwrap(hidden_summary.get(pe_types[0], {}).get("probe_accuracy_layerwise", [])))
    xticks = list(range(1, sample_n + 1)) + [sample_n + 1]
    xlabels = [f"L{i}" for i in range(1, sample_n + 1)] + ["Logit"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Representation space")
    ax.set_ylabel("Position probe accuracy")
    ax.set_title("Position probe: hidden layers vs logit space")
    ax.legend(loc="best")
    ax.axvline(x=sample_n + 0.5, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FL4 — DSM validation loss comparison
# ---------------------------------------------------------------------------

def fig_logit_dsm_loss(logit_results: dict, hidden_summary: dict) -> plt.Figure:
    """DSM validation loss across hidden layers + logit space."""
    _apply_style()
    pe_types = [pe for pe in ["rope", "alibi", "absolute"] if pe in logit_results]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pe in pe_types:
        pe_data = hidden_summary.get(pe, {})
        dsm_lw = _unwrap(pe_data.get("val_dsm_loss_layerwise", []))
        n_layers = len(dsm_lw)

        layers = list(range(1, n_layers + 1))
        ax.plot(layers, dsm_lw, 'o-', color=PE_COLORS.get(pe, "gray"),
                label=f"{PE_LABELS.get(pe, pe)} (hidden)")

        logit_dsm = np.mean([r["val_dsm_loss"] for r in logit_results[pe]])
        ax.plot(n_layers + 1, logit_dsm, 's', color=PE_COLORS.get(pe, "gray"),
                markersize=10, markeredgecolor="black", markeredgewidth=1.5)

    sample_n = len(_unwrap(hidden_summary.get(pe_types[0], {}).get("val_dsm_loss_layerwise", [])))
    xticks = list(range(1, sample_n + 1)) + [sample_n + 1]
    xlabels = [f"L{i}" for i in range(1, sample_n + 1)] + ["Logit"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Representation space")
    ax.set_ylabel("DSM validation loss")
    ax.set_title("Score model quality: hidden layers vs logit space")
    ax.legend(loc="best")
    ax.axvline(x=sample_n + 0.5, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate logit-space score-geometric figures."
    )
    parser.add_argument("--logit-dir", type=str, default=None,
                        help="Directory with *_logit_analysis.json files (Phase 1)")
    parser.add_argument("--hidden-summary", type=str, required=True,
                        help="Path to analysis_summary.json from hidden-state pipeline")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--pretrained-dir", type=str, default=None,
                        help="Directory with pretrained model logit analysis (Phase 2)")
    parser.add_argument("--pretrained-models", nargs="+",
                        default=["gpt2", "bloom-560m", "pythia-410m"],
                        help="Model keys for pretrained logit loading")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.hidden_summary) as f:
        hidden_summary = json.load(f)

    # Load logit results from Phase 1 (controlled) and/or Phase 2 (pretrained)
    logit_results = {}
    if args.logit_dir:
        logit_results = load_logit_results(Path(args.logit_dir), args.pe_types, args.seeds)
    if args.pretrained_dir:
        pretrained = load_pretrained_logit_results(
            Path(args.pretrained_dir), args.pretrained_models)
        # Merge: pretrained results grouped by PE type
        for pe, results_list in pretrained.items():
            logit_results.setdefault(pe, []).extend(results_list)

    if not logit_results:
        print("ERROR: No logit results found. Provide --logit-dir and/or --pretrained-dir.")
        return

    saved = []

    # FL1 — M_bar_r heatmaps
    try:
        fig = fig_logit_mbar_heatmaps(logit_results)
        p = out_dir / "FL1_logit_mbar_heatmaps.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
    except Exception as e:
        print(f"  [FL1 error]: {e}")

    # FL2 — Hidden vs Logit bars
    try:
        fig = fig_logit_vs_hidden_bars(logit_results, hidden_summary)
        p = out_dir / "FL2_logit_vs_hidden_bars.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
    except Exception as e:
        print(f"  [FL2 error]: {e}")

    # FL3 — Probe comparison
    try:
        fig = fig_logit_probe_comparison(logit_results, hidden_summary)
        p = out_dir / "FL3_logit_probe_comparison.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
    except Exception as e:
        print(f"  [FL3 error]: {e}")

    # FL4 — DSM loss
    try:
        fig = fig_logit_dsm_loss(logit_results, hidden_summary)
        p = out_dir / "FL4_logit_dsm_loss.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
    except Exception as e:
        print(f"  [FL4 error]: {e}")

    print(f"\nSaved {len(saved)} figures:")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    main()
