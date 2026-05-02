"""
Aggregate ablation results across direction sets (score-SVD, probe-hidden,
SAE variants, random-PCA) into one comparison table.

Reads the per-layer b2 JSONs produced by:
  - run_ablation_study.py          -> b2_probe_direction_ablation.json
                                       (probe_hidden, probe_abs, score_svd)
                                       dose_response.json (random_control)
  - run_sae_baseline.py            -> b2_sae_direction_ablation.json (L1 SAE)
  - run_modern_sae_baselines.py    -> b2_sae_direction_ablation.json
                                       (TopK / BatchTopK / T-SAE — same schema,
                                        each under its own per-variant subdir)

Multiple SAE variants can be compared in a single run by passing one or
more of --sae-l1-dir / --sae-topk-dir / --sae-batchtopk-dir / --sae-tsae-dir.
The legacy --sae-ablation-dir flag is preserved as an alias for L1.

Variants with sparse cell coverage (e.g. TopK on 4 cells, T-SAE on 1 cell)
work transparently — cells where the variant wasn't run render as "—".

Produces:
  <out-dir>/comparison_summary.json   per-cell Δ + ratios for every direction set
  <out-dir>/comparison_table.md       paper-friendly markdown table
  <out-dir>/comparison_table.tex      LaTeX-source table for paper3.tex
  <out-dir>/F_ablation_comparison.pdf 4-panel bar chart (1 per layer), log y-scale
  <out-dir>/F_score_vs_baselines_ratios.pdf  ratio bars (score / each baseline)

Usage
-----
    # Single SAE variant (legacy — equivalent to --sae-l1-dir)
    python scripts/compare_ablation_baselines.py \\
        --probe-ablation-dir results/lagpair_ablation_3seed/ablation \\
        --sae-ablation-dir   results/sae_baseline_3seed/ablation \\
        --layers 1 2 3 4 \\
        --pe-types rope alibi absolute \\
        --seeds 0 1 2 \\
        --out-dir results/ablation_comparison

    # All four SAE variants together
    python scripts/compare_ablation_baselines.py \\
        --probe-ablation-dir   results/lagpair_ablation_3seed/ablation \\
        --sae-l1-dir           results/sae_baseline_3seed/ablation \\
        --sae-topk-dir         results/sae_modern_3seed/topk/ablation \\
        --sae-batchtopk-dir    results/sae_modern_3seed/batchtopk/ablation \\
        --sae-tsae-dir         results/sae_modern_3seed/tsae/ablation \\
        --layers 1 2 3 4 \\
        --pe-types rope alibi absolute \\
        --seeds 0 1 2 \\
        --out-dir results/ablation_comparison_all_saes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NON_IID = ["variable_lag_copy", "absolute_anchor", "order_sensitive", "distance_bucket"]


# Ordered registry of supported SAE variants. Order here defines column
# order in tables and bar order in figures. Each entry: (key, label, color).
# `key` becomes the dict key as `sae_<key>` and the ratio key
# `score_vs_sae_<key>`.
SAE_VARIANT_REGISTRY: List[Tuple[str, str, str]] = [
    ("l1",         "L1 SAE",     "#2ca02c"),
    ("topk",       "TopK SAE",   "#9467bd"),
    ("batchtopk",  "BatchTopK",  "#e377c2"),
    ("tsae",       "T-SAE",      "#8c564b"),
]


# ============================================================================
# Loaders
# ============================================================================

def cell_key(pe: str, seed: int, layer: int, lag: int = 1) -> str:
    """Match the string-tuple keys used in the b2 JSONs."""
    return f"('{pe}', {seed}, {layer}, {lag})"


def load_json_or_none(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def family_avg_delta(
    losses_by_alpha: Dict[str, Dict[str, float]],
    alpha_max: str = "2.0",
    alpha_min: str = "0.0",
) -> Optional[float]:
    """Δ = mean over 4 non-iid families of (loss[α=α_max] - loss[α=α_min])."""
    if alpha_max not in losses_by_alpha or alpha_min not in losses_by_alpha:
        return None
    a0 = losses_by_alpha[alpha_min]
    aM = losses_by_alpha[alpha_max]
    deltas = [aM.get(f, 0.0) - a0.get(f, 0.0) for f in NON_IID if f in a0 and f in aM]
    if not deltas:
        return None
    return float(np.mean(deltas))


# ============================================================================
# Per-cell extraction
# ============================================================================

def extract_per_seed_deltas(
    probe_b2: Optional[Dict[str, Any]],
    dose: Optional[Dict[str, Any]],
    sae_variants_b2: Dict[str, Optional[Dict[str, Any]]],
    pe: str,
    seeds: List[int],
    layer: int,
    lag: int,
) -> Dict[str, List[Optional[float]]]:
    """For one (PE, layer), return {direction_set: [Δ_seed0, Δ_seed1, Δ_seed2]}.

    Direction sets handled:
      score_svd, probe_hidden, probe_abs   (from probe_b2)
      random_control                        (from dose, under "random_control" key)
      sae_<variant>                         (from sae_variants_b2[variant])
    Missing seeds/sets become None.
    """
    out: Dict[str, List[Optional[float]]] = {
        "score_svd": [],
        "probe_hidden": [],
        "probe_abs": [],
        "random_control": [],
    }
    for v in sae_variants_b2:
        out[f"sae_{v}"] = []

    for seed in seeds:
        k = cell_key(pe, seed, layer, lag)

        # probe-side direction sets
        if probe_b2 and k in probe_b2:
            cell = probe_b2[k]
            out["score_svd"].append(family_avg_delta(cell.get("score_svd", {})))
            out["probe_hidden"].append(family_avg_delta(cell.get("probe_hidden", {})))
            out["probe_abs"].append(family_avg_delta(cell.get("probe_abs", {})))
        else:
            out["score_svd"].append(None)
            out["probe_hidden"].append(None)
            out["probe_abs"].append(None)

        # random_control lives in dose_response.json
        if dose and k in dose:
            rc = dose[k].get("random_control", {})
            out["random_control"].append(family_avg_delta(rc))
        else:
            out["random_control"].append(None)

        # Each SAE variant uses the same per-cell key "sae_hidden"
        # (run_modern_sae_baselines writes the same schema as the L1 baseline,
        # one b2 JSON per variant under its own subdir).
        for v, b2 in sae_variants_b2.items():
            if b2 and k in b2:
                out[f"sae_{v}"].append(family_avg_delta(b2[k].get("sae_hidden", {})))
            else:
                out[f"sae_{v}"].append(None)

    return out


def aggregate(deltas: List[Optional[float]]) -> Dict[str, Any]:
    """Mean ± std across non-None seed deltas."""
    vals = [d for d in deltas if d is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "values": deltas}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n": len(vals),
        "values": deltas,
    }


def median_per_seed_ratio(
    numers: List[Optional[float]],
    denoms: List[Optional[float]],
    eps: float = 1e-6,
) -> Optional[float]:
    """Per-seed ratio (numer / max(denom, 0) + eps), then median across seeds.
    Returns None if no seed has both values."""
    ratios = []
    for n, d in zip(numers, denoms):
        if n is None or d is None:
            continue
        ratios.append(n / (max(d, 0.0) + eps))
    if not ratios:
        return None
    return float(np.median(ratios))


# ============================================================================
# Variant resolution — CLI flags → ordered list of active SAE variants
# ============================================================================

def resolve_active_variants(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return the ordered list of SAE variants the user asked us to compare.

    --sae-ablation-dir is a back-compat alias for --sae-l1-dir.
    Variants with no dir specified are silently skipped.
    """
    out: List[Dict[str, Any]] = []
    for key, label, color in SAE_VARIANT_REGISTRY:
        attr = f"sae_{key}_dir"
        path = getattr(args, attr, None)
        if key == "l1" and path is None and getattr(args, "sae_ablation_dir", None):
            path = args.sae_ablation_dir
        if path:
            out.append({
                "key": key,
                "label": label,
                "color": color,
                "dir": Path(path),
            })
    return out


# ============================================================================
# Table rendering
# ============================================================================

def _fmt(x: Optional[float], unit: str = "") -> str:
    if x is None:
        return "—"
    if unit == "x":
        if abs(x) >= 100:
            return f"{x:.0f}×"
        return f"{x:.1f}×"
    if abs(x) < 0.001 and x != 0:
        return f"{x:+.1e}"
    return f"{x:+.3f}"


def render_markdown_table(
    rows: List[Dict[str, Any]],
    pe_types: List[str],
    layers: List[int],
    active_variants: List[Dict[str, Any]],
    drop_random_column: bool = False,
) -> str:
    """One row per (PE, layer); Δ columns + ratio columns for each baseline.

    drop_random_column:
        Omits Δ_random-PCA and score/random columns. Random-PCA is the
        strict matched-subspace null whose discussion belongs in the
        appendix calibration, not the headline causal claim.
    """
    lines: List[str] = []
    lines.append("# Ablation baseline comparison")
    lines.append("")
    lines.append("All Δ values are mean across seeds of (CE at α=2 minus CE at α=0), "
                 "averaged over the 4 non-iid task families. Ratios are **median per-seed**, "
                 "matching the convention in `tab:ablation_probe_vs_score`.")
    lines.append("")
    if active_variants:
        lines.append(f"SAE variants in comparison: " +
                     ", ".join(v["label"] for v in active_variants))
        lines.append("")

    # ---- Δ values ----------------------------------------------------------
    lines.append("## Δ values (nats)")
    lines.append("")
    headers = ["PE", "Layer", "Δ score-SVD", "Δ probe-hidden"]
    headers += [f"Δ {v['label']}" for v in active_variants]
    if not drop_random_column:
        headers.append("Δ random-PCA")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        cells = [r["pe"], f"L{r['layer']}",
                 _fmt(r["score_svd"]["mean"]),
                 _fmt(r["probe_hidden"]["mean"])]
        for v in active_variants:
            cells.append(_fmt(r[f"sae_{v['key']}"]["mean"]))
        if not drop_random_column:
            cells.append(_fmt(r["random_control"]["mean"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Median per-seed ratios -------------------------------------------
    lines.append("## Median per-seed ratios (score-SVD vs each baseline)")
    lines.append("")
    headers = ["PE", "Layer", "score / probe-hidden"]
    headers += [f"score / {v['label']}" for v in active_variants]
    if not drop_random_column:
        headers.append("score / random-PCA")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        cells = [r["pe"], f"L{r['layer']}",
                 _fmt(r["ratios"]["score_vs_probe_hidden"], unit="x")]
        for v in active_variants:
            cells.append(_fmt(r["ratios"][f"score_vs_sae_{v['key']}"], unit="x"))
        if not drop_random_column:
            cells.append(_fmt(r["ratios"]["score_vs_random"], unit="x"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Reading guide -----------------------------------------------------
    lines.append("## Reading guide")
    lines.append("")
    lines.append("- **score / probe-hidden > 1**: SBTG identifies directions a linear probe in "
                 "hidden space cannot — the empirical Prop 1 result. Headline.")
    if active_variants:
        v_keys = {v["key"] for v in active_variants}
        if "l1" in v_keys:
            lines.append("- **score / L1 SAE > 1**: SBTG also beats non-linear, sparsity-encouraged "
                         "marginal-readout features. Stronger Prop 1 evidence than the linear-probe "
                         "comparison alone.")
        if "topk" in v_keys:
            lines.append("- **score / TopK SAE > 1**: the gap isn't just \"L1 is bad\" — modern "
                         "per-token-reconstruction objectives still miss the structure.")
        if "batchtopk" in v_keys:
            lines.append("- **score / BatchTopK SAE > 1**: even relaxing the per-token frame "
                         "(BatchTopK lets a token recruit more features when neighbours recruit "
                         "fewer) doesn't close the gap. Critical evidence given paper3 App F.1's "
                         "explicit conjecture about BatchTopK partially closing the gap.")
        if "tsae" in v_keys:
            lines.append("- **score / T-SAE > 1**: even an SAE objective explicitly augmented with "
                         "sequential structure (InfoNCE between adjacent positions) still misses "
                         "the joint coupling SBTG reads.")
    if not drop_random_column:
        lines.append("- **score / random-PCA > 1**: SBTG directions are privileged within the "
                     "active subspace, not arbitrary. Variable across cells (see "
                     "`docs/ablation_explainer.md`).")
    else:
        lines.append("- *Random-PCA column dropped from headline table* — see `comparison_summary.json` "
                     "for the per-cell values; calibration of the strict matched-subspace null "
                     "is discussed in the paper's appendix.")
    return "\n".join(lines)


def render_latex_table(
    rows: List[Dict[str, Any]],
    active_variants: List[Dict[str, Any]],
    drop_random_column: bool = False,
) -> str:
    """LaTeX source for the median-ratio table — slot into paper3.tex appendix.

    Score-SVD column followed by score/probe and one score/<SAE-variant>
    column per active variant, then optionally score/random.
    """
    n_ratio_cols = 1 + len(active_variants) + (0 if drop_random_column else 1)
    col_spec = "@{}ll" + "c" * (1 + n_ratio_cols) + "@{}"

    lines = [r"\begin{table}[h]", r"\centering"]
    if drop_random_column:
        lines.append(
            r"\caption{\textbf{Score-SVD vs marginal-readable baselines.} "
            r"$\Delta$ masked-CE at $\alpha = 2$, family-averaged excluding "
            r"\texttt{iid\_random}, mean across 3 seeds. Ratios are median "
            r"per-seed of $\Delta_{\text{score-SVD}} / \Delta_{\text{baseline}}$. "
            r"Calibration of the matched-subspace random-PCA null is "
            r"discussed in the appendix.}"
        )
    else:
        lines.append(
            r"\caption{\textbf{Score-SVD vs all baselines.} "
            r"$\Delta$ masked-CE at $\alpha = 2$, family-averaged excluding "
            r"\texttt{iid\_random}, mean across 3 seeds. Ratios are median "
            r"per-seed of $\Delta_{\text{score-SVD}} / \Delta_{\text{baseline}}$.}"
        )
    lines.append(r"\label{tab:ablation_three_baselines}")
    lines.append(r"\smallskip\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_cells = [r"\textbf{PE}", r"\textbf{Layer}", r"$\Delta_{\text{score}}$",
                    r"score/probe"]
    for v in active_variants:
        # Make the LaTeX header readable; escape underscores by avoiding them.
        header_cells.append(r"score/\textsc{" + v["label"].replace(" SAE", "").replace(" ", "") + "}")
    if not drop_random_column:
        header_cells.append(r"score/random")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    last_pe = None
    for r in rows:
        score = r["score_svd"]["mean"]
        score_str = "—" if score is None else f"${score:+.3f}$"
        row_cells = [r["pe"] if r["pe"] != last_pe else "", f"L{r['layer']}", score_str]
        last_pe = r["pe"]

        rp = r["ratios"]["score_vs_probe_hidden"]
        row_cells.append(_latex_ratio(rp))
        for v in active_variants:
            rv = r["ratios"][f"score_vs_sae_{v['key']}"]
            row_cells.append(_latex_ratio(rv))
        if not drop_random_column:
            rr = r["ratios"]["score_vs_random"]
            row_cells.append(_latex_ratio(rr))

        lines.append("  " + " & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _latex_ratio(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x >= 1:
        if x >= 100:
            return rf"$\mathbf{{{x:.0f}\times}}$"
        return rf"$\mathbf{{{x:.1f}\times}}$"
    return f"${x:.2f}\\times$"


# ============================================================================
# Figures
# ============================================================================

def fig_per_layer_bars(
    rows: List[Dict[str, Any]],
    layers: List[int],
    active_variants: List[Dict[str, Any]],
    out_path: Path,
    drop_random_column: bool = False,
):
    """One panel per layer; x-axis PE; bars per direction set; log y-scale."""
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    pe_order: List[str] = []
    for r in rows:
        if r["pe"] not in pe_order:
            pe_order.append(r["pe"])

    direction_sets: List[Tuple[str, str, str]] = [
        ("score_svd",       "Score-SVD",      "#1f77b4"),
        ("probe_hidden",    "Probe (hidden)", "#ff7f0e"),
    ]
    for v in active_variants:
        direction_sets.append((f"sae_{v['key']}", v["label"], v["color"]))
    if not drop_random_column:
        direction_sets.append(("random_control", "Random (PCA)", "#999999"))

    n_bars = len(direction_sets)
    # Auto-narrow bars when many baselines are present.
    width = max(0.10, min(0.20, 0.85 / n_bars))
    offset = (n_bars - 1) / 2.0

    for ax, layer in zip(axes, layers):
        ax.set_title(f"Layer {layer}")
        for i, (key, label, color) in enumerate(direction_sets):
            heights, errs = [], []
            for pe in pe_order:
                row = next((r for r in rows if r["pe"] == pe and r["layer"] == layer), None)
                if row is None or row[key]["mean"] is None:
                    heights.append(np.nan)
                    errs.append(0.0)
                else:
                    heights.append(max(row[key]["mean"], 1e-4))   # clip for log scale
                    errs.append(row[key]["std"] or 0.0)
            xs = np.arange(len(pe_order)) + (i - offset) * width
            ax.bar(xs, heights, width=width, color=color, label=label,
                   yerr=errs, error_kw={"capsize": 2, "elinewidth": 0.7})
        ax.set_xticks(np.arange(len(pe_order)))
        ax.set_xticklabels(pe_order)
        ax.set_yscale("log")
        ax.set_ylabel("Δ masked-CE (nats)" if layer == layers[0] else "")
        ax.grid(True, alpha=0.3, which="both", axis="y")

    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    suptitle = ("Ablation Δ across direction sets — score-SVD vs marginal-readable baselines"
                if drop_random_column else
                "Ablation Δ across direction sets")
    fig.suptitle(f"{suptitle} (mean ± std, 3 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_ratio_panels(
    rows: List[Dict[str, Any]],
    layers: List[int],
    active_variants: List[Dict[str, Any]],
    out_path: Path,
    drop_random_column: bool = False,
):
    """One panel per ratio: score/probe, score/<each SAE variant>, optional score/random."""
    pe_order: List[str] = []
    for r in rows:
        if r["pe"] not in pe_order:
            pe_order.append(r["pe"])

    panels: List[Tuple[str, str]] = [("score_vs_probe_hidden", "score-SVD / probe-hidden")]
    for v in active_variants:
        panels.append((f"score_vs_sae_{v['key']}", f"score-SVD / {v['label']}"))
    if not drop_random_column:
        panels.append(("score_vs_random", "score-SVD / random-PCA"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (rkey, title) in zip(axes, panels):
        for pe in pe_order:
            ratios = []
            for L in layers:
                row = next((r for r in rows if r["pe"] == pe and r["layer"] == L), None)
                if row is None:
                    ratios.append(np.nan)
                else:
                    v = row["ratios"].get(rkey)
                    ratios.append(v if v is not None else np.nan)
            ax.plot(layers, ratios, marker="o", label=pe, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", lw=0.8)
        ax.set_xlabel("layer")
        if ax is axes[0]:
            ax.set_ylabel("median per-seed ratio")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, which="both", axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Score-SVD vs each baseline — median per-seed ratios across (PE, layer)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-ablation-dir", type=str, required=True,
                   help="Path to .../ablation/ containing L<layer>/b2_probe_direction_ablation.json + dose_response.json")

    # Multi-variant SAE flags. Any subset can be active. The legacy
    # --sae-ablation-dir is treated as L1.
    p.add_argument("--sae-ablation-dir", type=str, default=None,
                   help="[legacy] Alias for --sae-l1-dir.")
    p.add_argument("--sae-l1-dir", type=str, default=None,
                   help="Path to .../ablation/ for the L1-penalized SAE baseline (run_sae_baseline.py output).")
    p.add_argument("--sae-topk-dir", type=str, default=None,
                   help="Path to .../<run>/topk/ablation/ (run_modern_sae_baselines.py --variant topk).")
    p.add_argument("--sae-batchtopk-dir", type=str, default=None,
                   help="Path to .../<run>/batchtopk/ablation/ (run_modern_sae_baselines.py --variant batchtopk).")
    p.add_argument("--sae-tsae-dir", type=str, default=None,
                   help="Path to .../<run>/tsae/ablation/ (run_modern_sae_baselines.py --variant tsae).")

    p.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--drop-random-column", action="store_true",
                   help="Omit Δ_random-PCA and score/random columns from the rendered "
                        "Markdown / LaTeX tables and figures. The data is still computed "
                        "and stored in comparison_summary.json. Use this for the headline "
                        "paper table.")
    args = p.parse_args()

    probe_root = Path(args.probe_ablation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_variants = resolve_active_variants(args)

    print("=" * 72)
    print(" Ablation baseline comparison")
    print("=" * 72)
    print(f"  probe ablation dir: {probe_root}")
    if active_variants:
        for v in active_variants:
            print(f"  SAE [{v['key']}]: {v['dir']}")
    else:
        print("  SAE: (no variant dir provided — SAE columns will read —)")
    print(f"  cells: {len(args.pe_types)} PE × {len(args.seeds)} seeds × {len(args.layers)} layers "
          f"= {len(args.pe_types) * len(args.seeds) * len(args.layers)}")
    print()

    rows: List[Dict[str, Any]] = []
    for layer in args.layers:
        probe_b2 = load_json_or_none(probe_root / f"L{layer}" / "b2_probe_direction_ablation.json")
        dose     = load_json_or_none(probe_root / f"L{layer}" / "dose_response.json")
        sae_variants_b2: Dict[str, Optional[Dict[str, Any]]] = {
            v["key"]: load_json_or_none(v["dir"] / f"L{layer}" / "b2_sae_direction_ablation.json")
            for v in active_variants
        }

        if probe_b2 is None:
            print(f"  WARN: missing probe b2 at L{layer}, skipping")
            continue

        for pe in args.pe_types:
            per_seed = extract_per_seed_deltas(
                probe_b2=probe_b2, dose=dose, sae_variants_b2=sae_variants_b2,
                pe=pe, seeds=args.seeds, layer=layer, lag=args.lag,
            )

            agg = {key: aggregate(per_seed[key]) for key in per_seed}

            ratios = {
                "score_vs_probe_hidden": median_per_seed_ratio(
                    per_seed["score_svd"], per_seed["probe_hidden"]),
                "score_vs_random": median_per_seed_ratio(
                    per_seed["score_svd"], per_seed["random_control"]),
            }
            for v in active_variants:
                ratios[f"score_vs_sae_{v['key']}"] = median_per_seed_ratio(
                    per_seed["score_svd"], per_seed[f"sae_{v['key']}"])

            rows.append({
                "pe": pe,
                "layer": layer,
                **agg,
                "ratios": ratios,
            })

    # ----- Print headline table to stdout ---------------------------------
    sae_headers = [v["key"][:6] for v in active_variants]
    head_line = (f"  {'PE':<10} {'L':<3} {'Δscore':>9}  {'Δprobe':>9}  "
                 + "  ".join(f"Δ{h:>6}" for h in sae_headers)
                 + ("  Δrand".rjust(8) if not args.drop_random_column else "")
                 + "  s/p".rjust(8)
                 + "".join(f"  s/{h}".rjust(8) for h in sae_headers)
                 + ("  s/r".rjust(8) if not args.drop_random_column else ""))
    print(head_line)
    print("  " + "-" * max(40, len(head_line)))

    def fnum(x, unit=""):
        if x is None:
            return "—".rjust(8)
        if unit == "x":
            if x >= 100:
                return f"{x:6.0f}×"
            return f"{x:6.1f}×"
        return f"{x:+.4f}"

    for r in rows:
        cells: List[str] = [f"  {r['pe']:<10} L{r['layer']:<2}",
                            f"{fnum(r['score_svd']['mean']):>9}",
                            f"{fnum(r['probe_hidden']['mean']):>9}"]
        for v in active_variants:
            sae_key = f"sae_{v['key']}"
            cells.append(f"{fnum(r[sae_key]['mean']):>9}")
        if not args.drop_random_column:
            cells.append(f"{fnum(r['random_control']['mean']):>9}")
        cells.append(f"{fnum(r['ratios']['score_vs_probe_hidden'], unit='x'):>8}")
        for v in active_variants:
            ratio_key = f"score_vs_sae_{v['key']}"
            cells.append(f"{fnum(r['ratios'][ratio_key], unit='x'):>8}")
        if not args.drop_random_column:
            cells.append(f"{fnum(r['ratios']['score_vs_random'], unit='x'):>8}")
        print("  ".join(cells))

    # ----- Persist artifacts ----------------------------------------------
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w") as fp:
        json.dump({
            "config": {
                "pe_types": args.pe_types,
                "seeds": args.seeds,
                "layers": args.layers,
                "lag": args.lag,
                "probe_ablation_dir": str(probe_root),
                "sae_variants": {
                    v["key"]: str(v["dir"]) for v in active_variants
                },
            },
            "rows": rows,
        }, fp, indent=2)

    md_path = out_dir / "comparison_table.md"
    md_path.write_text(render_markdown_table(
        rows, args.pe_types, args.layers, active_variants,
        drop_random_column=args.drop_random_column,
    ))

    tex_path = out_dir / "comparison_table.tex"
    tex_path.write_text(render_latex_table(
        rows, active_variants,
        drop_random_column=args.drop_random_column,
    ))

    fig_per_layer_bars(
        rows, args.layers, active_variants,
        out_dir / "F_ablation_comparison.pdf",
        drop_random_column=args.drop_random_column,
    )
    fig_ratio_panels(
        rows, args.layers, active_variants,
        out_dir / "F_score_vs_baselines_ratios.pdf",
        drop_random_column=args.drop_random_column,
    )

    print()
    print("  Wrote:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"    {f}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
