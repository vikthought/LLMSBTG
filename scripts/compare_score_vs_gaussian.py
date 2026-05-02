"""
Score vs parametric Gaussian comparison for the SBTG positional operator.

Reads two parallel files:

  --score-metrics    /.../lagpair_analysis_3seed/lagpair_metrics.json
                     (produced by run_lagpair_analysis.py — the score-model
                      version of M_r and its scalar diagnostics)

  --gaussian-metrics /.../gaussian_lagpair_metrics.json
                     (produced by run_lagpair_gaussian_baseline.py — the
                      parametric Gaussian Σ^{-1} version)

Both files share the same cell schema (key = `{pe}_s{seed}_L{layer}`), so
comparison is direct.  The script answers the reviewer question:

  > "What is the score model adding over a parametric Gaussian baseline?"

via three artifacts:

  1. comparison_table.md / .tex — per (PE, layer) means of the headline
     scalars (A_r(1), S_r(1), C_r(1), AS_r(1), beta, SI), score vs Gaussian
     side by side, plus a Δ column.
  2. F_envelope_overlay.pdf — empirical AS_r/AS_1 envelope vs lag at
     L4 for each PE, score (solid) and Gaussian (dashed) overlaid.
  3. F_scalar_scatter.pdf — per-cell scatter of score-derived vs Gaussian-
     derived A_r(1), C_r(1) etc., to see whether the agreement is uniform
     or uneven across cells.

The headline number reported at the end is the per-(PE, layer) Spearman
correlation between score and Gaussian C_r(1), which is the diagnostic that
distinguishes ALiBi (rank-1) from RoPE/Absolute (distributed) in the paper.
If Gaussian recovers that ranking, Outcome A or C; if it doesn't, Outcome B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# IO
# ============================================================================

def aggregate_cells_by_pe_layer(
    metrics: Dict, lag: int = 1
) -> Dict[Tuple[str, int], Dict[str, Tuple[float, float]]]:
    """Per-(PE, layer): mean ± std across seeds of headline scalars at given lag.

    Returns
    -------
    dict keyed by (pe, layer), value = {field: (mean, std)} for fields
    A_r, S_r, C_r, AS_r, plus SI and beta from the cell top-level.
    """
    bucket: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for k, c in metrics.items():
        pe = c.get("pe_type")
        L = int(c.get("layer", 0))
        seed = int(c.get("seed", -1))
        if not pe or L == 0:
            continue
        b = bucket.setdefault((pe, L), {
            "SI": [], "beta": [],
            "A_r": [], "S_r": [], "C_r": [], "AS_r": [],
        })
        b["SI"].append(c.get("SI", float("nan")))
        b["beta"].append(c.get("beta", float("nan")))
        lag_entry = c.get("lags", {}).get(str(lag), {})
        for f in ("A_r", "S_r", "C_r", "AS_r"):
            b[f].append(lag_entry.get(f, float("nan")))
    out = {}
    for key, fields in bucket.items():
        out[key] = {
            f: (float(np.nanmean(v)), float(np.nanstd(v)))
            for f, v in fields.items()
        }
    return out


# ============================================================================
# Side-by-side table
# ============================================================================

def render_markdown_table(
    score_agg: Dict, gauss_agg: Dict, pe_types: List[str], layers: List[int],
) -> str:
    lines = []
    lines.append("# Score vs parametric Gaussian — SBTG fingerprints")
    lines.append("")
    lines.append("Mean across 3 seeds at lag r=1.  S = score-model M_r, G = Gaussian Σ⁻¹ block.")
    lines.append("Δ = (S − G); a small Δ at every diagnostic means the residual-stream window distribution is well-approximated as Gaussian for this experiment, and the score model isn't doing extra work.  A large Δ on a specific diagnostic (especially C_r) would mean the score model is reading non-Gaussian structure that Σ⁻¹ misses.")
    lines.append("")
    for field in ("A_r", "S_r", "C_r", "AS_r"):
        lines.append(f"## {field} at lag 1")
        lines.append("")
        lines.append(f"| PE | Layer | {field}^S | {field}^G | Δ ({field}) |")
        lines.append("|---|---|---|---|---|")
        for pe in pe_types:
            for L in layers:
                s = score_agg.get((pe, L), {}).get(field)
                g = gauss_agg.get((pe, L), {}).get(field)
                if s is None or g is None:
                    continue
                s_mean, s_std = s
                g_mean, g_std = g
                d = s_mean - g_mean
                lines.append(
                    f"| {pe} | L{L} | {s_mean:+.4f} ± {s_std:.4f} | "
                    f"{g_mean:+.4f} ± {g_std:.4f} | {d:+.4f} |"
                )
        lines.append("")
    return "\n".join(lines)


def render_latex_table(
    score_agg: Dict, gauss_agg: Dict, pe_types: List[str], layers: List[int],
) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Score-model vs parametric Gaussian fingerprints at lag $r{=}1$.}  "
                 r"For each PE and layer, mean across 3 seeds; $S$ is the score-model "
                 r"$M_r$ scalar; $G$ is the same scalar computed from "
                 r"$M_r^{\text{Gauss}} = -\Lambda_{w-1, w-r-1}$ with "
                 r"$\Lambda = (\Sigma + \lambda I)^{-1}$ on the windowed test "
                 r"activations.  Small $|S - G|$ everywhere means the residual-stream "
                 r"window distribution is well-approximated as Gaussian for the diagnostic "
                 r"in question; large $|S - G|$ identifies which diagnostics genuinely "
                 r"require the score-model machinery.}")
    lines.append(r"\label{tab:gaussian_vs_score}")
    lines.append(r"\smallskip\small")
    lines.append(r"\begin{tabular}{@{}llcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"PE & Layer & $A_r^S$ & $A_r^G$ & $C_r^S$ & $C_r^G$ & $\mathrm{AS}_r^S$ & $\mathrm{AS}_r^G$ \\")
    lines.append(r"\midrule")
    last_pe = None
    for pe in pe_types:
        for L in layers:
            s = score_agg.get((pe, L), {})
            g = gauss_agg.get((pe, L), {})
            if not s or not g:
                continue
            row = (pe if pe != last_pe else "")
            last_pe = pe
            lines.append(
                f"  {row} & L{L} & "
                f"${s['A_r'][0]:+.3f}$ & ${g['A_r'][0]:+.3f}$ & "
                f"${s['C_r'][0]:.3f}$ & ${g['C_r'][0]:.3f}$ & "
                f"${s['AS_r'][0]:+.3f}$ & ${g['AS_r'][0]:+.3f}$ \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# Figures
# ============================================================================

def fig_envelope_overlay(
    score_metrics: Dict, gauss_metrics: Dict,
    pe_types: List[str], layer: int, max_lag: int, out_path: Path,
):
    """AS_r/AS_1 vs r at the given layer, score (solid) and Gaussian (dashed)
    overlaid, one line per PE.  This is the figure that visualizes whether
    the architectural fingerprints (rank-one ALiBi, oscillatory RoPE,
    high-amplitude Absolute) reproduce under the Gaussian fit."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = {"rope": "#1f77b4", "alibi": "#ff7f0e", "absolute": "#2ca02c"}

    rs = list(range(0, max_lag + 1))
    for pe in pe_types:
        s_envs, g_envs = [], []
        for seed in [0, 1, 2]:
            cell_s = score_metrics.get(f"{pe}_s{seed}_L{layer}")
            cell_g = gauss_metrics.get(f"{pe}_s{seed}_L{layer}")
            if cell_s is None or cell_g is None:
                continue
            s1 = cell_s["lags"]["1"]["AS_r"]
            g1 = cell_g["lags"]["1"]["AS_r"]
            s_envs.append([cell_s["lags"][str(r)]["AS_r"] / max(s1, 1e-12) for r in rs])
            g_envs.append([cell_g["lags"][str(r)]["AS_r"] / max(g1, 1e-12) for r in rs])
        if not s_envs:
            continue
        s_mean = np.mean(s_envs, axis=0); s_std = np.std(s_envs, axis=0)
        g_mean = np.mean(g_envs, axis=0); g_std = np.std(g_envs, axis=0)
        c = cmap.get(pe, "k")
        ax.plot(rs, s_mean, color=c, marker="o", linewidth=2, label=f"{pe} (score)")
        ax.fill_between(rs, s_mean - s_std, s_mean + s_std, color=c, alpha=0.15)
        ax.plot(rs, g_mean, color=c, marker="s", linewidth=1.5, linestyle="--",
                alpha=0.85, label=f"{pe} (Gaussian)")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lag r")
    ax.set_ylabel(r"$AS_r(r) / AS_r(1)$")
    ax.set_title(f"Score (solid) vs Gaussian (dashed) envelope at L{layer}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_scalar_scatter(
    score_metrics: Dict, gauss_metrics: Dict, max_lag: int,
    out_path: Path, lag: int = 1,
):
    """Per-cell scatter of score (x) vs Gaussian (y) for A_r, C_r, AS_r at
    fixed lag.  Cells line up on y = x ⇒ Gaussian matches score; deviation
    identifies which cells / diagnostics need the score model."""
    fields = ["A_r", "C_r", "AS_r"]
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4.5))
    cmap = {"rope": "#1f77b4", "alibi": "#ff7f0e", "absolute": "#2ca02c"}
    layer_marker = {1: "o", 2: "s", 3: "^", 4: "D"}

    for ax, field in zip(axes, fields):
        xs, ys, cs, ms = [], [], [], []
        for k, c in score_metrics.items():
            if k not in gauss_metrics:
                continue
            pe = c.get("pe_type"); L = int(c.get("layer", 0))
            v_s = c["lags"][str(lag)].get(field)
            v_g = gauss_metrics[k]["lags"][str(lag)].get(field)
            if v_s is None or v_g is None:
                continue
            xs.append(v_s); ys.append(v_g)
            cs.append(cmap.get(pe, "k"))
            ms.append(layer_marker.get(L, "o"))
        for x, y, color, marker in zip(xs, ys, cs, ms):
            ax.scatter(x, y, c=color, marker=marker, s=70, alpha=0.8, edgecolor="k", linewidths=0.5)
        if xs:
            lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=0.6, alpha=0.5)
        ax.set_xlabel(f"{field}^score (lag {lag})")
        ax.set_ylabel(f"{field}^Gaussian (lag {lag})")
        ax.set_title(field)
        ax.grid(True, alpha=0.3)

    # Legend (shared)
    handles = []
    for pe, c in cmap.items():
        handles.append(plt.Line2D([], [], marker="o", color=c, lw=0, label=pe, markersize=8))
    for L, mk in layer_marker.items():
        handles.append(plt.Line2D([], [], marker=mk, color="gray", lw=0, label=f"L{L}", markersize=8))
    axes[-1].legend(handles=handles, fontsize=8, loc="best", ncol=2)

    fig.suptitle("Per-cell scalar agreement: score vs Gaussian (cells on y=x ⇒ Gaussian matches)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score-metrics",    type=str, required=True)
    p.add_argument("--gaussian-metrics", type=str, required=True)
    p.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    p.add_argument("--layers",   nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--max-lag",  type=int, default=14)
    p.add_argument("--out-dir",  type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    score_metrics = json.loads(Path(args.score_metrics).read_text())
    gauss_metrics = json.loads(Path(args.gaussian_metrics).read_text())

    score_agg = aggregate_cells_by_pe_layer(score_metrics, lag=1)
    gauss_agg = aggregate_cells_by_pe_layer(gauss_metrics, lag=1)

    md = render_markdown_table(score_agg, gauss_agg, args.pe_types, args.layers)
    (out_dir / "comparison_table.md").write_text(md)
    tex = render_latex_table(score_agg, gauss_agg, args.pe_types, args.layers)
    (out_dir / "comparison_table.tex").write_text(tex)

    fig_envelope_overlay(
        score_metrics, gauss_metrics, args.pe_types, layer=4,
        max_lag=args.max_lag,
        out_path=out_dir / "F_envelope_overlay_L4.pdf",
    )
    fig_scalar_scatter(
        score_metrics, gauss_metrics, max_lag=args.max_lag,
        out_path=out_dir / "F_scalar_scatter_lag1.pdf",
    )
    print(f"Wrote: {out_dir / 'comparison_table.md'}")
    print(f"Wrote: {out_dir / 'comparison_table.tex'}")
    print(f"Wrote: {out_dir / 'F_envelope_overlay_L4.pdf'}")
    print(f"Wrote: {out_dir / 'F_scalar_scatter_lag1.pdf'}")

    # Headline assessment: does Gaussian C_r at L4 reproduce the
    # ALiBi-vs-others ordering?
    print()
    print("=" * 72)
    print("Headline: C_r at L4 — does Gaussian recover the rank-one ALiBi signature?")
    print("=" * 72)
    print(f"  {'PE':<10}  {'C_r^score (L4)':>17}  {'C_r^Gauss (L4)':>17}")
    for pe in args.pe_types:
        s = score_agg.get((pe, 4), {}).get("C_r")
        g = gauss_agg.get((pe, 4), {}).get("C_r")
        if s and g:
            print(f"  {pe:<10}  {s[0]:>+10.3f} ± {s[1]:.3f}  "
                  f"{g[0]:>+10.3f} ± {g[1]:.3f}")
    print()
    print("If C_r^Gauss[ALiBi] >> C_r^Gauss[RoPE], C_r^Gauss[Absolute],")
    print("the Gaussian fingerprint reproduces the rank-one ALiBi signature.")
    print("If not, the score model is reading non-Gaussian structure that")
    print("Σ⁻¹ misses (Outcome B; strongest motivation for the score model).")


if __name__ == "__main__":
    main()
