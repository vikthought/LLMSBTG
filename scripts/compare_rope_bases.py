"""
Cross-base aggregation for the RoPE base sweep.

Reads the per-base lagpair_metrics.json and logit_analysis/*.json produced by
cluster_rope_base_sweep.sh, then:
  * aggregates SBTG metrics (SI, A_r, S_r, C_r, AS_r, operator autocorrelation)
    per (base, layer) across seeds,
  * reads training-history validation losses per task family per seed,
  * reads logit-space metrics (RDI, A_r, concentration) per base across seeds,
  * produces figures and a single rope_base_summary.json consolidating the
    cross-base comparison.

Theoretical RoPE envelope (Su et al. 2024, eq. 15):
    ρ(r, base) = (1/K) · Σ_{k=0}^{K-1} cos(r · base^(-2k/d_head))
with K = d_head / 2.  Overlaid on the empirical AS_r curves to test the
paper's claim that empirical envelopes track ρ(r) in extremum locations.

Usage:
    python scripts/compare_rope_bases.py \\
        --sweep-root results/rope_base_sweep \\
        --bases 10 100 1000 10000 100000 1000000 \\
        --seeds 0 1 2 \\
        --out-dir results/rope_base_sweep/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------

def rope_theoretical_envelope(r_values: np.ndarray, base: float, d_head: int = 64) -> np.ndarray:
    """ρ(r) = (1/K) · Σ cos(r · base^(-2k/d_head)) for k = 0..K-1, K = d_head/2."""
    K = d_head // 2
    ks = np.arange(K)
    thetas = base ** (-2.0 * ks / d_head)
    r = np.asarray(r_values, dtype=np.float64)
    cos_mat = np.cos(r[:, None] * thetas[None, :])
    return cos_mat.mean(axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_lagpair(base_dir: Path, seeds: List[int]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Return {(seed, layer): lag-1 metrics + SI + full AS_r profile}."""
    p = base_dir / "lagpair_analysis" / "lagpair_metrics.json"
    if not p.exists():
        print(f"  WARN: missing {p}")
        return {}
    with open(p) as f:
        d = json.load(f)
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for seed in seeds:
        for L in (1, 2, 3, 4):
            tag = f"rope_s{seed}_L{L}"
            if tag not in d:
                continue
            lag1 = d[tag]["lags"].get("1", {})
            # Full AS_r profile across lags 0..max_lag
            as_r = []
            a_r_all = []
            max_lag = d[tag].get("max_lag", 14)
            for r in range(max_lag + 1):
                e = d[tag]["lags"].get(str(r), {})
                as_r.append(e.get("AS_r", np.nan))
                a_r_all.append(e.get("A_r", np.nan))
            out[(seed, L)] = {
                "SI":   d[tag]["SI"],
                "A_r":  lag1.get("A_r"),
                "S_r":  lag1.get("S_r"),
                "C_r":  lag1.get("C_r"),
                "AS_r": lag1.get("AS_r"),
                "AS_r_profile":      as_r,
                "A_r_profile":       a_r_all,
                "autocorr_lag1":     lag1.get("autocorr", []),
                "singular_values":   lag1.get("singular_values", []),
                "endpoint_eigvals":  lag1.get("endpoint_eigvals", []),
            }
    return out


def _load_logit(base_dir: Path, seeds: List[int]) -> Dict[int, Dict[str, Any]]:
    p_dir = base_dir / "logit_analysis"
    out: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        p = p_dir / f"rope_seed{seed}_logit_analysis.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        out[seed] = {
            "RDI":            d.get("RDI"),
            "A_r":            d.get("A_r", []),
            "H_r":            d.get("H_r", []),
            "beta":           d.get("beta"),
            "SCR_r":          d.get("SCR_r", []),
            "probe_accuracy": d.get("probe_accuracy"),
            "val_dsm_loss":   d.get("val_dsm_loss"),
            "best_hp":        d.get("best_hp", {}),
        }
    return out


def _load_training_history(base_dir: Path, seeds: List[int]) -> Dict[int, Dict[str, float]]:
    """Last-epoch val loss per task family per seed from training_history.json."""
    out: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        p = base_dir / f"rope_seed{seed}" / "training_history.json"
        if not p.exists():
            continue
        with open(p) as f:
            h = json.load(f)
        # val_loss is {family: [loss_per_epoch]} — take last epoch
        val = h.get("val_loss", {})
        if not val:
            continue
        out[seed] = {fam: (losses[-1] if losses else float("nan")) for fam, losses in val.items()}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BASE_CMAP = plt.get_cmap("viridis")


def _base_color(base: int, bases: List[int]) -> tuple:
    i = bases.index(base)
    return BASE_CMAP(i / max(1, len(bases) - 1))


def fig_val_loss_vs_base(
    per_base_training: Dict[int, Dict[int, Dict[str, float]]],
    bases: List[int],
    out_dir: Path,
) -> None:
    """One panel per task family; x=log(base), y=val loss mean±std across seeds."""
    # Gather families
    all_families: set = set()
    for tb in per_base_training.values():
        for sd in tb.values():
            all_families.update(sd.keys())
    fams = sorted(all_families - {"iid_random"})
    if not fams:
        print("  WARN: no training_history val_loss — skipping fig_val_loss_vs_base")
        return

    n_cols = min(4, len(fams))
    n_rows = (len(fams) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)
    axes_flat = [a for row in axes for a in row]

    for ax, fam in zip(axes_flat, fams):
        means, stds = [], []
        for base in bases:
            per_seed = [s.get(fam, np.nan) for s in per_base_training.get(base, {}).values()]
            per_seed = np.array([v for v in per_seed if not np.isnan(v)])
            means.append(per_seed.mean() if per_seed.size else np.nan)
            stds.append(per_seed.std() if per_seed.size else 0.0)
        means = np.array(means)
        stds  = np.array(stds)
        ax.errorbar(bases, means, yerr=stds, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("RoPE base θ")
        ax.set_ylabel("val cross-entropy")
        ax.set_title(fam)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(fams):]:
        ax.set_visible(False)

    fig.suptitle("Per-family validation loss vs. RoPE base (3 seeds, mean ± std)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "RB1_val_loss_vs_base.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sbtg_scalars_vs_base(
    per_base_lagpair: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]],
    bases: List[int],
    out_dir: Path,
) -> None:
    """2×4 grid: SI, A_r, C_r, AS_r vs log(base), one line per layer."""
    metrics = ["SI", "A_r", "C_r", "AS_r"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, metrics):
        for L in (1, 2, 3, 4):
            means, stds = [], []
            for base in bases:
                vals = [
                    per_base_lagpair.get(base, {}).get((s, L), {}).get(metric)
                    for s in range(10)
                ]
                vals = np.array([v for v in vals if v is not None], dtype=float)
                means.append(vals.mean() if vals.size else np.nan)
                stds.append(vals.std() if vals.size else 0.0)
            means, stds = np.array(means), np.array(stds)
            ax.errorbar(bases, means, yerr=stds, marker="o", capsize=3, label=f"L{L}")
        ax.set_xscale("log")
        ax.set_xlabel("RoPE base θ")
        ax.set_ylabel(metric + ("(r=1)" if metric != "SI" else ""))
        ax.set_title(f"{metric} vs base")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SBTG scalars vs RoPE base (lag 1, mean ± std across 3 seeds)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "RB2_sbtg_scalars_vs_base.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_as_r_envelope_overlay(
    per_base_lagpair: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]],
    bases: List[int],
    out_dir: Path,
    d_head: int = 64,
) -> None:
    """
    Empirical AS_r(r)/AS_r(1) at L4 for each base overlaid with theoretical
    ρ(r, base)/ρ(1, base).  Tests the paper's prediction that the envelope
    tracks ρ(r) in extremum locations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # Left: empirical
    ax = axes[0]
    for base in bases:
        as_lists = []
        for seed in range(10):
            entry = per_base_lagpair.get(base, {}).get((seed, 4))
            if entry is None:
                continue
            prof = np.array(entry["AS_r_profile"], dtype=float)
            if prof.size < 2 or not np.isfinite(prof[1]) or prof[1] <= 0:
                continue
            as_lists.append(prof / prof[1])   # normalize by AS_r(1)
        if not as_lists:
            continue
        arr = np.array(as_lists)                     # (n_seeds, max_lag+1)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        r_vals = np.arange(mean.size)
        color = _base_color(base, bases)
        ax.plot(r_vals, mean, color=color, marker="o", ms=4, label=f"base={base:g}")
        ax.fill_between(r_vals, mean - std, mean + std, color=color, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lag r")
    ax.set_ylabel(r"$\mathrm{AS}_r(r)/\mathrm{AS}_r(1)$")
    ax.set_title("Empirical AS_r envelope at L4 (3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: theoretical ρ(r)/ρ(1)
    ax = axes[1]
    r_vals = np.arange(15, dtype=float)
    for base in bases:
        rho = rope_theoretical_envelope(r_vals, base, d_head=d_head)
        if abs(rho[1]) > 1e-9:
            rho_norm = rho / rho[1]
        else:
            rho_norm = rho
        ax.plot(r_vals, rho_norm, color=_base_color(base, bases), marker="s", ms=4,
                label=f"base={base:g}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lag r")
    ax.set_ylabel(r"$\rho(r)/\rho(1)$")
    ax.set_title(f"Theoretical ρ(r) (d_head={d_head})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "RoPE envelope across bases — empirical at L4 vs. theory.\n"
        "θ_0 = 1 for every base, so the dip near r≈6 is base-invariant; "
        "what changes with base is the low-frequency tail.",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "RB3_as_r_envelope_overlay.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_dip_depth_vs_base(
    per_base_lagpair: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]],
    bases: List[int],
    out_dir: Path,
) -> None:
    """
    Depth of the r=6-7 dip at L4: 1 - min(AS_r(6..7)) / AS_r(1).  Larger
    depth means sharper rank-2 dip.  Tests prediction #1 from the sweep.
    """
    means, stds = [], []
    for base in bases:
        depths = []
        for seed in range(10):
            entry = per_base_lagpair.get(base, {}).get((seed, 4))
            if entry is None:
                continue
            p = np.array(entry["AS_r_profile"], dtype=float)
            if p.size < 8 or not np.isfinite(p[1]) or p[1] <= 0:
                continue
            dip_min = np.nanmin(p[6:8])
            depths.append(1.0 - dip_min / p[1])
        depths = np.array(depths)
        means.append(depths.mean() if depths.size else np.nan)
        stds.append(depths.std() if depths.size else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(bases, means, yerr=stds, marker="o", capsize=3, color="darkblue")
    ax.set_xscale("log")
    ax.set_xlabel("RoPE base θ")
    ax.set_ylabel(r"dip depth at $r\in\{6,7\}$ (relative to $r{=}1$)")
    ax.set_title("Depth of the RoPE period-2π dip at L4 vs base")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "RB4_dip_depth_vs_base.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_logit_vs_base(
    per_base_logit: Dict[int, Dict[int, Dict[str, Any]]],
    bases: List[int],
    out_dir: Path,
) -> None:
    """Logit RDI and probe accuracy vs base.  Tests prediction #3."""
    rdi_m, rdi_s, probe_m, probe_s = [], [], [], []
    for base in bases:
        rdis, probes = [], []
        for s, v in per_base_logit.get(base, {}).items():
            if v.get("RDI") is not None:
                rdis.append(v["RDI"])
            if v.get("probe_accuracy") is not None:
                probes.append(v["probe_accuracy"])
        rdis = np.array(rdis)
        probes = np.array(probes)
        rdi_m.append(rdis.mean() if rdis.size else np.nan)
        rdi_s.append(rdis.std() if rdis.size else 0.0)
        probe_m.append(probes.mean() if probes.size else np.nan)
        probe_s.append(probes.std() if probes.size else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(bases, rdi_m, yerr=rdi_s, marker="o", capsize=3, color="teal")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("RoPE base θ")
    axes[0].set_ylabel("Logit RDI")
    axes[0].set_title("Logit RDI vs base (prediction: unimodal, peak near 10^4)")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(bases, probe_m, yerr=probe_s, marker="s", capsize=3, color="crimson")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("RoPE base θ")
    axes[1].set_ylabel("Logit linear-probe accuracy")
    axes[1].set_title("Logit position probe vs base")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "RB5_logit_metrics_vs_base.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--bases", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--d-head", type=int, default=64, help="head dim for theoretical envelope")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir    = Path(args.out_dir)
    fig_dir    = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_base_lagpair: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    per_base_logit:   Dict[int, Dict[int, Dict[str, Any]]] = {}
    per_base_training: Dict[int, Dict[int, Dict[str, float]]] = {}

    for base in args.bases:
        base_dir = sweep_root / f"base_{base}"
        per_base_lagpair[base]  = _load_lagpair(base_dir, args.seeds)
        per_base_logit[base]    = _load_logit(base_dir, args.seeds)
        per_base_training[base] = _load_training_history(base_dir, args.seeds)
        n_lag = len({k[0] for k in per_base_lagpair[base].keys()})
        print(f"  base={base}: lagpair seeds={n_lag}, logit seeds={len(per_base_logit[base])}, "
              f"training seeds={len(per_base_training[base])}")

    # Generate figures
    print("\nGenerating figures...")
    fig_val_loss_vs_base(per_base_training, args.bases, fig_dir)
    fig_sbtg_scalars_vs_base(per_base_lagpair, args.bases, fig_dir)
    fig_as_r_envelope_overlay(per_base_lagpair, args.bases, fig_dir, d_head=args.d_head)
    fig_dip_depth_vs_base(per_base_lagpair, args.bases, fig_dir)
    fig_logit_vs_base(per_base_logit, args.bases, fig_dir)

    # Summary JSON
    summary: Dict[str, Any] = {
        "bases": args.bases,
        "seeds": args.seeds,
        "d_head": args.d_head,
        "per_base": {},
    }
    for base in args.bases:
        bd: Dict[str, Any] = {
            "lagpair_by_layer": {},
            "logit": {},
            "training_val_loss": {},
        }
        for L in (1, 2, 3, 4):
            for metric in ("SI", "A_r", "S_r", "C_r", "AS_r"):
                vals = [
                    per_base_lagpair.get(base, {}).get((s, L), {}).get(metric)
                    for s in args.seeds
                ]
                vals = np.array([v for v in vals if v is not None], dtype=float)
                bd["lagpair_by_layer"].setdefault(f"L{L}", {})[metric] = {
                    "mean": float(np.nanmean(vals)) if vals.size else None,
                    "std":  float(np.nanstd(vals))  if vals.size else None,
                    "n":    int(vals.size),
                }
        logit_vals = per_base_logit.get(base, {})
        for metric in ("RDI", "beta", "probe_accuracy", "val_dsm_loss"):
            vs = np.array([v.get(metric) for v in logit_vals.values() if v.get(metric) is not None],
                          dtype=float)
            bd["logit"][metric] = {
                "mean": float(vs.mean()) if vs.size else None,
                "std":  float(vs.std())  if vs.size else None,
                "n":    int(vs.size),
            }
        train_vals = per_base_training.get(base, {})
        if train_vals:
            all_fams = {f for s in train_vals.values() for f in s.keys()}
            for fam in all_fams:
                vs = np.array([s.get(fam, np.nan) for s in train_vals.values()], dtype=float)
                vs = vs[~np.isnan(vs)]
                bd["training_val_loss"][fam] = {
                    "mean": float(vs.mean()) if vs.size else None,
                    "std":  float(vs.std())  if vs.size else None,
                    "n":    int(vs.size),
                }
        summary["per_base"][str(base)] = bd

    out_path = out_dir / "rope_base_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Figures in {fig_dir}")


if __name__ == "__main__":
    main()
