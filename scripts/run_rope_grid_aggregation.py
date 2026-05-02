"""
RoPE base × context grid aggregator.

Reads the per-(base, context, seed) outputs from
results/rope_base_context_grid/ctx{C}/base_{B}/{rope_seed{S}, lagpair}/ and
produces:

    rope_grid_summary.json         consolidated per-cell summary
    F_val_loss_heatmap.pdf         2-D heatmap of val_loss on
                                   variable_lag_copy across (base, context)
    F_val_loss_u_curves.pdf        per-context U-curves overlaid
    F_as_r_envelope_grid.pdf       AS_r envelope at the deepest layer for
                                   each (base, context) cell, side by side

The headline figure is the U-curve overlay: if the optimum shifts toward
larger base as context grows, RoPE's base × context interaction is
visible in val loss alone, and the practical-advice claim
("min viable base ≈ context length") is supported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Theoretical envelope
# ============================================================================

def rho_theoretical(r: int, base: float, d_head: int) -> float:
    """ρ(r) = (1/K) Σ_k cos(r · base^(-2k/d_head)) for k = 0..K-1, K = d_head/2."""
    K = d_head // 2
    ks = np.arange(K)
    theta = base ** (-2.0 * ks / d_head)
    return float(np.mean(np.cos(r * theta)))


# ============================================================================
# Per-cell readers
# ============================================================================

def read_training_history(seed_dir: Path) -> Optional[Dict]:
    p = seed_dir / "training_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def collect_val_losses(
    base_dir: Path, seeds: List[int]
) -> Dict[str, Dict[str, float]]:
    """Per-family val loss at the last epoch, mean ± std across seeds."""
    families_seen = {}
    for s in seeds:
        seed_dir = base_dir / f"rope_seed{s}"
        h = read_training_history(seed_dir)
        if h is None:
            continue
        epochs = h.get("epochs", [])
        if not epochs:
            continue
        last = epochs[-1]
        for fam, v in last.get("val_losses", {}).items():
            families_seen.setdefault(fam, []).append(v)
    out = {}
    for fam, vs in families_seen.items():
        out[fam] = {
            "mean": float(np.mean(vs)),
            "std":  float(np.std(vs)),
            "n":    len(vs),
        }
    return out


def collect_effective_bandwidth(base_dir: Path) -> Optional[Dict]:
    """Read effective_bandwidth.json if present; return summary stats."""
    p = base_dir / "effective_bandwidth" / "effective_bandwidth.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("n_seeds", 0) == 0:
        return None
    return {
        "K":           int(d.get("K", 0)),
        "k_eff_mean":  float(d["k_eff_mean"]),
        "k_eff_std":   float(d["k_eff_std"]),
        "L_full_mean": float(d["L_full_mean"]),
        "n_seeds":     int(d["n_seeds"]),
    }


def collect_lagpair_envelope(
    lp_dir: Path,
    seeds: List[int],
    deepest_layer: int,
    max_lag: int,
) -> Dict[str, List[float]]:
    """AS_r/AS_1 envelope at the deepest layer, mean ± std across seeds."""
    p = lp_dir / "lagpair_metrics.json"
    if not p.exists():
        return {"r": [], "mean": [], "std": []}
    d = json.loads(p.read_text())

    rs = list(range(0, max_lag + 1))
    by_seed_envelope = []
    for s in seeds:
        cell = d.get(f"rope_s{s}_L{deepest_layer}")
        if cell is None:
            continue
        a1 = cell["lags"].get("1", {}).get("AS_r")
        if a1 is None or a1 == 0:
            continue
        env = []
        for r in rs:
            ar = cell["lags"].get(str(r), {}).get("AS_r")
            env.append(ar / a1 if ar is not None else np.nan)
        by_seed_envelope.append(env)
    if not by_seed_envelope:
        return {"r": rs, "mean": [], "std": []}
    arr = np.array(by_seed_envelope, dtype=float)
    return {
        "r":    rs,
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std":  np.nanstd(arr,  axis=0).tolist(),
        "n_seeds": int(arr.shape[0]),
    }


# ============================================================================
# Figures
# ============================================================================

def fig_val_loss_heatmap(
    summary: Dict,
    bases: List[int],
    contexts: List[int],
    family: str,
    out_path: Path,
):
    """Heatmap: columns = bases, rows = contexts, cells = mean val loss."""
    H = np.full((len(contexts), len(bases)), np.nan)
    for i, ctx in enumerate(contexts):
        for j, base in enumerate(bases):
            cell = summary["per_cell"].get(f"ctx{ctx}_base{base}", {})
            v = cell.get("val_loss", {}).get(family, {}).get("mean")
            if v is not None:
                H[i, j] = v
    fig, ax = plt.subplots(figsize=(1.4 * len(bases) + 2, 1.2 * len(contexts) + 2))
    im = ax.imshow(H, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([f"{b:g}" for b in bases])
    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels([str(c) for c in contexts])
    ax.set_xlabel("RoPE base")
    ax.set_ylabel("context length")
    ax.set_title(f"val loss on {family} (lower = better)")
    for i in range(len(contexts)):
        for j in range(len(bases)):
            v = H[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > np.nanmean(H) else "black",
                        fontsize=9)
    plt.colorbar(im, ax=ax, label="val loss (nats)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_val_loss_u_curves(
    summary: Dict,
    bases: List[int],
    contexts: List[int],
    family: str,
    out_path: Path,
):
    """U-curves: x = log(base), y = val loss; one line per context."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, ctx in enumerate(contexts):
        ys, errs = [], []
        for base in bases:
            cell = summary["per_cell"].get(f"ctx{ctx}_base{base}", {})
            entry = cell.get("val_loss", {}).get(family, {})
            ys.append(entry.get("mean", np.nan))
            errs.append(entry.get("std", 0.0))
        color = cmap(i / max(len(contexts) - 1, 1))
        ax.errorbar(bases, ys, yerr=errs, marker="o", color=color, capsize=3,
                    linewidth=2, label=f"ctx={ctx}")
    ax.set_xscale("log")
    ax.set_xlabel("RoPE base")
    ax.set_ylabel(f"val loss on {family}")
    ax.set_title("Val-loss U-curve vs RoPE base, by context")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_effective_bandwidth_heatmap(
    summary: Dict,
    bases: List[int],
    contexts: List[int],
    out_path: Path,
):
    """Heatmap of k_eff across (base, context).  Annotated with k_eff mean
    and the K=d_head/2 cap for reference."""
    K_max = None
    H_keff = np.full((len(contexts), len(bases)), np.nan)
    for i, ctx in enumerate(contexts):
        for j, base in enumerate(bases):
            cell = summary["per_cell"].get(f"ctx{ctx}_base{base}", {})
            bw = cell.get("effective_bandwidth")
            if bw is not None:
                H_keff[i, j] = bw["k_eff_mean"]
                K_max = bw["K"]
    fig, ax = plt.subplots(figsize=(1.4 * len(bases) + 2, 1.2 * len(contexts) + 2))
    im = ax.imshow(H_keff, aspect="auto", cmap="plasma", origin="lower",
                   vmin=0, vmax=K_max if K_max else None)
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([f"{b:g}" for b in bases])
    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels([str(c) for c in contexts])
    ax.set_xlabel("RoPE base (training)")
    ax.set_ylabel("context length")
    cap = f" (K={K_max})" if K_max else ""
    ax.set_title(f"Effective bandwidth $k_{{\\rm eff}}${cap}")
    for i in range(len(contexts)):
        for j in range(len(bases)):
            v = H_keff[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < (K_max or 32) * 0.5 else "black",
                        fontsize=10)
    plt.colorbar(im, ax=ax, label=r"$k_{\rm eff}$ (top-frequencies needed)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_as_r_envelope_grid(
    summary: Dict,
    bases: List[int],
    contexts: List[int],
    out_path: Path,
    d_head: int,
    deepest_layer: int,
):
    """Empirical AS_r/AS_1 envelope at L_max for each (base, context).
    Theoretical ρ(r)/ρ(1) overlaid as dashed lines for reference."""
    n_ctx = len(contexts)
    fig, axes = plt.subplots(1, n_ctx, figsize=(5 * n_ctx, 4.5),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax_i, ctx in enumerate(contexts):
        ax = axes[0, ax_i]
        for j, base in enumerate(bases):
            cell = summary["per_cell"].get(f"ctx{ctx}_base{base}", {})
            env = cell.get("envelope_L%d" % deepest_layer, {})
            rs = env.get("r", [])
            mean = env.get("mean", [])
            std = env.get("std", [])
            if not rs or not mean:
                continue
            color = cmap(j / max(len(bases) - 1, 1))
            ax.plot(rs, mean, marker="o", color=color, linewidth=2,
                    label=f"base={base:g}")
            if std:
                m = np.array(mean); s = np.array(std)
                ax.fill_between(rs, m - s, m + s, color=color, alpha=0.15)

            # theoretical overlay (dashed)
            theory = np.array([rho_theoretical(r, base, d_head) for r in rs])
            theory_norm = theory / theory[1] if abs(theory[1]) > 1e-12 else theory
            ax.plot(rs, theory_norm, linestyle="--", color=color, linewidth=1, alpha=0.6)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("lag r")
        if ax_i == 0:
            ax.set_ylabel(r"$AS_r(r) / AS_r(1)$")
        ax.set_title(f"ctx = {ctx}  (L{deepest_layer})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Empirical AS_r envelope (solid) vs theoretical ρ(r) (dashed), per context",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--bases", nargs="+", type=int, required=True)
    p.add_argument("--contexts", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--d-head", type=int, default=64,
                   help="Head dim used for the theoretical ρ(r) overlay")
    p.add_argument("--deepest-layer", type=int, default=2,
                   help="Layer index to use for the AS_r envelope figure "
                        "(matches the SBTG pipeline's 1-indexed layer).")
    p.add_argument("--max-lag", type=int, default=18)
    p.add_argument("--family", type=str, default="variable_lag_copy",
                   help="Task family for the val-loss heatmap and U-curves.")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-cell summary
    per_cell = {}
    for ctx in args.contexts:
        for base in args.bases:
            base_dir = sweep_root / f"ctx{ctx}/base_{base}"
            lp_dir = base_dir / "lagpair"
            cell_key = f"ctx{ctx}_base{base}"
            cell = {
                "ctx":  int(ctx),
                "base": int(base),
                "val_loss": collect_val_losses(base_dir, args.seeds),
                "effective_bandwidth": collect_effective_bandwidth(base_dir),
            }
            for L in range(1, args.deepest_layer + 1):
                cell[f"envelope_L{L}"] = collect_lagpair_envelope(
                    lp_dir, args.seeds, deepest_layer=L, max_lag=args.max_lag,
                )
            per_cell[cell_key] = cell

    summary = {
        "config": {
            "bases":    args.bases,
            "contexts": args.contexts,
            "seeds":    args.seeds,
            "d_head":   args.d_head,
            "deepest_layer": args.deepest_layer,
            "family":   args.family,
        },
        "per_cell": per_cell,
    }
    (out_dir / "rope_grid_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Wrote {out_dir / 'rope_grid_summary.json'}")

    # Figures
    fig_val_loss_heatmap(
        summary, args.bases, args.contexts, args.family,
        out_dir / "F_val_loss_heatmap.pdf",
    )
    fig_val_loss_u_curves(
        summary, args.bases, args.contexts, args.family,
        out_dir / "F_val_loss_u_curves.pdf",
    )
    fig_as_r_envelope_grid(
        summary, args.bases, args.contexts,
        out_dir / "F_as_r_envelope_grid.pdf",
        d_head=args.d_head,
        deepest_layer=args.deepest_layer,
    )
    fig_effective_bandwidth_heatmap(
        summary, args.bases, args.contexts,
        out_dir / "F_effective_bandwidth.pdf",
    )
    print(f"Wrote 4 figures to {out_dir}/")

    # Headline numbers
    print()
    print("=" * 70)
    print("Headline 1: val loss on", args.family, "across (base, context)")
    print("=" * 70)
    print(f"  {'base':>10}  | " + " | ".join(f"ctx={c:>4}" for c in args.contexts))
    for base in args.bases:
        line = f"  {base:>10}  | "
        for ctx in args.contexts:
            v = per_cell[f"ctx{ctx}_base{base}"]["val_loss"].get(args.family, {}).get("mean")
            line += (f"{v:>9.4f}" if v is not None else f"{'—':>9}") + "  | "
        print(line)

    print()
    print("=" * 70)
    print("Headline 2: effective bandwidth k_eff (top-frequencies needed)")
    print("=" * 70)
    print(f"  {'base':>10}  | " + " | ".join(f"ctx={c:>4}" for c in args.contexts))
    for base in args.bases:
        line = f"  {base:>10}  | "
        for ctx in args.contexts:
            bw = per_cell[f"ctx{ctx}_base{base}"].get("effective_bandwidth")
            if bw is None:
                line += f"{'—':>9}  | "
            else:
                line += f"{bw['k_eff_mean']:>5.1f}±{bw['k_eff_std']:.1f}  | "
        print(line)
    print()
    print(f"  K = d_head/2 = {args.d_head // 2}.  k_eff < K means the model")
    print(f"  doesn't use the lowest-frequency dimensions — bases that keep")
    print(f"  θ_{{k_eff}} in the active range are functionally equivalent.")


if __name__ == "__main__":
    main()
