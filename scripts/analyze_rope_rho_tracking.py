"""
Numerical tracking of empirical SBTG envelope vs RoPE rotation kernel ρ(r).

Reviewer-requested follow-up to paper3.tex §4.2: the existing figure
(LP_AS_r_rope_overlay.pdf) shows the empirical AS_r/AS_1 envelope tracks
ρ(r)/ρ(1) in extremum locations, but reports no numerical correlation.
This script computes Pearson / Spearman / Kendall against the theoretical
kernel and against three nulls:

  monotone_decay     fit α·exp(−β r) to the empirical envelope, then correlate
                     the empirical with this fit (a "no-oscillation" null)
  shifted_kernel_δ   ρ(r − δ)/ρ(1 − δ) for δ ∈ {−2..+2} (the unshifted
                     version should maximize correlation if the empirical
                     tracks ρ in extrema location, not just sign)
  permuted           random permutation of the empirical envelope (a
                     significance null for the correlation magnitude)

CPU-only — runs locally on existing JSONs from:
  results/lagpair_analysis_<3,6>seed/lagpair_metrics.json
  results/rope_base_context_grid/analysis/rope_grid_summary.json
  optionally per-base lagpair JSONs from a RoPE-base-sweep run.

Usage
-----
python scripts/analyze_rope_rho_tracking.py \\
    --matched-metrics results/lagpair_analysis_3seed/lagpair_metrics.json \\
    --grid-summary    results/rope_base_context_grid/analysis/rope_grid_summary.json \\
    --out-dir         results/rope_rho_tracking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau


# ---------------------------------------------------------------------------
# Theoretical kernel (matches scripts/run_lagpair_analysis.py:219-223 and
# scripts/compare_rope_bases.py:44)
# ---------------------------------------------------------------------------

def rope_kernel(r_values: np.ndarray, base: float = 10_000.0, d_head: int = 64) -> np.ndarray:
    """ρ(r) = (1/K) Σ_k cos(r · θ_k),  θ_k = base^(-2k/d_head),  k = 0..K-1.

    Matches the indexing convention used in run_lagpair_analysis.py.
    """
    K = d_head // 2
    inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)  # (K,)
    return np.array([np.mean(np.cos(r * inv_freq)) for r in r_values])


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _three_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Pearson / Spearman / Kendall, returning NaN where input is degenerate."""
    out = {}
    for name, fn in (("pearson", pearsonr), ("spearman", spearmanr), ("kendall", kendalltau)):
        try:
            r, p = fn(x, y)
            out[f"{name}_r"] = float(r)
            out[f"{name}_p"] = float(p)
        except Exception:
            out[f"{name}_r"] = float("nan")
            out[f"{name}_p"] = float("nan")
    return out


def _fit_monotone_decay(r: np.ndarray, env: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """Fit env(r) ≈ α · exp(−β r), return (params, predicted)."""
    pos_mask = env > 0
    if pos_mask.sum() < 2:
        return {"alpha": float("nan"), "beta": float("nan"), "fit_ok": False}, np.full_like(env, float("nan"))
    try:
        popt, _ = curve_fit(
            lambda r, a, b: a * np.exp(-b * r),
            r[pos_mask].astype(float), env[pos_mask].astype(float),
            p0=[float(env[pos_mask][0]), 0.1], maxfev=2000,
        )
        a, b = popt
        pred = a * np.exp(-b * r)
        return {"alpha": float(a), "beta": float(b), "fit_ok": True}, pred
    except Exception:
        return {"alpha": float("nan"), "beta": float("nan"), "fit_ok": False}, np.full_like(env, float("nan"))


def _permutation_null(emp: np.ndarray, theory: np.ndarray, n: int = 1000,
                      seed: int = 0) -> Dict[str, float]:
    """Distribution of Pearson correlations under random permutation of emp."""
    rng = np.random.default_rng(seed)
    obs_r, _ = pearsonr(emp, theory)
    null_rs = []
    for _ in range(n):
        perm = rng.permutation(len(emp))
        try:
            r, _ = pearsonr(emp[perm], theory)
            null_rs.append(r)
        except Exception:
            null_rs.append(float("nan"))
    null_rs = np.array(null_rs)
    pct = float(np.mean(np.abs(null_rs) >= abs(obs_r)))
    return {
        "observed_r":     float(obs_r),
        "null_r_mean":    float(np.nanmean(null_rs)),
        "null_r_std":     float(np.nanstd(null_rs)),
        "null_r_q05":     float(np.nanquantile(null_rs, 0.05)),
        "null_r_q95":     float(np.nanquantile(null_rs, 0.95)),
        "two_sided_pvalue_perm": pct,
    }


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def _normalize_lag1(arr: np.ndarray) -> np.ndarray:
    """Divide by arr[0] (= lag 1 if arr[0] is the AS_1 entry)."""
    if arr[0] == 0 or not np.isfinite(arr[0]):
        return arr
    return arr / arr[0]


def analyze_envelope_vs_rho(
    empirical: np.ndarray,
    r_values: np.ndarray,
    base: float,
    d_head: int = 64,
    shifts: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    n_permutations: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """Run all the correlations + nulls for one cell.

    `empirical`: the empirical envelope at lags `r_values` (length L).
    `r_values`:  array of integer lags, e.g. [1, 2, ..., 14].

    Both should be normalized to lag-1 (== empirical[0] = 1.0) by the caller.
    """
    rho     = rope_kernel(r_values, base=base, d_head=d_head)
    rho_n   = _normalize_lag1(rho)
    out: Dict = {
        "r_values":      r_values.tolist(),
        "empirical":     empirical.tolist(),
        "theoretical":   rho_n.tolist(),
        "base":          float(base),
        "d_head":        int(d_head),
    }

    # 1. Direct correlation
    out["vs_rho"] = _three_correlations(empirical, rho_n)

    # 2. Permutation null on Pearson
    out["permutation_null"] = _permutation_null(
        np.asarray(empirical, dtype=float),
        np.asarray(rho_n, dtype=float),
        n=n_permutations, seed=rng_seed,
    )

    # 3. Monotone-decay null
    fit_params, pred = _fit_monotone_decay(r_values.astype(float), empirical.astype(float))
    out["monotone_decay_fit"] = {
        **fit_params,
        "predicted":  pred.tolist(),
        "vs_empirical": _three_correlations(empirical, pred),
    }

    # 4. Shifted-kernel scan
    out["shifted_kernel_correlations"] = {}
    for delta in shifts:
        r_shift = r_values - delta
        rho_shift = rope_kernel(r_shift.astype(float), base=base, d_head=d_head)
        if rho_shift[0] == 0 or not np.isfinite(rho_shift[0]):
            rho_shift_n = rho_shift
        else:
            rho_shift_n = rho_shift / rho_shift[0]
        out["shifted_kernel_correlations"][str(delta)] = {
            **_three_correlations(empirical, rho_shift_n),
            "rho_shift_n": rho_shift_n.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Loaders for the two JSON formats we consume
# ---------------------------------------------------------------------------

def _empirical_from_lagpair_cell(cell: Dict, max_lag: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Pull AS_r at lags 1..max_lag from a lagpair_metrics.json cell."""
    lag_dict = cell.get("lags", {})
    rs, vals = [], []
    for r in range(1, max_lag + 1):
        sub = lag_dict.get(str(r))
        if sub is None:
            continue
        v = sub.get("AS_r")
        if v is None:
            continue
        rs.append(r); vals.append(float(v))
    return np.array(rs), np.array(vals)


def _empirical_from_grid_envelope(env: Dict, max_lag: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (r, mean) from envelope_L<deepest>.  Format: dict with keys r/mean/std."""
    rs   = np.array(env.get("r", []))
    mean = np.array(env.get("mean", []))
    if rs.size == 0 or mean.size == 0:
        return np.array([]), np.array([])
    keep = (rs >= 1) & (rs <= max_lag)
    return rs[keep], mean[keep]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_matched_model(
    metrics_path: Path, out_dir: Path,
    base: float = 10_000.0, d_head: int = 64, max_lag: int = 14,
    n_permutations: int = 1000,
) -> Dict:
    with open(metrics_path) as f:
        metrics = json.load(f)

    rope_cells = {k: v for k, v in metrics.items() if k.startswith("rope_")}
    print(f"[matched] {len(rope_cells)} rope cells in {metrics_path}")

    per_cell_results = {}
    for cell_key, cell in rope_cells.items():
        r_values, env = _empirical_from_lagpair_cell(cell, max_lag=max_lag)
        if env.size < 4:
            continue
        env_n = _normalize_lag1(env)
        per_cell_results[cell_key] = analyze_envelope_vs_rho(
            empirical=env_n, r_values=r_values,
            base=base, d_head=d_head,
            n_permutations=n_permutations,
            rng_seed=hash(cell_key) & 0xFFFFFFFF,
        )

    # Aggregate per layer (across seeds)
    per_layer_summary: Dict[int, Dict] = {}
    for L in range(1, 5):
        cells_L = [v for k, v in per_cell_results.items() if k.endswith(f"_L{L}")]
        if not cells_L:
            continue
        pearson_rs   = np.array([c["vs_rho"]["pearson_r"]   for c in cells_L])
        spearman_rs  = np.array([c["vs_rho"]["spearman_r"]  for c in cells_L])
        kendall_rs   = np.array([c["vs_rho"]["kendall_r"]   for c in cells_L])
        mono_pearson = np.array([c["monotone_decay_fit"]["vs_empirical"]["pearson_r"]
                                 for c in cells_L])
        # Best shift per cell (the δ maximizing |Pearson|)
        best_shifts = []
        for c in cells_L:
            shifts = c["shifted_kernel_correlations"]
            best_d = max(shifts.keys(),
                         key=lambda d: abs(shifts[d]["pearson_r"]) if not np.isnan(shifts[d]["pearson_r"]) else -1)
            best_shifts.append(int(best_d))
        per_layer_summary[L] = {
            "n_cells":           len(cells_L),
            "pearson_mean":      float(np.nanmean(pearson_rs)),
            "pearson_std":       float(np.nanstd(pearson_rs)),
            "spearman_mean":     float(np.nanmean(spearman_rs)),
            "spearman_std":      float(np.nanstd(spearman_rs)),
            "kendall_mean":      float(np.nanmean(kendall_rs)),
            "kendall_std":       float(np.nanstd(kendall_rs)),
            "monotone_decay_pearson_mean": float(np.nanmean(mono_pearson)),
            "monotone_decay_pearson_std":  float(np.nanstd(mono_pearson)),
            "best_shift_mode":   int(np.bincount([s + 5 for s in best_shifts]).argmax() - 5)
                                  if best_shifts else None,
            "best_shift_distribution": {str(d): int(np.sum([s == d for s in best_shifts]))
                                         for d in (-2, -1, 0, 1, 2)},
        }

    out = {
        "metrics_path": str(metrics_path),
        "config":       {"base": base, "d_head": d_head, "max_lag": max_lag,
                         "n_permutations": n_permutations},
        "per_cell":     per_cell_results,
        "per_layer":    per_layer_summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matched_model_rho_tracking.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[matched] wrote {out_path}")
    return out


def run_grid(
    grid_summary_path: Path, out_dir: Path,
    d_head: int = 64, max_lag: int = 14, n_permutations: int = 1000,
) -> Dict:
    with open(grid_summary_path) as f:
        grid = json.load(f)
    cfg = grid.get("config", {})
    deepest = int(cfg.get("deepest_layer", 2))
    cells = grid.get("per_cell", {})
    print(f"[grid] {len(cells)} cells in {grid_summary_path}, deepest_layer={deepest}")

    per_cell_results = {}
    for cell_key, cell in cells.items():
        env_block = cell.get(f"envelope_L{deepest}")
        if env_block is None:
            continue
        r_values, env = _empirical_from_grid_envelope(env_block, max_lag=max_lag)
        if env.size < 4:
            continue
        env_n = _normalize_lag1(env)
        base = float(cell.get("base", 10_000.0))
        ctx  = int(cell.get("ctx", 0))
        per_cell_results[cell_key] = {
            "ctx":     ctx,
            "base":    base,
            **analyze_envelope_vs_rho(
                empirical=env_n, r_values=r_values,
                base=base, d_head=d_head,
                n_permutations=n_permutations,
                rng_seed=hash(cell_key) & 0xFFFFFFFF,
            ),
        }

    # Per-base / per-context summary
    by_base_ctx: Dict[Tuple[int, int], List[Dict]] = {}
    for cell_key, c in per_cell_results.items():
        by_base_ctx.setdefault((c["ctx"], int(c["base"])), []).append(c)
    summary_rows = []
    for (ctx, base), rows in sorted(by_base_ctx.items()):
        pearsons   = np.array([r["vs_rho"]["pearson_r"]   for r in rows])
        spearmans  = np.array([r["vs_rho"]["spearman_r"]  for r in rows])
        mono_pearsons = np.array([r["monotone_decay_fit"]["vs_empirical"]["pearson_r"]
                                   for r in rows])
        summary_rows.append({
            "ctx":  ctx, "base": base, "n_cells": len(rows),
            "pearson_mean":  float(np.nanmean(pearsons)),
            "pearson_std":   float(np.nanstd(pearsons)),
            "spearman_mean": float(np.nanmean(spearmans)),
            "spearman_std":  float(np.nanstd(spearmans)),
            "monotone_decay_pearson_mean": float(np.nanmean(mono_pearsons)),
        })

    out = {
        "grid_summary_path": str(grid_summary_path),
        "config":            {"d_head": d_head, "max_lag": max_lag,
                              "n_permutations": n_permutations,
                              "deepest_layer": deepest},
        "per_cell":          per_cell_results,
        "summary_by_base_ctx": summary_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_rho_tracking.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[grid] wrote {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matched-metrics", type=str,
                   default="results/lagpair_analysis_3seed/lagpair_metrics.json",
                   help="lagpair_metrics.json from the matched-model run.")
    p.add_argument("--grid-summary",    type=str,
                   default="results/rope_base_context_grid/analysis/rope_grid_summary.json",
                   help="rope_grid_summary.json from the RoPE base × context grid.")
    p.add_argument("--out-dir",         type=str,
                   default="results/rope_rho_tracking")
    p.add_argument("--matched-base",    type=float, default=10_000.0,
                   help="Base for the matched-model RoPE (default 10000).")
    p.add_argument("--d-head",          type=int,   default=64)
    p.add_argument("--max-lag",         type=int,   default=14)
    p.add_argument("--n-permutations",  type=int,   default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    matched_path = Path(args.matched_metrics)
    if matched_path.exists():
        run_matched_model(
            metrics_path=matched_path, out_dir=out_dir,
            base=args.matched_base, d_head=args.d_head, max_lag=args.max_lag,
            n_permutations=args.n_permutations,
        )
    else:
        print(f"[matched] SKIP — {matched_path} not found")

    grid_path = Path(args.grid_summary)
    if grid_path.exists():
        run_grid(
            grid_summary_path=grid_path, out_dir=out_dir,
            d_head=args.d_head, max_lag=args.max_lag,
            n_permutations=args.n_permutations,
        )
    else:
        print(f"[grid] SKIP — {grid_path} not found")


if __name__ == "__main__":
    main()
