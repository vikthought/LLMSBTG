"""
Parametric Gaussian baseline for the SBTG positional operator family.

Why this exists
---------------
Section 2.4 of the paper motivates the score function by pointing out that
for a multivariate Gaussian, cross-block products of the score recover
precision-matrix entries: if Z ~ N(μ, Σ) and Λ = Σ⁻¹, then

    s(Z) = -Λ(Z - μ),
    E[s_a s_b^T] = Λ_{a,b}  (block (a, b) of Λ).

The score-model machinery only earns its keep if the residual-stream window
distribution is non-Gaussian *in ways that affect the cross-block structure
SBTG reads*.  If a parametric Gaussian fit gives the same fingerprints, the
score model isn't doing useful work on this experiment.

This script runs SBTG in its parametric Gaussian form so the comparison is
direct and quantitative.

Pipeline
--------
For each (PE, seed, layer) cell:

  1. Load test activations from the standard MODELS_DIR layout.
  2. Load PCA components and per-position layer-mean from the per-seed
     analysis JSON produced by run_positional_analysis.py.  Project the
     test activations the same way the score-model pipeline does (m=32 by
     default).
  3. Slide a window of width w=16 across each sequence to get
     (N_test, n_windows, m*w) windows.
  4. For each endpoint i (window index): fit a multivariate Gaussian to the
     N_test windows at that endpoint.  Σ_i ∈ R^(mw × mw); regularize with a
     small ridge (λ = 1e-3 by default).  Invert to Λ_i.  Read

         M_r^Gauss(i) = -Λ_i[(w-1)*m : w*m,  (w-r-1)*m : (w-r)*m]

     for r = 0, ..., max_lag.  Stack across endpoints:
     M_r_i ∈ R^((max_lag+1), n_windows, m, m).

  5. Average over endpoints (with the same skip_edges = 4 used by the score
     pipeline) to get M_bar_r.

  6. Feed (M_r_i, M_bar_r) through compute_extended_metrics() — the same
     function run_lagpair_analysis.py uses — to get per-lag scalars
     (A_r, S_r, C_r, AS_r), exactly the same way as the score-based pipeline.

Output
------
<out-dir>/gaussian_lagpair_metrics.json with the same per-cell schema as
lagpair_metrics.json (key = `{pe}_s{seed}_L{layer}`), so
compare_score_vs_gaussian.py can ingest both with identical readers.

Cost
----
36 cells × ~50 endpoint Gaussians × (~50ms each on a 512-dim ridge invert)
≈ 90 seconds on CPU, plus activation loading.  Trivial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sbtg.estimators.localized_multiblock_trainer import extract_windows
# We reuse the exact post-processing helper the score pipeline uses, so the
# scalar diagnostics are computed identically.  (Imported from the entry
# script rather than the library because that's where compute_extended_metrics
# lives.)
import importlib.util
_lp_spec = importlib.util.spec_from_file_location(
    "_run_lagpair", str(Path(__file__).resolve().parent / "run_lagpair_analysis.py")
)
_run_lagpair = importlib.util.module_from_spec(_lp_spec)
_lp_spec.loader.exec_module(_run_lagpair)
compute_extended_metrics = _run_lagpair.compute_extended_metrics


# ============================================================================
# Helpers
# ============================================================================

def transform_pca(
    acts: np.ndarray,            # (N, T, hidden_dim)
    pca_components: np.ndarray,  # (m, hidden_dim) — sklearn PCA convention
    mu_l: np.ndarray,            # (hidden_dim,)
) -> np.ndarray:
    """Center by mu_l, project onto pca_components.  Returns (N, T, m)."""
    N, T, H = acts.shape
    centered = acts.reshape(-1, H) - mu_l[None, :]
    proj = centered @ pca_components.T   # (N*T, m)
    m = pca_components.shape[0]
    return proj.reshape(N, T, m)


def fit_gaussian_M_r_per_endpoint(
    windows: np.ndarray,    # (N, n_windows, mw)
    m: int,
    w: int,
    max_lag: int,
    ridge: float = 1e-3,
) -> np.ndarray:
    """For each window endpoint, fit a Gaussian and read M_r from -Λ blocks.

    Returns
    -------
    M_r_i : (max_lag + 1, n_windows, m, m).  The block structure of Λ:
        Λ is (mw, mw); block (a, b) is the (m × m) sub-matrix
        Λ[a*m:(a+1)*m, b*m:(b+1)*m].  M_r at endpoint i extracts
        block (w-1, w-r-1) of Λ_i and negates.
    """
    N, n_windows, mw = windows.shape
    assert mw == m * w, f"window dim {mw} != m*w = {m}*{w}"
    M_r_i = np.zeros((max_lag + 1, n_windows, m, m), dtype=np.float64)

    ridge_eye = ridge * np.eye(mw)
    curr_slice = slice((w - 1) * m, w * m)

    for i in range(n_windows):
        X = windows[:, i, :]                                    # (N, mw)
        mu = X.mean(axis=0)
        Xc = X - mu[None, :]
        Sigma = (Xc.T @ Xc) / max(N - 1, 1)                     # (mw, mw)
        Lambda = np.linalg.inv(Sigma + ridge_eye)               # (mw, mw)
        for r in range(max_lag + 1):
            src = w - r - 1
            if src < 0:
                continue
            src_slice = slice(src * m, (src + 1) * m)
            M_r_i[r, i] = -Lambda[curr_slice, src_slice]

    return M_r_i


# ============================================================================
# Per-cell driver
# ============================================================================

def process_cell(
    pe: str,
    seed: int,
    layer: int,
    models_dir: Path,
    analysis_dir: Path,
    w: int,
    max_lag: int,
    pca_dim: int,
    skip_edges: int,
    ridge: float,
) -> Dict:
    """Run the Gaussian baseline for one (PE, seed, layer) cell.  Returns a
    dict in the same schema as one entry of lagpair_metrics.json."""
    seed_dir = models_dir / f"{pe}_seed{seed}"
    test_acts_p = seed_dir / "test_acts.npy"
    analysis_p = analysis_dir / f"{pe}_seed{seed}_analysis.json"

    if not test_acts_p.exists():
        raise FileNotFoundError(f"missing test_acts at {test_acts_p}")
    if not analysis_p.exists():
        raise FileNotFoundError(f"missing analysis JSON at {analysis_p}")

    analysis = json.loads(analysis_p.read_text())
    layer_stats = analysis["layer_stats"]
    # layer_stats is list, indexed by (layer-1) since layer is 1-indexed
    cell_meta = layer_stats[layer - 1]
    pca_components = np.asarray(cell_meta["pca_components"], dtype=np.float64)  # (m, hidden_dim)
    mu_l = np.asarray(cell_meta["mu_l"], dtype=np.float64)                      # (hidden_dim,)
    if pca_components.shape[0] != pca_dim:
        raise ValueError(
            f"pca_dim mismatch: analysis JSON has m={pca_components.shape[0]}, "
            f"flag has {pca_dim}"
        )

    test_acts_full = np.load(test_acts_p, mmap_mode="r")
    test_acts_l = np.asarray(test_acts_full[:, layer], dtype=np.float64)        # (N, T, hidden)
    N_test, T, H = test_acts_l.shape

    # PCA project + window
    test_pca = transform_pca(test_acts_l, pca_components, mu_l)                 # (N, T, m)
    test_windows = extract_windows(test_pca, w)                                 # (N, n_windows, mw)
    n_windows = test_windows.shape[1]
    if n_windows <= 2 * skip_edges:
        # Very short context — fall back to no skip
        eff_skip = 0
    else:
        eff_skip = skip_edges

    # Fit Gaussian per endpoint, read M_r
    M_r_i = fit_gaussian_M_r_per_endpoint(
        test_windows, m=pca_dim, w=w, max_lag=max_lag, ridge=ridge
    )

    # Endpoint-averaged operator across the *valid* (non-edge) range — same
    # convention as the score pipeline.
    M_bar_r = M_r_i[:, eff_skip:n_windows - eff_skip].mean(axis=1) \
        if n_windows > 2 * eff_skip else M_r_i.mean(axis=1)

    # Per-lag scalars via the same helper the score pipeline uses
    extended = compute_extended_metrics(
        M_r_i=M_r_i,
        M_bar_r=M_bar_r,
        skip_edges=eff_skip,
        top_k=3,
        side="source",
    )

    # SI: fraction of M̄_r-mass at lag 0 vs total (matches lagpair_metrics.json
    # convention for Stationarity Index).  Compute identically here.
    A_r_per_lag = np.array([float(np.linalg.norm(M_bar_r[r], 'fro'))
                            for r in range(max_lag + 1)])
    SI = float(A_r_per_lag[0] ** 2 / (np.sum(A_r_per_lag ** 2) + 1e-12))

    # RDI = sum_{r >= 1} A_r^2 / sum_r A_r^2 — the lagged-mass fraction
    RDI = float(np.sum(A_r_per_lag[1:] ** 2) / (np.sum(A_r_per_lag ** 2) + 1e-12))

    # beta: log-log slope of A_r over r in [1, max_lag]
    rs = np.arange(1, max_lag + 1)
    Ar_pos = np.maximum(A_r_per_lag[1:], 1e-12)
    beta = float(np.polyfit(np.log(rs), np.log(Ar_pos), 1)[0])

    # Per-lag entries (subset of fields that lagpair_metrics.json has — we
    # don't have score-model val_dsm_loss etc.; everything that's a function
    # of M_r is computable here).
    lags_out = {}
    for r in range(max_lag + 1):
        e = extended[r]
        lags_out[str(r)] = {
            "A_r":  float(e["A_r"]),
            "S_r":  float(e["S_r"]),
            "C_r":  float(e["C_r"]),
            "AS_r": float(e["AS_r"]),
            "singular_values": e.get("singular_values", []),
        }

    return {
        "pe_type": pe,
        "seed":    int(seed),
        "layer":   int(layer),
        "pca_dim": int(pca_dim),
        "w":       int(w),
        "max_lag": int(max_lag),
        "skip_edges": int(eff_skip),
        "ridge":   float(ridge),
        "n_windows": int(n_windows),
        "N_test":  int(N_test),
        "SI":  SI,
        "RDI": RDI,
        "beta": beta,
        "A_r_original": A_r_per_lag.tolist(),
        "lags": lags_out,
        "estimator": "gaussian_precision",
    }


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir",   type=str, required=True,
                   help="Directory containing {pe}_seed{seed}/test_acts.npy")
    p.add_argument("--analysis-dir", type=str, required=True,
                   help="Directory containing {pe}_seed{seed}_analysis.json "
                        "(produced by run_positional_analysis.py — has "
                        "pca_components and mu_l per layer)")
    p.add_argument("--out-dir",      type=str, required=True)

    p.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--layers",   nargs="+", type=int, default=[1, 2, 3, 4])

    p.add_argument("--w",          type=int, default=16)
    p.add_argument("--max-lag",    type=int, default=14)
    p.add_argument("--pca-dim",    type=int, default=32)
    p.add_argument("--skip-edges", type=int, default=4)
    p.add_argument("--ridge",      type=float, default=1e-3,
                   help="Ridge regularization on Σ before inversion (Σ + λI). "
                        "Required because mw=512 and N_test=20000 → empirical "
                        "covariance is rank-deficient.")
    args = p.parse_args()

    models_dir   = Path(args.models_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Gaussian baseline for SBTG positional operator")
    print(f"   models:    {models_dir}")
    print(f"   analysis:  {analysis_dir}  (for PCA components + mu_l)")
    print(f"   out:       {out_dir}")
    print(f"   w={args.w}  max_lag={args.max_lag}  m={args.pca_dim}  "
          f"skip_edges={args.skip_edges}  ridge={args.ridge:g}")
    print("=" * 70)

    out = {}
    for pe in args.pe_types:
        for seed in args.seeds:
            for layer in args.layers:
                key = f"{pe}_s{seed}_L{layer}"
                print(f"\n  --- {key} ---")
                try:
                    cell = process_cell(
                        pe=pe, seed=seed, layer=layer,
                        models_dir=models_dir, analysis_dir=analysis_dir,
                        w=args.w, max_lag=args.max_lag, pca_dim=args.pca_dim,
                        skip_edges=args.skip_edges, ridge=args.ridge,
                    )
                except FileNotFoundError as e:
                    print(f"    SKIP — {e}")
                    continue
                out[key] = cell
                print(f"    SI={cell['SI']:.3f}  beta={cell['beta']:+.3f}  "
                      f"A_r(1)={cell['lags']['1']['A_r']:.4f}  "
                      f"C_r(1)={cell['lags']['1']['C_r']:.3f}")

    out_path = out_dir / "gaussian_lagpair_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}  ({len(out)} cells)")


if __name__ == "__main__":
    main()
