"""
Statistical inference for SBTG edge discovery.

This module implements the full inference pipeline that converts learned
score functions into binary edge decisions (the functional connectome).

Mathematical overview
---------------------

1. **Mean transfer matrix**.  Given held-out score vectors
   s_past(t) in R^n and s_future(t) in R^n, define the (j, i) entry of
   the mean transfer matrix as:

       mu_hat(j, i) = (1/N) sum_{t=1}^{N} s_future_j(t) * s_past_i(t)

   Under correct model specification, mu_hat(j, i) converges to the (j, i)
   entry of the cross-block Jacobian, which is non-zero only when neuron i
   directly influences neuron j at the given time lag.

2. **HAC t-test (Newey-West)**.  Because the score products Y_ji(t) are
   serially correlated (time-series data), standard i.i.d. variance
   estimates would understate uncertainty.  The Newey-West estimator uses
   Bartlett kernel weights:

       sigma^2_HAC = gamma_0 + 2 * sum_{l=1}^{L} (1 - l/(L+1)) * gamma_l

   where gamma_l = (1/N) sum_t (Y_t - Y_bar)(Y_{t-l} - Y_bar).  The
   t-statistic is t = mu_hat * sqrt(N) / sigma_HAC, with a two-sided
   p-value from the standard normal approximation (valid for large N).

3. **FDR control**.  With n*(n-1) off-diagonal hypotheses tested, we apply
   either Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) FDR control.
   BY is more conservative and is valid under arbitrary dependence among
   the p-values -- appropriate here because score products share
   common score network parameters.

4. **Cross-fitting**.  To ensure valid p-values, we use K-fold cross-fitting:
   for each fold k, the score network is trained on the other K-1 folds,
   and scores are computed on fold k only.  This prevents over-fitting
   from inflating the apparent signal in mu_hat.

5. **Standardization**.  Training-fold mean and standard deviation are
   computed and applied to all windows (train + held-out).  This prevents
   information leakage from held-out windows into the normalisation.

Provides
--------
- ``newey_west_variance`` -- HAC variance estimator (Bartlett kernel).
- ``hac_test_mu_hat`` -- per-edge t-test for the mean transfer matrix.
- ``apply_fdr`` -- BH / BY multiple-testing correction.
- ``standardize_windows`` -- train-fold-only standardization with clipping.
- ``create_fold_assignments`` -- strided fold assignment respecting segments.
- ``cross_fit`` -- generic K-fold cross-fitting loop.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# HAC variance
# ---------------------------------------------------------------------------

def newey_west_variance(y: np.ndarray, max_lag: int) -> float:
    """
    Newey-West HAC variance estimator with Bartlett kernel weights.

    Parameters
    ----------
    y : 1-d array of length N
    max_lag : int
        Maximum lag for autocovariance summation.

    Returns
    -------
    float  (always >= 1e-12 for numerical safety)
    """
    N = len(y)
    yc = y - y.mean()
    gamma0 = np.dot(yc, yc) / N
    gamma_sum = 0.0
    L = min(max_lag, N - 1)
    for ell in range(1, L + 1):
        weight = 1.0 - ell / (L + 1)
        gamma_ell = np.dot(yc[ell:], yc[:-ell]) / N
        gamma_sum += 2.0 * weight * gamma_ell
    return max(gamma0 + gamma_sum, 1e-12)


# ---------------------------------------------------------------------------
# HAC t-test for mu_hat
# ---------------------------------------------------------------------------

def hac_test_mu_hat(
    s_future: np.ndarray,
    s_past: np.ndarray,
    hac_max_lag: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HAC t-test for the mean transfer matrix.

    mu_hat(j, i) = mean_t[ s_future_j(t) * s_past_i(t) ]

    Under the null (no edge j<-i), mu_hat(j,i) ~ N(0, sigma^2/N)
    where sigma^2 is estimated via Newey-West.

    Parameters
    ----------
    s_future : ndarray (N, n)  scores for the future block
    s_past   : ndarray (N, n)  scores for the past block
    hac_max_lag : int

    Returns
    -------
    mu_hat   : ndarray (n, n)
    p_values : ndarray (n, n)  two-sided p-values (diagonal = 1.0)
    """
    N, n = s_future.shape
    mu_hat = np.zeros((n, n))
    p_values = np.ones((n, n))

    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            Y_ji = s_future[:, j] * s_past[:, i]
            mu_hat[j, i] = Y_ji.mean()
            var_hac = newey_west_variance(Y_ji, hac_max_lag)
            se = np.sqrt(var_hac / N)
            if se > 1e-10:
                t_stat = mu_hat[j, i] / se
                p_values[j, i] = 2.0 * (1.0 - norm.cdf(np.abs(t_stat)))

    return mu_hat, p_values


# ---------------------------------------------------------------------------
# FDR control
# ---------------------------------------------------------------------------

def apply_fdr(
    p_values: np.ndarray,
    alpha: float = 0.1,
    method: str = "by",
) -> np.ndarray:
    """
    Apply FDR control to a (n, n) p-value matrix.

    Self-loops (diagonal) are always excluded.

    Parameters
    ----------
    p_values : ndarray (n, n)
    alpha : float
        Nominal FDR level.
    method : ``'bh'`` or ``'by'``

    Returns
    -------
    significant : bool ndarray (n, n)
    """
    n = p_values.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pvals = p_values[mask]
    m = len(pvals)
    if m == 0:
        return np.zeros_like(p_values, dtype=bool)

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    if method == "by":
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        thresholds = alpha * np.arange(1, m + 1) / (m * c_m)
    elif method == "bh":
        thresholds = alpha * np.arange(1, m + 1) / m
    else:
        raise ValueError(f"FDR method must be 'bh' or 'by', got '{method}'")

    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros_like(p_values, dtype=bool)

    k_max = np.where(below)[0][-1]
    threshold = sorted_p[k_max]

    return (p_values <= threshold) & mask


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize_windows(
    windows: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize windows using statistics computed on the training subset only.

    Clips extreme values to [-10, 10] after standardization.

    Returns
    -------
    windows_std : ndarray
    mean : ndarray (D,)
    std  : ndarray (D,)
    """
    windows = windows.astype(np.float64)
    train = windows[train_mask]
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True) + 1e-8
    out = np.clip((windows - mean) / std, -10.0, 10.0)
    return np.nan_to_num(out, nan=0.0), mean.squeeze(), std.squeeze()


# ---------------------------------------------------------------------------
# Fold assignment (respects segment boundaries)
# ---------------------------------------------------------------------------

def create_fold_assignments(
    stim_ids: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """
    Strided fold assignment within each segment so every fold contains
    windows from every stimulus and temporal locality is preserved.
    """
    fold_ids = np.zeros(len(stim_ids), dtype=np.int32)
    for u in np.unique(stim_ids):
        idx = np.where(stim_ids == u)[0]
        fold_ids[idx] = np.arange(len(idx)) % n_folds
    return fold_ids


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------

def cross_fit(
    windows: np.ndarray,
    stim_ids: np.ndarray,
    n_folds: int,
    model_factory: Callable[[], "nn.Module"],
    train_fn: Callable[["nn.Module", np.ndarray], None],
    score_fn: Callable[["nn.Module", np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    K-fold cross-fitting: for each fold k, train on the other K-1 folds
    and compute scores on fold k.  This ensures p-values are computed
    on truly held-out scores.

    Parameters
    ----------
    windows : ndarray (N, D)  raw (unstandardized) windows
    stim_ids : ndarray (N,)
    n_folds : int
    model_factory : callable  () -> nn.Module  (fresh model each fold)
    train_fn : callable  (model, train_windows_std) -> None
    score_fn : callable  (model, windows_std) -> ndarray (N_k, D)

    Returns
    -------
    scores_heldout : ndarray (N, D)
    """
    N, D = windows.shape
    fold_ids = create_fold_assignments(stim_ids, n_folds)
    scores_heldout = np.zeros((N, D), dtype=np.float32)

    for k in range(n_folds):
        train_mask = fold_ids != k
        heldout_mask = fold_ids == k
        if heldout_mask.sum() == 0:
            continue

        windows_std, _, _ = standardize_windows(windows, train_mask)
        model = model_factory()
        train_fn(model, windows_std[train_mask])
        scores_heldout[heldout_mask] = score_fn(model, windows_std[heldout_mask])

    return scores_heldout
