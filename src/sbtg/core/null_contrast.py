"""
Null-contrast objective for hyperparameter tuning.

The null contrast (NC) is a model-selection statistic that measures
whether the learned score network captures genuine temporal structure
beyond what would be expected from independent marginals.

Definition
----------
Given held-out scores s_future(t) and s_past(t), compute the mean
transfer matrix:

    mu_hat(j,i) = (1/N) sum_t  s_future_j(t) * s_past_i(t)

Under the null (no temporal dependence), s_future and s_past are
independent, so E[mu(j,i)] = 0.  To estimate the null level, we
temporally shuffle s_past:

    mu_null(j,i) = (1/N) sum_t  s_future_j(t) * s_past_i(pi(t))

where pi is a random permutation.  The null contrast is:

    NC = mean_{j!=i} |mu_hat(j,i)|  /  mean_{j!=i} |mu_null(j,i)|

Interpretation:
- NC >> 1 : the score network has learned meaningful cross-block signal.
- NC ~ 1  : no better than chance (bad hyper-parameters or insufficient data).
- NC < 1  : degenerate solution (extremely rare).

As an Optuna objective, NC is maximised during hyperparameter search.

Variants
--------
- ``compute_null_contrast_from_scores``: takes raw score arrays, averages
  over multiple shuffle replicates.
- ``compute_null_contrast_from_mu``: takes the already-computed mu_hat and
  shuffles its off-diagonal entries (cheaper, slightly less precise).
- ``compute_edge_stability``: bootstrap stability of mu_hat as a secondary
  diagnostic (not used in tuning, but useful for result inspection).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_null_contrast_from_scores(
    s_future: np.ndarray,
    s_past: np.ndarray,
    n_shuffles: int = 5,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Null contrast via temporal shuffle of s_past.

    Parameters
    ----------
    s_future, s_past : ndarray (N, n)
    n_shuffles : int
    seed : int

    Returns
    -------
    null_contrast : float  (>= 1.0 means real signal present)
    mu_hat : ndarray (n, n)
    """
    N, n = s_future.shape
    mask = ~np.eye(n, dtype=bool)

    mu_hat = (s_future.T @ s_past) / N
    real_mean = np.abs(mu_hat[mask]).mean()

    rng = np.random.default_rng(seed)
    null_means = []
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        mu_null = (s_future.T @ s_past[perm]) / N
        null_means.append(np.abs(mu_null[mask]).mean())

    null_mean = np.mean(null_means)
    if null_mean < 1e-10:
        return 1.0, mu_hat
    return real_mean / null_mean, mu_hat


def compute_null_contrast_from_mu(
    mu_hat: np.ndarray,
    n_shuffles: int = 10,
    seed: int = 42,
) -> float:
    """Simpler variant: shuffle off-diagonal entries of mu_hat itself."""
    n = mu_hat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    real_mean = np.abs(mu_hat[mask]).mean()

    rng = np.random.default_rng(seed)
    null_means = []
    for _ in range(n_shuffles):
        flat = mu_hat[mask].copy()
        rng.shuffle(flat)
        null_means.append(np.abs(flat).mean())

    null_mean = np.mean(null_means)
    return real_mean / null_mean if null_mean > 1e-10 else 1.0


def compute_edge_stability(
    scores: np.ndarray,
    n_bootstrap: int = 10,
    seed: int = 42,
) -> float:
    """
    Bootstrap stability of mu_hat: mean pairwise correlation across
    bootstrap resamples.  Higher = more reliable edges.
    """
    N = scores.shape[0]
    n = scores.shape[1] // 2
    rng = np.random.default_rng(seed)

    s_past = scores[:, :n]
    s_future = scores[:, n : 2 * n]

    mu_boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        mu_b = (s_future[idx].T @ s_past[idx]) / N
        mu_boots.append(mu_b.flatten())

    mu_boots = np.array(mu_boots)
    corrs = np.corrcoef(mu_boots)
    off_diag = ~np.eye(n_bootstrap, dtype=bool)
    return float(corrs[off_diag].mean())
