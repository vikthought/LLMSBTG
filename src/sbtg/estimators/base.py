"""
Base estimator interface and shared result container.

Every SBTG estimator (Approach A, C, or future variants) inherits from
``BaseEstimator`` and returns a ``MultiLagResult`` from ``fit()``.

``MultiLagResult`` is a self-contained snapshot of a multi-lag analysis:
it holds, for each lag, the mean transfer matrix (mu_hat), p-values,
FDR-thresholded binary adjacency, and continuous scores suitable for
AUROC evaluation.  It also carries metadata (hyper-parameters, device,
approach identifier) for reproducibility.

The ``save_result`` / ``load_result`` utilities in ``sbtg.utils.io``
serialise ``MultiLagResult`` to/from ``.npz`` + JSON for archival.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MultiLagResult:
    """Container for multi-lag SBTG analysis results.

    Attributes
    ----------
    mu_hat : dict[int, ndarray]
        lag -> (n, n) mean transfer matrix.
    p_values : dict[int, ndarray]
        lag -> (n, n) HAC p-values.
    significant : dict[int, ndarray]
        lag -> (n, n) binary adjacency after FDR.
    scores_raw : dict[int, ndarray]
        lag -> (n, n) continuous scores (|mu_hat|) for AUROC evaluation.
    n_neurons : int
    n_windows : dict[int, int]
        lag -> number of windows used.
    metadata : dict
        Arbitrary provenance (HP config, timing, approach name, etc.).
    """

    mu_hat: Dict[int, np.ndarray]
    p_values: Dict[int, np.ndarray]
    significant: Dict[int, np.ndarray]
    scores_raw: Dict[int, np.ndarray]
    n_neurons: int
    n_windows: Dict[int, int]
    metadata: dict = field(default_factory=dict)

    # Convenience ---------------------------------------------------------

    def lags(self) -> List[int]:
        return sorted(self.mu_hat.keys())

    def adjacency(self, lag: int) -> np.ndarray:
        """Binary adjacency for a given lag."""
        return self.significant.get(lag, np.zeros((self.n_neurons, self.n_neurons)))

    def continuous_scores(self, lag: int) -> np.ndarray:
        """Continuous edge scores (|mu_hat|) for a given lag (for AUROC)."""
        return self.scores_raw.get(lag, np.zeros((self.n_neurons, self.n_neurons)))

    def n_edges(self, lag: int) -> int:
        return int(self.adjacency(lag).sum())

    def summary_dict(self) -> dict:
        """One-line-per-lag summary for quick inspection."""
        rows = {}
        for lag in self.lags():
            mu = self.mu_hat[lag]
            mask = ~np.eye(self.n_neurons, dtype=bool)
            rows[lag] = {
                "lag": lag,
                "mean_abs_mu": float(np.abs(mu[mask]).mean()),
                "max_abs_mu": float(np.abs(mu[mask]).max()),
                "n_edges": self.n_edges(lag),
                "n_windows": self.n_windows.get(lag, 0),
            }
        return rows


class BaseEstimator(ABC):
    """Abstract base for all SBTG estimators."""

    @abstractmethod
    def fit(self, X_list: List[np.ndarray]) -> MultiLagResult:
        """
        Fit the model on a list of (T_u, n) time-series segments.

        Parameters
        ----------
        X_list : list of ndarray
            One (T, n_neurons) array per worm / recording segment.

        Returns
        -------
        MultiLagResult
        """
        ...
