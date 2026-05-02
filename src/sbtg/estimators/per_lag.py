"""
Approach A: Per-Lag 2-Block SBTG estimator.

Trains a separate 2-block model for each target lag.  Simpler than
Approach C but does not condition on intermediate lags.  Kept for
comparative analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from sbtg.core.score_nets import TwoBlockScoreNet
from sbtg.core.dsm import train_score_model, compute_scores
from sbtg.core.windows import build_two_block_windows
from sbtg.core.inference import (
    hac_test_mu_hat,
    apply_fdr,
    standardize_windows,
    create_fold_assignments,
)
from sbtg.core.null_contrast import compute_null_contrast_from_mu
from sbtg.estimators.base import BaseEstimator, MultiLagResult


class PerLagEstimator(BaseEstimator):
    """Approach A: independent 2-block SBTG per lag."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        *,
        noise_std: float = 0.1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 128,
        l1_lambda: float = 0.0,
        n_folds: int = 5,
        hac_max_lag: int = 5,
        fdr_alpha: float = 0.1,
        fdr_method: str = "by",
        device: Optional[str] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.lags = lags if lags is not None else [1, 2, 3, 5]
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.n_folds = n_folds
        self.hac_max_lag = hac_max_lag
        self.fdr_alpha = fdr_alpha
        self.fdr_method = fdr_method
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def fit(self, X_list: List[np.ndarray]) -> MultiLagResult:
        n_neurons = X_list[0].shape[1]
        mu_all: Dict[int, np.ndarray] = {}
        pv_all: Dict[int, np.ndarray] = {}
        sig_all: Dict[int, np.ndarray] = {}
        sc_all: Dict[int, np.ndarray] = {}
        nw_all: Dict[int, int] = {}

        for lag in self.lags:
            if self.verbose:
                print(f"\n[Approach A] lag={lag}")

            try:
                windows, stim_ids, _ = build_two_block_windows(X_list, lag)
            except ValueError as e:
                if "No windows could be built" in str(e):
                    if self.verbose:
                        print(f"  Skipping lag={lag}: {str(e)}")
                    continue
                raise e

            nw_all[lag] = len(windows)

            fold_ids = create_fold_assignments(stim_ids, self.n_folds)
            scores_out = np.zeros_like(windows, dtype=np.float32)

            for k in range(self.n_folds):
                if self.verbose:
                    print(f"    fold {k + 1}/{self.n_folds}", flush=True)
                train_mask = fold_ids != k
                heldout_mask = fold_ids == k
                if heldout_mask.sum() == 0:
                    continue

                w_std, _, _ = standardize_windows(windows, train_mask)
                model = TwoBlockScoreNet(
                    n_neurons, hidden_dim=self.hidden_dim, num_layers=self.num_layers
                )
                train_score_model(
                    model, w_std[train_mask],
                    noise_std=self.noise_std, lr=self.lr, epochs=self.epochs,
                    batch_size=self.batch_size, l1_lambda=self.l1_lambda,
                    device=self.device, verbose=False,
                )
                scores_out[heldout_mask] = compute_scores(
                    model, w_std[heldout_mask], device=self.device
                )

            s_past = scores_out[:, :n_neurons]
            s_future = scores_out[:, n_neurons:]
            mu, pv = hac_test_mu_hat(s_future, s_past, self.hac_max_lag)
            sig = apply_fdr(pv, self.fdr_alpha, self.fdr_method)

            mu_all[lag] = mu
            pv_all[lag] = pv
            sig_all[lag] = sig.astype(np.float64)
            sc_all[lag] = np.abs(mu)

            if self.verbose:
                nc = compute_null_contrast_from_mu(mu)
                print(f"  edges={int(sig.sum())}  NC={nc:.3f}")

        return MultiLagResult(
            mu_hat=mu_all, p_values=pv_all, significant=sig_all,
            scores_raw=sc_all, n_neurons=n_neurons, n_windows=nw_all,
            metadata={"approach": "A", "lags": self.lags},
        )
