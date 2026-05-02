"""
Approach C: Minimal Multi-Block SBTG estimator.

Theoretical motivation
----------------------
The goal of multi-lag analysis is to infer the *direct* lag-r Jacobian

    J_r(j, i) = d E[x_{t+r,j} | x_{<=t}] / d x_{t,i}

which is non-zero only if neuron i *directly* influences neuron j at
exactly lag r (as opposed to indirect effects routed through intermediate
time-steps).

**Why Approach C gives lag separation.**  For a lag-r analysis, we build
windows of (r+1) consecutive observations:

    z_t = (x_t, x_{t+1}, ..., x_{t+r})

and fit a structured score model whose energy includes *all* coupling
matrices W_1, ..., W_r.  The score of the joint distribution factorises
into marginal and cross-block terms:

    s(z) = -grad_z [ sum_k g_k(z^(k)) + sum_{l=1}^{r} z_{future}^T W_l z_{lag-l} ]

Because W_r is the *only* coupling matrix between the first block (z^0 = x_t)
and the last block (z^r = x_{t+r}), the mean transfer product
E[s_future_j * s_past_i] converges to J_r(j,i) -- the direct lag-r
effect -- while indirect paths through intermediate lags are absorbed by
the intermediate coupling matrices W_1, ..., W_{r-1} and the marginal
energies g_1, ..., g_{r-1}.

In contrast, Approach A (two-block, z = [x_t, x_{t+r}]) omits the
intermediate observations, so the bilinear coupling W conflates direct
lag-r effects with indirect effects through intermediate lags.

Per-lag models
--------------
Each target lag trains an independent model:

    lag 1 -> z = (x_t, x_{t+1})          -- 2 blocks, 1 W
    lag 2 -> z = (x_t, x_{t+1}, x_{t+2}) -- 3 blocks, 2 W's
    lag r -> z = (x_t, ..., x_{t+r})      -- (r+1) blocks, r W's

Independence means per-lag hyperparameter tuning is possible.

Cross-fitting
-------------
The estimator uses K-fold cross-fitting (default K=5) so that the
scores used for HAC inference are always from held-out data, giving
valid p-values and FDR control.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from sbtg.core.score_nets import TwoBlockScoreNet, MultiBlockScoreNet
from sbtg.core.dsm import train_score_model, compute_scores
from sbtg.core.windows import build_two_block_windows, build_minimal_multiblock_windows
from sbtg.core.inference import (
    hac_test_mu_hat,
    apply_fdr,
    cross_fit,
    standardize_windows,
    create_fold_assignments,
)
from sbtg.core.null_contrast import compute_null_contrast_from_mu
from sbtg.estimators.base import BaseEstimator, MultiLagResult


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MinimalMultiBlockEstimator(BaseEstimator):
    """
    Approach C estimator -- one (r+1)-block model per target lag.

    Parameters
    ----------
    lags : list[int]
        Target lags to analyse (e.g. [1, 2, 3, 5]).
    noise_std, hidden_dim, num_layers, lr, epochs : float / int
        DSM training hyper-parameters.
    batch_size : int
    l1_lambda : float
        L1 penalty on coupling matrices.
    n_folds : int
        Number of cross-fitting folds.
    hac_max_lag : int
        Maximum lag for Newey-West variance.
    fdr_alpha : float
        Nominal FDR level.
    fdr_method : str
        ``'bh'`` or ``'by'``.
    device : str or None
        Torch device (auto-detected if None).
    verbose : bool
    seed : int
    """

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
        self.device = device or _auto_device()
        self.verbose = verbose
        self.seed = seed

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fit(self, X_list: List[np.ndarray]) -> MultiLagResult:
        n_neurons = X_list[0].shape[1]

        mu_hat_all: Dict[int, np.ndarray] = {}
        p_values_all: Dict[int, np.ndarray] = {}
        sig_all: Dict[int, np.ndarray] = {}
        scores_all: Dict[int, np.ndarray] = {}
        nwin_all: Dict[int, int] = {}

        for lag_r in self.lags:
            if self.verbose:
                print(f"\n[Approach C] lag={lag_r}  ({lag_r + 1}-block model)")

            n_blocks = lag_r + 1

            # Build windows
            try:
                if n_blocks == 2:
                    windows, stim_ids, _ = build_two_block_windows(X_list, lag_r)
                else:
                    windows, stim_ids, _ = build_minimal_multiblock_windows(X_list, lag_r)
            except ValueError as e:
                if "No windows could be built" in str(e):
                    if self.verbose:
                        print(f"  Skipping lag={lag_r}: {str(e)}")
                    continue
                raise e

            nwin_all[lag_r] = len(windows)
            if self.verbose:
                print(f"  windows={len(windows)}  dim={windows.shape[1]}")

            # Cross-fitted scores
            scores_heldout = self._cross_fit_lag(
                windows, stim_ids, n_neurons, lag_r
            )

            # Extract block scores for past (block 0) and future (block r)
            scores_blocks = scores_heldout.reshape(len(scores_heldout), n_blocks, n_neurons)
            s_past = scores_blocks[:, 0, :]
            s_future = scores_blocks[:, -1, :]

            # HAC inference
            mu_hat, p_values = hac_test_mu_hat(s_future, s_past, self.hac_max_lag)
            significant = apply_fdr(p_values, self.fdr_alpha, self.fdr_method)

            mu_hat_all[lag_r] = mu_hat
            p_values_all[lag_r] = p_values
            sig_all[lag_r] = significant.astype(np.float64)
            scores_all[lag_r] = np.abs(mu_hat)

            if self.verbose:
                mask = ~np.eye(n_neurons, dtype=bool)
                nc = compute_null_contrast_from_mu(mu_hat)
                print(
                    f"  |mu| mean={np.abs(mu_hat[mask]).mean():.4f}  "
                    f"NC={nc:.3f}  edges={int(significant.sum())}"
                )

        return MultiLagResult(
            mu_hat=mu_hat_all,
            p_values=p_values_all,
            significant=sig_all,
            scores_raw=scores_all,
            n_neurons=n_neurons,
            n_windows=nwin_all,
            metadata={
                "approach": "C",
                "lags": self.lags,
                "epochs": self.epochs,
                "noise_std": self.noise_std,
                "hidden_dim": self.hidden_dim,
                "n_folds": self.n_folds,
                "device": self.device,
                "seed": self.seed,
            },
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _cross_fit_lag(
        self,
        windows: np.ndarray,
        stim_ids: np.ndarray,
        n_neurons: int,
        lag: int,
    ) -> np.ndarray:
        """Cross-fitted inference for one lag."""
        n_blocks = lag + 1
        fold_ids = create_fold_assignments(stim_ids, self.n_folds)
        N = len(windows)
        scores_out = np.zeros_like(windows, dtype=np.float32)

        for k in range(self.n_folds):
            if self.verbose:
                print(f"    fold {k + 1}/{self.n_folds}", flush=True)

            train_mask = fold_ids != k
            heldout_mask = fold_ids == k
            if heldout_mask.sum() == 0:
                continue

            windows_std, _, _ = standardize_windows(windows, train_mask)

            model = self._make_model(n_neurons, lag)
            train_score_model(
                model,
                windows_std[train_mask],
                noise_std=self.noise_std,
                lr=self.lr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                l1_lambda=self.l1_lambda,
                device=self.device,
                verbose=False,
            )
            heldout_scores = compute_scores(
                model,
                windows_std[heldout_mask],
                device=self.device,
            )
            scores_out[heldout_mask] = heldout_scores

        return scores_out

    def _make_model(self, n_neurons: int, lag: int):
        if lag + 1 == 2:
            return TwoBlockScoreNet(
                n_neurons,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
        return MultiBlockScoreNet(
            n_neurons,
            p_max=lag,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
