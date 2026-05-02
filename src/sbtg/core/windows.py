"""
Window construction for SBTG models.

Windows pair observations at different time lags:

- Two-block:  z_t = [x_t, x_{t+lag}]
- Minimal multi-block:  z_t = [x_t, x_{t+1}, ..., x_{t+lag}]

All builders return a consistent triple (windows, stim_ids, local_t) so that
downstream code (cross-fitting, standardization) works identically regardless
of window type.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_two_block_windows(
    X_list: List[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2-block windows:  z = (x_t, x_{t+lag}).

    Parameters
    ----------
    X_list : list of ndarray, each (T_u, n)
        One array per segment / worm.
    lag : int  (>= 1)
        Temporal offset between the two blocks.

    Returns
    -------
    windows : ndarray (N, 2n)
    stim_ids : ndarray (N,)   segment index for each window
    local_t  : ndarray (N,)   local time index within segment
    """
    windows, sids, lts = [], [], []
    for u, X in enumerate(X_list):
        X = np.asarray(X, dtype=np.float64)
        T, n = X.shape
        if T <= lag:
            continue
        n_win = T - lag
        x_past = X[:n_win]
        x_future = X[lag : lag + n_win]
        z = np.concatenate([x_past, x_future], axis=1)
        windows.append(z)
        sids.append(np.full(n_win, u, dtype=np.int32))
        lts.append(np.arange(n_win, dtype=np.int32))

    if not windows:
        raise ValueError(f"No windows could be built (lag={lag}, check input lengths).")
    return np.concatenate(windows), np.concatenate(sids), np.concatenate(lts)


def build_minimal_multiblock_windows(
    X_list: List[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build minimal multi-block windows for Approach C.

    For lag *r* the window has (r+1) consecutive blocks:
        z_t = [x_t, x_{t+1}, ..., x_{t+r}]

    This conditions on intermediate lags without including irrelevant
    future blocks, giving the minimal Markov blanket for identifying the
    direct lag-r Jacobian J_r.

    Parameters
    ----------
    X_list : list of ndarray, each (T_u, n)
    lag : int  (>= 1)

    Returns
    -------
    windows : ndarray (N, (lag+1)*n)
    stim_ids : ndarray (N,)
    local_t  : ndarray (N,)
    """
    n_blocks = lag + 1
    windows, sids, lts = [], [], []

    for u, X in enumerate(X_list):
        X = np.asarray(X, dtype=np.float64)
        T, n = X.shape
        if T < n_blocks:
            continue
        n_win = T - lag
        # Vectorised: stack shifted views
        blocks = np.concatenate([X[k : k + n_win] for k in range(n_blocks)], axis=1)
        windows.append(blocks)
        sids.append(np.full(n_win, u, dtype=np.int32))
        lts.append(np.arange(n_win, dtype=np.int32))

    if not windows:
        raise ValueError(f"No windows could be built (lag={lag}, check input lengths).")
    return np.concatenate(windows), np.concatenate(sids), np.concatenate(lts)
