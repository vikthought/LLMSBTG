"""
Denoising Score Matching (DSM) training and score evaluation.

Background
----------
Given data z ~ p(z), the *score function* is s(z) = grad_z log p(z).
Direct estimation of the score requires the intractable normalising
constant of p(z).  DSM sidesteps this by training against Gaussian-
corrupted data (Vincent 2011):

    L(theta) = E_{z ~ p(z), eps ~ N(0,I)}
        [ || s_theta(z + sigma * eps) + eps / sigma ||^2 ]

This objective has the same minimiser as the Fisher divergence between
s_theta and the true score, up to a constant independent of theta.

The noise scale sigma controls the bias--variance trade-off: larger
sigma smooths the loss landscape (easier optimisation) but biases the
score estimate toward the Gaussian kernel; smaller sigma gives less
bias but noisier gradients and more local minima.

Implementation
--------------
- ``dsm_loss``: single-batch loss computation (no reduction over
  the full dataset -- the caller iterates over a DataLoader).
- ``train_score_model``: full training loop with Adam, gradient
  clipping, optional L1 penalty on coupling matrices, and a progress
  callback hook.
- ``compute_scores``: evaluation-mode inference with ``torch.enable_grad``
  (needed because the score network uses ``torch.autograd.grad`` internally
  to differentiate the energy U with respect to z).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def dsm_loss(model: nn.Module, z: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Compute DSM loss for a batch of clean samples *z*."""
    eps = torch.randn_like(z)
    z_noisy = z + noise_std * eps
    target = -eps / noise_std
    pred = model(z_noisy)
    return ((pred - target) ** 2).mean()


def train_score_model(
    model: nn.Module,
    train_windows: np.ndarray,
    *,
    noise_std: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 128,
    l1_lambda: float = 0.0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, list]:
    """
    Train a structured score model using DSM.

    Parameters
    ----------
    model : nn.Module
        Score network (moved to *device* internally).
    train_windows : ndarray, shape (N, D)
        Standardized training windows.
    noise_std : float
        Gaussian corruption scale sigma.
    lr, epochs, batch_size : float / int
        Optimizer and schedule settings.
    l1_lambda : float
        L1 penalty on coupling matrices W (sparsity prior).
    device : str
        Torch device string.
    verbose : bool
        Print loss every ~10 % of epochs.
    progress_callback : callable, optional
        Called with (epoch, loss) after each epoch.

    Returns
    -------
    history : dict
        Keys ``'epoch'`` and ``'loss'`` with per-epoch values.
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tensor = torch.as_tensor(train_windows, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_windows) > batch_size,
    )

    history: Dict[str, list] = {"epoch": [], "loss": []}
    log_every = max(1, epochs // 10)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            loss = dsm_loss(model, batch, noise_std)

            if l1_lambda > 0.0 and hasattr(model, "W"):
                w = model.W
                if isinstance(w, nn.ParameterList):
                    l1 = sum(p.abs().sum() for p in w)
                else:
                    l1 = w.abs().sum()
                loss = loss + l1_lambda * l1

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["loss"].append(avg)

        if progress_callback is not None:
            progress_callback(epoch, avg)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.6f}")

    return history


def compute_scores(
    model: nn.Module,
    windows: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate the score network on *windows* in eval mode."""
    device = torch.device(device)
    model = model.to(device).eval()

    parts: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.as_tensor(
            windows[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.enable_grad():
            s = model(batch)
        parts.append(s.detach().cpu().numpy())

    return np.concatenate(parts, axis=0)
