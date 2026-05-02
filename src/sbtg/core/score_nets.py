"""
Structured score networks for SBTG.

All architectures define a scalar energy function U(z; theta) and compute
the score vector s(z; theta) = -grad_z U(z; theta) via PyTorch autograd.

The key design constraint is that cross-block coupling (the W matrices)
is *bilinear* and *explicit*:

    U_coupling = z_future^T W z_past

This forces the learned score to decompose into marginal components (the
g_k networks, which capture within-block structure) and cross-block
components (the W matrices, which encode direct inter-neuron influence).
After training via denoising score matching (DSM), the cross-block
coupling W converges to the Jacobian of the conditional expectation
E[x_future | x_past], which is exactly the functional connectivity
matrix we want to infer.

Architectures
-------------

TwoBlockScoreNet (Approach A / single-lag)
    z = (x_past, x_future),  both in R^n.
    U = g0(x_past) + g1(x_future) + x_future^T W x_past
    W is (n, n) -- one coupling matrix per pair.

MultiBlockScoreNet (Approach B / C)
    z = (z^0, z^1, ..., z^p),  each in R^n.
    U = sum_{k=0}^{p} g_k(z^k) + sum_{r=1}^{p} z_future^T W_r z_{lag-r}
    W_r is (n, n) for each lag r -- enables true lag separation.

FeatureBilinearScoreNet
    z = (x0, x1),  both in R^n.
    U = g0(x0) + g1(x1) + psi(x1)^T W phi(x0)
    phi, psi are learned feature maps R^n -> R^d; W is (d, d).
    Provides richer (non-linear) cross-block coupling at the cost of
    not directly reading off per-neuron connectivity from W.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScalarMLP(nn.Module):
    """MLP mapping R^d -> R (scalar energy)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d_in, hidden_dim), nn.SiLU()]
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class VectorMLP(nn.Module):
    """MLP mapping R^d -> R^r (feature vector)."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d_in, hidden_dim), nn.SiLU()]
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Two-block structured score net  (Approach A / single-lag)
# ---------------------------------------------------------------------------

class TwoBlockScoreNet(nn.Module):
    """
    Structured score model for 2-block windows.

    Energy:  U(z) = g0(x_past) + g1(x_future) + x_future^T W x_past
    Score:   s(z) = -grad_z U(z)
    """

    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n_neurons
        self.g0 = ScalarMLP(n_neurons, hidden_dim, num_layers)
        self.g1 = ScalarMLP(n_neurons, hidden_dim, num_layers)
        self.W = nn.Parameter(torch.empty(n_neurons, n_neurons))
        nn.init.uniform_(self.W, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = self.n
        z = z.clone().requires_grad_(True)
        x_past, x_future = z[:, :n], z[:, n:]

        U = (
            self.g0(x_past)
            + self.g1(x_future)
            + torch.einsum("bi,ij,bj->b", x_future, self.W, x_past)
        )

        (grad_z,) = torch.autograd.grad(U.sum(), z, create_graph=self.training)
        return -grad_z


# ---------------------------------------------------------------------------
# Multi-block structured score net  (Approach B / C)
# ---------------------------------------------------------------------------

class MultiBlockScoreNet(nn.Module):
    """
    Structured score model for (p+1)-block windows.

    Energy:  U(z) = sum_k g_k(z^(k)) + sum_{r=1}^{p} z_future^T W_r z_{lag-r}
    Score:   s(z) = -grad_z U(z)

    The W_r matrices encode direct lag-r connectivity, giving true lag
    separation when the model is trained on the full (p+1)-block window.
    """

    def __init__(
        self,
        n_neurons: int,
        p_max: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n_neurons
        self.p_max = p_max
        self.n_blocks = p_max + 1

        self.g = nn.ModuleList(
            [ScalarMLP(n_neurons, hidden_dim, num_layers) for _ in range(self.n_blocks)]
        )
        self.W = nn.ParameterList(
            [nn.Parameter(torch.empty(n_neurons, n_neurons)) for _ in range(p_max)]
        )
        for W_r in self.W:
            nn.init.uniform_(W_r, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        n = self.n
        z = z.clone().requires_grad_(True)
        blocks = z.view(B, self.n_blocks, n)

        U = torch.zeros(B, device=z.device, dtype=z.dtype)
        for k in range(self.n_blocks):
            U = U + self.g[k](blocks[:, k, :])

        z_future = blocks[:, self.p_max, :]
        for r in range(1, self.p_max + 1):
            z_lag_r = blocks[:, self.p_max - r, :]
            U = U + torch.einsum("bi,ij,bj->b", z_future, self.W[r - 1], z_lag_r)

        (grad_z,) = torch.autograd.grad(U.sum(), z, create_graph=self.training)
        return -grad_z


# ---------------------------------------------------------------------------
# Feature-bilinear score net  (richer coupling)
# ---------------------------------------------------------------------------

class FeatureBilinearScoreNet(nn.Module):
    """
    Feature-bilinear coupling for 2-block windows.

    Energy:  U(z) = g0(x0) + g1(x1) + psi(x1)^T W phi(x0)

    The learned feature maps phi, psi allow non-linear cross-block
    interactions through a low-rank bilinear coupling.
    """

    def __init__(
        self,
        n_neurons: int,
        feature_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        feature_hidden_dim: int = 64,
        feature_num_layers: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n_neurons
        self.g0 = ScalarMLP(n_neurons, hidden_dim, num_layers)
        self.g1 = ScalarMLP(n_neurons, hidden_dim, num_layers)
        self.phi = VectorMLP(n_neurons, feature_dim, feature_hidden_dim, feature_num_layers)
        self.psi = VectorMLP(n_neurons, feature_dim, feature_hidden_dim, feature_num_layers)
        self.W = nn.Parameter(torch.empty(feature_dim, feature_dim))
        nn.init.uniform_(self.W, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = self.n
        z = z.clone().requires_grad_(True)
        x0, x1 = z[:, :n], z[:, n:]

        p = self.phi(x0)
        q = self.psi(x1)
        U = self.g0(x0) + self.g1(x1) + ((q @ self.W) * p).sum(dim=-1)

        (grad_z,) = torch.autograd.grad(U.sum(), z, create_graph=self.training)
        return -grad_z
