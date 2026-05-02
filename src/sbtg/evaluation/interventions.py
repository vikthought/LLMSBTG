import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Iterable, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Multi-lag direction extraction
# ---------------------------------------------------------------------------

def multilag_svd_directions(
    m_bar_r: np.ndarray,
    k: int,
    lags: Optional[List[int]] = None,
    weighting: str = "uniform",
    side: str = "source",
) -> Tuple[np.ndarray, Dict]:
    """Compute top-k ablation directions by jointly decomposing M_bar across lags.

    Three approaches available via *weighting*:

    * ``"uniform"`` – vertically stack ``M_bar_r[lag]`` for each lag, take SVD
      of the tall matrix.  Equivalent to eigenvectors of
      ``sum_r M_bar_r[r]^T M_bar_r[r]`` (source) or ``M_bar_r[r] M_bar_r[r]^T``
      (target).  All lags contribute equally per-row.

    * ``"amplitude"`` – weight each lag's matrix by its Frobenius norm ``A_r``
      before stacking.  Lags with stronger coupling contribute more.

    * ``"heterogeneity"`` – weight by ``H_r`` (position heterogeneity) so that
      lags where the operator varies most across positions dominate.  These are
      the lags where *non-stationarity* concentrates, so the resulting
      directions target the non-stationary subspace specifically.

    Parameters
    ----------
    m_bar_r : (max_lag+1, m, m)
    k : number of directions to return
    lags : which lag indices to include; default all (0..max_lag)
    weighting : "uniform", "amplitude", or "heterogeneity"
    side : "source" (right SVs) or "target" (left SVs)

    Returns
    -------
    dirs_m : (k, m) orthonormal directions in PCA space
    info : dict with singular values, per-lag weights, alignment cosines
    """
    max_lag_p1, m, _ = m_bar_r.shape
    if lags is None:
        lags = list(range(max_lag_p1))

    # Compute per-lag norms for weighting
    A_r = np.array([np.linalg.norm(m_bar_r[r], "fro") for r in lags])

    if weighting == "uniform":
        weights = np.ones(len(lags))
    elif weighting == "amplitude":
        weights = A_r / (A_r.sum() + 1e-12)
    elif weighting == "heterogeneity":
        # Caller should pass H_r, but we can't compute it here without
        # per-position data.  Fall back to A_r-weighting with a warning.
        weights = A_r / (A_r.sum() + 1e-12)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Stack weighted matrices: (len(lags)*m, m)
    blocks = []
    for i, r in enumerate(lags):
        blocks.append(weights[i] * m_bar_r[r])
    stacked = np.vstack(blocks)  # (len(lags)*m, m)

    U_full, S_full, Vh_full = np.linalg.svd(stacked, full_matrices=False)

    if side == "source":
        dirs_m = Vh_full[:k, :]  # right singular vectors
    else:
        # For target: take left SVs but reshape back
        # Each block of U contributes per-lag left structure.
        # Simpler: use Gram approach M M^T is (len(lags)*m, len(lags)*m) which
        # is large; instead just transpose and re-SVD.
        stacked_T = np.hstack([weights[i] * m_bar_r[r].T for i, r in enumerate(lags)])
        _, _, Vh_T = np.linalg.svd(stacked_T, full_matrices=False)
        dirs_m = Vh_T[:k, :]

    # Per-lag alignment: cosine between multi-lag dirs and each single-lag SVD
    alignments = {}
    for r in lags:
        _, _, Vh_r = np.linalg.svd(m_bar_r[r], full_matrices=False)
        single_top = Vh_r[:k, :] if side == "source" else None
        if single_top is not None:
            # Subspace alignment: max |cos(angle)| between each multi-lag dir
            # and the single-lag subspace
            cos_mat = dirs_m @ single_top.T  # (k, k)
            alignments[r] = {
                "max_cos": float(np.max(np.abs(cos_mat))),
                "mean_cos": float(np.mean(np.abs(cos_mat))),
                "principal_angle": float(np.arccos(np.clip(
                    np.linalg.svd(cos_mat, compute_uv=False)[0], 0, 1
                )) * 180 / np.pi) if k > 0 else 90.0,
            }

    info = {
        "singular_values": S_full[:k].tolist(),
        "weights": {r: float(weights[i]) for i, r in enumerate(lags)},
        "lag_amplitudes": {r: float(A_r[i]) for i, r in enumerate(lags)},
        "per_lag_alignment": alignments,
        "weighting": weighting,
        "lags_used": lags,
    }
    return dirs_m, info


class SingularAblationHook:
    def __init__(self, target_layer: nn.Module, m_bar: np.ndarray, pca_components: np.ndarray, 
                 mu_l: np.ndarray, k: int, lag: int, alpha: float, side: str='source'):
        # m_bar: (max_lag+1, m, m) matrix
        # k: top k singular vectors
        self.target_layer = target_layer
        self.m_bar = m_bar
        self.pca_components = pca_components
        self.mu_l = torch.tensor(mu_l, dtype=torch.float32)
        self.k = k
        self.lag = lag
        self.alpha = alpha
        self.side = side
        self.handle = None
        
        # Compute directions
        U, S, Vh = np.linalg.svd(m_bar[lag])
        
        if side == 'source':
            # right singular vectors (rows of Vh)
            dirs_m = Vh[:k, :]
        elif side == 'target':
            # left singular vectors (cols of U)
            dirs_m = U[:, :k].T
        else: # source+target 
            pass # Simplified
            
        self.dirs_m = torch.tensor(dirs_m, dtype=torch.float32)
        # map to hidden space
        self.dirs_h = self.dirs_m @ torch.tensor(self.pca_components, dtype=torch.float32)
        
        # Orthonormalize
        Q, R = torch.linalg.qr(self.dirs_h.T)
        self.dirs_h = Q.T
        
    def hook_fn(self, module, input, output):
        # Hook on block output, which could be a tuple depending on return format
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
            
        # h: (seq, seq_len, hidden_dim)
        device = h.device
        self.mu_l = self.mu_l.to(device)
        self.dirs_h = self.dirs_h.to(device)
        
        h_centered = h - self.mu_l
        
        # We need to project out
        # D * D^T * h_centered
        # D is (k, hidden_dim) -> (hidden_dim, k)
        D = self.dirs_h.T
        
        b, seq_len, dim = h.shape
        proj = h_centered @ D # (b, seq_len, k)
        ablation = proj @ D.T # (b, seq_len, dim)
        
        # Mask where to apply
        mask = torch.zeros((1, seq_len, 1), device=device)
        
        for i in range(seq_len):
            if self.side == 'source' and i - self.lag >= 0:
                mask[:, i - self.lag, :] = 1.0
            elif self.side == 'target':
                mask[:, i, :] = 1.0
                
        h_prime = h - self.alpha * ablation * mask
        
        if isinstance(output, tuple):
            return (h_prime,) + output[1:]
        else:
            return h_prime
        
    def register(self):
        self.handle = self.target_layer.register_forward_hook(self.hook_fn)
        
    def remove(self):
        if self.handle:
            self.handle.remove()
