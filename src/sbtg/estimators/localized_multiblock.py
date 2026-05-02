import numpy as np

class LocalizedMultiBlockEstimator:
    def __init__(self, m: int, w: int, max_lag: int, skip_edges: int = 4):
        self.m = m
        self.w = w
        self.max_lag = max_lag
        self.skip_edges = skip_edges
        
    def estimate(self, model, test_windows_flat: np.ndarray, device="cuda"):
        # test_windows_flat: (num_test_seqs, num_windows, m*w)
        # model: Trained score model
        num_seqs, num_windows, mw = test_windows_flat.shape
        model.eval()
        
        # We need the scores!
        import torch
        with torch.no_grad():
            tensor_windows = torch.tensor(test_windows_flat.reshape(-1, mw), dtype=torch.float32).to(device)
            # Batch inference
            scores = []
            batch_size = 512
            for i in range(0, len(tensor_windows), batch_size):
                batch = tensor_windows[i:i+batch_size]
                scores.append(model(batch).cpu().numpy())
            scores = np.concatenate(scores, axis=0) # (num_seqs * num_windows, m*w)
            
        scores = scores.reshape(num_seqs, num_windows, self.w, self.m)
        
        M_r_i = np.zeros((self.max_lag+1, num_windows, self.m, self.m))
        
        # M_r^(l)(i) = -E [s_w, s_{w-r}^T]
        for lag in range(self.max_lag + 1):
            if self.w - lag - 1 >= 0:
                s_current = scores[:, :, -1, :] # size (N, num_windows, m)
                s_lagged = scores[:, :, self.w - lag - 1, :] # size (N, num_windows, m)
                
                # Average over sequences E_p
                # Resulting shape: (num_windows, m, m)
                M = -np.einsum('nwi,nwj->wij', s_current, s_lagged) / num_seqs
                M_r_i[lag] = M
                
        # Decomposition
        # Skip boundaries
        valid_windows = slice(self.skip_edges, num_windows - self.skip_edges)
        
        M_r_i_valid = M_r_i[:, valid_windows, :, :]
        M_bar_r = np.mean(M_r_i_valid, axis=1) # (max_lag+1, m, m)
        Delta_r_i = M_r_i_valid - M_bar_r[:, None, :, :]
        
        # Summaries
        A_r = np.linalg.norm(M_bar_r, ord='fro', axis=(1,2))
        
        denom = A_r**2 + 1e-8
        H_r = np.mean(np.linalg.norm(Delta_r_i, ord='fro', axis=(2,3))**2, axis=1) / denom
        
        num_SI = np.sum(np.mean(np.linalg.norm(Delta_r_i, ord='fro', axis=(2,3))**2, axis=1))
        den_SI = np.sum(np.mean(np.linalg.norm(M_r_i_valid, ord='fro', axis=(2,3))**2, axis=1)) + 1e-8
        SI = 1.0 - (num_SI / den_SI)
        
        # Recency slope Beta
        log_A = np.log(A_r[1:] + 1e-8)
        lags = np.arange(1, self.max_lag + 1)
        beta = np.polyfit(lags, log_A, 1)[0] if self.max_lag > 0 else 0
        
        # Spectral Concentration Ratio: how much of each lag operator's singular-value
        # mass is concentrated in the top-k directions.
        # SCR[r, k-1] = sum(sigma[:k]) / sum(sigma) for lag r.
        SCR_r = np.zeros((self.max_lag + 1, self.m))
        for r in range(self.max_lag + 1):
            mat = M_bar_r[r]
            if not np.all(np.isfinite(mat)):
                mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                _, sv, _ = np.linalg.svd(mat, full_matrices=False)
            except np.linalg.LinAlgError:
                sv = np.zeros(self.m)
            total = sv.sum() + 1e-8
            SCR_r[r] = np.cumsum(sv) / total

        # Per-position per-lag Frobenius norms — small arrays, enable F4 heatmap.
        # delta_frob[r, i] = ||Δ_r^(ℓ)(i)||_F   shape (max_lag+1, num_valid_windows)
        # M_frob[r, i]     = ||M_r^(ℓ)(i)||_F    same shape (context for normalisation)
        delta_frob = np.linalg.norm(Delta_r_i, ord='fro', axis=(2, 3))
        M_frob     = np.linalg.norm(M_r_i_valid, ord='fro', axis=(2, 3))

        return {
            'A_r':        A_r,
            'H_r':        H_r,
            'SI':         SI,
            'RDI':        SI,  # backward compat alias
            'beta':       beta,
            'M_bar_r':    M_bar_r,
            'M_r_i':      M_r_i,
            'SCR_r':      SCR_r,       # (max_lag+1, m)  cumulative sv fraction
            'delta_frob': delta_frob,  # (max_lag+1, num_valid_windows)
            'M_frob':     M_frob,      # (max_lag+1, num_valid_windows)
        }

    @staticmethod
    def compute_operator_autocorrelation(M_r_i, max_shift=None):
        """Measure how M_r(i) changes with position shift delta.

        For each lag r and shift delta, computes:
            C(r, delta) = E_i[ ||M_r(i) - M_r(i+delta)||_F ] / E_i[ ||M_r(i)||_F ]

        This is the operational definition of (non-)stationarity: if C(r, delta)
        is small for all delta, the operator is translation-invariant at lag r.

        Parameters
        ----------
        M_r_i : (max_lag+1, num_windows, m, m)
            Per-position operator matrices (from estimate()).
        max_shift : int, optional
            Maximum shift to compute. Default: num_windows // 2.

        Returns
        -------
        autocorr : (max_lag+1, max_shift+1) — normalized distance vs shift.
            autocorr[r, 0] = 0 by construction.
        """
        n_lags, n_win, m1, m2 = M_r_i.shape
        if max_shift is None:
            max_shift = n_win // 2

        autocorr = np.zeros((n_lags, max_shift + 1))

        for r in range(n_lags):
            # Denominator: mean Frobenius norm at this lag
            norms = np.linalg.norm(M_r_i[r], ord='fro', axis=(1, 2))  # (n_win,)
            denom = norms.mean() + 1e-8

            for delta in range(1, max_shift + 1):
                # ||M_r(i) - M_r(i+delta)||_F for valid i
                diff = M_r_i[r, :n_win - delta] - M_r_i[r, delta:]
                diff_norms = np.linalg.norm(diff, ord='fro', axis=(1, 2))
                autocorr[r, delta] = diff_norms.mean() / denom

        return autocorr
