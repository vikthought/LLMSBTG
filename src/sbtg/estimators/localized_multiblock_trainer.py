import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

class MinimalMLPScoreNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_features)
        )
        
    def forward(self, x, t=None):
        return self.net(x)

def extract_windows(data: np.ndarray, window_size: int):
    # data: (N_seqs, seq_len, m) — dtype preserved from input
    # returns: (N_seqs, seq_len - window_size + 1, window_size * m)
    #
    # Memory note: this materializes the full (N, num_windows, w*m) array.
    # At long context (seq_len ≥ 200) the array can exceed RAM.  Callers that
    # might hit that regime should subsample sequences in `data` first (see
    # the memory-cap logic in scripts/run_lagpair_analysis.py).  We preserve
    # the input dtype so float32 inputs stay float32 (half the memory of the
    # historical float64 default).
    N_seqs, seq_len, m = data.shape
    num_windows = seq_len - window_size + 1

    windows = np.zeros((N_seqs, num_windows, window_size * m), dtype=data.dtype)
    for i in range(num_windows):
        windows[:, i, :] = data[:, i:i+window_size, :].reshape(N_seqs, -1)

    return windows

def train_score_model_layer(train_windows: np.ndarray, val_windows: np.ndarray,
                            m: int, w: int, epochs: int=50, lr: float=1e-3,
                            sigma: float = 0.1, hidden_dim: int = 256,
                            batch_size: int=256, device="cuda"):
    # train_windows: (N, num_windows, mw)
    train_windows_flat = train_windows.reshape(-1, m*w)
    val_windows_flat = val_windows.reshape(-1, m*w)

    train_tensor = torch.tensor(train_windows_flat, dtype=torch.float32)
    val_tensor = torch.tensor(val_windows_flat, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=batch_size)

    model = MinimalMLPScoreNet(m*w, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            noise = torch.randn_like(batch) * sigma
            noisy_batch = batch + noise
            
            # Predict noise instead of pure score 
            # score = -noise / sigma^2
            pred_score = model(noisy_batch)
            target_score = -noise / (sigma ** 2)
            
            loss = torch.mean(torch.sum((pred_score - target_score)**2, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                noise = torch.randn_like(batch) * sigma
                noisy_batch = batch + noise
                pred_score = model(noisy_batch)
                target_score = -noise / (sigma ** 2)
                loss = torch.mean(torch.sum((pred_score - target_score)**2, dim=1))
                val_loss += loss.item()
                
        # print(f"Epoch {epoch+1} Layer train_loss: {train_loss/len(train_loader):.4f} val_loss: {val_loss/len(val_loader):.4f}")

    return model


def tune_score_model_hyperparams(
    train_windows_flat: np.ndarray,
    val_windows_flat: np.ndarray,
    in_features: int,
    n_trials: int = 70,
    tune_epochs: int = 10,
    max_tune_samples: int = 50_000,
    device: str = "cuda",
    seed: int = 42,
    sigma: float = 0.3,
) -> dict:
    """
    Optuna hyperparameter search for the DSM score model.

    Searches over ``hidden_dim`` and ``lr``.  The noise level ``sigma`` is
    fixed (default 0.3) to preserve sensitivity to position-dependent
    structure in the score-geometric diagnostics.

    Returns
    -------
    dict with keys ``sigma``, ``hidden_dim``, ``lr`` — ready to pass to
    ``train_score_model_layer``.
    """
    defaults = {"sigma": sigma, "hidden_dim": 256, "lr": 1e-3}
    if not _HAS_OPTUNA:
        print("  [tune] optuna not installed — using defaults")
        return defaults

    # Sub-sample so each trial is fast
    rng = np.random.default_rng(seed)
    def _subsample(arr, n):
        if len(arr) <= n:
            return arr
        idx = rng.choice(len(arr), n, replace=False)
        return arr[idx]

    t_flat = _subsample(train_windows_flat, max_tune_samples)
    v_flat = _subsample(val_windows_flat, min(10_000, len(val_windows_flat)))

    train_tensor = torch.tensor(t_flat, dtype=torch.float32)
    val_tensor   = torch.tensor(v_flat, dtype=torch.float32)

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)

        model = MinimalMLPScoreNet(in_features, hidden_dim=hidden_dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        loader = torch.utils.data.DataLoader(train_tensor, batch_size=256, shuffle=True)
        model.train()
        for _ in range(tune_epochs):
            for batch in loader:
                batch = batch.to(device)
                noise = torch.randn_like(batch) * sigma
                pred  = model(batch + noise)
                target = -noise / (sigma ** 2)
                loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=256)
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                noise = torch.randn_like(batch) * sigma
                pred  = model(batch + noise)
                target = -noise / (sigma ** 2)
                val_loss += torch.mean(torch.sum((pred - target) ** 2, dim=1)).item()
                n += 1
        return val_loss / max(1, n)

    n_startup = max(10, n_trials // 3)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    print(f"  [tune] best: sigma={sigma:.3f} (fixed)  hidden={bp['hidden_dim']}  "
          f"lr={bp['lr']:.2e}  val_loss={study.best_value:.4f}")
    return {"sigma": sigma, "hidden_dim": bp["hidden_dim"], "lr": bp["lr"]}


def tune_score_model_null_contrast(
    train_windows_flat: np.ndarray,
    val_windows_flat: np.ndarray,
    in_features: int,
    m: int,
    w: int,
    n_trials: int = 150,
    tune_epochs: int = 10,
    max_tune_samples: int = 50_000,
    n_shuffles: int = 5,
    sigma_range: tuple = (0.2, 0.4),
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Optuna HP search using **null-contrast** as the objective.

    Instead of minimizing val DSM loss, this maximises the null-contrast
    ratio (NC): the mean absolute cross-block score product under true
    temporal alignment divided by the same quantity under random
    permutation of the past block.  NC >> 1 means the score model
    captures genuine temporal/positional structure; NC ~ 1 means it
    learned nothing beyond independent marginals.

    Searches over ``sigma`` (noise level), ``hidden_dim``, and ``lr``.
    Trial 0 always evaluates the known-good defaults (sigma=0.3,
    hidden_dim=256, lr=1e-3) so the search starts from a reasonable
    baseline.

    Parameters
    ----------
    m : int
        PCA dim (block size in each time step).
    w : int
        Window size (number of time steps per window).
    n_shuffles : int
        Number of temporal shuffles for null estimate (default 5).
    sigma_range : tuple
        (min, max) for noise_std search (log-uniform). Default (0.2, 0.4).

    Returns
    -------
    dict with keys ``sigma``, ``hidden_dim``, ``lr``, ``best_nc``.
    """
    defaults = {"sigma": 0.3, "hidden_dim": 256, "lr": 1e-3}
    if not _HAS_OPTUNA:
        print("  [tune-NC] optuna not installed — using defaults")
        return defaults

    rng = np.random.default_rng(seed)

    def _subsample(arr, n):
        if len(arr) <= n:
            return arr
        idx = rng.choice(len(arr), n, replace=False)
        return arr[idx]

    t_flat = _subsample(train_windows_flat, max_tune_samples)
    v_flat = _subsample(val_windows_flat, min(10_000, len(val_windows_flat)))

    train_tensor = torch.tensor(t_flat, dtype=torch.float32)
    val_tensor   = torch.tensor(v_flat, dtype=torch.float32)

    def objective(trial):
        sigma      = trial.suggest_float("sigma", *sigma_range, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = MinimalMLPScoreNet(in_features, hidden_dim=hidden_dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        loader = torch.utils.data.DataLoader(train_tensor, batch_size=256, shuffle=True)
        model.train()
        for _ in range(tune_epochs):
            for batch in loader:
                batch = batch.to(device)
                noise = torch.randn_like(batch) * sigma
                pred  = model(batch + noise)
                target = -noise / (sigma ** 2)
                loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))
                opt.zero_grad()
                loss.backward()
                opt.step()

        # --- Null-contrast evaluation on val set ---
        model.eval()
        scores_parts = []
        with torch.no_grad():
            for i in range(0, len(val_tensor), 512):
                batch = val_tensor[i:i + 512].to(device)
                scores_parts.append(model(batch).cpu().numpy())
        scores_flat = np.concatenate(scores_parts, axis=0)

        # Window layout: [block_0 | block_1 | ... | block_{w-1}]
        # Each block has m dims.  Future = block_{w-1}, past = block_0.
        scores_blocks = scores_flat.reshape(-1, w, m)
        s_future = scores_blocks[:, -1, :]   # (N_windows, m)
        s_past   = scores_blocks[:,  0, :]   # (N_windows, m)

        N_w = s_future.shape[0]
        mu_hat = (s_future.T @ s_past) / N_w      # (m, m)
        mask = ~np.eye(m, dtype=bool)
        real_mean = np.abs(mu_hat[mask]).mean()

        shuffle_rng = np.random.default_rng(seed + trial.number)
        null_means = []
        for _ in range(n_shuffles):
            perm = shuffle_rng.permutation(N_w)
            mu_null = (s_future.T @ s_past[perm]) / N_w
            null_means.append(np.abs(mu_null[mask]).mean())
        null_mean = float(np.mean(null_means))

        if null_mean < 1e-10:
            return 1.0
        return real_mean / null_mean

    n_startup = max(10, n_trials // 3)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed with known-good defaults so trial 0 establishes a baseline
    study.enqueue_trial({"sigma": 0.3, "hidden_dim": 256, "lr": 1e-3})

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best_nc = study.best_value
    print(f"  [tune-NC] best: sigma={bp['sigma']:.3f}  hidden={bp['hidden_dim']}  "
          f"lr={bp['lr']:.2e}  NC={best_nc:.4f}")
    return {"sigma": bp["sigma"], "hidden_dim": bp["hidden_dim"], "lr": bp["lr"],
            "best_nc": best_nc}
