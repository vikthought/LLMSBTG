"""
Hyperparameter tuning for SBTG estimators via Optuna.

The tuning loop uses the TPE (Tree-structured Parzen Estimator)
sampler from Optuna and optimises the null-contrast (NC) objective:

    NC = mean(|mu_real|) / mean(|mu_null|)

The search space covers:
- ``noise_std`` : DSM corruption level (log-uniform).
- ``hidden_dim`` : MLP width (categorical).
- ``num_layers`` : MLP depth (categorical, 2--4).
- ``lr`` : Adam learning rate (log-uniform).

For each trial, a K-fold cross-fitted null contrast is computed with
reduced training epochs for speed.  The best configuration is returned
as an ``HPConfig`` dataclass.

Usage
-----
::

    from sbtg.estimators.tuning import tune_hyperparameters

    config = tune_hyperparameters(X_list, lag=2, n_trials=50, device="cuda")
    print(config.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sbtg.core.score_nets import TwoBlockScoreNet, MultiBlockScoreNet
from sbtg.core.dsm import train_score_model, compute_scores
from sbtg.core.windows import build_two_block_windows, build_minimal_multiblock_windows
from sbtg.core.inference import standardize_windows, create_fold_assignments
from sbtg.core.null_contrast import compute_null_contrast_from_scores


@dataclass
class HPConfig:
    """Hyperparameter configuration found by tuning."""

    noise_std: float = 0.1
    hidden_dim: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 100

    def to_dict(self) -> dict:
        return {
            "noise_std": self.noise_std,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "epochs": self.epochs,
        }


def tune_hyperparameters(
    X_list: List[np.ndarray],
    lag: int = 1,
    *,
    n_trials: int = 20,
    noise_std_range: Tuple[float, float] = (0.01, 0.5),
    hidden_dim_choices: Optional[List[int]] = None,
    lr_range: Tuple[float, float] = (5e-5, 5e-2),
    epochs_for_tuning: int = 20,
    n_folds: int = 3,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> HPConfig:
    """
    Tune HP for a single lag using Optuna TPE sampler + null-contrast.

    Parameters
    ----------
    X_list : list of (T, n) arrays
    lag : int
    n_trials : int
        Total Optuna trials (first half random, second half Bayesian).
    noise_std_range, lr_range : 2-tuples
    hidden_dim_choices : list[int]
    epochs_for_tuning : int
        Reduced epochs for faster evaluation.
    n_folds : int
    device, verbose, seed : standard args

    Returns
    -------
    HPConfig with the best parameters found.
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna is required for HP tuning: pip install optuna")

    if hidden_dim_choices is None:
        hidden_dim_choices = [32, 64, 128, 256]

    n_neurons = X_list[0].shape[1]
    n_blocks = lag + 1

    try:
        if n_blocks == 2:
            windows, stim_ids, _ = build_two_block_windows(X_list, lag)
        else:
            windows, stim_ids, _ = build_minimal_multiblock_windows(X_list, lag)
    except ValueError as e:
        if "No windows could be built" in str(e):
            if verbose:
                print(f"  Skipping HP tuning (lag={lag}): {str(e)}. Using defaults.")
            return HPConfig()
        raise e

    if verbose:
        print(f"  HP tuning: {n_trials} trials, {len(windows)} windows, lag={lag}")

    def objective(trial: optuna.Trial) -> float:
        ns = trial.suggest_float("noise_std", *noise_std_range)
        hd = trial.suggest_categorical("hidden_dim", hidden_dim_choices)
        lr = trial.suggest_float("lr", *lr_range, log=True)
        nl = trial.suggest_categorical("num_layers", [2, 3, 4])

        fold_ids = create_fold_assignments(stim_ids, n_folds)
        all_sf, all_sp = [], []

        for k in range(n_folds):
            train_mask = fold_ids != k
            heldout_mask = fold_ids == k
            w_std, _, _ = standardize_windows(windows, train_mask)

            if n_blocks == 2:
                model = TwoBlockScoreNet(n_neurons, hidden_dim=hd, num_layers=nl)
            else:
                model = MultiBlockScoreNet(n_neurons, p_max=lag, hidden_dim=hd, num_layers=nl)

            try:
                train_score_model(
                    model, w_std[train_mask],
                    noise_std=ns, lr=lr, epochs=epochs_for_tuning,
                    batch_size=128, device=device, verbose=False,
                )
            except Exception:
                return float("-inf")

            ho_scores = compute_scores(model, w_std[heldout_mask], device=device)
            if n_blocks == 2:
                all_sp.append(ho_scores[:, :n_neurons])
                all_sf.append(ho_scores[:, n_neurons:])
            else:
                reshaped = ho_scores.reshape(-1, n_blocks, n_neurons)
                all_sp.append(reshaped[:, 0, :])
                all_sf.append(reshaped[:, -1, :])

        sf = np.concatenate(all_sf)
        sp = np.concatenate(all_sp)
        nc, _ = compute_null_contrast_from_scores(sf, sp)
        return nc if np.isfinite(nc) else float("-inf")

    n_startup = min(n_trials // 2, 20)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _log(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if verbose:
            if trial.value is not None and trial.value > float("-inf"):
                tag = " *" if trial.value >= study.best_value else ""
                p = trial.params
                print(f"    Trial {trial.number + 1}/{n_trials} | NC={trial.value:.4f}{tag} | "
                      f"hidden={p.get('hidden_dim')}, num_layers={p.get('num_layers')}, "
                      f"noise={p.get('noise_std', 0.0):.3f}, lr={p.get('lr', 0.0):.1e}")
            else:
                print(f"    Trial {trial.number + 1}/{n_trials} | FAILED")

    study.optimize(objective, n_trials=n_trials, callbacks=[_log], show_progress_bar=False)

    bp = study.best_params
    best = HPConfig(
        noise_std=bp["noise_std"],
        hidden_dim=bp["hidden_dim"],
        num_layers=bp.get("num_layers", 2),
        lr=bp["lr"],
        epochs=epochs_for_tuning,
    )
    if verbose:
        print(f"  Best: noise_std={best.noise_std:.3f}  hidden={best.hidden_dim}  "
              f"lr={best.lr:.2e}  NC={study.best_value:.3f}")
    return best
