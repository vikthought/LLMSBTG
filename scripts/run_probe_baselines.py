"""
Probe baseline suite — B1 (OVERHAUL §B1a/B1b/B1c).

Fits four probe variants per layer per PE family and saves per-layer
accuracy + weight directions (used by run_ablation_study.py for B2).

Probe variants
--------------
  linear_abs    Linear LR predicting absolute token position (existing baseline,
                replicated here consistently)
  mlp_abs       2-layer MLP predicting absolute token position (B1c)
  rel_dist      Linear LR predicting bucketed |i-j| from concat(h_i, h_j) (B1a)
  near_far      Binary logistic regression: near (|i-j|≤4) vs far (|i-j|≥17) (B1b)

Output
------
  <out-dir>/probe_baselines_summary.json
      {
        "rope": {
          "linear_abs":  {"mean": [acc_l1, ...], "std": [...], "directions": [[w_l1], ...]},
          "mlp_abs":     {"mean": [...], "std": [...]},
          "rel_dist":    {"mean": [...], "std": [...]},
          "near_far":    {"mean": [...], "std": [...]},
        }, ...
      }
  The "directions" key (unit-norm vector in PCA space, per layer) is written
  for linear_abs and near_far; these feed into run_ablation_study.py --ablation-type probe_*.

Usage
-----
python scripts/run_probe_baselines.py \\
    --analysis-dir results/<run>/analysis \\
    --models-dir   results/<run>/models \\
    --out-dir      results/<run>/probes \\
    --pe-types rope alibi absolute \\
    --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# PCA helpers (duplicated from run_positional_analysis.py to keep standalone)
# ---------------------------------------------------------------------------

def _transform_pca(components: np.ndarray, mu: np.ndarray, acts_l: np.ndarray) -> np.ndarray:
    """acts_l: (N, seq_len, H) → (N, seq_len, m)"""
    N, seq_len, H = acts_l.shape
    flat = acts_l.reshape(-1, H) - mu[None, :]
    return flat @ components.T


# ---------------------------------------------------------------------------
# MLP probe (B1c)
# ---------------------------------------------------------------------------

class _MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    n_classes: int,
    hidden: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> float:
    model = _MLPProbe(X_train.shape[1], hidden, n_classes)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    ds  = torch.utils.data.TensorDataset(X_t, y_t)
    dl  = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        preds  = logits.argmax(dim=1).numpy()
    return float((preds == y_test).mean())


# ---------------------------------------------------------------------------
# Pair sampling helpers for relative probes
# ---------------------------------------------------------------------------

DIST_BINS = [1, 2, 3, 4, 8, 16]  # edges: [1], [2], [3], [4], [5-8], [9-16], [17+]


def _bucket_dist(d: int) -> int:
    for i, edge in enumerate(DIST_BINS):
        if d <= edge:
            return i
    return len(DIST_BINS)  # 17+


def _sample_pairs(
    acts: np.ndarray,        # (N, seq_len, m) — already PCA-projected
    n_pairs_per_seq: int = 50,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample random (i, j) pairs with j < i from each sequence.
    Returns X: (total_pairs, 2m), dists: (total_pairs,), near_far: (total_pairs,)
    where near_far encodes {0=near: |i-j|<=4, 1=far: |i-j|>=17, -1=exclude}.
    """
    N, seq_len, m = acts.shape
    rng = np.random.default_rng(rng_seed)

    X_list, dist_list, nf_list = [], [], []

    positions = np.arange(seq_len)
    for n in range(N):
        # All valid (i, j) with j < i
        pairs_i = []
        pairs_j = []
        for i in range(1, seq_len):
            for j in range(i):
                pairs_i.append(i)
                pairs_j.append(j)
        pairs_i = np.array(pairs_i)
        pairs_j = np.array(pairs_j)

        # Sub-sample
        total = len(pairs_i)
        idx   = rng.choice(total, min(n_pairs_per_seq, total), replace=False)
        pi, pj = pairs_i[idx], pairs_j[idx]

        feat = np.concatenate([acts[n, pi, :], acts[n, pj, :]], axis=1)  # (k, 2m)
        d    = pi - pj  # always > 0

        nf = np.where(d <= 4, 0, np.where(d >= 17, 1, -1))

        X_list.append(feat)
        dist_list.append(d)
        nf_list.append(nf)

    X     = np.concatenate(X_list, axis=0)
    dists = np.concatenate(dist_list, axis=0)
    nf    = np.concatenate(nf_list, axis=0)
    return X, dists, nf


# ---------------------------------------------------------------------------
# Per-layer probe fitting
# ---------------------------------------------------------------------------

def _probe_layer(
    train_acts_l: np.ndarray,  # (N_train, seq_len, m)
    test_acts_l:  np.ndarray,  # (N_test,  seq_len, m)
    seq_len: int,
    max_train_samples: int = 40_000,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Fit all four probe variants for one layer.
    Returns dict with accuracy and (where applicable) probe direction.
    """
    N_train, _, m = train_acts_l.shape
    N_test        = test_acts_l.shape[0]
    rng = np.random.default_rng(rng_seed)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _subsample(X, y, n):
        if len(X) <= n:
            return X, y
        idx = rng.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    def _scale(X_tr, X_te):
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)

    # ── Linear absolute position probe (B1 / existing baseline) ─────────────
    X_tr = train_acts_l.reshape(-1, m)
    y_tr = np.tile(np.arange(seq_len), N_train)
    X_te = test_acts_l.reshape(-1, m)
    y_te = np.tile(np.arange(seq_len), N_test)

    X_tr_sub, y_tr_sub = _subsample(X_tr, y_tr, max_train_samples)
    _sc      = StandardScaler()
    X_tr_sub = _sc.fit_transform(X_tr_sub)
    X_te_s   = _sc.transform(X_te)

    probe_lin = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe_lin.fit(X_tr_sub, y_tr_sub)
    acc_lin = float(probe_lin.score(X_te_s, y_te))

    # Top PC of weight matrix as probe direction (for B2 ablation)
    W = probe_lin.coef_    # (n_classes, m)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    dir_abs = Vt[0]           # (m,) unit-norm top direction

    # ── Nonlinear (MLP) absolute position probe (B1c) ────────────────────────
    # Reuse the already-standardised arrays
    acc_mlp = _train_mlp_probe(
        X_tr_sub, y_tr_sub, X_te_s, y_te,
        n_classes=seq_len, hidden=64, epochs=30,
    )

    # ── Relative-distance probe (B1a) ────────────────────────────────────────
    X_pairs_tr, d_tr, _ = _sample_pairs(train_acts_l, n_pairs_per_seq=50, rng_seed=rng_seed)
    X_pairs_te, d_te, _ = _sample_pairs(test_acts_l,  n_pairs_per_seq=50, rng_seed=rng_seed + 1)
    y_bucket_tr = np.array([_bucket_dist(d) for d in d_tr])
    y_bucket_te = np.array([_bucket_dist(d) for d in d_te])

    sc_rel = StandardScaler()
    X_pairs_tr_s = sc_rel.fit_transform(X_pairs_tr)
    X_pairs_te_s = sc_rel.transform(X_pairs_te)
    if len(X_pairs_tr_s) > max_train_samples:
        idx_r = rng.choice(len(X_pairs_tr_s), max_train_samples, replace=False)
        X_pairs_tr_s = X_pairs_tr_s[idx_r]
        y_bucket_tr  = y_bucket_tr[idx_r]

    probe_rel = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe_rel.fit(X_pairs_tr_s, y_bucket_tr)
    acc_rel = float(probe_rel.score(X_pairs_te_s, y_bucket_te))

    # ── Near/far binary probe (B1b) ───────────────────────────────────────────
    _, _, nf_tr = _sample_pairs(train_acts_l, n_pairs_per_seq=50, rng_seed=rng_seed + 2)
    _, _, nf_te = _sample_pairs(test_acts_l,  n_pairs_per_seq=50, rng_seed=rng_seed + 3)

    # Recompute pair features for consistency with the nf labels
    X_nf_tr, _, nf_tr = _sample_pairs(train_acts_l, n_pairs_per_seq=50, rng_seed=rng_seed + 4)
    X_nf_te, _, nf_te = _sample_pairs(test_acts_l,  n_pairs_per_seq=50, rng_seed=rng_seed + 5)
    mask_tr = nf_tr != -1
    mask_te = nf_te != -1

    acc_nf = float("nan")
    dir_nf = np.zeros(m * 2)
    if mask_tr.sum() >= 20 and mask_te.sum() >= 5:
        sc_nf = StandardScaler()
        Xnf_tr_s = sc_nf.fit_transform(X_nf_tr[mask_tr])
        Xnf_te_s = sc_nf.transform(X_nf_te[mask_te])
        y_nf_tr  = nf_tr[mask_tr]
        y_nf_te  = nf_te[mask_te]
        if len(Xnf_tr_s) > max_train_samples:
            idx_n = rng.choice(len(Xnf_tr_s), max_train_samples, replace=False)
            Xnf_tr_s = Xnf_tr_s[idx_n]
            y_nf_tr  = y_nf_tr[idx_n]
        probe_nf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", n_jobs=-1)
        probe_nf.fit(Xnf_tr_s, y_nf_tr)
        acc_nf = float(probe_nf.score(Xnf_te_s, y_nf_te))
        dir_nf = probe_nf.coef_[0]  # (2m,) — binary probe has single weight row
        nrm = np.linalg.norm(dir_nf) + 1e-12
        dir_nf = dir_nf / nrm

    return {
        "linear_abs": {"acc": acc_lin, "direction_pca": dir_abs.tolist()},
        "mlp_abs":    {"acc": acc_mlp},
        "rel_dist":   {"acc": acc_rel},
        "near_far":   {"acc": acc_nf, "direction_pca_concat": dir_nf.tolist()},
    }


def _probe_layer_hidden(
    train_acts_l: np.ndarray,  # (N_train, seq_len, hidden_size) — raw, no PCA
    test_acts_l:  np.ndarray,  # (N_test,  seq_len, hidden_size)
    seq_len: int,
    top_k: int = 3,
    max_train_samples: int = 40_000,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Fit a linear position probe directly in hidden space (no PCA).

    Returns accuracy and top-k SVD directions of the probe weight matrix
    in full hidden space — these can be used for PCA-free ablation.
    """
    N_train, _, hidden_size = train_acts_l.shape
    N_test = test_acts_l.shape[0]
    rng = np.random.default_rng(rng_seed)

    X_tr = train_acts_l.reshape(-1, hidden_size)
    y_tr = np.tile(np.arange(seq_len), N_train)
    X_te = test_acts_l.reshape(-1, hidden_size)
    y_te = np.tile(np.arange(seq_len), N_test)

    if len(X_tr) > max_train_samples:
        idx = rng.choice(len(X_tr), max_train_samples, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    probe = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe.fit(X_tr, y_tr)
    acc = float(probe.score(X_te, y_te))

    # Top-k SVD directions of weight matrix W: (n_classes, hidden_size)
    W = probe.coef_
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    dirs_hidden = Vt[:top_k]  # (top_k, hidden_size) — already in hidden space

    return {
        "acc": acc,
        "directions_hidden": dirs_hidden.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit probe baseline suite (B1a/b/c) for all PE types and layers."
    )
    parser.add_argument("--analysis-dir", type=str, required=True,
                        help="Directory containing <pe>_seed<N>_analysis.json files")
    parser.add_argument("--models-dir",   type=str, required=True)
    parser.add_argument("--out-dir",      type=str, required=True)
    parser.add_argument("--pe-types",     nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds",        nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    models_dir   = Path(args.models_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate: pe → probe_type → [per-layer acc per seed]
    results: Dict[str, Dict[str, List]] = {}

    for pe in args.pe_types:
        results[pe] = {
            "linear_abs": {"accs_by_seed": [], "directions_by_seed": []},
            "linear_hidden": {"accs_by_seed": [], "directions_by_seed": []},
            "mlp_abs":    {"accs_by_seed": []},
            "rel_dist":   {"accs_by_seed": []},
            "near_far":   {"accs_by_seed": [], "directions_by_seed": []},
        }

        for seed in args.seeds:
            seed_dir      = models_dir   / f"{pe}_seed{seed}"
            analysis_file = analysis_dir / f"{pe}_seed{seed}_analysis.json"

            if not seed_dir.exists() or not analysis_file.exists():
                print(f"  Skipping {pe} seed {seed}: missing files.")
                continue

            print(f"\n  {pe.upper()} seed {seed}")

            train_acts = np.load(seed_dir / "train_acts.npy")
            test_acts  = np.load(seed_dir / "test_acts.npy")

            with open(analysis_file) as f:
                analysis = json.load(f)

            n_layers_plus_one = train_acts.shape[1]
            seq_len           = train_acts.shape[2]

            layer_lin_accs, layer_lin_dirs = [], []
            layer_hidden_accs, layer_hidden_dirs = [], []
            layer_mlp_accs  = []
            layer_rel_accs  = []
            layer_nf_accs, layer_nf_dirs = [], []

            for l_idx, ls in enumerate(analysis["layer_stats"]):
                l = ls["layer"]
                print(f"    layer {l} …", end=" ", flush=True)

                # PCA project using saved components
                components = np.array(ls["pca_components"])  # (m, H)
                mu_l       = np.array(ls["mu_l"])            # (H,)
                m          = components.shape[0]

                train_pca = _transform_pca(components, mu_l, train_acts[:, l]).reshape(
                    train_acts.shape[0], seq_len, m)
                test_pca  = _transform_pca(components, mu_l, test_acts[:, l]).reshape(
                    test_acts.shape[0], seq_len, m)

                layer_result = _probe_layer(
                    train_pca, test_pca, seq_len,
                    rng_seed=seed * 100 + l_idx,
                )

                # Hidden-space probe (no PCA bottleneck)
                hidden_result = _probe_layer_hidden(
                    train_acts[:, l], test_acts[:, l], seq_len,
                    top_k=3, rng_seed=seed * 100 + l_idx + 50,
                )

                print(f"lin={layer_result['linear_abs']['acc']:.3f}  "
                      f"hidden={hidden_result['acc']:.3f}  "
                      f"mlp={layer_result['mlp_abs']['acc']:.3f}  "
                      f"rel={layer_result['rel_dist']['acc']:.3f}  "
                      f"nf={layer_result['near_far']['acc']:.3f}")

                layer_lin_accs.append(layer_result["linear_abs"]["acc"])
                layer_lin_dirs.append(layer_result["linear_abs"]["direction_pca"])
                layer_hidden_accs.append(hidden_result["acc"])
                layer_hidden_dirs.append(hidden_result["directions_hidden"])
                layer_mlp_accs.append(layer_result["mlp_abs"]["acc"])
                layer_rel_accs.append(layer_result["rel_dist"]["acc"])
                layer_nf_accs.append(layer_result["near_far"]["acc"])
                layer_nf_dirs.append(layer_result["near_far"]["direction_pca_concat"])

            results[pe]["linear_abs"]["accs_by_seed"].append(layer_lin_accs)
            results[pe]["linear_abs"]["directions_by_seed"].append(layer_lin_dirs)
            results[pe]["linear_hidden"]["accs_by_seed"].append(layer_hidden_accs)
            results[pe]["linear_hidden"]["directions_by_seed"].append(layer_hidden_dirs)
            results[pe]["mlp_abs"]["accs_by_seed"].append(layer_mlp_accs)
            results[pe]["rel_dist"]["accs_by_seed"].append(layer_rel_accs)
            results[pe]["near_far"]["accs_by_seed"].append(layer_nf_accs)
            results[pe]["near_far"]["directions_by_seed"].append(layer_nf_dirs)

    # Aggregate across seeds: mean ± std per layer
    summary: Dict[str, Any] = {}
    for pe in args.pe_types:
        summary[pe] = {}
        for probe_type in ("linear_abs", "linear_hidden", "mlp_abs", "rel_dist", "near_far"):
            accs = np.array(results[pe][probe_type]["accs_by_seed"])  # (n_seeds, n_layers)
            if accs.size == 0:
                continue
            entry: Dict[str, Any] = {
                "mean": accs.mean(axis=0).tolist(),
                "std":  accs.std(axis=0).tolist(),
                "per_seed": accs.tolist(),
            }
            # Save per-seed probe directions (for B2 ablation)
            for dir_key in ("directions_by_seed", ):
                if dir_key in results[pe][probe_type]:
                    entry[dir_key] = results[pe][probe_type][dir_key]
            summary[pe][probe_type] = entry

    out_path = out_dir / "probe_baselines_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Quick summary table
    print("\n=== Probe accuracy summary (mean across layers & seeds) ===")
    header = f"{'PE':10s}  {'linear_abs':>10s}  {'lin_hidden':>10s}  {'mlp_abs':>10s}  {'rel_dist':>10s}  {'near_far':>10s}"
    print(header)
    for pe in args.pe_types:
        row = f"{pe:10s}"
        for pt in ("linear_abs", "linear_hidden", "mlp_abs", "rel_dist", "near_far"):
            if pt in summary.get(pe, {}):
                avg = float(np.mean(summary[pe][pt]["mean"]))
                row += f"  {avg:>10.3f}"
            else:
                row += f"  {'---':>10s}"
        print(row)


if __name__ == "__main__":
    main()
