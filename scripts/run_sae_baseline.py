"""
Sparse-autoencoder baseline for the SBTG ablation comparison.

Why this exists
---------------
The current ablation pipeline compares score-SVD against a *linear* probe
(`probe_hidden`) — the literature standard for "what's readable from
activations" (Alain & Bengio 2016, Hewitt & Manning 2019, Belinkov 2022).
A reviewer's natural follow-up is: "what about non-linear, sparsity-
encouraged features?" That's what SAEs (Bricken et al. 2023; Cunningham
et al. 2023; Templeton et al. 2024) provide.

This script trains an SAE per (PE, seed, layer) cell, identifies the
top-k position-relevant SAE features via a probe trained on feature
activations, extracts their decoder columns as ablation directions in
hidden space, and runs the **same** ablation hook used elsewhere in the
pipeline. The result slots into the same comparison table as
`probe_hidden` and `score_svd`.

Apples-to-apples with `probe_hidden`
------------------------------------
| Step                       | probe_hidden                       | sae_hidden                                                                                                |
|----------------------------|------------------------------------|----------------------------------------------------------------------------------------------------------|
| Input                      | hidden activations h ∈ R^256       | hidden activations h ∈ R^256                                                                              |
| Feature extraction         | none                               | SAE: h → ReLU(W_enc h + b) → f ∈ R^F  (F = 4·256 = 1024)                                                  |
| Position-relevant readout  | LogReg(h → position) weight SVD    | LogReg(f → position) → top-k features by aggregated |coef|                                                |
| Direction set in R^256     | top-3 right SVs of probe weight    | decoder columns W_dec[:, top_k_features], QR-orthonormalized                                              |
| Ablation                   | identical SingularAblationHook     | identical SingularAblationHook                                                                            |

If score-SVD still beats sae_hidden, the Prop 1 logic survives the
strongest non-linear marginal-readout baseline available in the field.

Usage
-----
    python scripts/run_sae_baseline.py \\
        --summary-path  lagpair_ablation_3seed/analysis/analysis_summary.json \\
        --models-dir    results/transformer_pos_models_20260419_114958 \\
        --data-dir      data/transformer_pos_cluster \\
        --out-dir       results/sae_baseline_3seed \\
        --pe-types      rope alibi absolute \\
        --seeds         0 1 2 \\
        --layers        1 2 3 4 \\
        --feature-dim   1024 \\
        --sae-epochs    10 \\
        --l1-lambda     1e-3 \\
        --device        cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import GPT2Config
from src.sbtg.data.transformer_variants import create_transformer_variant


# ============================================================================
# 1. SAE architecture (Bricken-style: untied weights, unit-norm decoder columns)
# ============================================================================

class SimpleSAE(nn.Module):
    """Vanilla L1-penalized sparse autoencoder.

    h_centered  = h - b_dec
    f           = ReLU(W_enc h_centered + b_enc)
    h_recon     = W_dec f + b_dec

    Decoder columns are kept at unit norm after each step so feature
    activation magnitudes are interpretable. This is the standard
    Bricken-2023 setup; it isn't the most modern (Top-K / JumpReLU
    variants are state of the art) but it's sufficient for a
    methodology baseline.
    """

    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.W_enc = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, feature_dim))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        # Geometric initialisation: random Gaussian, then unit-norm decoder cols
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        return F.relu(F.linear(x_c, self.W_enc, self.b_enc))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return F.linear(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        x_recon = self.decode(f)
        return x_recon, f

    def normalize_decoder(self):
        """Project decoder columns onto unit norm. Call after each opt step."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


# ============================================================================
# 2. SAE training
# ============================================================================

def train_sae(
    activations: np.ndarray,           # (N_samples, hidden_dim) — flattened over (seq, position)
    feature_dim: int,
    n_epochs: int,
    l1_lambda: float,
    lr: float,
    batch_size: int,
    device: str,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[SimpleSAE, Dict[str, Any]]:
    torch.manual_seed(seed)
    hidden_dim = activations.shape[-1]

    sae = SimpleSAE(hidden_dim, feature_dim).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    X = torch.tensor(activations, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        X, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(n_epochs):
        sae.train()
        recon_loss_sum = 0.0
        sparsity_sum = 0.0
        active_features_sum = 0.0
        n_steps = 0
        for batch in loader:
            batch = batch.to(device)
            x_recon, f = sae(batch)
            recon_loss = ((batch - x_recon) ** 2).sum(dim=-1).mean()
            sparsity = f.abs().sum(dim=-1).mean()
            loss = recon_loss + l1_lambda * sparsity

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            recon_loss_sum += float(recon_loss.item())
            sparsity_sum += float(sparsity.item())
            active_features_sum += float((f > 0).float().sum(dim=-1).mean().item())
            n_steps += 1

        avg_recon = recon_loss_sum / max(n_steps, 1)
        avg_sparsity = sparsity_sum / max(n_steps, 1)
        avg_active = active_features_sum / max(n_steps, 1)
        history.append({
            "epoch": epoch,
            "recon_loss": avg_recon,
            "sparsity": avg_sparsity,
            "mean_active_features": avg_active,
        })
        if verbose:
            print(f"    [SAE] epoch {epoch:2d}  recon={avg_recon:.4f}  "
                  f"sparsity={avg_sparsity:.2f}  active/token={avg_active:.1f}")

    sae.eval()
    return sae, {"history": history, "final": history[-1] if history else {}}


def encode_activations(
    sae: SimpleSAE,
    activations: np.ndarray,           # (N_samples, hidden_dim)
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Run activations through the SAE encoder. Returns (N_samples, feature_dim)."""
    sae.eval()
    X = torch.tensor(activations, dtype=torch.float32)
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            f = sae.encode(batch)
            parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ============================================================================
# 3. Position probe on SAE features + top-k feature → hidden directions
# ============================================================================

def fit_position_probe_on_features(
    features_train: np.ndarray,        # (N_samples, feature_dim)
    labels_train: np.ndarray,          # (N_samples,) — position labels 0..seq_len-1
    features_test: np.ndarray,
    labels_test: np.ndarray,
    max_train_samples: int = 40_000,
    rng_seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """LogisticRegression on SAE feature activations, mirror of probe_hidden's
    LogReg on raw activations. Returns (test_accuracy, weight_matrix)."""
    rng = np.random.default_rng(rng_seed)

    if len(features_train) > max_train_samples:
        idx = rng.choice(len(features_train), max_train_samples, replace=False)
        features_train, labels_train = features_train[idx], labels_train[idx]

    sc = StandardScaler()
    Xtr = sc.fit_transform(features_train)
    Xte = sc.transform(features_test)

    probe = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs", n_jobs=-1)
    probe.fit(Xtr, labels_train)
    acc = float(probe.score(Xte, labels_test))

    return acc, probe.coef_   # (n_classes, feature_dim)


def top_k_sae_directions(
    sae: SimpleSAE,
    probe_W: np.ndarray,              # (n_classes, feature_dim)
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the top-k position-relevant SAE features by aggregated probe-weight
    magnitude, take their decoder columns as directions in hidden space, and
    QR-orthonormalize.

    Returns
    -------
    dirs_h       : (top_k, hidden_dim) — orthonormal directions in hidden space
    feat_idx     : (top_k,) — which feature indices were selected
    importance   : (top_k,) — their aggregated |probe weight| score
    """
    feature_importance = np.abs(probe_W).sum(axis=0)            # (feature_dim,)
    top_k_idx = np.argsort(feature_importance)[::-1][:top_k]    # (top_k,)
    top_k_idx = np.sort(top_k_idx)                              # stable order
    W_dec = sae.W_dec.data.cpu().numpy()                        # (hidden_dim, feature_dim)
    sae_dirs_h = W_dec[:, top_k_idx].T                          # (top_k, hidden_dim)
    Q, _ = np.linalg.qr(sae_dirs_h.T)                           # (hidden_dim, top_k)
    sae_dirs_h = Q.T                                            # (top_k, hidden_dim)
    return sae_dirs_h, top_k_idx, feature_importance[top_k_idx]


# ============================================================================
# 4. Ablation harness — same hook mechanic as run_ablation_study._eval_with_dirs
# ============================================================================

class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, masks):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.bool)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.masks[i]


def load_test_families(data_dir: Path, families: List[str]) -> Dict[str, torch.utils.data.DataLoader]:
    loaders = {}
    for fam in families:
        seqs = np.load(data_dir / f"{fam}_test.npy")
        masks = np.load(data_dir / f"{fam}_test_mask.npy")
        ds = _TokenDataset(seqs, masks)
        loaders[fam] = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    return loaders


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> float:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks = masks[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    return float((loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8))


def eval_with_dirs(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    target_layer: nn.Module,
    dirs_h: torch.Tensor,             # (k, hidden_dim) unit-norm orthogonal
    mu_l: np.ndarray,                 # (hidden_dim,)
    alpha_values: List[float],
    device: str,
    side: str = "source",
    lag: int = 1,
) -> Dict[float, Dict[str, float]]:
    """Same ablation mechanic as run_ablation_study._eval_with_dirs, factored
    so this script doesn't have to import the other entry point."""
    results: Dict[float, Dict[str, float]] = {}
    model.eval()
    model.to(device)

    mu_tensor = torch.tensor(mu_l, dtype=torch.float32).to(device)
    dirs_dev = dirs_h.to(device)

    for alpha in alpha_values:
        def _hook_fn(module, inp, out, _alpha=alpha, _side=side, _lag=lag,
                     _dirs=dirs_dev, _mu=mu_tensor):
            h = out[0] if isinstance(out, tuple) else out
            b, seq_len, dim = h.shape
            h_c = h - _mu
            D = _dirs.T
            proj = h_c @ D
            ablation = proj @ D.T
            mask = torch.zeros((1, seq_len, 1), device=h.device)
            for i in range(seq_len):
                src = i - _lag
                if _side == "source" and 0 <= src < seq_len:
                    mask[:, src, :] = 1.0
                elif _side == "target":
                    mask[:, i, :] = 1.0
            h_prime = h - _alpha * ablation * mask
            return (h_prime,) + out[1:] if isinstance(out, tuple) else h_prime

        handle = target_layer.register_forward_hook(_hook_fn)
        fam_losses: Dict[str, float] = {}
        with torch.no_grad():
            for fam, loader in loaders.items():
                losses = []
                for seqs, masks in loader:
                    seqs = seqs.to(device)
                    masks = masks.to(device)
                    out = model(input_ids=seqs)
                    losses.append(masked_ce(out.logits, seqs, masks))
                fam_losses[fam] = float(np.mean(losses))
        handle.remove()
        results[alpha] = fam_losses

    return results


# ============================================================================
# 5. Per-cell driver
# ============================================================================

def process_cell(
    pe_type: str,
    seed: int,
    layer: int,
    summary_path: Path,
    models_dir: Path,
    data_dir: Path,
    out_dir: Path,
    config: GPT2Config,
    families: List[str],
    loaders: Dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Train SAE → fit feature probe → extract top-k directions → ablate.
    Returns the per-cell record (dose response per α + metadata)."""
    seed_dir = models_dir / f"{pe_type}_seed{seed}"
    model_pt = seed_dir / "model.pt"
    train_acts_p = seed_dir / "train_acts.npy"
    test_acts_p = seed_dir / "test_acts.npy"

    analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"

    for p in (model_pt, train_acts_p, test_acts_p, analysis_file):
        if not p.exists():
            print(f"  SKIP {pe_type} seed{seed} L{layer}: missing {p.name}")
            return None

    cell_tag = f"{pe_type}_seed{seed}_L{layer}"
    sae_dir = out_dir / "analysis" / f"L{layer}" / "sae_models"
    sae_dir.mkdir(parents=True, exist_ok=True)
    sae_pt = sae_dir / f"{pe_type}_seed{seed}.pt"
    cell_meta_path = sae_dir / f"{pe_type}_seed{seed}_meta.json"

    print(f"\n  --- {cell_tag} ---")

    # Load activations and slice to the chosen layer.
    # train_acts shape: (N_train, n_layers+1, seq_len, hidden_dim).
    # Layer 0 = embeddings; we use 1-indexed layers consistent with the rest of the pipeline.
    train_acts_full = np.load(train_acts_p, mmap_mode="r")
    test_acts_full = np.load(test_acts_p, mmap_mode="r")
    train_acts_l = np.array(train_acts_full[:, layer])     # (N_train, seq_len, hidden)
    test_acts_l = np.array(test_acts_full[:, layer])       # (N_test,  seq_len, hidden)
    N_train, seq_len, hidden_dim = train_acts_l.shape
    N_test = test_acts_l.shape[0]

    # Subsample training activations for SAE if very large.
    flat_train = train_acts_l.reshape(-1, hidden_dim)
    rng = np.random.default_rng(seed * 100 + layer)
    if len(flat_train) > args.sae_max_samples:
        idx = rng.choice(len(flat_train), args.sae_max_samples, replace=False)
        flat_train_sae = flat_train[idx]
    else:
        flat_train_sae = flat_train

    # ---- Train (or reload) the SAE ----------------------------------------
    if sae_pt.exists() and not args.retrain_sae:
        print(f"    Loading existing SAE: {sae_pt.name}")
        sae = SimpleSAE(hidden_dim, args.feature_dim).to(args.device)
        sae.load_state_dict(torch.load(sae_pt, map_location=args.device))
        sae.eval()
        train_history = {}
    else:
        sae, train_history = train_sae(
            activations=flat_train_sae,
            feature_dim=args.feature_dim,
            n_epochs=args.sae_epochs,
            l1_lambda=args.l1_lambda,
            lr=args.sae_lr,
            batch_size=args.sae_batch_size,
            device=args.device,
            seed=seed,
        )
        torch.save(sae.state_dict(), sae_pt)
        print(f"    Saved SAE: {sae_pt.name}")

    # ---- Encode + position probe on features -------------------------------
    flat_test = test_acts_l.reshape(-1, hidden_dim)
    train_features = encode_activations(sae, flat_train, args.device)
    test_features = encode_activations(sae, flat_test, args.device)

    train_labels = np.tile(np.arange(seq_len), N_train)
    test_labels = np.tile(np.arange(seq_len), N_test)

    probe_acc, probe_W = fit_position_probe_on_features(
        train_features, train_labels, test_features, test_labels,
        max_train_samples=args.probe_max_samples,
        rng_seed=seed * 100 + layer,
    )
    chance = 1.0 / seq_len
    print(f"    Feature-probe accuracy: {probe_acc:.4f}  (chance ≈ {chance:.4f})")

    # ---- Top-k features → hidden-space directions --------------------------
    sae_dirs_h, top_k_idx, top_k_importance = top_k_sae_directions(
        sae, probe_W, top_k=args.top_k
    )
    sae_dirs_h_t = torch.tensor(sae_dirs_h, dtype=torch.float32)

    # ---- Compute mean-activation rate per selected feature (sparsity stat) -
    feat_active_rate = (train_features[:, top_k_idx] > 0).mean(axis=0)
    feat_mean_when_active = np.array([
        train_features[train_features[:, j] > 0, j].mean()
        if (train_features[:, j] > 0).any() else 0.0
        for j in top_k_idx
    ])

    # ---- Load model + ablate -----------------------------------------------
    with open(analysis_file) as f:
        analysis = json.load(f)
    layer_data = analysis["layer_stats"][layer - 1]
    mu_l = np.array(layer_data["mu_l"])

    model = create_transformer_variant(config, pe_type)
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(args.device)
    target_layer = model.transformer.h[layer - 1]

    print(f"    Ablating SAE-derived directions (α ∈ {args.alpha_values}) …")
    dose = eval_with_dirs(
        model=model,
        loaders=loaders,
        target_layer=target_layer,
        dirs_h=sae_dirs_h_t,
        mu_l=mu_l,
        alpha_values=args.alpha_values,
        device=args.device,
        side="source",
        lag=args.lag,
    )

    # Family-averaged Δ at α = max
    non_iid = [f for f in families if f != "iid_random"]
    a0 = dose[args.alpha_values[0]]
    a_max = dose[args.alpha_values[-1]]
    delta_per_fam = {f: a_max[f] - a0[f] for f in non_iid}
    delta_avg = float(np.mean(list(delta_per_fam.values())))
    print(f"    Δ_sae (mean over {len(non_iid)} families) = {delta_avg:+.4f} nats")

    # ---- Pack record + per-cell metadata ----------------------------------
    record = {
        "sae_hidden": {str(a): dose[a] for a in args.alpha_values},
        "sae_meta": {
            "feature_dim": args.feature_dim,
            "l1_lambda": args.l1_lambda,
            "sae_epochs": args.sae_epochs,
            "feature_probe_acc": probe_acc,
            "chance_acc": chance,
            "top_k_feature_idx": top_k_idx.tolist(),
            "top_k_feature_importance": top_k_importance.tolist(),
            "top_k_feature_active_rate": feat_active_rate.tolist(),
            "top_k_feature_mean_when_active": feat_mean_when_active.tolist(),
            "delta_per_family_alpha_max": delta_per_fam,
            "delta_avg_alpha_max": delta_avg,
        },
        "training_history": train_history,
    }

    cell_meta_path.write_text(json.dumps(record["sae_meta"], indent=2))

    # Free memory before next cell
    del model, train_features, test_features, train_acts_full, test_acts_full
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    return record


# ============================================================================
# 6. Main loop and aggregation
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-path", type=str, required=True,
                   help="Path to analysis_summary.json (provides mu_l per cell)")
    p.add_argument("--models-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--alpha-values", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])

    p.add_argument("--feature-dim", type=int, default=1024,
                   help="SAE feature_dim. Default 4× hidden_dim=256.")
    p.add_argument("--sae-epochs", type=int, default=10)
    p.add_argument("--l1-lambda", type=float, default=1e-3)
    p.add_argument("--sae-lr", type=float, default=1e-3)
    p.add_argument("--sae-batch-size", type=int, default=256)
    p.add_argument("--sae-max-samples", type=int, default=400_000,
                   help="Cap on total training tokens for SAE per cell")
    p.add_argument("--probe-max-samples", type=int, default=40_000)
    p.add_argument("--retrain-sae", action="store_true",
                   help="Force re-train SAEs even if checkpoints exist")

    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [warn] {args.device} not available, falling back to cpu")
        args.device = "cpu"

    summary_path = Path(args.summary_path)
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    families = meta.get("families", [
        "variable_lag_copy", "absolute_anchor",
        "order_sensitive", "distance_bucket", "iid_random",
    ])

    config = GPT2Config(
        vocab_size=meta["vocab_size"],
        n_positions=meta["seq_len"],
        n_embd=256, n_layer=4, n_head=4, n_inner=1024,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )

    loaders = load_test_families(data_dir, families)

    # Per-layer b2-style JSONs, mirroring lagpair_ablation_3seed/ablation/L<layer>/b2_*.json
    for layer in args.layers:
        layer_out = out_dir / "ablation" / f"L{layer}"
        layer_out.mkdir(parents=True, exist_ok=True)
        b2_path = layer_out / "b2_sae_direction_ablation.json"

        if b2_path.exists() and not args.retrain_sae:
            with open(b2_path) as f:
                b2_layer = json.load(f)
            print(f"\n  [L{layer}] Loading existing {b2_path.name} "
                  f"({len(b2_layer)} cells)")
        else:
            b2_layer = {}

        for pe_type in args.pe_types:
            for seed in args.seeds:
                key = f"('{pe_type}', {seed}, {layer}, {args.lag})"
                if key in b2_layer and "sae_hidden" in b2_layer[key] and not args.retrain_sae:
                    print(f"  SKIP (exists): {key}")
                    continue

                rec = process_cell(
                    pe_type=pe_type, seed=seed, layer=layer,
                    summary_path=summary_path, models_dir=models_dir,
                    data_dir=data_dir, out_dir=out_dir,
                    config=config, families=families, loaders=loaders,
                    args=args,
                )
                if rec is not None:
                    b2_layer[key] = rec
                    # Incremental persistence so a crash mid-layer doesn't lose work
                    with open(b2_path, "w") as f:
                        json.dump(b2_layer, f, indent=2)

        print(f"\n  Wrote {b2_path}  ({len(b2_layer)} cells)")

    print("\n  Done.  Outputs:")
    for layer in args.layers:
        b2 = out_dir / "ablation" / f"L{layer}" / "b2_sae_direction_ablation.json"
        n = sum(1 for _ in b2.read_text().split('"sae_hidden"')) - 1 if b2.exists() else 0
        print(f"    {b2}  ({n} cells)")


if __name__ == "__main__":
    main()
