"""
Modern SAE baselines (TopK, BatchTopK, T-SAE) for paper3 §4.3 ablation.

Why this exists
---------------
The current SAE baseline (`run_sae_baseline.py`) trains an L1-penalized SAE,
which is the weakest modern variant. Paper3 App F.1 explicitly conjectures
that BatchTopK might partially close the score-vs-SAE gap because it
relaxes the per-token-reconstruction frame. A reviewer reads that
conjecture as "the authors know their baseline is weak and didn't test the
harder one." This script tests it.

Three variants
--------------
  TopK SAE        Per-token top-k. Modern but still per-token-reconstruction.
                  Tests whether the gap is "L1 is bad" vs structurally
                  per-token. Run on 4 control cells.

  BatchTopK SAE   Top (B*k) across (batch, feature) flattened. Relaxes
                  the per-token frame — a single token can recruit more
                  features if others recruit fewer. Headline modern
                  comparison. Run on full 12-cell grid.

  T-SAE           TopK + InfoNCE temporal contrastive loss between
                  adjacent positions in the same sequence. Reproduction
                  of the Bhalla 2026 concept (token-independent SAE
                  objective + sequential structure). Run on ALiBi L4 only.

Apples-to-apples with `run_sae_baseline.py`
-------------------------------------------
Same input (full hidden h ∈ R^256), same readout target (absolute
position 0..63), same top-k=3 directions, same SingularAblationHook,
same α sweep. The only thing that changes is what feature dictionary
the LogReg reads from.

For TopK and BatchTopK, `k` is matched to the L1 SAE's measured average
L0 per cell so feature-budget differences can't drive the comparison.
The T-SAE variant inherits TopK's k.

Output layout
-------------
  {out_dir}/{variant}/analysis/L<layer>/sae_models/<pe>_seed<N>.pt
                                          <pe>_seed<N>_meta.json
  {out_dir}/{variant}/ablation/L<layer>/b2_sae_direction_ablation.json
    └─ same schema as results/sae_baseline_3seed/.../b2_sae_direction_ablation.json
       so compare_ablation_baselines.py works unchanged when pointed at the
       per-variant ablation dir.

Usage
-----
  # BatchTopK on full grid (headline)
  python scripts/run_modern_sae_baselines.py \\
      --variant batchtopk \\
      --summary-path  results/lagpair_ablation_3seed/analysis/analysis_summary.json \\
      --models-dir    results/transformer_pos_models_20260419_114958 \\
      --data-dir      data/transformer_pos_cluster \\
      --out-dir       results/sae_modern_3seed \\
      --pe-types rope alibi absolute \\
      --seeds 0 1 2 \\
      --layers 1 2 3 4 \\
      --l1-sae-dir results/sae_baseline_3seed/analysis

  # TopK on 4 cells (control)
  python scripts/run_modern_sae_baselines.py \\
      --variant topk \\
      ... \\
      --cells alibi:1 alibi:4 absolute:1 absolute:2 \\
      --seeds 0 1 2

  # T-SAE on ALiBi L4 (sole cell)
  python scripts/run_modern_sae_baselines.py \\
      --variant tsae \\
      ... \\
      --cells alibi:4 \\
      --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# 1. SAE architectures
# ============================================================================


class _SAEBase(nn.Module):
    """Shared parameters and decoder-normalization primitives."""

    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.W_enc = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, feature_dim))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        return F.linear(x_c, self.W_enc, self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return F.linear(f, self.W_dec) + self.b_dec

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


class SimpleSAE(_SAEBase):
    """L1-penalized SAE.

    Used here only to load saved L1 checkpoints from `run_sae_baseline.py`
    so we can measure their average L0 and match it for the new variants.
    Architecture is identical to `run_sae_baseline.SimpleSAE`.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self._pre_activation(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


class TopKSAE(_SAEBase):
    """Per-token top-k SAE.

    For each token row of the encoder pre-activation, keep only the top-k
    coordinates (by raw pre-activation value); zero the rest. No L1 needed
    — sparsity is structurally enforced. Modern but still per-token.
    """

    def __init__(self, hidden_dim: int, feature_dim: int, k_per_token: int):
        super().__init__(hidden_dim, feature_dim)
        self.k_per_token = int(k_per_token)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._pre_activation(x)
        return _topk_per_token(z, self.k_per_token)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


class BatchTopKSAE(_SAEBase):
    """Batch-wide top-k SAE.

    Stack the encoder pre-activations across the whole batch into a single
    (B*F)-vector, keep the top (B*k_per_token) entries, zero the rest, then
    reshape back to (B, F). A single token can recruit far more than k
    features if neighboring tokens recruit fewer — relaxing the strict
    per-token reconstruction frame.

    At inference (model.eval()) we fall back to per-token top-k with the
    same k_per_token, so encoded features are well-defined on a single
    activation. This is the standard sae_lens BatchTopK convention.
    """

    def __init__(self, hidden_dim: int, feature_dim: int, k_per_token: int):
        super().__init__(hidden_dim, feature_dim)
        self.k_per_token = int(k_per_token)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._pre_activation(x)
        if self.training:
            return _batchtopk(z, self.k_per_token)
        return _topk_per_token(z, self.k_per_token)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


class TSAE(TopKSAE):
    """TopK SAE with a temporal contrastive auxiliary loss.

    Reproduction of the concept in Bhalla et al. 2026 (T-SAE): a
    token-independent SAE objective augmented with sequential structure.
    Concretely: standard TopK reconstruction + InfoNCE between f_t and
    f_{t+1} of the same sequence (positives), with all other positions in
    the batch as negatives.

    NOTE: this is our reproduction of the concept rather than a port of
    the paper code, which was not available at the time of writing.
    Implementation details (temperature, choice of similarity space) are
    documented as defaults; see `train_tsae` for the full loss.
    """


def _topk_per_token(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k entries per row, zero the rest, ReLU."""
    if k >= z.shape[-1]:
        return F.relu(z)
    vals, idx = torch.topk(z, k=k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, F.relu(vals))
    return out


def _batchtopk(z: torch.Tensor, k_per_token: int) -> torch.Tensor:
    """Keep top (B * k_per_token) entries across the (B, F)-flattened tensor.

    A token can recruit up to F features (no per-token cap during training);
    sparsity is enforced only as a global budget across the batch.
    """
    B, Fdim = z.shape
    total_k = min(B * k_per_token, B * Fdim)
    flat = z.reshape(-1)
    vals, idx = torch.topk(flat, k=total_k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    gated = F.relu(flat) * mask
    return gated.reshape(B, Fdim)


# ============================================================================
# 2. Training routines
# ============================================================================


def measure_l1_l0(
    l1_sae_pt: Path,
    flat_acts: np.ndarray,
    hidden_dim: int,
    feature_dim: int,
    device: str,
    sample_size: int = 50_000,
    rng_seed: int = 0,
) -> Optional[float]:
    """Load an L1 SAE checkpoint, encode activations, return mean L0 per token.

    Returns None if the checkpoint can't be loaded; caller decides on
    fallback k.
    """
    if not l1_sae_pt.exists():
        return None
    try:
        sae = SimpleSAE(hidden_dim, feature_dim).to(device)
        state = torch.load(l1_sae_pt, map_location=device, weights_only=True)
        sae.load_state_dict(state)
        sae.eval()
    except Exception as e:
        print(f"    [warn] could not load L1 SAE at {l1_sae_pt.name}: {e}")
        return None

    rng = np.random.default_rng(rng_seed)
    if len(flat_acts) > sample_size:
        idx = rng.choice(len(flat_acts), sample_size, replace=False)
        flat_acts = flat_acts[idx]

    X = torch.tensor(flat_acts, dtype=torch.float32)
    active_counts: List[float] = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            batch = X[i:i + 1024].to(device)
            f = sae.encode(batch)
            active_counts.append(float((f > 0).float().sum(dim=-1).mean().item()))
    return float(np.mean(active_counts))


def train_topk_sae(
    activations: np.ndarray,
    feature_dim: int,
    k_per_token: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int = 0,
    verbose: bool = True,
    variant: str = "topk",  # "topk" or "batchtopk"
) -> Tuple[_SAEBase, Dict[str, Any]]:
    torch.manual_seed(seed)
    hidden_dim = activations.shape[-1]

    if variant == "topk":
        sae: _SAEBase = TopKSAE(hidden_dim, feature_dim, k_per_token).to(device)
    elif variant == "batchtopk":
        sae = BatchTopKSAE(hidden_dim, feature_dim, k_per_token).to(device)
    else:
        raise ValueError(f"variant must be 'topk' or 'batchtopk', got {variant!r}")

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    X = torch.tensor(activations, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        X, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(n_epochs):
        sae.train()
        recon_sum = 0.0
        active_sum = 0.0
        n_steps = 0
        for batch in loader:
            batch = batch.to(device)
            x_recon, f = sae(batch)
            recon_loss = ((batch - x_recon) ** 2).sum(dim=-1).mean()

            opt.zero_grad()
            recon_loss.backward()
            opt.step()
            sae.normalize_decoder()

            recon_sum += float(recon_loss.item())
            active_sum += float((f > 0).float().sum(dim=-1).mean().item())
            n_steps += 1

        avg_recon = recon_sum / max(n_steps, 1)
        avg_active = active_sum / max(n_steps, 1)
        history.append({
            "epoch": epoch,
            "recon_loss": avg_recon,
            "mean_active_features": avg_active,
        })
        if verbose:
            print(f"    [{variant}] epoch {epoch:2d}  recon={avg_recon:.4f}  "
                  f"active/token={avg_active:.1f}  (target k={k_per_token})")

    sae.eval()
    return sae, {"history": history, "final": history[-1] if history else {}}


def train_tsae(
    seq_activations: np.ndarray,        # (N_seqs, seq_len, hidden_dim)
    feature_dim: int,
    k_per_token: int,
    n_epochs: int,
    lr: float,
    batch_size_seqs: int,               # number of sequences per batch
    contrastive_weight: float,
    contrastive_temperature: float,
    device: str,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[TSAE, Dict[str, Any]]:
    """Train a TopK SAE with an InfoNCE temporal contrastive auxiliary loss.

    Loss = recon_mse + λ · InfoNCE(f_t, f_{t+1})

    The InfoNCE term takes anchors at each non-final position t in the
    batch, treats f_{t+1} (same sequence) as the positive, and treats all
    other (sequence, position) features in the batch as negatives.
    Cosine similarity in feature space, temperature τ.

    Reproduction of the Bhalla 2026 T-SAE concept rather than a port of
    paper code.
    """
    torch.manual_seed(seed)
    hidden_dim = seq_activations.shape[-1]
    sae = TSAE(hidden_dim, feature_dim, k_per_token).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    X = torch.tensor(seq_activations, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        X, batch_size=batch_size_seqs, shuffle=True, drop_last=True,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(n_epochs):
        sae.train()
        recon_sum = 0.0
        nce_sum = 0.0
        active_sum = 0.0
        n_steps = 0

        for batch_seqs in loader:
            batch_seqs = batch_seqs.to(device)              # (B, T, H)
            B, T, H = batch_seqs.shape
            flat = batch_seqs.reshape(B * T, H)
            x_recon, f_flat = sae(flat)                     # f_flat: (B*T, F)

            recon_loss = ((flat - x_recon) ** 2).sum(dim=-1).mean()

            # InfoNCE between adjacent positions
            f_seq = f_flat.reshape(B, T, -1)                # (B, T, F)
            anchors = f_seq[:, :-1, :].reshape(-1, feature_dim)   # (B*(T-1), F)
            positives = f_seq[:, 1:, :].reshape(-1, feature_dim)  # (B*(T-1), F)
            # All positions in the batch as candidate keys
            keys = f_flat                                          # (B*T, F)

            # L2-normalize before cosine similarity. Add a small epsilon for
            # the rare batch where TopK leaves a row strictly zero.
            eps = 1e-8
            anchors_n = anchors / (anchors.norm(dim=-1, keepdim=True) + eps)
            positives_n = positives / (positives.norm(dim=-1, keepdim=True) + eps)
            keys_n = keys / (keys.norm(dim=-1, keepdim=True) + eps)

            logits = anchors_n @ keys_n.T / contrastive_temperature  # (N_anchor, B*T)
            # Positive index: anchor at (b, t) → key at flat index b*T + (t+1)
            n_anchor = anchors_n.shape[0]
            anchor_idx = torch.arange(n_anchor, device=device)
            # anchor index a corresponds to (b, t) where b = a // (T-1), t = a % (T-1)
            b_a = anchor_idx // (T - 1)
            t_a = anchor_idx % (T - 1)
            pos_key_idx = b_a * T + (t_a + 1)

            # Mask out the anchor's own flat index from the negatives
            self_key_idx = b_a * T + t_a
            logits.scatter_(1, self_key_idx.unsqueeze(1), float("-inf"))

            nce_loss = F.cross_entropy(logits, pos_key_idx)

            loss = recon_loss + contrastive_weight * nce_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            recon_sum += float(recon_loss.item())
            nce_sum += float(nce_loss.item())
            active_sum += float((f_flat > 0).float().sum(dim=-1).mean().item())
            n_steps += 1

        avg_recon = recon_sum / max(n_steps, 1)
        avg_nce = nce_sum / max(n_steps, 1)
        avg_active = active_sum / max(n_steps, 1)
        history.append({
            "epoch": epoch,
            "recon_loss": avg_recon,
            "nce_loss": avg_nce,
            "mean_active_features": avg_active,
        })
        if verbose:
            print(f"    [tsae] epoch {epoch:2d}  recon={avg_recon:.4f}  "
                  f"nce={avg_nce:.4f}  active/token={avg_active:.1f}")

    sae.eval()
    return sae, {"history": history, "final": history[-1] if history else {}}


# ============================================================================
# 3. Probe + top-k features → hidden directions
# ============================================================================


def encode_activations(
    sae: _SAEBase,
    activations: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Per-token encoding (eval mode); returns (N, feature_dim)."""
    sae.eval()
    X = torch.tensor(activations, dtype=torch.float32)
    parts: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            f = sae.encode(batch)
            parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0)


def fit_position_probe_on_features(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    max_train_samples: int = 40_000,
    rng_seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """LogReg(SAE features → absolute position). Mirror of probe_hidden's setup."""
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
    return acc, probe.coef_


def top_k_sae_directions(
    sae: _SAEBase,
    probe_W: np.ndarray,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k features by aggregated |probe weight| → decoder columns → QR."""
    feature_importance = np.abs(probe_W).sum(axis=0)
    top_k_idx = np.argsort(feature_importance)[::-1][:top_k]
    top_k_idx = np.sort(top_k_idx)
    W_dec = sae.W_dec.data.cpu().numpy()
    sae_dirs_h = W_dec[:, top_k_idx].T
    Q, _ = np.linalg.qr(sae_dirs_h.T)
    sae_dirs_h = Q.T
    return sae_dirs_h, top_k_idx, feature_importance[top_k_idx]


# ============================================================================
# 4. Ablation harness — same hook as run_ablation_study and run_sae_baseline
# ============================================================================


class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, masks):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.bool)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.masks[i]


def load_test_families(
    data_dir: Path, families: List[str]
) -> Dict[str, torch.utils.data.DataLoader]:
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
    dirs_h: torch.Tensor,
    mu_l: np.ndarray,
    alpha_values: List[float],
    device: str,
    side: str = "source",
    lag: int = 1,
) -> Dict[float, Dict[str, float]]:
    """Identical to run_sae_baseline.eval_with_dirs / run_ablation_study._eval_with_dirs."""
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
    variant: str,
    pe_type: str,
    seed: int,
    layer: int,
    summary_path: Path,
    models_dir: Path,
    out_dir: Path,
    config: GPT2Config,
    families: List[str],
    loaders: Dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    seed_dir = models_dir / f"{pe_type}_seed{seed}"
    model_pt = seed_dir / "model.pt"
    train_acts_p = seed_dir / "train_acts.npy"
    test_acts_p = seed_dir / "test_acts.npy"
    analysis_file = summary_path.parent / f"{pe_type}_seed{seed}_analysis.json"

    for p in (model_pt, train_acts_p, test_acts_p, analysis_file):
        if not p.exists():
            print(f"  SKIP {pe_type} seed{seed} L{layer}: missing {p.name}")
            return None

    cell_tag = f"{variant}_{pe_type}_seed{seed}_L{layer}"
    sae_dir = out_dir / variant / "analysis" / f"L{layer}" / "sae_models"
    sae_dir.mkdir(parents=True, exist_ok=True)
    sae_pt = sae_dir / f"{pe_type}_seed{seed}.pt"
    cell_meta_path = sae_dir / f"{pe_type}_seed{seed}_meta.json"

    print(f"\n  --- {cell_tag} ---")

    # Load activations.
    train_acts_full = np.load(train_acts_p, mmap_mode="r")
    test_acts_full = np.load(test_acts_p, mmap_mode="r")
    train_acts_l = np.array(train_acts_full[:, layer])     # (N_train, seq_len, hidden)
    test_acts_l = np.array(test_acts_full[:, layer])       # (N_test,  seq_len, hidden)
    N_train, seq_len, hidden_dim = train_acts_l.shape
    N_test = test_acts_l.shape[0]
    flat_train = train_acts_l.reshape(-1, hidden_dim)
    flat_test = test_acts_l.reshape(-1, hidden_dim)

    rng = np.random.default_rng(seed * 100 + layer)

    # ---- Decide k_per_token ------------------------------------------------
    measured_l0: Optional[float] = None
    if args.k_mode == "match-l1" and args.l1_sae_dir is not None:
        l1_pt = Path(args.l1_sae_dir) / f"L{layer}" / "sae_models" / f"{pe_type}_seed{seed}.pt"
        # Subsample the train activations for the L0 measurement.
        if len(flat_train) > args.l0_measure_samples:
            idx = rng.choice(len(flat_train), args.l0_measure_samples, replace=False)
            flat_for_l0 = flat_train[idx]
        else:
            flat_for_l0 = flat_train
        measured_l0 = measure_l1_l0(
            l1_pt, flat_for_l0, hidden_dim, args.feature_dim, args.device,
            sample_size=args.l0_measure_samples, rng_seed=seed * 100 + layer,
        )
        if measured_l0 is None:
            print(f"    [k] L1 SAE missing at {l1_pt.name}; falling back to --k {args.k_fallback}")
            k_per_token = args.k_fallback
        else:
            k_per_token = max(1, int(round(measured_l0)))
            print(f"    [k] L1 mean L0 = {measured_l0:.2f} → using k_per_token = {k_per_token}")
    else:
        k_per_token = args.k_fallback
        print(f"    [k] using fixed k_per_token = {k_per_token}")

    # Subsample training activations.
    if len(flat_train) > args.sae_max_samples:
        idx = rng.choice(len(flat_train), args.sae_max_samples, replace=False)
        flat_train_sae = flat_train[idx]
    else:
        flat_train_sae = flat_train

    # ---- Train (or reload) the SAE variant --------------------------------
    if sae_pt.exists() and not args.retrain_sae:
        print(f"    Loading existing SAE: {sae_pt.name}")
        if variant == "topk":
            sae: _SAEBase = TopKSAE(hidden_dim, args.feature_dim, k_per_token).to(args.device)
        elif variant == "batchtopk":
            sae = BatchTopKSAE(hidden_dim, args.feature_dim, k_per_token).to(args.device)
        elif variant == "tsae":
            sae = TSAE(hidden_dim, args.feature_dim, k_per_token).to(args.device)
        else:
            raise ValueError(variant)
        state = torch.load(sae_pt, map_location=args.device, weights_only=True)
        sae.load_state_dict(state)
        sae.eval()
        train_history: Dict[str, Any] = {}
    else:
        if variant in ("topk", "batchtopk"):
            sae, train_history = train_topk_sae(
                activations=flat_train_sae,
                feature_dim=args.feature_dim,
                k_per_token=k_per_token,
                n_epochs=args.sae_epochs,
                lr=args.sae_lr,
                batch_size=args.sae_batch_size,
                device=args.device,
                seed=seed,
                variant=variant,
            )
        else:  # tsae
            # Sequence-shape activations are needed for the contrastive loss.
            # Cap on number of sequences for memory.
            if N_train > args.tsae_max_seqs:
                seq_idx = rng.choice(N_train, args.tsae_max_seqs, replace=False)
                seq_acts = train_acts_l[seq_idx]
            else:
                seq_acts = train_acts_l
            sae, train_history = train_tsae(
                seq_activations=seq_acts,
                feature_dim=args.feature_dim,
                k_per_token=k_per_token,
                n_epochs=args.sae_epochs,
                lr=args.sae_lr,
                batch_size_seqs=args.tsae_batch_size_seqs,
                contrastive_weight=args.tsae_contrastive_weight,
                contrastive_temperature=args.tsae_temperature,
                device=args.device,
                seed=seed,
            )
        torch.save(sae.state_dict(), sae_pt)
        print(f"    Saved SAE: {sae_pt.name}")

    # ---- Encode + position probe ------------------------------------------
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

    # ---- Top-k features → hidden-space directions -------------------------
    sae_dirs_h, top_k_idx, top_k_importance = top_k_sae_directions(
        sae, probe_W, top_k=args.top_k
    )
    sae_dirs_h_t = torch.tensor(sae_dirs_h, dtype=torch.float32)

    # Per-feature activation stats (eval-mode encoding, so applies to all variants)
    feat_active_rate = (train_features[:, top_k_idx] > 0).mean(axis=0)
    feat_mean_when_active = np.array([
        train_features[train_features[:, j] > 0, j].mean()
        if (train_features[:, j] > 0).any() else 0.0
        for j in top_k_idx
    ])

    # Eval-mode mean L0 of the trained variant (for transparency)
    realised_l0 = float((train_features > 0).sum(axis=1).mean())

    # ---- Load model + ablate ---------------------------------------------
    with open(analysis_file) as f:
        analysis = json.load(f)
    layer_data = analysis["layer_stats"][layer - 1]
    mu_l = np.array(layer_data["mu_l"])

    model = create_transformer_variant(config, pe_type)
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(args.device)
    target_layer = model.transformer.h[layer - 1]

    print(f"    Ablating {variant} directions (α ∈ {args.alpha_values}) …")
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

    non_iid = [f for f in families if f != "iid_random"]
    a0 = dose[args.alpha_values[0]]
    a_max = dose[args.alpha_values[-1]]
    delta_per_fam = {f: a_max[f] - a0[f] for f in non_iid}
    delta_avg = float(np.mean(list(delta_per_fam.values())))
    print(f"    Δ_{variant} (mean over {len(non_iid)} families) = {delta_avg:+.4f} nats")

    record = {
        # Schema-compatible with run_sae_baseline output:
        # compare_ablation_baselines.py keys this off "sae_hidden".
        "sae_hidden": {str(a): dose[a] for a in args.alpha_values},
        "sae_meta": {
            "variant": variant,
            "feature_dim": args.feature_dim,
            "k_per_token": k_per_token,
            "k_mode": args.k_mode,
            "measured_l1_l0": measured_l0,
            "realised_eval_l0": realised_l0,
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

    del model, train_features, test_features, train_acts_full, test_acts_full
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return record


# ============================================================================
# 6. Cell-spec parsing
# ============================================================================


def parse_cells(
    cells_arg: Optional[List[str]],
    pe_types: List[str],
    layers: List[int],
) -> List[Tuple[str, int]]:
    """Return the (pe_type, layer) cells to process.

    If --cells is provided, parse it as ['pe:layer', ...] and return those.
    Otherwise return the cartesian product of --pe-types × --layers.
    """
    if cells_arg:
        out: List[Tuple[str, int]] = []
        for c in cells_arg:
            if ":" not in c:
                raise ValueError(f"--cells entry must be 'pe:layer', got {c!r}")
            pe, l = c.split(":", 1)
            out.append((pe.strip(), int(l)))
        return out
    return [(pe, l) for pe in pe_types for l in layers]


# ============================================================================
# 7. Main
# ============================================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["topk", "batchtopk", "tsae"], required=True,
                   help="Which modern SAE variant to train + ablate")

    p.add_argument("--summary-path", type=str, required=True,
                   help="Path to analysis_summary.json (provides mu_l per cell)")
    p.add_argument("--models-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--cells", nargs="+", default=None,
                   help="Override grid; e.g. --cells alibi:1 alibi:4 absolute:1 absolute:2")

    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--alpha-values", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])

    p.add_argument("--feature-dim", type=int, default=1024,
                   help="SAE feature dim (4× hidden_dim=256 by default).")
    p.add_argument("--k-mode", choices=["match-l1", "fixed"], default="match-l1",
                   help="Either match L1 SAE's measured average L0 per cell, or use --k-fallback")
    p.add_argument("--k-fallback", type=int, default=32,
                   help="k_per_token used when k_mode=fixed or L1 checkpoint is missing")
    p.add_argument("--l1-sae-dir", type=str, default=None,
                   help="Path to results/sae_baseline_3seed/analysis (contains L<L>/sae_models/<pe>_seed<N>.pt)")

    p.add_argument("--sae-epochs", type=int, default=10)
    p.add_argument("--sae-lr", type=float, default=1e-3)
    p.add_argument("--sae-batch-size", type=int, default=256)
    p.add_argument("--sae-max-samples", type=int, default=400_000)
    p.add_argument("--l0-measure-samples", type=int, default=50_000)
    p.add_argument("--probe-max-samples", type=int, default=40_000)

    # T-SAE-specific knobs
    p.add_argument("--tsae-batch-size-seqs", type=int, default=32,
                   help="Number of sequences per T-SAE training batch")
    p.add_argument("--tsae-max-seqs", type=int, default=8000,
                   help="Cap on number of sequences fed to T-SAE training")
    p.add_argument("--tsae-contrastive-weight", type=float, default=0.1,
                   help="λ on the InfoNCE temporal contrastive loss")
    p.add_argument("--tsae-temperature", type=float, default=0.1,
                   help="τ for InfoNCE")

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

    cells = parse_cells(args.cells, args.pe_types, args.layers)
    layers_to_process = sorted({l for _, l in cells})
    print(f"  Variant: {args.variant}")
    print(f"  Cells:   {len(cells)} (PE, layer) × {len(args.seeds)} seeds = "
          f"{len(cells) * len(args.seeds)} runs")
    print(f"  Layers seen: {layers_to_process}")
    print()

    # Per-layer JSONs, schema-compatible with run_sae_baseline.py output.
    for layer in layers_to_process:
        layer_out = out_dir / args.variant / "ablation" / f"L{layer}"
        layer_out.mkdir(parents=True, exist_ok=True)
        b2_path = layer_out / "b2_sae_direction_ablation.json"

        if b2_path.exists() and not args.retrain_sae:
            with open(b2_path) as f:
                b2_layer = json.load(f)
            print(f"  [L{layer}] Loading existing {b2_path.name} ({len(b2_layer)} cells)")
        else:
            b2_layer = {}

        for pe_type, l in cells:
            if l != layer:
                continue
            for seed in args.seeds:
                key = f"('{pe_type}', {seed}, {layer}, {args.lag})"
                if key in b2_layer and "sae_hidden" in b2_layer[key] and not args.retrain_sae:
                    print(f"  SKIP (exists): {key}")
                    continue

                rec = process_cell(
                    variant=args.variant,
                    pe_type=pe_type, seed=seed, layer=layer,
                    summary_path=summary_path, models_dir=models_dir,
                    out_dir=out_dir, config=config,
                    families=families, loaders=loaders,
                    args=args,
                )
                if rec is not None:
                    b2_layer[key] = rec
                    with open(b2_path, "w") as f:
                        json.dump(b2_layer, f, indent=2)

        print(f"  Wrote {b2_path}  ({len(b2_layer)} cells)")

    print("\n  Done.  Outputs:")
    for layer in layers_to_process:
        b2 = out_dir / args.variant / "ablation" / f"L{layer}" / "b2_sae_direction_ablation.json"
        if b2.exists():
            n = len(json.loads(b2.read_text()))
            print(f"    {b2}  ({n} cells)")


if __name__ == "__main__":
    main()
