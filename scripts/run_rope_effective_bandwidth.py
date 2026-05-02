"""
RoPE effective bandwidth diagnostic (Option A).

For each trained RoPE model in a (base, context, seed) grid cell, this script
measures how many of the d_head/2 rotation frequency dimensions the model
actually *uses*.  The standard architectural argument prescribes
$\\theta_k = \\text{base}^{-2k/d_{\\text{head}}}$ for $k = 0, \\ldots, K-1$
($K = d_{\\text{head}}/2$); whether the trained model attends to all of those
dimensions or only the highest-frequency tail is an empirical question.

Protocol
--------
For each model:
  1.  Record baseline val loss with the full RoPE config.
  2.  Sweep $k_{\\text{keep}} \\in \\{32, 24, 16, 12, 8, 6, 4, 2, 1\\}$
      (or the user-specified levels, clipped to $\\le K$).
  3.  For each $k_{\\text{keep}}$: zero out
      $\\text{inv\\_freq}[k_{\\text{keep}}{:}\\,]$ (i.e., kill the
      $K - k_{\\text{keep}}$ lowest-frequency dimensions, keep the highest
      $k_{\\text{keep}}$).  Restore after measurement.
  4.  Define **effective bandwidth** $k_{\\text{eff}}$ as the smallest
      $k_{\\text{keep}}$ at which val loss is within
      $(1 + \\text{tolerance}) \\cdot L_{\\text{full}}$.  Default tolerance = 0.10.

Practical claim if $k_{\\text{eff}} \\ll K$: the model only uses the top-
$k_{\\text{eff}}$ frequencies; bases that keep $\\theta_{k_{\\text{eff}}}$ in
the active range work equivalently.  Different from NTK-aware (which
prescribes a base from architectural arguments) — this is what the model is
actually doing.

Output
------
<out-dir>/effective_bandwidth.json with per-seed sweeps and a per-cell
$k_{\\text{eff}}$ summary.

Note: only ablates RoPE.  Absolute / ALiBi cells should be skipped at
the cluster level — this script asserts the model is RoPE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import GPT2Config
from src.sbtg.data.transformer_variants import (
    create_transformer_variant, CustomCAttnRoPE,
)


# ============================================================================
# Helpers
# ============================================================================

def find_rope_modules(model: nn.Module) -> List[CustomCAttnRoPE]:
    """Return all CustomCAttnRoPE instances in the model (one per layer)."""
    mods = []
    for m in model.modules():
        if isinstance(m, CustomCAttnRoPE):
            mods.append(m)
    return mods


def zero_low_freq_dims(rope_modules: List[CustomCAttnRoPE], k_keep: int) -> List[torch.Tensor]:
    """Modify each module's inv_freq in place: keep top k_keep entries, zero the rest.
    Returns the original tensors so they can be restored.

    Note: inv_freq is indexed 0 = highest frequency, K-1 = lowest frequency.
    Keeping the top k_keep means slots [0, ..., k_keep-1] survive;
    [k_keep, ..., K-1] (the low-frequency tail) are zeroed.
    """
    originals = []
    for m in rope_modules:
        orig = m.inv_freq.detach().clone()
        originals.append(orig)
        new = orig.clone()
        if k_keep < new.numel():
            new[k_keep:] = 0.0
        m.inv_freq.data = new
    return originals


def restore_inv_freq(rope_modules: List[CustomCAttnRoPE], originals: List[torch.Tensor]):
    for m, orig in zip(rope_modules, originals):
        m.inv_freq.data = orig.clone()


def masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks = masks[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    return float((loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8))


def evaluate_loss_on_family(
    model: nn.Module,
    seqs: np.ndarray,
    masks: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> float:
    """Mean masked-CE on a family's val/test set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            s = torch.tensor(seqs[i:i + batch_size], dtype=torch.long, device=device)
            m = torch.tensor(masks[i:i + batch_size], dtype=torch.bool, device=device)
            out = model(input_ids=s)
            losses.append(masked_ce(out.logits, s, m))
    return float(np.mean(losses))


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str, required=True,
                   help="Data dir for the model's training context (has metadata.json + per-family npy)")
    p.add_argument("--models-dir", type=str, required=True,
                   help="Directory containing rope_seed{N}/model.pt files")
    p.add_argument("--out-dir",    type=str, required=True)
    p.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--family",     type=str, default="variable_lag_copy",
                   help="Position-sensitive task family for the effective-bandwidth measurement")
    p.add_argument("--split",      type=str, default="test",
                   choices=["val", "test"],
                   help="Use val_*.npy or test_*.npy")
    p.add_argument("--k-keep-levels", nargs="+", type=int,
                   default=[32, 24, 16, 12, 8, 6, 4, 2, 1],
                   help="Which k_keep values to evaluate.  Will be clipped to ≤ K=d_head/2.")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="Effective bandwidth threshold: smallest k_keep with "
                        "loss ≤ (1 + tol) · L_full")
    # Architecture knobs (must match training)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd",  type=int, default=128)
    p.add_argument("--n-head",  type=int, default=2)
    p.add_argument("--n-inner", type=int, default=512)
    p.add_argument("--device",  type=str, default="cuda:0")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [warn] {args.device} not available, falling back to cpu")
        args.device = "cpu"

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata + the chosen task family's data
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    seqs = np.load(data_dir / f"{args.family}_{args.split}.npy")
    masks = np.load(data_dir / f"{args.family}_{args.split}_mask.npy")

    config = GPT2Config(
        vocab_size=meta["vocab_size"],
        n_positions=meta["seq_len"],
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, n_inner=args.n_inner,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    K = config.n_embd // config.n_head // 2  # d_head / 2
    k_levels = sorted(set(min(k, K) for k in args.k_keep_levels), reverse=True)

    print("=" * 70)
    print(" RoPE effective-bandwidth diagnostic")
    print(f"   models:    {models_dir}")
    print(f"   data:      {data_dir}  (family={args.family}, split={args.split})")
    print(f"   K = d_head/2 = {K}")
    print(f"   k_keep levels: {k_levels}")
    print(f"   tolerance: {args.tolerance}")
    print("=" * 70)

    per_seed = {}
    for seed in args.seeds:
        seed_dir = models_dir / f"rope_seed{seed}"
        model_pt = seed_dir / "model.pt"
        if not model_pt.exists():
            print(f"  SKIP seed {seed}: no model.pt at {model_pt}")
            continue

        model = create_transformer_variant(config, "rope")
        state = torch.load(model_pt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(args.device)

        rope_modules = find_rope_modules(model)
        if not rope_modules:
            print(f"  SKIP seed {seed}: no CustomCAttnRoPE modules found")
            continue
        print(f"\n  seed {seed}: {len(rope_modules)} RoPE modules, "
              f"inv_freq shape {tuple(rope_modules[0].inv_freq.shape)}")

        # Baseline
        L_full = evaluate_loss_on_family(model, seqs, masks, args.device)
        print(f"    full RoPE val loss: {L_full:.4f}")

        # Sweep k_keep
        sweep = []
        for k_keep in k_levels:
            originals = zero_low_freq_dims(rope_modules, k_keep)
            try:
                L = evaluate_loss_on_family(model, seqs, masks, args.device)
            finally:
                restore_inv_freq(rope_modules, originals)
            ratio = L / max(L_full, 1e-8)
            sweep.append({
                "k_keep": int(k_keep),
                "loss":   float(L),
                "ratio_vs_full": float(ratio),
            })
            print(f"    k_keep={k_keep:>3}  loss={L:.4f}  ratio={ratio:.3f}")

        # Effective bandwidth: smallest k_keep where loss ≤ (1+tol) · L_full
        threshold = (1 + args.tolerance) * L_full
        k_eff = K  # default to full bandwidth if we never find a clean knee
        for entry in sorted(sweep, key=lambda e: e["k_keep"]):
            if entry["loss"] <= threshold:
                k_eff = entry["k_keep"]
                break
        print(f"    → k_eff = {k_eff}  (loss ≤ {threshold:.4f})")

        per_seed[str(seed)] = {
            "L_full":   float(L_full),
            "threshold": float(threshold),
            "k_eff":    int(k_eff),
            "sweep":    sweep,
        }

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Aggregate
    if per_seed:
        k_effs = [v["k_eff"] for v in per_seed.values()]
        L_fulls = [v["L_full"] for v in per_seed.values()]
        agg = {
            "K":            int(K),
            "k_levels":     k_levels,
            "tolerance":    float(args.tolerance),
            "family":       args.family,
            "split":        args.split,
            "k_eff_mean":   float(np.mean(k_effs)),
            "k_eff_std":    float(np.std(k_effs)),
            "L_full_mean":  float(np.mean(L_fulls)),
            "L_full_std":   float(np.std(L_fulls)),
            "n_seeds":      len(per_seed),
            "per_seed":     per_seed,
        }
    else:
        agg = {"K": int(K), "n_seeds": 0, "per_seed": {}}

    out_path = out_dir / "effective_bandwidth.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"\nWrote {out_path}")
    if agg.get("n_seeds", 0) > 0:
        print(f"  k_eff = {agg['k_eff_mean']:.1f} ± {agg['k_eff_std']:.1f}  "
              f"(K={K}, fraction = {agg['k_eff_mean']/K:.2f})")


if __name__ == "__main__":
    main()
