"""
RoPE context-extension validator (Option B).

Take models trained at (base = B0, context = C0) and ask: if we apply NTK-aware
base scaling to extend to (context = C1) WITHOUT retraining, does the
resulting empirical signature land cleanly?

The validation criterion: the extended model's val loss on a position-
sensitive task family at the new context should match a from-scratch model
trained at (base = B1, context = C1).  If it does, NTK extension worked
without retraining; if not, retraining (or YaRN-style frequency-band gating)
is needed.

NTK-aware scaling formula (bloc97 / EleutherAI blog, formalized in YaRN
[Peng et al. 2023]):

    s     = (C1 / C0) ** (d_head / (d_head - 2))
    B1    = B0 * s

For d_head = 64, C0 = 64, C1 = 256:
    s ≈ 4 ** (64/62) ≈ 4.16
    B1 ≈ 41600 (if B0 = 10000)

Inputs
------
--source-models-dir : path containing rope_seed{N}/model.pt — the (B0, C0) cell.
--source-data-dir   : data dir for context C0 (used to build model config).
--target-data-dir   : data dir for context C1 (where val loss is measured).
--target-base       : base of the from-scratch comparison cell at C1, if
                      available; the script will also report the matched-
                      from-scratch loss if --comparison-models-dir is given.

Output
------
<out-dir>/extension_validation.json

    {
      "source_base":         B0,
      "source_context":      C0,
      "target_context":      C1,
      "ntk_scaled_base":     B1,
      "per_seed": {
        "0": {
          "L_no_extension":   loss_at_C1_with_unchanged_inv_freq,
          "L_ntk_extended":   loss_at_C1_with_NTK_scaled_inv_freq,
          "L_pi_extended":    loss_at_C1_with_PI_style_scaled_inv_freq,
        }, ...
      },
      "L_from_scratch":  optional, loss of from-scratch model at (target_base, C1)
    }

Practical claim if NTK_extended ≈ from_scratch and no_extension ≫ from_scratch:
SBTG (or just task loss) confirms NTK-aware extension lands cleanly without
retraining; readers can use this validation as a cheap pre-check before
committing to fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import GPT2Config
from src.sbtg.data.transformer_variants import (
    create_transformer_variant, CustomCAttnRoPE,
)


def find_rope_modules(model: nn.Module) -> List[CustomCAttnRoPE]:
    return [m for m in model.modules() if isinstance(m, CustomCAttnRoPE)]


def build_inv_freq(base: float, d_head: int) -> torch.Tensor:
    """θ_k = base^(-2k/d_head) for k = 0..d_head/2-1."""
    return 1.0 / (float(base) ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))


def set_inv_freq(rope_modules: List[CustomCAttnRoPE], inv_freq: torch.Tensor):
    """Replace every RoPE module's inv_freq with the same (rescaled) values."""
    for m in rope_modules:
        m.inv_freq.data = inv_freq.clone().to(m.inv_freq.device)


def masked_ce(logits, targets, masks):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks = masks[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    return float((loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8))


def evaluate_loss(model, seqs, masks, device, batch_size=128):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            s = torch.tensor(seqs[i:i + batch_size], dtype=torch.long, device=device)
            m = torch.tensor(masks[i:i + batch_size], dtype=torch.bool, device=device)
            out = model(input_ids=s)
            losses.append(masked_ce(out.logits, s, m))
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-models-dir", type=str, required=True,
                   help="Directory containing rope_seed{N}/model.pt at (B0, C0)")
    p.add_argument("--source-data-dir", type=str, required=True,
                   help="Data dir for context C0 (provides metadata + n_positions)")
    p.add_argument("--target-data-dir", type=str, required=True,
                   help="Data dir for context C1 (eval target)")
    p.add_argument("--source-base", type=float, required=True,
                   help="B0 — the base used during training of the source models")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--family", type=str, default="variable_lag_copy")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--comparison-models-dir", type=str, default=None,
                   help="(Optional) directory containing rope_seed{N}/model.pt of a "
                        "FROM-SCRATCH model trained at the target (B1, C1).  If "
                        "provided, its val loss is reported alongside the extended "
                        "loss for direct comparison.")
    p.add_argument("--comparison-data-dir", type=str, default=None,
                   help="(Optional) data dir for the from-scratch comparison cell. "
                        "Defaults to --target-data-dir.")
    # Architecture (must match the trained models)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd",  type=int, default=128)
    p.add_argument("--n-head",  type=int, default=2)
    p.add_argument("--n-inner", type=int, default=512)
    p.add_argument("--device",  type=str, default="cuda:0")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [warn] {args.device} not available, falling back to cpu")
        args.device = "cpu"

    src_models = Path(args.source_models_dir)
    src_data   = Path(args.source_data_dir)
    tgt_data   = Path(args.target_data_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src_data / "metadata.json") as f:
        src_meta = json.load(f)
    with open(tgt_data / "metadata.json") as f:
        tgt_meta = json.load(f)
    C0 = int(src_meta["seq_len"])
    C1 = int(tgt_meta["seq_len"])
    if C1 <= C0:
        print(f"  [warn] target context {C1} ≤ source context {C0}; this isn't "
              f"an extension scenario.")

    seqs  = np.load(tgt_data / f"{args.family}_{args.split}.npy")
    masks = np.load(tgt_data / f"{args.family}_{args.split}_mask.npy")

    d_head = args.n_embd // args.n_head
    s_ntk = (float(C1) / float(C0)) ** (d_head / max(d_head - 2, 1))
    B0 = float(args.source_base)
    B1 = B0 * s_ntk
    print("=" * 72)
    print(" RoPE context-extension validator (NTK-aware, Option B)")
    print(f"   source: ({B0:.0f}, ctx={C0})  →  target: ctx={C1}")
    print(f"   d_head = {d_head}")
    print(f"   NTK scale s = ({C1}/{C0})^(d/(d-2)) = {s_ntk:.3f}")
    print(f"   NTK-scaled base B1 = {B1:.0f}")
    print(f"   PI-equivalent base B_PI = {B0 * (C1/C0):.0f}  (for reference)")
    print("=" * 72)

    # Build target-context config (n_positions = C1 — needed because the model
    # is asked to handle context C1 at inference time).
    tgt_config = GPT2Config(
        vocab_size=tgt_meta["vocab_size"],
        n_positions=C1,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, n_inner=args.n_inner,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )

    inv_no_change = build_inv_freq(B0, d_head)
    inv_ntk       = build_inv_freq(B1, d_head)
    inv_pi        = build_inv_freq(B0 * (C1 / C0), d_head)  # PI-equivalent

    per_seed = {}
    for seed in args.seeds:
        seed_dir = src_models / f"rope_seed{seed}"
        model_pt = seed_dir / "model.pt"
        if not model_pt.exists():
            print(f"  SKIP seed {seed}: no model.pt at {model_pt}")
            continue

        model = create_transformer_variant(tgt_config, "rope")
        state = torch.load(model_pt, map_location="cpu", weights_only=True)
        # Filter out the inv_freq buffers from the loaded state (they were
        # registered with the SOURCE config; the new config rebuilds them
        # at the right C1-friendly default — we'll override below anyway).
        # GPT2 doesn't reinitialise position embeddings on load_state_dict if
        # n_positions matches, so we expand the embedding table if necessary.
        # To keep this simple, we only rely on attention's RoPE pathway and
        # absolute position embeddings of GPT2 are zeroed by create_transformer_variant.
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"  WARN seed {seed}: load_state_dict raised {type(e).__name__}: {e}")
        model.to(args.device)

        rope_modules = find_rope_modules(model)
        if not rope_modules:
            print(f"  SKIP seed {seed}: no CustomCAttnRoPE found")
            continue

        # 1) No extension — use source base
        set_inv_freq(rope_modules, inv_no_change)
        L_no = evaluate_loss(model, seqs, masks, args.device)
        # 2) NTK-aware extension
        set_inv_freq(rope_modules, inv_ntk)
        L_ntk = evaluate_loss(model, seqs, masks, args.device)
        # 3) PI-style equivalent (linear)
        set_inv_freq(rope_modules, inv_pi)
        L_pi = evaluate_loss(model, seqs, masks, args.device)

        per_seed[str(seed)] = {
            "L_no_extension": float(L_no),
            "L_ntk_extended": float(L_ntk),
            "L_pi_extended":  float(L_pi),
        }
        print(f"  seed {seed}: no-ext={L_no:.4f}  NTK={L_ntk:.4f}  PI={L_pi:.4f}")

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Optional: from-scratch comparison
    L_from_scratch = None
    if args.comparison_models_dir:
        comp_dir  = Path(args.comparison_models_dir)
        comp_data = Path(args.comparison_data_dir or args.target_data_dir)
        with open(comp_data / "metadata.json") as f:
            comp_meta = json.load(f)
        comp_seqs  = np.load(comp_data / f"{args.family}_{args.split}.npy")
        comp_masks = np.load(comp_data / f"{args.family}_{args.split}_mask.npy")
        comp_config = GPT2Config(
            vocab_size=comp_meta["vocab_size"],
            n_positions=int(comp_meta["seq_len"]),
            n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, n_inner=args.n_inner,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        losses = []
        for seed in args.seeds:
            mp = comp_dir / f"rope_seed{seed}/model.pt"
            if not mp.exists():
                continue
            model = create_transformer_variant(comp_config, "rope")
            try:
                model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True), strict=False)
            except Exception:
                pass
            model.to(args.device)
            losses.append(evaluate_loss(model, comp_seqs, comp_masks, args.device))
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if losses:
            L_from_scratch = {
                "mean": float(np.mean(losses)),
                "std":  float(np.std(losses)),
                "n":    len(losses),
            }
            print(f"  from-scratch ({args.comparison_models_dir}): {L_from_scratch['mean']:.4f} ± {L_from_scratch['std']:.4f}")

    # Aggregate
    if per_seed:
        no_extension  = [v["L_no_extension"] for v in per_seed.values()]
        ntk_extension = [v["L_ntk_extended"] for v in per_seed.values()]
        pi_extension  = [v["L_pi_extended"]  for v in per_seed.values()]
        agg = {
            "source_base":     B0,
            "source_context":  C0,
            "target_context":  C1,
            "d_head":          d_head,
            "ntk_scale_factor": float(s_ntk),
            "ntk_scaled_base": float(B1),
            "pi_scaled_base":  float(B0 * (C1 / C0)),
            "family":          args.family,
            "split":           args.split,
            "L_no_extension":  {"mean": float(np.mean(no_extension)),  "std": float(np.std(no_extension)),  "n": len(no_extension)},
            "L_ntk_extended":  {"mean": float(np.mean(ntk_extension)), "std": float(np.std(ntk_extension)), "n": len(ntk_extension)},
            "L_pi_extended":   {"mean": float(np.mean(pi_extension)),  "std": float(np.std(pi_extension)),  "n": len(pi_extension)},
            "L_from_scratch":  L_from_scratch,
            "per_seed":        per_seed,
        }
    else:
        agg = {"source_base": B0, "source_context": C0, "target_context": C1, "n_seeds": 0, "per_seed": {}}

    out_path = out_dir / "extension_validation.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"\nWrote {out_path}")

    if per_seed and L_from_scratch:
        # Headline: how close did NTK extension get to from-scratch?
        delta_ntk = agg["L_ntk_extended"]["mean"] - L_from_scratch["mean"]
        delta_no  = agg["L_no_extension"]["mean"]  - L_from_scratch["mean"]
        print()
        print("=" * 72)
        print(f"  from-scratch loss (B={agg['ntk_scaled_base']:.0f}-ish, ctx={C1}): "
              f"{L_from_scratch['mean']:.4f}")
        print(f"  no-extension gap:  Δ = {delta_no:+.4f}  "
              f"(model trained at B0 used at C1 with unchanged base)")
        print(f"  NTK-extended gap:  Δ = {delta_ntk:+.4f}  "
              f"({'CLEAN' if abs(delta_ntk) < 0.1 * L_from_scratch['mean'] else 'needs retraining'})")
        print("=" * 72)


if __name__ == "__main__":
    main()
