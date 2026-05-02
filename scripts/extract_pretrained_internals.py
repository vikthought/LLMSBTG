"""
Extract PCA-reduced internal activations + logits from pretrained HF models.

Extension of `extract_pretrained_logits.py` for the size-sweep + multi-layer
study (paper3 §5.4 follow-up). Differences from the existing extractor:

  1. Larger model registry covering three within-family size sweeps
     (Pythia for RoPE, BLOOM for ALiBi, GPT-2 for Absolute) — the only
     fairness tier that holds tokenizer + training corpus + architecture
     constant and varies size alone.

  2. Internal hidden states extracted at chosen layers (default: relative
     depths 10% / 50% / 90% of the model's depth). The 4-layer matched-toy
     story shows the PE signature lives in early layers (ALiBi rank-one at
     L1–L3); the logit-only fingerprint smears across all 24+ layers and
     reads weakly. Multi-layer pretrained extraction is the natural way to
     test whether the layered signature reproduces at scale.

  3. IncrementalPCA streaming so that 2.8B / 3B models don't need the full
     hidden-state tensor in memory. Per-layer storage at PCA-dim 32 ≈ 16 MB
     regardless of source hidden_dim.

The existing `extract_pretrained_logits.py` is untouched and remains the
entry point for the current paper3 logit-fingerprinting pipeline.

Usage
-----
  python scripts/extract_pretrained_internals.py \\
      --models pythia-1.4b \\
      --mode both \\
      --n-sequences 2000 --seq-len 64 \\
      --logit-pca-dim 256 --inner-pca-dim 32 \\
      --out-dir results/pretrained_size_sweep \\
      --device cuda:0

  --mode {logit, internal, both}        what to extract
  --layers L1 L2 ...                    layer indices (overrides default 10/50/90% depths)
  --layer-relative-depths 0.1 0.5 0.9   alternative spec via relative depth
  --inner-pca-dim 32                    final PCA dim for both internal and logit pipelines
  --logit-pca-dim 256                   first-stage PCA on logits (paper3 sets 256;
                                         512 is the §8.5 robustness check)

Output
------
  <out_dir>/<model_key>/
    metadata.json              model registry entry, sequence info, fairness tier
    corpus_ids.npy             token IDs used (per-model tokenizer)
    logit_pca.npy              (N, seq_len, inner_pca_dim) — if mode in {logit, both}
    logit_pca_stage1.pkl       fitted IncrementalPCA, vocab → logit_pca_dim
    logit_pca_stage2.pkl       fitted PCA, logit_pca_dim → inner_pca_dim
    hidden_L<idx>_pca.npy      (N, seq_len, inner_pca_dim) — per chosen layer
    hidden_L<idx>_pca.pkl      fitted IncrementalPCA, hidden_dim → inner_pca_dim
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA


# ============================================================================
# Model registry — three size-sweep families + matched-scale cross-family
# ============================================================================
#
# Each entry encodes:
#   hf_id       — HuggingFace model ID
#   pe_type     — {absolute, alibi, rope}
#   family      — {gpt2, bloom, pythia, opt, openllama, ...}
#   tier        — {1 = within-family size sweep, 2 = matched-scale cross-family,
#                  3 = cross-corpus same-PE diagnostic}
#   training_corpus — short tag for the training corpus
#   tokenizer_family — short tag (different tokenizers = vocab-distribution confound)
#   dtype       — load dtype on GPU (fp16 for the bigger models)
#
# We do NOT hard-code n_layer / hidden_dim — those are read from the model
# config at extraction time, since HF may revise them. Only PE/family/etc. are
# registry-level metadata.

@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    pe_type: str
    family: str
    tier: int
    training_corpus: str
    tokenizer_family: str
    dtype: torch.dtype = torch.float32


PRETRAINED_REGISTRY: Dict[str, ModelSpec] = {
    # -- Tier 1: Pythia size sweep (RoPE, Pile, GPT-NeoX tokenizer) --------
    "pythia-410m": ModelSpec("EleutherAI/pythia-410m", "rope", "pythia", 1,
                              "the_pile", "gpt-neox", torch.float32),
    "pythia-1b":   ModelSpec("EleutherAI/pythia-1b",   "rope", "pythia", 1,
                              "the_pile", "gpt-neox", torch.float16),
    "pythia-1.4b": ModelSpec("EleutherAI/pythia-1.4b", "rope", "pythia", 1,
                              "the_pile", "gpt-neox", torch.float16),
    "pythia-2.8b": ModelSpec("EleutherAI/pythia-2.8b", "rope", "pythia", 1,
                              "the_pile", "gpt-neox", torch.float16),

    # -- Tier 1: BLOOM size sweep (ALiBi, ROOTS, BLOOM tokenizer) ---------
    "bloom-560m": ModelSpec("bigscience/bloom-560m", "alibi", "bloom", 1,
                             "roots", "bloom", torch.float16),
    "bloom-1b1":  ModelSpec("bigscience/bloom-1b1",  "alibi", "bloom", 1,
                             "roots", "bloom", torch.float16),
    "bloom-1b7":  ModelSpec("bigscience/bloom-1b7",  "alibi", "bloom", 1,
                             "roots", "bloom", torch.float16),
    "bloom-3b":   ModelSpec("bigscience/bloom-3b",   "alibi", "bloom", 1,
                             "roots", "bloom", torch.float16),

    # -- Tier 1: GPT-2 size sweep (Absolute learned, WebText, GPT-2 BPE) --
    "gpt2":         ModelSpec("gpt2",         "absolute", "gpt2", 1,
                              "webtext", "gpt2", torch.float32),
    "gpt2-medium":  ModelSpec("gpt2-medium",  "absolute", "gpt2", 1,
                              "webtext", "gpt2", torch.float32),
    "gpt2-large":   ModelSpec("gpt2-large",   "absolute", "gpt2", 1,
                              "webtext", "gpt2", torch.float16),
    "gpt2-xl":      ModelSpec("gpt2-xl",      "absolute", "gpt2", 1,
                              "webtext", "gpt2", torch.float16),

    # -- Tier 2: matched-scale cross-family at ~350M-560M ------------------
    # The existing paper3 trio (gpt2-small + pythia-410m + bloom-560m) is
    # already this comparison; we add opt-350m as an additional Absolute
    # data point on a different corpus.
    "opt-350m": ModelSpec("facebook/opt-350m", "absolute", "opt", 2,
                          "various", "opt", torch.float32),

    # -- Tier 3: cross-corpus same-PE diagnostic ---------------------------
    # OpenLLaMA-3B is RoPE on RedPajama vs Pythia on the Pile. If the RoPE
    # signature reproduces here, the fingerprint isn't Pile-specific.
    "openllama-3b": ModelSpec("openlm-research/open_llama_3b", "rope", "openllama", 3,
                               "redpajama", "llama", torch.float16),
}


# ============================================================================
# Layer selection
# ============================================================================

def hidden_states_module_indices(n_layer: int) -> Dict[str, int]:
    """HF returns hidden_states as a tuple of (n_layer + 1) tensors.
       Index 0 = embedding output; indices 1..n_layer = post-block outputs.
       We use 1-indexed block outputs throughout (matches paper3 toy-model
       convention where 'layer 1' means after block 0)."""
    return {"first": 1, "middle": (n_layer // 2) + 1, "last": n_layer}


def select_layers(
    n_layer: int,
    explicit: Optional[Sequence[int]] = None,
    relative: Optional[Sequence[float]] = None,
) -> List[int]:
    """Return 1-indexed layer indices into HF's hidden_states tuple.

    Defaults to relative depths [0.1, 0.5, 0.9] of the model — the matched-toy
    paper3 study found PE signatures most concentrated at relative depth ~0.25
    (L1 of 4 = 0.25); to span the layer stack we use 0.1 / 0.5 / 0.9.
    """
    if explicit is not None:
        for idx in explicit:
            if not (1 <= idx <= n_layer):
                raise ValueError(f"layer index {idx} out of range [1, {n_layer}]")
        return list(explicit)
    if relative is None:
        relative = [0.1, 0.5, 0.9]
    raw = sorted({max(1, min(n_layer, round(r * n_layer))) for r in relative})
    return raw


# ============================================================================
# Corpus loading (shared text, per-model tokenization)
# ============================================================================

def load_wikitext_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return "\n".join(line for line in ds["text"] if line.strip())


def tokenize_corpus(tokenizer, raw_text: str, n_sequences: int, seq_len: int,
                    seed: int = 42) -> np.ndarray:
    encoded = tokenizer(raw_text, return_tensors="np", truncation=False,
                        add_special_tokens=False)
    all_ids = encoded["input_ids"][0]
    n_avail = len(all_ids) - seq_len
    if n_avail <= 0:
        raise RuntimeError(f"Corpus too short: {len(all_ids)} tokens, need {seq_len}")
    rng = np.random.default_rng(seed)
    n = min(n_sequences, n_avail)
    starts = np.sort(rng.choice(n_avail, size=n, replace=False))
    return np.array([all_ids[s:s + seq_len] for s in starts], dtype=np.int64)


# ============================================================================
# Streamed extraction with IncrementalPCA per layer
# ============================================================================

def _safe_key(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_key)


def stream_extract(
    model,
    tokenizer,
    corpus: np.ndarray,
    chosen_layers: List[int],
    do_logits: bool,
    inner_pca_dim: int,
    logit_pca_dim: int,
    device: str,
    batch_size: int,
    out_dir: Path,
    progress_every: int = 25,
) -> Dict[str, dict]:
    """Run the model over the corpus in batches, fitting IncrementalPCA per
    target representation as we go, and accumulating PCA-reduced features.

    Returns a dict of metadata (per-stream variance ratios, shapes, etc.).
    Saves per-layer reduced features to disk under `out_dir`.

    Memory profile: never materializes the full (N, seq, hidden_dim) tensor.
    Per-layer working footprint = batch_size × seq_len × hidden_dim (~10s of
    MB) plus the persistent (N, seq_len, inner_pca_dim) reduced output (16 MB).
    """
    model.eval()
    N, seq_len = corpus.shape

    # --- Storage allocation --------------------------------------------------
    reduced_hidden: Dict[int, np.ndarray] = {}
    pca_hidden: Dict[int, IncrementalPCA] = {}
    for L in chosen_layers:
        reduced_hidden[L] = np.empty((N, seq_len, inner_pca_dim), dtype=np.float32)
        pca_hidden[L] = IncrementalPCA(n_components=inner_pca_dim)

    if do_logits:
        # Two-stage logit PCA: stage 1 (vocab → logit_pca_dim) is the heavy
        # one; we IncrementalPCA-fit it during a first pass, transform during
        # a second pass. To avoid double forwards, we accumulate flattened
        # raw logits in memory if it fits; otherwise we'd need a two-pass.
        # For paper3 sizes (vocab up to 250K) and N=2000, seq=64, the float16
        # buffer is up to 2000*64*250K*2 = 64 GB — too big. So we use
        # IncrementalPCA stage 1 directly during streaming.
        pca_logit_stage1 = IncrementalPCA(n_components=logit_pca_dim)
        # Stage 2 (logit_pca_dim → inner_pca_dim) is fit on the post-stage1
        # output, which is small enough to materialize.
        stage1_buffer = np.empty((N, seq_len, logit_pca_dim), dtype=np.float32)

    # --- Pass 1: fit IncrementalPCA --------------------------------------------
    print(f"  Pass 1/2: fit IncrementalPCA per representation ...")
    n_batches = (N + batch_size - 1) // batch_size
    with torch.no_grad():
        for b_idx in range(n_batches):
            s, e = b_idx * batch_size, min(N, (b_idx + 1) * batch_size)
            batch = torch.tensor(corpus[s:e], dtype=torch.long, device=device)
            outputs = model(input_ids=batch, output_hidden_states=True)

            for L in chosen_layers:
                # outputs.hidden_states is tuple of (n_layer + 1) tensors;
                # outputs.hidden_states[L] is the post-block-(L-1) output for
                # 1-indexed L (matches our convention).
                h = outputs.hidden_states[L].float().cpu().numpy()  # (B, seq, hd)
                pca_hidden[L].partial_fit(h.reshape(-1, h.shape[-1]))

            if do_logits:
                lg = outputs.logits.float().cpu().numpy()  # (B, seq, V)
                pca_logit_stage1.partial_fit(lg.reshape(-1, lg.shape[-1]))

            if (b_idx % progress_every) == 0:
                print(f"    fit batch {b_idx + 1}/{n_batches}", flush=True)

    # --- Pass 2: transform ---------------------------------------------------
    print(f"  Pass 2/2: transform + accumulate ...")
    with torch.no_grad():
        for b_idx in range(n_batches):
            s, e = b_idx * batch_size, min(N, (b_idx + 1) * batch_size)
            batch = torch.tensor(corpus[s:e], dtype=torch.long, device=device)
            outputs = model(input_ids=batch, output_hidden_states=True)

            for L in chosen_layers:
                h = outputs.hidden_states[L].float().cpu().numpy()
                B, T, D = h.shape
                reduced = pca_hidden[L].transform(h.reshape(-1, D)).reshape(B, T, -1)
                reduced_hidden[L][s:e] = reduced

            if do_logits:
                lg = outputs.logits.float().cpu().numpy()
                B, T, V = lg.shape
                stage1_buffer[s:e] = pca_logit_stage1.transform(
                    lg.reshape(-1, V)).reshape(B, T, -1)

            if (b_idx % progress_every) == 0:
                print(f"    transform batch {b_idx + 1}/{n_batches}", flush=True)

    # --- Save hidden-state reductions ----------------------------------------
    metadata: Dict[str, dict] = {"hidden_layers": {}, "logit": {}}
    for L in chosen_layers:
        np.save(out_dir / f"hidden_L{L}_pca.npy", reduced_hidden[L])
        with open(out_dir / f"hidden_L{L}_pca.pkl", "wb") as fp:
            pickle.dump(pca_hidden[L], fp)
        metadata["hidden_layers"][str(L)] = {
            "shape": list(reduced_hidden[L].shape),
            "explained_variance_ratio_sum": float(pca_hidden[L].explained_variance_ratio_.sum()),
        }

    # --- Logit stage 2 + save ------------------------------------------------
    if do_logits:
        flat_stage1 = stage1_buffer.reshape(-1, logit_pca_dim)
        pca_logit_stage2 = PCA(n_components=inner_pca_dim)
        reduced_logit = pca_logit_stage2.fit_transform(flat_stage1).reshape(
            N, seq_len, inner_pca_dim)

        np.save(out_dir / "logit_pca.npy", reduced_logit)
        with open(out_dir / "logit_pca_stage1.pkl", "wb") as fp:
            pickle.dump(pca_logit_stage1, fp)
        with open(out_dir / "logit_pca_stage2.pkl", "wb") as fp:
            pickle.dump(pca_logit_stage2, fp)
        metadata["logit"] = {
            "shape": list(reduced_logit.shape),
            "stage1_explained_variance_ratio_sum":
                float(pca_logit_stage1.explained_variance_ratio_.sum()),
            "stage2_explained_variance_ratio_sum":
                float(pca_logit_stage2.explained_variance_ratio_.sum()),
            "stage1_dim": logit_pca_dim,
        }

    return metadata


# ============================================================================
# Per-model driver
# ============================================================================

def process_model(
    model_key: str,
    spec: ModelSpec,
    args: argparse.Namespace,
    raw_text: str,
) -> None:
    out_root = Path(args.out_dir) / _safe_key(model_key)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "metadata.json"
    expected_outputs: List[Path] = []
    if args.mode in ("logit", "both"):
        expected_outputs.append(out_root / "logit_pca.npy")
    # we don't yet know layer indices — finalized after model load

    print(f"\n{'=' * 72}")
    print(f"  {model_key}  ({spec.hf_id})  PE={spec.pe_type}  tier={spec.tier}")
    print(f"  family={spec.family}  corpus={spec.training_corpus}  "
          f"tokenizer={spec.tokenizer_family}")
    print(f"{'=' * 72}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Loading tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=spec.dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    n_layer = model.config.num_hidden_layers
    hidden_dim = getattr(model.config, "hidden_size",
                          getattr(model.config, "n_embd", None))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_layer={n_layer}  hidden_dim={hidden_dim}  params={n_params/1e6:.1f}M")

    # Layer selection
    if args.mode in ("internal", "both"):
        chosen_layers = select_layers(
            n_layer,
            explicit=args.layers,
            relative=args.layer_relative_depths,
        )
        print(f"  Chosen layers (1-indexed): {chosen_layers}  "
              f"(relative depths: {[round(L/n_layer, 2) for L in chosen_layers]})")
    else:
        chosen_layers = []

    # Tokenize corpus with this model's tokenizer (same raw text across models)
    corpus = tokenize_corpus(tokenizer, raw_text, args.n_sequences, args.seq_len,
                              seed=args.corpus_seed)
    np.save(out_root / "corpus_ids.npy", corpus)
    print(f"  Corpus: {corpus.shape}")

    # Idempotency: if every expected output exists, skip extraction.
    expected = [out_root / f"hidden_L{L}_pca.npy" for L in chosen_layers]
    if args.mode in ("logit", "both"):
        expected.append(out_root / "logit_pca.npy")
    if all(p.exists() for p in expected) and not args.force:
        print(f"  All expected outputs exist; skipping extraction. "
              f"(use --force to re-extract)")
        return

    # Run streamed extraction
    do_logits = args.mode in ("logit", "both")
    extraction_meta = stream_extract(
        model=model,
        tokenizer=tokenizer,
        corpus=corpus,
        chosen_layers=chosen_layers,
        do_logits=do_logits,
        inner_pca_dim=args.inner_pca_dim,
        logit_pca_dim=args.logit_pca_dim,
        device=args.device,
        batch_size=args.batch_size,
        out_dir=out_root,
    )

    # Save metadata
    metadata = {
        "model_key": model_key,
        "hf_id": spec.hf_id,
        "pe_type": spec.pe_type,
        "family": spec.family,
        "tier": spec.tier,
        "training_corpus": spec.training_corpus,
        "tokenizer_family": spec.tokenizer_family,
        "n_params": int(n_params),
        "n_layer": int(n_layer),
        "hidden_dim": int(hidden_dim),
        "vocab_size": int(model.config.vocab_size),
        "seq_len": int(args.seq_len),
        "n_sequences": int(corpus.shape[0]),
        "corpus_seed": int(args.corpus_seed),
        "inner_pca_dim": int(args.inner_pca_dim),
        "logit_pca_dim": int(args.logit_pca_dim),
        "chosen_layers": list(chosen_layers),
        "chosen_layers_relative": [float(L / n_layer) for L in chosen_layers],
        "extraction": extraction_meta,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {meta_path}")

    # Free GPU memory before next model
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Model keys from PRETRAINED_REGISTRY.")
    p.add_argument("--list-models", action="store_true",
                   help="Print the registry and exit.")
    p.add_argument("--mode", choices=["logit", "internal", "both"], default="both")

    p.add_argument("--n-sequences", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--corpus-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--inner-pca-dim", type=int, default=32,
                   help="Final PCA dim for both internal layers and post-stage1 logits "
                        "(matches paper3 m=32 throughout).")
    p.add_argument("--logit-pca-dim", type=int, default=256,
                   help="First-stage logit PCA dim (paper3 default 256; 512 is the "
                        "§8.5 robustness check).")

    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Explicit 1-indexed layer indices into hidden_states (overrides "
                        "--layer-relative-depths).")
    p.add_argument("--layer-relative-depths", nargs="+", type=float, default=None,
                   help="Relative depths in (0, 1] to extract (default: 0.1 0.5 0.9). "
                        "Each rounds to nearest 1-indexed layer.")

    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if outputs exist.")

    args = p.parse_args()

    if args.list_models:
        print(f"{'key':<16} {'hf_id':<35} {'pe':<10} {'family':<10} "
              f"{'tier':<5} {'corpus':<12} {'tokenizer'}")
        for key in sorted(PRETRAINED_REGISTRY,
                          key=lambda k: (PRETRAINED_REGISTRY[k].tier,
                                         PRETRAINED_REGISTRY[k].family, k)):
            s = PRETRAINED_REGISTRY[key]
            print(f"{key:<16} {s.hf_id:<35} {s.pe_type:<10} {s.family:<10} "
                  f"{s.tier:<5} {s.training_corpus:<12} {s.tokenizer_family}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate model keys
    for key in args.models:
        if key not in PRETRAINED_REGISTRY:
            raise SystemExit(f"Unknown model key: {key}. Use --list-models.")

    print(f"Loading WikiText-103 ...")
    raw_text = load_wikitext_text()
    print(f"  Raw text: {len(raw_text):,} chars")

    # Persist registry slice + run config
    run_meta = {
        "mode": args.mode,
        "n_sequences": args.n_sequences,
        "seq_len": args.seq_len,
        "inner_pca_dim": args.inner_pca_dim,
        "logit_pca_dim": args.logit_pca_dim,
        "models": {
            k: {
                "hf_id": PRETRAINED_REGISTRY[k].hf_id,
                "pe_type": PRETRAINED_REGISTRY[k].pe_type,
                "family": PRETRAINED_REGISTRY[k].family,
                "tier": PRETRAINED_REGISTRY[k].tier,
                "training_corpus": PRETRAINED_REGISTRY[k].training_corpus,
                "tokenizer_family": PRETRAINED_REGISTRY[k].tokenizer_family,
            }
            for k in args.models
        },
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    for key in args.models:
        process_model(key, PRETRAINED_REGISTRY[key], args, raw_text)

    print(f"\nAll done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
