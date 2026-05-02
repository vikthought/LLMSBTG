"""
Extract PCA-reduced logits from pretrained HuggingFace models with known PE types.

Target models:
    gpt2           — Absolute (learned) positional embeddings
    bloom-560m     — ALiBi (attention linear biases)
    pythia-410m    — RoPE (rotary positional embeddings)

All models are fed the same input corpus for fair comparison.
By default, uses natural text from WikiText-103 so that pretrained models'
positional mechanisms are meaningfully engaged.  Falls back to random tokens
if --corpus random is specified.

Outputs per model:
    <out_dir>/<model_key>_logits_pca.npy   — (N, seq_len, logit_pca_dim)
    <out_dir>/<model_key>_logit_pca.pkl    — fitted PCA object
    <out_dir>/<model_key>_metadata.json    — model info, PE type, shapes

Usage
-----
python scripts/extract_pretrained_logits.py \
    --models gpt2 bloom-560m pythia-410m \
    --seq-len 64 --n-sequences 2000 \
    --logit-pca-dim 256 \
    --out-dir results/pretrained_logits/ \
    --device cuda:0
"""

import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Model registry: model_key -> (HF model ID, PE type, optional kwargs)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "gpt2": {
        "hf_id":   "gpt2",
        "pe_type": "absolute",
        "dtype":   torch.float32,
    },
    "bloom-560m": {
        "hf_id":   "bigscience/bloom-560m",
        "pe_type": "alibi",
        "dtype":   torch.float16,
    },
    "pythia-410m": {
        "hf_id":   "EleutherAI/pythia-410m",
        "pe_type": "rope",
        "dtype":   torch.float32,
    },
    "opt-350m": {
        "hf_id":   "facebook/opt-350m",
        "pe_type": "absolute",
        "dtype":   torch.float32,
    },
    "llama-2-7b": {
        "hf_id":   "meta-llama/Llama-2-7b-hf",
        "pe_type": "rope",
        "dtype":   torch.float16,
    },
}


def generate_random_corpus(tokenizer, n_sequences, seq_len, seed=42):
    """Generate a shared input corpus of random token sequences.

    Uses a fixed random seed so all models see identical inputs.
    Avoids special tokens (pad, eos, bos) to prevent model-specific behavior.
    """
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.vocab_size

    # Build set of normal token IDs (exclude special tokens)
    special_ids = set()
    for attr in ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    normal_ids = np.array([i for i in range(vocab_size) if i not in special_ids])

    input_ids = rng.choice(normal_ids, size=(n_sequences, seq_len))
    return input_ids.astype(np.int64)


def generate_natural_text_corpus(tokenizer, n_sequences, seq_len, seed=42):
    """Generate a shared corpus from WikiText-103 natural text.

    Tokenizes WikiText-103 test split into fixed-length chunks. All models
    share the same token IDs by re-tokenizing the raw text with each model's
    own tokenizer at extraction time — but the underlying TEXT is shared.

    Returns raw text chunks (list of str) plus tokenized IDs for this tokenizer.
    """
    from datasets import load_dataset

    print("  Loading WikiText-103 (test split) ...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Concatenate all text, filter empty lines
    all_text = "\n".join(line for line in ds["text"] if line.strip())

    # Tokenize the full text
    print("  Tokenizing ...")
    encoded = tokenizer(all_text, return_tensors="np", truncation=False,
                        add_special_tokens=False)
    all_ids = encoded["input_ids"][0]  # (total_tokens,)

    # Slice into fixed-length sequences
    n_available = len(all_ids) // seq_len
    if n_available < n_sequences:
        print(f"  WARNING: Only {n_available} sequences available "
              f"(requested {n_sequences}), using all")
        n_sequences = n_available

    rng = np.random.default_rng(seed)
    # Sample random starting positions for diversity
    max_start = len(all_ids) - seq_len
    starts = rng.choice(max_start, size=n_sequences, replace=False)
    starts.sort()

    corpus = np.array([all_ids[s:s + seq_len] for s in starts], dtype=np.int64)
    return corpus


def extract_logits_batched(model, input_ids_np, device, batch_size=8, dtype=torch.float32):
    """Run model forward pass and collect logits.

    Returns raw logits as float32 ndarray: (N, seq_len, vocab_size).
    """
    model.eval()
    all_logits = []
    N = len(input_ids_np)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = torch.tensor(input_ids_np[start:start + batch_size],
                                 dtype=torch.long).to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits.float().cpu().numpy()  # always float32 for PCA
            all_logits.append(logits)

            if (start // batch_size) % 50 == 0:
                print(f"    batch {start // batch_size + 1}/{(N + batch_size - 1) // batch_size}")

    return np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract PCA-reduced logits from pretrained HF models."
    )
    parser.add_argument("--models", nargs="+", default=["gpt2", "bloom-560m", "pythia-410m"],
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model keys to process")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-sequences", type=int, default=2000)
    parser.add_argument("--logit-pca-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus-seed", type=int, default=42,
                        help="Random seed for generating shared input corpus")
    parser.add_argument("--corpus", type=str, default="natural",
                        choices=["natural", "random"],
                        help="Corpus type: 'natural' (WikiText-103, recommended) "
                             "or 'random' (random token IDs)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Pre-load all tokenizers
    tokenizers = {}
    for model_key in args.models:
        info = MODEL_REGISTRY[model_key]
        print(f"Loading tokenizer for {model_key} ({info['hf_id']}) ...")
        tok = AutoTokenizer.from_pretrained(info["hf_id"], trust_remote_code=True)
        tokenizers[model_key] = tok

    # Generate corpus
    use_natural = (args.corpus == "natural")

    if use_natural:
        # Natural text: tokenize per-model (different vocabs), but same raw text
        # We generate the corpus from the first tokenizer, then re-tokenize per model
        print(f"\nCorpus mode: natural text (WikiText-103)")
        # Load raw text once, then tokenize per-model below
        from datasets import load_dataset
        print("  Loading WikiText-103 (test split) ...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        raw_text = "\n".join(line for line in ds["text"] if line.strip())
        print(f"  Raw text length: {len(raw_text):,} chars")
    else:
        # Random tokens: shared vocab range, single corpus for all models
        min_vocab = min(tok.vocab_size for tok in tokenizers.values())
        print(f"\nCorpus mode: random tokens")
        print(f"Shared vocabulary range: [0, {min_vocab})")
        print(f"Generating {args.n_sequences} sequences of length {args.seq_len} ...")
        first_tok = tokenizers[args.models[0]]
        corpus = generate_random_corpus(first_tok, args.n_sequences, args.seq_len,
                                         seed=args.corpus_seed)
        corpus = corpus % min_vocab
        np.save(out_dir / "shared_corpus.npy", corpus)
        print(f"Saved shared corpus: {corpus.shape}")

    # Process each model
    for model_key in args.models:
        info = MODEL_REGISTRY[model_key]
        print(f"\n{'='*55}")
        print(f"  {model_key} ({info['hf_id']})  PE={info['pe_type']}")
        print(f"{'='*55}")

        # Prepare per-model corpus
        if use_natural:
            tok = tokenizers[model_key]
            print(f"  Tokenizing with {model_key} tokenizer ...")
            encoded = tok(raw_text, return_tensors="np", truncation=False,
                          add_special_tokens=False)
            all_ids = encoded["input_ids"][0]
            max_start = len(all_ids) - args.seq_len
            n_avail = max_start
            n_seqs = min(args.n_sequences, n_avail)
            rng = np.random.default_rng(args.corpus_seed)
            starts = rng.choice(max_start, size=n_seqs, replace=False)
            starts.sort()
            model_corpus = np.array([all_ids[s:s + args.seq_len] for s in starts],
                                     dtype=np.int64)
            print(f"  Corpus: {model_corpus.shape} (from {len(all_ids):,} tokens)")
        else:
            model_corpus = corpus

        print(f"  Loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            info["hf_id"],
            torch_dtype=info["dtype"],
            trust_remote_code=True,
        ).to(args.device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e6:.1f}M")

        print(f"  Extracting logits ...")
        raw_logits = extract_logits_batched(
            model, model_corpus, args.device,
            batch_size=args.batch_size, dtype=info["dtype"],
        )
        print(f"  Raw logits shape: {raw_logits.shape}")  # (N, seq_len, vocab_size)

        # Free model memory before PCA
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # PCA reduction
        N, S, V = raw_logits.shape
        print(f"  Fitting PCA: {V} -> {args.logit_pca_dim} ...")
        flat = raw_logits.reshape(-1, V)
        pca = PCA(n_components=args.logit_pca_dim)
        reduced = pca.fit_transform(flat).reshape(N, S, args.logit_pca_dim)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        del raw_logits, flat  # free memory

        # Save
        np.save(out_dir / f"{model_key}_logits_pca.npy", reduced)
        with open(out_dir / f"{model_key}_logit_pca.pkl", "wb") as fp:
            pickle.dump(pca, fp)

        metadata = {
            "model_key":    model_key,
            "hf_id":        info["hf_id"],
            "pe_type":      info["pe_type"],
            "n_params":     n_params,
            "vocab_size":   int(V),
            "seq_len":      args.seq_len,
            "n_sequences":  int(reduced.shape[0]),
            "corpus_type":  args.corpus,
            "logit_pca_dim": args.logit_pca_dim,
            "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
            "output_shape": list(reduced.shape),
        }
        with open(out_dir / f"{model_key}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {out_dir}/{model_key}_*")

    # Save registry info for downstream scripts
    registry_out = {k: {"hf_id": v["hf_id"], "pe_type": v["pe_type"]}
                    for k, v in MODEL_REGISTRY.items() if k in args.models}
    with open(out_dir / "model_registry.json", "w") as f:
        json.dump(registry_out, f, indent=2)

    print(f"\nAll models processed. Results in {out_dir}")


if __name__ == "__main__":
    main()
