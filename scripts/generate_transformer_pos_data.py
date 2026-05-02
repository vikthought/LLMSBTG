import argparse
import os
import json
import numpy as np
from pathlib import Path

def generate_iid_random(n_seqs: int, seq_len: int, vocab_size: int, seed: int):
    """Generates pure IID random tokens. No deterministic targets."""
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, vocab_size, size=(n_seqs, seq_len), dtype=np.int64)
    # Mask is all zeros because there is no learnable target
    masks = np.zeros((n_seqs, seq_len), dtype=np.bool_)
    return seqs, masks

def generate_variable_lag_copy(n_seqs: int, seq_len: int, vocab_size: int, seed: int):
    """
    Dense variable-lag copy:
    Repeatedly writes a context block, then a special trigger, then forces copying
    of the context block. Yields multiple answer-bearing positions.
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, vocab_size - 1, size=(n_seqs, seq_len), dtype=np.int64)
    masks = np.zeros((n_seqs, seq_len), dtype=np.bool_)
    trigger_token = vocab_size - 1

    for i in range(n_seqs):
        # Generate 2 to 4 copy tasks per sequence
        n_tasks = rng.integers(2, 5)
        cursor = 0
        for _ in range(n_tasks):
            if cursor >= seq_len - 4:
                break
            # Block length to copy
            block_len = rng.integers(1, 6)
            lag = rng.integers(2, 10)

            # Ensure it fits
            if cursor + block_len + lag + 1 + block_len >= seq_len:
                break

            # Place trigger
            copy_start = cursor + block_len + lag
            seqs[i, copy_start] = trigger_token

            # Perform copy
            seqs[i, copy_start + 1 : copy_start + 1 + block_len] = seqs[i, cursor : cursor + block_len]
            masks[i, copy_start + 1 : copy_start + 1 + block_len] = True

            cursor = copy_start + 1 + block_len

    return seqs, masks

def generate_absolute_anchor(n_seqs: int, seq_len: int, vocab_size: int, seed: int):
    """
    Dense absolute-anchor:
    Places an anchor symbol at fixed absolute positions (e.g., 0, 20, 40).
    A few steps after each anchor, specific deterministic tokens are placed.
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, vocab_size - 2, size=(n_seqs, seq_len), dtype=np.int64)
    masks = np.zeros((n_seqs, seq_len), dtype=np.bool_)
    anchor_token = vocab_size - 1
    target_token = vocab_size - 2

    anchor_positions = [0, seq_len // 3, (2 * seq_len) // 3]

    for pos in anchor_positions:
        if pos < seq_len:
            seqs[:, pos] = anchor_token
            # Place dense targets after the anchor
            for offset in [3, 5, 7]:
                if pos + offset < seq_len:
                    seqs[:, pos + offset] = target_token
                    masks[:, pos + offset] = True

    return seqs, masks

def generate_order_sensitive(n_seqs: int, seq_len: int, vocab_size: int, seed: int):
    """
    Dense order-sensitive:
    Repeating motifs (e.g. A B C D) that tile the sequence. Every token after the
    first motif is completely determinable and thus answer-bearing.
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, vocab_size, size=(n_seqs, seq_len), dtype=np.int64)
    masks = np.zeros((n_seqs, seq_len), dtype=np.bool_)

    motif_len = 4
    for i in range(n_seqs):
        motif = rng.integers(0, vocab_size, size=motif_len)
        for j in range(0, seq_len, motif_len):
            end = min(j + motif_len, seq_len)
            seqs[i, j:end] = motif[:end - j]
            if j >= motif_len:
                # The first motif is purely random context, subsequent repeats are targets
                masks[i, j:end] = True

    return seqs, masks


def generate_distance_bucket(n_seqs: int, seq_len: int, vocab_size: int, seed: int):
    """
    Distance-bucket task (paper Section 7.2):
    Places MARKER_A at a random position and MARKER_B at another random position.
    Two positions after MARKER_B, a target token encodes which distance bucket the
    pair falls into: near (0–3), medium (4–9), or far (10+).

    Three distinct bucket-indicator tokens are placed in the upper quarter of the
    vocab, away from the special marker tokens, to keep them distinguishable.
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, vocab_size - 4, size=(n_seqs, seq_len), dtype=np.int64)
    masks = np.zeros((n_seqs, seq_len), dtype=np.bool_)

    MARKER_A = vocab_size - 4
    MARKER_B = vocab_size - 3
    # Bucket indicator tokens placed in upper-middle range so they are distinctive
    # but not in the very-top-of-vocab special range
    BUCKET_BASE = vocab_size // 4  # tokens: BUCKET_BASE, BUCKET_BASE+1, BUCKET_BASE+2

    for i in range(n_seqs):
        pos_a = rng.integers(0, seq_len // 2)
        min_b = pos_a + 1
        max_b = seq_len - 3
        if min_b >= max_b:
            continue
        pos_b = rng.integers(min_b, max_b)

        dist = int(pos_b - pos_a)
        if dist <= 3:
            bucket = 0  # near
        elif dist <= 9:
            bucket = 1  # medium
        else:
            bucket = 2  # far

        seqs[i, pos_a] = MARKER_A
        seqs[i, pos_b] = MARKER_B

        target_pos = pos_b + 2
        if target_pos < seq_len:
            seqs[i, target_pos] = BUCKET_BASE + bucket
            masks[i, target_pos] = True

    return seqs, masks


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transformer sequences.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-train", type=int, default=100000, help="Total train sequences across all mixed families")
    parser.add_argument("--n-val", type=int, default=5000, help="Total val sequences across all mixed families")
    parser.add_argument("--n-test", type=int, default=5000, help="Total test sequences across all mixed families")
    parser.add_argument("--seq-len", type=int, default=64, help="Context length")
    parser.add_argument("--vocab-size", type=int, default=128, help="Vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_order = [
        "variable_lag_copy",
        "absolute_anchor",
        "order_sensitive",
        "distance_bucket",
        "iid_random",
    ]

    # Mix ratios: iid_random is only a tiny calibration fraction (no supervised targets)
    mix_ratios = {
        "variable_lag_copy": 0.40,
        "absolute_anchor":   0.20,
        "order_sensitive":   0.20,
        "distance_bucket":   0.19,
        "iid_random":        0.01,
    }

    families_fn = {
        "iid_random":        generate_iid_random,
        "variable_lag_copy": generate_variable_lag_copy,
        "absolute_anchor":   generate_absolute_anchor,
        "order_sensitive":   generate_order_sensitive,
        "distance_bucket":   generate_distance_bucket,
    }

    metadata = {
        "n_train_total": args.n_train,
        "n_val_total":   args.n_val,
        "n_test_total":  args.n_test,
        "seq_len":       args.seq_len,
        "vocab_size":    args.vocab_size,
        "seed":          args.seed,
        "mix_ratios":    mix_ratios,
        "families":      family_order,       # explicit list so downstream scripts don't rely on defaults
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for name in family_order:
        gen_fn = families_fn[name]
        ratio  = mix_ratios[name]
        n_train = max(1, int(args.n_train * ratio))
        n_val   = max(1, int(args.n_val   * ratio))
        n_test  = max(1, int(args.n_test  * ratio))

        print(f"Generating {name} (train: {n_train}, val: {n_val}, test: {n_test})...")
        base_seed = args.seed + sum(map(ord, name))
        train_seqs, train_masks = gen_fn(n_train, args.seq_len, args.vocab_size, base_seed * 1)
        val_seqs,   val_masks   = gen_fn(n_val,   args.seq_len, args.vocab_size, base_seed * 2)
        test_seqs,  test_masks  = gen_fn(n_test,  args.seq_len, args.vocab_size, base_seed * 3)

        # Save sequences
        np.save(out_dir / f"{name}_train.npy",      train_seqs)
        np.save(out_dir / f"{name}_val.npy",         val_seqs)
        np.save(out_dir / f"{name}_test.npy",        test_seqs)

        # Save corresponding boolean masks for loss-masking during training
        np.save(out_dir / f"{name}_train_mask.npy", train_masks)
        np.save(out_dir / f"{name}_val_mask.npy",   val_masks)
        np.save(out_dir / f"{name}_test_mask.npy",  test_masks)

    print(f"Data generation complete. Saved to {out_dir}")

if __name__ == "__main__":
    main()
