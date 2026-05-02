"""
Compare logit-space score-geometric signatures between controlled toy models
(Phase 1) and pretrained HF models (Phase 2).

The key question: do pretrained models' logit signatures cluster with the
correct PE family from the controlled experiments?

Reads:
    <phase1_dir>/<pe>_seed<s>_logit_analysis.json    — controlled model results
    <phase2_dir>/<model_key>_logit_analysis.json      — pretrained model results
    <phase2_dir>/model_registry.json                  — model_key -> pe_type mapping

Produces:
    <out_dir>/controlled_vs_pretrained_comparison.json
    Console: metric distances, nearest-neighbor classification, confusion matrix
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_phase1_results(phase1_dir: Path, pe_types, seeds):
    """Load Phase 1 (controlled) logit analysis results."""
    results = {}
    for pe in pe_types:
        pe_seeds = []
        for s in seeds:
            p = phase1_dir / f"{pe}_seed{s}_logit_analysis.json"
            if p.exists():
                with open(p) as f:
                    pe_seeds.append(json.load(f))
        if pe_seeds:
            results[pe] = pe_seeds
    return results


def load_phase2_results(phase2_dir: Path, model_keys):
    """Load Phase 2 (pretrained) logit analysis results."""
    results = {}
    for mk in model_keys:
        p = phase2_dir / f"{mk}_logit_analysis.json"
        if p.exists():
            with open(p) as f:
                results[mk] = json.load(f)
    return results


def extract_signature_vector(result):
    """Extract a compact signature vector from analysis result.

    Returns (RDI, mean_H_r, mean_A_r, beta, SCR_lag1_top1) as ndarray.
    """
    rdi  = result["RDI"]
    hr   = np.mean(result["H_r"])
    ar   = np.mean(result["A_r"])
    beta = result["beta"]
    scr  = result["SCR_r"][1][0] if len(result["SCR_r"]) > 1 else 0.0
    return np.array([rdi, hr, ar, beta, scr])


def main():
    parser = argparse.ArgumentParser(
        description="Compare controlled vs pretrained logit-space signatures."
    )
    parser.add_argument("--phase1-dir", type=str, required=True,
                        help="Directory with Phase 1 logit analysis JSONs")
    parser.add_argument("--phase2-dir", type=str, required=True,
                        help="Directory with Phase 2 logit analysis JSONs")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--models", nargs="+", default=["gpt2", "bloom-560m", "pythia-410m"])

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = load_phase1_results(Path(args.phase1_dir), args.pe_types, args.seeds)
    phase2 = load_phase2_results(Path(args.phase2_dir), args.models)

    # Load registry for ground-truth PE types
    registry_path = Path(args.phase2_dir) / "model_registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)

    # Compute Phase 1 centroid signatures per PE type
    print(f"\n{'='*65}")
    print("Phase 1 — Controlled model signature centroids")
    print(f"{'='*65}")

    pe_centroids = {}
    sig_names = ["RDI", "H_r_mean", "A_r_mean", "beta", "SCR_lag1_top1"]

    for pe in args.pe_types:
        if pe not in phase1:
            continue
        sigs = [extract_signature_vector(r) for r in phase1[pe]]
        centroid = np.mean(sigs, axis=0)
        pe_centroids[pe] = centroid
        print(f"\n  {pe.upper()} (n={len(sigs)} seeds):")
        for name, val in zip(sig_names, centroid):
            print(f"    {name:<15} = {val:.6f}")

    # Phase 2 signatures
    print(f"\n{'='*65}")
    print("Phase 2 — Pretrained model signatures")
    print(f"{'='*65}")

    pretrained_sigs = {}
    pretrained_pe_types = {}

    for mk in args.models:
        if mk not in phase2:
            print(f"  {mk}: not found")
            continue
        sig = extract_signature_vector(phase2[mk])
        pretrained_sigs[mk] = sig
        pretrained_pe_types[mk] = phase2[mk].get("pe_type", registry.get(mk, {}).get("pe_type", "unknown"))

        print(f"\n  {mk} (PE={pretrained_pe_types[mk]}):")
        for name, val in zip(sig_names, sig):
            print(f"    {name:<15} = {val:.6f}")

    # Distance matrix and nearest-neighbor classification
    print(f"\n{'='*65}")
    print("Distance matrix (Euclidean in signature space)")
    print(f"{'='*65}")

    # Normalize signatures for fair distance computation
    all_sigs = list(pe_centroids.values()) + list(pretrained_sigs.values())
    all_sigs_arr = np.array(all_sigs)
    sig_mean = all_sigs_arr.mean(axis=0)
    sig_std  = all_sigs_arr.std(axis=0) + 1e-8

    pe_centroids_norm = {k: (v - sig_mean) / sig_std for k, v in pe_centroids.items()}
    pretrained_sigs_norm = {k: (v - sig_mean) / sig_std for k, v in pretrained_sigs.items()}

    # Print distance table
    pe_list = sorted(pe_centroids_norm.keys())
    header = f"  {'Model':<15}" + "".join(f"  {pe.upper():>10}" for pe in pe_list)
    print(header)
    print(f"  {'-'*len(header)}")

    classification_results = {}
    for mk in args.models:
        if mk not in pretrained_sigs_norm:
            continue
        sig = pretrained_sigs_norm[mk]
        distances = {}
        for pe in pe_list:
            dist = np.linalg.norm(sig - pe_centroids_norm[pe])
            distances[pe] = float(dist)

        nearest = min(distances, key=distances.get)
        true_pe = pretrained_pe_types.get(mk, "unknown")
        correct = (nearest == true_pe)

        row = f"  {mk:<15}" + "".join(f"  {distances[pe]:>10.4f}" for pe in pe_list)
        marker = " <-- correct" if correct else f" <-- WRONG (true={true_pe})"
        print(f"{row}  -> {nearest.upper()}{marker}")

        classification_results[mk] = {
            "true_pe":      true_pe,
            "predicted_pe": nearest,
            "correct":      correct,
            "distances":    distances,
        }

    # Summary
    n_correct = sum(1 for r in classification_results.values() if r["correct"])
    n_total = len(classification_results)

    print(f"\n{'='*65}")
    print(f"Classification accuracy: {n_correct}/{n_total}")
    if n_total > 0:
        print(f"  ({100 * n_correct / n_total:.0f}%)")
    print(f"{'='*65}")

    # Per-metric comparison table
    print(f"\n{'='*65}")
    print("Per-metric comparison (raw values)")
    print(f"{'='*65}")

    for metric_idx, metric_name in enumerate(sig_names):
        print(f"\n  {metric_name}:")
        print(f"    {'Source':<20}", end="")
        for pe in pe_list:
            print(f"  {pe.upper():>12}", end="")
        print()

        # Controlled
        print(f"    {'Controlled':<20}", end="")
        for pe in pe_list:
            print(f"  {pe_centroids[pe][metric_idx]:>12.6f}", end="")
        print()

        # Pretrained (grouped by PE type)
        pe_to_models = defaultdict(list)
        for mk, pe in pretrained_pe_types.items():
            if mk in pretrained_sigs:
                pe_to_models[pe].append(mk)

        for pe in pe_list:
            for mk in pe_to_models.get(pe, []):
                print(f"    {mk:<20}", end="")
                val = pretrained_sigs[mk][metric_idx]
                for pe2 in pe_list:
                    if pe2 == pe:
                        print(f"  {val:>12.6f}", end="")
                    else:
                        print(f"  {'':>12}", end="")
                print()

    # Save results
    output = {
        "phase1_centroids": {k: v.tolist() for k, v in pe_centroids.items()},
        "phase2_signatures": {k: v.tolist() for k, v in pretrained_sigs.items()},
        "phase2_pe_types": pretrained_pe_types,
        "signature_names": sig_names,
        "normalization": {"mean": sig_mean.tolist(), "std": sig_std.tolist()},
        "classification": classification_results,
        "accuracy": n_correct / n_total if n_total > 0 else None,
    }
    out_path = out_dir / "controlled_vs_pretrained_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
