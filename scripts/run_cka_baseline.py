import argparse
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from sbtg.evaluation.cka import linear_cka

def parse_args():
    parser = argparse.ArgumentParser(description="Computes Centered Kernel Alignment (CKA) between layers and architectures.")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing tuned model folders (where test_acts.npy is stored)")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save the resulting cka_summary.json")
    parser.add_argument("--pe-types", nargs="+", default=["rope", "alibi", "absolute"], help="List of PE architectures to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of random seeds targeted")
    return parser.parse_args()

def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cka_results = {}
    acts_cache = {}

    print("Loading test_acts.npy across all seeds...")
    # 1. Load Acts array memory cache
    for pe in args.pe_types:
        acts_cache[pe] = {}
        for seed in args.seeds:
            folder = models_dir / f"{pe}_seed{seed}"
            acts_path = folder / "test_acts.npy"
            if acts_path.exists():
                acts = np.load(acts_path)
                # Ensure consistency (N_test, n_layers+1, seq_len, hidden_size)
                acts_cache[pe][seed] = acts
            else:
                print(f"Warning: {acts_path} not found. Skipping {pe} seed {seed}.")
    
    print("Computing Cross-Layer CKA mappings...")
    # 2. Cross-layer within same Architecture (PE)
    for pe in args.pe_types:
        cka_results[pe] = {
            "cross_layer": {},
            "cross_pe": {}
        }
        
        layer_ckas = {}
        for seed in args.seeds:
            if seed not in acts_cache[pe]: continue
            acts = acts_cache[pe][seed]
            n_layers = acts.shape[1]
            
            # Cross comb of layer interactions
            for l1, l2 in combinations(range(n_layers), 2):
                pair = f"l{l1}_l{l2}"
                if pair not in layer_ckas:
                    layer_ckas[pair] = []
                
                # Flatten the test set items into aligned (batch*seq_len) arrays for comprehensive alignment probing
                A1 = acts[:, l1, ...].reshape(-1, acts.shape[-1])
                A2 = acts[:, l2, ...].reshape(-1, acts.shape[-1])
                
                val = linear_cka(A1, A2)
                layer_ckas[pair].append(val)
                
        # Aggregate logic
        for pair, vals in layer_ckas.items():
            cka_results[pe]["cross_layer"][pair] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    print("Computing Cross-Architecture CKA interactions...")
    # 3. Cross-Architecture (PE) at the matching corresponding seed and layer depths
    pe_pairs = list(combinations(args.pe_types, 2))
    for pe1, pe2 in pe_pairs:
        pe_pair_key = f"{pe1}_vs_{pe2}"
        layer_cross_ckas = {}
        for seed in args.seeds:
            if seed not in acts_cache[pe1] or seed not in acts_cache[pe2]: continue
            acts1 = acts_cache[pe1][seed]
            acts2 = acts_cache[pe2][seed]
            
            n_layers = acts1.shape[1]
            for l in range(n_layers):
                layer_key = f"l{l}"
                if layer_key not in layer_cross_ckas:
                    layer_cross_ckas[layer_key] = []
                
                # Flatten test items (batch*seq_len) per layer architecture
                A1 = acts1[:, l, ...].reshape(-1, acts1.shape[-1])
                A2 = acts2[:, l, ...].reshape(-1, acts2.shape[-1])
                
                val = linear_cka(A1, A2)
                layer_cross_ckas[layer_key].append(val)
        
        # Mirror outputs to both architectures for quick lookups later
        for layer_key, vals in layer_cross_ckas.items():
            stat = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            if pe_pair_key not in cka_results[pe1]["cross_pe"]:
                cka_results[pe1]["cross_pe"][pe_pair_key] = {}
            if pe_pair_key not in cka_results[pe2]["cross_pe"]:
                cka_results[pe2]["cross_pe"][pe_pair_key] = {}
                
            cka_results[pe1]["cross_pe"][pe_pair_key][layer_key] = stat
            cka_results[pe2]["cross_pe"][pe_pair_key][layer_key] = stat

    out_path = out_dir / "cka_summary.json"
    with open(out_path, "w") as f:
        json.dump(cka_results, f, indent=4)
    print(f"✅ Saved analytical CKA summary to {out_path}")

if __name__ == "__main__":
    main()
