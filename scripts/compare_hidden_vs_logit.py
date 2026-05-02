"""
Compare hidden-state score-geometric signatures (from run_positional_analysis.py)
with logit-space signatures (from run_logit_analysis.py).

Reads:
    <hidden_dir>/analysis_summary.json     — aggregated hidden-state results
    <logit_dir>/<pe>_seed<s>_logit_analysis.json — per-(pe,seed) logit results

Produces:
    <out_dir>/logit_vs_hidden_comparison.json — structured comparison
    Console table summary
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_logit_results(logit_dir: Path, pe_types, seeds):
    """Load per-(pe, seed) logit analysis JSONs into a dict keyed by pe_type."""
    results = {}
    for pe in pe_types:
        pe_seeds = []
        for s in seeds:
            p = logit_dir / f"{pe}_seed{s}_logit_analysis.json"
            if p.exists():
                with open(p) as f:
                    pe_seeds.append(json.load(f))
        if pe_seeds:
            results[pe] = pe_seeds
    return results


def _get_mean(field):
    """Extract the mean list from a layerwise field.

    Handles both formats:
      - {"mean": [...], "std": [...]}  (aggregated summary)
      - [...]                          (flat list, legacy)
    """
    if isinstance(field, dict):
        return field.get("mean", [])
    return field


def extract_hidden_layer_metrics(summary, pe_type, layer_idx):
    """Pull metrics for a specific layer from the hidden-state analysis_summary."""
    pe_data = summary.get(pe_type, {})
    rdi_lw   = _get_mean(pe_data.get("RDI_layerwise", []))
    hr_lw    = _get_mean(pe_data.get("H_r_layerwise", []))
    ar_lw    = _get_mean(pe_data.get("A_r_layerwise", []))
    scr_lw   = _get_mean(pe_data.get("SCR_r_layerwise", []))
    beta_lw  = _get_mean(pe_data.get("beta_layerwise", []))
    dsm_lw   = _get_mean(pe_data.get("val_dsm_loss_layerwise", []))
    probe_lw = _get_mean(pe_data.get("probe_accuracy_layerwise", []))

    if layer_idx >= len(rdi_lw):
        return None

    return {
        "RDI":   rdi_lw[layer_idx],
        "H_r":   hr_lw[layer_idx] if layer_idx < len(hr_lw) else None,
        "A_r":   ar_lw[layer_idx] if layer_idx < len(ar_lw) else None,
        "beta":  beta_lw[layer_idx] if layer_idx < len(beta_lw) else None,
        "SCR_r": scr_lw[layer_idx] if layer_idx < len(scr_lw) else None,
        "val_dsm_loss":   dsm_lw[layer_idx] if layer_idx < len(dsm_lw) else None,
        "probe_accuracy": probe_lw[layer_idx] if layer_idx < len(probe_lw) else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare hidden-state vs logit-space score-geometric signatures."
    )
    parser.add_argument("--hidden-summary", type=str, required=True,
                        help="Path to analysis_summary.json from hidden-state pipeline")
    parser.add_argument("--logit-dir", type=str, required=True,
                        help="Directory containing *_logit_analysis.json files")
    parser.add_argument("--out-dir",   type=str, required=True)
    parser.add_argument("--pe-types",  nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=[0])
    parser.add_argument("--compare-layer", type=int, default=-1,
                        help="Hidden-state layer to compare against logits "
                             "(-1 = last layer, i.e. L4 in 4-layer model)")

    args = parser.parse_args()

    with open(args.hidden_summary) as f:
        hidden_summary = json.load(f)

    logit_dir = Path(args.logit_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_results = load_logit_results(logit_dir, args.pe_types, args.seeds)

    # Determine comparison layer index
    # In analysis_summary, layers are 1-indexed (L1..L4), stored as list index 0..3
    sample_pe = args.pe_types[0]
    n_layers = len(hidden_summary.get(sample_pe, {}).get("RDI_layerwise", []))
    compare_idx = args.compare_layer if args.compare_layer >= 0 else (n_layers - 1)
    compare_layer_name = f"L{compare_idx + 1}"

    print(f"\nComparing hidden-state {compare_layer_name} vs logit space")
    print(f"{'='*65}")

    comparison = {}

    for pe in args.pe_types:
        hidden_metrics = extract_hidden_layer_metrics(hidden_summary, pe, compare_idx)
        if hidden_metrics is None:
            print(f"  {pe}: no hidden-state data at layer index {compare_idx}")
            continue

        if pe not in logit_results:
            print(f"  {pe}: no logit analysis found")
            continue

        # Average logit metrics across seeds
        logit_seeds = logit_results[pe]
        logit_rdi   = np.mean([r["RDI"] for r in logit_seeds])
        logit_hr    = np.mean([np.mean(r["H_r"]) for r in logit_seeds])
        logit_ar    = np.mean([np.mean(r["A_r"]) for r in logit_seeds])
        logit_beta  = np.mean([r["beta"] for r in logit_seeds])
        logit_dsm   = np.mean([r["val_dsm_loss"] for r in logit_seeds])
        logit_probe = np.mean([r["probe_accuracy"] for r in logit_seeds])
        logit_scr1  = np.mean([r["SCR_r"][1][0] for r in logit_seeds])  # SCR at lag=1, top-1

        hidden_hr  = np.mean(hidden_metrics["H_r"]) if hidden_metrics["H_r"] is not None else None
        hidden_ar  = np.mean(hidden_metrics["A_r"]) if hidden_metrics["A_r"] is not None else None
        hidden_scr1 = hidden_metrics["SCR_r"][1][0] if hidden_metrics["SCR_r"] is not None else None

        entry = {
            "hidden_layer": compare_layer_name,
            "hidden": {
                "RDI":            hidden_metrics["RDI"],
                "H_r_mean":       hidden_hr,
                "A_r_mean":       hidden_ar,
                "beta":           hidden_metrics["beta"],
                "val_dsm_loss":   hidden_metrics["val_dsm_loss"],
                "probe_accuracy": hidden_metrics["probe_accuracy"],
                "SCR_lag1_top1":  hidden_scr1,
            },
            "logit": {
                "RDI":            float(logit_rdi),
                "H_r_mean":       float(logit_hr),
                "A_r_mean":       float(logit_ar),
                "beta":           float(logit_beta),
                "val_dsm_loss":   float(logit_dsm),
                "probe_accuracy": float(logit_probe),
                "SCR_lag1_top1":  float(logit_scr1),
            },
            "n_logit_seeds": len(logit_seeds),
        }
        comparison[pe] = entry

        # Print table row
        print(f"\n  {pe.upper()}")
        print(f"  {'Metric':<20} {'Hidden ' + compare_layer_name:>15} {'Logit':>15}")
        print(f"  {'-'*50}")
        print(f"  {'RDI':<20} {hidden_metrics['RDI']:>15.4f} {logit_rdi:>15.4f}")
        if hidden_hr is not None:
            print(f"  {'H_r (mean)':<20} {hidden_hr:>15.4f} {logit_hr:>15.4f}")
        if hidden_ar is not None:
            print(f"  {'A_r (mean)':<20} {hidden_ar:>15.4f} {logit_ar:>15.4f}")
        if hidden_metrics["beta"] is not None:
            print(f"  {'beta':<20} {hidden_metrics['beta']:>15.4f} {logit_beta:>15.4f}")
        if hidden_metrics["val_dsm_loss"] is not None:
            print(f"  {'DSM val loss':<20} {hidden_metrics['val_dsm_loss']:>15.4f} {logit_dsm:>15.4f}")
        if hidden_metrics["probe_accuracy"] is not None:
            print(f"  {'Probe accuracy':<20} {hidden_metrics['probe_accuracy']:>15.4f} {logit_probe:>15.4f}")

    # ----------------------------------------------------------------
    # Cross-PE discrimination analysis
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Cross-PE discrimination in logit space")
    print(f"{'='*65}")

    pe_rdis = {}
    for pe in args.pe_types:
        if pe in logit_results:
            pe_rdis[pe] = np.mean([r["RDI"] for r in logit_results[pe]])

    if len(pe_rdis) >= 2:
        pes = sorted(pe_rdis.keys())
        for i in range(len(pes)):
            for j in range(i+1, len(pes)):
                diff = abs(pe_rdis[pes[i]] - pe_rdis[pes[j]])
                print(f"  |RDI({pes[i]}) - RDI({pes[j]})| = {diff:.4f}")

        rdi_range = max(pe_rdis.values()) - min(pe_rdis.values())
        print(f"\n  RDI range across PE types: {rdi_range:.4f}")
        print(f"  (hidden {compare_layer_name} range: "
              f"{max(comparison[pe]['hidden']['RDI'] for pe in comparison) - min(comparison[pe]['hidden']['RDI'] for pe in comparison):.4f})")

    # Save
    output = {
        "compare_layer": compare_layer_name,
        "pe_comparison": comparison,
        "logit_pe_rdis": {k: float(v) for k, v in pe_rdis.items()},
    }
    out_path = out_dir / "logit_vs_hidden_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
