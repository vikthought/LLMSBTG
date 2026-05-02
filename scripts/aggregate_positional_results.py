"""
Aggregate per-(pe_type, seed) analysis JSON files produced by
run_positional_analysis.py into a single analysis_summary.json.

Also computes cross-seed statistics (mean ± std) for every scalar
summary (A_r, H_r, RDI, beta, probe_accuracy, SCR_r).

Usage
-----
python scripts/aggregate_positional_results.py --out-dir results/transformer_pos_analysis_<timestamp>
"""

import argparse
import json
import numpy as np
from pathlib import Path


def _collect_per_layer_stat(records: list, key: str) -> dict:
    """
    records : list of per-seed dicts, each with 'layer_stats' list.
    Returns { 'mean': [...], 'std': [...] } — one entry per layer.
    """
    n_layers = len(records[0]["layer_stats"])
    vals_per_layer = [[] for _ in range(n_layers)]
    for rec in records:
        for l_idx, ls in enumerate(rec["layer_stats"]):
            v = ls[key]
            vals_per_layer[l_idx].append(np.array(v))
    means = [np.mean(vs, axis=0).tolist() for vs in vals_per_layer]
    stds  = [np.std( vs, axis=0).tolist() for vs in vals_per_layer]
    return {"mean": means, "std": stds}


def _collect_bootstrap_ci(records: list) -> dict:
    """
    Collect and aggregate bootstrap CI dicts across seeds.

    Each layer_stats entry may have 'bootstrap_ci' with keys
    RDI_ci, H_r_ci, A_r_ci, beta_ci.  Each CI dict has
    {'lower': ..., 'upper': ..., 'mean': ...} where values
    are scalars or lists.

    Returns per-stat aggregated CIs:
      {
        'RDI_ci':  { 'lower': [mean_lower_l0, ...],
                     'upper': [mean_upper_l0, ...],
                     'mean':  [mean_mean_l0,  ...] },
        ...
      }
    """
    n_layers = len(records[0]["layer_stats"])

    # Gather all per-seed CIs, keyed by stat name
    ci_keys = ["RDI_ci", "H_r_ci", "A_r_ci", "beta_ci"]
    per_stat: dict[str, dict[str, list]] = {
        k: {"lower": [[] for _ in range(n_layers)],
            "upper": [[] for _ in range(n_layers)],
            "mean":  [[] for _ in range(n_layers)]}
        for k in ci_keys
    }

    has_any = False
    for rec in records:
        for l_idx, ls in enumerate(rec["layer_stats"]):
            bci = ls.get("bootstrap_ci")
            if bci is None:
                continue
            has_any = True
            for ck in ci_keys:
                if ck not in bci:
                    continue
                for bound in ("lower", "upper", "mean"):
                    v = bci[ck].get(bound)
                    if v is not None:
                        per_stat[ck][bound][l_idx].append(np.array(v))

    if not has_any:
        return {}

    # Aggregate: mean over seeds per layer
    result: dict = {}
    for ck in ci_keys:
        agg: dict[str, list] = {}
        for bound in ("lower", "upper", "mean"):
            agg[bound] = []
            for l_idx in range(n_layers):
                vs = per_stat[ck][bound][l_idx]
                if vs:
                    agg[bound].append(np.mean(vs, axis=0).tolist())
                else:
                    agg[bound].append(None)
        result[ck] = agg

    return result


def _collect_val_dsm_loss(records: list) -> dict:
    """
    Collect val_dsm_loss across seeds per layer.
    Returns {'mean': [l0, l1, ...], 'std': [l0, l1, ...], 'per_seed': [[l0,l1,...], ...]}.
    """
    n_layers = len(records[0]["layer_stats"])
    vals_per_layer = [[] for _ in range(n_layers)]
    for rec in records:
        for l_idx, ls in enumerate(rec["layer_stats"]):
            v = ls.get("val_dsm_loss")
            if v is not None:
                vals_per_layer[l_idx].append(float(v))

    if not any(vals_per_layer):
        return {}

    means = [float(np.mean(vs)) if vs else None for vs in vals_per_layer]
    stds  = [float(np.std(vs))  if vs else None for vs in vals_per_layer]
    return {"mean": means, "std": stds, "per_seed": vals_per_layer}


def aggregate(out_dir: Path) -> dict:
    pattern = "*_seed*_analysis.json"
    files   = sorted(out_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No '*_seed*_analysis.json' files found in {out_dir}.\n"
            "Have you run run_positional_analysis.py first?"
        )

    # Group by pe_type
    by_pe: dict[str, list] = {}
    for fp in files:
        with open(fp) as f:
            rec = json.load(f)
        pe = rec["pe_type"]
        by_pe.setdefault(pe, []).append(rec)

    summary = {}
    for pe_type, records in by_pe.items():
        records_sorted = sorted(records, key=lambda r: r["seed"])
        n_seeds        = len(records_sorted)
        n_layers       = len(records_sorted[0]["layer_stats"])

        entry = {
            "pe_type":  pe_type,
            "n_seeds":  n_seeds,
            "seeds":    [r["seed"] for r in records_sorted],
            "pca_dim":  records_sorted[0]["pca_dim"],
            "w":        records_sorted[0]["w"],
            "max_lag":  records_sorted[0]["max_lag"],
            "best_hps": [r["best_hp"] for r in records_sorted],
        }

        # Per-layer scalar statistics across seeds
        for stat_key in ("A_r", "H_r", "RDI", "beta", "probe_accuracy", "SCR_r", "M_bar_r",
                         "delta_frob", "M_frob"):
            try:
                entry[f"{stat_key}_layerwise"] = _collect_per_layer_stat(records_sorted, stat_key)
            except (KeyError, IndexError):
                pass   # stat may not be present in all versions

        # Bootstrap CIs (B4)
        bci = _collect_bootstrap_ci(records_sorted)
        if bci:
            entry["bootstrap_ci_layerwise"] = bci

        # Validation DSM loss per layer across seeds (D5)
        vdl = _collect_val_dsm_loss(records_sorted)
        if vdl:
            entry["val_dsm_loss_layerwise"] = vdl

        # Attention stats (if saved — shape per record: per-layer/head arrays)
        attn_all = [r.get("attn_stats") for r in records_sorted if r.get("attn_stats")]
        if attn_all:
            entry["attn_stats_seeds"] = attn_all

        # Per-seed raw layer_stats (kept for detailed downstream analysis)
        entry["per_seed"] = records_sorted

        summary[pe_type] = entry

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory containing *_seed*_analysis.json files "
                             "(same dir passed to run_positional_analysis.py)")
    args   = parser.parse_args()
    out_dir = Path(args.out_dir)

    print(f"Aggregating results in {out_dir} …")
    summary = aggregate(out_dir)

    out_path = out_dir / "analysis_summary.json"
    with open(out_path, "w") as f:
        def _default(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.generic):  return obj.item()
            raise TypeError(type(obj))
        json.dump(summary, f, default=_default, indent=2)

    pe_types = list(summary.keys())
    print(f"Done. {len(pe_types)} PE type(s): {pe_types}")
    for pe, data in summary.items():
        seeds = data["seeds"]
        print(f"  {pe}: seeds {seeds}")
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
