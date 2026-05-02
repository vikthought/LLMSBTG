"""
RoPE base sweep — finalize Phase 4 analysis on already-trained data.

Designed for the case where cluster_rope_base_sweep.sh ran out of wall time
after Phases 1-3 (training + per-base lagpair + per-base logit) but before
Phase 4 (cross-base aggregation).  Uses no GPU; reads only small JSONs.

What it does
------------
1. Auto-detects which bases under <sweep-root> have the required artifacts:
     - <sweep-root>/base_{B}/lagpair_analysis/lagpair_metrics.json
     - <sweep-root>/base_{B}/logit_analysis/rope_seed{N}_logit_analysis.json
     - <sweep-root>/base_{B}/rope_seed{N}/training_history.json
2. Calls compare_rope_bases.py via its main() (Phase 4 logic) on the
   completed bases — produces rope_base_summary.json + 5 figures.
3. Bundles the source JSONs into <sweep-root>/analysis/per_base/ so the
   whole analysis/ directory is self-contained and easy to scp.
4. Writes a small README to <sweep-root>/analysis/ describing the layout
   and the four predictions the figures address.

Usage
-----
    python scripts/run_rope_sweep_finalize.py \\
        --sweep-root results/rope_base_sweep \\
        [--bases 10 100 1000 10000 100000 1000000]   # auto-detected if omitted
        [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def find_completed_bases(
    sweep_root: Path,
    candidate_bases: List[int],
    seeds: List[int],
) -> Tuple[List[int], dict]:
    """For each candidate base, check artifact completeness.  Return the list
    of bases with at least lagpair_metrics.json present (logit + training are
    optional — figures degrade gracefully)."""
    completed: List[int] = []
    status = {}
    for base in candidate_bases:
        bd = sweep_root / f"base_{base}"
        lp = bd / "lagpair_analysis" / "lagpair_metrics.json"
        n_logit = sum(
            1 for s in seeds
            if (bd / "logit_analysis" / f"rope_seed{s}_logit_analysis.json").exists()
        )
        n_training = sum(
            1 for s in seeds
            if (bd / f"rope_seed{s}" / "training_history.json").exists()
        )
        info = {
            "lagpair_metrics_present": lp.exists(),
            "n_logit_seeds": n_logit,
            "n_training_seeds": n_training,
        }
        status[base] = info
        if info["lagpair_metrics_present"]:
            completed.append(base)
    return completed, status


def autodetect_bases(sweep_root: Path) -> List[int]:
    """Find directories matching base_<int> under sweep_root."""
    found = []
    for child in sweep_root.iterdir():
        if not child.is_dir() or not child.name.startswith("base_"):
            continue
        try:
            found.append(int(child.name[len("base_"):]))
        except ValueError:
            continue
    return sorted(found)


def bundle_source_jsons(
    sweep_root: Path,
    out_dir: Path,
    bases: List[int],
    seeds: List[int],
) -> dict:
    """Copy the small JSONs into out_dir/per_base/base_{B}/ so the analysis
    directory is self-contained for download.  Returns a manifest describing
    what was copied."""
    manifest = {}
    for base in bases:
        src = sweep_root / f"base_{base}"
        dst = out_dir / "per_base" / f"base_{base}"
        dst.mkdir(parents=True, exist_ok=True)
        copied = []

        # lagpair_metrics.json
        lp = src / "lagpair_analysis" / "lagpair_metrics.json"
        if lp.exists():
            shutil.copy2(lp, dst / "lagpair_metrics.json")
            copied.append("lagpair_metrics.json")

        # logit per-seed JSONs
        for s in seeds:
            lj = src / "logit_analysis" / f"rope_seed{s}_logit_analysis.json"
            if lj.exists():
                shutil.copy2(lj, dst / f"logit_seed{s}.json")
                copied.append(f"logit_seed{s}.json")

        # training history per-seed
        for s in seeds:
            th = src / f"rope_seed{s}" / "training_history.json"
            if th.exists():
                shutil.copy2(th, dst / f"training_history_seed{s}.json")
                copied.append(f"training_history_seed{s}.json")

        # attn_stats per-seed (small, sometimes useful for the prediction-1
        # diagnostic at base=10 — not required by Phase 4 figures)
        for s in seeds:
            asd = src / f"rope_seed{s}" / "attn_stats.json"
            if asd.exists():
                shutil.copy2(asd, dst / f"attn_stats_seed{s}.json")
                copied.append(f"attn_stats_seed{s}.json")

        manifest[base] = copied
    return manifest


README_TEMPLATE = """# RoPE base sweep — analysis bundle

Self-contained output of the Phase 4 cross-base aggregation, suitable for
downloading via `scp -r` and re-analyzing locally.

## Why this experiment

Vary RoPE's base parameter across {bases} — keeping training data, seeds,
and architecture identical — and test four predictions:

1. **Dip depth at $r \\in \\{{6, 7\\}}$ at L4 monotonically decreases with
   $\\log(\\text{{base}})$.** (figures: `RB3`, `RB4`)
2. **$C_1$ at L4 is monotonically non-decreasing in $\\log(\\text{{base}})$**:
   larger base → fewer effective rotation dimensions → coupling concentrates.
   (figure: `RB2`)
3. **Logit RDI is unimodal in $\\log(\\text{{base}})$, peaking near $10^4$**:
   driven by task performance rather than RoPE geometry directly. (`RB5`)
4. **Validation loss on `variable_lag_copy` worst at base=10**: severe
   aliasing across all dimensions. (`RB1`)

The decisive figure is **RB3** (`figures/RB3_as_r_envelope_overlay.pdf`):
empirical $\\mathrm{{AS}}_r/\\mathrm{{AS}}_1$ at L4 vs theoretical
$\\rho(r)/\\rho(1)$, both varying base.  If the empirical curve tracks
theory in shape ordering across all bases, the paper's $\\rho(r)$ claim
generalizes from one base to a parametric law.

## Layout

```
analysis/
├── README.md                              # this file
├── rope_base_summary.json                 # consolidated cross-base summary
├── figures/
│   ├── RB1_val_loss_vs_base.pdf           # task losses (prediction 4)
│   ├── RB2_sbtg_scalars_vs_base.pdf       # SI/A/S/C/AS at lag 1 (prediction 2)
│   ├── RB3_as_r_envelope_overlay.pdf      # empirical vs theoretical (decisive)
│   ├── RB4_dip_depth_vs_base.pdf          # dip depth (prediction 1)
│   └── RB5_logit_metrics_vs_base.pdf      # logit RDI etc. (prediction 3)
├── per_base/
│   ├── base_{example_base}/
│   │   ├── lagpair_metrics.json           # SI, A_r, S_r, C_r, AS_r per seed/layer/lag
│   │   ├── logit_seed{{0,1,2}}.json         # per-seed logit metrics
│   │   ├── training_history_seed{{0,1,2}}.json   # per-epoch per-family val loss
│   │   └── attn_stats_seed{{0,1,2}}.json    # mean attention dist + entropy
│   └── ... (one folder per completed base)
└── manifest.json                          # what was found per base
```

## Per-base completeness

See `manifest.json` in this directory.  A base appears in the figures iff
`lagpair_metrics.json` is present; logit/training are optional.

## Re-running locally

If you want to recompute the figures with different bases, $d_{{\\text{{head}}}}$,
or seed-subsets:

```bash
python scripts/compare_rope_bases.py \\
    --sweep-root <path-to-this-bundle's-parent> \\
    --bases {bases} \\
    --seeds 0 1 2 \\
    --out-dir <new-output-dir>
```

(Requires the `per_base/` mirror at the layout above, OR the original
`base_{{B}}/` tree from the cluster.)
"""


def write_readme(out_dir: Path, bases: List[int]) -> None:
    bases_str = "{" + ", ".join(str(b) for b in bases) + "}"
    bases_inline = " ".join(str(b) for b in bases)
    readme = README_TEMPLATE.format(
        bases=bases_str,
        bases_inline=bases_inline,
        example_base=bases[0] if bases else 10000,
    )
    # The template uses literal command line, replace placeholder
    readme = readme.replace("{bases}", bases_inline, 1)
    (out_dir / "README.md").write_text(readme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=str,
                   default="results/rope_base_sweep",
                   help="Root containing base_<B>/ subdirectories")
    p.add_argument("--bases", nargs="*", type=int, default=None,
                   help="Bases to include (default: auto-detect)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write analysis/ (default: <sweep-root>/analysis)")
    p.add_argument("--d-head", type=int, default=64,
                   help="head dim for theoretical envelope")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    if not sweep_root.is_dir():
        sys.exit(f"ERROR: sweep root does not exist: {sweep_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else sweep_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Detect bases ------------------------------------------------------
    candidates = args.bases if args.bases else autodetect_bases(sweep_root)
    if not candidates:
        sys.exit(f"ERROR: no base_* directories found under {sweep_root}")
    print(f"Candidate bases: {candidates}")

    completed, status = find_completed_bases(sweep_root, candidates, args.seeds)
    print()
    print("Per-base artifact status:")
    print(f"  {'base':>10}  {'lagpair':>8}  {'logit':>6}  {'training':>9}")
    for base in candidates:
        s = status[base]
        print(f"  {base:>10}  {'YES' if s['lagpair_metrics_present'] else 'no':>8}  "
              f"{s['n_logit_seeds']}/{len(args.seeds):<2}  "
              f"{s['n_training_seeds']}/{len(args.seeds):<2}")

    if not completed:
        sys.exit("ERROR: no base has lagpair_metrics.json — nothing to aggregate.")

    missing = [b for b in candidates if b not in completed]
    if missing:
        print(f"\nProceeding with {len(completed)} completed bases; "
              f"skipping {missing} (no lagpair_metrics.json).")

    # ---- Run Phase 4 aggregation via compare_rope_bases.main() -------------
    print("\nRunning cross-base aggregation (compare_rope_bases.py logic)...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compare_rope_bases as crb

    # Patch sys.argv so its argparse main() picks up our parameters.
    saved_argv = sys.argv
    sys.argv = [
        "compare_rope_bases.py",
        "--sweep-root", str(sweep_root),
        "--bases", *(str(b) for b in completed),
        "--seeds", *(str(s) for s in args.seeds),
        "--out-dir", str(out_dir),
        "--d-head", str(args.d_head),
    ]
    try:
        crb.main()
    finally:
        sys.argv = saved_argv

    # ---- Bundle source JSONs for self-contained download -------------------
    print("\nBundling source JSONs for self-contained download...")
    manifest = bundle_source_jsons(sweep_root, out_dir, completed, args.seeds)
    (out_dir / "manifest.json").write_text(json.dumps({
        "bases_included": completed,
        "bases_skipped": missing,
        "seeds": args.seeds,
        "d_head": args.d_head,
        "per_base_files": manifest,
        "status": status,
    }, indent=2))

    # ---- README ------------------------------------------------------------
    write_readme(out_dir, completed)

    # ---- Summary -----------------------------------------------------------
    print()
    print("=" * 60)
    print(" Finalize done")
    print("=" * 60)
    print(f"  Output:  {out_dir}")
    print(f"  Bases:   {completed}")
    print(f"  Files:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            print(f"    {rel}")
    print()
    print(f"  Total bytes: "
          f"{sum(f.stat().st_size for f in out_dir.rglob('*') if f.is_file()):,}")


if __name__ == "__main__":
    main()
