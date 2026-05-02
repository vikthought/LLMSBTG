#!/bin/bash
#SBATCH --job-name=gauss_baseline
#SBATCH --output=logs/gauss_baseline_%j.out
#SBATCH --error=logs/gauss_baseline_%j.err
#SBATCH -p week
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ===========================================================================
# Parametric Gaussian baseline for the SBTG positional operator
# ===========================================================================
#
# Addresses the reviewer-pre-emptive question: "what is the score model
# adding over a parametric Gaussian baseline?"  For each (PE, seed, layer)
# cell we already have, this script fits a multivariate Gaussian per
# window endpoint, reads M_r from the precision-matrix block
# Λ_{w-1, w-r-1}, runs the same scalar diagnostics through the same helper
# (compute_extended_metrics), and writes a JSON in the same schema as
# lagpair_metrics.json.  Then compare_score_vs_gaussian.py renders the
# side-by-side table and figures.
#
# CPU-only (no GPU), no large compute: numpy einsums + 36 cells × ~50
# endpoints × (mw=512)³ matrix inversion ≈ 90 seconds of real work.  The
# 2 h walltime is a buffer for activation-loading I/O.
#
# Submit (after cluster_lagpair_3seed.sh + lp3 chain Phase 0 outputs are
# in place):
#
#   sbatch scripts/cluster_lagpair_gaussian_baseline.sh
#
# Output: results/lagpair_gaussian_baseline_3seed/gaussian_lagpair_metrics.json
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m pip install -e ".[all]" --quiet

# ---------------------------------------------------------------------------
# Configuration — match the 3-seed pipeline's settings exactly
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"

# The per-seed analysis JSONs live in lagpair_ablation_3seed/analysis/
# (produced by the lp3 chain Phase 0 / monolithic ablation Phase 0).
ANALYSIS_DIR="results/lagpair_ablation_3seed/analysis"

OUT_DIR="results/lagpair_gaussian_baseline_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
LAYERS="1 2 3 4"

W=16
MAX_LAG=14
PCA_DIM=32
SKIP_EDGES=4
RIDGE=1e-3

mkdir -p "$OUT_DIR" logs

echo "============================================================"
echo " Gaussian baseline for SBTG operator"
echo " Models:    $MODELS_DIR"
echo " Analysis:  $ANALYSIS_DIR"
echo " Output:    $OUT_DIR"
echo " Started:   $(date)"
echo "============================================================"

# Sanity preflight — soft: enumerate every (pe, seed) cell, log which files
# are present and which are missing, but only abort if NOTHING is runnable.
# The Python entry point catches FileNotFoundError per cell and skips, so
# partial coverage is fine; we just want to surface the gap up front instead
# of letting the user wait for the Python loop to find it cell-by-cell.
echo ""
echo "  Preflight: checking activations + per-seed analysis JSONs"
n_ok=0
n_missing=0
missing_lines=()
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        ACTS="$MODELS_DIR/${PE}_seed${SEED}/test_acts.npy"
        ANL="$ANALYSIS_DIR/${PE}_seed${SEED}_analysis.json"
        miss=""
        [ -f "$ACTS" ] || miss+=" acts"
        [ -f "$ANL"  ] || miss+=" anl"
        if [ -z "$miss" ]; then
            n_ok=$((n_ok + 1))
        else
            n_missing=$((n_missing + 1))
            missing_lines+=("    MISS ${PE}_seed${SEED}:${miss}")
            [ -f "$ACTS" ] || missing_lines+=("           ${ACTS}")
            [ -f "$ANL"  ] || missing_lines+=("           ${ANL}")
        fi
    done
done

if [ "$n_missing" -gt 0 ]; then
    echo "  ${n_missing} (pe, seed) cell(s) have missing inputs:" >&2
    for line in "${missing_lines[@]}"; do
        echo "$line" >&2
    done
fi

if [ "$n_ok" -eq 0 ]; then
    echo "  ERROR: no runnable cells — aborting." >&2
    exit 1
fi
echo "  Preflight: ${n_ok} cell(s) ready, ${n_missing} skipped."

python scripts/run_lagpair_gaussian_baseline.py \
    --models-dir   "$MODELS_DIR" \
    --analysis-dir "$ANALYSIS_DIR" \
    --out-dir      "$OUT_DIR" \
    --pe-types     $PE_TYPES \
    --seeds        $SEEDS \
    --layers       $LAYERS \
    --w            "$W" \
    --max-lag      "$MAX_LAG" \
    --pca-dim      "$PCA_DIM" \
    --skip-edges   "$SKIP_EDGES" \
    --ridge        "$RIDGE"

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
echo "  Next: download $OUT_DIR/gaussian_lagpair_metrics.json and run"
echo "  python scripts/compare_score_vs_gaussian.py \\"
echo "      --score-metrics    lagpair_analysis_3seed/lagpair_metrics.json \\"
echo "      --gaussian-metrics $OUT_DIR/gaussian_lagpair_metrics.json \\"
echo "      --out-dir          results/score_vs_gaussian_comparison"
