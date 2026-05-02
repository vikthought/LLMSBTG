#!/bin/bash
#SBATCH --job-name=rope_finalize
#SBATCH --output=logs/rope_finalize_%j.out
#SBATCH --error=logs/rope_finalize_%j.err
#SBATCH -p week
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ===========================================================================
# RoPE base sweep — finalize Phase 4 on already-trained data
# ===========================================================================
#
# Why a separate, lean script:
#   The main cluster_rope_base_sweep.sh trains 18 RoPE models, runs hidden
#   SBTG, runs logit SBTG, and finally calls compare_rope_bases.py for
#   cross-base aggregation.  When the GPU job times out before reaching
#   the cross-base aggregation, all the training is salvageable but the
#   summary + figures are missing.  This script picks up from there.
#
# This job:
#   * runs on the CPU partition (no --gpus directive — GPU-idle policy
#     literally cannot apply, no keepalive needed),
#   * auto-detects which bases under results/rope_base_sweep/ have the
#     required artifacts,
#   * runs scripts/run_rope_sweep_finalize.py which calls
#     compare_rope_bases.py to produce rope_base_summary.json + 5 figures,
#     then bundles the small source JSONs into a self-contained
#     analysis/ directory,
#   * the entire output (analysis/) is sub-megabyte and trivial to scp
#     down for local analysis.
#
# Submit:
#   sbatch scripts/cluster_rope_sweep_finalize.sh
#
# Output:
#   results/rope_base_sweep/analysis/
#     README.md
#     rope_base_summary.json
#     figures/RB{1..5}.pdf
#     per_base/base_{B}/{lagpair_metrics.json, logit_seed{0..2}.json,
#                        training_history_seed{0..2}.json,
#                        attn_stats_seed{0..2}.json}
#     manifest.json
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Lightweight install (no GPU torch needed, but harmless)
python -m pip install -e ".[all]" --quiet

SWEEP_ROOT="results/rope_base_sweep"

mkdir -p logs

echo "============================================================"
echo " RoPE base sweep — finalize Phase 4"
echo " Sweep root: $SWEEP_ROOT"
echo " Started:    $(date)"
echo "============================================================"

if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: $SWEEP_ROOT does not exist." >&2
    exit 1
fi

python scripts/run_rope_sweep_finalize.py --sweep-root "$SWEEP_ROOT"

echo ""
echo "============================================================"
echo " Finalize complete — $(date)"
echo "============================================================"
echo ""
echo "  Output bundle: $SWEEP_ROOT/analysis/"
echo "  To download locally:"
echo "    rsync -av <cluster>:$PROJECT_DIR/$SWEEP_ROOT/analysis/ ./rope_analysis/"
echo "  or:"
echo "    scp -r <cluster>:$PROJECT_DIR/$SWEEP_ROOT/analysis/ ./rope_analysis/"
echo ""
