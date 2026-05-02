#!/bin/bash
#SBATCH --job-name=lp3_p12
#SBATCH --output=logs/lp3_phase12_%j.out
#SBATCH --error=logs/lp3_phase12_%j.err
#SBATCH -p week
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G

# ===========================================================================
# 3-seed extended ablation — Phases 1 + 2 (CPU): probes + CKA
# ===========================================================================
#
# Part 2 of the 3-job chain.  No --gpus directive → no GPU allocated → the
# cluster's GPU-idle policy literally cannot apply.  Two phases here:
#
#   Phase 1 — run_probe_baselines.py
#       fits 5 probe variants per (PE, seed, layer):
#         linear_abs, mlp_abs, rel_dist, near_far  (PCA space)
#         linear_hidden                            (full hidden space)
#       saves per-cell weight directions for the Phase 3 ablation comparison.
#
#   Phase 2 — run_cka_baseline.py
#       cross-layer + cross-PE CKA scalars.  Independent of Phase 0/1.
#
# Submit with:
#   PHASE0=$(sbatch --parsable scripts/cluster_lp3_phase0_gpu.sh)
#   sbatch --dependency=afterok:$PHASE0 scripts/cluster_lp3_phase12_cpu.sh
#
# Phase 1 needs Phase 0's per-seed analysis JSONs (for pca_components,
# mu_l, M_bar_r) → Phase 0 must complete first.  Phase 2 doesn't, but it's
# folded in here for simplicity (~1h additional CPU work).
#
# Output: results/lagpair_ablation_3seed/probes/probe_baselines_summary.json
#         results/lagpair_ablation_3seed/cka/cka_summary.json
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

python -m pip install -e ".[all]" --quiet
python -c "import torch; print(f'CPU threads: {torch.get_num_threads()}')"

# ---------------------------------------------------------------------------
# Configuration — same OUT_DIR as Phase 0 / Phase 3
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
OUT_DIR="results/lagpair_ablation_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"

mkdir -p "$OUT_DIR/probes" "$OUT_DIR/cka" logs

echo "============================================================"
echo " 3-seed ablation Phases 1+2 (CPU): probes + CKA"
echo " Output:  $OUT_DIR/{probes,cka}/"
echo " Threads: ${OMP_NUM_THREADS}"
echo " Started: $(date)"
echo "============================================================"

# Sanity: Phase 0 outputs must exist
ANALYSIS_DIR="$OUT_DIR/analysis"
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
if [ ! -f "$SUMMARY_PATH" ]; then
    echo "ERROR: $SUMMARY_PATH not found.  Run Phase 0 first:" >&2
    echo "       sbatch scripts/cluster_lp3_phase0_gpu.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: probe baselines
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: probe baselines (sklearn) ---"
PROBE_PATH="$OUT_DIR/probes/probe_baselines_summary.json"
if [ -f "$PROBE_PATH" ]; then
    echo "  SKIP (exists): $PROBE_PATH"
else
    python scripts/run_probe_baselines.py \
        --analysis-dir "$ANALYSIS_DIR" \
        --models-dir   "$MODELS_DIR" \
        --out-dir      "$OUT_DIR/probes" \
        --pe-types     $PE_TYPES \
        --seeds        $SEEDS
fi

# ---------------------------------------------------------------------------
# Phase 2: CKA baseline
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 2: CKA baseline (numpy) ---"
CKA_PATH="$OUT_DIR/cka/cka_summary.json"
if [ -f "$CKA_PATH" ]; then
    echo "  SKIP (exists): $CKA_PATH"
else
    python scripts/run_cka_baseline.py \
        --models-dir "$MODELS_DIR" \
        --out-dir    "$OUT_DIR/cka" \
        --pe-types   $PE_TYPES \
        --seeds      $SEEDS
fi

echo ""
echo "============================================================"
echo " Phase 1+2 DONE — $(date)"
echo " Probes: $PROBE_PATH"
echo " CKA:    $CKA_PATH"
echo " Next:   sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/cluster_lp3_phase3_gpu.sh"
echo "============================================================"
