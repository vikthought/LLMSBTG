#!/bin/bash
#
# Submit the 3-job chain for the 6-seed extended ablation:
#
#   Phase 0 (GPU)  →  Phase 1+2 (CPU)  →  Phase 3 (GPU)
#
# Each phase is resource-pure (no CPU work in GPU jobs, no GPU allocated
# for CPU jobs).  SLURM --dependency=afterok ensures the chain runs in
# order even if jobs queue.  Output goes to results/lagpair_ablation_6seed/.
#
# This chain does NOT include Phase 4 (logit gap-fill) from the monolith
# cluster_lagpair_6seed_ablation.sh.  Run that script (or
# cluster_lagpair_6seed.sh's logit phase) separately if you need the 18
# per-seed logit JSONs.
#
# Usage:
#   bash scripts/submit_lp6_chain.sh
#
#   # Or override the trained-models dir at submit time (forwarded to all
#   # three phases via --export=ALL):
#   MODELS_DIR=results/extended_pos_models_20260420_152301 \
#       bash scripts/submit_lp6_chain.sh
#
# Notes:
#   * Re-running is safe.  Each phase is idempotent — already-complete
#     phases are skipped at the per-cell level.  If you only need to re-run
#     one phase, sbatch its script directly without dependencies.
#   * The PHASE3 job's --dependency lists BOTH PHASE0 and PHASE12.  SLURM
#     waits for both to complete with exit code 0.
#   * MODELS_DIR resolution inside each phase script (precedence top-down):
#       1. --export=ALL,MODELS_DIR=... (set by this wrapper if MODELS_DIR
#          is in the caller's env)
#       2. MODELS_DIR_DEFAULT hard-coded near the top of each phase script
#       3. Newest results/extended_pos_models_* by name sort
#

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

# Forward MODELS_DIR to all phases if the caller set it; otherwise let each
# script fall through to MODELS_DIR_DEFAULT / auto-discover.
if [[ -n "${MODELS_DIR:-}" ]]; then
    EXPORT_ARG="--export=ALL,MODELS_DIR=$MODELS_DIR"
    echo "Using MODELS_DIR override: $MODELS_DIR"
else
    EXPORT_ARG="--export=ALL"
    echo "MODELS_DIR not set in env — each phase will use its own DEFAULT / auto-discovery."
fi

# Phase 0 (GPU): per-seed analysis JSONs
PHASE0=$(sbatch --parsable $EXPORT_ARG scripts/cluster_lp6_phase0_gpu.sh)
echo "Phase 0  (GPU, analysis JSONs):       jobid $PHASE0"

# Phase 1+2 (CPU): probes + CKA, depends on Phase 0
PHASE12=$(sbatch --parsable --dependency=afterok:"$PHASE0" $EXPORT_ARG \
    scripts/cluster_lp6_phase12_cpu.sh)
echo "Phase 1+2 (CPU, probes + CKA):        jobid $PHASE12  (after:$PHASE0)"

# Phase 3 (GPU): ablation, depends on both Phase 0 AND Phase 12
PHASE3=$(sbatch --parsable --dependency=afterok:"$PHASE0":"$PHASE12" $EXPORT_ARG \
    scripts/cluster_lp6_phase3_gpu.sh)
echo "Phase 3   (GPU, ablation × 4 layers): jobid $PHASE3  (after:$PHASE0,$PHASE12)"

echo ""
echo "============================================================"
echo " 6-seed ablation chain submitted."
echo "============================================================"
echo "  squeue -u \$USER -j $PHASE0,$PHASE12,$PHASE3"
echo ""
echo "  follow logs:"
echo "    tail -f logs/lp6_phase0_${PHASE0}.out"
echo "    tail -f logs/lp6_phase12_${PHASE12}.out"
echo "    tail -f logs/lp6_phase3_${PHASE3}.out"
echo ""
echo "  output dir: results/lagpair_ablation_6seed/"
echo "============================================================"
