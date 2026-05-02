#!/bin/bash
#
# Submit the 3-job chain for the 3-seed extended ablation:
#
#   Phase 0 (GPU)  →  Phase 1+2 (CPU)  →  Phase 3 (GPU)
#
# Each phase is resource-pure (no CPU work in GPU jobs, no GPU allocated
# for CPU jobs).  SLURM --dependency=afterok ensures the chain runs in
# order even if jobs queue.  Output goes to results/lagpair_ablation_3seed/.
#
# Usage:
#   bash scripts/submit_lp3_chain.sh
#
# Notes:
#   * Re-running is safe.  Each phase is idempotent — already-complete
#     phases are skipped at the per-cell level.  If you only need to re-run
#     one phase, sbatch its script directly without dependencies.
#   * The PHASE3 job's --dependency lists BOTH PHASE0 and PHASE12.  SLURM
#     waits for both to complete with exit code 0.
#

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

# Phase 0 (GPU): per-seed analysis JSONs
PHASE0=$(sbatch --parsable scripts/cluster_lp3_phase0_gpu.sh)
echo "Phase 0  (GPU, analysis JSONs):       jobid $PHASE0"

# Phase 1+2 (CPU): probes + CKA, depends on Phase 0
PHASE12=$(sbatch --parsable --dependency=afterok:"$PHASE0" \
    scripts/cluster_lp3_phase12_cpu.sh)
echo "Phase 1+2 (CPU, probes + CKA):        jobid $PHASE12  (after:$PHASE0)"

# Phase 3 (GPU): ablation, depends on both Phase 0 AND Phase 12
PHASE3=$(sbatch --parsable --dependency=afterok:"$PHASE0":"$PHASE12" \
    scripts/cluster_lp3_phase3_gpu.sh)
echo "Phase 3   (GPU, ablation × 4 layers): jobid $PHASE3  (after:$PHASE0,$PHASE12)"

echo ""
echo "============================================================"
echo " 3-seed ablation chain submitted."
echo "============================================================"
echo "  squeue -u \$USER -j $PHASE0,$PHASE12,$PHASE3"
echo ""
echo "  follow logs:"
echo "    tail -f logs/lp3_phase0_${PHASE0}.out"
echo "    tail -f logs/lp3_phase12_${PHASE12}.out"
echo "    tail -f logs/lp3_phase3_${PHASE3}.out"
echo ""
echo "  output dir: results/lagpair_ablation_3seed/"
echo "============================================================"
