#!/bin/bash
#SBATCH --job-name=lp3_p3
#SBATCH --output=logs/lp3_phase3_%j.out
#SBATCH --error=logs/lp3_phase3_%j.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# 3-seed extended ablation — Phase 3 (GPU): ablation study, all 4 layers
# ===========================================================================
#
# Part 3 of the 3-job chain.  Continuously GPU-busy: 4 layer-loops over
# run_ablation_study.py, each doing forward passes through the transformer
# at α ∈ {0, 0.5, 1, 2} for score-SVD, probe-hidden, probe-abs, random-PCA,
# wrong-layer, wrong-position direction sets.  No CPU-only window inside
# the script → no GPU-idle policy risk.
#
# Submit with:
#   PHASE0=$(sbatch --parsable scripts/cluster_lp3_phase0_gpu.sh)
#   PHASE12=$(sbatch --parsable --dependency=afterok:$PHASE0 \
#       scripts/cluster_lp3_phase12_cpu.sh)
#   sbatch --dependency=afterok:$PHASE0:$PHASE12 \
#       scripts/cluster_lp3_phase3_gpu.sh
#
# Or use the wrapper:
#   bash scripts/submit_lp3_chain.sh
#
# Output: results/lagpair_ablation_3seed/ablation/L{1..4}/
#           dose_response.json
#           b2_probe_direction_ablation.json
#           b3_controls.json
#           F6_dose_response_*.pdf  +  F6b, F7, T6a
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_GPU:-12}

python -m pip install -e ".[all]" --quiet

# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
_raw_gpus=""
if   [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then _raw_gpus="$CUDA_VISIBLE_DEVICES"
elif [[ -n "${SLURM_STEP_GPUS:-}"      ]]; then _raw_gpus="$SLURM_STEP_GPUS"
elif [[ -n "${SLURM_JOB_GPUS:-}"       ]]; then _raw_gpus="$SLURM_JOB_GPUS"
else _raw_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//'); fi
GPU_IDS=()
IFS=',' read -ra _raw_ids <<< "$_raw_gpus"
for _id in "${_raw_ids[@]}"; do
    _id="${_id#"${_id%%[![:space:]]*}"}"
    _id="${_id%"${_id##*[![:space:]]}"}"
    [[ -n "$_id" ]] && GPU_IDS+=("$_id")
done
[[ ${#GPU_IDS[@]} -eq 0 ]] && { echo "ERROR: no GPU IDs detected." >&2; exit 1; }
GPU_ID=${GPU_IDS[0]}
CUDA_VISIBLE_DEVICES=${GPU_ID} python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Light keepalive — Phase 3 is continuously GPU-busy; this just covers the
# inter-layer model-load / setup gaps.
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -c "
import torch, time
torch.cuda.init()
A = torch.randn(2048, 2048, device='cuda')
B = torch.randn(2048, 2048, device='cuda')
print('[keepalive] light burst mode', flush=True)
while True:
    C = A @ B
    A = C / (C.abs().max() + 1e-6)
    torch.cuda.synchronize()
    time.sleep(5)
" >> logs/lp3_phase3_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"
OUT_DIR="results/lagpair_ablation_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
LAYERS="1 2 3 4"

LAG=1
TOP_K=3
ALPHAS="0.0 0.5 1.0 2.0"
WRONG_POS_OFFSET=10

ANALYSIS_DIR="$OUT_DIR/analysis"
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
PROBE_PATH="$OUT_DIR/probes/probe_baselines_summary.json"

mkdir -p "$OUT_DIR/ablation" logs

echo "============================================================"
echo " 3-seed ablation Phase 3 (GPU): ablation study × 4 layers"
echo " Output:  $OUT_DIR/ablation/"
echo " Started: $(date)"
echo "============================================================"

# Sanity: Phase 0 + Phase 1 outputs must exist
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: Phase 0 output $SUMMARY_PATH missing" >&2; exit 1; }
[ -f "$PROBE_PATH" ]   || { echo "ERROR: Phase 1 output $PROBE_PATH missing" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Phase 3: ablation study per layer
# ---------------------------------------------------------------------------
for LAYER in $LAYERS; do
    LAYER_OUT="$OUT_DIR/ablation/L${LAYER}"
    if [ -f "$LAYER_OUT/b2_probe_direction_ablation.json" ] \
       && [ -f "$LAYER_OUT/dose_response.json" ]; then
        echo "  SKIP (exists): L${LAYER}"
        continue
    fi
    mkdir -p "$LAYER_OUT"
    echo ""
    echo "  --- layer $LAYER ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_ablation_study.py \
        --summary-path           "$SUMMARY_PATH" \
        --models-dir             "$MODELS_DIR" \
        --data-dir               "$DATA_DIR" \
        --out-dir                "$LAYER_OUT" \
        --pe-types               $PE_TYPES \
        --seeds                  $SEEDS \
        --layers                 "$LAYER" \
        --lag                    "$LAG" \
        --top-k                  "$TOP_K" \
        --alpha-values           $ALPHAS \
        --probe-baselines-path   "$PROBE_PATH" \
        --wrong-position-offset  "$WRONG_POS_OFFSET" \
        --ablation-alpha-b3      2.0 \
        --device                 cuda:0 \
        || { echo "ERROR: ablation failed at L${LAYER}" >&2; exit 1; }
done

echo ""
echo "============================================================"
echo " Phase 3 DONE — $(date)"
echo " All outputs:"
for L in $LAYERS; do
    _ld="$OUT_DIR/ablation/L${L}"
    echo "   $_ld/"
    for f in "$_ld"/*.json; do [ -f "$f" ] && echo "     $(basename "$f")"; done
done
echo ""
echo " 3-seed ablation chain complete."
echo "============================================================"
