#!/bin/bash
#SBATCH --job-name=lp3_p0_cont
#SBATCH --output=logs/lp3_phase0_cont_%j.out
#SBATCH --error=logs/lp3_phase0_cont_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# 3-seed extended ablation — Phase 0 CONTINUATION (GPU)
# ===========================================================================
#
# Picks up where cluster_lp3_phase0_gpu.sh ran out of wall-time.  Same job as
# Phase 0, but with two changes:
#
#   1) Wall-time bumped 12h → 24h.
#   2) Per-(pe, seed) skip: each cell is called as its own Python invocation
#      and skipped if its analysis JSON already exists.  (The original Phase 0
#      passes all 9 cells in one Python call, so any cell that didn't get to
#      the json.dump line gets fully redone — including its 150-trial Optuna
#      sweep — even if the per-layer training already happened.  That's why
#      a 12h job that died mid-cell loses ~30 min of progress.)
#
# Output dir is the same — results/lagpair_ablation_3seed/analysis/ — so
# Phase 1+2 and Phase 3 don't need to know which Phase 0 variant produced it.
#
# Usage:
#   sbatch scripts/cluster_lp3_phase0_cont_gpu.sh
#
# To re-thread the rest of the chain off this job:
#   CONT=$(sbatch --parsable scripts/cluster_lp3_phase0_cont_gpu.sh)
#   PH12=$(sbatch --parsable --dependency=afterok:$CONT scripts/cluster_lp3_phase12_cpu.sh)
#   sbatch --dependency=afterok:$CONT:$PH12 scripts/cluster_lp3_phase3_gpu.sh
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

# Light keepalive — same belt-and-suspenders pattern as Phase 0.  Each per-cell
# Python call has a brief CPU window between cells; this keeps the GPU-idle
# monitor from firing during those gaps.
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
" >> logs/lp3_phase0_cont_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration — must match cluster_lp3_phase0_gpu.sh
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"
OUT_DIR="results/lagpair_ablation_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"

W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
BOOTSTRAP_N=100

ANALYSIS_DIR="$OUT_DIR/analysis"
mkdir -p "$ANALYSIS_DIR" logs

echo "============================================================"
echo " 3-seed ablation Phase 0 CONTINUATION (GPU)"
echo " Models:  $MODELS_DIR"
echo " Output:  $ANALYSIS_DIR/"
echo " Started: $(date)"
echo "============================================================"

# Sanity check models + activations (re-extract logits if test_acts is missing).
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        _seed_dir="$MODELS_DIR/${PE}_seed${SEED}"
        if [ ! -f "$_seed_dir/model.pt" ]; then
            echo "ERROR: $_seed_dir/model.pt not found." >&2; exit 1
        fi
        if [ ! -f "$_seed_dir/test_acts.npy" ] || [ ! -f "$_seed_dir/train_acts.npy" ]; then
            echo "  (re-extracting acts: ${PE}_seed${SEED})"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
                --data-dir "$DATA_DIR" --out-dir "$MODELS_DIR" \
                --pe-types "$PE" --seed "$SEED" \
                --skip-training --save-logits --device cuda:0
        fi
    done
done

# ---------------------------------------------------------------------------
# Phase 0a: per-(pe, seed) analysis JSONs — one Python call per cell, skip
# any cell whose JSON already exists.
# ---------------------------------------------------------------------------
echo ""
echo "Existing analysis JSONs:"
ls "$ANALYSIS_DIR"/*_analysis.json 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo ""

for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        OUT_JSON="$ANALYSIS_DIR/${PE}_seed${SEED}_analysis.json"
        if [ -f "$OUT_JSON" ]; then
            echo "  [skip] $OUT_JSON already exists"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "  RUNNING ${PE} seed=${SEED}  ($(date))"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_positional_analysis.py \
            --data-dir              "$DATA_DIR" \
            --models-dir            "$MODELS_DIR" \
            --out-dir               "$ANALYSIS_DIR" \
            --pe-types              "$PE" \
            --seeds                 "$SEED" \
            --w                     "$W" \
            --max-lag               "$MAX_LAG" \
            --pca-dim               "$PCA_DIM" \
            --epochs                "$SCORE_EPOCHS" \
            --score-tuning-trials   "$TUNING_TRIALS" \
            --tune-objective        null_contrast \
            --bootstrap-n           "$BOOTSTRAP_N" \
            --save-score-models \
            --device                cuda:0

        if [ ! -f "$OUT_JSON" ]; then
            echo "ERROR: cell ${PE} seed=${SEED} did not produce $OUT_JSON" >&2
            exit 1
        fi
    done
done

# ---------------------------------------------------------------------------
# Phase 0b: aggregate to analysis_summary.json
# ---------------------------------------------------------------------------
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
if [ ! -f "$SUMMARY_PATH" ]; then
    python scripts/aggregate_positional_results.py --out-dir "$ANALYSIS_DIR"
fi
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH missing" >&2; exit 1; }

echo ""
echo "============================================================"
echo " Phase 0 continuation DONE — $(date)"
echo " Output: $ANALYSIS_DIR/  ($(ls $ANALYSIS_DIR/*.json 2>/dev/null | wc -l) files)"
echo " Next:   sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/cluster_lp3_phase12_cpu.sh"
echo "============================================================"
