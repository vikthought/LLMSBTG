#!/bin/bash
#SBATCH --job-name=lp3_p0
#SBATCH --output=logs/lp3_phase0_%j.out
#SBATCH --error=logs/lp3_phase0_%j.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# 3-seed extended ablation — Phase 0 (GPU): per-seed analysis JSONs
# ===========================================================================
#
# Part 1 of a 3-job chain that replaces the monolithic
# cluster_lagpair_3seed_ablation.sh.  Splits the original 4-phase pipeline
# into three resource-pure SLURM jobs so each GPU job stays continuously
# GPU-busy (no CPU-only window for the cluster's GPU-idle monitor to
# trigger on), and the CPU work runs in a separate CPU-partition job.
#
#   Phase 0 (this script, GPU):  run_positional_analysis.py
#                                trains 36 score models (3 PE × 3 seeds × 4 layers)
#                                with Optuna 150 trials + 50 DSM epochs each.
#                                Continuously GPU-bound — should not trigger
#                                idle policy at all.
#                                Output: results/lagpair_ablation_3seed/analysis/
#                                  + analysis_summary.json
#
#   Phase 1+2 (cluster_lp3_phase12_cpu.sh, CPU partition):
#                                probe baselines + CKA, no --gpus directive,
#                                no GPU-idle policy applies.
#                                Submit with --dependency=afterok:<this_job>
#
#   Phase 3 (cluster_lp3_phase3_gpu.sh, GPU):
#                                run_ablation_study.py × 4 layers.
#                                Submit with --dependency=afterok:<this_job>:<phase12_job>
#
# Idempotent: skips cells whose JSON already exists.  Output dir is
# results/lagpair_ablation_3seed/ — same as the old monolithic, so any
# partial state from a previous run is reused.
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

# Light keepalive — Phase 0 is continuously GPU-busy, so this is just
# belt-and-suspenders for the brief inter-cell CPU moments.
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
" >> logs/lp3_phase0_keepalive.log 2>&1 &
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

W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
BOOTSTRAP_N=100

mkdir -p "$OUT_DIR/analysis" logs

echo "============================================================"
echo " 3-seed ablation Phase 0 (GPU): per-seed analysis JSONs"
echo " Models:  $MODELS_DIR"
echo " Output:  $OUT_DIR/analysis/"
echo " Started: $(date)"
echo "============================================================"

# Sanity check models + activations
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
# Phase 0a: per-seed analysis JSONs
# ---------------------------------------------------------------------------
ANALYSIS_DIR="$OUT_DIR/analysis"

# Skip if already complete
ALL_PRESENT=1
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        if [ ! -f "$ANALYSIS_DIR/${PE}_seed${SEED}_analysis.json" ]; then
            ALL_PRESENT=0; break 2
        fi
    done
done

if [ "$ALL_PRESENT" = "1" ]; then
    echo "  All per-seed analysis JSONs already exist — skipping."
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_positional_analysis.py \
        --data-dir              "$DATA_DIR" \
        --models-dir            "$MODELS_DIR" \
        --out-dir               "$ANALYSIS_DIR" \
        --pe-types              $PE_TYPES \
        --seeds                 $SEEDS \
        --w                     "$W" \
        --max-lag               "$MAX_LAG" \
        --pca-dim               "$PCA_DIM" \
        --epochs                "$SCORE_EPOCHS" \
        --score-tuning-trials   "$TUNING_TRIALS" \
        --tune-objective        null_contrast \
        --bootstrap-n           "$BOOTSTRAP_N" \
        --save-score-models \
        --device                cuda:0
fi

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
echo " Phase 0 DONE — $(date)"
echo " Output: $ANALYSIS_DIR/  ($(ls $ANALYSIS_DIR/*.json 2>/dev/null | wc -l) files)"
echo " Next:   sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/cluster_lp3_phase12_cpu.sh"
echo "============================================================"
