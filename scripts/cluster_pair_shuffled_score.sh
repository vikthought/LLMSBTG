#!/bin/bash
#SBATCH --job-name=lp3_psh
#SBATCH --output=logs/lp3_pair_shuffled_%j.out
#SBATCH --error=logs/lp3_pair_shuffled_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# 3-seed pair-shuffled score-model ablation (Sprint 2)
# ===========================================================================
#
# Reviewer-requested control: re-train the score model on data where the
# joint between the focal block (position w-1) and earlier blocks is
# destroyed by per-window permutation, then ablate with directions from
# the resulting M_bar[1]. If SBTG's signal requires joint structure, the
# shuffled-score directions should match the random-control Δ; if they
# match the real-score Δ, SBTG is reading marginal structure and the
# paper's framing in §4.3 / App E needs pulling back.
#
# Per (PE, seed, layer) cell:
#   1. PCA-project train + test acts using the cell's existing
#      pca_components / mu_l (same coordinate system as the real model)
#   2. Window into width-w=16 blocks
#   3. Pair-shuffle position w-1 across windows in BOTH train and test
#      (preserves all per-position marginals; breaks joint between
#       block w-1 and every earlier block)
#   4. Train a fresh DSM score model on the shuffled training data using
#      the cell's existing Optuna-tuned (sigma, hidden_dim, lr)
#   5. Compute M_bar[1] on shuffled test data with this model
#   6. Top-3 right SVs → back-project through pca_components
#   7. Ablate the REAL model through the same forward hook used by
#      score-SVD / probe-hidden / the second-order baselines
#
# Output:
#   $OUT_DIR/pair_shuffled_score/<pe>_seed<N>_L<layer>_pair_shuffled.json
#
# Cost (rough):
#   Per cell: PCA proj + windowing (~30s) + score train (50 epochs DSM,
#             ~2-3 min on rtx_5000_ada) + estimator forward (~30s) +
#             real-model ablation (~30s) ≈ 4-5 min/cell
#   Total: 36 cells × ~5 min ≈ 3-4h. 24h wall is generous.
#
# Tuning down for a faster smoke test:
#   sbatch --export=ALL,SCORE_EPOCHS=20 scripts/cluster_pair_shuffled_score.sh
#
# Idempotent: skips per-cell JSONs that already exist on resubmit.
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

# Light keepalive
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
" >> logs/lp3_pair_shuffled_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="${MODELS_DIR:-results/transformer_pos_models_20260419_114958}"
DATA_DIR="${DATA_DIR:-data/transformer_pos_cluster}"
OUT_DIR="${OUT_DIR:-results/lagpair_ablation_3seed}"
ANALYSIS_DIR="$OUT_DIR/analysis"
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
LAYERS="1 2 3 4"

LAG=1
TOP_K=3
ALPHA=2.0
W=16
MAX_LAG=14
SCORE_EPOCHS="${SCORE_EPOCHS:-50}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-256}"

mkdir -p "$OUT_DIR/pair_shuffled_score" logs

echo "============================================================"
echo " 3-seed pair-shuffled score-model ablation (Sprint 2)"
echo " Models:   $MODELS_DIR"
echo " Analysis: $ANALYSIS_DIR"
echo " Output:   $OUT_DIR/pair_shuffled_score/"
echo " Score:    epochs=$SCORE_EPOCHS  batch=$SCORE_BATCH_SIZE"
echo " Started:  $(date)"
echo "============================================================"

[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH missing — run cluster_lp3_phase0_gpu.sh first" >&2; exit 1; }
[ -d "$MODELS_DIR" ]   || { echo "ERROR: $MODELS_DIR not a directory" >&2; exit 1; }
[ -d "$DATA_DIR" ]     || { echo "ERROR: $DATA_DIR not a directory" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_pair_shuffled_score_ablation.py \
    --summary-path     "$SUMMARY_PATH" \
    --models-dir       "$MODELS_DIR" \
    --data-dir         "$DATA_DIR" \
    --out-dir          "$OUT_DIR/pair_shuffled_score" \
    --pe-types         $PE_TYPES \
    --seeds            $SEEDS \
    --layers           $LAYERS \
    --lag              $LAG \
    --top-k            $TOP_K \
    --alpha            $ALPHA \
    --w                $W \
    --max-lag          $MAX_LAG \
    --score-epochs     $SCORE_EPOCHS \
    --score-batch-size $SCORE_BATCH_SIZE \
    --device           cuda:0

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Output: $OUT_DIR/pair_shuffled_score/  ($(ls "$OUT_DIR/pair_shuffled_score"/*.json 2>/dev/null | wc -l) JSONs)"
echo "============================================================"
