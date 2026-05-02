#!/bin/bash
#SBATCH --job-name=lp3_mlag
#SBATCH --output=logs/lp3_multilag_%j.out
#SBATCH --error=logs/lp3_multilag_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# 3-seed multi-lag and higher-k score-SVD ablation experiment
# ===========================================================================
#
# Why this exists:
#   Study 1 (random_pca_distribution) showed score_lag1_k3 fails the PCA-
#   random null in 18/36 cells, with a mechanism-aligned pattern: it wins
#   where coupling is forced onto high-variance modes (ALiBi rank-one,
#   Absolute shallow) and loses where coupling spans many lower-variance
#   modes (RoPE deep layers, Absolute deep). The lag-1-only extraction in
#   run_ablation_study.py / SingularAblationHook reads only ONE of 14 lag
#   slices the score model produces, while SBTG's per-lag operator family
#   M_r is the full picture.
#
# This job tests 5 alternative score-SVD direction-extraction strategies
# vs matched-k PCA-random null distributions:
#
#   score_lag1_k3                — current pipeline baseline
#   score_multilag_uniform_k3    — stack M_bar[1..max_lag], SVD top-3
#   score_multilag_amplitude_k3  — A_r-weighted stack (emphasizes RoPE's
#                                   active lags over its quiet ones)
#   score_lag1_k5                — does extra capacity at lag-1 help?
#   score_multilag_uniform_k5    — both fixes combined
#
# Random nulls at matched k (3 and 5) are drawn N=20 times per cell; each
# score-SVD condition is reported with its percentile vs the matched-k null.
#
# Prerequisites:
#   * Phase 0 outputs at $ANALYSIS_DIR/<pe>_seed<N>_analysis.json
#   * Trained transformers at $MODELS_DIR
#   * Test data at $DATA_DIR
#
# Idempotent: skips per-cell JSON files that already exist on resubmit.
#
# Output:
#   $OUT_DIR/multilag_ablation/<pe>_seed<N>_L<layer>_multilag_ablation.json
#
# Cost:
#   Per cell: 1 baseline + 5 score conditions + 2 × 20 random draws = 46 fwd passes
#   Each pass: ~10-30s on rtx_5000_ada
#   Total: 36 cells × ~20 min = ~12h. 24h wall leaves 2× slack.
#
# Tuning down for a faster run:
#   sbatch --export=ALL,NUM_RANDOM_DRAWS=10 scripts/cluster_multilag_ablation.sh
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

# Light keepalive (this script is continuously GPU-busy; just covers
# inter-cell model-load gaps).
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
" >> logs/lp3_multilag_keepalive.log 2>&1 &
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
LAG_MIN=1
ALPHA=2.0
NUM_RANDOM_DRAWS="${NUM_RANDOM_DRAWS:-20}"

mkdir -p "$OUT_DIR/multilag_ablation" logs

echo "============================================================"
echo " 3-seed multi-lag / higher-k ablation experiment"
echo " Models:        $MODELS_DIR"
echo " Analysis:      $ANALYSIS_DIR"
echo " Output:        $OUT_DIR/multilag_ablation/"
echo " Random draws:  $NUM_RANDOM_DRAWS per matched-k null"
echo " Started:       $(date)"
echo "============================================================"

# Sanity
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH missing" >&2; exit 1; }
[ -d "$MODELS_DIR" ]   || { echo "ERROR: $MODELS_DIR not a directory" >&2; exit 1; }
[ -d "$DATA_DIR" ]     || { echo "ERROR: $DATA_DIR not a directory" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_multilag_ablation.py \
    --summary-path     "$SUMMARY_PATH" \
    --models-dir       "$MODELS_DIR" \
    --data-dir         "$DATA_DIR" \
    --out-dir          "$OUT_DIR/multilag_ablation" \
    --pe-types         $PE_TYPES \
    --seeds            $SEEDS \
    --layers           $LAYERS \
    --lag              $LAG \
    --lag-min          $LAG_MIN \
    --num-random-draws $NUM_RANDOM_DRAWS \
    --alpha            $ALPHA \
    --device           cuda:0

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Output: $OUT_DIR/multilag_ablation/"
echo " ($(ls "$OUT_DIR/multilag_ablation"/*.json 2>/dev/null | wc -l) per-cell JSONs)"
echo "============================================================"
