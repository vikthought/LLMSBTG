#!/bin/bash
#SBATCH --job-name=sbtg_lp6
#SBATCH --output=logs/sbtg_lagpair_6seed_%j.out
#SBATCH --error=logs/sbtg_lagpair_6seed_%j.err
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# EXTENDED METRIC SUITE — 6-SEED RUN
# ===========================================================================
#
# TRAINS NEW transformer models (6 seeds × 3 PE types = 18 models), then
# runs the full extended metric suite on hidden states AND logits.
#
# Pipeline:
#   Phase 0 — regenerate data (20K test)
#   Phase 1 — train transformers + extract activations + logits  (GPU)
#   Phase 2 — hidden-state extended metric suite                 (GPU)
#   Phase 3 — logit-space score-geometric analysis               (GPU)
#   Phase 4 — logit figures                                      (CPU)
#
# GPU is used continuously:
#   Phase 0: CPU-only data generation (~2 min)
#   Phase 1: 18 transformer training runs + forward passes       (~4-6 hrs)
#   Phase 2: 72 score models (150-trial Optuna + 50-epoch DSM)   (~8-14 hrs)
#   Phase 3: 18 logit score models                               (~2-3 hrs)
#   Phase 4: CPU-only figure generation                          (~1 min)
#
# Score models: 18 configs × 4 layers = 72 hidden + 18 logit = 90 total
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

# ---------------------------------------------------------------------------
# GPU keepalive (continuous burst, ~100% util) for the brief CPU phases
# (data regen, logit comparison).  Auto-killed on script exit.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -c "
import torch, time
torch.cuda.init()
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
print('[keepalive] started — continuous burst mode', flush=True)
last_log = time.time(); n = 0
while True:
    C = A @ B
    A = C / (C.abs().max() + 1e-6)
    torch.cuda.synchronize()
    n += 1
    if time.time() - last_log > 300:
        print(f'[keepalive] alive, iters={n}', flush=True); last_log = time.time()
" >> logs/sbtg_lagpair_6seed_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="data/transformer_pos_cluster"
MODELS_DIR="results/extended_pos_models_${TIMESTAMP}"
OUT_DIR="results/lagpair_analysis_6seed"
LOGIT_OUT="${OUT_DIR}/logit_analysis"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2 3 4 5"

# Training
EPOCHS=15
BATCH_SIZE=128

# Hidden-state extended metrics
W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
TOP_K=3

# Logit analysis
LOGIT_PCA_DIM=256
LOGIT_TUNING_TRIALS=60
LOGIT_BOOTSTRAP_N=100

mkdir -p "$MODELS_DIR" "$OUT_DIR" "$LOGIT_OUT" logs

echo ""
echo "============================================================"
echo " Extended Metric Suite — 6-seed (full train + analyze)"
echo " Models:  $MODELS_DIR"
echo " Output:  $OUT_DIR"
echo " GPU:     $GPU_ID"
echo " Started: $(date)"
echo "============================================================"

# ===================================================================
# Phase 0: Regenerate test data (20K)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: Regenerate data (N_test=20000)"
echo "============================================================"

python scripts/generate_transformer_pos_data.py \
    --out-dir    "$DATA_DIR" \
    --n-train    100000 \
    --n-val      5000 \
    --n-test     20000 \
    --seq-len    64 \
    --vocab-size 128 \
    --seed       42

# ===================================================================
# Phase 1: Train transformers + extract activations + logits
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: Train 18 transformers + extract acts & logits"
echo "============================================================"

for SEED in $SEEDS; do
    echo ""
    echo "  --- Training seed=${SEED} (all PE types) ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
        --data-dir      "$DATA_DIR"      \
        --out-dir       "$MODELS_DIR"    \
        --epochs        "$EPOCHS"        \
        --batch-size    "$BATCH_SIZE"    \
        --pe-types      $PE_TYPES        \
        --seed          "$SEED"          \
        --save-logits                    \
        --logit-pca-dim "$LOGIT_PCA_DIM" \
        --device        cuda:0
done

echo " Phase 1 complete — 18 models trained, activations + logits extracted"

# ===================================================================
# Phase 2: Hidden-state extended metric suite
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: Hidden-state extended metrics (72 score models)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_lagpair_analysis.py \
    --models-dir     "$MODELS_DIR" \
    --data-dir       "$DATA_DIR" \
    --out-dir        "$OUT_DIR" \
    --pe-types       $PE_TYPES \
    --seeds          $SEEDS \
    --layers         1 2 3 4 \
    --w              $W \
    --max-lag        $MAX_LAG \
    --pca-dim        $PCA_DIM \
    --top-k          $TOP_K \
    --tuning-trials  $TUNING_TRIALS \
    --score-epochs   $SCORE_EPOCHS \
    --alpha-values   0.0 0.5 1.0 2.0 \
    --device         cuda:0

# ===================================================================
# Phase 3: Logit-space analysis
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Logit-space analysis (18 score models)"
echo "============================================================"

for pe in $PE_TYPES; do
    for seed in $SEEDS; do
        _json="$LOGIT_OUT/${pe}_seed${seed}_logit_analysis.json"
        if [ -f "$_json" ]; then
            echo "  SKIP (exists): $_json"
            continue
        fi
        echo "  Analyzing logits: pe=${pe} seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_logit_analysis.py \
            --models-dir         "$MODELS_DIR"       \
            --out-dir            "$LOGIT_OUT"        \
            --pe-types           "$pe"               \
            --seeds              "$seed"             \
            --w                  "$W"                \
            --max-lag            "$MAX_LAG"          \
            --pca-dim            "$PCA_DIM"          \
            --epochs             "$SCORE_EPOCHS"     \
            --score-tuning-trials "$LOGIT_TUNING_TRIALS" \
            --bootstrap-n        "$LOGIT_BOOTSTRAP_N" \
            --save-score-models                      \
            --device             cuda:0              \
            || echo "WARNING: Logit analysis failed for ${pe}_seed${seed}" >&2
    done
done

# ===================================================================
# Phase 4: Logit figures
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 4: Logit comparison figures"
echo "============================================================"

LOGIT_FIGS="${LOGIT_OUT}/figures"
mkdir -p "$LOGIT_FIGS"

# Hidden vs logit comparison requires analysis_summary.json from the original
# pipeline, not lagpair_metrics.json. Check common locations.
ORIG_SUMMARY=""
for _candidate in \
    "transformer_pos_analysis_"*/analysis_summary.json \
    "results/transformer_pos_analysis_"*/analysis_summary.json \
    "extended_pos_analysis_"*/analysis_summary.json \
    "results/extended_pos_analysis_"*/analysis_summary.json; do
    if [ -f "$_candidate" ]; then
        ORIG_SUMMARY="$_candidate"
        break
    fi
done

if [ -n "$ORIG_SUMMARY" ]; then
    echo "  Using hidden summary: $ORIG_SUMMARY"
    python scripts/compare_hidden_vs_logit.py \
        --hidden-summary "$ORIG_SUMMARY" \
        --logit-dir      "$LOGIT_OUT" \
        --out-dir        "$LOGIT_OUT" \
        --pe-types       $PE_TYPES \
        --seeds          $SEEDS \
        || echo "WARNING: Hidden vs logit comparison failed — continuing" >&2
else
    echo "  SKIP: No analysis_summary.json from original pipeline found"
fi

# ---------------------------------------------------------------------------
# Print summary of outputs
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
echo ""
echo "  Trained models: $MODELS_DIR"
echo "    $(ls "$MODELS_DIR"/*/model.pt 2>/dev/null | wc -l) transformer checkpoints"
echo ""
echo "  Hidden-state results:"
echo "    $OUT_DIR/lagpair_metrics.json"
echo "    $OUT_DIR/figures/"
for f in "$OUT_DIR"/figures/*.pdf; do
    [ -f "$f" ] && echo "      $(basename "$f")"
done
echo ""
echo "  Logit results:"
echo "    $LOGIT_OUT/"
for f in "$LOGIT_OUT"/*.json; do
    [ -f "$f" ] && echo "      $(basename "$f")"
done
echo ""
echo "  Score models:"
echo "    $OUT_DIR/score_models/ ($(ls "$OUT_DIR"/score_models/*.pt 2>/dev/null | wc -l) hidden)"
echo "    $LOGIT_OUT/             ($(ls "$LOGIT_OUT"/*.pt 2>/dev/null | wc -l) logit)"
echo ""

exit 0
