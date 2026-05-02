#!/bin/bash
#SBATCH --job-name=sbtg_lp3
#SBATCH --output=logs/sbtg_lagpair_3seed_%j.out
#SBATCH --error=logs/sbtg_lagpair_3seed_%j.err
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256

# ===========================================================================
# EXTENDED METRIC SUITE — 3-SEED RUN
# ===========================================================================
#
# Uses EXISTING trained transformers (no retraining).
# Regenerates test data (N_test=20K) and re-extracts activations + logits.
#
# Pipeline:
#   Phase 0 — regenerate data (20K test) + re-extract activations & logits
#   Phase 1 — hidden-state extended metric suite (A_r, S_r, C_r, AS_r)
#   Phase 2 — logit-space score-geometric analysis
#   Phase 3 — logit figures (hidden vs logit comparison)
#
# GPU is used throughout:
#   Phase 0: forward passes for activation/logit extraction
#   Phase 1: 150-trial Optuna tuning + DSM training per score model (36 total)
#            + ablation forward passes
#   Phase 2: logit score models (3 PE × 3 seeds = 9 models)
#   Phase 3: CPU-only figures (GPU idle briefly)
#
# Input models:  $MODELS_DIR (set below — user provides from full pipeline)
# Output:        lagpair_analysis_3seed/
#
# Score models: 3 PE × 3 seeds × 4 layers = 36 hidden + 9 logit = 45 total
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
# (data regen, hidden-vs-logit comparison).  Auto-killed on script exit.
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
" >> logs/sbtg_lagpair_3seed_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"
OUT_DIR="results/lagpair_analysis_3seed"
LOGIT_OUT="${OUT_DIR}/logit_analysis"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"

W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
TOP_K=3

LOGIT_PCA_DIM=256
LOGIT_TUNING_TRIALS=60
LOGIT_BOOTSTRAP_N=100

mkdir -p "$OUT_DIR" "$LOGIT_OUT" logs

echo ""
echo "============================================================"
echo " Extended Metric Suite — 3-seed"
echo " Models:  $MODELS_DIR"
echo " Output:  $OUT_DIR"
echo " GPU:     $GPU_ID"
echo " Started: $(date)"
echo "============================================================"

# ===================================================================
# Phase 0: Regenerate test data (20K) and re-extract activations + logits
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: Regenerate data (N_test=20000) & re-extract"
echo "============================================================"

python scripts/generate_transformer_pos_data.py \
    --out-dir    "$DATA_DIR" \
    --n-train    100000 \
    --n-val      5000 \
    --n-test     20000 \
    --seq-len    64 \
    --vocab-size 128 \
    --seed       42

for SEED in $SEEDS; do
    echo "  Re-extracting: seed=${SEED} (all PE types)"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
        --data-dir      "$DATA_DIR"      \
        --out-dir       "$MODELS_DIR"    \
        --pe-types      $PE_TYPES        \
        --seed          "$SEED"          \
        --skip-training                  \
        --save-logits                    \
        --logit-pca-dim "$LOGIT_PCA_DIM" \
        --device        cuda:0
done

echo " Phase 0 complete — activations + logits re-extracted with 20K test"

# ===================================================================
# Phase 1: Hidden-state extended metric suite
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: Hidden-state extended metrics"
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
# Phase 2: Logit-space analysis
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: Logit-space analysis"
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
# Phase 3: Logit figures (hidden vs logit comparison)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Logit comparison figures"
echo "============================================================"

LOGIT_FIGS="${LOGIT_OUT}/figures"
mkdir -p "$LOGIT_FIGS"

# Hidden vs logit comparison requires analysis_summary.json from the original
# pipeline (run_positional_analysis.py), not lagpair_metrics.json.
# Check common locations for the original pipeline summary.
ORIG_SUMMARY=""
for _candidate in \
    "transformer_pos_analysis_"*/analysis_summary.json \
    "results/transformer_pos_analysis_"*/analysis_summary.json; do
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
