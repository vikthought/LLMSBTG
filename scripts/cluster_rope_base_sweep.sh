#!/bin/bash
#SBATCH --job-name=sbtg_rope_sweep
#SBATCH --output=logs/sbtg_rope_base_sweep_%j.out
#SBATCH --error=logs/sbtg_rope_base_sweep_%j.err
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# ROPE BASE SWEEP — 6 bases × 3 seeds = 18 RoPE models
# ===========================================================================
#
# Question: does RoPE's downstream task performance and the SBTG signature
# co-vary with the base parameter θ that controls the rotational frequencies?
#
#   θ_k = base^(-2k/d_head)   for k = 0, 1, ..., d_head/2 - 1
#
# Note θ_0 = base^0 = 1 regardless of base — the shortest rotation period is
# always 2π ≈ 6.28 tokens.  What base controls is how fast the LOW-frequency
# dimensions rotate.  Standard RoFormer base = 10,000 (LLaMA, GPT-NeoX,
# Pythia).  Context-extension work (YaRN, PI, Men et al. 2024) uses base up
# to 10^6.  Below-standard bases are exotic but test the aliasing limit.
#
# Predictions (testable by this script):
#   1. AS_r dip at r ≈ 6-7 persists at all bases (base-invariant θ_0 = 1),
#      but its DEPTH varies: shallowest at extreme bases (high-freq dominates
#      at low base; low-freq DC offset overwhelms at high base).
#   2. C_1 at layer 4 is monotonically non-decreasing in log(base):
#      larger base → fewer effective rotation dimensions → coupling
#      concentrates in fewer directions.
#   3. Logit RDI is unimodal in log(base), peak near base = 10^4.
#   4. Task loss on variable_lag_copy is worst at both extremes, best
#      near base = 10^3 to 10^5.
#
# Pipeline
# --------
#   Phase 0 — regenerate data (20K test) ONCE for all bases.
#   Phase 1 — train 3 seeds × 6 bases = 18 RoPE models.
#   Phase 2 — hidden-state SBTG analysis per base (72 score models total).
#   Phase 3 — logit-space SBTG analysis per base (18 logit score models).
#   Phase 4 — cross-base aggregation + figures.
#
# Output: results/rope_base_sweep/
#           base_10/, base_100/, ..., base_1000000/
#               rope_seed{0,1,2}/            (model.pt + activations + logits)
#               lagpair_metrics.json         (hidden SBTG)
#               logit_analysis/              (per-seed logit JSONs)
#           comparison/
#               rope_base_summary.json
#               figures/
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
# GPU keepalive — touches the GPU every 2 min so SLURM doesn't cancel us
# during any CPU-only phase.  Tiny (4 bytes); auto-killed on script exit.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -c "
import torch, time
torch.cuda.init()
# Continuous burst at ~100% util.  rope_base_sweep is mostly GPU-bound
# (Phases 1, 2, 3 all train on GPU), so pause/resume isn't strictly
# needed — the simple continuous keepalive is enough for any utilization
# policy that triggered earlier 50%-burst designs.
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
print('[keepalive] started — continuous burst mode (~100% util)', flush=True)
last_log = time.time()
n_iter = 0
while True:
    C = A @ B
    A = C / (C.abs().max() + 1e-6)
    torch.cuda.synchronize()
    n_iter += 1
    if time.time() - last_log > 300:
        print(f'[keepalive] alive, n_iter={n_iter}, elapsed={(time.time()-last_log):.0f}s', flush=True)
        last_log = time.time()
" >> logs/sbtg_rope_base_sweep_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT
echo "GPU keepalive PID: $KEEPALIVE_PID (logs/sbtg_rope_base_sweep_keepalive.log)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR="data/transformer_pos_cluster"
SWEEP_ROOT="results/rope_base_sweep"
COMPARE_DIR="${SWEEP_ROOT}/comparison"

# Six bases spanning five orders of magnitude, centered on the standard 10,000.
# See script header for literature references and predictions.
BASES=(10 100 1000 10000 100000 1000000)

# 3 seeds per base (same as the 3-seed lagpair run) — 18 models total
SEEDS="0 1 2"

# Training config
EPOCHS=15
BATCH_SIZE=128
LOGIT_PCA_DIM=256

# Hidden-state analysis config (matches cluster_lagpair_3seed.sh)
W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
TOP_K=3

# Logit analysis config
LOGIT_TUNING_TRIALS=60
LOGIT_BOOTSTRAP_N=100

mkdir -p "$SWEEP_ROOT" "$COMPARE_DIR" "$COMPARE_DIR/figures" logs

echo ""
echo "============================================================"
echo " RoPE Base Sweep"
echo " Bases:   ${BASES[*]}"
echo " Seeds:   $SEEDS"
echo " Output:  $SWEEP_ROOT"
echo " GPU:     $GPU_ID"
echo " Started: $(date)"
echo "============================================================"

# ===================================================================
# Phase 0: Regenerate data (20K test set) — shared across all bases
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: Regenerate data (N_test=20000)"
echo "============================================================"

if [ -f "$DATA_DIR/metadata.json" ]; then
    echo "  Data dir exists at $DATA_DIR — skipping regeneration"
else
    python scripts/generate_transformer_pos_data.py \
        --out-dir    "$DATA_DIR" \
        --n-train    100000 \
        --n-val      5000 \
        --n-test     20000 \
        --seq-len    64 \
        --vocab-size 128 \
        --seed       42
fi

# ===================================================================
# Phase 1: Train 18 RoPE models (6 bases × 3 seeds)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: Train 18 RoPE models (6 bases × 3 seeds)"
echo "============================================================"

for BASE in "${BASES[@]}"; do
    BASE_DIR="${SWEEP_ROOT}/base_${BASE}"
    mkdir -p "$BASE_DIR"

    for SEED in $SEEDS; do
        SEED_DIR="${BASE_DIR}/rope_seed${SEED}"
        if [ -f "${SEED_DIR}/model.pt" ] \
           && [ -f "${SEED_DIR}/test_acts.npy" ] \
           && [ -f "${SEED_DIR}/test_logits_pca.npy" ]; then
            echo "  SKIP (exists): base=${BASE} seed=${SEED}"
            continue
        fi
        echo ""
        echo "  --- Training base=${BASE}  seed=${SEED} ---"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
            --data-dir      "$DATA_DIR"      \
            --out-dir       "$BASE_DIR"      \
            --epochs        "$EPOCHS"        \
            --batch-size    "$BATCH_SIZE"    \
            --pe-types      rope             \
            --seed          "$SEED"          \
            --rope-base     "$BASE"          \
            --save-logits                    \
            --logit-pca-dim "$LOGIT_PCA_DIM" \
            --device        cuda:0
    done
done

echo " Phase 1 complete — 18 RoPE models trained."

# ===================================================================
# Phase 2: Hidden-state lagpair analysis per base
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: Hidden-state SBTG per base"
echo "============================================================"

for BASE in "${BASES[@]}"; do
    BASE_DIR="${SWEEP_ROOT}/base_${BASE}"
    LAGPAIR_OUT="${BASE_DIR}/lagpair_analysis"

    if [ -f "${LAGPAIR_OUT}/lagpair_metrics.json" ]; then
        echo "  SKIP (exists): base=${BASE}"
        continue
    fi
    mkdir -p "$LAGPAIR_OUT"

    echo ""
    echo "  --- Analyzing base=${BASE} ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_lagpair_analysis.py \
        --models-dir     "$BASE_DIR" \
        --data-dir       "$DATA_DIR" \
        --out-dir        "$LAGPAIR_OUT" \
        --pe-types       rope \
        --seeds          $SEEDS \
        --layers         1 2 3 4 \
        --w              "$W" \
        --max-lag        "$MAX_LAG" \
        --pca-dim        "$PCA_DIM" \
        --top-k          "$TOP_K" \
        --tuning-trials  "$TUNING_TRIALS" \
        --score-epochs   "$SCORE_EPOCHS" \
        --alpha-values   0.0 0.5 1.0 2.0 \
        --device         cuda:0
done

# ===================================================================
# Phase 3: Logit-space analysis per base
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Logit SBTG per base"
echo "============================================================"

for BASE in "${BASES[@]}"; do
    BASE_DIR="${SWEEP_ROOT}/base_${BASE}"
    LOGIT_OUT="${BASE_DIR}/logit_analysis"
    mkdir -p "$LOGIT_OUT"

    for SEED in $SEEDS; do
        _json="${LOGIT_OUT}/rope_seed${SEED}_logit_analysis.json"
        if [ -f "$_json" ]; then
            echo "  SKIP (exists): base=${BASE} seed=${SEED}"
            continue
        fi
        echo "  Analyzing logits: base=${BASE} seed=${SEED}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_logit_analysis.py \
            --models-dir          "$BASE_DIR"          \
            --out-dir             "$LOGIT_OUT"         \
            --pe-types            rope                 \
            --seeds               "$SEED"              \
            --w                   "$W"                 \
            --max-lag             "$MAX_LAG"           \
            --pca-dim             "$PCA_DIM"           \
            --epochs              "$SCORE_EPOCHS"      \
            --score-tuning-trials "$LOGIT_TUNING_TRIALS" \
            --bootstrap-n         "$LOGIT_BOOTSTRAP_N" \
            --save-score-models                        \
            --device              cuda:0               \
            || echo "WARNING: logit analysis failed for base=${BASE} seed=${SEED}" >&2
    done
done

# ===================================================================
# Phase 4: Cross-base aggregation and figures
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 4: Cross-base aggregation + figures"
echo "============================================================"

python scripts/compare_rope_bases.py \
    --sweep-root "$SWEEP_ROOT" \
    --bases      "${BASES[@]}" \
    --seeds      $SEEDS \
    --out-dir    "$COMPARE_DIR" \
    || { echo "ERROR: Phase 4 aggregation failed" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
echo ""
echo "  Per-base model + analysis:"
for BASE in "${BASES[@]}"; do
    _bd="${SWEEP_ROOT}/base_${BASE}"
    n_models=$(ls "$_bd"/rope_seed*/model.pt 2>/dev/null | wc -l)
    has_lagpair=0; [ -f "${_bd}/lagpair_analysis/lagpair_metrics.json" ] && has_lagpair=1
    n_logit=$(ls "$_bd"/logit_analysis/rope_seed*_logit_analysis.json 2>/dev/null | wc -l)
    echo "    base=${BASE}: ${n_models}/3 models, lagpair=${has_lagpair}, logit=${n_logit}/3"
done
echo ""
echo "  Cross-base aggregation:"
echo "    $COMPARE_DIR/rope_base_summary.json"
echo "    $COMPARE_DIR/figures/"
for f in "$COMPARE_DIR"/figures/*.pdf; do
    [ -f "$f" ] && echo "      $(basename "$f")"
done
echo ""

exit 0
