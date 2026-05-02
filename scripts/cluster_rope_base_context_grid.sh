#!/bin/bash
#SBATCH --job-name=rope_grid
#SBATCH --output=logs/rope_grid_%j.out
#SBATCH --error=logs/rope_grid_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G

# ===========================================================================
# RoPE base × context grid (small architecture, fast)
# ===========================================================================
#
# Goal
# ----
# The original cluster_rope_base_sweep.sh tested base ∈ {10..10⁶} at fixed
# context 64 with the paper's full 4L × 256H × 4H architecture.  Result
# (rope_base_sweep/analysis): the empirical L4 AS_r envelope is approximately
# base-invariant at context 64; only the behavioral signal at base=10 picks
# out a clean failure.  Diagnosis: at context 64, the lowest-frequency
# rotation dimension does ~1 full rotation only at base ≈ 50 — for base ≥ 100
# the entire low-frequency tail sits near DC and the architectural
# differences between bases are *intrinsically muted* by the short context.
#
# This experiment turns base × context into a 2-D grid at a smaller, faster
# architecture so we can show the U-curve of position-task val loss shifting
# *with* context length — the practical-advice claim that "min viable base
# scales roughly with context length."
#
# Design
# ------
#   Architecture:  2 layers × hidden 128 × 2 heads × MLP 512
#                  (head_dim = 64, identical RoPE math; ~4× faster training
#                  per model than the paper's matched arch).
#   Bases:         {30, 100, 300, 1000, 10000}  — five points bracketing the
#                  predicted Goldilocks zone for context ∈ {64, 256}.
#   Contexts:      {64, 256}  — 64 anchors against existing data; 256 is
#                  where the architectural base × context interaction becomes
#                  visible.  (Skipping 128 for speed; can add as a follow-up.)
#   Seeds:         {0, 1, 2}  — 3 per (base, context) cell.
#   Total:         5 × 2 × 3 = 30 transformer models.
#
# Speed cuts (vs the paper's standard pipeline)
# ---------------------------------------------
#   Transformer epochs: 10 (paper uses 15; val loss plateaus by ~10 in the
#                       existing data).
#   Score epochs:       30 (paper uses 50).
#   Optuna trials:      30 (paper uses 150).
#   PCA dim:            full hidden (no bottleneck) — addresses the
#                       low-energy-direction concern from the v1 sweep.
#   Window:             24 (paper uses 16) — wider than ~2π so the score
#                       model can resolve sub-θ₀ rotation structure.
#   Skip ablation:      yes — this experiment is purely about the empirical
#                       envelope and val-loss U-curve, not causal subspace
#                       claims.
#
# Estimated cost: ~12 h cluster.
#
# Submit
# ------
#   sbatch scripts/cluster_rope_base_context_grid.sh
#
# Output
# ------
#   results/rope_base_context_grid/
#     ├── data_seq{64,256}/                       <- per-context data dirs
#     ├── ctx{64,256}/base_{B}/rope_seed{0,1,2}/  <- model.pt + activations
#     ├── ctx{64,256}/base_{B}/lagpair/           <- SBTG metrics
#     └── analysis/                               <- 2-D heatmap + U-curves
#         ├── rope_grid_summary.json
#         ├── F_val_loss_heatmap.pdf
#         ├── F_val_loss_u_curves.pdf
#         └── F_as_r_envelope_grid.pdf
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
# GPU discovery (same pattern as the lp3 chain scripts)
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
" >> logs/rope_grid_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_ROOT="results/rope_base_context_grid"

BASES=(30 100 300 1000 10000)
CONTEXTS=(64 256)
SEEDS="0 1 2"

# Architecture (small)
N_LAYER=2
N_EMBD=128
N_HEAD=2
N_INNER=512

# Training
N_TRAIN=100000
N_VAL=5000
N_TEST=20000
EPOCHS=10
BATCH_SIZE=128

# SBTG
W=24
PCA_DIM=128       # = N_EMBD, no bottleneck
MAX_LAG=18
TUNING_TRIALS=30
SCORE_EPOCHS=30
SCORE_BATCH_SIZE=256

mkdir -p "$OUT_ROOT" logs

echo "============================================================"
echo " RoPE base × context grid"
echo " Output:    $OUT_ROOT/"
echo " Arch:      ${N_LAYER}L × ${N_EMBD}H × ${N_HEAD}-head × MLP ${N_INNER}"
echo " Bases:     ${BASES[*]}"
echo " Contexts:  ${CONTEXTS[*]}"
echo " Seeds:     $SEEDS"
echo " Total:     $(( ${#BASES[@]} * ${#CONTEXTS[@]} * 3 )) models"
echo " Started:   $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0: per-context data generation (idempotent)
# ---------------------------------------------------------------------------
for CTX in "${CONTEXTS[@]}"; do
    DATA_DIR="$OUT_ROOT/data_seq${CTX}"
    META="$DATA_DIR/metadata.json"
    if [ -f "$META" ]; then
        echo "  [skip] $DATA_DIR exists"
        continue
    fi
    mkdir -p "$DATA_DIR"
    echo ""
    echo "  --- generating data: ctx=${CTX} ---"
    python scripts/generate_transformer_pos_data.py \
        --out-dir    "$DATA_DIR" \
        --n-train    "$N_TRAIN" \
        --n-val      "$N_VAL" \
        --n-test     "$N_TEST" \
        --seq-len    "$CTX" \
        --vocab-size 128 \
        --seed       42
done

# ---------------------------------------------------------------------------
# Phase 1: train all (base, context, seed) cells
# ---------------------------------------------------------------------------
for CTX in "${CONTEXTS[@]}"; do
    DATA_DIR="$OUT_ROOT/data_seq${CTX}"
    for BASE in "${BASES[@]}"; do
        BASE_DIR="$OUT_ROOT/ctx${CTX}/base_${BASE}"
        mkdir -p "$BASE_DIR"
        for SEED in $SEEDS; do
            SEED_DIR="$BASE_DIR/rope_seed${SEED}"
            if [ -f "$SEED_DIR/model.pt" ]; then
                echo "  [skip] ${SEED_DIR}/model.pt exists"
                continue
            fi
            echo ""
            echo "  --- training: ctx=${CTX} base=${BASE} seed=${SEED} ---"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
                --data-dir   "$DATA_DIR" \
                --out-dir    "$BASE_DIR" \
                --pe-types   rope \
                --seed       "$SEED" \
                --epochs     "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --rope-base  "$BASE" \
                --n-layer    "$N_LAYER" \
                --n-embd     "$N_EMBD" \
                --n-head     "$N_HEAD" \
                --n-inner    "$N_INNER" \
                --device     cuda:0
        done
    done
done

# ---------------------------------------------------------------------------
# Phase 2: SBTG (lagpair) per (base, context).  Skip ablation.
# ---------------------------------------------------------------------------
for CTX in "${CONTEXTS[@]}"; do
    DATA_DIR="$OUT_ROOT/data_seq${CTX}"
    for BASE in "${BASES[@]}"; do
        BASE_DIR="$OUT_ROOT/ctx${CTX}/base_${BASE}"
        LP_DIR="$BASE_DIR/lagpair"
        if [ -f "$LP_DIR/lagpair_metrics.json" ]; then
            echo "  [skip] $LP_DIR/lagpair_metrics.json exists"
            continue
        fi
        mkdir -p "$LP_DIR"
        echo ""
        echo "  --- lagpair: ctx=${CTX} base=${BASE} ---"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_lagpair_analysis.py \
            --data-dir       "$DATA_DIR" \
            --models-dir     "$BASE_DIR" \
            --out-dir        "$LP_DIR" \
            --pe-types       rope \
            --seeds          $SEEDS \
            --layers         1 2 \
            --w              "$W" \
            --max-lag        "$MAX_LAG" \
            --pca-dim        "$PCA_DIM" \
            --tuning-trials  "$TUNING_TRIALS" \
            --score-epochs   "$SCORE_EPOCHS" \
            --max-windows-gb 16 \
            --skip-ablation \
            --device         cuda:0
    done
done

# ---------------------------------------------------------------------------
# Phase 3: Effective-bandwidth diagnostic (Option A)
# ---------------------------------------------------------------------------
# For each (base, context) cell, sweep k_keep ∈ {K, K*0.75, K/2, K/4, ...}
# and find the smallest top-frequencies-only setting that retains
# (1 + tolerance) · full-RoPE loss.  k_eff much smaller than K=d_head/2 means
# the model is wasting low-frequency capacity — bases that keep θ_{k_eff} in
# the active range work equivalently.
# ---------------------------------------------------------------------------
for CTX in "${CONTEXTS[@]}"; do
    DATA_DIR="$OUT_ROOT/data_seq${CTX}"
    for BASE in "${BASES[@]}"; do
        BASE_DIR="$OUT_ROOT/ctx${CTX}/base_${BASE}"
        BW_DIR="$BASE_DIR/effective_bandwidth"
        if [ -f "$BW_DIR/effective_bandwidth.json" ]; then
            echo "  [skip] $BW_DIR/effective_bandwidth.json exists"
            continue
        fi
        mkdir -p "$BW_DIR"
        echo ""
        echo "  --- effective bandwidth: ctx=${CTX} base=${BASE} ---"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_rope_effective_bandwidth.py \
            --data-dir   "$DATA_DIR" \
            --models-dir "$BASE_DIR" \
            --out-dir    "$BW_DIR" \
            --seeds      $SEEDS \
            --family     variable_lag_copy \
            --split      test \
            --n-layer    "$N_LAYER" \
            --n-embd     "$N_EMBD" \
            --n-head     "$N_HEAD" \
            --n-inner    "$N_INNER" \
            --device     cuda:0
    done
done

# ---------------------------------------------------------------------------
# Phase 4: Aggregate the 2-D grid
# ---------------------------------------------------------------------------
echo ""
echo "  --- aggregating grid → $OUT_ROOT/analysis/ ---"
python scripts/run_rope_grid_aggregation.py \
    --sweep-root "$OUT_ROOT" \
    --bases      "${BASES[@]}" \
    --contexts   "${CONTEXTS[@]}" \
    --seeds      $SEEDS \
    --out-dir    "$OUT_ROOT/analysis" \
    --d-head     $(( N_EMBD / N_HEAD ))

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Output: $OUT_ROOT/analysis/"
echo "============================================================"
