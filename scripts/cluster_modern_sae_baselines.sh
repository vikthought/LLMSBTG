#!/bin/bash
#SBATCH --job-name=sae_modern
#SBATCH --output=logs/sae_modern_%j.out
#SBATCH --error=logs/sae_modern_%j.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# Modern SAE baselines (3-seed) — paper3 §4.3 Experiment 2
# ===========================================================================
#
# Trains TopK, BatchTopK, and T-SAE variants on the same activations as
# the L1 SAE baseline (run_sae_baseline.sh / cluster_sae_baseline.sh) and
# runs the same probe → top-3 features → orthonormalize → ablate hook.
#
#   BatchTopK   — full 3 PE × 3 seeds × 4 layers grid (36 cells).
#                 Headline modern comparison: relaxes per-token frame.
#   TopK        — 4 cells × 3 seeds (12 cells).
#                 Control: per-token but modern. Cells chosen as the most
#                 discriminating in the L1 SAE table — ALiBi L1, L4
#                 (where the marginal carries no position info) and
#                 Absolute L1, L2 (where it does).
#   T-SAE       — ALiBi L4 × 3 seeds (3 cells).
#                 Reproduction of Bhalla 2026 (TopK + InfoNCE temporal
#                 contrastive). Most direct match to "leveraging
#                 sequential structure."
#
# Total: 51 cell-runs. ~3-7 min/cell on 1× RTX 5000 Ada.
# Walltime budget: 12h with comfortable headroom for T-SAE (heavier
# than TopK/BatchTopK because of the contrastive batch).
#
# k matching for fairness
# -----------------------
# For TopK and BatchTopK we set k_per_token equal to the L1 SAE's
# average L0 measured on the same activations (so the new variants
# can't win by simply being denser/sparser than the L1 baseline).
# T-SAE inherits TopK's k. The matched value is logged per cell in
# {variant}/analysis/L<L>/sae_models/<pe>_seed<N>_meta.json.
#
# To match L1 L0, we need the saved L1 SAE checkpoints from a previous
# `cluster_sae_baseline.sh` run. The script auto-detects which directory
# they live in. If neither exists, the script falls back to a fixed
# k_per_token (default 32, ~3% of feature_dim) and logs that in meta.
#
# Output layout
# -------------
#   results/sae_modern_3seed/
#     batchtopk/
#       analysis/L<L>/sae_models/<pe>_seed<N>.{pt,_meta.json}
#       ablation/L<L>/b2_sae_direction_ablation.json   ← compare_ablation_baselines pickup point
#     topk/      same layout, fewer cells
#     tsae/      same layout, fewer cells
#
# Submit
# ------
#   sbatch scripts/cluster_modern_sae_baselines.sh
#
# Re-run a single variant after a partial timeout
# -----------------------------------------------
# The Python driver is idempotent at the per-cell level (skips cells
# whose key is already present in the variant's b2 JSON). Run any of
# the three blocks below independently, or set RUN_<VARIANT>=0 to
# skip it.
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
# GPU discovery (same pattern as cluster_sae_baseline.sh and lp3 chain)
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

mkdir -p logs
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
" >> logs/sae_modern_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"

# analysis_summary.json + per-seed analysis JSONs (mu_l + layer_stats), produced
# by Phase 0 of the lp3 chain. Used by every variant.
SUMMARY_PATH=""
for candidate in \
    "results/lagpair_ablation_3seed/analysis/analysis_summary.json" \
    "results/lagpair_ablation_3seed_cpu/analysis/analysis_summary.json" \
    "lagpair_ablation_3seed/analysis/analysis_summary.json"; do
    if [ -f "$candidate" ]; then
        SUMMARY_PATH="$candidate"
        break
    fi
done
if [ -z "$SUMMARY_PATH" ]; then
    echo "ERROR: cannot locate analysis_summary.json — Phase 0 of the lp3 chain must run first." >&2
    exit 1
fi
echo "  Using summary at: $SUMMARY_PATH"

# L1 SAE checkpoints — used only to measure the per-cell average L0 we
# match against. Auto-detect across the two known locations.
L1_SAE_DIR=""
for candidate in \
    "results/sae_baseline_3seed/analysis" \
    "results/sae_baseline_3seed_legacy/analysis"; do
    if [ -d "$candidate" ]; then
        L1_SAE_DIR="$candidate"
        break
    fi
done
if [ -n "$L1_SAE_DIR" ]; then
    echo "  L1 SAE checkpoints at: $L1_SAE_DIR  (k will match per-cell L1 L0)"
    K_MODE_FLAGS="--k-mode match-l1 --l1-sae-dir $L1_SAE_DIR"
else
    echo "  [warn] no L1 SAE checkpoint dir found; falling back to fixed k=32"
    K_MODE_FLAGS="--k-mode fixed"
fi

OUT_DIR="results/sae_modern_3seed"
mkdir -p "$OUT_DIR"

SEEDS="0 1 2"
LAG=1
TOP_K=3
ALPHAS="0.0 0.5 1.0 2.0"

# SAE hyperparameters — feature_dim and epoch budget identical to the L1
# SAE so the comparison is matched on capacity and training duration.
FEATURE_DIM=1024
SAE_EPOCHS=10
SAE_LR=1e-3
SAE_BATCH_SIZE=256
SAE_MAX_SAMPLES=400000
PROBE_MAX_SAMPLES=40000
K_FALLBACK=32        # used if L1 checkpoint is missing for a cell

# T-SAE-specific
TSAE_BATCH_SIZE_SEQS=32
TSAE_MAX_SEQS=8000
TSAE_CONTRASTIVE_WEIGHT=0.1
TSAE_TEMPERATURE=0.1

# Variant toggles (set to 0 to skip)
RUN_BATCHTOPK="${RUN_BATCHTOPK:-1}"
RUN_TOPK="${RUN_TOPK:-1}"
RUN_TSAE="${RUN_TSAE:-1}"

echo "============================================================"
echo " Modern SAE baselines (3-seed) — paper3 §4.3 Exp. 2"
echo " Models:   $MODELS_DIR"
echo " Output:   $OUT_DIR"
echo " Toggles:  batchtopk=$RUN_BATCHTOPK  topk=$RUN_TOPK  tsae=$RUN_TSAE"
echo " Started:  $(date)"
echo "============================================================"

# Sanity: every (PE, seed) cell must have a trained model + activations.
for PE in rope alibi absolute; do
    for SEED in $SEEDS; do
        _seed_dir="$MODELS_DIR/${PE}_seed${SEED}"
        for f in model.pt train_acts.npy test_acts.npy; do
            if [ ! -f "$_seed_dir/$f" ]; then
                echo "ERROR: missing $_seed_dir/$f" >&2
                exit 1
            fi
        done
    done
done
echo "  Sanity check passed: all 9 (PE, seed) cells have model + activations."
echo ""

# ---------------------------------------------------------------------------
# Variant 1: BatchTopK — full 12-cell grid (headline modern comparison)
# ---------------------------------------------------------------------------
if [ "$RUN_BATCHTOPK" = "1" ]; then
    echo "------------------------------------------------------------"
    echo " [1/3] BatchTopK SAE — full grid (3 PE × 3 seeds × 4 layers)"
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_modern_sae_baselines.py \
        --variant             batchtopk \
        --summary-path        "$SUMMARY_PATH" \
        --models-dir          "$MODELS_DIR" \
        --data-dir            "$DATA_DIR" \
        --out-dir             "$OUT_DIR" \
        --pe-types            rope alibi absolute \
        --seeds               $SEEDS \
        --layers              1 2 3 4 \
        --lag                 "$LAG" \
        --top-k               "$TOP_K" \
        --alpha-values        $ALPHAS \
        --feature-dim         "$FEATURE_DIM" \
        $K_MODE_FLAGS \
        --k-fallback          "$K_FALLBACK" \
        --sae-epochs          "$SAE_EPOCHS" \
        --sae-lr              "$SAE_LR" \
        --sae-batch-size      "$SAE_BATCH_SIZE" \
        --sae-max-samples     "$SAE_MAX_SAMPLES" \
        --probe-max-samples   "$PROBE_MAX_SAMPLES" \
        --device              cuda:0
fi

# ---------------------------------------------------------------------------
# Variant 2: TopK — 4 control cells × 3 seeds
# ---------------------------------------------------------------------------
if [ "$RUN_TOPK" = "1" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo " [2/3] TopK SAE — control cells (ALiBi L1/L4 + Absolute L1/L2)"
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_modern_sae_baselines.py \
        --variant             topk \
        --summary-path        "$SUMMARY_PATH" \
        --models-dir          "$MODELS_DIR" \
        --data-dir            "$DATA_DIR" \
        --out-dir             "$OUT_DIR" \
        --cells               alibi:1 alibi:4 absolute:1 absolute:2 \
        --seeds               $SEEDS \
        --lag                 "$LAG" \
        --top-k               "$TOP_K" \
        --alpha-values        $ALPHAS \
        --feature-dim         "$FEATURE_DIM" \
        $K_MODE_FLAGS \
        --k-fallback          "$K_FALLBACK" \
        --sae-epochs          "$SAE_EPOCHS" \
        --sae-lr              "$SAE_LR" \
        --sae-batch-size      "$SAE_BATCH_SIZE" \
        --sae-max-samples     "$SAE_MAX_SAMPLES" \
        --probe-max-samples   "$PROBE_MAX_SAMPLES" \
        --device              cuda:0
fi

# ---------------------------------------------------------------------------
# Variant 3: T-SAE — ALiBi L4 only × 3 seeds
# ---------------------------------------------------------------------------
if [ "$RUN_TSAE" = "1" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo " [3/3] T-SAE — ALiBi L4 (temporal contrastive variant)"
    echo "------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_modern_sae_baselines.py \
        --variant                   tsae \
        --summary-path              "$SUMMARY_PATH" \
        --models-dir                "$MODELS_DIR" \
        --data-dir                  "$DATA_DIR" \
        --out-dir                   "$OUT_DIR" \
        --cells                     alibi:4 \
        --seeds                     $SEEDS \
        --lag                       "$LAG" \
        --top-k                     "$TOP_K" \
        --alpha-values              $ALPHAS \
        --feature-dim               "$FEATURE_DIM" \
        $K_MODE_FLAGS \
        --k-fallback                "$K_FALLBACK" \
        --sae-epochs                "$SAE_EPOCHS" \
        --sae-lr                    "$SAE_LR" \
        --sae-batch-size            "$SAE_BATCH_SIZE" \
        --sae-max-samples           "$SAE_MAX_SAMPLES" \
        --probe-max-samples         "$PROBE_MAX_SAMPLES" \
        --tsae-batch-size-seqs      "$TSAE_BATCH_SIZE_SEQS" \
        --tsae-max-seqs             "$TSAE_MAX_SEQS" \
        --tsae-contrastive-weight   "$TSAE_CONTRASTIVE_WEIGHT" \
        --tsae-temperature          "$TSAE_TEMPERATURE" \
        --device                    cuda:0
fi

echo ""
echo "============================================================"
echo " Modern SAE baselines DONE — $(date)"
echo "============================================================"
echo "  Output JSONs (b2_sae_direction_ablation.json schema-compatible"
echo "  with results/sae_baseline_3seed/, so compare_ablation_baselines.py"
echo "  works unchanged when pointed at each variant's ablation dir):"
for VARIANT in batchtopk topk tsae; do
    for L in 1 2 3 4; do
        _b2="$OUT_DIR/$VARIANT/ablation/L${L}/b2_sae_direction_ablation.json"
        if [ -f "$_b2" ]; then
            n_cells=$(python3 -c "import json; print(len(json.load(open('$_b2'))))")
            echo "    [$VARIANT/L${L}] $_b2  ($n_cells cells)"
        fi
    done
done
echo ""
echo "  To produce the comparison table per variant, run:"
echo "    for V in batchtopk topk tsae; do"
echo "        python scripts/compare_ablation_baselines.py \\"
echo "            --probe-ablation-dir results/lagpair_ablation_3seed/ablation \\"
echo "            --sae-ablation-dir   results/sae_modern_3seed/\$V/ablation \\"
echo "            --layers 1 2 3 4 \\"
echo "            --pe-types rope alibi absolute \\"
echo "            --seeds 0 1 2 \\"
echo "            --out-dir results/ablation_comparison_\$V"
echo "    done"
echo ""
