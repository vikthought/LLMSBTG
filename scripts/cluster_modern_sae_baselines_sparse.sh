#!/bin/bash
#SBATCH --job-name=sae_modern_k32
#SBATCH --output=logs/sae_modern_k32_%j.out
#SBATCH --error=logs/sae_modern_k32_%j.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# Modern SAE baselines — SPARSE-REGIME follow-up (paper3 §4.3 Experiment 2)
# ===========================================================================
#
# Why this exists
# ---------------
# The matched-L0 run (`cluster_modern_sae_baselines.sh`) sets k_per_token to
# the L1 SAE's measured average L0, which turned out to be ~580–800 features
# active per token out of feature_dim=1024 — i.e. 57–78% density. At that
# operating point, BatchTopK's "give a token more budget when neighbours
# use less" structural advantage is mostly inert because nearly every
# token already gets ≥57% of the dictionary.
#
# This run pins k_per_token = 32 (≈3% density), the standard sae_lens
# BatchTopK regime. Question: does BatchTopK close the gap to score-SVD
# in its native operating point?
#
# Expected outcomes
# -----------------
#   - Score-SVD continues to dominate on ALiBi L1–L3 (the App F.1 prediction:
#     no per-token-reconstruction objective can recover what the marginal
#     discards).
#   - BatchTopK may close more of the gap on Absolute (where position is
#     genuinely linearly readable; a 4×-overcomplete dictionary at k=32 has
#     enough specificity to find position-indexed features).
#   - If BatchTopK closes the gap to <2× anywhere, that's the cell to
#     reframe in the paper as "score-SVD and BatchTopK access overlapping
#     but distinct structure".
#
# Cell coverage (defaults)
# ------------------------
#   BatchTopK   — full 12-cell grid (3 PE × 4 layers × 3 seeds = 36 runs)
#   TopK        — disabled (RUN_TOPK=0); enable with `RUN_TOPK=1 sbatch ...`
#                  matched-L0 already showed TopK ≈ BatchTopK at all overlap
#                  cells, so the per-token-vs-batch frame test isn't the
#                  bottleneck here.
#   T-SAE       — disabled (RUN_TSAE=0); enable with `RUN_TSAE=1 sbatch ...`
#                  T-SAE's contrastive loss interacts non-trivially with
#                  hard sparsity at k=32; useful as a robustness check.
#
# Output layout (separate from matched-L0 to avoid collisions)
# ------------------------------------------------------------
#   results/sae_modern_3seed_k32/
#     batchtopk/
#       analysis/L<L>/sae_models/<pe>_seed<N>.{pt,_meta.json}
#       ablation/L<L>/b2_sae_direction_ablation.json
#     (topk/ and tsae/ only populated if RUN_TOPK=1 / RUN_TSAE=1)
#
# Compute budget
# --------------
#   36 cells × ~3 min each on 1× RTX 5000 Ada ≈ 2 h. SAE training at k=32
#   is faster than at k≈600 because fewer features are active in the
#   forward pass, but the difference is small (decode is the cost).
#   12 h walltime gives comfortable headroom and matches the matched-L0
#   wrapper for ops simplicity.
#
# Submit
# ------
#   sbatch scripts/cluster_modern_sae_baselines_sparse.sh                 # BatchTopK only (default)
#   RUN_TOPK=1 sbatch scripts/cluster_modern_sae_baselines_sparse.sh      # + TopK on the 4 control cells
#   RUN_TSAE=1 RUN_TOPK=1 sbatch scripts/cluster_modern_sae_baselines_sparse.sh  # all three
#
# Aggregation (after job finishes)
# --------------------------------
#   python scripts/compare_ablation_baselines.py \
#     --probe-ablation-dir   results/lagpair_ablation_3seed/ablation \
#     --sae-l1-dir           results/sae_baseline_3seed_legacy/ablation \
#     --sae-batchtopk-dir    results/sae_modern_3seed_k32/batchtopk/ablation \
#     --layers 1 2 3 4 \
#     --pe-types rope alibi absolute \
#     --seeds 0 1 2 \
#     --out-dir results/ablation_comparison_k32 \
#     --drop-random-column
#
#   To produce the matched-L0 vs sparse-regime side-by-side table, point
#   --sae-batchtopk-dir at one and --sae-topk-dir at the other (the labels
#   will need a manual rename in comparison_table.{md,tex}, or just run
#   twice and concatenate).
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
# GPU discovery + keepalive (identical pattern to matched-L0 wrapper)
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
" >> logs/sae_modern_k32_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"

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

# Output directory deliberately separate from the matched-L0 run.
OUT_DIR="results/sae_modern_3seed_k32"
mkdir -p "$OUT_DIR"

# Hard-coded sparse k. NO --l1-sae-dir flag — we explicitly do NOT match L1's L0.
# k_per_token = 32 ≈ 3% density at feature_dim=1024 — sae_lens BatchTopK default.
K_MODE_FLAGS="--k-mode fixed --k-fallback 32"
echo "  Sparse regime: k_per_token = 32 (≈3% of feature_dim=1024)"

SEEDS="0 1 2"
LAG=1
TOP_K=3
ALPHAS="0.0 0.5 1.0 2.0"

# Match matched-L0 wrapper on all other knobs so the only changing variable
# is k_per_token.
FEATURE_DIM=1024
SAE_EPOCHS=10
SAE_LR=1e-3
SAE_BATCH_SIZE=256
SAE_MAX_SAMPLES=400000
PROBE_MAX_SAMPLES=40000

TSAE_BATCH_SIZE_SEQS=32
TSAE_MAX_SEQS=8000
TSAE_CONTRASTIVE_WEIGHT=0.1
TSAE_TEMPERATURE=0.1

# Variant toggles — defaults focused on BatchTopK (the headline question).
RUN_BATCHTOPK="${RUN_BATCHTOPK:-1}"
RUN_TOPK="${RUN_TOPK:-0}"
RUN_TSAE="${RUN_TSAE:-0}"

echo "============================================================"
echo " Modern SAE baselines — SPARSE REGIME (k=32) — paper3 §4.3"
echo " Models:   $MODELS_DIR"
echo " Output:   $OUT_DIR"
echo " Toggles:  batchtopk=$RUN_BATCHTOPK  topk=$RUN_TOPK  tsae=$RUN_TSAE"
echo " Started:  $(date)"
echo "============================================================"

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
# Variant 1: BatchTopK at k=32 — full 12-cell grid (headline)
# ---------------------------------------------------------------------------
if [ "$RUN_BATCHTOPK" = "1" ]; then
    echo "------------------------------------------------------------"
    echo " [BatchTopK @ k=32] full grid (3 PE × 3 seeds × 4 layers)"
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
        --sae-epochs          "$SAE_EPOCHS" \
        --sae-lr              "$SAE_LR" \
        --sae-batch-size      "$SAE_BATCH_SIZE" \
        --sae-max-samples     "$SAE_MAX_SAMPLES" \
        --probe-max-samples   "$PROBE_MAX_SAMPLES" \
        --device              cuda:0
fi

# ---------------------------------------------------------------------------
# Variant 2 (optional): TopK at k=32 — same 4 control cells as matched-L0
# ---------------------------------------------------------------------------
if [ "$RUN_TOPK" = "1" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo " [TopK @ k=32] control cells (ALiBi L1/L4 + Absolute L1/L2)"
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
        --sae-epochs          "$SAE_EPOCHS" \
        --sae-lr              "$SAE_LR" \
        --sae-batch-size      "$SAE_BATCH_SIZE" \
        --sae-max-samples     "$SAE_MAX_SAMPLES" \
        --probe-max-samples   "$PROBE_MAX_SAMPLES" \
        --device              cuda:0
fi

# ---------------------------------------------------------------------------
# Variant 3 (optional): T-SAE at k=32 — ALiBi L4 only
# ---------------------------------------------------------------------------
if [ "$RUN_TSAE" = "1" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo " [T-SAE @ k=32] ALiBi L4 (temporal contrastive variant)"
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
echo " Modern SAE baselines (sparse k=32) DONE — $(date)"
echo "============================================================"
echo "  Output JSONs:"
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
echo "  Sparse-regime comparison vs matched-L0:"
echo "    python scripts/compare_ablation_baselines.py \\"
echo "      --probe-ablation-dir  results/lagpair_ablation_3seed/ablation \\"
echo "      --sae-l1-dir          results/sae_baseline_3seed_legacy/ablation \\"
echo "      --sae-batchtopk-dir   results/sae_modern_3seed_k32/batchtopk/ablation \\"
echo "      --layers 1 2 3 4 --pe-types rope alibi absolute --seeds 0 1 2 \\"
echo "      --out-dir results/ablation_comparison_k32 --drop-random-column"
echo ""
