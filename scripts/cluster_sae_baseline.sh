#!/bin/bash
#SBATCH --job-name=sae_baseline
#SBATCH --output=logs/sae_baseline_%j.out
#SBATCH --error=logs/sae_baseline_%j.err
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# SAE baseline (3-seed) — sparse-autoencoder analog of probe_hidden
# ===========================================================================
#
# Why this exists:
#   The current ablation pipeline compares score-SVD against a *linear*
#   probe (paper2.tex tab:ablation_probe_vs_score).  A reviewer's natural
#   follow-up is "what about non-linear, sparsity-encouraged features?"
#   That's what SAEs (Bricken 2023; Cunningham 2023; Templeton 2024)
#   provide.  This job trains an SAE per (PE, seed, layer) cell, identifies
#   the top-k=3 position-relevant features via a probe on feature
#   activations, extracts their decoder columns as ablation directions in
#   hidden space, and runs the **same** ablation hook used elsewhere in
#   the pipeline.
#
#   The new directions slot into the same (PE, seed, layer, lag) →
#   {sae_hidden: {α: {family: loss}}, sae_meta: {...}} schema as the
#   existing b2 JSONs.  Drop-in for the comparison table.
#
# Apples-to-apples with probe_hidden:
#   - Same input (full-hidden activations h ∈ R^256, no PCA bottleneck).
#   - Same readout target (position label 0..63).
#   - Same dimensionality of ablation subspace (top-k=3 directions).
#   - Same intervention hook with identical α sweep.
#   The only thing that changes vs probe_hidden is what the LogisticRegression
#   reads from: raw h (probe_hidden) vs SAE feature activations f = ReLU(W_enc h)
#   (sae_hidden).
#
# Submit:
#   sbatch scripts/cluster_sae_baseline.sh
#
# Output:
#   results/sae_baseline_3seed/
#     analysis/L<layer>/sae_models/<pe>_seed<N>.pt           SAE checkpoints
#                                  <pe>_seed<N>_meta.json    per-cell SAE metadata
#     ablation/L<layer>/b2_sae_direction_ablation.json       per-cell α sweep
#       schema mirrors lagpair_ablation_3seed/.../b2_probe_direction_ablation.json
#
# Compute budget:
#   36 cells (3 PE × 3 seeds × 4 layers).  Per cell: ~1-3 min SAE training
#   + ~1-2 min ablation forward across 4 task families × 4 α values =
#   ~3-5 min/cell × 36 ≈ 2-3 h total on 1× RTX 5000 Ada.  6h walltime
#   buffer absorbs Optuna-style variance.
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
# GPU discovery (same pattern as the lp3 chain)
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

# Light keepalive — both SAE training and ablation are continuously
# GPU-busy, so this is just belt-and-suspenders for inter-cell setup gaps.
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
" >> logs/sae_baseline_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration — matches the 3-seed lp3 chain so the SAE results live in
# the same coordinate system as score_svd / probe_hidden / random_control.
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"

# We need an analysis_summary.json + per-seed analysis JSONs (for mu_l +
# layer_stats), produced by Phase 0 of the lp3 chain.  Both legacy
# locations work — pick whichever is populated on this cluster.
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

OUT_DIR="results/sae_baseline_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
LAYERS="1 2 3 4"

LAG=1
TOP_K=3
ALPHAS="0.0 0.5 1.0 2.0"

# SAE hyperparameters.  Defaults match Bricken et al. 2023 conventions
# scaled for our small 256-d hidden space.
FEATURE_DIM=1024              # 4× hidden_dim, modest overcomplete
SAE_EPOCHS=10
L1_LAMBDA=1e-3
SAE_LR=1e-3
SAE_BATCH_SIZE=256
SAE_MAX_SAMPLES=400000        # cap on training tokens per cell
PROBE_MAX_SAMPLES=40000       # matches probe_hidden's max_train_samples

mkdir -p "$OUT_DIR" logs

echo "============================================================"
echo " SAE baseline (3-seed)"
echo " Models:   $MODELS_DIR"
echo " Summary:  $SUMMARY_PATH"
echo " Output:   $OUT_DIR"
echo " Cells:    3 PE × 3 seeds × 4 layers = 36"
echo " Started:  $(date)"
echo "============================================================"

# Sanity check models + activations exist
for PE in $PE_TYPES; do
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
echo "  Sanity check passed: all 9 (pe, seed) cells have model + activations."
echo ""

# ---------------------------------------------------------------------------
# Run the per-layer driver.  The Python script is idempotent at the
# per-cell level — already-completed cells skip immediately, so a re-run
# after a partial timeout picks up where it left off.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_sae_baseline.py \
    --summary-path        "$SUMMARY_PATH" \
    --models-dir          "$MODELS_DIR" \
    --data-dir            "$DATA_DIR" \
    --out-dir             "$OUT_DIR" \
    --pe-types            $PE_TYPES \
    --seeds               $SEEDS \
    --layers              $LAYERS \
    --lag                 "$LAG" \
    --top-k               "$TOP_K" \
    --alpha-values        $ALPHAS \
    --feature-dim         "$FEATURE_DIM" \
    --sae-epochs          "$SAE_EPOCHS" \
    --l1-lambda           "$L1_LAMBDA" \
    --sae-lr              "$SAE_LR" \
    --sae-batch-size      "$SAE_BATCH_SIZE" \
    --sae-max-samples     "$SAE_MAX_SAMPLES" \
    --probe-max-samples   "$PROBE_MAX_SAMPLES" \
    --device              cuda:0

echo ""
echo "============================================================"
echo " SAE baseline DONE — $(date)"
echo "============================================================"
echo "  Output JSONs:"
for L in $LAYERS; do
    _b2="$OUT_DIR/ablation/L${L}/b2_sae_direction_ablation.json"
    if [ -f "$_b2" ]; then
        n_cells=$(python3 -c "import json; print(len(json.load(open('$_b2'))))")
        echo "    L${L}: $_b2  ($n_cells / 9 cells)"
    fi
done
echo ""
echo "  Next steps:"
echo "    1. Download $OUT_DIR/ablation/L*/b2_sae_direction_ablation.json"
echo "       (small, ~10s of KB per layer)."
echo "    2. Compare against lagpair_ablation_3seed/ablation/L*/b2_probe_direction_ablation.json"
echo "       to compute the score_svd-vs-sae-vs-probe ratio table."
echo ""
