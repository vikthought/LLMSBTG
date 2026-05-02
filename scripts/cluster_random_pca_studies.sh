#!/bin/bash
#SBATCH --job-name=lp3_rand
#SBATCH --output=logs/lp3_random_pca_%j.out
#SBATCH --error=logs/lp3_random_pca_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# 3-seed PCA-random null distribution + PC-stratified ablation studies
# ===========================================================================
#
# Why this exists:
#   The single PCA-random draw in run_ablation_study.py gives ratio<1 in
#   21/36 per-seed cells (most prominently RoPE deep layers and Absolute
#   L3-L4) because (a) it's a point estimate not a distribution, and
#   (b) it conflates variance loading with coupling alignment. This job
#   runs two follow-up studies that disentangle both:
#
#   Study 1  run_random_pca_distribution.py
#     N=100 PCA-random orthonormal 3-dim subspaces per cell, all ablated
#     through the same forward hook as score-SVD. Reports score-SVD's
#     percentile against the resulting Δ distribution. Robust to a single
#     unlucky random draw.
#
#   Study 2  run_pc_stratified_ablation.py
#     Splits the m=32 PCA subspace into a top-K_var "high variance" block
#     (K_var=8 default) and a bottom-(m-K_var) "low variance" block, then
#     runs both random and score-SVD-restricted ablations in each block.
#     If score-SVD's signal is variance-loading, the bottom-block Δ → 0.
#     If it's coupling-loading, the bottom-block Δ stays large.
#
# Prerequisites:
#   * Phase 0 outputs at $ANALYSIS_DIR/<pe>_seed<N>_analysis.json
#     (cluster_lp3_phase0_gpu.sh / submit_lp3_chain.sh).
#   * Trained transformers at $MODELS_DIR (default
#     results/transformer_pos_models_20260419_114958, matching the 3-seed
#     pipeline).
#   * Test data at $DATA_DIR.
#
# Idempotent: each study script skips per-cell JSON files that already
# exist. Re-submit safely after a wall-time hit; the same SLURM script
# resumes where it left off.
#
# Output:
#   $OUT_DIR/random_pca_distribution/<pe>_seed<N>_L<layer>_random_distribution.json
#   $OUT_DIR/pc_stratified/<pe>_seed<N>_L<layer>_pc_stratified.json
#
# Cost (rough):
#   Study 1: 36 cells × (1 baseline + 100 random) = 3636 forward passes
#   Study 2: 36 cells × (1 baseline + 3 score + 3*N_draws random) ≈ 2400
#   Each pass: ~10-30s on rtx_5000_ada. Total estimate: 18-22h. 24h wall
#   leaves slack.
#
# Tuning down for a faster run:
#   sbatch --export=ALL,NUM_RANDOM_DRAWS=50,NUM_STRATIFIED_DRAWS=10 \\
#       scripts/cluster_random_pca_studies.sh
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
# GPU discovery (matches lp3 phase scripts)
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

# Light keepalive — both studies are continuously GPU-busy, this just
# covers inter-cell model-load gaps.
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
" >> logs/lp3_random_pca_keepalive.log 2>&1 &
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
TOP_K_VAR=8
ALPHA=2.0

NUM_RANDOM_DRAWS="${NUM_RANDOM_DRAWS:-100}"
NUM_STRATIFIED_DRAWS="${NUM_STRATIFIED_DRAWS:-20}"

mkdir -p "$OUT_DIR/random_pca_distribution" \
         "$OUT_DIR/pc_stratified" \
         logs

echo "============================================================"
echo " 3-seed PCA-random studies"
echo " Models:        $MODELS_DIR"
echo " Analysis:      $ANALYSIS_DIR"
echo " Output:        $OUT_DIR/{random_pca_distribution,pc_stratified}/"
echo " Random draws:  $NUM_RANDOM_DRAWS (Study 1)"
echo "                $NUM_STRATIFIED_DRAWS per condition (Study 2)"
echo " Started:       $(date)"
echo "============================================================"

# Sanity checks
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH missing — run cluster_lp3_phase0_gpu.sh first" >&2; exit 1; }
[ -d "$MODELS_DIR" ]   || { echo "ERROR: $MODELS_DIR not a directory" >&2; exit 1; }
[ -d "$DATA_DIR" ]     || { echo "ERROR: $DATA_DIR not a directory" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Study 1: PCA-random null distribution (N draws per cell)
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo " Study 1: PCA-random null distribution (N=$NUM_RANDOM_DRAWS draws)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_random_pca_distribution.py \
    --summary-path     "$SUMMARY_PATH" \
    --models-dir       "$MODELS_DIR" \
    --data-dir         "$DATA_DIR" \
    --out-dir          "$OUT_DIR/random_pca_distribution" \
    --pe-types         $PE_TYPES \
    --seeds            $SEEDS \
    --layers           $LAYERS \
    --lag              $LAG \
    --top-k            $TOP_K \
    --num-random-draws $NUM_RANDOM_DRAWS \
    --alpha            $ALPHA \
    --device           cuda:0

# ---------------------------------------------------------------------------
# Study 2: PC-stratified ablation
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo " Study 2: PC-stratified ablation"
echo "   (top-K_var=$TOP_K_VAR vs bottom block, ${NUM_STRATIFIED_DRAWS} random per condition)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_pc_stratified_ablation.py \
    --summary-path     "$SUMMARY_PATH" \
    --models-dir       "$MODELS_DIR" \
    --data-dir         "$DATA_DIR" \
    --out-dir          "$OUT_DIR/pc_stratified" \
    --pe-types         $PE_TYPES \
    --seeds            $SEEDS \
    --layers           $LAYERS \
    --lag              $LAG \
    --top-k            $TOP_K \
    --top-k-var        $TOP_K_VAR \
    --num-random-draws $NUM_STRATIFIED_DRAWS \
    --alpha            $ALPHA \
    --device           cuda:0

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Study 1 output: $OUT_DIR/random_pca_distribution/"
echo "   ($(ls "$OUT_DIR/random_pca_distribution"/*.json 2>/dev/null | wc -l) per-cell JSONs)"
echo " Study 2 output: $OUT_DIR/pc_stratified/"
echo "   ($(ls "$OUT_DIR/pc_stratified"/*.json 2>/dev/null | wc -l) per-cell JSONs)"
echo "============================================================"
