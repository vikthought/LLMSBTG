#!/bin/bash
#SBATCH --job-name=lp3_so
#SBATCH --output=logs/lp3_second_order_%j.out
#SBATCH --error=logs/lp3_second_order_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# 3-seed second-order / conditional / pair-probe ablation baselines (Sprint 1)
# ===========================================================================
#
# Reviewer-requested baseline suite that competes with score-SVD at the
# same forward hook. Tests whether second-order statistics (cross-cov,
# CCA, conditional regression), pure variance (top-PCA), shuffled-score,
# or pair probes can match score-SVD's per-cell ablation Δ.
#
# Conditions per (PE, seed, layer) cell, all k=3 in m=32 PCA, ablated at
# lag=1 source-side through the existing _eval_with_dirs hook:
#
#   score_lag1_k3              top-3 right SVs of M_bar[1]   (reference)
#   cross_cov_svd_k3           top-3 right SVs of E[u_i u_{i-1}^T]
#   cca_k3                     top-3 source-side canonical directions
#   cond_regression_k3         top-3 right SVs of W in u_i = W u_{i-1} + b
#   top_pca_k3                 first 3 PCA components (variance-only)
#   shuffled_score_k3          M_bar[1] right SVs with each row's
#                              coefficients permuted across PCs
#   pair_probe_concat_k3       top-3 SVs of LR(concat(u_i, u_{i-1}) → lag)
#                              source-side block
#   pair_probe_difference_k3   top-3 SVs of LR(u_i - u_{i-1} → lag)
#
# Prerequisites:
#   * Phase 0 outputs at $ANALYSIS_DIR/<pe>_seed<N>_analysis.json
#   * Trained transformers at $MODELS_DIR (with test_acts.npy alongside)
#   * Test data at $DATA_DIR
#
# Idempotent: skips per-cell JSON files that already exist.
#
# Output:
#   $OUT_DIR/second_order_baselines/<pe>_seed<N>_L<layer>_second_order.json
#
# Cost (rough):
#   Per cell: 1 baseline + 8 conditions × 1 fwd at α=2 = 9 forward passes
#   plus per-cell direction-set construction (~30s each: regression / CCA / probes)
#   Each fwd pass: ~10-30s on rtx_5000_ada
#   Total: 36 cells × ~5-7 min ≈ 3-4h. 24h wall is generous.
#
# Tuning down for a faster run:
#   sbatch --export=ALL,MAX_PAIRS=100000 scripts/cluster_second_order_baselines.sh
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
" >> logs/lp3_second_order_keepalive.log 2>&1 &
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
MAX_PAIRS="${MAX_PAIRS:-200000}"
MAX_LAG_FOR_PROBES="${MAX_LAG_FOR_PROBES:-14}"

mkdir -p "$OUT_DIR/second_order_baselines" logs

echo "============================================================"
echo " 3-seed second-order baselines (Sprint 1)"
echo " Models:   $MODELS_DIR"
echo " Analysis: $ANALYSIS_DIR"
echo " Output:   $OUT_DIR/second_order_baselines/"
echo " Pairs:    max=$MAX_PAIRS, max_lag=$MAX_LAG_FOR_PROBES"
echo " Started:  $(date)"
echo "============================================================"

[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH missing — run cluster_lp3_phase0_gpu.sh first" >&2; exit 1; }
[ -d "$MODELS_DIR" ]   || { echo "ERROR: $MODELS_DIR not a directory" >&2; exit 1; }
[ -d "$DATA_DIR" ]     || { echo "ERROR: $DATA_DIR not a directory" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_second_order_baselines.py \
    --summary-path        "$SUMMARY_PATH" \
    --models-dir          "$MODELS_DIR" \
    --data-dir            "$DATA_DIR" \
    --out-dir             "$OUT_DIR/second_order_baselines" \
    --pe-types            $PE_TYPES \
    --seeds               $SEEDS \
    --layers              $LAYERS \
    --lag                 $LAG \
    --top-k               $TOP_K \
    --alpha               $ALPHA \
    --max-pairs           $MAX_PAIRS \
    --max-lag-for-probes  $MAX_LAG_FOR_PROBES \
    --device              cuda:0

echo ""
echo "------------------------------------------------------------"
echo " Sprint 1 analysis-only: RoPE ρ(r) tracking + nulls"
echo "------------------------------------------------------------"
python scripts/analyze_rope_rho_tracking.py \
    --matched-metrics "results/lagpair_analysis_3seed/lagpair_metrics.json" \
    --grid-summary    "results/rope_base_context_grid/analysis/rope_grid_summary.json" \
    --out-dir         "$OUT_DIR/rope_rho_tracking"

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Second-order baselines: $OUT_DIR/second_order_baselines/  ($(ls "$OUT_DIR/second_order_baselines"/*.json 2>/dev/null | wc -l) JSONs)"
echo " ρ(r) tracking analysis: $OUT_DIR/rope_rho_tracking/"
echo "============================================================"
