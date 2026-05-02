#!/bin/bash
#SBATCH --job-name=lp3_p0_abs_fill
#SBATCH --output=logs/lp3_phase0_abs_fill_%j.out
#SBATCH --error=logs/lp3_phase0_abs_fill_%j.err
#SBATCH -p gpu
#SBATCH -t 04:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G

# ===========================================================================
# Phase 0 fill-in: regenerate per-seed analysis JSONs for absolute_seed{1,2}
# ===========================================================================
#
# Background.  The 3-seed extended ablation pipeline writes per-seed analysis
# JSONs at results/lagpair_ablation_3seed/analysis/{pe}_seed{seed}_analysis.json.
# These are the inputs that downstream baselines (Gaussian, SAE legacy) need
# in order to share a PCA basis with the score-pipeline run.
#
# A previous run of cluster_lp3_phase0_gpu.sh / the older monolith left two
# cells without these JSONs:
#
#   results/lagpair_ablation_3seed/analysis/absolute_seed1_analysis.json   MISSING
#   results/lagpair_ablation_3seed/analysis/absolute_seed2_analysis.json   MISSING
#
# (rope_seed{0,1,2}, alibi_seed{0,1,2}, and absolute_seed0 are present.)
#
# This wrapper runs the same script (run_positional_analysis.py) the chain's
# Phase 0 calls, but limited to --pe-types absolute --seeds 1 2.  Output
# format and hyperparameters match Phase 0 exactly so the new JSONs are
# drop-in compatible with everything that already consumed Phase 0 output.
#
# Cost.  2 cells × (Optuna 150 trials + 4 layers × 50 DSM epochs + bootstrap)
# is roughly 60-90 minutes on one RTX 5000 Ada.  Walltime 4h is generous.
#
# Submit:
#   sbatch scripts/cluster_lp3_phase0_absolute_fillgap.sh
#
# After completion: re-run the Gaussian baseline to fill its missing 8 cells:
#   sbatch scripts/cluster_lagpair_gaussian_baseline.sh
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
# GPU discovery (same pattern as cluster_lp3_phase0_gpu.sh)
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

# Light keepalive — same pattern as cluster_lp3_phase0_gpu.sh.
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
" >> logs/lp3_phase0_abs_fill_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Configuration — match cluster_lp3_phase0_gpu.sh exactly so new JSONs are
# drop-in compatible.
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"
OUT_DIR="results/lagpair_ablation_3seed"
ANALYSIS_DIR="$OUT_DIR/analysis"

PE_TYPES="absolute"
SEEDS="1 2"

W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
BOOTSTRAP_N=100

mkdir -p "$ANALYSIS_DIR" logs

echo "============================================================"
echo " Phase 0 fill-in: absolute_seed{1,2} per-seed analysis JSONs"
echo " Models:  $MODELS_DIR"
echo " Output:  $ANALYSIS_DIR"
echo " Cells:   $PE_TYPES × seeds {$SEEDS} = 2 cells × 4 layers"
echo " Started: $(date)"
echo "============================================================"

# Sanity check: both seed dirs exist and have the activations we need.
# (Models themselves: present for all seeds since rope/alibi seed{0,1,2} +
# absolute_seed0 all worked through Phase 0.  test_acts.npy might not yet
# exist for absolute_seed{1,2}; if not, train_transformer_toys.py is invoked
# in --skip-training mode to extract.)
for SEED in $SEEDS; do
    _seed_dir="$MODELS_DIR/absolute_seed${SEED}"
    if [ ! -f "$_seed_dir/model.pt" ]; then
        echo "ERROR: $_seed_dir/model.pt missing — cannot fill gap." >&2
        exit 1
    fi
    if [ ! -f "$_seed_dir/test_acts.npy" ] || [ ! -f "$_seed_dir/train_acts.npy" ]; then
        echo "  Re-extracting activations: absolute_seed${SEED}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
            --data-dir "$DATA_DIR" --out-dir "$MODELS_DIR" \
            --pe-types absolute --seed "$SEED" \
            --skip-training --save-logits --device cuda:0
    fi
done

# ---------------------------------------------------------------------------
# Skip if both target JSONs already exist (idempotent).
# ---------------------------------------------------------------------------
ALL_PRESENT=1
for SEED in $SEEDS; do
    if [ ! -f "$ANALYSIS_DIR/absolute_seed${SEED}_analysis.json" ]; then
        ALL_PRESENT=0; break
    fi
done

if [ "$ALL_PRESENT" = "1" ]; then
    echo "  Both target JSONs already exist — nothing to do."
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_positional_analysis.py \
        --data-dir              "$DATA_DIR" \
        --models-dir            "$MODELS_DIR" \
        --out-dir               "$ANALYSIS_DIR" \
        --pe-types              $PE_TYPES \
        --seeds                 $SEEDS \
        --w                     "$W" \
        --max-lag               "$MAX_LAG" \
        --pca-dim               "$PCA_DIM" \
        --epochs                "$SCORE_EPOCHS" \
        --score-tuning-trials   "$TUNING_TRIALS" \
        --tune-objective        null_contrast \
        --bootstrap-n           "$BOOTSTRAP_N" \
        --save-score-models \
        --device                cuda:0
fi

# ---------------------------------------------------------------------------
# Verify and refresh analysis_summary.json so anything that consumes the
# aggregated summary (compare_hidden_vs_logit, etc.) sees the new cells.
# ---------------------------------------------------------------------------
echo ""
echo "  Per-seed JSONs in $ANALYSIS_DIR (absolute only):"
for SEED in $SEEDS; do
    _f="$ANALYSIS_DIR/absolute_seed${SEED}_analysis.json"
    if [ -f "$_f" ]; then
        echo "    OK   $_f"
    else
        echo "    MISS $_f" >&2
    fi
done

# Re-run aggregate so summary picks up the new cells.
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
echo ""
echo "  Refreshing $SUMMARY_PATH"
python scripts/aggregate_positional_results.py --out-dir "$ANALYSIS_DIR"

echo ""
echo "============================================================"
echo " Phase 0 fill-in DONE — $(date)"
echo " Next:  sbatch scripts/cluster_lagpair_gaussian_baseline.sh"
echo "        (will now produce 36/36 cells instead of 28/36)"
echo "============================================================"
