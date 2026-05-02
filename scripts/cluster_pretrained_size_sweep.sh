#!/bin/bash
#SBATCH --job-name=pretrained_sweep
#SBATCH --output=logs/pretrained_sweep_%A_%a.out
#SBATCH --error=logs/pretrained_sweep_%A_%a.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G
#SBATCH --array=0-13

# ===========================================================================
# Pretrained-LLM size sweep + multi-layer extraction (paper3 §5.4 follow-up)
# ===========================================================================
#
# What this runs
# --------------
# A SLURM array job — one task per pretrained model. For each model:
#   1. extract_pretrained_internals.py    → per-model PCA-reduced features
#   2. run_pretrained_internal_analysis.py → per-cell metrics JSON
#
# Cells per model = {logit} ∪ {hidden_L<N> for each chosen depth}
# Default depths: 10% / 50% / 90% of the model's layer stack (1-indexed).
#
# Models — three fairness tiers
# ------------------------------
# Tier 1 (within-family size sweeps; the only "clean" comparison):
#   RoPE     — pythia 410m, 1b, 1.4b, 2.8b   (Pile, GPT-NeoX tokenizer)
#   ALiBi    — bloom 560m, 1b1, 1b7, 3b      (ROOTS, BLOOM tokenizer)
#   Absolute — gpt2 small, medium, large, xl (WebText, GPT-2 BPE)
#
# Tier 2 (matched-scale cross-family — already present in paper3 §5.4):
#   gpt2 (124M)+ pythia-410m + bloom-560m + opt-350m
#
# Tier 3 (cross-corpus same-PE diagnostic; upside, not core):
#   openllama-3b vs pythia-2.8b — both RoPE, RedPajama vs Pile
#
# Walltime
# --------
# Per-model task: extraction (10–60 min depending on size) + analysis
# (3 cells × ~10 min Optuna = ~30 min). Total per task: 1–2 h for the
# small models, up to 4–5 h for 2.8B / 3B / openllama-3b. 12 h walltime
# is comfortable headroom; stretches: 7B-class would need a separate
# config (different SBATCH options).
#
# Memory
# ------
# extract_pretrained_internals.py uses IncrementalPCA streaming, so a
# 2.8B model never materializes the full hidden-state tensor in RAM
# (per-layer working footprint = batch_size × seq_len × hidden_dim).
# The analyzer reads pre-PCA'd features at inner_pca_dim=32, so the
# analysis stage is memory-trivial.
#
# Submit
# ------
#   sbatch scripts/cluster_pretrained_size_sweep.sh
#
#   # Restrict to a specific tier:
#   sbatch --array=0-3  scripts/cluster_pretrained_size_sweep.sh   # Pythia only
#   sbatch --array=4-7  scripts/cluster_pretrained_size_sweep.sh   # BLOOM only
#   sbatch --array=8-11 scripts/cluster_pretrained_size_sweep.sh   # GPT-2 only
#   sbatch --array=12   scripts/cluster_pretrained_size_sweep.sh   # opt-350m
#   sbatch --array=13   scripts/cluster_pretrained_size_sweep.sh   # openllama-3b (Tier 3)
#
#   # Logit-only run (B6 PCA-512 robustness check at the same time):
#   MODE=logit LOGIT_PCA_DIM=512 OUT_SUFFIX=_pca512 \
#     sbatch scripts/cluster_pretrained_size_sweep.sh
#
# Aggregation (after the array finishes)
# --------------------------------------
#   python scripts/aggregate_pretrained_size_sweep.py \
#     --analysis-dir results/pretrained_size_sweep_analysis \
#     --out-dir      results/pretrained_size_sweep_summary
#
# (aggregator not yet written; per-model JSONs are self-contained and
#  easy to load directly. Add the aggregator when the run finishes.)
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
# Array task → model key
# ---------------------------------------------------------------------------
# Order encodes the tier structure. Adjust --array=K-K to hit one model.
MODELS=(
    # Tier 1: Pythia size sweep (RoPE)         ── indices 0-3
    "pythia-410m"
    "pythia-1b"
    "pythia-1.4b"
    "pythia-2.8b"
    # Tier 1: BLOOM size sweep (ALiBi)          ── indices 4-7
    "bloom-560m"
    "bloom-1b1"
    "bloom-1b7"
    "bloom-3b"
    # Tier 1: GPT-2 size sweep (Absolute)       ── indices 8-11
    "gpt2"
    "gpt2-medium"
    "gpt2-large"
    "gpt2-xl"
    # Tier 2: matched-scale cross-family extra  ── index 12
    "opt-350m"
    # Tier 3: cross-corpus same-PE diagnostic   ── index 13
    "openllama-3b"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this script is meant to run as a SLURM array job. Submit with sbatch." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} )); then
    echo "Array index $SLURM_ARRAY_TASK_ID out of range (max ${#MODELS[@]}-1)." >&2
    exit 1
fi

MODEL_KEY="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Configuration knobs (env-var overridable)
# ---------------------------------------------------------------------------
MODE="${MODE:-both}"                       # logit | internal | both
N_SEQUENCES="${N_SEQUENCES:-2000}"
SEQ_LEN="${SEQ_LEN:-64}"
INNER_PCA_DIM="${INNER_PCA_DIM:-32}"       # paper3 m = 32
LOGIT_PCA_DIM="${LOGIT_PCA_DIM:-256}"      # paper3 default; 512 = §8.5 robustness
W="${W:-16}"
MAX_LAG="${MAX_LAG:-14}"
TUNING_TRIALS="${TUNING_TRIALS:-60}"       # paper3 logit pipeline default
EPOCHS="${EPOCHS:-5}"                      # matched-toy default
BOOTSTRAP_N="${BOOTSTRAP_N:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"              # forward-pass batch size, not score-train

# Layer selection: comma-separated list of explicit indices, or empty for
# defaults (10/50/90% relative depths).
LAYERS="${LAYERS:-}"
LAYER_RELATIVE_DEPTHS="${LAYER_RELATIVE_DEPTHS:-0.1 0.5 0.9}"

OUT_SUFFIX="${OUT_SUFFIX:-}"
EXTRACT_DIR="results/pretrained_size_sweep${OUT_SUFFIX}"
ANALYSIS_DIR="results/pretrained_size_sweep_analysis${OUT_SUFFIX}"

mkdir -p "$EXTRACT_DIR" "$ANALYSIS_DIR" logs

echo "============================================================"
echo " Pretrained size sweep — array task $SLURM_ARRAY_TASK_ID"
echo " Model:           $MODEL_KEY"
echo " Mode:            $MODE"
echo " inner-pca-dim:   $INNER_PCA_DIM"
echo " logit-pca-dim:   $LOGIT_PCA_DIM"
echo " seqs × seq_len:  $N_SEQUENCES × $SEQ_LEN"
echo " Extract dir:     $EXTRACT_DIR"
echo " Analysis dir:    $ANALYSIS_DIR"
echo " Started:         $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: extraction
# ---------------------------------------------------------------------------
LAYER_FLAGS=""
if [[ -n "$LAYERS" ]]; then
    LAYER_FLAGS="--layers $LAYERS"
elif [[ -n "$LAYER_RELATIVE_DEPTHS" ]]; then
    LAYER_FLAGS="--layer-relative-depths $LAYER_RELATIVE_DEPTHS"
fi

echo ""
echo "----- Phase 1: extract_pretrained_internals.py -----"
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/extract_pretrained_internals.py \
    --models           "$MODEL_KEY" \
    --mode             "$MODE" \
    --n-sequences      "$N_SEQUENCES" \
    --seq-len          "$SEQ_LEN" \
    --inner-pca-dim    "$INNER_PCA_DIM" \
    --logit-pca-dim    "$LOGIT_PCA_DIM" \
    --batch-size       "$BATCH_SIZE" \
    $LAYER_FLAGS \
    --out-dir          "$EXTRACT_DIR" \
    --device           cuda:0

# ---------------------------------------------------------------------------
# Phase 2: analysis
# ---------------------------------------------------------------------------
# Map the script's --mode to the analyzer's --modes ({logit, hidden}).
case "$MODE" in
    logit)    MODES_ARG="logit" ;;
    internal) MODES_ARG="hidden" ;;
    both)     MODES_ARG="logit hidden" ;;
    *)        echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac

echo ""
echo "----- Phase 2: run_pretrained_internal_analysis.py -----"
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_pretrained_internal_analysis.py \
    --pretrained-dir   "$EXTRACT_DIR" \
    --models           "$MODEL_KEY" \
    --modes            $MODES_ARG \
    --out-dir          "$ANALYSIS_DIR" \
    --w                "$W" \
    --max-lag          "$MAX_LAG" \
    --pca-dim          "$INNER_PCA_DIM" \
    --epochs           "$EPOCHS" \
    --tuning-trials    "$TUNING_TRIALS" \
    --bootstrap-n      "$BOOTSTRAP_N" \
    --device           cuda:0

echo ""
echo "============================================================"
echo " Task $SLURM_ARRAY_TASK_ID ($MODEL_KEY) DONE — $(date)"
echo "============================================================"
echo "  Extracted: $EXTRACT_DIR/$(echo "$MODEL_KEY" | tr -c '[:alnum:]._-' '_')/"
echo "  Analyzed:  $ANALYSIS_DIR/$(echo "$MODEL_KEY" | tr -c '[:alnum:]._-' '_')/"
