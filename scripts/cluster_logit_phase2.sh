#!/bin/bash
#SBATCH --job-name=sbtg_logit_phase2
#SBATCH --output=logs/sbtg_logit_phase2_%j.out
#SBATCH --error=logs/sbtg_logit_phase2_%j.err
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=128G

# ===========================================================================
# SBTG Logit-Space Positional Signature Analysis — Phase 2 (Pretrained Models)
#
# Extracts logits from pretrained HF models (GPT-2, BLOOM, Pythia) with
# known PE types, runs score-geometric analysis, and compares against
# controlled toy model signatures from Phase 1.
#
# Prerequisites: Phase 1 results must exist for the comparison step.
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true

if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
echo "Python: $(which python) ($(python --version))"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_GPU:-6}

python -m pip install -e ".[all]" --quiet
python -m pip install datasets>=2.14.0 --quiet

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Target models — all manageable on a single GPU
MODELS=("gpt2" "bloom-560m" "pythia-410m")
SEQ_LEN=64
N_SEQUENCES=2000
LOGIT_PCA_DIM=256
BATCH_SIZE=8

# Output directories
PRETRAINED_LOGITS="results/pretrained_logits_${TIMESTAMP}"
PRETRAINED_ANALYSIS="results/pretrained_logit_analysis_${TIMESTAMP}"
COMPARISON_OUT="results/logit_phase2_comparison_${TIMESTAMP}"

mkdir -p "$PRETRAINED_LOGITS" "$PRETRAINED_ANALYSIS" "$COMPARISON_OUT" logs

# ---------------------------------------------------------------------------
# Find Phase 1 results for cross-phase comparison
# ---------------------------------------------------------------------------
PHASE1_DIR=""
# Check standalone logit phase1 dirs first, then analysis_figures nested dirs
for d in results/logit_analysis_phase1_* results/transformer_pos_analysis_*/logit_analysis; do
    if [ -d "$d" ]; then
        if ls "$d"/*_logit_analysis.json 1>/dev/null 2>&1; then
            PHASE1_DIR="$d"
            break
        fi
    fi
done

if [ -n "$PHASE1_DIR" ]; then
    echo "Found Phase 1 results: $PHASE1_DIR"
else
    echo "WARNING: No Phase 1 results found — cross-phase comparison will be skipped"
fi

# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
_raw_gpus=""
if   [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _raw_gpus="$CUDA_VISIBLE_DEVICES"
elif [[ -n "${SLURM_STEP_GPUS:-}"      ]]; then
    _raw_gpus="$SLURM_STEP_GPUS"
elif [[ -n "${SLURM_JOB_GPUS:-}"       ]]; then
    _raw_gpus="$SLURM_JOB_GPUS"
else
    _raw_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader \
                | tr '\n' ',' | sed 's/,$//')
fi

GPU_IDS=()
IFS=',' read -ra _raw_ids <<< "$_raw_gpus"
for _id in "${_raw_ids[@]}"; do
    _id="${_id#"${_id%%[![:space:]]*}"}"
    _id="${_id%"${_id##*[![:space:]]}"}"
    [[ -n "$_id" ]] && GPU_IDS+=("$_id")
done
[[ ${#GPU_IDS[@]} -eq 0 ]] && { echo "ERROR: no GPU IDs detected." >&2; exit 1; }
GPU_ID=${GPU_IDS[0]}
echo "Using GPU: $GPU_ID"

# ---------------------------------------------------------------------------
# STEP 1: Extract logits from pretrained models
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 1: Extract pretrained model logits"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/extract_pretrained_logits.py \
    --models       "${MODELS[@]}" \
    --seq-len      $SEQ_LEN \
    --n-sequences  $N_SEQUENCES \
    --logit-pca-dim $LOGIT_PCA_DIM \
    --batch-size   $BATCH_SIZE \
    --corpus       natural \
    --out-dir      "$PRETRAINED_LOGITS" \
    --device       cuda:0 \
    || { echo "ERROR: Logit extraction failed" >&2; exit 1; }

echo "Logit extraction complete."

# ---------------------------------------------------------------------------
# STEP 2: Score-geometric analysis on pretrained logits
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 2: Pretrained logit-space score-geometric analysis"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_pretrained_logit_analysis.py \
    --pretrained-dir "$PRETRAINED_LOGITS" \
    --out-dir        "$PRETRAINED_ANALYSIS" \
    --models         "${MODELS[@]}" \
    --w              16 \
    --max-lag        5 \
    --pca-dim        32 \
    --epochs         5 \
    --score-tuning-trials 150 \
    --tune-objective null_contrast \
    --bootstrap-n    500 \
    --save-score-models \
    --device         cuda:0 \
    || echo "WARNING: Pretrained analysis failed — continuing" >&2

echo "Pretrained analysis complete."

# ---------------------------------------------------------------------------
# STEP 3: Cross-phase comparison (controlled vs pretrained)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 3: Cross-phase comparison"
echo "============================================================"

if [ -n "$PHASE1_DIR" ]; then
    python scripts/compare_controlled_vs_pretrained.py \
        --phase1-dir "$PHASE1_DIR" \
        --phase2-dir "$PRETRAINED_ANALYSIS" \
        --out-dir    "$COMPARISON_OUT" \
        --pe-types   rope alibi absolute \
        --seeds      0 \
        --models     "${MODELS[@]}" \
        || echo "WARNING: Cross-phase comparison failed — continuing" >&2
else
    echo "  SKIP: No Phase 1 results available"
fi

# ---------------------------------------------------------------------------
# STEP 4: Generate figures (Phase 2 logit figures)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 4: Generating figures"
echo "============================================================"

# Find hidden-state summary for comparison figures
HIDDEN_SUMMARY=""
for d in results/transformer_pos_analysis_*; do
    if [ -d "$d" ] && [ -f "$d/analysis_summary.json" ]; then
        HIDDEN_SUMMARY="$d/analysis_summary.json"
        break
    fi
done

FIGS_DIR="${COMPARISON_OUT}/figures"
mkdir -p "$FIGS_DIR"

if [ -n "$HIDDEN_SUMMARY" ]; then
    # Use Phase 2 analysis results with the logit plotting script
    python scripts/plot_logit_signatures.py \
        --pretrained-dir     "$PRETRAINED_ANALYSIS" \
        --pretrained-models  "${MODELS[@]}" \
        --hidden-summary     "$HIDDEN_SUMMARY" \
        --out-dir            "$FIGS_DIR" \
        || echo "WARNING: Figure generation failed — continuing" >&2
else
    echo "  SKIP: No hidden-state summary for comparison figures"
fi

echo ""
echo "============================================================"
echo " PHASE 2 COMPLETE"
echo "============================================================"
echo "  pretrained logits    -> $PRETRAINED_LOGITS"
echo "  pretrained analysis  -> $PRETRAINED_ANALYSIS"
echo "  comparison results   -> $COMPARISON_OUT"
echo "  figures              -> $FIGS_DIR"
