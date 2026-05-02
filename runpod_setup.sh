#!/bin/bash
# ===========================================================================
# RunPod bootstrap for the 4 pending matched-toy experiments + Sprint 4
# ===========================================================================
#
# Usage:
#   POD_ROLE=matched_toy    bash runpod_setup.sh        # Pod 1
#   POD_ROLE=pretrained     bash runpod_setup.sh        # Pods 2 / 3
#
# Assumes the pod has src/, scripts/, requirements.txt at the working dir
# already (you upload those manually). Matched-toy pod additionally needs the
# pre-staged results/ tree per the manifest in CurrentRun.md.
#
# After this script completes, the matched-toy pod can run any of:
#
#   bash run_pod1_jobs.sh                                # sequential, all 5 jobs
#   nohup bash run_pod1_jobs.sh > pod1.log 2>&1 &        # background
#
# Pretrained pods run via:
#
#   POD_TIER=A bash run_pretrained_jobs.sh               # array 0-6
#   POD_TIER=B bash run_pretrained_jobs.sh               # array 7-13
#
# ===========================================================================

set -euo pipefail

POD_ROLE="${POD_ROLE:?must set POD_ROLE=matched_toy or POD_ROLE=pretrained}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

echo "============================================================"
echo " RunPod setup — POD_ROLE=$POD_ROLE"
echo " Working dir: $PROJECT_DIR"
echo " Started:     $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Sanity-check uploaded files
# ---------------------------------------------------------------------------
for f in src scripts requirements.txt; do
    if [ ! -e "$f" ]; then
        echo "ERROR: $f missing in $PROJECT_DIR — upload it before running this script." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 2. Python env
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 not on PATH. Use a RunPod template with Python 3.10+." >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo ""
    echo "--- Creating venv ---"
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

# Make `src.sbtg` importable everywhere without an editable install
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Sanity GPU check (non-fatal — Sprint 1b is CPU-only)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# 3. Role-specific setup
# ---------------------------------------------------------------------------

if [ "$POD_ROLE" = "matched_toy" ]; then

    MODELS_DIR="${MODELS_DIR:-results/transformer_pos_models_20260419_114958}"
    ANALYSIS_DIR="${ANALYSIS_DIR:-results/lagpair_ablation_3seed/analysis}"
    DATA_DIR="${DATA_DIR:-data/transformer_pos_cluster}"

    PE_TYPES="rope alibi absolute"
    SEEDS="0 1 2"

    # 3a. Verify uploaded payload
    echo ""
    echo "--- Verifying uploaded artifacts ---"
    miss=0
    for pe in $PE_TYPES; do
        for s in $SEEDS; do
            for f in "$MODELS_DIR/${pe}_seed${s}/model.pt" \
                     "$ANALYSIS_DIR/${pe}_seed${s}_analysis.json"; do
                if [ ! -f "$f" ]; then
                    echo "  MISSING: $f" >&2; miss=$((miss+1))
                fi
            done
        done
    done
    for f in "$ANALYSIS_DIR/analysis_summary.json" \
             "results/lagpair_analysis_3seed/lagpair_metrics.json" \
             "results/rope_base_context_grid/analysis/rope_grid_summary.json"; do
        if [ ! -f "$f" ]; then
            echo "  MISSING: $f" >&2; miss=$((miss+1))
        fi
    done
    if [ "$miss" -gt 0 ]; then
        echo "ERROR: $miss required files missing. Upload from old cluster per CurrentRun.md manifest." >&2
        exit 1
    fi
    echo "  All required artifacts present."

    # 3b. Regenerate synthetic test data deterministically (--seed 42 matches cluster)
    if [ ! -f "$DATA_DIR/metadata.json" ]; then
        echo ""
        echo "--- Generating synthetic data ($DATA_DIR) ---"
        python scripts/generate_transformer_pos_data.py \
            --out-dir   "$DATA_DIR" \
            --seed      42 \
            --n-train   100000 \
            --n-val     5000 \
            --n-test    20000 \
            --seq-len   64 \
            --vocab-size 128
    else
        echo "  Data dir already populated — skipping regen."
    fi

    # 3c. Regenerate activations from model.pt where missing
    echo ""
    echo "--- Regenerating activations from model.pt where needed ---"
    for pe in $PE_TYPES; do
        for s in $SEEDS; do
            d="$MODELS_DIR/${pe}_seed${s}"
            if [ ! -f "$d/test_acts.npy" ] || [ ! -f "$d/train_acts.npy" ]; then
                echo "  ${pe}_seed${s}: extracting acts (model.pt + deterministic test data)"
                python scripts/train_transformer_toys.py \
                    --data-dir      "$DATA_DIR" \
                    --out-dir       "$MODELS_DIR" \
                    --pe-types      "$pe" \
                    --seed          "$s" \
                    --skip-training \
                    --save-logits \
                    --device        cuda:0
            else
                echo "  ${pe}_seed${s}: acts present, skipping"
            fi
        done
    done

    # 3d. Pre-create output dirs the sprints will write into
    mkdir -p \
        results/lagpair_ablation_3seed/multilag_ablation \
        results/lagpair_ablation_3seed/second_order_baselines \
        results/lagpair_ablation_3seed/rope_rho_tracking \
        results/lagpair_ablation_3seed/pair_shuffled_score \
        results/sae_modern_3seed_k32/batchtopk/ablation \
        logs

    # 3e. Drop a job runner that executes the 5 sprints sequentially
    cat > run_pod1_jobs.sh <<'JOBS'
#!/bin/bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

MODELS_DIR="${MODELS_DIR:-results/transformer_pos_models_20260419_114958}"
ANALYSIS_DIR="${ANALYSIS_DIR:-results/lagpair_ablation_3seed/analysis}"
DATA_DIR="${DATA_DIR:-data/transformer_pos_cluster}"
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
OUT_DIR="results/lagpair_ablation_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
LAYERS="1 2 3 4"

# --- Study 3: multilag (longest, ~12h) ---
log "Study 3 (multilag) START"
python scripts/run_multilag_ablation.py \
    --summary-path     "$SUMMARY_PATH" \
    --models-dir       "$MODELS_DIR" \
    --data-dir         "$DATA_DIR" \
    --out-dir          "$OUT_DIR/multilag_ablation" \
    --pe-types         $PE_TYPES \
    --seeds            $SEEDS \
    --layers           $LAYERS \
    --num-random-draws "${NUM_RANDOM_DRAWS:-20}" \
    --alpha 2.0 --device cuda:0
log "Study 3 (multilag) DONE"

# --- Sprint 1a: second-order baselines ---
log "Sprint 1a (second-order) START"
python scripts/run_second_order_baselines.py \
    --summary-path "$SUMMARY_PATH" --models-dir "$MODELS_DIR" --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR/second_order_baselines" \
    --pe-types $PE_TYPES --seeds $SEEDS --layers $LAYERS \
    --top-k 3 --alpha 2.0 --device cuda:0
log "Sprint 1a DONE"

# --- Sprint 1b: RoPE rho tracking (CPU, seconds) ---
log "Sprint 1b (RoPE rho tracking) START"
python scripts/analyze_rope_rho_tracking.py \
    --matched-metrics results/lagpair_analysis_3seed/lagpair_metrics.json \
    --grid-summary    results/rope_base_context_grid/analysis/rope_grid_summary.json \
    --out-dir         "$OUT_DIR/rope_rho_tracking"
log "Sprint 1b DONE"

# --- Sprint 2: pair-shuffled score model retrain ---
log "Sprint 2 (pair-shuffled) START"
python scripts/run_pair_shuffled_score_ablation.py \
    --summary-path "$SUMMARY_PATH" --models-dir "$MODELS_DIR" --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR/pair_shuffled_score" \
    --pe-types $PE_TYPES --seeds $SEEDS --layers $LAYERS \
    --score-epochs "${SCORE_EPOCHS:-50}" --device cuda:0
log "Sprint 2 DONE"

# --- Sprint 3: Modern SAE k=32 (BatchTopK only by default) ---
log "Sprint 3 (Modern SAE k=32) START"
python scripts/run_modern_sae_baselines.py \
    --summary-path "$SUMMARY_PATH" --models-dir "$MODELS_DIR" --data-dir "$DATA_DIR" \
    --out-dir "results/sae_modern_3seed_k32/batchtopk" \
    --pe-types $PE_TYPES --seeds $SEEDS --layers $LAYERS \
    --variant batchtopk --k-mode fixed --k-fallback 32 \
    --device cuda:0
log "Sprint 3 DONE"

log "ALL POD 1 JOBS COMPLETE"
JOBS
    chmod +x run_pod1_jobs.sh

    echo ""
    echo "============================================================"
    echo " Pod 1 (matched_toy) READY"
    echo "============================================================"
    echo ""
    echo " Run all 5 jobs sequentially (foreground):"
    echo "   bash run_pod1_jobs.sh"
    echo ""
    echo " Or in background:"
    echo "   nohup bash run_pod1_jobs.sh > pod1.log 2>&1 &"
    echo "   tail -f pod1.log"
    echo ""
    echo " Outputs land in:"
    echo "   results/lagpair_ablation_3seed/{multilag_ablation,second_order_baselines,rope_rho_tracking,pair_shuffled_score}/"
    echo "   results/sae_modern_3seed_k32/batchtopk/"
    echo ""

elif [ "$POD_ROLE" = "pretrained" ]; then

    POD_TIER="${POD_TIER:-A}"   # A = indices 0-6, B = indices 7-13

    # 3a. Verify HF cache writable (downloads will go here)
    HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
    mkdir -p "$HF_HOME"
    export HF_HOME

    # 3b. Pre-create output dirs
    mkdir -p \
        results/pretrained_size_sweep \
        results/pretrained_size_sweep_analysis \
        logs

    # 3c. Drop a job runner that loops over the tier's array indices
    cat > run_pretrained_jobs.sh <<'JOBS'
#!/bin/bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Mirror the MODELS=(...) array from cluster_pretrained_size_sweep.sh.
# Tier A = indices 0-6, Tier B = indices 7-13.
POD_TIER="${POD_TIER:-A}"
case "$POD_TIER" in
    A) IDXS=(0 1 2 3 4 5 6) ;;
    B) IDXS=(7 8 9 10 11 12 13) ;;
    *) echo "ERROR: POD_TIER must be A or B" >&2; exit 1 ;;
esac

for IDX in "${IDXS[@]}"; do
    log "Sprint 4 array idx=$IDX (tier=$POD_TIER) START"
    SLURM_ARRAY_TASK_ID="$IDX" \
    MODE="${MODE:-both}" \
    LOGIT_PCA_DIM="${LOGIT_PCA_DIM:-256}" \
        bash scripts/cluster_pretrained_size_sweep.sh || \
        log "WARNING: idx=$IDX failed — continuing"
    log "Sprint 4 array idx=$IDX DONE"
done

log "ALL POD ${POD_TIER} JOBS COMPLETE"
JOBS
    chmod +x run_pretrained_jobs.sh

    echo ""
    echo "============================================================"
    echo " Pod (pretrained, tier $POD_TIER) READY"
    echo "============================================================"
    echo ""
    echo " Run the tier's models sequentially:"
    echo "   POD_TIER=$POD_TIER bash run_pretrained_jobs.sh"
    echo ""
    echo " Or in background:"
    echo "   nohup env POD_TIER=$POD_TIER bash run_pretrained_jobs.sh > pretrained_$POD_TIER.log 2>&1 &"
    echo ""
    echo " Outputs land in:"
    echo "   results/pretrained_size_sweep/<model_key>/"
    echo "   results/pretrained_size_sweep_analysis/<model_key>/"
    echo ""

else
    echo "ERROR: unknown POD_ROLE=$POD_ROLE — use matched_toy or pretrained" >&2
    exit 1
fi

echo " Setup complete: $(date)"
