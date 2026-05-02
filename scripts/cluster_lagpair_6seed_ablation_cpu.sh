#!/bin/bash
#SBATCH --job-name=sbtg_lp6_abl_cpu
#SBATCH --output=logs/sbtg_lagpair_6seed_ablation_cpu_%j.out
#SBATCH --error=logs/sbtg_lagpair_6seed_ablation_cpu_%j.err
#SBATCH -p week
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G

# ===========================================================================
# EXTENDED ABLATION SUITE — 6-seed (CPU-ONLY FALLBACK)
# ===========================================================================
#
# Mirror of cluster_lagpair_6seed_ablation.sh but runs ENTIRELY on CPU.
# Use this when the GPU version gets cancelled by the cluster's GPU-idle
# policy and the keepalive doesn't catch the policy variant in use.
#
#   * No `--gpus` SLURM directive → no GPU allocated → GPU-idle policy
#     does not apply.
#   * Submit to a CPU partition (default `week`; change `-p week` above to
#     match your cluster — Yale Grace `week` allows up to 7 days, `day`
#     caps at 24 h, `pi_sz25_cpu` is your group's CPU partition).
#   * 72 h wall: end-to-end is ~24-36 h on a 24-core node; the buffer
#     absorbs Optuna variance and Phase 4's 18 logit cells.
#
# Output goes to SEPARATE directories (so it does NOT collide with any
# GPU run, partial or complete):
#   results/lagpair_ablation_6seed_cpu/         ← analysis, probes, cka, ablation
#   results/lagpair_analysis_6seed_cpu/         ← Phase 4 logit JSONs
#
# Compute knobs trimmed for CPU tractability (the only differences vs the
# GPU script's science settings):
#   TUNING_TRIALS  150 → 30   (Optuna is the CPU bottleneck)
#   SCORE_EPOCHS    50 → 25   (DSM converges by ~20 epochs anyway)
#   LOGIT_TUNING_TRIALS    60 → 20
#   LOGIT_SCORE_EPOCHS     50 → 20
# Everything else (PCA dim, window width, max lag, top-k, alphas, bootstrap
# count) is identical to the GPU version, so the *direction* of every
# reported metric is comparable.  Magnitudes may be ~5-10% noisier from
# fewer trials.
#
# Idempotent: each phase skips if its output JSON exists.  Resubmitting
# after a partial run resumes from the first unfinished phase.
#
# Pipeline
# --------
#   Phase 0 — per-seed analysis JSONs (reuses GPU-run output if available).
#   Phase 1 — probe baselines (sklearn — CPU native).
#   Phase 2 — CKA baseline (numpy — CPU native).
#   Phase 3 — Extended ablation study on CPU.
#   Phase 4 — 6-seed logit completion to results/lagpair_analysis_6seed_cpu/.
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true

if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
# CPU thread budget: PyTorch + sklearn + numpy all share these
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

python -m pip install -e ".[all]" --quiet

python -c "import torch; print(f'CPU threads: {torch.get_num_threads()}')"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# >>> EDIT THIS <<<
#
# MODELS_DIR must point at the 6-seed training run directory (the one that
# produced the 18 model.pt files + test_acts.npy under
# {rope,alibi,absolute}_seed{0..5}/).  cluster_lagpair_6seed.sh writes to a
# timestamped results/extended_pos_models_${TIMESTAMP} dir, so the exact
# path depends on when training ran.  Options, in order of precedence:
#
#   1. sbatch --export=ALL,MODELS_DIR=results/extended_pos_models_20260420_152301 ...
#   2. export MODELS_DIR=... ; sbatch cluster_lagpair_6seed_ablation_cpu.sh
#   3. Edit MODELS_DIR_DEFAULT below to hard-code the path.
#   4. Leave blank — the script picks the newest results/extended_pos_models_*.
# ---------------------------------------------------------------------------
MODELS_DIR_DEFAULT="results/extended_pos_models_20260422_131901"   # match the GPU script's pin

if [[ -z "${MODELS_DIR:-}" ]]; then
    if [[ -n "$MODELS_DIR_DEFAULT" ]]; then
        MODELS_DIR="$MODELS_DIR_DEFAULT"
    else
        _cand=$(ls -d results/extended_pos_models_* 2>/dev/null | sort -r | head -n 1)
        if [[ -z "$_cand" ]]; then
            echo "ERROR: no MODELS_DIR set and no results/extended_pos_models_* found." >&2
            echo "       Edit MODELS_DIR_DEFAULT at the top of this script, or export" >&2
            echo "       MODELS_DIR=/path/to/run before sbatch." >&2
            exit 1
        fi
        MODELS_DIR="$_cand"
    fi
fi

[ -d "$MODELS_DIR" ] || { echo "ERROR: MODELS_DIR '$MODELS_DIR' not a directory" >&2; exit 1; }

DATA_DIR="data/transformer_pos_cluster"
LAGPAIR_DIR="results/lagpair_analysis_6seed"          # GPU pipeline's logit dir (read-only here)
OUT_DIR="results/lagpair_ablation_6seed_cpu"          # CPU ablation output
LOGIT_CPU_DIR="results/lagpair_analysis_6seed_cpu/logit_analysis"   # CPU logit output

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2 3 4 5"
LAYERS="1 2 3 4"

W=16
MAX_LAG=14
PCA_DIM=32
TOP_K=3
LAG=1
ALPHAS="0.0 0.5 1.0 2.0"
TUNING_TRIALS=30          # << CPU-trimmed (GPU: 150)
SCORE_EPOCHS=25           # << CPU-trimmed (GPU: 50)
BOOTSTRAP_N=100
WRONG_POS_OFFSET=10

LOGIT_TUNING_TRIALS=20    # << CPU-trimmed (GPU: 60)
LOGIT_SCORE_EPOCHS=20     # << CPU-trimmed (GPU: 50)

mkdir -p "$OUT_DIR" \
         "$OUT_DIR/analysis" \
         "$OUT_DIR/probes" \
         "$OUT_DIR/cka" \
         "$OUT_DIR/ablation" \
         "$LOGIT_CPU_DIR" \
         logs

echo ""
echo "============================================================"
echo " Extended Ablation Suite — 6-seed (CPU-ONLY)"
echo " Models:        $MODELS_DIR"
echo " Output:        $OUT_DIR"
echo " Logit (CPU):   $LOGIT_CPU_DIR"
echo " Threads:       ${OMP_NUM_THREADS}"
echo " Started:       $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Sanity: models and activations must be present.
# ---------------------------------------------------------------------------
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        _seed_dir="$MODELS_DIR/${PE}_seed${SEED}"
        if [ ! -f "$_seed_dir/model.pt" ]; then
            echo "ERROR: $_seed_dir/model.pt not found — run cluster_lagpair_6seed.sh first." >&2
            exit 1
        fi
        if [ ! -f "$_seed_dir/test_acts.npy" ] || [ ! -f "$_seed_dir/train_acts.npy" ]; then
            echo "WARNING: missing acts for $_seed_dir — re-extracting on CPU"
            python scripts/train_transformer_toys.py \
                --data-dir    "$DATA_DIR"    \
                --out-dir     "$MODELS_DIR"  \
                --pe-types    "$PE"          \
                --seed        "$SEED"        \
                --skip-training              \
                --save-logits                \
                --device      cpu
        fi
    done
done

# ===================================================================
# Phase 0: per-seed analysis JSONs
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: per-seed <pe>_seed<N>_analysis.json [CPU]"
echo "============================================================"

# Reuse existing run_positional_analysis output if any compatible run is
# already on disk — these JSONs are device-independent (no model.pt inside).
ANALYSIS_DIR=""
for _candidate in \
    "$OUT_DIR/analysis" \
    "results/lagpair_ablation_6seed/analysis" \
    "$LAGPAIR_DIR/analysis" \
    "results/extended_pos_analysis_"* \
    "results/transformer_pos_analysis_"*; do
    if [ -d "$_candidate" ]; then
        _all_present=1
        for PE in $PE_TYPES; do
            for SEED in $SEEDS; do
                if [ ! -f "$_candidate/${PE}_seed${SEED}_analysis.json" ]; then
                    _all_present=0
                    break 2
                fi
            done
        done
        if [ "$_all_present" = "1" ]; then
            ANALYSIS_DIR="$_candidate"
            break
        fi
    fi
done

if [ -n "$ANALYSIS_DIR" ]; then
    echo "  Reusing per-seed analysis JSONs from: $ANALYSIS_DIR"
else
    ANALYSIS_DIR="$OUT_DIR/analysis"
    echo "  Generating per-seed analysis JSONs into: $ANALYSIS_DIR"
    python scripts/run_positional_analysis.py \
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
        --device                cpu
fi

SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
if [ ! -f "$SUMMARY_PATH" ]; then
    echo "  Aggregating into: $SUMMARY_PATH"
    python scripts/aggregate_positional_results.py \
        --out-dir "$ANALYSIS_DIR"
fi
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH still missing" >&2; exit 1; }

# ===================================================================
# Phase 1: probe baselines (sklearn — CPU native)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: probe baselines [CPU]"
echo "============================================================"

PROBE_PATH="$OUT_DIR/probes/probe_baselines_summary.json"
if [ -f "$PROBE_PATH" ]; then
    echo "  SKIP (exists): $PROBE_PATH"
else
    python scripts/run_probe_baselines.py \
        --analysis-dir "$ANALYSIS_DIR" \
        --models-dir   "$MODELS_DIR" \
        --out-dir      "$OUT_DIR/probes" \
        --pe-types     $PE_TYPES \
        --seeds        $SEEDS
fi

# ===================================================================
# Phase 2: CKA baseline (numpy — CPU native)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: CKA baseline [CPU]"
echo "============================================================"

CKA_PATH="$OUT_DIR/cka/cka_summary.json"
if [ -f "$CKA_PATH" ]; then
    echo "  SKIP (exists): $CKA_PATH"
else
    python scripts/run_cka_baseline.py \
        --models-dir "$MODELS_DIR" \
        --out-dir    "$OUT_DIR/cka" \
        --pe-types   $PE_TYPES \
        --seeds      $SEEDS
fi

# ===================================================================
# Phase 3: Extended ablation study [CPU]
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Extended ablation study [CPU]"
echo "============================================================"

for LAYER in $LAYERS; do
    LAYER_OUT="$OUT_DIR/ablation/L${LAYER}"
    if [ -f "$LAYER_OUT/b2_probe_direction_ablation.json" ] \
       && [ -f "$LAYER_OUT/dose_response.json" ]; then
        echo "  SKIP (exists): $LAYER_OUT"
        continue
    fi
    mkdir -p "$LAYER_OUT"
    echo ""
    echo "  --- layer $LAYER ---"
    python scripts/run_ablation_study.py \
        --summary-path           "$SUMMARY_PATH" \
        --models-dir             "$MODELS_DIR" \
        --data-dir               "$DATA_DIR" \
        --out-dir                "$LAYER_OUT" \
        --pe-types               $PE_TYPES \
        --seeds                  $SEEDS \
        --layers                 "$LAYER" \
        --lag                    "$LAG" \
        --top-k                  "$TOP_K" \
        --alpha-values           $ALPHAS \
        --probe-baselines-path   "$PROBE_PATH" \
        --wrong-position-offset  "$WRONG_POS_OFFSET" \
        --ablation-alpha-b3      2.0 \
        --device                 cpu \
        || { echo "ERROR: ablation failed at layer $LAYER" >&2; exit 1; }
done

# ===================================================================
# Phase 4: Finish the 6-seed LOGIT analysis (CPU, separate output dir)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 4: 6-seed logit analysis [CPU]"
echo "   (writing to $LOGIT_CPU_DIR — independent of GPU run's"
echo "    lagpair_analysis_6seed/logit_analysis/)"
echo "============================================================"

for pe in $PE_TYPES; do
    for seed in $SEEDS; do
        _json="${LOGIT_CPU_DIR}/${pe}_seed${seed}_logit_analysis.json"
        if [ -f "$_json" ]; then
            echo "  SKIP (exists): ${pe}_seed${seed}"
            continue
        fi
        echo "  Analyzing logits: pe=${pe} seed=${seed}"
        python scripts/run_logit_analysis.py \
            --models-dir          "$MODELS_DIR"            \
            --out-dir             "$LOGIT_CPU_DIR"         \
            --pe-types            "$pe"                    \
            --seeds               "$seed"                  \
            --w                   "$W"                     \
            --max-lag             "$MAX_LAG"               \
            --pca-dim             "$PCA_DIM"               \
            --epochs              "$LOGIT_SCORE_EPOCHS"    \
            --score-tuning-trials "$LOGIT_TUNING_TRIALS"   \
            --bootstrap-n         "$BOOTSTRAP_N"           \
            --save-score-models                            \
            --device              cpu                      \
            || echo "WARNING: logit analysis failed for ${pe}_seed${seed}" >&2
    done
done

# Hidden-vs-logit comparison figure (uses Phase-0 analysis JSONs as the
# 6-seed hidden summary).
echo ""
echo "  Generating hidden-vs-logit comparison JSON + figures"
python scripts/compare_hidden_vs_logit.py \
    --hidden-summary "$SUMMARY_PATH" \
    --logit-dir      "$LOGIT_CPU_DIR" \
    --out-dir        "$LOGIT_CPU_DIR" \
    --pe-types       $PE_TYPES \
    --seeds          $SEEDS \
    || echo "WARNING: Hidden vs logit comparison failed — continuing" >&2

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
echo ""
echo "  Per-seed analysis:    $ANALYSIS_DIR/"
echo "  Probe baselines:      $PROBE_PATH"
echo "  CKA baseline:         $CKA_PATH"
echo "  Extended ablation:"
for LAYER in $LAYERS; do
    _ld="$OUT_DIR/ablation/L${LAYER}"
    echo "    $_ld/"
    for f in "$_ld"/*.json "$_ld"/*.pdf; do
        [ -f "$f" ] && echo "      $(basename "$f")"
    done
done
echo ""
echo "  6-seed logit analysis (CPU output, independent of GPU run):"
echo "    $LOGIT_CPU_DIR/"
echo "    $(ls "$LOGIT_CPU_DIR"/*_logit_analysis.json 2>/dev/null | wc -l) / 18 per-seed JSONs"
[ -f "$LOGIT_CPU_DIR/logit_vs_hidden_comparison.json" ] && \
    echo "    logit_vs_hidden_comparison.json present"
echo ""

exit 0
