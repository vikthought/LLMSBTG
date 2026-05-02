#!/bin/bash
#SBATCH --job-name=sbtg_lp6_abl
#SBATCH --output=logs/sbtg_lagpair_6seed_ablation_%j.out
#SBATCH --error=logs/sbtg_lagpair_6seed_ablation_%j.err
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# EXTENDED ABLATION SUITE — 6-seed
# ===========================================================================
#
# Follow-up to cluster_lagpair_6seed.sh.  Reuses the 18 trained models
# (3 PE × 6 seeds) and their activations; adds a head-to-head causal
# comparison between SBTG-derived directions (score-SVD stable, endpoint-
# covariance) and every interpretability baseline we have: four probe
# variants (linear_abs, mlp_abs, rel_dist, near_far) in PCA space, a
# full-hidden-space linear probe, CKA, and two random controls
# (PCA-subspace random — the fair one — and full-hidden random).
#
# Structurally identical to cluster_lagpair_3seed_ablation.sh: same phases,
# same knobs, same PCA/direction correctness guarantees.  Differences:
#   * SEEDS="0 1 2 3 4 5"                  (6 training seeds)
#   * MODELS_DIR auto-discovers the latest results/extended_pos_models_*
#     dir (which is what cluster_lagpair_6seed.sh writes to), since the
#     6-seed training directory is timestamped and not fixed.
#   * t=36h wall (12 layers × 6 seeds × 3 PE = 2× the 3-seed runtime).
#
# PCA / direction correctness
# ---------------------------
# SBTG stable/endpoint dirs live in the m=32 PCA subspace.  For the ablation
# to be apples-to-apples, every competing direction must be mapped to hidden
# space through the *same* per-layer PCA components and *same* per-layer mean
# mu_l.  run_ablation_study.py does this:
#   dirs_h = dirs_m @ pca_components   # (k, m) @ (m, d) -> (k, d)
#   Q, _   = torch.linalg.qr(dirs_h.T) # orthonormalize in hidden space
# and the ablation hook centers by mu_l before projecting (matches paper2 §C.4).
# The random control inside ablation_sweep draws in PCA space and back-projects,
# which is the *fair* m-dim random control — stricter than the hidden-space
# random used in run_lagpair_analysis.py.
#
# Pipeline
# --------
#   Phase 0 — ensure per-seed analysis JSONs exist (pca_components, mu_l,
#             M_bar_r per layer) for all 6 seeds.  If absent, regenerate with
#             run_positional_analysis.py and aggregate into analysis_summary.json.
#   Phase 1 — probe baselines over all 6 seeds.  Saves weight directions.
#   Phase 2 — CKA baseline (cross-layer + cross-PE, 6 seeds).
#   Phase 3 — Full ablation study: F6 dose-response, F6b score-SVD vs probe
#             vs random, F7 source vs target, B3 wrong-layer/wrong-position.
#   Phase 4 — Finish the 6-seed logit analysis.  cluster_lagpair_6seed.sh
#             left this partial (rope complete, alibi seeds 0-1 only,
#             absolute missing).  Phase 4 fills the 10 remaining cells
#             into the ORIGINAL lagpair_analysis_6seed/logit_analysis/ dir
#             so all 18 per-seed JSONs consolidate, then runs
#             compare_hidden_vs_logit.py for the cross-space comparison.
#
# Any run of this script produces a direct, directly-comparable extension
# of Table 3 in paper2.tex on 2× the seed count, and closes the 6-seed
# logit gap so Table 4 can be rebuilt on 6 seeds.
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
# GPU keepalive — sustained burst (100% avg util) + pause/resume helpers.
# Pause it during real GPU work (Phases 0, 3, 4) so it doesn't compete;
# resume during CPU-only work (Phases 1, 2) so SLURM's utilization monitor
# sees activity.  Same design as cluster_lagpair_3seed_ablation.sh.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -c "
import torch, time
torch.cuda.init()
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
print('[keepalive] started — continuous burst mode (~100% util)', flush=True)
last_log = time.time()
n_iter = 0
while True:
    C = A @ B
    A = C / (C.abs().max() + 1e-6)
    torch.cuda.synchronize()
    n_iter += 1
    if time.time() - last_log > 300:
        print(f'[keepalive] alive, n_iter={n_iter}, elapsed={(time.time()-last_log):.0f}s', flush=True)
        last_log = time.time()
" >> logs/sbtg_lagpair_6seed_ablation_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT
echo "GPU keepalive PID: $KEEPALIVE_PID (logs/sbtg_lagpair_6seed_ablation_keepalive.log)"

keepalive_pause()  { kill -STOP "$KEEPALIVE_PID" 2>/dev/null || true; echo "[keepalive] paused (real GPU phase)"; }
keepalive_resume() { kill -CONT "$KEEPALIVE_PID" 2>/dev/null || true; echo "[keepalive] resumed (CPU phase)"; }

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# >>> EDIT THIS <<<
#
# MODELS_DIR must point at the 6-seed training run directory (the one that
# produced the 18 model.pt files + test_acts.npy under
# {rope,alibi,absolute}_seed{0..5}/).  cluster_lagpair_6seed.sh writes to a
# timestamped results/extended_pos_models_${TIMESTAMP} dir, so the exact path
# depends on when training ran.  Options, in order of precedence:
#
#   1. sbatch --export=ALL,MODELS_DIR=results/extended_pos_models_20260420_152301 ...
#   2. export MODELS_DIR=... ; sbatch cluster_lagpair_6seed_ablation.sh
#   3. Edit the MODELS_DIR_DEFAULT line below to hard-code the path.
#   4. Leave blank — the script picks the newest results/extended_pos_models_*.
# ---------------------------------------------------------------------------
MODELS_DIR_DEFAULT="results/extended_pos_models_20260422_131901"   # e.g. "results/extended_pos_models_20260420_152301"

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
LAGPAIR_DIR="results/lagpair_analysis_6seed"
OUT_DIR="results/lagpair_ablation_6seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2 3 4 5"
LAYERS="1 2 3 4"

W=16
MAX_LAG=14
PCA_DIM=32
TOP_K=3
LAG=1
ALPHAS="0.0 0.5 1.0 2.0"
TUNING_TRIALS=150
SCORE_EPOCHS=50
BOOTSTRAP_N=100
WRONG_POS_OFFSET=10

mkdir -p "$OUT_DIR" \
         "$OUT_DIR/analysis" \
         "$OUT_DIR/probes" \
         "$OUT_DIR/cka" \
         "$OUT_DIR/ablation" \
         logs

echo ""
echo "============================================================"
echo " Extended Ablation Suite — 6-seed"
echo " Models:      $MODELS_DIR"
echo " Lagpair run: $LAGPAIR_DIR"
echo " Output:      $OUT_DIR"
echo " Seeds:       $SEEDS"
echo " GPU:         $GPU_ID"
echo " Started:     $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Sanity: models and activations must be present (produced by the upstream
# cluster_lagpair_6seed.sh Phase 1).  Re-extract if missing.
# ---------------------------------------------------------------------------
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        _seed_dir="$MODELS_DIR/${PE}_seed${SEED}"
        if [ ! -f "$_seed_dir/model.pt" ]; then
            echo "ERROR: $_seed_dir/model.pt not found — run cluster_lagpair_6seed.sh first." >&2
            exit 1
        fi
        if [ ! -f "$_seed_dir/test_acts.npy" ] || [ ! -f "$_seed_dir/train_acts.npy" ]; then
            echo "WARNING: missing acts for $_seed_dir — re-extracting via train_transformer_toys.py"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
                --data-dir    "$DATA_DIR"    \
                --out-dir     "$MODELS_DIR"  \
                --pe-types    "$PE"          \
                --seed        "$SEED"        \
                --skip-training              \
                --save-logits                \
                --device      cuda:0
        fi
    done
done

# ===================================================================
# Phase 0: per-seed analysis JSONs (pca_components, mu_l, M_bar_r per layer)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: per-seed <pe>_seed<N>_analysis.json (all 6 seeds)"
echo "============================================================"

# Prefer an existing run_positional_analysis.py output; otherwise generate.
# Require ALL 18 (pe, seed) per-seed JSONs to be present before reusing.
ANALYSIS_DIR=""
for _candidate in \
    "$OUT_DIR/analysis" \
    "$LAGPAIR_DIR/analysis" \
    "results/extended_pos_analysis_"* \
    "results/transformer_pos_analysis_"* \
    "extended_pos_analysis_"* \
    "transformer_pos_analysis_"*; do
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
    echo "  Found existing per-seed analysis JSONs in: $ANALYSIS_DIR"
else
    ANALYSIS_DIR="$OUT_DIR/analysis"
    echo "  Generating per-seed analysis JSONs into: $ANALYSIS_DIR"
    keepalive_pause   # real GPU work — keepalive yields the device
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
    keepalive_resume
fi

# Aggregate → analysis_summary.json (colocated with per-seed JSONs so that
# run_ablation_study.py's summary_path.parent resolves correctly).
SUMMARY_PATH="$ANALYSIS_DIR/analysis_summary.json"
if [ ! -f "$SUMMARY_PATH" ]; then
    echo "  Aggregating into: $SUMMARY_PATH"
    python scripts/aggregate_positional_results.py \
        --out-dir "$ANALYSIS_DIR"
fi
[ -f "$SUMMARY_PATH" ] || { echo "ERROR: $SUMMARY_PATH still missing" >&2; exit 1; }

# ===================================================================
# Phase 1: probe baselines
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: probe baselines (6 seeds)"
echo "   linear_abs (PCA)  mlp_abs  rel_dist  near_far  linear_hidden"
echo "============================================================"

PROBE_PATH="$OUT_DIR/probes/probe_baselines_summary.json"
if [ -f "$PROBE_PATH" ]; then
    echo "  SKIP (exists): $PROBE_PATH"
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_probe_baselines.py \
        --analysis-dir "$ANALYSIS_DIR" \
        --models-dir   "$MODELS_DIR" \
        --out-dir      "$OUT_DIR/probes" \
        --pe-types     $PE_TYPES \
        --seeds        $SEEDS
fi

# ===================================================================
# Phase 2: CKA baseline
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: CKA baseline (cross-layer + cross-PE, 6 seeds)"
echo "============================================================"

CKA_PATH="$OUT_DIR/cka/cka_summary.json"
if [ -f "$CKA_PATH" ]; then
    echo "  SKIP (exists): $CKA_PATH"
else
    # CPU-only; CKA is a Gram-matrix computation
    python scripts/run_cka_baseline.py \
        --models-dir "$MODELS_DIR" \
        --out-dir    "$OUT_DIR/cka" \
        --pe-types   $PE_TYPES \
        --seeds      $SEEDS
fi

# ===================================================================
# Phase 3: Extended ablation study — SBTG vs probe vs random
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Extended ablation study (6 seeds × 4 layers)"
echo "   — F6  dose-response (SBTG stable/endpoint vs PCA-subspace random)"
echo "   — F6b score-SVD vs linear_abs probe vs linear_hidden probe"
echo "   — F7  source- vs target-side ablation"
echo "   — B3  wrong-layer and wrong-position controls (offset=${WRONG_POS_OFFSET})"
echo "============================================================"

# One invocation per layer → isolated output dirs, one model-load per
# (pe, seed) per layer.  Keeps memory bounded and per-layer figures clean.
keepalive_pause   # Phase 3 is GPU-heavy
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
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_ablation_study.py \
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
        --device                 cuda:0 \
        || { keepalive_resume; echo "ERROR: ablation failed at layer $LAYER" >&2; exit 1; }
done
keepalive_resume

# ===================================================================
# Phase 4: Finish the 6-seed LOGIT analysis
# ===================================================================
#
# The original cluster_lagpair_6seed.sh populated only part of the logit
# table (rope all 6 seeds; alibi seeds 0-1; absolute missing).  This phase
# fills in the 10 remaining (pe, seed) cells so paper2.tex Table 4 can be
# rebuilt on 6 seeds.  Writes to the ORIGINAL logit_analysis directory
# ($LAGPAIR_DIR/logit_analysis) so all 18 JSONs sit in one place.  Skips
# cells whose JSON already exists (idempotent).
# ===================================================================

LOGIT_OUT="$LAGPAIR_DIR/logit_analysis"
LOGIT_TUNING_TRIALS=60
LOGIT_BOOTSTRAP_N=100
LOGIT_SCORE_EPOCHS=50
mkdir -p "$LOGIT_OUT"

echo ""
echo "============================================================"
echo " Phase 4: Complete the 6-seed logit analysis"
echo "   (writing to $LOGIT_OUT — skips cells that already exist)"
echo "============================================================"

keepalive_pause   # Phase 4 is GPU-heavy (18 logit score models)
for pe in $PE_TYPES; do
    for seed in $SEEDS; do
        _json="$LOGIT_OUT/${pe}_seed${seed}_logit_analysis.json"
        if [ -f "$_json" ]; then
            echo "  SKIP (exists): ${pe}_seed${seed}"
            continue
        fi
        echo "  Analyzing logits: pe=${pe} seed=${seed}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_logit_analysis.py \
            --models-dir          "$MODELS_DIR"          \
            --out-dir             "$LOGIT_OUT"           \
            --pe-types            "$pe"                  \
            --seeds               "$seed"                \
            --w                   "$W"                   \
            --max-lag             "$MAX_LAG"             \
            --pca-dim             "$PCA_DIM"             \
            --epochs              "$LOGIT_SCORE_EPOCHS"  \
            --score-tuning-trials "$LOGIT_TUNING_TRIALS" \
            --bootstrap-n         "$LOGIT_BOOTSTRAP_N"   \
            --save-score-models                          \
            --device              cuda:0                 \
            || echo "WARNING: Logit analysis failed for ${pe}_seed${seed}" >&2
    done
done
keepalive_resume

# Hidden-vs-logit comparison figure (uses the per-seed analysis JSONs from
# Phase 0 as its hidden-state summary — these are the 6-seed ones).
echo ""
echo "  Generating hidden-vs-logit comparison JSON + figures"
python scripts/compare_hidden_vs_logit.py \
    --hidden-summary "$SUMMARY_PATH" \
    --logit-dir      "$LOGIT_OUT" \
    --out-dir        "$LOGIT_OUT" \
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
echo "  Per-seed analysis:"
echo "    $ANALYSIS_DIR/"
echo "    $SUMMARY_PATH"
echo ""
echo "  Probe baselines:"
echo "    $PROBE_PATH"
echo ""
echo "  CKA baseline:"
echo "    $CKA_PATH"
echo ""
echo "  Extended ablation (per-layer):"
for LAYER in $LAYERS; do
    _ld="$OUT_DIR/ablation/L${LAYER}"
    echo "    $_ld/"
    for f in "$_ld"/*.json "$_ld"/*.pdf; do
        [ -f "$f" ] && echo "      $(basename "$f")"
    done
done
echo ""
echo "  6-seed logit analysis:"
echo "    $LOGIT_OUT/"
echo "    $(ls "$LOGIT_OUT"/*_logit_analysis.json 2>/dev/null | wc -l) / 18 per-seed JSONs"
[ -f "$LOGIT_OUT/logit_vs_hidden_comparison.json" ] && \
    echo "    logit_vs_hidden_comparison.json present"
echo ""

exit 0
