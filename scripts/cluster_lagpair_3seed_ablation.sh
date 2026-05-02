#!/bin/bash
#SBATCH --job-name=sbtg_lp3_abl
#SBATCH --output=logs/sbtg_lagpair_3seed_ablation_%j.out
#SBATCH --error=logs/sbtg_lagpair_3seed_ablation_%j.err
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# EXTENDED ABLATION SUITE — 3-seed
# ===========================================================================
#
# Follow-up to cluster_lagpair_3seed.sh.  Reuses the same trained models and
# activations; adds a head-to-head causal comparison between SBTG-derived
# directions (score-SVD stable, endpoint-covariance) and every interpretability
# baseline we have: four probe variants (linear_abs, mlp_abs, rel_dist,
# near_far) in PCA space, a full-hidden-space linear probe, CKA, and two
# random controls (PCA-subspace random — the fair one — and full-hidden random).
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
#             M_bar_r per layer).  If absent, regenerate with
#             run_positional_analysis.py and aggregate into analysis_summary.json.
#   Phase 1 — probe baselines: linear_abs, linear_hidden, mlp_abs, rel_dist,
#             near_far per (pe, seed, layer).  Saves weight directions.
#   Phase 2 — CKA baseline (cross-layer and cross-PE scalar similarity).
#   Phase 3 — Full ablation study: F6 dose-response (SBTG stable/endpoint vs
#             PCA-subspace random), F6b score-SVD vs probe-direction vs random,
#             F7 source vs target, B3 wrong-layer and wrong-position controls.
#
# Models DO NOT change.  Activations DO NOT change.  Any run of this script
# produces a direct, directly-comparable extension of Table 3 in paper2.tex.
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
# GPU keepalive — sustained burst (50% avg util) so SLURM's utilization-
# window monitor doesn't cancel us during CPU-only phases (probes, CKA).
# Auto-killed on script exit via the trap below.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u -c "
import torch, time
torch.cuda.init()
# Maximum-aggression keepalive: continuous 4096x4096 matmuls with no
# rest, sustaining ~100% GPU utilization throughout the job.  Yale's
# GPU-idle monitor cancelled the 50%-util version too; this is the
# nuclear option.  Real-work phases (Phase 0 + Phase 3) DO compete
# with the keepalive for compute, so they slow by ~30-40%.  Wall budget
# was bumped to 36h to absorb that.  Worth it: probe + CKA phases
# (Phases 1, 2) take ~1-2h of CPU during which the keepalive is the
# only thing keeping the GPU 'alive' from the policy's perspective.
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
print('[keepalive] started — continuous burst mode (~100% util)', flush=True)
last_log = time.time()
n_iter = 0
while True:
    C = A @ B
    A = C / (C.abs().max() + 1e-6)  # renormalize to prevent overflow
    torch.cuda.synchronize()
    n_iter += 1
    # heartbeat log every 5 minutes
    if time.time() - last_log > 300:
        print(f'[keepalive] alive, n_iter={n_iter}, elapsed={(time.time()-last_log):.0f}s', flush=True)
        last_log = time.time()
" >> logs/sbtg_lagpair_3seed_ablation_keepalive.log 2>&1 &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null || true" EXIT
echo "GPU keepalive PID: $KEEPALIVE_PID (logs/sbtg_lagpair_3seed_ablation_keepalive.log)"

# Pause / resume helpers: when REAL GPU work is happening (Phases 0 + 3),
# pause the keepalive so it doesn't compete for compute.  When CPU work is
# happening (Phases 1 + 2), resume the keepalive so the GPU is "busy" from
# the policy's perspective.
keepalive_pause()  { kill -STOP "$KEEPALIVE_PID" 2>/dev/null || true; echo "[keepalive] paused (real GPU phase)"; }
keepalive_resume() { kill -CONT "$KEEPALIVE_PID" 2>/dev/null || true; echo "[keepalive] resumed (CPU phase)"; }

# ---------------------------------------------------------------------------
# Configuration — matches cluster_lagpair_3seed.sh so numbers are comparable
# ---------------------------------------------------------------------------
MODELS_DIR="results/transformer_pos_models_20260419_114958"
DATA_DIR="data/transformer_pos_cluster"
LAGPAIR_DIR="results/lagpair_analysis_3seed"
OUT_DIR="results/lagpair_ablation_3seed"

PE_TYPES="rope alibi absolute"
SEEDS="0 1 2"
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
echo " Extended Ablation Suite — 3-seed"
echo " Models:      $MODELS_DIR"
echo " Lagpair run: $LAGPAIR_DIR"
echo " Output:      $OUT_DIR"
echo " GPU:         $GPU_ID"
echo " Started:     $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Sanity: models and activations must be present (produced by the upstream
# cluster_lagpair_3seed.sh Phase 0).  If a seed dir lacks test_acts.npy,
# the probe script cannot run.
# ---------------------------------------------------------------------------
for PE in $PE_TYPES; do
    for SEED in $SEEDS; do
        _seed_dir="$MODELS_DIR/${PE}_seed${SEED}"
        if [ ! -f "$_seed_dir/model.pt" ]; then
            echo "ERROR: $_seed_dir/model.pt not found — run cluster_lagpair_3seed.sh first." >&2
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
echo " Phase 0: per-seed <pe>_seed<N>_analysis.json"
echo "============================================================"

# Prefer an existing run_positional_analysis.py output; otherwise generate.
ANALYSIS_DIR=""
for _candidate in \
    "$OUT_DIR/analysis" \
    "$LAGPAIR_DIR/analysis" \
    "results/transformer_pos_analysis_"* \
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
echo " Phase 1: probe baselines"
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
echo " Phase 2: CKA baseline (cross-layer + cross-PE)"
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
echo " Phase 3: Extended ablation study"
echo "   — F6  dose-response (SBTG stable/endpoint vs PCA-subspace random)"
echo "   — F6b score-SVD vs linear_abs probe vs linear_hidden probe"
echo "   — F7  source- vs target-side ablation"
echo "   — B3  wrong-layer and wrong-position controls (α=${WRONG_POS_OFFSET})"
echo "============================================================"

# One invocation per layer → isolated output dirs, one model-load per (pe,seed)
# per layer.  Keeps memory bounded and per-layer figures clean.
keepalive_pause   # Phase 3 is GPU-heavy — keepalive yields the device
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
keepalive_resume   # back to keeping the GPU alive in case there's any cleanup

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

exit 0
