#!/bin/bash
#SBATCH --job-name=sbtg_lp6_2gpu
#SBATCH --output=logs/sbtg_lagpair_6seed_2gpu_%j.out
#SBATCH --error=logs/sbtg_lagpair_6seed_2gpu_%j.err
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=256G

# ===========================================================================
# EXTENDED METRIC SUITE — 6-SEED, 2 GPU
# ===========================================================================
#
# Same pipeline as cluster_lagpair_6seed.sh but parallelised across 2 GPUs:
#
#   Phase 0 — regenerate data (CPU)
#   Phase 1 — train 18 transformers, 2 at a time (pool)
#   Phase 2 — hidden-state analysis split by seed group:
#             GPU0: seeds 0,1,2  →  _g0/
#             GPU1: seeds 3,4,5  →  _g1/
#   Phase 3 — merge JSONs + regenerate combined figures (CPU)
#   Phase 4 — logit analysis, 2 at a time (pool)
#   Phase 5 — logit comparison figures (CPU)
#
# Expected wall time: ~18-22 hrs (vs ~36 hrs on 1 GPU)
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
[[ ${#GPU_IDS[@]} -lt 2 ]] && { echo "ERROR: need 2 GPUs, got ${#GPU_IDS[@]}." >&2; exit 1; }
N_GPUS=${#GPU_IDS[@]}

echo "Allocated ${N_GPUS} GPU(s): ${GPU_IDS[*]}"

# Validate GPUs
for _id in "${GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${_id} python -c \
        "import torch; print(f'  GPU ${_id}: {torch.cuda.get_device_name(0)}')" \
        || { echo "ERROR: GPU ${_id} failed CUDA check" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# Worker-pool helpers (from cluster_full_pipeline.sh)
# ---------------------------------------------------------------------------
declare -a _SLOT_PIDS
for (( _i=0; _i<N_GPUS; _i++ )); do _SLOT_PIDS[$_i]=0; done
_FREE_SLOT=-1

pool_acquire() {
    while true; do
        for (( _s=0; _s<N_GPUS; _s++ )); do
            local _p=${_SLOT_PIDS[$_s]:-0}
            if [[ $_p -eq 0 ]] || ! kill -0 "$_p" 2>/dev/null; then
                if [[ $_p -ne 0 ]]; then
                    local _rc=0
                    wait "$_p" || _rc=$?
                    [[ $_rc -ne 0 ]] && echo "WARNING: slot ${_s} (PID ${_p}) exited ${_rc}" >&2
                fi
                _SLOT_PIDS[$_s]=0
                _FREE_SLOT=$_s
                return
            fi
        done
        sleep 3
    done
}

pool_drain() {
    local _rc=0
    for (( _s=0; _s<N_GPUS; _s++ )); do
        local _p=${_SLOT_PIDS[$_s]:-0}
        [[ $_p -ne 0 ]] && { wait "$_p" || _rc=$?; _SLOT_PIDS[$_s]=0; }
    done
    return $_rc
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="data/transformer_pos_cluster"
MODELS_DIR="results/extended_pos_models_${TIMESTAMP}"
OUT_DIR="results/lagpair_analysis_6seed"
LOGIT_OUT="${OUT_DIR}/logit_analysis"

PE_TYPES=(rope alibi absolute)
ALL_SEEDS=(0 1 2 3 4 5)
SEEDS_G0=(0 1 2)
SEEDS_G1=(3 4 5)

EPOCHS=15
BATCH_SIZE=128

W=16
MAX_LAG=14
PCA_DIM=32
TUNING_TRIALS=150
SCORE_EPOCHS=50
TOP_K=3

LOGIT_PCA_DIM=256
LOGIT_TUNING_TRIALS=60
LOGIT_BOOTSTRAP_N=100

mkdir -p "$MODELS_DIR" "$OUT_DIR" "$LOGIT_OUT" logs

echo ""
echo "============================================================"
echo " Extended Metric Suite — 6-seed, 2 GPUs"
echo " Models:  $MODELS_DIR"
echo " Output:  $OUT_DIR"
echo " GPUs:    ${GPU_IDS[*]}"
echo " Started: $(date)"
echo "============================================================"

# ===================================================================
# Phase 0: Regenerate test data (20K)
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 0: Regenerate data (N_test=20000)"
echo "============================================================"

python scripts/generate_transformer_pos_data.py \
    --out-dir    "$DATA_DIR" \
    --n-train    100000 \
    --n-val      5000 \
    --n-test     20000 \
    --seq-len    64 \
    --vocab-size 128 \
    --seed       42

# ===================================================================
# Phase 1: Train 18 transformers across 2 GPUs
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 1: Train 18 transformers (2 GPUs)"
echo "============================================================"

for SEED in "${ALL_SEEDS[@]}"; do
    pool_acquire
    slot=$_FREE_SLOT
    GPU_ID=${GPU_IDS[$slot]}
    LOG="logs/train_seed${SEED}_${TIMESTAMP}.log"
    echo "  Training seed=${SEED} → GPU ${GPU_ID} (slot ${slot})"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_transformer_toys.py \
        --data-dir      "$DATA_DIR"      \
        --out-dir       "$MODELS_DIR"    \
        --epochs        "$EPOCHS"        \
        --batch-size    "$BATCH_SIZE"    \
        --pe-types      "${PE_TYPES[@]}" \
        --seed          "$SEED"          \
        --save-logits                    \
        --logit-pca-dim "$LOGIT_PCA_DIM" \
        --device        cuda:0           \
        >> "$LOG" 2>&1 &

    _SLOT_PIDS[$slot]=$!
done

echo "  Waiting for all training to finish..."
_train_rc=0
pool_drain || _train_rc=$?
[[ $_train_rc -ne 0 ]] && echo "WARNING: some training jobs failed (rc=${_train_rc})" >&2
echo " Phase 1 complete — 18 models trained"

# ===================================================================
# Phase 2: Hidden-state analysis — 2 GPUs, split by seed group
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 2: Hidden-state analysis (2 GPUs, split seeds)"
echo "============================================================"

OUT_G0="${OUT_DIR}/_g0"
OUT_G1="${OUT_DIR}/_g1"
mkdir -p "$OUT_G0" "$OUT_G1"

LOG_G0="logs/analysis_g0_${TIMESTAMP}.log"
LOG_G1="logs/analysis_g1_${TIMESTAMP}.log"

echo "  GPU ${GPU_IDS[0]}: seeds ${SEEDS_G0[*]} → ${OUT_G0}/"
echo "  GPU ${GPU_IDS[1]}: seeds ${SEEDS_G1[*]} → ${OUT_G1}/"

# GPU 0: seeds 0,1,2
CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} python scripts/run_lagpair_analysis.py \
    --models-dir     "$MODELS_DIR" \
    --data-dir       "$DATA_DIR" \
    --out-dir        "$OUT_G0" \
    --pe-types       "${PE_TYPES[@]}" \
    --seeds          "${SEEDS_G0[@]}" \
    --layers         1 2 3 4 \
    --w              $W \
    --max-lag        $MAX_LAG \
    --pca-dim        $PCA_DIM \
    --top-k          $TOP_K \
    --tuning-trials  $TUNING_TRIALS \
    --score-epochs   $SCORE_EPOCHS \
    --alpha-values   0.0 0.5 1.0 2.0 \
    --device         cuda:0 \
    >> "$LOG_G0" 2>&1 &
_PID_G0=$!

# GPU 1: seeds 3,4,5
CUDA_VISIBLE_DEVICES=${GPU_IDS[1]} python scripts/run_lagpair_analysis.py \
    --models-dir     "$MODELS_DIR" \
    --data-dir       "$DATA_DIR" \
    --out-dir        "$OUT_G1" \
    --pe-types       "${PE_TYPES[@]}" \
    --seeds          "${SEEDS_G1[@]}" \
    --layers         1 2 3 4 \
    --w              $W \
    --max-lag        $MAX_LAG \
    --pca-dim        $PCA_DIM \
    --top-k          $TOP_K \
    --tuning-trials  $TUNING_TRIALS \
    --score-epochs   $SCORE_EPOCHS \
    --alpha-values   0.0 0.5 1.0 2.0 \
    --device         cuda:0 \
    >> "$LOG_G1" 2>&1 &
_PID_G1=$!

echo "  Waiting for both analysis runs..."
_analysis_rc=0
wait $_PID_G0 || _analysis_rc=$?
wait $_PID_G1 || _analysis_rc=$?
[[ $_analysis_rc -ne 0 ]] && echo "WARNING: analysis job(s) failed (rc=${_analysis_rc})" >&2
echo " Phase 2 complete"

# ===================================================================
# Phase 3: Merge JSONs + regenerate combined figures
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 3: Merge results + combined figures"
echo "============================================================"

python - "$OUT_G0" "$OUT_G1" "$OUT_DIR" "$MAX_LAG" <<'MERGE_PY'
import sys, json, os, shutil
import numpy as np

g0_dir, g1_dir, out_dir, max_lag_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
max_lag = int(max_lag_str)

# --- Merge lagpair_metrics.json ---
merged = {}
for gdir in [g0_dir, g1_dir]:
    jp = os.path.join(gdir, "lagpair_metrics.json")
    if os.path.exists(jp):
        with open(jp) as f:
            merged.update(json.load(f))

out_json = os.path.join(out_dir, "lagpair_metrics.json")
with open(out_json, "w") as f:
    json.dump(merged, f, indent=2, default=str)
print(f"  Merged {len(merged)} entries → {out_json}")

# --- Copy score models ---
sm_dir = os.path.join(out_dir, "score_models")
os.makedirs(sm_dir, exist_ok=True)
for gdir in [g0_dir, g1_dir]:
    gsd = os.path.join(gdir, "score_models")
    if os.path.isdir(gsd):
        for f in os.listdir(gsd):
            shutil.copy2(os.path.join(gsd, f), os.path.join(sm_dir, f))

# --- Copy .npy direction files ---
for gdir in [g0_dir, g1_dir]:
    for f in os.listdir(gdir):
        if f.endswith(".npy"):
            shutil.copy2(os.path.join(gdir, f), os.path.join(out_dir, f))

# --- Regenerate combined figures ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PE_COLORS = {"rope": "#2CA02C", "alibi": "#FF7F0E", "absolute": "#1F77B4"}
PE_LABELS = {"rope": "RoPE", "alibi": "ALiBi", "absolute": "Absolute"}

# Parse available PE types, seeds, layers from merged keys
pe_types = sorted(set(v["pe_type"] for v in merged.values()))
seeds    = sorted(set(v["seed"] for v in merged.values()))
layers   = sorted(set(v["layer"] for v in merged.values()))
all_lags = list(range(max_lag + 1))

fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

def collect_metric(pe_type, layer, metric_key):
    lags_out, means, stds = [], [], []
    for lag in all_lags:
        vals = []
        for seed in seeds:
            tag = f"{pe_type}_s{seed}_L{layer}"
            if tag in merged:
                for k in [lag, str(lag)]:
                    if k in merged[tag]["lags"]:
                        v = merged[tag]["lags"][k].get(metric_key)
                        if v is not None:
                            vals.append(v)
                        break
        if vals:
            lags_out.append(lag)
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
    return lags_out, means, stds

def plot_metric(metric_key, ylabel, title, filename, ylim=None):
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5),
                             sharey=bool(ylim))
    if n_layers == 1:
        axes = [axes]
    for li, layer in enumerate(layers):
        ax = axes[li]
        for pe in pe_types:
            lgs, mn, sd = collect_metric(pe, layer, metric_key)
            if lgs:
                ax.errorbar(lgs, mn, yerr=sd, fmt="o-",
                            color=PE_COLORS.get(pe, "gray"),
                            label=PE_LABELS.get(pe, pe),
                            capsize=3, linewidth=2, markersize=4)
        ax.set_xlabel("Lag $r$")
        ax.set_title(f"L{layer}")
        if li == 0:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2)
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

plot_metric("A_r", "$A_r$", "Lag Amplitude", "LP_A_r_profile.pdf")
plot_metric("S_r", "$S_r$", "Lag Stationarity", "LP_S_r_profile.pdf",
            ylim=(-0.05, 1.05))
plot_metric("C_r", "$C_r$", "Concentration", "LP_C_r_profile.pdf",
            ylim=(-0.05, 1.05))
plot_metric("AS_r", "$A_r S_r$", "Stable Lag Mass", "LP_AS_r_profile.pdf")

# --- AS_r with RoPE overlay ---
def compute_rope_theoretical_profile(head_dim=64, max_lag=14, base=10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (np.arange(half) / half))
    periods = 2 * np.pi / freqs
    lags = np.arange(max_lag + 1)
    profile = np.zeros(max_lag + 1)
    for lag in lags:
        profile[lag] = np.sum(np.cos(freqs * lag))
    return lags, profile, periods

lags_theo, rope_profile, _ = compute_rope_theoretical_profile(
    head_dim=64, max_lag=max_lag)

n_layers = len(layers)
fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5), sharey=False)
if n_layers == 1:
    axes = [axes]
for li, layer in enumerate(layers):
    ax = axes[li]
    for pe in pe_types:
        lgs, mn, sd = collect_metric(pe, layer, "AS_r")
        if lgs and mn[0] > 1e-8:
            norm = mn[0]
            ax.errorbar(lgs, [v / norm for v in mn],
                        yerr=[v / norm for v in sd], fmt="o-",
                        color=PE_COLORS.get(pe, "gray"),
                        label=PE_LABELS.get(pe, pe),
                        capsize=3, linewidth=2, markersize=4)
    rp = rope_profile[1:max_lag + 1]
    if len(rp) > 0 and abs(rp[0]) > 1e-8:
        ax.plot(range(1, len(rp) + 1), rp / rp[0],
                "k--", linewidth=1.5, alpha=0.5, label="RoPE theory")
    ax.set_xlabel("Lag $r$")
    ax.set_title(f"L{layer}")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    if li == 0:
        ax.set_ylabel("$A_r S_r$ (normalized)")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
fig.suptitle("Stable Lag Mass with RoPE Theoretical Envelope", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "LP_AS_r_rope_overlay.pdf"),
            bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: LP_AS_r_rope_overlay.pdf")

# --- Operator autocorrelation ---
fig, axes = plt.subplots(len(pe_types), n_layers,
                         figsize=(4 * n_layers, 3 * len(pe_types)),
                         squeeze=False)
for pi, pe in enumerate(pe_types):
    for li, layer in enumerate(layers):
        ax = axes[pi][li]
        for lag_show in [1, 2, 4]:
            ac_all = []
            for seed in seeds:
                tag = f"{pe}_s{seed}_L{layer}"
                if tag in merged:
                    for k in [lag_show, str(lag_show)]:
                        if k in merged[tag]["lags"]:
                            ac = merged[tag]["lags"][k].get("autocorr", [])
                            if ac:
                                ac_all.append(ac)
                            break
            if ac_all:
                min_len = min(len(a) for a in ac_all)
                ac_arr = np.array([a[:min_len] for a in ac_all])
                ac_mean = ac_arr.mean(0)
                ax.plot(np.arange(len(ac_mean)), ac_mean, linewidth=1.5,
                        label=f"lag {lag_show}", alpha=0.8)
        ax.set_xlabel("Shift $\\delta$")
        ax.set_title(f"{PE_LABELS.get(pe, pe)} L{layer}", fontsize=9)
        if li == 0:
            ax.set_ylabel("$C(r, \\delta)$")
        if pi == 0 and li == 0:
            ax.legend(fontsize=7)
fig.suptitle("Operator Autocorrelation", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "LP_operator_autocorrelation.pdf"),
            bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: LP_operator_autocorrelation.pdf")

print(f"  All figures saved to {fig_dir}/")
MERGE_PY

echo " Phase 3 complete"

# ===================================================================
# Phase 4: Logit analysis across 2 GPUs
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 4: Logit analysis (2 GPUs)"
echo "============================================================"

for pe in "${PE_TYPES[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
        _json="$LOGIT_OUT/${pe}_seed${seed}_logit_analysis.json"
        [ -f "$_json" ] && { echo "  SKIP: $_json"; continue; }

        pool_acquire
        slot=$_FREE_SLOT
        GPU_ID=${GPU_IDS[$slot]}
        LOG="logs/logit_${pe}_s${seed}_${TIMESTAMP}.log"
        echo "  Logit ${pe} seed=${seed} → GPU ${GPU_ID} (slot ${slot})"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/run_logit_analysis.py \
            --models-dir         "$MODELS_DIR"       \
            --out-dir            "$LOGIT_OUT"        \
            --pe-types           "$pe"               \
            --seeds              "$seed"             \
            --w                  "$W"                \
            --max-lag            "$MAX_LAG"          \
            --pca-dim            "$PCA_DIM"          \
            --epochs             "$SCORE_EPOCHS"     \
            --score-tuning-trials "$LOGIT_TUNING_TRIALS" \
            --bootstrap-n        "$LOGIT_BOOTSTRAP_N" \
            --save-score-models                      \
            --device             cuda:0              \
            >> "$LOG" 2>&1 &

        _SLOT_PIDS[$slot]=$!
    done
done

echo "  Waiting for all logit analysis to finish..."
_logit_rc=0
pool_drain || _logit_rc=$?
[[ $_logit_rc -ne 0 ]] && echo "WARNING: some logit analysis jobs failed (rc=${_logit_rc})" >&2
echo " Phase 4 complete"

# ===================================================================
# Phase 5: Logit comparison figures
# ===================================================================
echo ""
echo "============================================================"
echo " Phase 5: Logit comparison figures"
echo "============================================================"

ORIG_SUMMARY=""
for _c in "extended_pos_analysis_"*/analysis_summary.json \
          "results/extended_pos_analysis_"*/analysis_summary.json \
          "transformer_pos_analysis_"*/analysis_summary.json \
          "results/transformer_pos_analysis_"*/analysis_summary.json; do
    [ -f "$_c" ] && { ORIG_SUMMARY="$_c"; break; }
done

if [ -n "$ORIG_SUMMARY" ]; then
    echo "  Using hidden summary: $ORIG_SUMMARY"
    python scripts/compare_hidden_vs_logit.py \
        --hidden-summary "$ORIG_SUMMARY" \
        --logit-dir      "$LOGIT_OUT" \
        --out-dir        "$LOGIT_OUT" \
        --pe-types       "${PE_TYPES[@]}" \
        --seeds          "${ALL_SEEDS[@]}" \
        || echo "WARNING: Hidden vs logit comparison failed" >&2
else
    echo "  SKIP: No analysis_summary.json found"
fi

# ===================================================================
echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
echo ""
echo "  Trained models: $MODELS_DIR"
echo "    $(ls "$MODELS_DIR"/*/model.pt 2>/dev/null | wc -l) transformer checkpoints"
echo ""
echo "  Hidden-state results:"
echo "    $OUT_DIR/lagpair_metrics.json (merged, all 6 seeds)"
echo "    $OUT_DIR/figures/"
for f in "$OUT_DIR"/figures/*.pdf; do
    [ -f "$f" ] && echo "      $(basename "$f")"
done
echo ""
echo "  Logit results:"
echo "    $LOGIT_OUT/"
for f in "$LOGIT_OUT"/*.json; do
    [ -f "$f" ] && echo "      $(basename "$f")"
done
echo ""
echo "  Score models:"
echo "    $OUT_DIR/score_models/ ($(ls "$OUT_DIR"/score_models/*.pt 2>/dev/null | wc -l) hidden)"
echo ""

exit 0
