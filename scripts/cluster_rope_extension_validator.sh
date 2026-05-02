#!/bin/bash
#SBATCH --job-name=rope_ext_val
#SBATCH --output=logs/rope_ext_val_%j.out
#SBATCH --error=logs/rope_ext_val_%j.err
#SBATCH -p week
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ===========================================================================
# RoPE context-extension validator (Option B)
# ===========================================================================
#
# Tests whether NTK-aware base scaling extends a (B0, C0=64) trained model
# to context C1=256 *without retraining*, by comparing val loss under
# three regimes (no extension, NTK-aware scaling, PI-style scaling) against
# a from-scratch (nearest-NTK base, C1=256) baseline.
#
# Inputs are the trained models from cluster_rope_base_context_grid.sh:
#   results/rope_base_context_grid/ctx{64,256}/base_{B}/rope_seed{N}/model.pt
#
# CPU-only (no GPU): the validator does inference on 30 small models; even
# without GPU acceleration each (source, target, regime) eval is ~5s.
# Total runtime ≈ 5 minutes; the 2 h walltime is buffer.
#
# Submit:
#   sbatch scripts/cluster_rope_extension_validator.sh
#
# Output:
#   results/rope_base_context_grid/extension_validation/
#     ├── from_b30_c64_to_b100_c256/extension_validation.json
#     ├── from_b100_c64_to_b300_c256/extension_validation.json
#     ├── from_b300_c64_to_b1000_c256/extension_validation.json
#     ├── from_b1000_c64_to_b10000_c256/extension_validation.json
#     └── from_b10000_c64_to_b10000_c256/extension_validation.json   (extrap)
#
# Pairing rationale: for each source base B0 at ctx=64, NTK-aware extension
# to ctx=256 prescribes B1 = B0 × (256/64)^(d/(d-2)) ≈ B0 × 4.16.  We match
# each B1 to the *nearest available trained-from-scratch base in our grid*
# (in log space) so the from-scratch comparison is meaningful.  The last
# pairing (10000 → 10000) is an extrapolation: NTK predicts B1 ≈ 41600
# which we don't have, so the comparison is to the same-base ctx=256 cell.
# ===========================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/sk3373/project_pi_sz25/sk3373/SBTG_v1}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
if   [ -d "env"   ]; then source env/bin/activate
elif [ -d ".venv" ]; then source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m pip install -e ".[all]" --quiet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_ROOT="results/rope_base_context_grid"
OUT_ROOT="$GRID_ROOT/extension_validation"

DATA_C64="$GRID_ROOT/data_seq64"
DATA_C256="$GRID_ROOT/data_seq256"

# Architecture (must match training; matches the grid orchestrator's defaults)
N_LAYER=2
N_EMBD=128
N_HEAD=2
N_INNER=512

SEEDS="0 1 2"
FAMILY="variable_lag_copy"

mkdir -p "$OUT_ROOT" logs

echo "============================================================"
echo " RoPE context-extension validator (Option B, NTK-aware)"
echo " Grid:    $GRID_ROOT"
echo " Output:  $OUT_ROOT"
echo " Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Validation pairings: source (B0, C0=64) → target ctx=256
#                      compare against from-scratch (B1_nearest, ctx=256).
# Format: "B0:B1_nearest"  (B1_nearest matches available trained cells)
# ---------------------------------------------------------------------------
declare -a PAIRINGS=(
    "30:100"        # NTK predicts B1≈125; nearest avail = 100
    "100:300"       # NTK predicts B1≈416; nearest in log space = 300
    "300:1000"      # NTK predicts B1≈1248; nearest = 1000
    "1000:10000"    # NTK predicts B1≈4160; nearest in log space = 10000
    "10000:10000"   # NTK predicts B1≈41600; not in grid — same-base extrap
)

for PAIR in "${PAIRINGS[@]}"; do
    B0="${PAIR%:*}"
    B1_CMP="${PAIR#*:}"

    SRC_DIR="$GRID_ROOT/ctx64/base_${B0}"
    CMP_DIR="$GRID_ROOT/ctx256/base_${B1_CMP}"
    OUT_DIR="$OUT_ROOT/from_b${B0}_c64_to_b${B1_CMP}_c256"

    if [ -f "$OUT_DIR/extension_validation.json" ]; then
        echo "  [skip] $OUT_DIR/extension_validation.json exists"
        continue
    fi

    if [ ! -d "$SRC_DIR" ]; then
        echo "  WARN: source dir $SRC_DIR not found; skipping pairing $PAIR" >&2
        continue
    fi
    if [ ! -d "$CMP_DIR" ]; then
        echo "  WARN: comparison dir $CMP_DIR not found; running without --comparison-models-dir" >&2
        CMP_FLAGS=""
    else
        CMP_FLAGS="--comparison-models-dir $CMP_DIR"
    fi

    mkdir -p "$OUT_DIR"
    echo ""
    echo "  --- source: (base=${B0}, ctx=64) → target ctx=256 ---"
    echo "      comparison from-scratch: (base=${B1_CMP}, ctx=256)"
    python scripts/run_rope_extension_validator.py \
        --source-models-dir   "$SRC_DIR" \
        --source-data-dir     "$DATA_C64" \
        --target-data-dir     "$DATA_C256" \
        --source-base         "$B0" \
        --seeds               $SEEDS \
        --family              "$FAMILY" \
        --split               test \
        --n-layer             "$N_LAYER" \
        --n-embd              "$N_EMBD" \
        --n-head              "$N_HEAD" \
        --n-inner             "$N_INNER" \
        --device              cpu \
        $CMP_FLAGS \
        --out-dir             "$OUT_DIR"
done

# ---------------------------------------------------------------------------
# Aggregate: print a one-line summary per pairing
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Extension validation summary"
echo "============================================================"
python -u - <<'PYEOF'
import json
from pathlib import Path

root = Path("results/rope_base_context_grid/extension_validation")
print(f"  {'pairing':<32}  {'L_no':>8}  {'L_NTK':>8}  {'L_PI':>8}  {'L_scratch':>10}  verdict")
print("-" * 90)
for p in sorted(root.glob("from_*/extension_validation.json")):
    d = json.loads(p.read_text())
    name = p.parent.name
    L_no  = d.get("L_no_extension",  {}).get("mean")
    L_ntk = d.get("L_ntk_extended",  {}).get("mean")
    L_pi  = d.get("L_pi_extended",   {}).get("mean")
    L_fs  = (d.get("L_from_scratch") or {}).get("mean")
    def f(v): return "—" if v is None else f"{v:.4f}"
    verdict = ""
    if L_ntk is not None and L_fs is not None:
        gap = L_ntk - L_fs
        rel = abs(gap) / max(L_fs, 1e-6)
        verdict = "CLEAN" if rel < 0.10 else f"gap {rel*100:.0f}%"
    print(f"  {name:<32}  {f(L_no):>8}  {f(L_ntk):>8}  {f(L_pi):>8}  {f(L_fs):>10}  {verdict}")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
