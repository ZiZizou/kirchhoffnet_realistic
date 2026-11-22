#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=narma_gating_e0x
#SBATCH --output=%x-%j.out

# NARMA-10 Phase-1 EXTENSION grid (follow-up to narma_gating_e0-59691417).
#
# Prior run (48 corners, gm -4..-1 x isat -10..-5, leak 0.1): 0/48 pass,
# rails 0.0% everywhere, best corner on the grid EDGE (gm=-1, isat=-5:
# ridge 0.6347, MC 2.26, PR 1.49). Gain was monotonic (gm -4->-1 and
# isat -10->-5 improve ridge+MC+PR together); drive was second-order.
#
# This leg pushes past the edge at zero rail risk:
#   6 pairs: gm {-1,0,1} x isat {-5,-3} (positional zip) x drive {0.25,0.5,1.0}
#   = 18 corners, fixed leak 0.1 (numeric token).
# The (-1,-5) pair re-runs the prior best as a bridge anchor.
#
# Zero-training, eval-only. Same instruments: ridge-from-states, washout MC,
# Jacobian, rail fraction, gate-zero add-on.
# Pass gate (code): MC>3.0, ridge<=0.40, state-PR>=6.0, rail<=5%.
#
# Submit with:
#   sbatch narma_gating_e0_extend_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_gating_e0_extend_batch.sh
#   sbatch --export=LEG_TIMEOUT=7200 narma_gating_e0_extend_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/gating_e0_extend narma_gating_e0_extend_batch.sh
#
# NOTE: *.sh is gitignored -- ship with `git add -f` like the other batches.

set -uo pipefail
# NOTE: no `set -e` on purpose -- one failed leg must not kill the rest.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load StdEnv/2023
module load python/3.13
module load cuda/12.2
module load gcc

source "/home/annaik/Documents/tron_eq/venv/bin/activate"

# Fail fast if the GPU Python environment is incomplete.
python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

# Record the exact probe code used for this run.
sha256sum narma_advisor_probes.py probe_audit1.py narma_gating_e0_extend_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_advisor_probes.py probe_audit1.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-5400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/gating_e0_extend}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[gating-e0x] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[gating-e0x] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_advisor_probes.py "$@" --output "$out"; then
    echo "[gating-e0x] leg $name DONE"
  else
    echo "[gating-e0x] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Extension axes (POSITIONAL gm x isat pairing -- lengths must match).
# 6 pairs: (-1,-5) anchor + (-1,-3), (0,-5), (0,-3), (1,-5), (1,-3).
GM_GRID="-1,-1,0,0,1,1"
ISAT_GRID="-5,-3,-5,-3,-5,-3"

# ---- Remote preflight: 2 corners must parse and run before the full grid.
echo ""
echo "=================================================================="
echo "[gating-e0x] preflight: 2 extension corners"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_advisor_probes.py e0 \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 300 --jacobian-samples 1 \
  --t-span 1.0 --num-steps 8 --hidden-dim 25 \
  --gain-grid="$GM_GRID" --isat-grid="$ISAT_GRID" \
  --leak-grid="0.1" --drive-grid="0.25,0.5,1.0" \
  --max-corners 2 --output "$PROBE_ROOT/e0_preflight"; then
  echo "[gating-e0x] preflight DONE"
else
  echo "[gating-e0x] preflight FAILED; aborting before the full grid" >&2
  exit 2
fi

# ---- Full 18-corner high-gain extension grid ----
run_probe_leg "phase1-op-point-extend" \
  "$PROBE_ROOT/e0_op_point_extend" "e0_sweep.json" \
  e0 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 4 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 25 \
    --gain-grid="$GM_GRID" \
    --isat-grid="$ISAT_GRID" \
    --leak-grid="0.1" \
    --drive-grid="0.25,0.5,1.0"

echo ""
echo "=================================================================="
echo "[gating-e0x] sweep summary"
echo "=================================================================="
python - "$PROBE_ROOT/e0_op_point_extend/e0_sweep.json" <<'PY'
import json
import sys

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

path = sys.argv[1]
data = load(path)
if data is not None:
    rows = data.get("rows", [])
    normal = [r for r in rows if r.get("pass_all")]
    print(f"--- {path}")
    print(f"corners={len(rows)} pass_all={len(normal)}")
    for row in sorted(rows, key=lambda r: r["ridge_nrmse"])[:10]:
        print(
            f"gm={row['gm_init']} isat={row.get('isat_init')} "
            f"leak={row['leak_mode']} drive={row['drive']} "
            f"ridge={row['ridge_nrmse']:.4f} mc={row['mc_total_washout_corrected']:.2f} "
            f"state_pr={row['state_pr']:.2f} rail={100*row['rail_frac']:.1f}% "
            f"pass={row.get('pass_all')}"
        )
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[gating-e0x] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[gating-e0x] Phase-1 extension grid complete"
