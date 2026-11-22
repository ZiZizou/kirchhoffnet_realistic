#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=post_twin_gates
#SBATCH --output=%x-%j.out

# Post-twin fatal-risk gates (plan post-twin-gates, feature spec
# flow-sparse-gates). Eval-only, zero-training. Three gates stand between
# the confirmed digital twin and any multistage work; on all-pass the run
# emits post_twin_gates_buildable_spec.json (lambda_T operating point,
# edge-count target, mismatch tolerance, W_in geometry) as the sole input
# to the frozen-multistage experiment.
#
# What runs (locked grids, flow-sparse-gates spec):
#   preflight : audit_post_twin_gates.py (21 checks; FATAL on failure).
#   all       : post_twin_gates.py all (taps8 main 144 corners +
#               scalar confirm 18 corners at the locked confirm grid +
#               Gate-2 sparsity 17 rows + Gate-2 mismatch 37 rows +
#               Gate-3 W_in 18 rows, all at the resolved best corner),
#               then per-gate verdicts + buildable spec or redirect.
#
# Submit:
#   sbatch post_twin_gates_batch.sh
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 post_twin_gates_batch.sh
#   sbatch --export=DEVICE=cpu post_twin_gates_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/post_twin_gates post_twin_gates_batch.sh
#   sbatch --export=TRAIN_SAMPLES=300 post_twin_gates_batch.sh
#   sbatch --export=LEG_TIMEOUT=5400 post_twin_gates_batch.sh
#
# NOTE: *.sh is gitignored in this repo; ship with `git add -f
# post_twin_gates_batch.sh` (plus post_twin_gates.py +
# audit_post_twin_gates.py via normal git add) when the remote needs it.

set -uo pipefail
# No `set -e`: a single failed leg must not kill the rest.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load StdEnv/2023
module load python/3.13
module load cuda/12.2
module load gcc

VENV="${VENV:-/home/annaik/Documents/tron_eq/venv/bin/activate}"
source "$VENV"

python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

sha256sum post_twin_gates.py audit_post_twin_gates.py post_twin_gates_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- post_twin_gates.py audit_post_twin_gates.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-5400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-900}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/post_twin_gates}"

DEVICE="${DEVICE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-25}"
N_TAPS="${N_TAPS:-8}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2500}"
WASHOUT="${WASHOUT:-200}"
MAX_DELAY="${MAX_DELAY:-20}"
JACOBIAN_SAMPLES="${JACOBIAN_SAMPLES:-3}"

FAILED_LEGS=()

run_leg() {
  local tag="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[gates] $tag -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[gates] $tag already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u post_twin_gates.py "$@" --output "$out"; then
    echo "[gates] $tag DONE"
  else
    echo "[gates] $tag FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$tag")
  fi
}

# ---- Remote preflight: full audit must pass before any gate burns allocation.
echo ""
echo "=================================================================="
echo "[gates] preflight: audit_post_twin_gates.py (21 checks)"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u audit_post_twin_gates.py; then
  echo "[gates] preflight DONE"
else
  echo "[gates] preflight FAILED; aborting before the gate legs" >&2
  exit 2
fi

# ---- Gate 1 (taps8 main) + scalar confirm + Gate 2 + Gate 3, one `all` run.
# The run resolves the best taps8 corner, replays Gates 2/3 there, records
# per-gate verdicts, and emits the buildable spec on all-pass.
run_leg "all" \
  "$PROBE_ROOT/all" "post_twin_gates_verdict.json" \
  all \
    --order 10 \
    --device "$DEVICE" \
    --hidden-dim "$HIDDEN_DIM" \
    --n-taps "$N_TAPS" \
    --harness taps8 \
    --n-streams 1 \
    --train-samples "$TRAIN_SAMPLES" \
    --washout "$WASHOUT" \
    --max-delay "$MAX_DELAY" \
    --jacobian-samples "$JACOBIAN_SAMPLES"

echo ""
echo "=================================================================="
echo "[gates] summary"
echo "=================================================================="
python - "$PROBE_ROOT/all" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
verdict_path = out / "post_twin_gates_verdict.json"
print(f"--- dir={out}")
if not verdict_path.exists():
    print("--- post_twin_gates_verdict.json MISSING (leg failed or skipped)")
    sys.exit(0)
v = json.loads(verdict_path.read_text())

g1 = v.get("gate1", {})
print(f"--- gate1 taps8 main : {g1.get('verdict')} "
      f"viable_lamT={g1.get('viable_lamT')} "
      f"surviving={g1.get('surviving_by_lamT')}")
g1c = v.get("gate1_scalar_confirm", {})
print(f"--- gate1 scalar cfm : {g1c.get('verdict')} "
      f"(record-only) surviving={g1c.get('surviving_by_lamT')}")
print(f"--- best_corner      : {v.get('best_corner')}")

g2s = v.get("gate2_sparsity", {})
print(f"--- gate2 sparsity   : {g2s.get('verdict')} "
      f"edge_target={g2s.get('edge_target')} "
      f"buckets={g2s.get('mean_mc_by_edge_count')}")
g2m = v.get("gate2_mismatch", {})
print(f"--- gate2 mismatch   : {g2m.get('verdict')} "
      f"by_sigma={g2m.get('mean_mc_by_sigma')}")
g3 = v.get("gate3", {})
print(f"--- gate3 w_in       : {g3.get('verdict')} "
      f"by_leg={g3.get('mean_mc_by_leg')}")
print()

spec_path = out / "post_twin_gates_buildable_spec.json"
if v.get("buildable_spec_emitted") and spec_path.exists():
    spec = json.loads(spec_path.read_text())
    print("--- BUILDABLE SPEC EMITTED:")
    print(f"    lambda_T={spec['lambda_T']} "
          f"edge_target={spec['edge_target']} "
          f"mismatch_tolerance={spec['mismatch_tolerance']} "
          f"w_in_geometry={spec['w_in_geometry']}")
else:
    blocked = [name for name, g in
               (("gate1", g1), ("gate2_sparsity", g2s),
                ("gate2_mismatch", g2m), ("gate3", g3))
               if g.get("verdict") not in
               ("FLOW_VIABLE", "ANNEAL_PRUNE",
                "SILICON_PLAUSIBLE", "STRUCTURED")]
    print(f"--- NO BUILDABLE SPEC; blocked by: {blocked or ['unknown']}")
print()
print(f"--- artifacts: gate1_flowtwin.csv gate1_scalar_confirm.csv "
      f"gate2_sparsity.csv gate2_mismatch.csv gate3_w_in.csv + *.json "
      f"under {out}")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[gates] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[gates] all legs complete"
