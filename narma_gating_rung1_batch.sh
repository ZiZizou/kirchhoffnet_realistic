#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_rung1
#SBATCH --output=%x-%j.out

# NARMA-10 Phase-2 rung-1 full confirmation (knet-gated-memory plan).
#
# Rung-1: tanh-core native-linear control. Small-signal OTA fills
# (tanh linear regime, shunt killed), ESN-mapped uniform leak,
# drive matched to the ESN winner's input scaling, in-harness ESN
# parity reference on the identical stream.
# Question: does the Heun/KCL/windowing pipeline hold linear memory
# (MC ~14)? Local h25 CPU smoke says NO (MC 0.01, J max|lambda| 0.0046)
# -- this batch certifies that verdict at full stream length, plus
# diagnostic variants (finer/coarser integration, hotter gm fills).
#
# Submit with:
#   sbatch narma_gating_rung1_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_gating_rung1_batch.sh
#   sbatch --export=RUNG1_SEEDS="0 1" narma_gating_rung1_batch.sh
#   sbatch --export=ESN_LEAK=0.5 narma_gating_rung1_batch.sh
#   sbatch --export=ESN_SCALE=0.5 narma_gating_rung1_batch.sh
#   sbatch --export=RUNG1_ROOT=./outputs/gating_rung1 narma_gating_rung1_batch.sh

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

# Record the exact code used for this run.
sha256sum narma_linear_controls.py narma_experiment.py probe_audit2.py narma_gating_rung1_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_linear_controls.py narma_experiment.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-1800}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
RUNG1_ROOT="${RUNG1_ROOT:-./outputs/gating_rung1}"
RUNG1_SEEDS="${RUNG1_SEEDS:-0}"
# ESN-winner mapping (defaults: commonly observed winner region; override
# per seed from the in-harness run_baselines grid when known).
ESN_LEAK="${ESN_LEAK:-1.0}"
ESN_SCALE="${ESN_SCALE:-0.2}"

FAILED_LEGS=()

run_rung1_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[rung1] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[rung1] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_linear_controls.py "$@" --output "$out"; then
    echo "[rung1] leg $name DONE"
  else
    echo "[rung1] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Remote preflight: short-stream rung1 must parse and run first.
echo ""
echo "=================================================================="
echo "[rung1] preflight: 300-sample rung1"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_linear_controls.py rung1 \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 300 --hidden-dim 25 \
  --esn-leak="$ESN_LEAK" --esn-input-scaling="$ESN_SCALE" \
  --output "$RUNG1_ROOT/preflight"; then
  echo "[rung1] preflight DONE"
else
  echo "[rung1] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- Full-length confirmation + diagnostic variants, per seed ----
for SEED in $RUNG1_SEEDS; do
  # Canonical: full 2500-sample stream, t_span=1.0, 8 steps.
  run_rung1_leg "canonical-seed$SEED" \
    "$RUNG1_ROOT/canonical_seed$SEED" "rung1.json" \
    rung1 \
      --order 10 --seed "$SEED" --device cuda \
      --n-streams 1 --train-samples 2500 --hidden-dim 25 \
      --t-span 1.0 --num-steps 8 \
      --esn-leak="$ESN_LEAK" --esn-input-scaling="$ESN_SCALE"
  # Finer integration: 16 Heun steps (tests step-size destruction).
  run_rung1_leg "steps16-seed$SEED" \
    "$RUNG1_ROOT/steps16_seed$SEED" "rung1.json" \
    rung1 \
      --order 10 --seed "$SEED" --device cuda \
      --n-streams 1 --train-samples 2500 --hidden-dim 25 \
      --t-span 1.0 --num-steps 16 \
      --esn-leak="$ESN_LEAK" --esn-input-scaling="$ESN_SCALE"
  # Hotter core: gm_raw -6 (tests weight-scale starvation).
  run_rung1_leg "gm6-seed$SEED" \
    "$RUNG1_ROOT/gm6_seed$SEED" "rung1.json" \
    rung1 \
      --order 10 --seed "$SEED" --device cuda \
      --n-streams 1 --train-samples 2500 --hidden-dim 25 \
      --t-span 1.0 --num-steps 8 \
      --esn-leak="$ESN_LEAK" --esn-input-scaling="$ESN_SCALE" \
      --gm-raw-fill -6.0 --isat-raw-fill -2.0
done

echo ""
echo "=================================================================="
echo "[rung1] leg summary"
echo "=================================================================="
python - "$RUNG1_ROOT" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "*", "rung1.json"))):
    leg = os.path.basename(os.path.dirname(path))
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"--- {leg} (missing)")
        continue
    inst = data.get("instrument", {})
    print(f"--- {leg}")
    print(
        f"  ridge_nrmse={inst.get('nrmse')} r2={inst.get('r2')} "
        f"mc={inst.get('mc_total')} state_pr={inst.get('state_pr')} "
        f"jac_max={inst.get('jac_max_abs')} rail={inst.get('rail_frac')}"
    )
    print(
        f"  esn_ref: nrmse={data.get('esn_nrmse')} mc={data.get('esn_mc_total')} "
        f"pass_all={data.get('pass_all')}"
    )
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[rung1] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[rung1] all rung-1 legs complete"
echo "Verdict rule: canonical MC >= 10 with ridge <= 0.45 -> pipeline holds"
echo "linear memory (investigate trainability next). Else the integrator /"
echo "windowing semantics are the bottleneck -- see steps16/gm6 variants"
echo "for whether it is step size or weight scale."
