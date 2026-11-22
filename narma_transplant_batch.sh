#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_transplant
#SBATCH --output=%x-%j.out

# NARMA-10 regime-transplant full grid (plan narma-regime-transplant).
#
# Pre-registered trigger: canary leg read CORE-USEFUL-NO-PASS
# (anchor ridge 0.6398 in [0.60, 0.68] -> instrument calibrated;
# priority corner ridge 0.7611 / MC 1.68 with a gate-zero flip
# gz 0.8122 -> core load-bearing without a gate pass).
# Per the outcome table: full grid with core-Isat as lead axis.
#
# Grid (spec regime-transplant-grid): 2 gm x 3 Isat x 3 raw-leak x 3 swing
# = 54 corners, ~1.2 GPU-hr. Isat=None entries are coupled (isat == gm);
# raw:<val> leaks are programmable fills (effective softplus(val)).
# This leg remains zero-training and eval-only.
#
# NOTE: *.sh is gitignored in this repo; ship with `git add -f` like the
# other narma_*_batch.sh scripts.
#
# Submit with:
#   sbatch narma_transplant_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_transplant_batch.sh
#   sbatch --export=LEG_TIMEOUT=7200 narma_transplant_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/transplant narma_transplant_batch.sh

set -uo pipefail
# NOTE: no `set -e` on purpose -- one failed leg must not kill the rest.
# Failures are collected in FAILED_LEGS and reported at the end.

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
sha256sum narma_advisor_probes.py probe_audit_transplant.py narma_transplant_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_advisor_probes.py probe_audit_transplant.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-5400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-900}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/transplant}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[transplant] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[transplant] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  # -u streams progress live; `timeout` bounds a runaway leg. rc=124 means
  # the per-leg cap was hit.
  if timeout "$LEG_TIMEOUT" python -u narma_advisor_probes.py "$@" --output "$out"; then
    echo "[transplant] leg $name DONE"
  else
    echo "[transplant] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Preflight (fatal): audits + one transplant corner must pass on CUDA.
echo ""
echo "=================================================================="
echo "[transplant] preflight: transplant audit + one corner"
echo "=================================================================="
python -B -m py_compile narma_advisor_probes.py probe_audit_transplant.py \
  || { echo "ERROR: preflight compile failed." >&2; exit 3; }
python -B probe_audit_transplant.py \
  || { echo "ERROR: transplant audit failed. Aborting before any GPU work." >&2; exit 3; }
if ! timeout "$PREFLIGHT_TIMEOUT" python -u narma_advisor_probes.py transplant \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 300 --jacobian-samples 1 \
  --t-span 1.0 --num-steps 8 --hidden-dim 25 \
  --max-corners 1 \
  --output "$PROBE_ROOT/preflight"; then
  echo "ERROR: preflight corner failed. Aborting before the full grid." >&2
  exit 3
fi

# ---- Full 54-corner transplant grid (per-corner flush + resume).
run_probe_leg "transplant-full" "$PROBE_ROOT/full" "transplant_sweep.json" \
  transplant \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 4 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 25

echo ""
echo "=================================================================="
echo "[transplant] sweep summary"
echo "=================================================================="
if [[ -f "$PROBE_ROOT/full/transplant_sweep.txt" ]]; then
  cat "$PROBE_ROOT/full/transplant_sweep.txt"
else
  echo "--- $PROBE_ROOT/full/transplant_sweep.txt (missing)"
  FAILED_LEGS+=("transplant-full-summary")
fi

echo ""
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[transplant] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[transplant] all legs complete."
