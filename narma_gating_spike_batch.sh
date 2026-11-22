#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_gating_spike
#SBATCH --output=%x-%j.out

# NARMA-10 spike train test (knet-gated-memory plan, Step 3).
#
# 20-epoch legs from the best-known operating point:
#   A = baseline (no spike flags)
#   B = --dynamic-leak only
#   C = --vca-core --vca-use-hidden only
#   D = both spike flags
# Go/no-go on MC/NRMSE vs baseline decides whether Phase-1/2 grids proceed.
#
# Spike epoch-0 contract (audited locally, knet_gating_audit1.py ALL GREEN):
# dynamic-leak and hidden-VCA forwards are bit-identical to legacy at init.
#
# Submit with:
#   sbatch narma_gating_spike_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_gating_spike_batch.sh
#   sbatch --export=SPIKE_EPOCHS=20 narma_gating_spike_batch.sh
#   sbatch --export=SPIKE_SEEDS="0 1" narma_gating_spike_batch.sh
#   sbatch --export=SPIKE_ROOT=./outputs/gating_spike narma_gating_spike_batch.sh

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
sha256sum narma_experiment.py differential_stage.py topology.py config.py kirchhoff_net.py knet_gating_audit1.py narma_gating_spike_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_experiment.py differential_stage.py topology.py config.py kirchhoff_net.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
SPIKE_ROOT="${SPIKE_ROOT:-./outputs/gating_spike}"
SPIKE_EPOCHS="${SPIKE_EPOCHS:-20}"
SPIKE_SEEDS="${SPIKE_SEEDS:-0}"

FAILED_LEGS=()

run_train_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[gating-spike] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[gating-spike] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_experiment.py "$@" --output "$out"; then
    echo "[gating-spike] leg $name DONE"
  else
    echo "[gating-spike] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# Shared train geometry: canonical NARMA-10 protocol, tanh_free, k2 ladder.
# NOTE: --flag=value form throughout (Alliance batch convention).
BASE_ARGS="--order=10 --device=cuda --epochs=$SPIKE_EPOCHS --cell-library=tanh_free --t-span=1.0 --num-steps=8 --core-refresh-interval=2 --no-progress --checkpoint-every=5"

# ---- Remote preflight: 1-epoch spike build+train must work before the full legs.
echo ""
echo "=================================================================="
echo "[gating-spike] preflight: 1-epoch spike leg"
echo "=================================================================="
# shellcheck disable=SC2086
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_experiment.py $BASE_ARGS \
  --epochs=1 --train-samples=300 --n-streams=1 \
  --seed=0 --fabric-only \
  --vca-boundary --vca-core --vca-use-hidden --dynamic-leak \
  --output "$SPIKE_ROOT/preflight"; then
  echo "[gating-spike] preflight DONE"
else
  echo "[gating-spike] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- A/B/C/D legs per seed ----
for SEED in $SPIKE_SEEDS; do
  # shellcheck disable=SC2086
  run_train_leg "A-baseline-seed$SEED" \
    "$SPIKE_ROOT/legA_baseline_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only
  # shellcheck disable=SC2086
  run_train_leg "B-dynleak-seed$SEED" \
    "$SPIKE_ROOT/legB_dynleak_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only \
    --vca-boundary --dynamic-leak
  # shellcheck disable=SC2086
  run_train_leg "C-vcaH-seed$SEED" \
    "$SPIKE_ROOT/legC_vcaH_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only \
    --vca-boundary --vca-core --vca-use-hidden
  # shellcheck disable=SC2086
  run_train_leg "D-both-seed$SEED" \
    "$SPIKE_ROOT/legD_both_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only \
    --vca-boundary --vca-core --vca-use-hidden --dynamic-leak
done

echo ""
echo "=================================================================="
echo "[gating-spike] leg summary"
echo "=================================================================="
python - "$SPIKE_ROOT" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "leg*", "results_table.csv"))):
    leg = os.path.basename(os.path.dirname(path))
    try:
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        print(f"--- {leg} (missing)")
        continue
    fab = [r for r in rows if r.get("condition", "").startswith("fabric_")]
    print(f"--- {leg}")
    for r in fab:
        print(
            f"  {r.get('condition')} seed={r.get('seed')} "
            f"nrmse={r.get('nrmse')} r2={r.get('r2')} "
            f"mc={r.get('mc_total')} params={r.get('total_params')}"
        )
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[gating-spike] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[gating-spike] all spike legs complete"
echo "Go/no-go: compare B/C/D MC+NMRSE vs A. If no leg moves MC above ~1"
echo "or ridge below ~0.6, proceed to Phase-1 (narma_gating_e0_batch.sh)."
