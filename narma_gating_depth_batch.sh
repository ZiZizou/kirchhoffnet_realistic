#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=narma_depth
#SBATCH --output=%x-%j.out

# NARMA-10 Phase-4 depth screen (knet-gated-memory plan).
#
# 1/2/3-stage comparison from the best operating point. Each sample's
# t_span window is SPLIT across stages (times even, steps proportional);
# fresh drive enters stage 1 only; readout reads the last stage;
# residual-relu-tanh transfers (identity at init).
# Local 2-stage CPU smoke trains end-to-end (loss finite, grads flow);
# knet_multistage_audit.py ALL GREEN (split/identity/drive/equivalence).
#
# Question: does depth increase nonlinear mixing (MC/NRMSE vs 1-stage)?
#
# Submit with:
#   sbatch narma_gating_depth_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_gating_depth_batch.sh
#   sbatch --export=DEPTH_EPOCHS=80 narma_gating_depth_batch.sh
#   sbatch --export=DEPTH_SEEDS="0 1" narma_gating_depth_batch.sh
#   sbatch --export=DEPTH_GAIN=-2 narma_gating_depth_batch.sh
#   sbatch --export=DEPTH_ROOT=./outputs/gating_depth narma_gating_depth_batch.sh

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
sha256sum narma_experiment.py differential_stage.py topology.py config.py kirchhoff_net.py knet_multistage_audit.py narma_gating_depth_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_experiment.py differential_stage.py topology.py config.py kirchhoff_net.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-5400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
DEPTH_ROOT="${DEPTH_ROOT:-./outputs/gating_depth}"
DEPTH_EPOCHS="${DEPTH_EPOCHS:-80}"
DEPTH_SEEDS="${DEPTH_SEEDS:-0}"
# Best operating point from Phase-1 (defaults: canonical init; override
# with the winning gm/leak once the 48-corner grid lands).
DEPTH_GAIN="${DEPTH_GAIN:-}"
DEPTH_LEAK="${DEPTH_LEAK:-}"

FAILED_LEGS=()

run_depth_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[depth] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[depth] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_experiment.py "$@" --output "$out"; then
    echo "[depth] leg $name DONE"
  else
    echo "[depth] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# Shared geometry: canonical NARMA-10 protocol, tanh_free, k2 ladder.
# NOTE: --flag=value form throughout (Alliance batch convention).
BASE_ARGS="--order=10 --device=cuda --epochs=$DEPTH_EPOCHS --cell-library=tanh_free --t-span=1.0 --num-steps=8 --core-refresh-interval=2 --no-progress --checkpoint-every=10 --interstage-activation=residual-relu-tanh"
if [[ -n "$DEPTH_GAIN" ]]; then
  BASE_ARGS="$BASE_ARGS --gain-init=$DEPTH_GAIN"
fi
if [[ -n "$DEPTH_LEAK" ]]; then
  BASE_ARGS="$BASE_ARGS --leak-init=$DEPTH_LEAK"
fi

# ---- Remote preflight: 1-epoch 2-stage leg must build+train first.
echo ""
echo "=================================================================="
echo "[depth] preflight: 1-epoch 2-stage leg"
echo "=================================================================="
# shellcheck disable=SC2086
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_experiment.py $BASE_ARGS \
  --epochs=1 --train-samples=300 --n-streams=1 \
  --seed=0 --fabric-only --num-stages=2 \
  --output "$DEPTH_ROOT/preflight"; then
  echo "[depth] preflight DONE"
else
  echo "[depth] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- 1/2/3-stage comparison per seed ----
for SEED in $DEPTH_SEEDS; do
  # shellcheck disable=SC2086
  run_depth_leg "s1-seed$SEED" \
    "$DEPTH_ROOT/s1_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only --num-stages=1
  # shellcheck disable=SC2086
  run_depth_leg "s2-seed$SEED" \
    "$DEPTH_ROOT/s2_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only --num-stages=2
  # shellcheck disable=SC2086
  run_depth_leg "s3-seed$SEED" \
    "$DEPTH_ROOT/s3_seed$SEED" "results_table.csv" \
    $BASE_ARGS --seed="$SEED" --fabric-only --num-stages=3
done

echo ""
echo "=================================================================="
echo "[depth] leg summary"
echo "=================================================================="
python - "$DEPTH_ROOT" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "s?_seed*", "results_table.csv"))):
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
            f"mc={r.get('mc_total')} params={r.get('total_params')} "
            f"stages={r.get('num_stages')}"
        )
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[depth] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[depth] all depth legs complete"
echo "Deliverable: results table comparing 1/2/3 stages (MC + NRMSE)."
echo "If depth helps without saturation, keep it; else the fabric answer"
echo "stays single-stage and effort returns to init/gating levers."
