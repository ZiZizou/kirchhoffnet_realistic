#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_advisor_probes
#SBATCH --output=%x-%j.out

# NARMA-10 advisor-ordered zero-training probe batch.
#
# This is the Alliance/GPU counterpart to the local CPU harness checks in
# `narma_advisor_probes.py` and `probe_audit1.py`. It runs the three
# pre-registered probe legs in one Slurm allocation:
#
#   P0  In-harness baselines calibration (ridge, ESN, MLP, LSTM variants).
#   P1  Frozen-state probes on the existing trained k2 and k8 checkpoints.
#   P2  Full locked 36-point E0 gain/leak/drive sweep.
#
# Expected decisions:
#   P0: ESN in [0.15, 0.29] -> band calibrated; ESN >= 0.60 -> task-setup halt.
#   P1: ridge < 0.45 promotes the dense-readout fix; ridge >= 0.60 leaves
#       init-regime as the sole primary.
#   P2: a corner with MC > 3 and ridge < 0.5 warrants init search plus one
#       confirmation train; otherwise a flat sweep escalates to structural talk.
#
# Submit with:
#   sbatch narma_advisor_probes_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_advisor_probes_batch.sh
#   sbatch --export=K2_CKPT=/path/to/k2.pt,K8_CKPT=/path/to/k8.pt ...
#   sbatch --export=LEG_TIMEOUT=3600 ...

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

# Fail fast if the GPU Python environment is incomplete. A missing triton
# installation can otherwise cause silent eager fallback and burn GPU hours.
python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

# Re-run completed legs unless SKIP_DONE=0 (for example after code changes).
SKIP_DONE="${SKIP_DONE:-1}"
# Per-leg wall-clock cap in seconds. The E0 sweep is eval-only, but 36 fresh
# nets plus 26-dimensional Jacobians deserve generous headroom.
LEG_TIMEOUT="${LEG_TIMEOUT:-2400}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/advisor_probes}"
K2_CKPT="${K2_CKPT:-./outputs/narma_refresh2/fabric_refresh_k2_seed0.pt}"
K8_CKPT="${K8_CKPT:-./outputs/narma_diag_e3_k8/fabric_refresh_k8_seed0.pt}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[advisor-probes] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[advisor-probes] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  # -u streams progress live; `timeout` bounds a runaway leg. rc=124 means
  # the per-leg cap was hit.
  if timeout "$LEG_TIMEOUT" python -u narma_advisor_probes.py "$@" --output "$out"; then
    echo "[advisor-probes] leg $name DONE"
  else
    echo "[advisor-probes] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- P0: in-harness baselines calibration ----
run_probe_leg "P0-baselines" "$PROBE_ROOT/baselines" "baselines.json" \
  baselines \
    --order 10 \
    --seed 0 \
    --device cuda

# ---- P1: frozen-state probes on trained checkpoints ----
# Architecture flags must match each training run. Both canonical k2/k8
# runs used tanh_free, t_span=1.0, and num_steps=8.
if [[ -f "$K2_CKPT" ]]; then
  run_probe_leg "P1-frozen-k2" "$PROBE_ROOT/frozen_k2" "frozen_state.json" \
    frozen \
      --order 10 \
      --seed 0 \
      --device cuda \
      --ckpt "$K2_CKPT" \
      --cell-library tanh_free \
      --core-refresh-interval 2 \
      --t-span 1.0 \
      --num-steps 8 \
      --input-scale 1.0 \
      --n-streams 4 \
      --train-samples 2500 \
      --mlp-hidden 32 \
      --mlp-epochs 300
else
  echo "[advisor-probes] P1-frozen-k2 SKIPPED: checkpoint not found: $K2_CKPT"
  FAILED_LEGS+=("P1-frozen-k2 (missing ckpt)")
fi

if [[ -f "$K8_CKPT" ]]; then
  run_probe_leg "P1-frozen-k8" "$PROBE_ROOT/frozen_k8" "frozen_state.json" \
    frozen \
      --order 10 \
      --seed 0 \
      --device cuda \
      --ckpt "$K8_CKPT" \
      --cell-library tanh_free \
      --core-refresh-interval 8 \
      --t-span 1.0 \
      --num-steps 8 \
      --input-scale 1.0 \
      --n-streams 4 \
      --train-samples 2500 \
      --mlp-hidden 32 \
      --mlp-epochs 300
else
  echo "[advisor-probes] P1-frozen-k8 SKIPPED: checkpoint not found: $K8_CKPT"
  FAILED_LEGS+=("P1-frozen-k8 (missing ckpt)")
fi

# ---- P2: full locked 36-point E0 sweep ----
# Explicitly pass the locked gain/leak/drive grids even though they are also
# the module defaults, so the submitted protocol is self-documenting.
run_probe_leg "P2-e0-full" "$PROBE_ROOT/e0_full" "e0_sweep.json" \
  e0 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 4 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --gain-grid="-5,-2,0,1.5" \
    --leak-grid="slow-fixed,randomized,0.15" \
    --drive-grid="0.25,0.5,1.0"

echo ""
echo "=================================================================="
echo "[advisor-probes] probe-file summary"
echo "=================================================================="
for probe_file in \
  "$PROBE_ROOT/baselines/baselines.json" \
  "$PROBE_ROOT/frozen_k2/frozen_state.json" \
  "$PROBE_ROOT/frozen_k8/frozen_state.json" \
  "$PROBE_ROOT/e0_full/e0_sweep.json"; do
  if [[ -f "$probe_file" ]]; then
    echo "--- $probe_file"
    python -c "import json,sys; data=json.load(open(sys.argv[1])); print(json.dumps({k: data.get(k) for k in ('decision','esn_nrmse','row','n_corners','elapsed_s') if k in data}, indent=2))" "$probe_file"
  else
    echo "--- $probe_file (missing)"
  fi
done
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[advisor-probes] FAILED/SKIPPED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[advisor-probes] all legs complete"
