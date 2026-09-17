#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_p8_tied_gate
#SBATCH --output=%x-%j.out

# Phase-8 tied complementary tap-gate probe (E0-style falsification).
#
# Mechanism: B shared causal rails r_b = tanh(alpha_b * (a_b' * u_taps + c_b))
# from the 8 causal delay-bank taps drive a per-node gate
# z_j = sigmoid(beta_j + sum_b w_jb * r_b), m_j = 2 * (1 - z_j).
# m_j multiplies the ENTIRE KCL accumulator (core + boundary + shunt + drive)
# and the leak; clip stays outside as a safety. Rails/z are computed once per
# sample and held across the 8 Heun substeps (same convention as freeze_read).
# Identity init (w = 0, beta = 0 -> m = 1) is bit-identical to the static
# Phase-8 banded core, so the identity_gate row IS the static reference.
#
# Jobs: preflight audit (fatal) + random-gate existence probe for B = 2, 3, 4.
# The w-scale sweep extends the spec-small defaults (0.02, 0.05, 0.10) with
# 0.3 and 1.0: local CPU smokes showed z-std ~0.0004-0.002 at small scales,
# far below the 0.02 void threshold, so small scales alone cannot falsify.
# Step-4 training legs are NOT included (not yet implemented).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc

ENV_DIR="${ENV_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/ENV}"
source "$ENV_DIR/bin/activate"
PYTHON_BIN="$ENV_DIR/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing $PYTHON_BIN" >&2; exit 3; }
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

OUT="${OUT:-./output/narma_phase8_tied_gate}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-3000}"
WASHOUT="${WASHOUT:-200}"
CORE_SEEDS="${CORE_SEEDS:-0,1,2}"
N_LOCAL_JAC="${N_LOCAL_JAC:-8}"
W_SCALES="${W_SCALES:-0.02,0.05,0.10,0.3,1.0}"
ALPHA_GRID="${ALPHA_GRID:-0.40,0.50,0.60,0.70,0.80,0.90,1.0}"
ACTIVITY_TARGET="${ACTIVITY_TARGET:-0.49}"
RAIL_MAX="${RAIL_MAX:-0.20}"
mkdir -p "$OUT"

# Preflight: compile + bit-identity / gradient audit. Fatal on failure.
"$PYTHON_BIN" -m py_compile cell_library.py differential_stage.py topology.py config.py narma_experiment.py narma_spectral_norm.py narma_advisor_probes.py narma_phase7_leak_bands.py tap_rails.py narma_phase8_tied_gate_probe.py narma_phase8_tied_gates_audit.py
"$PYTHON_BIN" narma_phase8_tied_gates_audit.py | tee "$OUT/tied_gate_audit.log"

# Probe grid: one job per rail count, B = 2 first (spec sequencing:
# if B = 2 is flat and void, B = 4 is unlikely to rescue it).
for B in 2 3 4; do
  BOUT="$OUT/B${B}"
  mkdir -p "$BOUT"
  "$PYTHON_BIN" narma_phase8_tied_gate_probe.py \
    --out "$BOUT" --device "$DEVICE" --samples "$SAMPLES" --washout "$WASHOUT" \
    --core-seeds "$CORE_SEEDS" --B "$B" --w-scales "$W_SCALES" \
    --alpha-grid "$ALPHA_GRID" --activity-target "$ACTIVITY_TARGET" \
    --rail-max "$RAIL_MAX" --n-local-jac "$N_LOCAL_JAC" \
    | tee "$BOUT/tied_gate_probe.log"
done

echo "RESULTS_WRITTEN $OUT (per-B tied_gate_probe.json + tied_gate_audit.log)"
