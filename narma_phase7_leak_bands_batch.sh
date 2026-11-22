#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_p8_activity_leaks
#SBATCH --output=%x-%j.out

# Phase 8: activity-restored paired three-band leak experiment.  The directed 5x5 torus,
# 200 delay-bank OTAs (including their isat_raw=-4.4 pattern), integration,
# activation, and readout are fixed.  Only the 25 programmable core leaks
# differ between baseline, the banked alpha=1 reference, one scalar-scaled
# banded core, and an independently activity-matched uniform control.
# In particular, do not re-run spectral normalization after replacing leaks.
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

OUT="${OUT:-./output/narma_phase8_activity_leaks}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-3000}"
WASHOUT="${WASHOUT:-200}"
CORE_SEEDS="${CORE_SEEDS:-0,1,2}"
N_LOCAL_JAC="${N_LOCAL_JAC:-8}"
BAND_ALPHA_GRID="${BAND_ALPHA_GRID:-0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0}"
UNIFORM_LEAK_GRID="${UNIFORM_LEAK_GRID:-0.01,0.02,0.03,0.04,0.051,0.08,0.12,0.18,0.223,0.32,0.42,0.511,0.65}"
ACTIVITY_TARGET="${ACTIVITY_TARGET:-0.49}"
ACTIVITY_TOLERANCE="${ACTIVITY_TOLERANCE:-0.01}"
mkdir -p "$OUT"

"$PYTHON_BIN" -m py_compile cell_library.py differential_stage.py topology.py config.py narma_experiment.py narma_spectral_norm.py narma_advisor_probes.py narma_phase7_leak_bands.py
"$PYTHON_BIN" narma_phase7_leak_bands.py \
  --out "$OUT" --device "$DEVICE" --samples "$SAMPLES" --washout "$WASHOUT" \
  --core-seeds "$CORE_SEEDS" --n-local-jac "$N_LOCAL_JAC" \
  --band-alpha-grid "$BAND_ALPHA_GRID" --uniform-leak-grid "$UNIFORM_LEAK_GRID" \
  --activity-target "$ACTIVITY_TARGET" --activity-tolerance "$ACTIVITY_TOLERANCE" | tee "$OUT/leak_bands.log"
