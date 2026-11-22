#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=24:00:00             # Time limit (30 trials x 300 ep; requeue/resume safe: kn_bayes_opt auto-resumes the study.db)
#SBATCH --output=%x-%j.out          # Output file name

# Hardware-constrained Phase-A MLP-guided BO (30 trials).  One study dir per
# HW_PRESET; rerun or chain:
#   j1=$(sbatch --parsable scripts/slurm/retrain_hw_bo.sh)
#   j2=$(sbatch --parsable --dependency=afterany:$j1 scripts/slurm/retrain_hw_bo.sh)
#   sbatch --dependency=afterany:$j2 scripts/slurm/retrain_hw_bo.sh
#
# The failure rule is the surgery's single constant DEGRADE_REL_THRESHOLD = 0.10
# ("fail if >= 2 of the 4 specs degrade by >= 10 %"); no threshold flag is
# passed.  Set DEGRADE_THR in the environment to run an experiment at another
# threshold (--zig-degrade-thr, added by workstream W4).
#
# There are NO --kn-moe-* flags: the surgery removed them from the argparse.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --------------------------------------------------------------------------
# 1. cluster paths (override everything through the environment)
# --------------------------------------------------------------------------
ENV_DIR="${ENV_DIR:-${SLURM_SUBMIT_DIR}/ENV}"          # virtualenv on the cluster
TEACHER_CKPT="${TEACHER_CKPT:-${SLURM_SUBMIT_DIR}/assets/teachers/trial_0019/dagger_student_plain.pt}"
CANONICAL="${CANONICAL:-${SLURM_SUBMIT_DIR}/outputs/canonical_ctle_phase_a_v2.npz}"
ZIG_DIR="${ZIG_DIR:-/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2}"
DATA_DIR="${DATA_DIR:-/home/annaik/Documents/augmented-cvae-ctle}"

# --------------------------------------------------------------------------
# 2. run knobs
# --------------------------------------------------------------------------
HW_PRESET="${HW_PRESET:-phase3a_full}"                # none | phase3a_core | phase3a_full | phase3b_tm
HW_CONFIG="${HW_CONFIG:-}"                             # optional JSON overriding preset keys
HW_EXTRA="${HW_EXTRA:-}"                               # extra --hw-* flags
DEGRADE_THR="${DEGRADE_THR:-}"                         # optional override; empty = unified DEGRADE_REL_THRESHOLD = 0.10
N_TRIALS="${N_TRIALS:-30}"
SEED="${SEED:-100}"

OUT="${SLURM_SUBMIT_DIR}/outputs/hw_retrain_bo/${HW_PRESET}"   # same dir on every resubmission
PREPARE_SCRATCH="${SLURM_SUBMIT_DIR}/outputs/phase_a_prepare_scratch_v2"

# --------------------------------------------------------------------------
# 3. environment
# --------------------------------------------------------------------------
module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc

# The virtualenv of the 2026 run is not part of the repository checkout; fall
# back to the module python when it is absent (open question 8 of Phase 3b).
if [ -f "${ENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
else
  echo "WARNING: ${ENV_DIR}/bin/activate not found; using the module python" >&2
fi

# --------------------------------------------------------------------------
# 4. sanity checks
# --------------------------------------------------------------------------
test -f "$TEACHER_CKPT" || { echo "ERROR: teacher ckpt missing: $TEACHER_CKPT"; exit 1; }
test -f "$CANONICAL"    || { echo "ERROR: copy canonical_ctle_phase_a_v2.npz from ssr0 first"; exit 1; }

# --------------------------------------------------------------------------
# 5. hardware flags (all default OFF)
# --------------------------------------------------------------------------
HW_ARGS=(--hw-preset "$HW_PRESET")
if [ -n "$HW_CONFIG" ]; then HW_ARGS+=(--hw-config "$HW_CONFIG"); fi
if [ -n "$HW_EXTRA" ]; then read -r -a _extra <<< "$HW_EXTRA"; HW_ARGS+=("${_extra[@]}"); fi

# Failure threshold: unified DEGRADE_REL_THRESHOLD = 0.10 unless DEGRADE_THR
# overrides it.
THR_ARGS=()
if [ -n "$DEGRADE_THR" ]; then THR_ARGS+=(--zig-degrade-thr "$DEGRADE_THR"); fi

# --------------------------------------------------------------------------
# 6. the study
# --------------------------------------------------------------------------
python kn_bayes_opt.py \
  --dataset ctle --ctle-phase-a \
  --ctle-canonical-dataset "$CANONICAL" --ctle-objective validation \
  --ctle-phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --ctle-phase-a-mlp-weight 1.0 --ctle-phase-a-fwd-weight 1.0 --ctle-phase-a-power-weight 0.0005 \
  --ctle-phase-a-validity-weight 0.3 --ctle-phase-a-validity-ramp-start 10 --ctle-phase-a-validity-ramp-epochs 30 \
  --ctle-phase-a-power-norm train_mean \
  --readout shared-x2 --learnable-clip-sharpness \
  --gln-rails --gln-B 4 --gln-rank 2 --gln-families boundary,readout \
  --param-budget 7000 --param-tolerance 0.15 \
  --epochs 300 --n-trials "$N_TRIALS" --seed "$SEED" --device cuda \
  --no-seed-trial \
  ${THR_ARGS[@]+"${THR_ARGS[@]}"} \
  ${HW_ARGS[@]+"${HW_ARGS[@]}"} \
  --output "$OUT"

echo "DONE bo ${HW_PRESET} -> ${OUT}"
