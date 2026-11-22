#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=03:00:00             # Time limit (paper trial: 300 ep x 19-21 s = 1.6-1.75 h; hw flags add overhead)
#SBATCH --output=%x-%j.out          # Output file name

# HR-KNet hardware-constrained retrain, fixed paper topology (7x11, trial_0011
# hyper-parameters).  One rung of the ladder of docs/logs (Phase 3b section 3.4):
#   sbatch --export=ALL,HW_PRESET=phase3a_core,SEED=100 scripts/slurm/retrain_hw_phase_a.sh
#
# HW_PRESET=none          -> unconstrained baseline (R0), no --hw-* flags at all
# HW_PRESET=phase3a_core  -> R1: ratio A/B, slope bound, IC coupling, no alpha
# HW_PRESET=phase3a_full  -> R2: + compliance rail, physical c_node, GLN tail, Vref/xbar
# HW_PRESET=phase3b_tm    -> R4: + S&H, injection, droop, kT/C, time-multiplexed fabric
#
# The failure rule is the surgery's single constant DEGRADE_REL_THRESHOLD = 0.10
# ("fail if >= 2 of the 4 specs degrade by >= 10 %"), which is the rule the paper
# run already used (train 7.67 %, val 8.05 %, test 8.15 %).  No threshold flag
# is passed; set DEGRADE_THR in the environment to run an experiment at another
# threshold (that flag only exists if workstream W4 adds it).
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
HW_PRESET="${HW_PRESET:-phase3a_full}"      # none | phase3a_core | phase3a_full | phase3b_tm
HW_CONFIG="${HW_CONFIG:-}"                  # optional JSON overriding preset keys
HW_EXTRA="${HW_EXTRA:-}"                    # extra --hw-* flags, e.g. "--hw-tm --hw-tm-c-hold-ff 100"
DEGRADE_THR="${DEGRADE_THR:-}"             # optional override; empty = unified DEGRADE_REL_THRESHOLD = 0.10
SEED="${SEED:-100}"
RUN_TAG="${RUN_TAG:-${HW_PRESET}_s${SEED}}"

OUT="${SLURM_SUBMIT_DIR}/outputs/hw_retrain/${RUN_TAG}"
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
test -d "$ZIG_DIR"      || { echo "ERROR: ZIG model dir missing: $ZIG_DIR"; exit 1; }
if [ -d "$OUT" ]; then
  echo "ERROR: $OUT exists; refusing to warm-start from a stale checkpoint"; exit 1
fi

# Loss weights as in the C-full run (single_shot_distillation_2.sh:44-46)
MLP_W=1.0; FWD_W=1.0; PWR_W=0.0005

# --------------------------------------------------------------------------
# 5. canonical dataset: copy the paper's file when possible; rebuild only if
#    absent (a rebuild is not guaranteed bit-identical, so copying is preferred)
# --------------------------------------------------------------------------
if [ ! -f "$CANONICAL" ]; then
  python dagger-nuance-distillation-kirchhoffnet.py \
    --prepare-canonical-dataset "$CANONICAL" \
    --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
    --phase-a-n-samples 20000 --phase-a-boundary-ratio 0.5 --phase-a-power-norm train_mean \
    --seed 100 --teacher-dir "$ZIG_DIR" --data-dir "$DATA_DIR" \
    --device cuda --output "$PREPARE_SCRATCH"
fi

# --------------------------------------------------------------------------
# 6. hardware flags (all default OFF; HW_PRESET=none gives the baseline)
# --------------------------------------------------------------------------
HW_ARGS=()
KN_ISAT_MAX=2.868855e+01                    # paper search-bound value
if [ "$HW_PRESET" != "none" ]; then
  HW_ARGS+=(--hw-preset "$HW_PRESET")
  KN_ISAT_MAX=16                            # hardware tail ceiling (Phase 3a section 3.10)
fi
if [ -n "$HW_CONFIG" ]; then HW_ARGS+=(--hw-config "$HW_CONFIG"); fi
if [ -n "$HW_EXTRA" ]; then read -r -a _extra <<< "$HW_EXTRA"; HW_ARGS+=("${_extra[@]}"); fi

# Failure threshold: the surgery's unified DEGRADE_REL_THRESHOLD = 0.10 is the
# rule the paper run already used, so no flag is passed unless DEGRADE_THR is
# set explicitly (an experiment).
THR_ARGS=()
if [ -n "$DEGRADE_THR" ]; then THR_ARGS+=(--zig-degrade-thr "$DEGRADE_THR"); fi

# --------------------------------------------------------------------------
# 7. the trial
# --------------------------------------------------------------------------
python dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \
  --phase-a-epochs 300 --kn-num-stages 7 --kn-num-hidden 11 --kn-small-world-k 2 --kn-small-world-p 0.200000 --kn-vca-rank 3 \
  --kn-x-max 4.000000 --kn-gm-max 4.241970e+00 --kn-isat-max "$KN_ISAT_MAX" \
  --kn-mapper-lr-scale 1.0 --kn-struct-lr-scale 4.0 --kn-dyn-lr-scale 1.0 \
  --lr 7.069359e-03 --weight-decay 6.087302e-03 --batch-size 512 --earlystop-eval-every 5 \
  --phase-a-validity-weight 0.300000 --phase-a-validity-ramp-start 10 --phase-a-validity-ramp-epochs 30 \
  --phase-a-mlp-weight "$MLP_W" --phase-a-fwd-weight "$FWD_W" --phase-a-power-weight "$PWR_W" \
  --phase-a-power-norm train_mean --kn-readout shared-x2 \
  --kn-learnable-clip-sharpness --kn-gln-rails --kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout \
  --t-span 4.660583 --boundary-fan-out '{"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}' \
  --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --teacher-dir "$ZIG_DIR" --data-dir "$DATA_DIR" \
  ${THR_ARGS[@]+"${THR_ARGS[@]}"} \
  --device cuda --seed "$SEED" --output "$OUT" \
  ${HW_ARGS[@]+"${HW_ARGS[@]}"}

# --------------------------------------------------------------------------
# 8. post-train hardware audit (hw/audit.py: reload phase_a_best.pt, check the
#    slope bound, the rail excursion, the Heun stiffness, the code-value
#    quantisation and the SPICE-export parameter ranges).
#    hw/audit.py does not exist yet -> the block is a no-op until it lands.
# --------------------------------------------------------------------------
if [ -f hw/audit.py ] && [ "$HW_PRESET" != "none" ]; then
  python -m hw.audit --checkpoint "$OUT/phase_a_best.pt" --hw-preset "$HW_PRESET" ${HW_CONFIG:+--hw-config "$HW_CONFIG"}
fi

echo "DONE ${RUN_TAG} -> ${OUT}"
