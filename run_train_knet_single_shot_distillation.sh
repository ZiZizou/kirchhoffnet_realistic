#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=03:00:00             # Time limit (C-full trials are ~1.3x legacy; requeue/resume safe: study.db auto-resumes)
#SBATCH --output=%x-%j.out          # Output file name

# Phase-A MLP-guided KNet BO, run C-full (plan docs/phase-a-mlp-guided-knet-plan.md):
#   1. Build the shared schema-2
canonical dataset if missing
#      (flow top-8 + frozen min-power tie-break + trial_0019 MLP logits,
#      20k specs, seed 100).
#   2. Run kn_bayes_opt CTLE in Phase-A mode with the 4-term guided loss:
#        MSE(student, trial_0019 teacher) + FWD*forward_mse
#        + 0.3*validity_NLL (ramp 10+30) + PWR*absolute_power,
#      lexicographic BO objective (val failure, valid_mean_power, val mlp_mse),
#      at 7000/7000 budget parity.
# Rerunning this script resumes both steps (dataset guard + study.db resume).

# Ensure execution in the directory where sbatch was run
cd "$SLURM_SUBMIT_DIR"

# 1. Clear and reload modules with correct capitalization
module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc

# 2. Activate virtual environment using absolute submit path
source "${SLURM_SUBMIT_DIR}/ENV/bin/activate"

# Frozen MLP teacher (trial_0019 BO winner) + schema-2 canonical dataset.
# NOTE: *_v2 name on purpose — do NOT reuse the schema-1 canonical_ctle_phase_a.npz.
TEACHER_CKPT="${SLURM_SUBMIT_DIR}/outputs/generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019/dagger_student_plain.pt"
CANONICAL="${SLURM_SUBMIT_DIR}/outputs/canonical_ctle_phase_a_v2.npz"
# Plan phase-a-fixes, bug 1: --output to a scratch dir so the prepare step
# never reads/writes the shared default CHECKPOINT_PATH
# (${OUTPUT_DIR}/dagger_checkpoint.pt); a stale checkpoint there would
# silently warm-start the trial.
PREPARE_SCRATCH="${SLURM_SUBMIT_DIR}/outputs/phase_a_prepare_scratch_v2"

test -f "$TEACHER_CKPT" || { echo "ERROR: teacher ckpt missing: $TEACHER_CKPT"; exit 1; }

# C-full loss weights (plan section 3). The epoch-1 "term magnitudes" line in
# each trial log shows the actual contribution split; retune here if one term
# dominates (e.g. power >30% of total -> lower PWR_W).
MLP_W=1.0
FWD_W=1.0
PWR_W=0.0005

# 3. Canonical dataset (skip if present; harness also loads instead of rebuilding)
if [ ! -f "$CANONICAL" ]; then
  python dagger-nuance-distillation-kirchhoffnet.py \
    --prepare-canonical-dataset "$CANONICAL" \
    --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
    --phase-a-n-samples 20000 \
    --phase-a-boundary-ratio 0.5 \
    --phase-a-power-norm train_mean \
    --seed 100 \
    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \
    --data-dir /home/annaik/Documents/augmented-cvae-ctle \
    --device cuda \
    --output "$PREPARE_SCRATCH"
fi

# 4. KNet BO, Phase-A MLP-guided (C-full)
/home/annaik/Documents/kirchhoffnet_realistic/ENV/bin/python /home/annaik/Documents/kirchhoffnet_realistic/dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \
  --phase-a-epochs 300 --kn-num-stages 6 --kn-num-hidden 12 --kn-small-world-k 4 --kn-small-world-p 0.200000 --kn-vca-rank 2 \
  --kn-x-max 4.000000 --kn-gm-max 1.351251e+00 --kn-isat-max 1.364811e+01 --kn-moe-num-experts 3 --kn-moe-gate-rank 2 --kn-moe-top-k 1 \
  --kn-mapper-lr-scale 1.0 --kn-struct-lr-scale 4.0 --kn-dyn-lr-scale 1.0 --lr 6.002040e-03 --weight-decay 7.893409e-04 --batch-size 512 --earlystop-eval-every 5 \
  --phase-a-validity-weight 0.300000 --phase-a-validity-ramp-start 10 --phase-a-validity-ramp-epochs 30 --phase-a-mlp-weight "$MLP_W" --phase-a-fwd-weight "$FWD_W" --phase-a-power-weight "$PWR_W" \
  --phase-a-power-norm train_mean --kn-readout shared-x2 --output /home/annaik/Documents/kirchhoffnet_realistic/outputs/manual_knet_distillation_mix_trials/trial_0005 \
  --device cuda --seed 100 --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --kn-learnable-clip-sharpness --kn-gln-rails --kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout \
  --t-span 5.084841 --boundary-fan-out "{\"0\": [0, 4], \"1\": [1, 5], \"2\": [2, 6], \"3\": [3, 7]}"
