#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=24:00:00             # Time limit (30 trials x 300ep KNet; requeue/resume safe: kn_bayes_opt auto-resumes the study.db)
#SBATCH --output=%x-%j.out          # Output file name

# Canonical Phase-A KNet BO (plan canonical-ctle-unify):
#   1. Build the shared canonical flow-labelled dataset if missing
#      (flow top-8 + frozen min-power tie-break, 20k specs, seed 100).
#   2. Run kn_bayes_opt CTLE in Phase-A mode (static data, Huber-only,
#      Friedman differential-LR recipe, validation-failure objective)
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

CANONICAL="${SLURM_SUBMIT_DIR}/outputs/canonical_ctle_phase_a.npz"
# Plan phase-a-fixes, bug 1: --output to a scratch dir so the prepare step
# never reads/writes the shared default CHECKPOINT_PATH
# (${OUTPUT_DIR}/dagger_checkpoint.pt); a stale 52-edge / VCA-rank-3
# checkpoint there would silently warm-start the 56-edge / VCA-rank-2 trial.
PREPARE_SCRATCH="${SLURM_SUBMIT_DIR}/outputs/phase_a_prepare_scratch"

# 3. Canonical dataset (skip if present; harness also loads instead of rebuilding)
if [ ! -f "$CANONICAL" ]; then
  python dagger-nuance-distillation-kirchhoffnet.py \
    --prepare-canonical-dataset "$CANONICAL" \
    --phase-a-n-samples 20000 \
    --phase-a-boundary-ratio 0.5 \
    --seed 100 \
    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \
    --data-dir /home/annaik/Documents/augmented-cvae-ctle \
    --device cuda \
    --output "$PREPARE_SCRATCH"
fi

# 4. KNet BO only (Phase-A)
python kn_bayes_opt.py \
  --dataset ctle \
  --ctle-phase-a \
  --ctle-canonical-dataset "$CANONICAL" \
  --ctle-objective validation \
  --param-budget 7000 --param-tolerance 0.15 \
  --epochs 300 --n-trials 30 --seed 100 \
  --device cuda \
  --output outputs/phase_a_knet_bo
