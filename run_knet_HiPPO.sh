#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=4:00:00             # Time limit (30 trials x 300ep KNet; requeue/resume safe: kn_bayes_opt auto-resumes the study.db)
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

python test_memory_knet.py 