#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=03:00:00             # ~1h of legs when compiled; headroom for cold compile
#SBATCH --job-name=narma_diag
#SBATCH --output=%x-%j.out

# NARMA-10 memory-failure diagnostics batch (E1-E4).
#
# Context: k=2 run (outputs/narma_refresh2) plateaued at val NRMSE 0.675 with
# MC=0.23 while hidden RMS climbed to 1.57 (x_max=3.0) -- suspected
# compliance-gating saturation shutdown, not a recurrence problem (k=4 hit
# the same 0.67 wall). E5 (learned gated carry) is STASHED pending these
# logs; build it only if E2 shows carryover is actively harmful.
#
# Legs (sequential in ONE job: E2-on-k8 needs E3's checkpoint):
#   E3  k=8 frozen baseline, full 80ep (resume-first; ridge diagnostic only
#       on the fresh path -- the resume path's value is already logged)
#   E4a k=2 mapper LR x10 (two-timescale readout), 35ep + ridge diagnostic
#       (k=2 untrained-reservoir variant of E0)
#   E4b k=2 input drive x0.5 (half rail swing), 35ep
#   E4c k=2 fixed leak 0.15 (faster washout), 35ep
#   E4d k=2 mapper LR x10 + drive x0.5 combo, 35ep
#   E1/E2-k2  post-hoc saturation stats + carry ablation on the EXISTING
#             k2 checkpoint (no training; needs K2_CKPT below)
#   E2-k8     carry ablation on E3's k8 checkpoint
#
# What to read in the log afterwards:
#   "[ridge diagnostic]"  E0 fork: ~0.4-0.6 = reservoir OK (optimizer issue);
#                         ~1.0+ = reservoir itself has no memory (regime issue)
#   "xmaxr=/rail="        E1: rail% rising as val plateaus = saturation shutdown
#   "MASK "               E2 verdict: reset-hidden beating full = carryover
#                         harmful (green-light E5); else the reset idea is out
#   "RESULT:"             per-leg NRMSE/R2/MC summary
#
# TIMEOUT recovery: each leg is skippable -- rerun with SKIP_DONE=0, or resume
# a killed train leg manually, e.g.:
#   python -u narma_experiment.py <same leg args> \
#     --init-from <leg-out>/fabric_refresh_k<N>_seed0.pt
# (Checkpoints land every 10 epochs; --init-from needs single-k + single-seed.)

set -uo pipefail
# NOTE: no `set -e` on purpose -- one failed leg must not kill the rest.
# Failures are collected in FAILED_LEGS and reported at the end (exit != 0).

# ${VAR:-default} form so the script can also be dry-run outside Slurm
# (set -u would otherwise abort on the unset SLURM_* variables).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load StdEnv/2023
module load python/3.13
module load cuda/12.2
module load gcc

source "/home/annaik/Documents/tron_eq/venv/bin/activate"

# Fail-fast gate: this exact failure pattern (torch present, triton absent)
# burned 3h of GPU on the previous run because the compile wrapper silently
# fell back to eager (13x slower). Catch it in seconds here so a missing
# triton or wrong venv fails the job before any training starts.
python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed in /home/annaik/Documents/tron_eq/venv. Aborting before any GPU work." >&2; exit 3; }

# Re-run completed legs unless SKIP_DONE=0 (e.g. after code changes).
SKIP_DONE="${SKIP_DONE:-1}"
# Per-leg wall-clock cap (seconds). A compiled 35ep leg takes ~10 min and a
# fresh 80ep E3 ~20 min; 2400 is generous headroom. If a leg silently falls
# back to eager (~13x slower), `timeout` kills it at the cap (rc=124) so it
# can't eat the allocation and starve the remaining legs -- progress to that
# point survives in the every-10-epoch checkpoints and the leg is flagged
# FAILED below. Override: sbatch --export=LEG_TIMEOUT=3600 ...
LEG_TIMEOUT="${LEG_TIMEOUT:-2400}"
# Existing k2 checkpoint (E1-posthoc + E2-k2 legs). Override if yours lives
# elsewhere:  sbatch --export=K2_CKPT=/path/to/fabric_refresh_k2_seed0.pt ...
K2_CKPT="${K2_CKPT:-./outputs/narma_refresh2/fabric_refresh_k2_seed0.pt}"
# Output root for ALL legs below (mirrors PROBE_ROOT in
# narma_advisor_probes_batch.sh). Rerun to a fresh directory while keeping
# old results, e.g.:
#   sbatch --export=DIAG_ROOT=./outputs/narma_diag_rerun1,SKIP_DONE=0 narma_diag_batch.sh
# NOTE: K2_CKPT above is an INPUT and intentionally does NOT follow DIAG_ROOT;
# the E3 k8 checkpoint (input to E2-k8) DOES follow it via E3_CKPT.
DIAG_ROOT="${DIAG_ROOT:-./outputs}"

COMMON=(--order 10 --device cuda --compile --fabric-only
        --cell-library tanh_free --n-streams 4 --train-samples 2500
        --tbptt-chunk 25 --t-span 1.0 --num-steps 8 --seeds 0
        --val-every 5 --no-progress --checkpoint-every 10)

FAILED_LEGS=()

run_train_leg() {
  local name="$1"; local out="$2"; shift 2
  echo ""
  echo "=================================================================="
  echo "[diag] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/results_table.csv" ]]; then
    echo "[diag] leg $name already complete (results_table.csv exists); skipping."
    return 0
  fi
  # -u streams epoch lines live (otherwise Python block-buffers under sbatch).
  # `timeout` bounds runaway legs (see LEG_TIMEOUT); rc=124 means the cap hit.
  if timeout "$LEG_TIMEOUT" python -u narma_experiment.py "${COMMON[@]}" --output "$out" "$@"; then
    echo "[diag] leg $name DONE"
  else
    echo "[diag] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

run_eval_leg() {
  local name="$1"; local out="$2"; local ckpt="$3"; shift 3
  echo ""
  echo "=================================================================="
  echo "[diag] leg $name -> $out  (ckpt $ckpt)"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/carry_ablation.csv" ]]; then
    echo "[diag] leg $name already complete (carry_ablation.csv exists); skipping."
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[diag] leg $name SKIPPED: checkpoint not found: $ckpt"
    FAILED_LEGS+=("$name (missing ckpt)")
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_experiment.py "${COMMON[@]}" --output "$out" \
      --eval-masks full,reset-hidden,reset-all --init-from "$ckpt" "$@"; then
    echo "[diag] leg $name DONE"
  else
    echo "[diag] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- E3: k=8 frozen baseline (full run; cheapest config).
#      Resume-first: if the epoch-79 checkpoint from the cancelled run
#      exists, resume it to completion (empty train loop -> eval -> MC).
#      Otherwise start fresh. Either path lands the same results_table.csv.
E3_EXTRA=(--core-refresh-interval 8 --epochs 80 --early-stop-patience 15)
E3_CKPT="$DIAG_ROOT/narma_diag_e3_k8/fabric_refresh_k8_seed0.pt"
if [[ -f "$E3_CKPT" ]]; then
  # Resume path: the ridge value for this exact config is already in the
  # killed run's log (0.7194), so skip the duplicate diagnostic here.
  echo "[diag] E3-k8: resuming from $E3_CKPT"
  E3_EXTRA+=(--init-from "$E3_CKPT")
else
  echo "[diag] E3-k8: no checkpoint at $E3_CKPT; starting fresh"
  E3_EXTRA+=(--ridge-diagnostic)
fi
run_train_leg "E3-k8" "$DIAG_ROOT/narma_diag_e3_k8" "${E3_EXTRA[@]}"

# ---- E4: regime screens (short 35ep legs, k=2 to match the problem run) ----
run_train_leg "E4a-mapper10" "$DIAG_ROOT/narma_diag_e4a_mapper10" \
  --core-refresh-interval 2 --epochs 35 --early-stop-patience 10 \
  --mapper-lr-scale 10 --ridge-diagnostic

run_train_leg "E4b-drive05" "$DIAG_ROOT/narma_diag_e4b_drive05" \
  --core-refresh-interval 2 --epochs 35 --early-stop-patience 10 \
  --input-scale 0.5

run_train_leg "E4c-leak015" "$DIAG_ROOT/narma_diag_e4c_leak015" \
  --core-refresh-interval 2 --epochs 35 --early-stop-patience 10 \
  --leak-constant 0.15

run_train_leg "E4d-combo" "$DIAG_ROOT/narma_diag_e4d_combo" \
  --core-refresh-interval 2 --epochs 35 --early-stop-patience 10 \
  --mapper-lr-scale 10 --input-scale 0.5

# ---- E1-posthoc + E2 on the existing k2 checkpoint (arch flags must match
#      that run: tanh_free, k=2, defaults otherwise) ----
run_eval_leg "E2-k2" "$DIAG_ROOT/narma_diag_e2_k2" "$K2_CKPT" \
  --core-refresh-interval 2

# ---- E2 on the E3 k8 checkpoint (depends on E3 above) ----
run_eval_leg "E2-k8" "$DIAG_ROOT/narma_diag_e2_k8" \
  "$E3_CKPT" \
  --core-refresh-interval 8

echo ""
echo "=================================================================="
echo "[diag] batch summary (RESULT / MASK / SAT / ridge lines)"
echo "=================================================================="
_SELF_OUT="${SLURM_SUBMIT_DIR:-$PWD}/${SLURM_JOB_NAME:-narma_diag}-${SLURM_JOB_ID:-manual}.out"
grep -h -E "RESULT:|MASK |SAT \(|ridge diagnostic|Verdict:" "$_SELF_OUT" 2>/dev/null \
  || echo "[diag] (self-grep unavailable; inspect the .out file directly)"
echo ""
echo "--- E1 saturation series (validation epochs only: loss/xmaxr/rail/NRMSE) ---"
grep -h -E "fabric epoch.*NRMSE=" "$_SELF_OUT" 2>/dev/null \
  || echo "[diag] (no validation-epoch lines found)"
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[diag] FAILED/SKIPPED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[diag] all legs complete"
