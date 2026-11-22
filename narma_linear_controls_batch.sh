#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_linear_controls
#SBATCH --output=%x-%j.out

# NARMA-10 linear reservoir controls (plan narma-linear-controls,
# feature spec linear-control-probes).
#
# Scope: R0 Ridge-instrument reconciliation matrix, C0 ESN calibration
# control, C1 linear-reservoir-in-fabric control at hidden-49, and the
# C2/C3/C4 single-element bisection onto a passing C1. All controls are
# zero-training and eval-only.
#
# C1 PASS rule: MC > 3, washed-state PR >= 6, Ridge within matched
# parity (E0 Ridge <= raw-delay Ridge + 0.03). C2/C3/C4 run only when
# the C1 report's pass_all flag is true.
#
# Submit with:
#   sbatch narma_linear_controls_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_linear_controls_batch.sh
#   sbatch --export=LEG_TIMEOUT=3600 narma_linear_controls_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/linear_controls narma_linear_controls_batch.sh
#   sbatch --export=C3_GM_INIT=-2 narma_linear_controls_batch.sh
#     (best E0 sweep gain for the C3 tanh-core restoration leg)

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

# Fail fast if the GPU Python environment is incomplete.
python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

# C1b PASS rule: MC >= 5, rails <= 5%, J in [0.97, 0.99].
# Parity is diagnostic-only for linear legs.  The C3 sweep needs the
# C1b sidecar (c1b_tuned_base.pt + .json).

# Record the exact probe code used for this run.
sha256sum narma_linear_controls.py probe_audit2.py narma_linear_controls_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_linear_controls.py probe_audit2.py narma_experiment.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/linear_controls}"
C3_GM_INIT="${C3_GM_INIT:-0.0}"

FAILED_LEGS=()

run_control_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[linear-controls] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[linear-controls] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  # -u streams progress live; `timeout` bounds a runaway leg. rc=124 means
  # the per-leg cap was hit.
  if timeout "$LEG_TIMEOUT" python -u narma_linear_controls.py "$@" --output "$out"; then
    echo "[linear-controls] leg $name DONE"
  else
    echo "[linear-controls] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Remote preflight: tiny C1 must parse and run before the full legs.
echo ""
echo "=================================================================="
echo "[linear-controls] preflight: tiny C1 (hidden-9, 250 samples)"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_linear_controls.py c1 \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 250 --jacobian-samples 1 \
  --t-span 1.0 --num-steps 8 --hidden-dim 9 \
  --output "$PROBE_ROOT/c1_preflight"; then
  echo "[linear-controls] preflight DONE"
else
  echo "[linear-controls] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- R0 reconciliation matrix (canonical streams, hidden-25 to match E0).
run_control_leg "R0" \
  "$PROBE_ROOT/r0" "r0_reconciliation.json" \
  r0 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 1 \
    --train-samples 300 \
    --hidden-dim 25 \
    --t-span 1.0 \
    --num-steps 8

# ---- C0 ESN calibration control (canonical streams).
run_control_leg "C0" \
  "$PROBE_ROOT/c0" "c0_esn.json" \
  c0 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 1 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --max-delay 20 \
    --n-reservoir 25 \
    --spectral-radius 0.9 \
    --input-scaling 1.0 \
    --leak 1.0

# ---- C1 linear-reservoir-in-fabric control (hidden-49, full streams).
run_control_leg "C1" \
  "$PROBE_ROOT/c1" "c1_linear_reservoir.json" \
  c1 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 1 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 49 \
    --drive-scale 1.0 \
    --max-delay 20

# ---- C1b orthogonal-mixing revision (hidden-49, full streams).
# Persists the tuned base sidecar for the C3 sweep below.
run_control_leg "C1B" \
  "$PROBE_ROOT/c1b" "c1b.json" \
  c1b \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 1 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 49 \
    --drive-scale 1.0 \
    --w-seed 0 \
    --g-in-seed 0 \
    --leak-value 0.05 \
    --j-low 0.97 \
    --j-high 0.99 \
    --max-delay 20

# ---- C1b-real resistive-core variant (hidden-49, full streams).
run_control_leg "C1B_REAL" \
  "$PROBE_ROOT/c1b_real" "c1b_real.json" \
  c1b_real \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 1 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 49 \
    --drive-scale 1.0 \
    --boundary-drive-scale 0.5 \
    --w-seed 0 \
    --g-in-seed 0 \
    --leak-value 0.05 \
    --gm-raw-fill -10.0 \
    --isat-raw-fill -10.0 \
    --j-uniform-min 0.90 \
    --j-stability-cap 0.995 \
    --max-delay 20

# ---- C3 tanh-crossover sweep on the persisted C1b base.
if [[ -f "$PROBE_ROOT/c1b/c1b_tuned_base.pt" ]]; then
  run_control_leg "C3_SWEEP" \
    "$PROBE_ROOT/c3_sweep" "c3_sweep.json" \
    c3_sweep \
      --order 10 \
      --seed 0 \
      --device cuda \
      --n-streams 1 \
      --train-samples 2500 \
      --jacobian-samples 3 \
      --t-span 1.0 \
      --num-steps 8 \
      --hidden-dim 49 \
      --drive-scale 1.0 \
      --max-delay 20 \
      --c1b-sidecar-dir "$PROBE_ROOT/c1b"
else
  echo "[linear-controls] C1b sidecar missing; skipping C3 sweep"
fi

#test for sync
# ---- C2/C3/C4 bisection (only meaningful when C1 passes; the module
# itself skips the legs and reports c1-not-pass otherwise).
if [[ -f "$PROBE_ROOT/c1/c1_linear_reservoir.json" ]]; then
  run_control_leg "C2C3C4" \
    "$PROBE_ROOT/c2c3c4" "c2c3c4.json" \
    c2c3c4 \
      --order 10 \
      --seed 0 \
      --device cuda \
      --n-streams 1 \
      --train-samples 2500 \
      --jacobian-samples 3 \
      --t-span 1.0 \
      --num-steps 8 \
      --hidden-dim 49 \
      --drive-scale 1.0 \
      --max-delay 20 \
      --c3-gm-init "$C3_GM_INIT" \
      --c1-json "$PROBE_ROOT/c1/c1_linear_reservoir.json"
else
  echo "[linear-controls] C1 report missing; skipping C2/C3/C4"
fi

echo ""
echo "=================================================================="
echo "[linear-controls] summary"
echo "=================================================================="
python - "$PROBE_ROOT/r0/r0_reconciliation.json" "$PROBE_ROOT/c0/c0_esn.json" "$PROBE_ROOT/c1/c1_linear_reservoir.json" "$PROBE_ROOT/c2c3c4/c2c3c4.json" "$PROBE_ROOT/c1b/c1b.json" "$PROBE_ROOT/c1b_real/c1b_real.json" "$PROBE_ROOT/c3_sweep/c3_sweep.json" <<'PY'
import json
import sys

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

r0_path, c0_path, c1_path, c234_path, c1b_path, c1br_path, c3_path = sys.argv[1:8]
r0 = load(r0_path)
if r0 is not None:
    rule = r0.get("matched_parity_rule", {})
    print(f"--- {r0_path}")
    print(f"corners={r0.get('n_corners')} rule_holds={rule.get('rule_holds')} "
          f"delta={rule.get('delta_e0_minus_raw')}")
c0 = load(c0_path)
if c0 is not None:
    row = c0.get("row", {})
    print(f"--- {c0_path}")
    print(f"mc={row.get('mc_total')} pr={row.get('state_pr')} "
          f"nrmse={row.get('nrmse')} jac_max={row.get('jac_max_abs')}")
c1 = load(c1_path)
if c1 is not None:
    inst = c1.get("instrument", {})
    print(f"--- {c1_path}")
    print(f"pass_all={c1.get('pass_all')} mc={inst.get('mc_total')} "
          f"pr={inst.get('state_pr')} ridge={inst.get('nrmse')} "
          f"parity_delta={c1.get('matched_parity_delta')} "
          f"tuning_iters={c1.get('n_tuning_iterations')}")
c234 = load(c234_path)
if c234 is not None:
    print(f"--- {c234_path}")
    print(f"suppressor={c234.get('suppressor')}")
    for leg in c234.get("legs", []):
        print(f"  {leg.get('config_tag', '')[:40]} mc={leg.get('mc_total')} "
              f"pr={leg.get('state_pr')} ridge={leg.get('nrmse')}")
c1b = load(c1b_path)
if c1b is not None:
    inst = c1b.get("instrument", {})
    print(f"--- {c1b_path}")
    print(f"pass_all={c1b.get('pass_all')} mc={inst.get('mc_total')} "
          f"pr={inst.get('state_pr')} pr_std={c1b.get('state_pr_standardized')} "
          f"ridge={inst.get('nrmse')} jac_max={inst.get('jac_max_abs')} "
          f"rail={inst.get('rail_frac')} "
          f"tuning_iters={c1b.get('n_tuning_iterations')} "
          f"sidecar={c1b.get('sidecar_path')}")
c1br = load(c1br_path)
if c1br is not None:
    inst = c1br.get("instrument", {})
    print(f"--- {c1br_path}")
    print(f"pass_all={c1br.get('pass_all')} mc={inst.get('mc_total')} "
          f"pr={inst.get('state_pr')} ridge={inst.get('nrmse')} "
          f"jac_max={inst.get('jac_max_abs')} rail={inst.get('rail_frac')} "
          f"tuning_iters={c1br.get('n_tuning_iterations')}")
c3 = load(c3_path)
if c3 is not None:
    print(f"--- {c3_path}")
    print(f"c1b_mc={c3.get('c1b_mc_total')} "
          f"crossover_gm_raw_fill={c3.get('crossover_gm_raw_fill')}")
    for leg in c3.get("rows", []):
        print(f"  gm={leg.get('gm_raw_fill')} mc={leg.get('mc_total')} "
              f"ridge={leg.get('ridge_nrmse')} collapsed={leg.get('collapsed')}")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[linear-controls] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[linear-controls] all legs complete"
