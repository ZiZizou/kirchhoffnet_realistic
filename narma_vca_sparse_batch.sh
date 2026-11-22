#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_vca_sparse
#SBATCH --output=%x-%j.out

# Round-2 execution batch: sparse-drive E0 + VCA-boundary legs
# (docs/narma-revised-plan.md section 13).
#
# Sequencing (author decision, plan section 13): sparse-drive E0 and VCA
# plumbing proceed in parallel; VCA trains run after, on both corners
# sequentially; ablations isolate each lever (one enabled / other off).
#
#   E0-sparse  Sparse-drive E0 grid (tests the forceful-drive hypothesis):
#       boundary fan-out {0: [0,5,10,15,20]} x gain {-5,-2,0} x leak
#       {slow-fixed, randomized, hetero:1.0:40.0} x drive {0.5, 1.0} =
#       18 corners (~25 min). Torus only. The S2b full-fan-out 48 serve as
#       the control arm -- no recompute.
#       Telltale (pre-registered): the gate-zero delta. Sparse drive
#       flipping it (gzMC << MC, gzR >> ridge) confirms full fan-out was
#       suppressing the core; gz still >= everywhere exonerates the core
#       independent of drive. Either outcome is decisive.
#   VCA-1      Full drive + VCA on (best-corner init: gain-init -2,
#       randomized leak, drive 0.25). Comparator exists: temporal k2 0.676.
#       Fallback to canonical gm=-5 init if day-0 rails crash it (set
#       VCA1_GAIN=-5 on resubmit).
#   Plain-sparse  No-VCA control for VCA-2: sparse fan-out + canonical
#       init, same recipe.
#   VCA-2      Sparse fan-out + VCA on, sparse-winner init.
#
# All trains: k=2, 80 epochs, early-stop 15, val every 5, checkpoint every
# 10, fp32 (--no-amp), --compile, temporal readout. Isolation matrix
# {full, sparse} x {VCA off, VCA on}.
#
# Attribution (pre-registered, plan section 13.3): VCA contains static
# per-edge gains (learned separability, present even at fixed input),
# dynamic u-dependent multiplication (the product path), but no true zeros
# (unlike ESN masks). Post-hoc read: product-pairs/IPC-2 jump with PR flat
# -> product path did it; PR jumps too -> separability contributed.
#
# Fork outcomes (pre-registered, plan section 13.5):
# - Sparse E0 flips gate-zero + VCA-2 wins -> drive map was the suppressor;
#   full-fan-out work stops.
# - VCA moves NRMSE/product-pairs with PR flat -> product path confirmed;
#   separability work stops.
# - VCA moves nothing on either corner -> multiplicative mechanism must come
#   from cells (S3-bistable revival) or the forget gate (Step 6 activates).
#
# Submit with:
#   sbatch narma_vca_sparse_batch.sh
#
# Useful overrides (sparse-winner init is unknown until E0-sparse lands;
# set the winner on resubmit and skip the done E0 leg):
#   sbatch --export=SKIP_DONE=0 narma_vca_sparse_batch.sh
#   sbatch --export=SPARSE_GAIN=-2,SPARSE_LEAK=randomized,SPARSE_DRIVE=0.5 narma_vca_sparse_batch.sh
#   sbatch --export=VCA1_GAIN=-5 narma_vca_sparse_batch.sh
#   sbatch --export=LEG_TIMEOUT=3600,E0_LEG_TIMEOUT=3600,TRAIN_TIMEOUT=7200 narma_vca_sparse_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/narma_round2 narma_vca_sparse_batch.sh
#
# Files this batch needs in the submit dir (ship alongside this script;
# *.sh and *audit* are gitignored, so copy them explicitly):
#   narma_revised_plan.py narma_advisor_probes.py narma_experiment.py
#   config.py topology.py cell_library.py kirchhoff_net.py io_mapper.py
#   differential_stage.py probe_audit_revised.py
#
# Explicitly NOT built (plan section 13.6): static random-sign sparse W_in
# (conditional on VCA attribution); forget-gate leak (parked behind
# VCA/cells); hetero-capacitance; GLN; small-world and all multi-topology
# grids; Mackey-Glass-17 (scope creep until a leg passes).

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

# Record the exact probe code used for this run.
sha256sum narma_revised_plan.py probe_audit_revised.py narma_advisor_probes.py narma_experiment.py narma_vca_sparse_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_revised_plan.py narma_advisor_probes.py narma_experiment.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
# Sparse E0 runs 18 corners at the measured ~80 s/corner pace (two full Heun
# forwards + 6 Jacobians per corner): ~25 min, 1 h cap leaves headroom.
# Kills stay cheap: the sweep resumes from e0_progress.json on resubmit.
E0_LEG_TIMEOUT="${E0_LEG_TIMEOUT:-3600}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-7200}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/narma_round2}"

# ---- Round-2 grids and inits (all overridable via sbatch --export) ----
# Canonical sparse-drive map (plan section 13.1): 5 evenly spaced nodes.
SPARSE_BFO="${SPARSE_BFO:-{\"0\":[0,5,10,15,20]}}"
# VCA-1 init (plan section 13.3): best-corner gain -2 / randomized / 0.25.
VCA1_GAIN="${VCA1_GAIN:--2}"
VCA1_LEAK="${VCA1_LEAK:-randomized}"
VCA1_DRIVE="${VCA1_DRIVE:-0.25}"
# Sparse-winner init (VCA-2 + plain-sparse control share it so the VCA
# ablation is exact). Defaults are the author's prior; set the E0-sparse
# winner on resubmit after reading the E0 summary below.
SPARSE_GAIN="${SPARSE_GAIN:--2}"
SPARSE_LEAK="${SPARSE_LEAK:-randomized}"
SPARSE_DRIVE="${SPARSE_DRIVE:-0.5}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  local script="$4"
  shift 4
  echo ""
  echo "=================================================================="
  echo "[round2] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[round2] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  # Log the exact command (%q preserves values) so the next failure is
  # diagnosable from the .out log without re-reading this script.
  printf '  cmd: python -u %s' "$script"
  printf ' %q' "$@"
  printf ' --output %q\n' "$out"
  # -u streams progress live; `timeout` bounds a runaway leg. rc=124 means
  # the per-leg cap was hit. RUN_TIMEOUT (one-shot env prefix at the call
  # site) overrides LEG_TIMEOUT for slow legs; a kill is cheap now that
  # the e0 path resumes corner-by-corner from e0_progress.json.
  if timeout "${RUN_TIMEOUT:-$LEG_TIMEOUT}" python -u "$script" "$@" --output "$out"; then
    echo "[round2] leg $name DONE"
  else
    echo "[round2] leg $name FAILED (rc=$?; 124=timeout hit)"
    FAILED_LEGS+=("$name")
  fi
}

run_train_leg() {
  local name="$1"
  local out="$2"
  shift 2
  echo ""
  echo "=================================================================="
  echo "[round2] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/results_table.csv" ]]; then
    echo "[round2] leg $name already complete (results_table.csv exists); skipping."
    return 0
  fi
  printf '  cmd: python -u narma_experiment.py'
  printf ' %q' "$@"
  printf ' --output %q\n' "$out"
  if timeout "$TRAIN_TIMEOUT" python -u narma_experiment.py "$@" --output "$out"; then
    echo "[round2] leg $name DONE"
  else
    echo "[round2] leg $name FAILED (rc=$?; 124=TRAIN_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Preflight 1: sparse + VCA build smoke (validates imports + the new
# plumbing -- boundary fan-out override, VCA gate allocation, and the
# gain/leak init-override path -- before any GPU work).
echo ""
echo "=================================================================="
echo "[round2] preflight: sparse + VCA build smoke"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u -c "
import torch, narma_experiment as ne, narma_revised_plan as nrp
torch.manual_seed(0)
net, _, _, log = nrp.build_vca_boundary_sparse_drive_net(
    order=10, seed=0, boundary_fan_out=nrp.DEFAULT_SPARSE_DRIVE_BFO,
    vca_enabled=True, vca_rank=2,
)
s = net.core.stages[0]
assert s.vca_enabled and s.vca_v_boundary.shape == (5, 2)
assert not s._vca_core_enabled and not s.vca_gate_shunt
print('  sparse+VCA build OK:', log)
"; then
  echo "[round2] preflight build DONE"
else
  echo "[round2] preflight build FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- Preflight 2: the revised-plan audit suite (CPU-only, ~2 min with the
# Round-2 N/Q checks). probe_audit_revised.py is gitignored like the other
# audits; it must be shipped alongside this script. Skipped (not failed)
# when absent.
if [[ -f probe_audit_revised.py ]]; then
  echo ""
  echo "=================================================================="
  echo "[round2] preflight: probe_audit_revised.py"
  echo "=================================================================="
  if timeout "$PREFLIGHT_TIMEOUT" python -u -B probe_audit_revised.py; then
    echo "[round2] preflight audit DONE"
  else
    echo "[round2] preflight audit FAILED; aborting before the full legs" >&2
    exit 2
  fi
else
  echo "[round2] preflight audit SKIPPED (probe_audit_revised.py not shipped)"
fi

# ---- E0-sparse: sparse-drive E0 grid, torus only (18 corners) ----
# Gain x {slow-fixed, randomized, hetero-log-uniform} x drive, with the
# canonical full-fan-out S2b 48 as the control arm (no recompute).
# Grids starting with '-' travel in --flag=value form (argparse
# negative-number regex); the hetero token avoids commas (colon form).
RUN_TIMEOUT="$E0_LEG_TIMEOUT" run_probe_leg "E0-sparse" "$PROBE_ROOT/e0_sparse" "e0_sweep.json" \
  narma_advisor_probes.py \
  e0 \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 4 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 25 \
    --gain-grid="-5,-2,0" \
    --leak-grid="slow-fixed,randomized,hetero:1.0:40.0" \
    --drive-grid="0.5,1.0" \
    --boundary-fan-out="$SPARSE_BFO"

# ---- VCA trains: shared recipe (k=2, 80 epochs, tbptt 25, tanh_free,
# temporal readout, --compile, fp32). --no-amp (fp32): the S3 dense leg died
# at epoch 0 on a non-finite BPTT gradient under CUDA AMP; the VCA legs keep
# fp32 for the same single-variable caution (drop --no-amp as a follow-up
# ablation once a VCA leg lands, not before).
_TRAIN_BASE=(
  --order 10
  --seeds 0
  --epochs 80
  --tbptt-chunk 25
  --n-streams 4
  --train-samples 2500
  --device cuda
  --t-span 1.0
  --num-steps 8
  --core-refresh-interval 2
  --cell-library tanh_free
  --ridge-diagnostic
  --compile
  --no-amp
  --val-every 5
  --early-stop-patience 15
  --checkpoint-every 10
  --fabric-only
  --no-progress
)

# ---- VCA-1: full drive + VCA on (best-corner init) ----
# No --boundary-fan-out flag: the canonical preset full fan-out is the
# full-drive arm. Comparator exists: temporal k2 0.676.
run_train_leg "VCA-1-full-vca" "$PROBE_ROOT/vca1_full" \
  "${_TRAIN_BASE[@]}" \
  --vca-boundary \
  --gain-init "$VCA1_GAIN" \
  --leak-init "$VCA1_LEAK" \
  --input-scale "$VCA1_DRIVE"

# ---- Plain-sparse: sparse fan-out, no VCA (control for VCA-2) ----
run_train_leg "plain-sparse" "$PROBE_ROOT/plain_sparse" \
  "${_TRAIN_BASE[@]}" \
  --boundary-fan-out="$SPARSE_BFO" \
  --gain-init "$SPARSE_GAIN" \
  --leak-init "$SPARSE_LEAK" \
  --input-scale "$SPARSE_DRIVE"

# ---- VCA-2: sparse fan-out + VCA on (sparse-winner init) ----
run_train_leg "VCA-2-sparse-vca" "$PROBE_ROOT/vca2_sparse" \
  "${_TRAIN_BASE[@]}" \
  --boundary-fan-out="$SPARSE_BFO" \
  --vca-boundary \
  --gain-init "$SPARSE_GAIN" \
  --leak-init "$SPARSE_LEAK" \
  --input-scale "$SPARSE_DRIVE"

echo ""
echo "=================================================================="
echo "[round2] gate summary"
echo "=================================================================="
python - "$PROBE_ROOT/e0_sparse/e0_sweep.json" "$PROBE_ROOT/vca1_full/results_table.csv" "$PROBE_ROOT/plain_sparse/results_table.csv" "$PROBE_ROOT/vca2_sparse/results_table.csv" <<'PY'
import csv
import json
import sys

e0_path, vca1_csv, plain_csv, vca2_csv = sys.argv[1:5]

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

# E0-sparse: full table + gate-zero deltas (the pre-registered telltale).
e0 = load(e0_path)
if e0 is not None:
    rows = e0.get("rows", []) or []
    print(f"--- {e0_path}")
    print(f"corners={len(rows)} fan_out={e0.get('canonical_net', {}).get('boundary_fan_out')}")
    print(f"{'gm':>6} {'leak':>28} {'drive':>5} {'ridge':>7} "
          f"{'mc':>6} {'sPR':>6} {'gzR':>7} {'gzMC':>6} "
          f"{'d_gzR':>7} {'d_gzMC':>7} {'rail%':>6} {'pass':>4}")
    flipped = 0
    for r in rows:
        d_gz_r = r.get("gate_zero_ridge_nrmse", float("nan")) - r.get("ridge_nrmse", float("nan"))
        d_gz_mc = r.get("gate_zero_mc_total", float("nan")) - r.get("mc_total_washout_corrected", float("nan"))
        # Gate-zero flip: core contributes (gz worse than full by margin).
        if d_gz_r > 0.05 and d_gz_mc < -0.5:
            flipped += 1
        print(f"{r.get('gm_init'):>6.2f} {str(r.get('leak_mode')):>28} {r.get('drive'):>5.2f} "
              f"{r.get('ridge_nrmse'):>7.4f} {r.get('mc_total_washout_corrected'):>6.2f} "
              f"{r.get('state_pr'):>6.2f} {r.get('gate_zero_ridge_nrmse'):>7.4f} "
              f"{r.get('gate_zero_mc_total'):>6.2f} "
              f"{d_gz_r:>+7.4f} {d_gz_mc:>+7.2f} "
              f"{100.0 * r.get('rail_frac', 0.0):>5.1f}% "
              f"{'PASS' if r.get('pass_all') else 'FAIL':>4}")
    passing = [r for r in rows if r.get("pass_all")]
    print(f"passing={len(passing)}/{len(rows)} gate_zero_flipped={flipped}/{len(rows)}")
    if flipped:
        print("TELLTALE: gate-zero flip present -- full fan-out was suppressing the core.")
    else:
        print("TELLTALE: no gate-zero flip -- core exonerated independent of drive.")

# Trains: per-leg rows + VCA-vs-control deltas.
def read_trains(path):
    try:
        with open(path) as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return []

legs = {}
for label, path in (("VCA-1", vca1_csv), ("plain-sparse", plain_csv), ("VCA-2", vca2_csv)):
    rows = read_trains(path)
    legs[label] = rows
    print(f"--- {path}")
    for r in rows:
        print(f"  {r.get('condition')} seed={r.get('seed')} "
              f"NRMSE={r.get('nrmse')} R^2={r.get('r2')} "
              f"MC={r.get('mc_total')} params={r.get('trained_params')} "
              f"vca={r.get('vca_enabled')} fan={r.get('boundary_fan_out')} "
              f"gi={r.get('gain_init')} le={r.get('leak_init')}")

def nrmse_of(rows):
    try:
        return float(rows[0].get("nrmse", "nan"))
    except (IndexError, TypeError, ValueError):
        return float("nan")

v1, pl, v2 = (nrmse_of(legs[k]) for k in ("VCA-1", "plain-sparse", "VCA-2"))
if v1 == v1 and pl == pl:
    print(f"delta VCA-1 - plain-sparse: {v1 - pl:+.4f} (negative = VCA helps on full drive)")
if v2 == v2 and pl == pl:
    print(f"delta VCA-2 - plain-sparse: {v2 - pl:+.4f} (negative = VCA helps on sparse drive)")
if v2 == v2 and v1 == v1:
    print(f"delta VCA-2 - VCA-1: {v2 - v1:+.4f} (negative = sparse drive helps under VCA)")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[round2] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[round2] all legs complete"
