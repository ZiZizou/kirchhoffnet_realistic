#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_revised_plan
#SBATCH --output=%x-%j.out

# Revised NARMA plan, Steps 1-4 (docs/narma-revised-plan.md): the
# measurement-first execution order that supersedes the S0-S5 ordering in
# narma-fabric-memory-problem-statement.md section 11.7.
#
#   S1  S1-extended memory probes on the frozen k2 checkpoint (k8 as a
#       bonus leg): u-MC curve, y-MC curve, IPC-2 curve, NARMA-10 lag-9
#       product-pair R^2. Tells you which deficit dominates.
#   S2a Tuned-ESN calibration: the full 288-corner grid. Settles the
#       0.15-0.29 band. Gate: best clearly below ~0.4 -> band stands;
#       stuck >= ~0.4 -> band miscalibrated, retire NARMA-10 as the gate.
#   S2b S2 E0 grid on two topologies: canonical torus AND small-world,
#       each over gain x {slow-fixed, randomized, 0.15, hetero-log-uniform}
#       x drive (48 corners per leg). One graph, no new machinery.
#   S3  Multi-tap trained readout: one dense-readout k=2 train (80 epochs,
#       same recipe as the k2 baseline). Zero dynamics risk.
#
# Each step gates the next; the gate verdicts are printed by the summary
# section at the end of this script. S3 runs regardless (the plan promotes
# or demotes H3 after S1, but the dense leg is cheap either way).
#
# NOT included: Step 5 (VCA-boundary leg -- needs vca_enabled plumbing
# through _build_fabric_net, not yet implemented; run on the best S2
# corner only once it exists) and Step 6 (forget-gate leak -- genuine
# architecture work, design still open). Two-core / systolic / feedback /
# Sugiura stay deferred per plan sections 8-9.
#
# Submit with:
#   sbatch narma_revised_plan_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_revised_plan_batch.sh
#   sbatch --export=K2_CKPT=/path/to/k2.pt narma_revised_plan_batch.sh
#   sbatch --export=LEG_TIMEOUT=3600,E0_LEG_TIMEOUT=10800,TRAIN_TIMEOUT=7200 narma_revised_plan_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/narma_revised narma_revised_plan_batch.sh
#
# Files this batch needs in the submit dir (ship alongside this script;
# *.sh and *audit* are gitignored, so copy them explicitly):
#   narma_revised_plan.py narma_advisor_probes.py narma_experiment.py
#   config.py topology.py cell_library.py kirchhoff_net.py io_mapper.py
#   differential_stage.py probe_audit_revised.py

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
sha256sum narma_revised_plan.py probe_audit_revised.py narma_advisor_probes.py narma_experiment.py narma_revised_plan_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_revised_plan.py narma_advisor_probes.py narma_experiment.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
# E0 legs run 48 corners at a measured >75 s/corner pace (two full Heun
# forwards + 6 Jacobians per corner), so they get a 3 h cap. Kills stay
# cheap: the sweep resumes from e0_progress.json on the next submit.
E0_LEG_TIMEOUT="${E0_LEG_TIMEOUT:-10800}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-7200}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/narma_revised}"
K2_CKPT="${K2_CKPT:-./outputs/narma_refresh2/fabric_refresh_k2_seed0.pt}"
K8_CKPT="${K8_CKPT:-./outputs/narma_diag_e3_k8/fabric_refresh_k8_seed0.pt}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  local script="$4"
  shift 4
  echo ""
  echo "=================================================================="
  echo "[revised-plan] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[revised-plan] leg $name already complete ($done_file exists); skipping."
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
    echo "[revised-plan] leg $name DONE"
  else
    echo "[revised-plan] leg $name FAILED (rc=$?; 124=timeout hit)"
    FAILED_LEGS+=("$name")
  fi
}

run_train_leg() {
  local name="$1"
  local out="$2"
  shift 2
  echo ""
  echo "=================================================================="
  echo "[revised-plan] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/results_table.csv" ]]; then
    echo "[revised-plan] leg $name already complete (results_table.csv exists); skipping."
    return 0
  fi
  printf '  cmd: python -u narma_experiment.py'
  printf ' %q' "$@"
  printf ' --output %q\n' "$out"
  if timeout "$TRAIN_TIMEOUT" python -u narma_experiment.py "$@" --output "$out"; then
    echo "[revised-plan] leg $name DONE"
  else
    echo "[revised-plan] leg $name FAILED (rc=$?; 124=TRAIN_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Preflight 1: hetero-leak init smoke (builds one net, forwards 200
# samples; validates imports + build path before any GPU work).
echo ""
echo "=================================================================="
echo "[revised-plan] preflight: hetero-leak init smoke"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_revised_plan.py hetero-leak-init \
  --order 10 --seed 0 --device cuda \
  --hidden-dim 25 --tau-lo 1.0 --tau-hi 40.0 \
  --t-span 1.0 --num-steps 8 \
  --output "$PROBE_ROOT/preflight"; then
  echo "[revised-plan] preflight hetero-leak DONE"
else
  echo "[revised-plan] preflight hetero-leak FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- Preflight 2: the revised-plan audit suite (CPU-only, ~1 min).
# probe_audit_revised.py is gitignored like the other audits; it must be
# shipped alongside this script. Skipped (not failed) when absent.
if [[ -f probe_audit_revised.py ]]; then
  echo ""
  echo "=================================================================="
  echo "[revised-plan] preflight: probe_audit_revised.py"
  echo "=================================================================="
  if timeout "$PREFLIGHT_TIMEOUT" python -u -B probe_audit_revised.py; then
    echo "[revised-plan] preflight audit DONE"
  else
    echo "[revised-plan] preflight audit FAILED; aborting before the full legs" >&2
    exit 2
  fi
else
  echo "[revised-plan] preflight audit SKIPPED (probe_audit_revised.py not shipped)"
fi

# ---- S1: S1-extended memory probe on the frozen k2 checkpoint ----
# Architecture flags must match the training run: tanh_free, k=2,
# t_span=1.0, num_steps=8, full-rail drive.
if [[ -f "$K2_CKPT" ]]; then
  run_probe_leg "S1-memory-k2" "$PROBE_ROOT/s1_k2" "s1_extended.json" \
    narma_revised_plan.py \
    memory \
      --order 10 \
      --seed 0 \
      --device cuda \
      --ckpt "$K2_CKPT" \
      --cell-library tanh_free \
      --core-refresh-interval 2 \
      --input-scale 1.0 \
      --n-streams 4 \
      --train-samples 2500
else
  echo "[revised-plan] S1-memory-k2 SKIPPED: checkpoint not found: $K2_CKPT"
  FAILED_LEGS+=("S1-memory-k2 (missing ckpt)")
fi

# ---- S1 bonus: same probe on the frozen k8 checkpoint ----
# Best-effort only: a missing k8 checkpoint skips quietly (no FAILED entry).
if [[ -f "$K8_CKPT" ]]; then
  run_probe_leg "S1-memory-k8" "$PROBE_ROOT/s1_k8" "s1_extended.json" \
    narma_revised_plan.py \
    memory \
      --order 10 \
      --seed 0 \
      --device cuda \
      --ckpt "$K8_CKPT" \
      --cell-library tanh_free \
      --core-refresh-interval 8 \
      --input-scale 1.0 \
      --n-streams 4 \
      --train-samples 2500
else
  echo "[revised-plan] S1-memory-k8 SKIPPED: checkpoint not found: $K8_CKPT"
fi

# ---- S2a: tuned-ESN calibration (full 288-corner grid) ----
# Grids passed explicitly even though they are the module defaults, so the
# submitted protocol is self-documenting.
run_probe_leg "S2-esn-cal" "$PROBE_ROOT/s2_esn_cal" "esn_cal.json" \
  narma_revised_plan.py \
  esn-cal \
    --order 10 \
    --seed 0 \
    --device cuda \
    --train-samples 2500 \
    --spectral-radius="0.7,0.9,1.0,1.1" \
    --leak="0.3,0.5,0.8,1.0" \
    --n-reservoir="25,50,100" \
    --input-scaling="0.2,0.5,1.0" \
    --ridge-lambda="1e-4,1e-2"

# ---- S2b: S2 E0 grid, canonical torus (48 corners) ----
# Gain x {slow-fixed, randomized, 0.15, hetero-log-uniform} x drive.
RUN_TIMEOUT="$E0_LEG_TIMEOUT" run_probe_leg "S2-e0-torus" "$PROBE_ROOT/s2_e0_torus" "e0_sweep.json" \
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
    --gain-grid="-5,-2,0,1.5" \
    --leak-grid="slow-fixed,randomized,0.15,hetero:1.0:40.0" \
    --drive-grid="0.25,0.5,1.0"

# ---- S2c: S2 E0 grid, small-world core (48 corners) ----
# Same grid as S2b; isolates the topology axis (plan section 7).
RUN_TIMEOUT="$E0_LEG_TIMEOUT" run_probe_leg "S2-e0-smallworld" "$PROBE_ROOT/s2_e0_smallworld" "e0_sweep.json" \
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
    --gain-grid="-5,-2,0,1.5" \
    --leak-grid="slow-fixed,randomized,0.15,hetero:1.0:40.0" \
    --drive-grid="0.25,0.5,1.0" \
    --use-small-world \
    --small-world-k 4 \
    --small-world-p 0.2 \
    --small-world-seed 1

# ---- S3: dense-readout k=2 train (same recipe as the k2 baseline) ----
# 80 epochs, early-stop 15, val every 5, checkpoint every 10. The dense
# readout (1080 params) replaces the OTA mesh + accumulator + OutputAffine;
# dynamics are unchanged. Condition + checkpoint carry the _dense suffix.
# --no-amp (fp32): the 2026-09-12 run died at epoch 0 on a non-finite
# BPTT gradient under CUDA AMP -- the dense read backprops through all 25
# hidden dims (vs the 1-D accumulator bottleneck), overflowing fp16 in the
# 25-chunk x 8-step chain, and the finiteness guard fires before the
# GradScaler can skip-and-recover. fp32 trains the identical regime cleanly
# (verified locally: same seed/streams, 200 chunks, loss descending).
# --compile stays ON (single-variable change; if S3 fails again without
# AMP, remove --compile next to test the inductor hypothesis).
run_train_leg "S3-dense-train" "$PROBE_ROOT/s3_dense_train" \
  --order 10 \
  --seeds 0 \
  --epochs 80 \
  --tbptt-chunk 25 \
  --n-streams 4 \
  --train-samples 2500 \
  --device cuda \
  --t-span 1.0 \
  --num-steps 8 \
  --core-refresh-interval 2 \
  --cell-library tanh_free \
  --readout dense \
  --ridge-diagnostic \
  --compile \
  --no-amp \
  --val-every 5 \
  --early-stop-patience 15 \
  --checkpoint-every 10 \
  --fabric-only \
  --no-progress

# ---- S5 note (not runnable yet) ----
echo ""
echo "[revised-plan] Step 5 (VCA-boundary leg) is NOT run by this batch:"
echo "  it needs vca_enabled plumbing through _build_fabric_net (not yet"
echo "  implemented) and runs on the best S2 corner only. See the S2b/S2c"
echo "  summaries below for the corner to target."

echo ""
echo "=================================================================="
echo "[revised-plan] gate summary"
echo "=================================================================="
python - "$PROBE_ROOT/s1_k2/s1_extended.json" "$PROBE_ROOT/s1_k8/s1_extended.json" "$PROBE_ROOT/s2_esn_cal/esn_cal.json" "$PROBE_ROOT/s2_e0_torus/e0_sweep.json" "$PROBE_ROOT/s2_e0_smallworld/e0_sweep.json" "$PROBE_ROOT/s3_dense_train/results_table.csv" <<'PY'
import csv
import json
import sys

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

s1k2_path, s1k8_path, esn_path, e0t_path, e0s_path, dense_csv = sys.argv[1:7]

for label, path in (("S1-k2", s1k2_path), ("S1-k8", s1k8_path)):
    data = load(path)
    if data is None:
        continue
    row = data.get("row", {})
    pairs = row.get("product_pair_r2", {}) or {}
    finite = {k: v for k, v in pairs.items()
              if isinstance(v, (int, float)) and v == v}  # drop NaN
    best_pair = max(finite.items(), key=lambda kv: kv[1]) if finite else ("none", float("nan"))
    print(f"--- {path}")
    print(f"decision={data.get('decision')} "
          f"u_mc={row.get('u_mc_total')} y_mc={row.get('y_mc_total')} "
          f"ipc2={row.get('ipc2_total')} best_pair={best_pair[0]}:{best_pair[1]}")

esn = load(esn_path)
if esn is not None:
    print(f"--- {esn_path}")
    print(f"corners={esn.get('n_corners')} best_nrmse={esn.get('best_nrmse')} "
          f"decision={esn.get('decision')}")
    for r in (esn.get("rows", []) or [])[:5]:
        print(f"  sr={r.get('spectral_radius')} leak={r.get('leak')} "
              f"nr={r.get('n_reservoir')} ins={r.get('input_scaling')} "
              f"rl={r.get('ridge_lambda')} -> NRMSE={r.get('nrmse')} "
              f"MC={r.get('mc_total')} PR={r.get('state_pr')}")

for label, path in (("S2b-torus", e0t_path), ("S2c-smallworld", e0s_path)):
    data = load(path)
    if data is None:
        continue
    rows = data.get("rows", []) or []
    passing = [r for r in rows if r.get("pass_all")]
    print(f"--- {path}")
    print(f"corners={len(rows)} passing={len(passing)}")
    for r in sorted(passing, key=lambda r: r.get("ridge_nrmse", 9e9))[:5]:
        print(f"  PASS gm={r.get('gm_init')} leak={r.get('leak_mode')} "
              f"drive={r.get('drive')} ridge={r.get('ridge_nrmse')} "
              f"mc={r.get('mc_total_washout_corrected')} pr={r.get('state_pr')} "
              f"rail={r.get('rail_frac')}")

try:
    with open(dense_csv) as handle:
        reader = csv.DictReader(handle)
        print(f"--- {dense_csv}")
        for r in reader:
            print(f"  {r.get('condition')} seed={r.get('seed')} "
                  f"NRMSE={r.get('nrmse')} R^2={r.get('r2')} "
                  f"MC={r.get('mc_total')} params={r.get('trained_params')}")
except FileNotFoundError:
    print(f"--- {dense_csv} (missing)")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[revised-plan] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[revised-plan] all legs complete"
