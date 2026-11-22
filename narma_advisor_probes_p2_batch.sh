#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_p2_rerun
#SBATCH --output=%x-%j.out

# NARMA-10 P2-only E0 rerun.
#
# Scope: rerun the failed P2 probe leg only. P0 and P1 are already complete
# and are intentionally not recomputed here.
#
# This leg remains zero-training and eval-only. It screens untrained
# reservoir configurations; it does not train the fabric.
#
# Ridge-parity-first PASS rule for a normal-trajectory corner:
#   ridge <= 0.40, MC > 3, washed-state PR >= 6, rail fraction <= 5%.
# The analogous gate-zero fields define the boundary-only PASS rule.
#
# Submit with:
#   sbatch narma_advisor_probes_p2_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_advisor_probes_p2_batch.sh
#   sbatch --export=LEG_TIMEOUT=3600 narma_advisor_probes_p2_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/advisor_probes narma_advisor_probes_p2_batch.sh

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

# Record the exact probe code used for this rerun.
sha256sum narma_advisor_probes.py probe_audit1.py narma_advisor_probes_p2_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_advisor_probes.py probe_audit1.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
# Each full 36-corner width now includes normal and boundary-only passes.
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/advisor_probes}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[p2-rerun] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[p2-rerun] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  # -u streams progress live; `timeout` bounds a runaway leg. rc=124 means
  # the per-leg cap was hit.
  if timeout "$LEG_TIMEOUT" python -u narma_advisor_probes.py "$@" --output "$out"; then
    echo "[p2-rerun] leg $name DONE"
  else
    echo "[p2-rerun] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Remote preflight: one small corner must parse and run before the full grids.
echo ""
echo "=================================================================="
echo "[p2-rerun] preflight: one hidden-25 corner"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_advisor_probes.py e0 \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 300 --jacobian-samples 1 \
  --t-span 1.0 --num-steps 8 --hidden-dim 25 \
  --gain-grid="-5" --leak-grid="slow-fixed" --drive-grid="0.5" \
  --max-corners 1 --output "$PROBE_ROOT/e0_preflight"; then
  echo "[p2-rerun] preflight DONE"
else
  echo "[p2-rerun] preflight FAILED; aborting before the full grids" >&2
  exit 2
fi

# ---- Full locked grids at both widths ----
for hidden_dim in 25 49; do
  run_probe_leg "P2-e0-h${hidden_dim}" \
    "$PROBE_ROOT/e0_full_h${hidden_dim}" "e0_sweep.json" \
    e0 \
      --order 10 \
      --seed 0 \
      --device cuda \
      --n-streams 4 \
      --train-samples 2500 \
      --jacobian-samples 3 \
      --t-span 1.0 \
      --num-steps 8 \
      --hidden-dim "$hidden_dim" \
      --gain-grid="-5,-2,0,1.5" \
      --leak-grid="slow-fixed,randomized,0.15" \
      --drive-grid="0.25,0.5,1.0"
done

echo ""
echo "=================================================================="
echo "[p2-rerun] sweep summary"
echo "=================================================================="
python - "$PROBE_ROOT/e0_full_h25/e0_sweep.json" "$PROBE_ROOT/e0_full_h49/e0_sweep.json" <<'PY'
import json
import sys

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

def summarize(path, data):
    rows = data.get("rows", [])
    normal = [row for row in rows if row.get("pass_all")]
    boundary = [row for row in rows if row.get("gate_zero_pass_all")]
    print(f"--- {path}")
    print(f"corners={len(rows)} normal_pass={len(normal)} boundary_pass={len(boundary)}")
    for label, winner in (
        ("best normal", min(normal, key=lambda row: row["ridge_nrmse"]) if normal else None),
        ("best boundary-only", min(boundary, key=lambda row: row["gate_zero_ridge_nrmse"]) if boundary else None),
    ):
        if winner is None:
            print(f"{label}: none")
        else:
            print(
                f"{label}: gm={winner['gm_init']} leak={winner['leak_mode']} "
                f"drive={winner['drive']} ridge={winner['ridge_nrmse']:.4f} "
                f"gz_ridge={winner['gate_zero_ridge_nrmse']:.4f} "
                f"mc={winner['mc_total_washout_corrected']:.2f} "
                f"state_pr={winner['state_pr']:.2f}"
            )

narrow_path, wide_path = sys.argv[1], sys.argv[2]
narrow, wide = load(narrow_path), load(wide_path)
if narrow is not None:
    summarize(narrow_path, narrow)
if wide is not None:
    summarize(wide_path, wide)
if narrow is not None and wide is not None:
    narrow_rows = {(row["gm_init"], row["leak_mode"], row["drive"]): row for row in narrow.get("rows", [])}
    wide_rows = {(row["gm_init"], row["leak_mode"], row["drive"]): row for row in wide.get("rows", [])}
    print("--- paired hidden-49 minus hidden-25 changes, ordered by Ridge improvement")
    paired = []
    for key, narrow_row in narrow_rows.items():
        wide_row = wide_rows.get(key)
        if wide_row is None:
            continue
        paired.append((
            wide_row["ridge_nrmse"] - narrow_row["ridge_nrmse"],
            key,
            narrow_row,
            wide_row,
        ))
    for delta, key, narrow_row, wide_row in sorted(paired, key=lambda item: item[0])[:10]:
        print(
            f"gm={key[0]} leak={key[1]} drive={key[2]} "
            f"ridge_delta={delta:+.4f} "
            f"mc={narrow_row['mc_total_washout_corrected']:.2f}->{wide_row['mc_total_washout_corrected']:.2f} "
            f"state_pr={narrow_row['state_pr']:.2f}->{wide_row['state_pr']:.2f}"
        )
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[p2-rerun] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[p2-rerun] all P2 legs complete; P0/P1 were intentionally not rerun"
