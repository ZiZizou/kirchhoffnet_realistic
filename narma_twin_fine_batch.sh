#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=narma_twin_fine
#SBATCH --output=%x-%j.out

# NARMA-10 digital node-tanh twin FOLLOW-UP sweep (plan twin-node-tanh,
# feature spec twin-probe-spec). First grid (narma_twin-60146708.out,
# 144 corners) confirmed ESN-class memory (taps8 MC mean 8.10, max 10.29)
# with error floor ~0.50 (best 0.5047 at r=1.1/a=0.3/is=0.2/seed1).
# This batch concentrates on the winning region:
#   taps8 MAIN:  input_scale {0.10,0.15,0.20,0.25,0.30}
#              x leak_a {0.20,0.30,0.50,1.00}
#              x radius {1.00,1.10}
#              x w_seed {0,1,2}  (W-seed search: first grid used one W/corner)
#              x seeds {0,1,2}   = 360 legs
#   scalar CONFIRM: 3 best-guess combos x seeds {0,1,2} = 9 legs
# Total ~369 eval-only legs (~1.5 min/leg) ≈ 10 h. Resume-safe.
#
# Submit:
#   sbatch narma_twin_fine_batch.sh
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_twin_fine_batch.sh
#   sbatch --export=LEG_TIMEOUT=2400 narma_twin_fine_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/twin_fine narma_twin_fine_batch.sh
#   sbatch --export=WIN_SEED=1 narma_twin_fine_batch.sh

set -uo pipefail
# No `set -e`: a single failed leg must not kill the rest.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load StdEnv/2023
module load python/3.13
module load cuda/12.2
module load gcc

source "/home/annaik/Documents/tron_eq/venv/bin/activate"

python -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
  || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

sha256sum narma_linear_controls.py probe_audit2.py narma_twin_fine_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_linear_controls.py probe_audit2.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-2400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-300}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/twin_fine}"
WIN_SEED="${WIN_SEED:-0}"

# Winning-region grids (override any axis via env, comma-separated).
RADII="${RADII:-1.0,1.1}"
LEAKS="${LEAKS:-0.2,0.3,0.5,1.0}"
INPUT_SCALES="${INPUT_SCALES:-0.1,0.15,0.2,0.25,0.3}"
WSEEDS="${WSEEDS:-0,1,2}"
SEEDS="${SEEDS:-0,1,2}"

FAILED_LEGS=()

run_leg() {
  local tag="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[twin-fine] $tag -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[twin-fine] $tag already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_linear_controls.py "$@" --output "$out"; then
    echo "[twin-fine] $tag DONE"
  else
    echo "[twin-fine] $tag FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$tag")
  fi
}

# ---- Remote preflight: tiny twin must parse and run before the full legs.
echo ""
echo "=================================================================="
echo "[twin-fine] preflight: tiny taps8 (h=9, 200 samples)"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_linear_controls.py twin \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 200 --jacobian-samples 1 \
  --washout 50 --hidden-dim 9 \
  --harness taps8 --radius 1.1 --leak-a 0.3 --input-scale 0.2 \
  --output "$PROBE_ROOT/preflight"; then
  echo "[twin-fine] preflight DONE"
else
  echo "[twin-fine] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

IFS=',' read -ra RARR <<< "$RADII"
IFS=',' read -ra LARR <<< "$LEAKS"
IFS=',' read -ra IARR <<< "$INPUT_SCALES"
IFS=',' read -ra WARR <<< "$WSEEDS"
IFS=',' read -ra SARR <<< "$SEEDS"

# ---- MAIN: taps8 winning-region sweep with W-seed search.
for r in "${RARR[@]}"; do
  for la in "${LARR[@]}"; do
    for insc in "${IARR[@]}"; do
      for ws in "${WARR[@]}"; do
        for sd in "${SARR[@]}"; do
          tag="htaps8_r${r}_a${la}_is${insc}_ws${ws}_seed${sd}"
          run_leg "$tag" \
            "$PROBE_ROOT/$tag" "twin.json" \
            twin \
              --order 10 \
              --seed "$sd" \
              --device cuda \
              --harness taps8 \
              --radius "$r" \
              --leak-a "$la" \
              --input-scale "$insc" \
              --w-seed "$ws" \
              --win-seed "$WIN_SEED" \
              --n-taps 8 \
              --hidden-dim 25 \
              --n-streams 1 \
              --train-samples 2500 \
              --washout 200 \
              --max-delay 20 \
              --jacobian-samples 3
        done
      done
    done
  done
done

# ---- CONFIRM: scalar harness at best-guess combos (taps8 winner first).
# Combos: (r=1.1,a=0.3,is=0.2), (r=1.1,a=1.0,is=0.2), (r=1.0,a=1.0,is=0.2).
for combo in "1.1,0.3,0.2" "1.1,1.0,0.2" "1.0,1.0,0.2"; do
  IFS=',' read -ra C <<< "$combo"
  cr="${C[0]}"; ca="${C[1]}"; ci="${C[2]}"
  for sd in "${SARR[@]}"; do
    tag="hscalar_r${cr}_a${ca}_is${ci}_seed${sd}"
    run_leg "$tag" \
      "$PROBE_ROOT/$tag" "twin.json" \
      twin \
        --order 10 \
        --seed "$sd" \
        --device cuda \
        --harness scalar \
        --radius "$cr" \
        --leak-a "$ca" \
        --input-scale "$ci" \
        --w-seed 0 \
        --win-seed "$WIN_SEED" \
        --hidden-dim 25 \
        --n-streams 1 \
        --train-samples 2500 \
        --washout 200 \
        --max-delay 20 \
        --jacobian-samples 3
  done
done

echo ""
echo "=================================================================="
echo "[twin-fine] summary"
echo "=================================================================="
python - "$PROBE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load(tag):
    p = root / tag / "twin.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    inst = d.get("instrument", {})
    return {
        "tag": tag,
        "nrmse": inst.get("nrmse"),
        "mc": inst.get("mc_total"),
        "pr": inst.get("state_pr"),
        "activity": d.get("activity"),
        "jac_max": inst.get("jac_max_abs"),
        "engaged": d.get("tanh_engaged"),
        "esn_nrmse": d.get("esn_nrmse"),
        "esn_mc": d.get("esn_mc_total"),
        "verdict": d.get("verdict"),
    }

rows = []
for sub in sorted(root.iterdir()):
    if not sub.is_dir():
        continue
    s = load(sub.name)
    if s is not None and s["nrmse"] is not None:
        rows.append(s)

print(f"--- root={root} n_corners={len(rows)}")
taps = [r for r in rows if r["tag"].startswith("htaps8")]
scal = [r for r in rows if r["tag"].startswith("hscalar")]
for name, sub in (("taps8", taps), ("scalar", scal)):
    if not sub:
        continue
    mean_n = sum(r["nrmse"] for r in sub) / len(sub)
    mean_m = sum(r["mc"] for r in sub) / len(sub)
    n_pass = sum(1 for r in sub if r["verdict"] == "PASS_ARCH")
    n_fail = sum(1 for r in sub if r["verdict"] == "FAIL_FABRIC")
    n_void = sum(1 for r in sub if r["verdict"] == "VOID")
    print(f"== {name}: n={len(sub)} mean_nrmse={mean_n:.4f} "
          f"mean_mc={mean_m:.3f} pass={n_pass} fail={n_fail} void={n_void}")
    print(f"  {'tag':>42} {'nrmse':>7} {'mc':>6} {'pr':>6} "
          f"{'esn':>7} {'verdict':>12}")
    for r in sorted(sub, key=lambda x: x["nrmse"])[:10]:
        print(f"  {r['tag']:>42} {r['nrmse']:>7.4f} {r['mc']:>6.3f} "
              f"{r['pr']:>6.2f} {r['esn_nrmse']:>7.4f} {r['verdict']:>12}")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[twin-fine] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[twin-fine] all legs complete"
