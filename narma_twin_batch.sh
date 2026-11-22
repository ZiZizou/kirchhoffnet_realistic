#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_twin
#SBATCH --output=%x-%j.out

# NARMA-10 digital node-tanh twin probe (plan twin-node-tanh,
# feature spec twin-probe-spec). Tests the bottom-right cell of the
# linear-vs-nonlinear 2x2: dense linear mixing -> sum -> per-node tanh,
# x_{t+1} = (1-a)*x + a*tanh(W_mix@x + W_in@drive), with retention via
# the mixing matrix spectral radius (not leak gymnastics).
#
# Grid (locked, twin-probe-spec): radius {0.7, 0.9, 1.0, 1.1} x leak_a
# {0.3, 1.0} x input_scale {0.2, 0.5, 1.0} x seeds {0, 1, 2} = 72
# corners x 2 harnesses (taps8 + scalar), both run unconditionally.
#
# Submit:
#   sbatch narma_twin_batch.sh
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_twin_batch.sh
#   sbatch --export=LEG_TIMEOUT=2400 narma_twin_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/twin narma_twin_batch.sh

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

sha256sum narma_linear_controls.py probe_audit2.py narma_twin_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_linear_controls.py probe_audit2.py 2>/dev/null || true
fi

SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-2400}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-300}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/twin}"

RADII="${RADII:-0.7,0.9,1.0,1.1}"
LEAKS="${LEAKS:-0.3,1.0}"
INPUT_SCALES="${INPUT_SCALES:-0.2,0.5,1.0}"
SEEDS="${SEEDS:-0,1,2}"
HARNESSES="${HARNESSES:-taps8,scalar}"

FAILED_LEGS=()

run_leg() {
  local tag="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[twin] $tag -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[twin] $tag already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_linear_controls.py "$@" --output "$out"; then
    echo "[twin] $tag DONE"
  else
    echo "[twin] $tag FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$tag")
  fi
}

# ---- Remote preflight: tiny twin must parse and run before the full legs.
echo ""
echo "=================================================================="
echo "[twin] preflight: tiny scalar (h=9, 200 samples)"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u narma_linear_controls.py twin \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 200 --jacobian-samples 1 \
  --washout 50 --hidden-dim 9 \
  --harness scalar --radius 0.9 --leak-a 1.0 --input-scale 1.0 \
  --output "$PROBE_ROOT/preflight"; then
  echo "[twin] preflight DONE"
else
  echo "[twin] preflight FAILED; aborting before the full legs" >&2
  exit 2
fi

IFS=',' read -ra RARR <<< "$RADII"
IFS=',' read -ra LARR <<< "$LEAKS"
IFS=',' read -ra IARR <<< "$INPUT_SCALES"
IFS=',' read -ra SARR <<< "$SEEDS"
IFS=',' read -ra HARR <<< "$HARNESSES"

for h in "${HARR[@]}"; do
  for r in "${RARR[@]}"; do
    for la in "${LARR[@]}"; do
      for is in "${IARR[@]}"; do
        for sd in "${SARR[@]}"; do
          tag="h${h}_r${r}_a${la}_is${is}_seed${sd}"
          run_leg "$tag" \
            "$PROBE_ROOT/$tag" "twin.json" \
            twin \
              --order 10 \
              --seed "$sd" \
              --device cuda \
              --harness "$h" \
              --radius "$r" \
              --leak-a "$la" \
              --input-scale "$is" \
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

echo ""
echo "=================================================================="
echo "[twin] summary"
echo "=================================================================="
python - "$PROBE_ROOT" "$RADII" "$LEAKS" "$INPUT_SCALES" "$SEEDS" "$HARNESSES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
radii = sys.argv[2].split(",")
leaks = sys.argv[3].split(",")
inscales = sys.argv[4].split(",")
seeds = sys.argv[5].split(",")
harnesses = sys.argv[6].split(",")

print(f"--- root={root}")
print(f"--- radii={radii} leak_a={leaks} input_scale={inscales} seeds={seeds} harnesses={harnesses}")
print()

def summarize(json_path):
    if not json_path.exists():
        return None
    d = json.loads(json_path.read_text())
    inst = d.get("instrument", {})
    return {
        "nrmse": inst.get("nrmse"),
        "r2": inst.get("r2"),
        "mc_total": inst.get("mc_total"),
        "state_pr": inst.get("state_pr"),
        "state_pr_std": inst.get("state_pr_standardized"),
        "activity": d.get("activity"),
        "jac_max_abs": inst.get("jac_max_abs"),
        "tanh_engaged": d.get("tanh_engaged"),
        "esn_nrmse": d.get("esn_nrmse"),
        "esn_mc_total": d.get("esn_mc_total"),
        "pass_arch": d.get("pass_arch"),
        "fail_fabric": d.get("fail_fabric"),
        "void": d.get("void"),
        "verdict": d.get("verdict"),
        "matched_parity_delta": d.get("matched_parity_delta"),
    }

for h in harnesses:
    print(f"== harness={h} ==")
    print(f"  {'radius':>6} {'leak_a':>6} {'insc':>5} {'seed':>5} "
          f"{'nrmse':>7} {'mc':>6} {'state_pr':>9} "
          f"{'esn_nrmse':>9} {'esn_mc':>6} {'verdict':>15}")
    pass_arch_count = 0
    fail_fabric_count = 0
    void_count = 0
    nrmse_vals = []
    mc_vals = []
    for r in radii:
        for la in leaks:
            for insc in inscales:
                for sd in seeds:
                    tag = f"h{h}_r{r}_a{la}_is{insc}_seed{sd}"
                    s = summarize(root / tag / "twin.json")
                    if s is None:
                        print(f"  {r:>6} {la:>6} {insc:>5} {sd:>5} "
                              f"{'MISSING':>7}")
                        continue
                    if s["pass_arch"]:
                        pass_arch_count += 1
                    if s["fail_fabric"]:
                        fail_fabric_count += 1
                    if s["void"]:
                        void_count += 1
                    if s["nrmse"] is not None and s["mc_total"] is not None:
                        nrmse_vals.append(s["nrmse"])
                        mc_vals.append(s["mc_total"])
                    print(f"  {r:>6} {la:>6} {insc:>5} {sd:>5} "
                          f"{s['nrmse']:>7.4f} {s['mc_total']:>6.3f} "
                          f"{s['state_pr']:>9.3f} "
                          f"{s['esn_nrmse']:>9.4f} {s['esn_mc_total']:>6.3f} "
                          f"{s['verdict']:>15}")
    n = len(nrmse_vals)
    if n:
        mean_nrmse = sum(nrmse_vals) / n
        mean_mc = sum(mc_vals) / n
        print(f"  n_corners={n} pass_arch={pass_arch_count} "
              f"fail_fabric={fail_fabric_count} void={void_count}")
        print(f"  mean_nrmse={mean_nrmse:.4f} mean_mc={mean_mc:.3f}")
    print()
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[twin] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[twin] all legs complete"
