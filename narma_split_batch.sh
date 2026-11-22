#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=narma_split
#SBATCH --output=%x-%j.out

# NARMA-10 split leg: per-node diagnostics + shunt-vs-tanh arms
# (plan canary-core-split, specs per-node-diagnostics + core-decomp-legs).
#
# Context: canary verdict CORE-USEFUL-NO-PASS (priority corner ridge 0.7611
# -> gz 0.8122, MC 1.68 -> gz 0.86, rails 15.2% -> gz 0%) with a bimodal
# per-node pattern (4 hot / 5 dead nodes). This leg decomposes the flip:
# Phase 0 per-node inspection (shunt-G sums, sign balance, boundary
# current vs max|x|/xmax) + Phase 1 arms A (shunt-only kill) / B
# (tanh-only kill) / C (gate-zero control). All probe-side fills, zero
# dynamics changes. Eval-only, ~5 corners x ~80 s.
#
# NOTE: *.sh is gitignored in this repo; ship with `git add -f` like the
# other narma_*_batch.sh scripts.
#
# Submit with:
#   sbatch narma_split_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_split_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/split narma_split_batch.sh

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
sha256sum narma_advisor_probes.py probe_audit_split.py narma_split_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_advisor_probes.py probe_audit_split.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-900}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/split}"

FAILED_LEGS=()

run_probe_leg() {
  local name="$1"
  local out="$2"
  local done_file="$3"
  shift 3
  echo ""
  echo "=================================================================="
  echo "[split] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[split] leg $name already complete ($done_file exists); skipping."
    return 0
  fi
  if timeout "$LEG_TIMEOUT" python -u narma_advisor_probes.py "$@" --output "$out"; then
    echo "[split] leg $name DONE"
  else
    echo "[split] leg $name FAILED (rc=$?; 124=LEG_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Preflight (fatal): compile + split audit + thin split smoke on CUDA.
echo ""
echo "=================================================================="
echo "[split] preflight: split audit + thin smoke"
echo "=================================================================="
python -B -m py_compile narma_advisor_probes.py probe_audit_split.py \
  || { echo "ERROR: preflight compile failed." >&2; exit 3; }
python -B probe_audit_split.py \
  || { echo "ERROR: split audit failed. Aborting before any GPU work." >&2; exit 3; }
if ! timeout "$PREFLIGHT_TIMEOUT" python -u narma_advisor_probes.py split \
  --order 10 --seed 0 --device cuda \
  --n-streams 1 --train-samples 300 --jacobian-samples 1 \
  --t-span 1.0 --num-steps 8 --hidden-dim 25 \
  --output "$PROBE_ROOT/preflight"; then
  echo "ERROR: preflight split smoke failed. Aborting before the full leg." >&2
  exit 3
fi

# ---- Full split leg (base + arms A/B/C, full streams, per-node diags).
run_probe_leg "split-full" "$PROBE_ROOT/full" "split.json" \
  split \
    --order 10 \
    --seed 0 \
    --device cuda \
    --n-streams 4 \
    --train-samples 2500 \
    --jacobian-samples 3 \
    --t-span 1.0 \
    --num-steps 8 \
    --hidden-dim 25

# ---- Verdict: determinism check vs canary.json + arm reading table.
echo ""
echo "=================================================================="
echo "[split] verdict"
echo "=================================================================="
if [[ -f "$PROBE_ROOT/full/split.json" ]]; then
  python - "$PROBE_ROOT/full/split.json" <<'EOF'
import json, sys
split = json.load(open(sys.argv[1]))
base = split["base"]["score"]
arm_c = split["arm_C"]["score"]
ok = True
for label, got, exp in (("base ridge", base["ridge_nrmse"], 0.7611),
                        ("base MC", base["mc_total"], 1.68),
                        ("base rail", base["rail_frac"], 0.152),
                        ("armC ridge", arm_c["ridge_nrmse"], 0.8122),
                        ("armC MC", arm_c["mc_total"], 0.86)):
    drift = abs(got - exp)
    flag = "OK " if drift <= 0.05 else "VOID"
    if drift > 0.05:
        ok = False
    print(f"  [{flag}] {label}: got {got:.4f} vs canary {exp:.4f} (drift {drift:.4f})")
bn = split["base"]["node"]
print(f"  base corrs: shunt/max={bn['corr_shunt_vs_max']:.3f} "
      f"sign/max={bn['corr_sign_vs_max']:.3f} "
      f"bnd/max={bn['corr_boundary_vs_max']:.3f}")
for arm in ("A", "B", "C"):
    s = split[f"arm_{arm}"]["score"]
    n = split[f"arm_{arm}"]["node"]
    hot = sum(1 for v in n["max_abs_ratio"] if v > 0.9)
    print(f"  arm {arm}: ridge={s['ridge_nrmse']:.4f} MC={s['mc_total']:.2f} "
          f"PR={s['state_pr']:.2f} rail={100.0 * s['rail_frac']:.1f}% hot={hot}")
print("  determinism: PASS" if ok else "  determinism: VOID -- do not interpret arms")
sys.exit(0 if ok else 2)
EOF
  if [[ "$?" == "2" ]]; then
    FAILED_LEGS+=("split-verdict-void")
  fi
else
  echo "--- $PROBE_ROOT/full/split.json (missing)"
  FAILED_LEGS+=("split-full-summary")
fi

echo ""
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[split] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[split] all legs complete."
