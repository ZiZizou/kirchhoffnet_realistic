#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_round2_gated
#SBATCH --output=%x-%j.out

# Round-2 GATED execution batch: sparse-vs-dense E0 verdict FIRST, then only
# the winning fanout's train legs (docs/narma-revised-plan.md section 13).
#
# This is the gated companion to narma_vca_sparse_batch.sh (which runs every
# leg unconditionally). Run THIS script when you want the E0 evidence to
# decide where the GPU train budget goes:
#
#   Stage A (gate): matched 18-corner E0 grids on BOTH arms --
#       E0-sparse (boundary fan-out {0: [0,5,10,15,20]}) and E0-dense
#       (canonical full fan-out, same gain/leak/drive grid). Apples-to-apples:
#       gain {-5,-2,0} x leak {slow-fixed, randomized, hetero:1.0:40.0} x
#       drive {0.5, 1.0}, torus only (~25 min per arm).
#       (The S2b full-fan-out 48 stay a supplementary reference -- set
#       FULL_E0_REF to their e0_sweep.json and the summary prints their best
#       line. No recompute.)
#   Gate verdict (pre-registered, plan section 13.1 kill criteria):
#       - sparse best-MC clearly above dense best-MC (margin MC_MARGIN,
#         default 0.5) *with* >=1 gate-zero flip
#         -> winner=sparse: full-fan-out trains stop.
#       - sparse best-MC <= dense best-MC *and* zero gate-zero flips
#         -> winner=dense: sparse nets are dropped.
#       - anything else -> winner=ambiguous: run the full matrix (safe
#         fallback; identical to the ungated batch).
#       Best-MC is over rail-sane corners only (rail_frac <= 0.05).
#       Telltale: gate-zero flip (gzR >> ridge AND gzMC << MC) means full
#       fan-out was suppressing the core; gz still >= everywhere exonerates
#       the core independent of drive. Either outcome is decisive.
#   Stage B (conditional trains, shared recipe k=2 / 80 ep / tbptt 25 /
#   tanh_free / temporal / --compile / fp32):
#       - winner=sparse -> plain-sparse (control) + VCA-2 (sparse+VCA).
#       - winner=dense  -> plain-full (control) + VCA-1 (full+VCA).
#       - winner=ambiguous -> all four (full {full,sparse}x{VCA off,on}).
#   Inits (plan section 13.3): VCA-1 = best-corner (gain -2 / randomized /
#   drive 0.25; fallback gm=-5 on day-0 rail crash via VCA1_GAIN=-5).
#   plain-sparse + VCA-2 share the sparse-winner init (SPARSE_*; defaults are
#   the author's prior -- set the E0-sparse winner on resubmit). plain-full
#   uses the pure canonical init (no overrides; comparator temporal k2 0.676
#   exists).
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
#   sbatch narma_round2_gated_batch.sh
#
# Useful overrides:
#   sbatch --export=SKIP_DONE=0 narma_round2_gated_batch.sh
#   sbatch --export=FANOUT_WINNER=sparse narma_round2_gated_batch.sh
#       # force a branch without (re)computing the verdict: sparse | dense |
#       # ambiguous (= full matrix). Default auto = compute from E0 JSONs.
#   sbatch --export=FORCE_REGATE=1 narma_round2_gated_batch.sh
#       # recompute the verdict even if gate.env already exists.
#   sbatch --export=SPARSE_GAIN=-2,SPARSE_LEAK=randomized,SPARSE_DRIVE=0.5 narma_round2_gated_batch.sh
#   sbatch --export=VCA1_GAIN=-5 narma_round2_gated_batch.sh
#   sbatch --export=MC_MARGIN=0.5 narma_round2_gated_batch.sh
#   sbatch --export=FULL_E0_REF=./outputs/narma_revised/s2_e0_torus/e0_sweep.json narma_round2_gated_batch.sh
#   sbatch --export=LEG_TIMEOUT=3600,E0_LEG_TIMEOUT=3600,TRAIN_TIMEOUT=7200 narma_round2_gated_batch.sh
#   sbatch --export=PROBE_ROOT=./outputs/narma_round2_gated narma_round2_gated_batch.sh
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
sha256sum narma_revised_plan.py probe_audit_revised.py narma_advisor_probes.py narma_experiment.py narma_round2_gated_batch.sh 2>/dev/null || true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --short HEAD 2>/dev/null || true
  git status --short -- narma_revised_plan.py narma_advisor_probes.py narma_experiment.py 2>/dev/null || true
fi

# Re-run completed legs unless SKIP_DONE=0.
SKIP_DONE="${SKIP_DONE:-1}"
LEG_TIMEOUT="${LEG_TIMEOUT:-3600}"
# Each E0 arm runs 18 corners at the measured ~80 s/corner pace (two full Heun
# forwards + 6 Jacobians per corner): ~25 min per arm, 1 h cap leaves headroom.
# Kills stay cheap: each sweep resumes from its own e0_progress.json on resubmit.
E0_LEG_TIMEOUT="${E0_LEG_TIMEOUT:-3600}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-7200}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
PROBE_ROOT="${PROBE_ROOT:-./outputs/narma_round2_gated}"

# ---- Gate controls ----
# FANOUT_WINNER: auto (default) = compute from the two E0 JSONs; or force
# sparse | dense | ambiguous (= full train matrix, no gating).
FANOUT_WINNER="${FANOUT_WINNER:-auto}"
FORCE_REGATE="${FORCE_REGATE:-0}"
# "Clearly above" margin for the sparse-best-MC vs dense-best-MC comparison
# (pre-registered kill criterion, plan section 13.1).
MC_MARGIN="${MC_MARGIN:-0.5}"
# Supplementary S2b 48-corner reference (printed only, never recomputed).
FULL_E0_REF="${FULL_E0_REF:-./outputs/narma_revised/s2_e0_torus/e0_sweep.json}"

# ---- Round-2 grids and inits (all overridable via sbatch --export) ----
# Canonical sparse-drive map (plan section 13.1): 5 evenly spaced nodes.
SPARSE_BFO="${SPARSE_BFO:-{\"0\":[0,5,10,15,20]}}"
# VCA-1 init (plan section 13.3): best-corner gain -2 / randomized / 0.25.
VCA1_GAIN="${VCA1_GAIN:--2}"
VCA1_LEAK="${VCA1_LEAK:-randomized}"
VCA1_DRIVE="${VCA1_DRIVE:-0.25}"
# Sparse-winner init (VCA-2 + plain-sparse control share it so the VCA
# ablation is exact). Defaults are the author's prior; set the E0-sparse
# winner on resubmit after reading the gate summary below.
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
  echo "[round2-gated] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/$done_file" ]]; then
    echo "[round2-gated] leg $name already complete ($done_file exists); skipping."
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
  # each sweep resumes corner-by-corner from its own e0_progress.json.
  if timeout "${RUN_TIMEOUT:-$LEG_TIMEOUT}" python -u "$script" "$@" --output "$out"; then
    echo "[round2-gated] leg $name DONE"
  else
    echo "[round2-gated] leg $name FAILED (rc=$?; 124=timeout hit)"
    FAILED_LEGS+=("$name")
  fi
}

run_train_leg() {
  local name="$1"
  local out="$2"
  shift 2
  echo ""
  echo "=================================================================="
  echo "[round2-gated] leg $name -> $out"
  echo "=================================================================="
  if [[ "$SKIP_DONE" == "1" && -f "$out/results_table.csv" ]]; then
    echo "[round2-gated] leg $name already complete (results_table.csv exists); skipping."
    return 0
  fi
  printf '  cmd: python -u narma_experiment.py'
  printf ' %q' "$@"
  printf ' --output %q\n' "$out"
  if timeout "$TRAIN_TIMEOUT" python -u narma_experiment.py "$@" --output "$out"; then
    echo "[round2-gated] leg $name DONE"
  else
    echo "[round2-gated] leg $name FAILED (rc=$?; 124=TRAIN_TIMEOUT hit)"
    FAILED_LEGS+=("$name")
  fi
}

# ---- Preflight 1: sparse + VCA build smoke (validates imports + the new
# plumbing -- boundary fan-out override, VCA gate allocation, and the
# gain/leak init-override path -- before any GPU work).
echo ""
echo "=================================================================="
echo "[round2-gated] preflight: sparse + VCA build smoke"
echo "=================================================================="
if timeout "$PREFLIGHT_TIMEOUT" python -u -c "
import torch, narma_revised_plan as nrp
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
  echo "[round2-gated] preflight build DONE"
else
  echo "[round2-gated] preflight build FAILED; aborting before the full legs" >&2
  exit 2
fi

# ---- Preflight 2: the revised-plan audit suite (CPU-only).
# probe_audit_revised.py is gitignored like the other audits; it must be
# shipped alongside this script. Skipped (not failed) when absent.
if [[ -f probe_audit_revised.py ]]; then
  echo ""
  echo "=================================================================="
  echo "[round2-gated] preflight: probe_audit_revised.py"
  echo "=================================================================="
  if timeout "$PREFLIGHT_TIMEOUT" python -u -B probe_audit_revised.py; then
    echo "[round2-gated] preflight audit DONE"
  else
    echo "[round2-gated] preflight audit FAILED; aborting before the full legs" >&2
    exit 2
  fi
else
  echo "[round2-gated] preflight audit SKIPPED (probe_audit_revised.py not shipped)"
fi

# ---- Stage A, gate 1: matched 18-corner E0 grids on BOTH arms ----
# Identical gain/leak/drive grids so the sparse-vs-dense comparison is exact.
# Grids starting with '-' travel in --flag=value form (argparse
# negative-number regex); the hetero token avoids commas (colon form).
_E0_GRID=(
  e0
    --order 10
    --seed 0
    --device cuda
    --n-streams 4
    --train-samples 2500
    --jacobian-samples 3
    --t-span 1.0
    --num-steps 8
    --hidden-dim 25
    --gain-grid="-5,-2,0"
    --leak-grid="slow-fixed,randomized,hetero:1.0:40.0"
    --drive-grid="0.5,1.0"
)

# E0-dense: canonical full fan-out control (no --boundary-fan-out flag).
RUN_TIMEOUT="$E0_LEG_TIMEOUT" run_probe_leg "E0-dense" "$PROBE_ROOT/e0_dense" "e0_sweep.json" \
  narma_advisor_probes.py \
  "${_E0_GRID[@]}"

# E0-sparse: sparse-drive arm (tests the forceful-drive hypothesis, §13.1).
RUN_TIMEOUT="$E0_LEG_TIMEOUT" run_probe_leg "E0-sparse" "$PROBE_ROOT/e0_sparse" "e0_sweep.json" \
  narma_advisor_probes.py \
  "${_E0_GRID[@]}" \
  --boundary-fan-out="$SPARSE_BFO"

# ---- Gate verdict: sparse-vs-dense, pre-registered criteria ----
# Writes $PROBE_ROOT/gate_verdict.json (full numbers) + $PROBE_ROOT/gate.env
# (FANOUT_WINNER=<sparse|dense|ambiguous>, sourced below). Reused on resubmit
# unless FORCE_REGATE=1. A missing E0 arm -> ambiguous (safe fallback: full
# train matrix, same as the ungated batch).
echo ""
echo "=================================================================="
echo "[round2-gated] gate verdict: sparse vs dense E0"
echo "=================================================================="
if [[ "$FANOUT_WINNER" != "auto" ]]; then
  echo "[round2-gated] FANOUT_WINNER=$FANOUT_WINNER forced via env; skipping verdict computation."
  printf 'FANOUT_WINNER=%s\n' "$FANOUT_WINNER" > "$PROBE_ROOT/gate.env"
elif [[ "$FORCE_REGATE" != "1" && -f "$PROBE_ROOT/gate.env" && -f "$PROBE_ROOT/gate_verdict.json" ]]; then
  echo "[round2-gated] reusing existing verdict (FORCE_REGATE=1 to recompute):"
  cat "$PROBE_ROOT/gate.env"
else
  python - "$PROBE_ROOT/e0_dense/e0_sweep.json" "$PROBE_ROOT/e0_sparse/e0_sweep.json" "$PROBE_ROOT/gate_verdict.json" "$PROBE_ROOT/gate.env" "$MC_MARGIN" <<'PY'
import json
import math
import sys

dense_path, sparse_path, verdict_path, env_path, margin_s = sys.argv[1:6]
margin = float(margin_s)
RAIL_SANE_AT_OR_BELOW = 0.05  # RAIL_DISQUALIFY_ABOVE in narma_advisor_probes.py
FLIP_D_GZR_ABOVE = 0.05       # gzR - ridge: core removal hurts ridge
FLIP_D_GZMC_BELOW = -0.5      # gzMC - MC: core removal hurts memory

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, OSError, ValueError):
        return None

def finite(x):
    return isinstance(x, (int, float)) and x == x and math.isfinite(x)

def best_mc_of(data):
    """Best rail-sane MC of an arm, or NaN when the arm has no sane corner."""
    rows = (data or {}).get("rows", []) or []
    sane = [r for r in rows
            if finite(r.get("rail_frac")) and r["rail_frac"] <= RAIL_SANE_AT_OR_BELOW
            and finite(r.get("mc_total_washout_corrected"))]
    if not sane:
        return float("nan")
    return max(r["mc_total_washout_corrected"] for r in sane)

def summarize(data):
    """Best rail-sane MC + gate-zero flip count over an E0 arm."""
    rows = (data or {}).get("rows", []) or []
    sane = [r for r in rows
            if finite(r.get("rail_frac")) and r["rail_frac"] <= RAIL_SANE_AT_OR_BELOW
            and finite(r.get("mc_total_washout_corrected"))]
    best = min(sane, key=lambda r: r["ridge_nrmse"]) if sane else None
    flips = 0
    for r in sane:
        if not (finite(r.get("ridge_nrmse")) and finite(r.get("gate_zero_ridge_nrmse"))
                and finite(r.get("mc_total_washout_corrected")) and finite(r.get("gate_zero_mc_total"))):
            continue
        d_r = r["gate_zero_ridge_nrmse"] - r["ridge_nrmse"]
        d_mc = r["gate_zero_mc_total"] - r["mc_total_washout_corrected"]
        if d_r > FLIP_D_GZR_ABOVE and d_mc < FLIP_D_GZMC_BELOW:
            flips += 1
    return {
        "n_corners": len(rows),
        "n_sane": len(sane),
        "best_mc": best["mc_total_washout_corrected"] if best else float("nan"),
        "best_ridge": best["ridge_nrmse"] if best else float("nan"),
        "best_corner": ({k: best.get(k) for k in
                         ("gm_init", "leak_mode", "drive", "state_pr", "rail_frac")}
                        if best else None),
        "flips": flips,
    }

dense = load(dense_path)
sparse = load(sparse_path)
if dense is None or sparse is None:
    missing = [p for p, d in ((dense_path, dense), (sparse_path, sparse)) if d is None]
    verdict = {"winner": "ambiguous",
               "reason": f"missing E0 arm(s): {missing}; safe fallback = full train matrix"}
else:
    d, s = summarize(dense), summarize(sparse)
    if s["best_mc"] > d["best_mc"] + margin and s["flips"] >= 1:
        winner = "sparse"
        reason = (f"sparse best-MC {s['best_mc']:.2f} > dense {d['best_mc']:.2f} "
                  f"+ margin {margin} with {s['flips']} gate-zero flip(s): "
                  f"drive map was the suppressor; full-fan-out trains stop")
    elif not (s["best_mc"] > d["best_mc"]) and s["flips"] == 0:
        winner = "dense"
        reason = (f"sparse best-MC {s['best_mc']:.2f} <= dense {d['best_mc']:.2f} "
                  f"with 0 gate-zero flips: drive sparsity is not a lever; "
                  f"sparse nets are dropped")
    else:
        winner = "ambiguous"
        reason = (f"sparse best-MC {s['best_mc']:.2f} vs dense {d['best_mc']:.2f} "
                  f"(margin {margin}), sparse flips {s['flips']}: no decisive "
                  f"reading; run the full train matrix")
    verdict = {"winner": winner, "reason": reason, "margin": margin,
               "dense": d, "sparse": s,
               "flip_thresholds": {"d_gzR_above": FLIP_D_GZR_ABOVE,
                                   "d_gzMC_below": FLIP_D_GZMC_BELOW,
                                   "rail_sane_at_or_below": RAIL_SANE_AT_OR_BELOW}}

with open(verdict_path, "w") as handle:
    json.dump(verdict, handle, indent=2)
with open(env_path, "w") as handle:
    handle.write(f"FANOUT_WINNER={verdict['winner']}\n")
print(f"winner={verdict['winner']}: {verdict['reason']}")
PY
  if [[ ! -f "$PROBE_ROOT/gate.env" ]]; then
    echo "[round2-gated] verdict computation FAILED; falling back to ambiguous (full matrix)" >&2
    printf 'FANOUT_WINNER=%s\n' "ambiguous" > "$PROBE_ROOT/gate.env"
    FAILED_LEGS+=("gate-verdict")
  fi
fi
# shellcheck disable=SC1090
# The file is written by this script (printf above / python heredoc) or by an
# explicit sbatch --export override -- never shipped — so a missing file here
# means the heredoc failed without setting -e; guard instead of assuming.
if [[ ! -f "$PROBE_ROOT/gate.env" ]]; then
  echo "[round2-gated] gate.env missing after verdict block; falling back to ambiguous" >&2
  printf 'FANOUT_WINNER=%s\n' "ambiguous" > "$PROBE_ROOT/gate.env"
  FAILED_LEGS+=("gate-verdict-missing")
fi
# shellcheck disable=SC1090
source "$PROBE_ROOT/gate.env"
echo "[round2-gated] gated fanout for Stage B: $FANOUT_WINNER"
if [[ "$FANOUT_WINNER" != "sparse" && "$FANOUT_WINNER" != "dense" && "$FANOUT_WINNER" != "ambiguous" ]]; then
  echo "[round2-gated] WARNING: unexpected FANOUT_WINNER=$FANOUT_WINNER; treating as ambiguous" >&2
  FANOUT_WINNER="ambiguous"
fi

# ---- Stage B: train legs on the winning fanout ----
# Shared recipe (k=2, 80 epochs, tbptt 25, tanh_free, temporal readout,
# --compile, fp32). --no-amp (fp32): the S3 dense leg died at epoch 0 on a
# non-finite BPTT gradient under CUDA AMP; the VCA legs keep fp32 for the
# same single-variable caution (drop --no-amp as a follow-up ablation once
# a VCA leg lands, not before).
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

# Dense arms (run when winner is dense or ambiguous).
if [[ "$FANOUT_WINNER" == "dense" || "$FANOUT_WINNER" == "ambiguous" ]]; then
  # ---- Plain-full: canonical init, no VCA (dense control) ----
  # No --boundary-fan-out flag: the canonical preset full fan-out is the
  # full-drive arm. Comparator exists: temporal k2 0.676.
  run_train_leg "plain-full" "$PROBE_ROOT/plain_full" \
    "${_TRAIN_BASE[@]}"

  # ---- VCA-1: full drive + VCA on (best-corner init) ----
  run_train_leg "VCA-1-full-vca" "$PROBE_ROOT/vca1_full" \
    "${_TRAIN_BASE[@]}" \
    --vca-boundary \
    --gain-init "$VCA1_GAIN" \
    --leak-init "$VCA1_LEAK" \
    --input-scale "$VCA1_DRIVE"
else
  echo "[round2-gated] dense trains SKIPPED (gate winner: $FANOUT_WINNER)"
fi

# Sparse arms (run when winner is sparse or ambiguous). VCA-2 + plain-sparse
# share the sparse-winner init so the VCA ablation is exact.
if [[ "$FANOUT_WINNER" == "sparse" || "$FANOUT_WINNER" == "ambiguous" ]]; then
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
else
  echo "[round2-gated] sparse trains SKIPPED (gate winner: $FANOUT_WINNER)"
fi

echo ""
echo "=================================================================="
echo "[round2-gated] gate summary"
echo "=================================================================="
python - "$PROBE_ROOT/gate_verdict.json" "$PROBE_ROOT/e0_dense/e0_sweep.json" "$PROBE_ROOT/e0_sparse/e0_sweep.json" "$FULL_E0_REF" "$PROBE_ROOT/plain_full/results_table.csv" "$PROBE_ROOT/vca1_full/results_table.csv" "$PROBE_ROOT/plain_sparse/results_table.csv" "$PROBE_ROOT/vca2_sparse/results_table.csv" <<'PY'
import csv
import json
import sys

verdict_path, e0d_path, e0s_path, ref_path = sys.argv[1:5]
train_csvs = {"plain-full": sys.argv[5], "VCA-1": sys.argv[6],
              "plain-sparse": sys.argv[7], "VCA-2": sys.argv[8]}

def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"--- {path} (missing)")
        return None

# Gate verdict + both E0 arms with gate-zero deltas (the telltale).
verdict = load(verdict_path)
if verdict is not None:
    print(f"--- {verdict_path}")
    print(f"winner={verdict.get('winner')}: {verdict.get('reason')}")
    for arm in ("dense", "sparse"):
        s = verdict.get(arm) or {}
        print(f"  {arm}: corners={s.get('n_corners')} sane={s.get('n_sane')} "
              f"best_mc={s.get('best_mc')} best_ridge={s.get('best_ridge')} "
              f"flips={s.get('flips')} best_corner={s.get('best_corner')}")

for label, path in (("E0-dense", e0d_path), ("E0-sparse", e0s_path)):
    data = load(path)
    if data is None:
        continue
    rows = data.get("rows", []) or []
    print(f"--- {path}")
    print(f"corners={len(rows)} fan_out={data.get('canonical_net', {}).get('boundary_fan_out')}")
    print(f"{'gm':>6} {'leak':>28} {'drive':>5} {'ridge':>7} "
          f"{'mc':>6} {'sPR':>6} {'gzR':>7} {'gzMC':>6} "
          f"{'d_gzR':>7} {'d_gzMC':>7} {'rail%':>6} {'pass':>4}")
    flipped = 0
    for r in rows:
        d_gz_r = r.get("gate_zero_ridge_nrmse", float("nan")) - r.get("ridge_nrmse", float("nan"))
        d_gz_mc = r.get("gate_zero_mc_total", float("nan")) - r.get("mc_total_washout_corrected", float("nan"))
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

# Supplementary S2b reference (printed only).
ref = load(ref_path)
if ref is not None:
    rows = ref.get("rows", []) or []
    sane = [r for r in rows if isinstance(r.get("rail_frac"), (int, float)) and r["rail_frac"] <= 0.05]
    if sane:
        best = max(sane, key=lambda r: r.get("mc_total_washout_corrected", float("-inf")))
        print(f"--- S2b reference {ref_path}")
        print(f"  corners={len(rows)} best rail-sane: gm={best.get('gm_init')} "
              f"leak={best.get('leak_mode')} drive={best.get('drive')} "
              f"ridge={best.get('ridge_nrmse')} mc={best.get('mc_total_washout_corrected')}")

# Trains: whichever arms ran + applicable VCA-vs-control deltas.
def read_trains(path):
    try:
        with open(path) as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        print(f"--- {path} (missing -- gated out or failed)")
        return []

legs = {}
for label, path in train_csvs.items():
    rows = read_trains(path)
    legs[label] = rows
    if not rows:
        continue
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

pf, v1, ps, v2 = (nrmse_of(legs[k]) for k in ("plain-full", "VCA-1", "plain-sparse", "VCA-2"))
pairs = (("VCA-1 - plain-full (VCA effect, full drive)", v1, pf),
         ("VCA-2 - plain-sparse (VCA effect, sparse drive)", v2, ps),
         ("VCA-2 - VCA-1 (sparse-drive effect under VCA)", v2, v1),
         ("plain-sparse - plain-full (sparse-drive effect, no VCA)", ps, pf))
for text, a, b in pairs:
    if a == a and b == b:
        print(f"delta {text}: {a - b:+.4f}")
PY
echo "=================================================================="
if [[ "${#FAILED_LEGS[@]}" -gt 0 ]]; then
  echo "[round2-gated] FAILED legs: ${FAILED_LEGS[*]}"
  exit 1
fi
echo "[round2-gated] all legs complete (gate winner: $FANOUT_WINNER)"
