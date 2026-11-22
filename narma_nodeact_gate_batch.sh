#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=narma_nodeact_gate
#SBATCH --output=%x-%j.out

# narma-node-activation Step-3 gate batch (Alliance H100, gitignored: ship with `git add -f`).
#
# Phase A (E0 gate, GPU-eager, minutes): three arms on seed 0, full 3000-sample
#   stream, washout 200 --
#     legacy    : defaults (node none, full-rail drive)
#     proposal  : node tanh + gm -1 / isat -5 / leak 0.1 / drive 0.2 + sr 0.95
#     identity  : same as proposal with node identity (Step-4 bounding arm;
#                 clip is provably vacuous here -- rail 0% -- so no --no-clip
#                 leg is needed for the decision table)
#   PASS = proposal MC > 5 AND PR > 4 AND rails < 10% (plan gate).
# Phase B (only on PASS): 40-epoch trains of proposal (seeds) + legacy seed 0.
#
# Usage: sbatch narma_nodeact_gate_batch.sh   (or: bash narma_nodeact_gate_batch.sh)
# Env overrides: OUT, EPOCHS, SEEDS, LEGACY_SEEDS, TIME, ACCOUNT, GPU.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc

source "${SLURM_SUBMIT_DIR}/ENV/bin/activate"
PYTHON_BIN="${SLURM_SUBMIT_DIR}/ENV/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: expected Python executable is missing or not executable: $PYTHON_BIN" >&2
  exit 3
fi
echo "python=$PYTHON_BIN"

# Fail fast if the GPU Python environment is incomplete.
"$PYTHON_BIN" -c "import torch, triton; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} triton={triton.__version__}')" \
    || { echo "ERROR: torch/triton import failed. Aborting before any GPU work." >&2; exit 3; }
if ! "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: CUDA is not available to torch. Aborting before any GPU work." >&2
  exit 3
fi

OUT="${OUT:-./output/narma_nodeact_gate}"
EPOCHS="${EPOCHS:-40}"
SEEDS="${SEEDS:-0,1,2}"
LEGACY_SEEDS="${LEGACY_SEEDS:-0}"
DEVICE=cuda

mkdir -p "$OUT"
echo "=== narma-nodeact gate batch ==="
echo "out=$OUT epochs=$EPOCHS seeds=$SEEDS device=$DEVICE"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# ---- Preflight (fatal): compile + flag surface + lightweight audit ----
"$PYTHON_BIN" -m py_compile \
    cell_library.py config.py differential_stage.py kirchhoff_net.py \
    narma_advisor_probes.py narma_experiment.py narma_spectral_norm.py \
    probe_audit_nodeact.py topology.py || exit 11
"$PYTHON_BIN" narma_experiment.py --help | grep -q -- --node-activation || exit 12
"$PYTHON_BIN" narma_experiment.py --help | grep -q -- --spectral-radius-target || exit 13
"$PYTHON_BIN" narma_experiment.py --help | grep -q -- --isat-init || exit 14
"$PYTHON_BIN" probe_audit_nodeact.py | tee "$OUT/preflight_audit.log" | tail -3
grep -q ALL_AUDITS_OK "$OUT/preflight_audit.log" || exit 15
echo "PREFLIGHT_OK"

# ---- Phase A: E0 gate (exact train-path fill order, GPU) ----
# NOTE: gate FAIL exits 2 by design (plan: stop and reassess, no trains).
# `set +e` + PIPESTATUS keeps the log flush before we decide Phase B.
set +e
"$PYTHON_BIN" - "$OUT" "$DEVICE" <<'EOF' | tee "$OUT/e0_gate.log"
import sys
out_dir, device = sys.argv[1], sys.argv[2]
import torch
import narma_experiment as ne
from narma_advisor_probes import memory_capacity_washout
from narma_experiment import _scale_drive

XMAX = 3.0
torch.manual_seed(0)
rows = []

def arm(tag, *, node_act, fills, leak_const, input_scale, sr_target):
    net, ts, ns = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library="tanh_free", core_refresh_interval=0,
        leak_constant=leak_const, compile_sequence=False, hidden_dim=25,
        node_activation=node_act, spectral_radius_target=None,
    )
    if fills:
        gm_f, is_f = fills
        with torch.no_grad():
            for st in net.core.stages:
                for ln in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
                    lib = getattr(st, ln, None)
                    if lib is None:
                        continue
                    if hasattr(lib, "gm_raw"):
                        lib.gm_raw.data.fill_(gm_f)
                    if hasattr(lib, "isat_raw"):
                        lib.isat_raw.data.fill_(is_f)
    sr = None
    if sr_target is not None:
        ne._apply_spectral_norm(net, sr_target, ts, ns, tag=tag, max_passes=15)
        r0 = net.spectral_norm_report["stages"][0]
        sr = (round(r0["sr_before"], 4), round(r0["sr_after"], 4), r0["passes"])
    u, y = ne.narma(3000, order=10, seed=0)
    u_s = _scale_drive(u, bipolar=True, order=10, input_scale=input_scale)
    states = ne.collect_fabric_states(net, u_s.to(device), device=device).cpu()
    _, mc = memory_capacity_washout(states, u, washout=200)
    ridge = ne.ridge_readout_diagnostic(net, u_s, y, washout=200, device=device)
    w = states[200:]
    # PR is defensive: rank-collapsed (dead-node) states make the raw
    # covariance singular and eigh may not converge (CPU and CUDA alike).
    # Diagonal jitter keeps the instrument alive; NaN fallback never kills
    # the gate (MC/ridge/rails still decide).
    try:
        cov = torch.cov(w.T) + 1e-6 * torch.eye(w.shape[1])
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
    except Exception as e:
        print(f"  WARNING: PR failed ({e}); recording NaN", flush=True)
        pr = float("nan")
    rail = float((w.abs() > 0.9 * XMAX).float().mean())
    sat = float(w.abs().max() / XMAX)
    row = {"arm": tag, "mc": round(float(mc), 3),
           "ridge": round(float(ridge["ridge_nrmse"]), 4),
           "pr": round(pr, 2), "rail_pct": round(rail * 100, 2),
           "sat": round(sat, 3), "sr": sr}
    import json as _json
    rows.append(row)
    print(f"E0 {tag}: MC={row['mc']} ridge={row['ridge']} PR={row['pr']} "
          f"rail={row['rail_pct']}% sat={row['sat']} sr={sr}", flush=True)

arm("legacy", node_act="none", fills=None, leak_const=None,
    input_scale=1.0, sr_target=None)
arm("proposal", node_act="tanh", fills=(-1.0, -5.0), leak_const=0.1,
    input_scale=0.2, sr_target=0.95)
arm("identity", node_act="identity", fills=(-1.0, -5.0), leak_const=0.1,
    input_scale=0.2, sr_target=0.95)

import json
with open(f"{out_dir}/e0_gate.json", "w") as f:
    json.dump(rows, f, indent=2)
prop = rows[1]
verdict = (prop["mc"] > 5 and prop["pr"] > 4 and prop["rail_pct"] < 10)
with open(f"{out_dir}/e0_verdict.txt", "w") as f:
    f.write(f"proposal MC={prop['mc']} PR={prop['pr']} rail={prop['rail_pct']}% "
            f"-> {'PASS' if verdict else 'FAIL'}\n")
print(f"GATE_VERDICT={'PASS' if verdict else 'FAIL'}", flush=True)
sys.exit(0 if verdict else 2)
EOF
GATE_RC=${PIPESTATUS[0]}
set -e
echo "gate exit: $GATE_RC"
if [ -f "$OUT/e0_verdict.txt" ]; then
    cat "$OUT/e0_verdict.txt"
else
    echo "GATE_VERDICT=ERROR: E0 gate crashed before writing e0_verdict.txt" >&2
    echo "See $OUT/e0_gate.log for the original Python traceback." >&2
fi
if [ "$GATE_RC" -ne 0 ]; then
    echo "GATE FAIL: plan says stop and reassess -- no training legs. See $OUT/e0_gate.json."
    exit 2
fi

# ---- Phase B: 40-epoch trains (proposal seeds + legacy control) ----
"$PYTHON_BIN" narma_experiment.py \
    --order 10 --seeds "$SEEDS" --cell-library tanh_free \
    --t-span 1.0 --num-steps 8 \
    --node-activation tanh --gain-init -1 --isat-init -5 --leak-init 0.1 \
    --input-scale 0.2 --spectral-radius-target 0.95 \
    --epochs "$EPOCHS" --device "$DEVICE" --no-progress --ridge-diagnostic \
    --output "$OUT/trains_proposal" 2>&1 | tee "$OUT/trains_proposal.log"

"$PYTHON_BIN" narma_experiment.py \
    --order 10 --seeds "$LEGACY_SEEDS" --cell-library tanh_free \
    --t-span 1.0 --num-steps 8 \
    --epochs "$EPOCHS" --device "$DEVICE" --no-progress --ridge-diagnostic \
    --output "$OUT/trains_legacy" 2>&1 | tee "$OUT/trains_legacy.log"

echo "=== batch complete ==="
tail -5 "$OUT/trains_proposal.log"
tail -5 "$OUT/trains_legacy.log"
