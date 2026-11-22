#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=narma_phase5
#SBATCH --output=%x-%j.out

# Phase 5: diagnose rank collapse before spending a training budget.
#
# This is deliberately an *evaluation-only*, six-leg experiment--not an
# operating-point sweep.  Each leg collects MC, PR, rails, covariance
# geometry, and the per-node boundary-current geometry on the same NARMA-10
# stream.  The two LinearOTA legs distinguish a linear mix of bounded states
# from FreeTanh's independently saturated edge contributions; the diversity
# legs isolate recurrent-gain and input-drive rank.
#
#   5a  rung1_free_small: current small-signal FreeTanh correlation probe.
#   5b  free_moderate / linear_ota_moderate: matched structural comparison.
#   5c  free_hetero_core: random per-core-edge gm_raw initialization.
#   5d  free_sparse_input: sparse fan-out + random boundary gm_raw.
#   D1  linear_ota_diverse: LinearOTA + both diversity levers.
#
# Submit: sbatch narma_phase5_batch.sh
# Overrides: OUT, ENV_DIR, DEVICE, SAMPLES, WASHOUT, SR_PASSES, SKIP_DONE.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc

ENV_DIR="${ENV_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/ENV}"
source "$ENV_DIR/bin/activate"
PYTHON_BIN="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: missing cluster venv Python: $PYTHON_BIN" >&2
  exit 3
fi
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print('torch=', torch.__version__, 'cuda=', torch.cuda.get_device_name(0))"

OUT="${OUT:-./output/narma_phase5}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-3000}"
WASHOUT="${WASHOUT:-200}"
SR_PASSES="${SR_PASSES:-15}"
SKIP_DONE="${SKIP_DONE:-1}"
mkdir -p "$OUT"

if [[ "$SKIP_DONE" == "1" && -f "$OUT/phase5_results.json" ]]; then
  echo "Phase-5 results already exist: $OUT/phase5_results.json"
  exit 0
fi

"$PYTHON_BIN" -m py_compile cell_library.py config.py differential_stage.py topology.py narma_experiment.py narma_spectral_norm.py
"$PYTHON_BIN" -c "from cell_library import make_cell_library; assert type(make_cell_library('linear_ota')).__name__ == 'LinearOTALibrary'"

echo "=== Phase 5 rank/cell diagnostics ==="
echo "out=$OUT device=$DEVICE samples=$SAMPLES washout=$WASHOUT"

"$PYTHON_BIN" - "$OUT" "$DEVICE" "$SAMPLES" "$WASHOUT" "$SR_PASSES" <<'PY' | tee "$OUT/phase5.log"
import json
import math
import sys

import torch

import narma_experiment as ne
from narma_advisor_probes import memory_capacity_washout

out_dir, device, samples, washout, sr_passes = sys.argv[1:]
samples, washout, sr_passes = int(samples), int(washout), int(sr_passes)
torch.manual_seed(0)


def _fill(lib, gm, isat, *, hetero=False, seed=0):
    if lib is None:
        return
    with torch.no_grad():
        if hasattr(lib, "gm_raw"):
            if hetero:
                gen = torch.Generator(device=lib.gm_raw.device).manual_seed(seed)
                # Deliberately bounded: break the uniform eigenspectrum without
                # manufacturing a few near-max-gm edges.
                lib.gm_raw.copy_((gm + 0.75 * torch.randn(lib.gm_raw.shape,
                    generator=gen, device=lib.gm_raw.device)).clamp(-3.0, 1.5))
            else:
                lib.gm_raw.fill_(gm)
        if hasattr(lib, "isat_raw"):
            lib.isat_raw.fill_(isat)
        if hasattr(lib, "g_resistive_raw"):
            lib.g_resistive_raw.fill_(-20.0)


def _mean_pair_corr(values):
    """Mean off-diagonal Pearson correlation; robust to dead nodes."""
    z = values - values.mean(0, keepdim=True)
    denom = z.square().sum(0).sqrt().clamp_min(1e-12)
    corr = (z.T @ z) / (denom[:, None] * denom[None, :])
    n = corr.shape[0]
    return float((corr.sum() - torch.diagonal(corr).sum()) / max(n * (n - 1), 1))


def _pr(cov):
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(0)
    return float(eig.sum().square() / eig.square().sum().clamp_min(1e-30))


def _boundary_accumulator(stage, states, u):
    """Per-node boundary injection matching the dynamic RHS semantics."""
    x = states
    u_b = u[:, None]
    x_dst = x[:, stage.boundary_dst]
    if stage.node_activation == "tanh":
        x_dst = stage._node_broadcast(x)[:, stage.boundary_dst]
    i = stage.boundary_cell_lib(
        u_b[:, stage.boundary_src], x_dst, stage.x_max,
    ) * torch.sigmoid(stage.boundary_z_logits)[None, :]
    acc = torch.zeros_like(x)
    acc.index_add_(1, stage.boundary_dst, i)
    return acc


def run_case(tag, *, library, node_act, gm, isat, leak, drive,
             core_hetero=False, sparse=False):
    # One input channel means sparse fan-out alone changes injection support,
    # not its input rank.  Random boundary gm is included with the sparse leg
    # so the injected current vectors are genuinely non-identical.
    fanout = None
    if sparse:
        fanout = {0: [0, 3, 6, 9, 12, 15, 18, 21]}
    net, t_span, n_steps = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library=library, core_refresh_interval=0, leak_constant=leak,
        compile_sequence=False, hidden_dim=25, node_activation=node_act,
        boundary_fan_out=fanout, spectral_radius_target=None,
    )
    stage = net.core.stages[0]
    _fill(stage.cell_lib, gm, isat, hetero=core_hetero, seed=101)
    _fill(stage.boundary_cell_lib, gm, isat, hetero=sparse, seed=202)
    _fill(stage.output_ode_cell_lib, gm, isat, seed=303)
    ne._apply_spectral_norm(net, 0.95, t_span, n_steps, tag=tag,
                            max_passes=sr_passes)

    u_raw, y = ne.narma(samples, order=10, seed=0)
    u = ne._scale_drive(u_raw, bipolar=True, order=10, input_scale=drive)
    states = ne.collect_fabric_states(net, u.to(device), device=device).cpu()
    hidden = states[washout:, :25]
    cov = torch.cov(hidden.T) + 1e-8 * torch.eye(25)
    evals, evecs = torch.linalg.eigh(cov)
    lead = evecs[:, -1].abs()
    bacc = _boundary_accumulator(stage.cpu(), states[washout:, :], u[washout:])[:, :25]
    b_rms = bacc.square().mean(0).sqrt()
    b_cov = torch.cov(bacc.T) + 1e-8 * torch.eye(25)
    _, mc = memory_capacity_washout(states, u_raw, washout=washout)
    ridge = ne.ridge_readout_diagnostic(net, u, y, washout=washout, device=device)
    rail = float((hidden.abs() > 0.9 * stage.x_max).float().mean())
    sr = net.spectral_norm_report["stages"][0]
    row = {
        "tag": tag, "library": library, "node_activation": node_act,
        "core_hetero_gm": core_hetero, "sparse_input": sparse,
        "gm_raw": gm, "isat_raw": isat, "leak": leak, "input_scale": drive,
        "mc": float(mc), "ridge_nrmse": float(ridge["ridge_nrmse"]),
        "state_pr": _pr(cov), "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max),
        "rail_frac": rail, "state_mean_pair_corr": _mean_pair_corr(hidden),
        "state_diag_cv": float(torch.diagonal(cov).std() / torch.diagonal(cov).mean().clamp_min(1e-12)),
        "lead_effective_support": float(1.0 / lead.pow(4).sum().clamp_min(1e-30)),
        "boundary_pr": _pr(b_cov),
        "boundary_mean_pair_corr": _mean_pair_corr(bacc),
        "boundary_rms_cv": float(b_rms.std() / b_rms.mean().clamp_min(1e-12)),
        "spectral_before": float(sr["sr_before"]), "spectral_after": float(sr["sr_after"]),
        "spectral_passes": int(sr["passes"]),
        # Preserve the raw diagnostic objects for post-hoc inspection, not
        # just their scalar summaries.  These are 25x25 by construction.
        "state_covariance": cov.tolist(), "boundary_current_covariance": b_cov.tolist(),
        "leading_state_eigenvector_abs": lead.tolist(),
    }
    print("RESULT " + json.dumps(row, sort_keys=True), flush=True)
    return row


rows = []
# 5a: current small-signal control. Keep this separate from moderate tests.
rows.append(run_case("5a_rung1_free_small", library="tanh_free", node_act="none",
    gm=-8.0, isat=-2.0, leak=0.1, drive=0.2))
# 5b: matched pair at the moderate operating point where cells can diverge.
rows.append(run_case("5b_free_moderate", library="tanh_free", node_act="tanh",
    gm=-1.0, isat=-4.0, leak=0.1, drive=1.0))
rows.append(run_case("5b_linear_ota_moderate", library="linear_ota", node_act="tanh",
    gm=-1.0, isat=-4.0, leak=0.1, drive=1.0))
# 5c / 5d: isolate recurrent diversity and drive geometry.
rows.append(run_case("5c_free_hetero_core", library="tanh_free", node_act="tanh",
    gm=-1.0, isat=-4.0, leak=0.1, drive=1.0, core_hetero=True))
rows.append(run_case("5d_free_sparse_input", library="tanh_free", node_act="tanh",
    gm=-1.0, isat=-4.0, leak=0.1, drive=1.0, sparse=True))
# Decisive combined diversity control: the best diagnostic/performance leg.
rows.append(run_case("d1_linear_ota_diverse", library="linear_ota", node_act="tanh",
    gm=-1.0, isat=-4.0, leak=0.1, drive=1.0, core_hetero=True, sparse=True))

with open(f"{out_dir}/phase5_results.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\n=== Phase 5 summary ===")
for r in rows:
    print(f"{r['tag']:27s} MC={r['mc']:.3f} PR={r['state_pr']:.2f} "
          f"|x|/rail={r['mean_abs_over_rail']:.3f} rail={100*r['rail_frac']:.2f}% "
          f"state_corr={r['state_mean_pair_corr']:.3f} boundary_PR={r['boundary_pr']:.2f} "
          f"boundary_corr={r['boundary_mean_pair_corr']:.3f}")
PY

echo "=== Phase 5 complete ==="
echo "Results: $OUT/phase5_results.json"
