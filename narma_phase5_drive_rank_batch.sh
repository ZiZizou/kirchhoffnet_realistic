#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=narma_p5_drive_rank
#SBATCH --output=%x-%j.out

# One targeted Phase-5 follow-up, not a sweep.
#
# The previous job proved that the nominal "moderate" configuration was at
# 1.3% of rail, so both cell types stayed linear.  This job fixes that
# experiment by (1) spectral-normalising the recurrent core, then (2)
# restoring an independently strong, heterogeneous boundary injection.
# It uses a full 25-node fan-out, heterogeneous core gm/A/B/sign, and
# per-node leaks around 0.1.  At most ONE calibrated retry is permitted:
# - <0.25 mean |x|/rail: increase boundary Isat from raw -4 to raw -2.
# - >0.60 mean |x|/rail or >20% rail occupancy: reduce it to raw -5.
# - otherwise: stop after the primary run.
#
# This batch also runs a deliberately nonphysical delay-bank crossbar control:
# x+ = .4x + .6*tanh(Wx + Win*[u[t],...,u[t-7]]).  It uses dense signed
# random W and independent signed input weights, with the same NARMA stream
# and diagnostics. A strong result means the task is sound and the next
# production experiment should be a crossbar + input-delay topology, not
# another OTA-cell variation.
#
# A scalar input necessarily has temporal rank one at the boundary.  The
# goal is not to make its 25 currents have rank 25; it is to give them a
# heterogeneous spatial injection vector that excites multiple recurrent
# modes with distinct time constants.  boundary_rms_cv records this.
#
# Submit: sbatch narma_phase5_drive_rank_batch.sh
# Overrides: OUT, ENV_DIR, DEVICE, SAMPLES, WASHOUT, SR_PASSES.
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
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing $PYTHON_BIN" >&2; exit 3; }
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

OUT="${OUT:-./output/narma_phase5_drive_rank}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-3000}"
WASHOUT="${WASHOUT:-200}"
SR_PASSES="${SR_PASSES:-15}"
mkdir -p "$OUT"

"$PYTHON_BIN" -m py_compile cell_library.py differential_stage.py topology.py narma_experiment.py narma_spectral_norm.py

"$PYTHON_BIN" - "$OUT" "$DEVICE" "$SAMPLES" "$WASHOUT" "$SR_PASSES" <<'PY' | tee "$OUT/drive_rank.log"
import json
import math
import sys
import torch

import narma_experiment as ne
from narma_advisor_probes import memory_capacity_washout
from cell_library import make_cell_library
from config import make_narma_preset
from topology import build_net_from_config

out_dir, device, samples, washout, sr_passes = sys.argv[1:]
samples, washout, sr_passes = int(samples), int(washout), int(sr_passes)


def softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))


def heterogenize_core(stage):
    """Break core symmetry while preserving the requested ~0.1 leak scale."""
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(510)
        lib = stage.cell_lib
        # Independent differential mixes and signed gains.  The clamp avoids
        # a few near-maximum edges dominating the spectrum before norm.
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.a_raw.device))
        lib.b_raw.copy_(torch.randn(lib.b_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.b_raw.device))
        lib.s_raw.copy_(torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5,
            torch.ones_like(lib.s_raw.cpu()), -torch.ones_like(lib.s_raw.cpu())).to(lib.s_raw.device))
        lib.gm_raw.copy_((-1.0 + .45 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2.5, .8).to(lib.gm_raw.device))
        lib.isat_raw.fill_(-4.0)
        # Log-uniform leak rates in [0.06, 0.16], centred near 0.1.  This is
        # heterogeneous leak; capacitance remains scalar in this architecture.
        rates = torch.empty(stage.raw_leak.shape).uniform_(math.log(.06), math.log(.16), generator=gen).exp()
        stage.raw_leak.copy_(softplus_inverse(rates).to(stage.raw_leak.device))
        stage.leak_mode = "programmable"


def configure_boundary(stage, isat_mean):
    """Make source injection strong and distinct, but almost state-independent.

    b_raw=-6 suppresses the boundary destination term.  Consequently,
    restoring this drive after core spectral normalisation does not
    materially change the recurrent Jacobian, while a/b/gm/Isat/sign vary
    across the 25 boundary edges and generate unequal currents.
    """
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(511)
        lib = stage.boundary_cell_lib
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1.0, 1.5).to(lib.a_raw.device))
        lib.b_raw.fill_(-6.0)
        lib.s_raw.copy_(torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5,
            torch.ones_like(lib.s_raw.cpu()), -torch.ones_like(lib.s_raw.cpu())).to(lib.s_raw.device))
        lib.gm_raw.copy_((-1.0 + .40 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2.0, .5).to(lib.gm_raw.device))
        lib.isat_raw.copy_((isat_mean + .30 * torch.randn(lib.isat_raw.shape, generator=gen)).clamp(-5.5, .0).to(lib.isat_raw.device))


def evaluate(tag, boundary_isat):
    net, t_span, n_steps = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library="linear_ota", core_refresh_interval=0,
        leak_constant=None, compile_sequence=False, hidden_dim=25,
        node_activation="tanh", spectral_radius_target=None,
    )
    stage = net.core.stages[0]
    heterogenize_core(stage)
    configure_boundary(stage, boundary_isat)
    # The normalizer currently visits all gm_raw libraries.  Keep a snapshot
    # of the intentionally strong boundary source drive and restore it after
    # normalising the recurrent fabric.
    boundary_saved = {name: getattr(stage.boundary_cell_lib, name).detach().clone()
                      for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw")}
    ne._apply_spectral_norm(net, .95, t_span, n_steps, tag=tag, max_passes=sr_passes)
    with torch.no_grad():
        for name, value in boundary_saved.items():
            getattr(stage.boundary_cell_lib, name).copy_(value)

    u_raw, y = ne.narma(samples, order=10, seed=0)
    # Never exceed input_scale=1: above it the ideal terminal itself enters
    # compliance gating.  Boundary current, not terminal voltage, is tuned.
    u = ne._scale_drive(u_raw, bipolar=True, order=10, input_scale=1.0)
    states = ne.collect_fabric_states(net, u.to(device), device=device).cpu()
    hidden = states[washout:, :25]
    cov = torch.cov(hidden.T) + 1e-8 * torch.eye(25)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = float(eig.sum().square() / eig.square().sum().clamp_min(1e-30))
    # Boundary-current amplitude diversity at the washed states.
    st_cpu = stage.cpu()
    xdst = st_cpu._node_broadcast(states[washout:])[:, st_cpu.boundary_dst]
    current = st_cpu.boundary_cell_lib(u[washout:, None][:, st_cpu.boundary_src], xdst, st_cpu.x_max)
    current = current * torch.sigmoid(st_cpu.boundary_z_logits)[None, :]
    rms = current.square().mean(0).sqrt()
    _, mc = memory_capacity_washout(states, u_raw, washout=washout)
    rail = float((hidden.abs() > .9 * stage.x_max).float().mean())
    report = net.spectral_norm_report["stages"][0]
    row = {
        "tag": tag, "boundary_isat_raw_mean": boundary_isat,
        "mc": float(mc), "state_pr": pr,
        "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max),
        "rail_frac": rail,
        "boundary_rms_cv": float(rms.std() / rms.mean().clamp_min(1e-12)),
        "boundary_current_nonzero_nodes": int((rms > 1e-6).sum()),
        "spectral_before": float(report["sr_before"]),
        "spectral_after": float(report["sr_after"]),
        "spectral_passes": int(report["passes"]),
    }
    print("RESULT " + json.dumps(row, sort_keys=True), flush=True)
    return row


def crossbar_delay_bank_control():
    """Unconstrained working-design control: delay-bank dense crossbar.

    State is normalized to [-1, 1], so ``mean_abs_over_bound`` and
    ``rail_frac`` have the same interpretation as the OTA fractions but
    use a unit bound. This deliberately supplies eight input delay channels;
    it has no KCL/passivity/compliance claim.
    """
    u_raw, y = ne.narma(samples, order=10, seed=0)
    n_nodes, n_taps = 25, 8
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(n_nodes, n_nodes, generator=gen)
    W = W * (.95 / torch.linalg.eigvals(W).abs().max().clamp_min(1e-9))
    W_in = torch.randn(n_nodes, n_taps, generator=gen)
    taps = torch.stack([
        u_raw if lag == 0 else torch.cat((torch.zeros(lag), u_raw[:-lag]))
        for lag in range(n_taps)
    ], dim=1)
    x = torch.zeros(n_nodes)
    states = []
    for t in range(samples):
        x = .4 * x + .6 * torch.tanh(W @ x + W_in @ taps[t])
        states.append(x)
    states = torch.stack(states)
    hidden = states[washout:]
    cov = torch.cov(hidden.T) + 1e-8 * torch.eye(25)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = float(eig.sum().square() / eig.square().sum().clamp_min(1e-30))
    _, mc = memory_capacity_washout(states, u_raw, washout=washout)
    X = torch.cat([states, torch.ones(samples, 1)], dim=1)
    ridge_l2 = 1e-2
    readout = torch.linalg.solve(
        X.T @ X + ridge_l2 * torch.eye(n_nodes + 1), X.T @ y,
    )
    pred = X @ readout
    row = {
        "tag": "relaxed_delay_bank_crossbar",
        "architecture": "x+=.4x+.6*tanh(Wx+Win@[u[t],...,u[t-7]])",
        "physical_realism": "intentionally relaxed; dense signed crossbar plus eight-tap input delay bank",
        "n_nodes": n_nodes, "n_input_taps": n_taps,
        "spectral_radius": .95, "leak": .60,
        "mc": float(mc), "state_pr": pr,
        "mean_abs_over_bound": float(hidden.abs().mean()),
        "rail_frac": float((hidden.abs() > .9).float().mean()),
        "ridge_nrmse": float(ne.nrmse(pred[washout:], y[washout:])),
        "ridge_r2": float(ne.r2(pred[washout:], y[washout:])),
        "win_rms": float(W_in.square().mean().sqrt()),
    }
    print("CROSSBAR_CONTROL " + json.dumps(row, sort_keys=True), flush=True)
    return row


def _delay_bank(u, n_taps):
    """Causal digital/analog-delay-line model: [u[t], ..., u[t-taps+1]]."""
    return torch.stack([
        u if lag == 0 else torch.cat((torch.zeros(lag), u[:-lag]))
        for lag in range(n_taps)
    ], dim=1)


def _ridge_features(features, target, washout, ridge_l2=1e-2):
    X = features[washout:]
    y = target[washout:]
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    W = torch.linalg.solve(
        X_aug.T @ X_aug + ridge_l2 * torch.eye(X_aug.shape[1]), X_aug.T @ y,
    )
    return float(ne.nrmse(X_aug @ W, y)), float(ne.r2(X_aug @ W, y))


def knet_delay_bank_control():
    """Real KNet/O​DE experiment with eight physical input-delay terminals.

    Every hidden KCL node receives one independently parameterized OTA from
    each delayed terminal.  This needs ``allow_boundary_target_overlap``:
    summing multiple current sources into a node is normal KCL behavior, not
    a digital crossbar shortcut.  The only digital block is the causal delay
    line, which can equally be implemented with sampled capacitors/buffers.
    """
    n_taps, hidden_dim = 8, 25
    preset = make_narma_preset(order=10, hidden_dim=hidden_dim,
        num_steps_per_sample=8, t_span=1.0, leak_constant=None)
    preset["stages"] = [dict(s, num_inputs=n_taps) for s in preset["stages"]]
    # Full input-bank fan-in: 8 delayed source terminals × 25 hidden KCL
    # nodes = 200 independent boundary OTAs.  This is the analog equivalent
    # of a delay-line crossbar and intentionally opts into current summation.
    boundary_map = {tap: list(range(hidden_dim)) for tap in range(n_taps)}
    net = build_net_from_config(
        cfg=preset, cell_lib=make_cell_library("linear_ota"),
        boundary_fan_out=boundary_map, allow_boundary_target_overlap=True,
        enable_temporal_readout=False, freeze_read=False,
        node_activation="tanh", allow_experimental_cells=True,
    )
    stage = net.core.stages[0]
    heterogenize_core(stage)
    configure_boundary(stage, -4.0)
    boundary_saved = {name: getattr(stage.boundary_cell_lib, name).detach().clone()
                      for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw")}
    ne._apply_spectral_norm(net, .95, 1.0, 8, tag="knet-delay-bank", max_passes=sr_passes)
    with torch.no_grad():
        for name, value in boundary_saved.items():
            getattr(stage.boundary_cell_lib, name).copy_(value)

    u_raw, y = ne.narma(samples, order=10, seed=0)
    # Each delay tap is a rail-scaled source voltage.  No future samples are
    # used, so this is a causal delay bank rather than target leakage.
    u_rail = ne._scale_drive(u_raw, bipolar=True, order=10, input_scale=1.0)
    taps_rail = _delay_bank(u_rail, n_taps)
    states = ne.collect_fabric_states(net, taps_rail, device=device).cpu()
    hidden = states[washout:, :hidden_dim]
    cov = torch.cov(hidden.T) + 1e-8 * torch.eye(hidden_dim)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = float(eig.sum().square() / eig.square().sum().clamp_min(1e-30))
    _, mc = memory_capacity_washout(states[:, :hidden_dim], u_raw, washout=washout)
    knet_nrmse, knet_r2 = _ridge_features(states[:, :hidden_dim], y, washout)
    delay_nrmse, delay_r2 = _ridge_features(_delay_bank(u_raw, n_taps), y, washout)
    # Measure the *net* KCL injection after the 200 source currents have
    # been summed into their 25 destination nodes.  A positive CV alone only
    # says edge magnitudes differ; this PR says whether the injected spatial
    # patterns actually span more than one useful direction over time.
    stage_cpu = stage.cpu()
    x_dst = stage_cpu._node_broadcast(states)[:, stage_cpu.boundary_dst]
    edge_current = stage_cpu.boundary_cell_lib(
        taps_rail[:, stage_cpu.boundary_src], x_dst, stage_cpu.x_max,
    ) * torch.sigmoid(stage_cpu.boundary_z_logits)[None, :]
    node_current = torch.zeros(samples, hidden_dim).index_add_(
        1, stage_cpu.boundary_dst, edge_current,
    )
    current_cov = torch.cov(node_current[washout:].T) + 1e-8 * torch.eye(hidden_dim)
    current_eig = torch.linalg.eigvalsh(current_cov).clamp_min(0)
    current_pr = float(
        current_eig.sum().square() / current_eig.square().sum().clamp_min(1e-30)
    )
    current_rms = node_current[washout:].square().mean(0).sqrt()
    report = net.spectral_norm_report["stages"][0]
    row = {
        "tag": "real_knet_linear_ota_delay_bank",
        "architecture": "8 causal delayed rail terminals -> 200 LinearOTA boundary currents -> recurrent KCL core",
        "physical_realism": "delay line is digital/analog plausible; boundary-current summation is KCL",
        "n_input_taps": n_taps, "n_boundary_otas": n_taps * hidden_dim,
        "mc": float(mc), "state_pr": pr,
        "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max),
        "rail_frac": float((hidden.abs() > .9 * stage.x_max).float().mean()),
        "boundary_injection_pr": current_pr,
        "boundary_injection_rms_cv": float(
            current_rms.std() / current_rms.mean().clamp_min(1e-12)
        ),
        "knet_ridge_nrmse": knet_nrmse, "knet_ridge_r2": knet_r2,
        "raw_delay_ridge_nrmse": delay_nrmse, "raw_delay_ridge_r2": delay_r2,
        "knet_beats_raw_delay_ridge": knet_nrmse < delay_nrmse,
        "spectral_before": float(report["sr_before"]),
        "spectral_after": float(report["sr_after"]),
        "spectral_passes": int(report["passes"]),
    }
    print("KNET_DELAY_BANK " + json.dumps(row, sort_keys=True), flush=True)
    return row


rows = [evaluate("primary", -4.0)]
primary = rows[0]
amp, rail = primary["mean_abs_over_rail"], primary["rail_frac"]
if amp < .25:
    print("CALIBRATION: under-driven; running one stronger-boundary retry", flush=True)
    rows.append(evaluate("retry_stronger_boundary", -2.0))
elif amp > .60 or rail > .20:
    print("CALIBRATION: over-driven; running one weaker-boundary retry", flush=True)
    rows.append(evaluate("retry_weaker_boundary", -5.0))
else:
    print("CALIBRATION: primary is in the target amplitude band; no retry", flush=True)

best = rows[-1]
verdict = {
    "amplitude_target_met": .30 <= best["mean_abs_over_rail"] <= .60,
    "rank_target_met": best["state_pr"] > 4.0,
    "memory_target_met": best["mc"] > 5.0,
    "rail_safe": best["rail_frac"] <= .20,
}
crossbar = crossbar_delay_bank_control()
delay_knet = knet_delay_bank_control()
crossbar_verdict = {
    "rank_target_met": crossbar["state_pr"] > 4.0,
    "memory_target_met": crossbar["mc"] > 5.0,
    "moderate_bound_activity": .30 <= crossbar["mean_abs_over_bound"] <= .60,
    "rail_safe": crossbar["rail_frac"] <= .20,
}
with open(f"{out_dir}/drive_rank_results.json", "w") as f:
    json.dump({"ota_rows": rows, "ota_verdict": verdict,
               "crossbar_control": crossbar,
               "crossbar_verdict": crossbar_verdict,
               "real_knet_delay_bank": delay_knet}, f, indent=2)
print("OTA_VERDICT " + json.dumps(verdict, sort_keys=True), flush=True)
print("CROSSBAR_VERDICT " + json.dumps(crossbar_verdict, sort_keys=True), flush=True)
PY

echo "Results: $OUT/drive_rank_results.json"
