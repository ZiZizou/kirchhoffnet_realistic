#!/bin/bash
#SBATCH --account=def-shiyusu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=narma_p6_core_eigen
#SBATCH --output=%x-%j.out

# Phase 6: real KNet delay-bank experiment, with the input mechanism fixed.
#
# Phase 5 established that eight causal delay terminals give a useful KCL
# injection (injection PR about 6), but the recurrent state collapsed to PR
# about 1.9.  This batch therefore changes only the recurrent core topology
# and its edge/leak realization.  It deliberately does NOT add VCA or another
# boundary-symmetry mechanism.
#
# The drive is first calibrated on the legacy directed 5x5 torus, selecting
# from a small lower-drive ladder the setting closest to 0.50 mean |x| / rail.
# The selected *same* boundary drive is then used for every core condition;
# that makes topology comparisons interpretable.  The core sweep compares:
#   - the legacy directed 5x5 torus,
#   - a true bidirectional 5x5 torus,
#   - degree-matched Watts-Strogatz small-world cores (p=0.25 and 0.50), and
#   - a bidirectional degree-matched small-world core (p=0.25).
#
# In addition to NARMA ridge/MC/state PR, the batch reports the zero-input
# one-sample Jacobian spectrum, a finite-horizon controllability-Gramian PR
# from the eight delay terminals, and local driven-Jacobian summaries.  Thus
# a result can distinguish "input has many directions" from "the core keeps
# those directions controllable while driven".
#
# Submit: sbatch narma_phase6_core_eigen_batch.sh
# Overrides: OUT, ENV_DIR, DEVICE, SAMPLES, WASHOUT, SR_PASSES, CORE_SEEDS,
#            N_LOCAL_JAC, DRIVE_CANDIDATES.
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

OUT="${OUT:-./output/narma_phase6_core_eigen}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-3000}"
WASHOUT="${WASHOUT:-200}"
# The Phase-5 normalizer reached 0.996 after 15 passes.  Use more passes and
# record the measured landing point; no row is treated as a nominal 0.95.
SR_PASSES="${SR_PASSES:-30}"
CORE_SEEDS="${CORE_SEEDS:-0,1,2}"
N_LOCAL_JAC="${N_LOCAL_JAC:-16}"
# Isat raw -4.0 gave 0.689 in the Phase-5 delay-bank KNet.  These settings
# lower the current scale without altering the relative 200-edge injection.
DRIVE_CANDIDATES="${DRIVE_CANDIDATES:--4.2,-4.4,-4.6}"
mkdir -p "$OUT"

"$PYTHON_BIN" -m py_compile cell_library.py differential_stage.py topology.py narma_experiment.py narma_spectral_norm.py

"$PYTHON_BIN" - "$OUT" "$DEVICE" "$SAMPLES" "$WASHOUT" "$SR_PASSES" "$CORE_SEEDS" "$N_LOCAL_JAC" "$DRIVE_CANDIDATES" <<'PY' | tee "$OUT/core_eigen.log"
import json
import math
import os
import sys
from collections import defaultdict

import torch

import narma_experiment as ne
from cell_library import make_cell_library
from config import make_narma_preset
from narma_advisor_probes import _one_sample_transition, memory_capacity_washout
from narma_spectral_norm import _jacobian_at_zero
from topology import build_net_from_config


(out_dir, device, samples, washout, sr_passes, core_seeds_raw,
 n_local_jac, drive_candidates_raw) = sys.argv[1:]
samples = int(samples)
washout = int(washout)
sr_passes = int(sr_passes)
n_local_jac = int(n_local_jac)
core_seeds = [int(x) for x in core_seeds_raw.split(",") if x.strip()]
drive_candidates = [float(x) for x in drive_candidates_raw.split(",") if x.strip()]
if not core_seeds:
    raise ValueError("CORE_SEEDS must contain at least one integer")
if not drive_candidates:
    raise ValueError("DRIVE_CANDIDATES must contain at least one raw Isat value")

N_TAPS = 8
HIDDEN_DIM = 25
T_SPAN = 1.0
N_STEPS = 8
SR_TARGET = 0.95
GRAMIAN_HORIZON = 20


def softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))


def delay_bank(u, n_taps=N_TAPS):
    """Causal taps [u[t], ..., u[t-n_taps+1]] with zero initial history."""
    return torch.stack([
        u if lag == 0 else torch.cat((u.new_zeros(lag), u[:-lag]))
        for lag in range(n_taps)
    ], dim=1)


def pr_from_eigs(eigs):
    eigs = eigs.real.clamp_min(0)
    return float(eigs.sum().square() / eigs.square().sum().clamp_min(1e-30))


def covariance_pr(x):
    # x is CPU for collected states but CUDA for the input Jacobian in the
    # controllability probe.  Keep the regularizer co-located with x.
    cov = torch.cov(x.T) + 1e-8 * torch.eye(
        x.shape[1], dtype=x.dtype, device=x.device
    )
    return pr_from_eigs(torch.linalg.eigvalsh(cov))


def heterogenize_core(stage, seed):
    """Same core parameter family as Phase 5, with a reproducible seed."""
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(510 + int(seed))
        lib = stage.cell_lib
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.a_raw.device))
        lib.b_raw.copy_(torch.randn(lib.b_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.b_raw.device))
        signs = torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5,
                            torch.ones_like(lib.s_raw.cpu()), -torch.ones_like(lib.s_raw.cpu()))
        lib.s_raw.copy_(signs.to(lib.s_raw.device))
        lib.gm_raw.copy_((-1.0 + .45 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2.5, .8).to(lib.gm_raw.device))
        lib.isat_raw.fill_(-4.0)
        rates = torch.empty(stage.raw_leak.shape).uniform_(math.log(.06), math.log(.16), generator=gen).exp()
        stage.raw_leak.copy_(softplus_inverse(rates).to(stage.raw_leak.device))
        stage.leak_mode = "programmable"


def configure_boundary(stage, isat_mean):
    """Fixed heterogeneous delay-bank injection; only its common scale moves."""
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(511)
        lib = stage.boundary_cell_lib
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1.0, 1.5).to(lib.a_raw.device))
        # Suppress destination dependence, so input calibration does not
        # itself redesign the recurrent Jacobian.
        lib.b_raw.fill_(-6.0)
        signs = torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5,
                            torch.ones_like(lib.s_raw.cpu()), -torch.ones_like(lib.s_raw.cpu()))
        lib.s_raw.copy_(signs.to(lib.s_raw.device))
        lib.gm_raw.copy_((-1.0 + .40 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2.0, .5).to(lib.gm_raw.device))
        lib.isat_raw.copy_((float(isat_mean) + .30 * torch.randn(lib.isat_raw.shape, generator=gen)).clamp(-6.0, .0).to(lib.isat_raw.device))


TOPOLOGIES = (
    {
        "tag": "torus_directed",
        "family": "torus",
        "kwargs": {"height": 5, "width": 5, "kernel_size": 3, "bidirectional": False},
    },
    {
        "tag": "torus_bidirectional",
        "family": "torus",
        "kwargs": {"height": 5, "width": 5, "kernel_size": 3, "bidirectional": True},
    },
    # k=8 gives 25*8/2 = 100 undirected branches, matching the directed
    # 5x5 kernel-3 torus.  p changes only the long-range rewiring fraction.
    {
        "tag": "small_world_k8_p025",
        "family": "small_world",
        "kwargs": {"k": 8, "p": .25, "bidirectional": False},
    },
    {
        "tag": "small_world_k8_p050",
        "family": "small_world",
        "kwargs": {"k": 8, "p": .50, "bidirectional": False},
    },
    {
        "tag": "small_world_k8_p025_bidirectional",
        "family": "small_world",
        "kwargs": {"k": 8, "p": .25, "bidirectional": True},
    },
)


def build_delay_knet(topology, core_seed, boundary_isat):
    """Build exactly the physical KCL delay-bank path used in Phase 5."""
    preset = make_narma_preset(order=10, hidden_dim=HIDDEN_DIM,
                               num_steps_per_sample=N_STEPS, t_span=T_SPAN,
                               leak_constant=None)
    topo_kwargs = dict(topology["kwargs"])
    if topology["family"] == "small_world":
        # New deterministic graph realization per core seed, not per drive
        # calibration candidate.  This prevents a topology seed from being an
        # unreported source of variation.
        topo_kwargs["seed"] = 1000 + int(core_seed)
    preset["stages"] = [dict(
        preset["stages"][0], num_inputs=N_TAPS,
        hidden_family=topology["family"], hidden_kwargs=topo_kwargs,
    )]
    boundary_map = {tap: list(range(HIDDEN_DIM)) for tap in range(N_TAPS)}
    net = build_net_from_config(
        cfg=preset, cell_lib=make_cell_library("linear_ota"),
        boundary_fan_out=boundary_map, allow_boundary_target_overlap=True,
        enable_temporal_readout=False, freeze_read=False,
        node_activation="tanh", allow_experimental_cells=True,
    )
    stage = net.core.stages[0]
    heterogenize_core(stage, core_seed)
    configure_boundary(stage, boundary_isat)
    # Normalize the recurrent fabric only.  The normalizer presently visits
    # every gm library, so save and restore the fixed boundary injection.
    boundary_saved = {
        name: getattr(stage.boundary_cell_lib, name).detach().clone()
        for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw")
    }
    ne._apply_spectral_norm(net, SR_TARGET, T_SPAN, N_STEPS,
                            tag=f"{topology['tag']}-seed{core_seed}",
                            max_passes=sr_passes)
    with torch.no_grad():
        for name, value in boundary_saved.items():
            getattr(stage.boundary_cell_lib, name).copy_(value)
    return net


def ridge_features(features, target):
    x = features[washout:]
    y = target[washout:].to(device=x.device, dtype=x.dtype)
    x_aug = torch.cat([x, x.new_ones(x.shape[0], 1)], dim=1)
    w = torch.linalg.solve(
        x_aug.T @ x_aug + 1e-2 * torch.eye(
            x_aug.shape[1], dtype=x_aug.dtype, device=x_aug.device
        ), x_aug.T @ y,
    )
    pred = x_aug @ w
    return float(ne.nrmse(pred, y)), float(ne.r2(pred, y))


def spectrum_metrics(j):
    vals = torch.linalg.eigvals(j)
    mag = vals.abs()
    commutator = j.T @ j - j @ j.T
    return {
        "jac_spectral_radius": float(mag.max()),
        "jac_mean_abs_eig": float(mag.mean()),
        "jac_eig_pr": float(mag.sum().square() / mag.square().sum().clamp_min(1e-30)),
        "jac_modes_gt_060": int((mag > .60).sum()),
        "jac_modes_gt_080": int((mag > .80).sum()),
        "jac_complex_modes": int((vals.imag.abs() > 1e-5).sum()),
        "jac_nonnormality": float(
            torch.linalg.vector_norm(commutator) /
            torch.linalg.vector_norm(j).square().clamp_min(1e-30)
        ),
    }


def controllability_metrics(stage):
    """Finite-horizon Wc for the actual one-sample map and eight tap inputs."""
    stage_device = next(stage.parameters()).device
    j = _jacobian_at_zero(stage, t_span=T_SPAN, num_steps=N_STEPS,
                          device=str(stage_device))
    x0 = torch.zeros(HIDDEN_DIM, device=stage_device)
    u0 = torch.zeros(N_TAPS, device=stage_device)
    dt = T_SPAN / N_STEPS

    def input_map(u_flat):
        return _one_sample_transition(stage, x0.view(1, -1),
                                      u_flat.view(1, -1), dt, N_STEPS)

    b = torch.autograd.functional.jacobian(input_map, u0,
                                           create_graph=False)
    wc = torch.zeros_like(j)
    power = torch.eye(HIDDEN_DIM, dtype=j.dtype, device=j.device)
    bb_t = b @ b.T
    for _ in range(GRAMIAN_HORIZON):
        wc = wc + power @ bb_t @ power.T
        power = j @ power
    gram_eigs = torch.linalg.eigvalsh((wc + wc.T) * .5)
    out = spectrum_metrics(j)
    out.update({
        "input_jacobian_pr": covariance_pr(b.T),
        "controllability_pr": pr_from_eigs(gram_eigs),
        "controllability_rank_gt_1e6": int((gram_eigs > 1e-6).sum()),
    })
    return out


def driven_jacobian_metrics(stage, states, taps):
    """Summarize local transition spectra at washed, actually driven states."""
    if n_local_jac <= 0:
        return {}
    stage_device = next(stage.parameters()).device
    indices = torch.linspace(washout, states.shape[0] - 2,
                             steps=min(n_local_jac, states.shape[0] - washout - 1))
    indices = indices.round().to(torch.long).unique().tolist()
    dt = T_SPAN / N_STEPS
    metrics = []
    for index in indices:
        x_from = states[index].to(stage_device).detach().clone().requires_grad_(True)
        u_next = taps[index + 1].to(stage_device).view(1, -1)

        def transition_map(x_flat):
            return _one_sample_transition(stage, x_flat.view(1, -1),
                                          u_next, dt, N_STEPS)

        j = torch.autograd.functional.jacobian(transition_map, x_from,
                                               create_graph=False)
        metrics.append(spectrum_metrics(j))
    keys = ("jac_spectral_radius", "jac_mean_abs_eig", "jac_eig_pr",
            "jac_modes_gt_060", "jac_modes_gt_080", "jac_complex_modes",
            "jac_nonnormality")
    return {f"driven_{key}_mean": float(sum(row[key] for row in metrics) / len(metrics))
            for key in keys} | {"n_driven_jacobians": len(metrics)}


u_raw, y = ne.narma(samples, order=10, seed=0)
u_rail = ne._scale_drive(u_raw, bipolar=True, order=10, input_scale=1.0)
taps_rail = delay_bank(u_rail)


def collect_for(topology, core_seed, boundary_isat):
    net = build_delay_knet(topology, core_seed, boundary_isat)
    # collect_fabric_states is decorated with torch.inference_mode().  Moving
    # a CPU-built module to CUDA *inside* that context turns its buffers into
    # inference tensors, which cannot subsequently participate in the
    # autograd Jacobian/Gramian probes below.  Move it here, in normal mode;
    # the no-op .to(device) inside collect_fabric_states then preserves normal
    # autograd-capable parameters and buffers.
    net.to(device)
    states = ne.collect_fabric_states(net, taps_rail, device=device).cpu()
    return net, states


# Calibrate once on the original directed torus, then freeze this boundary
# setting for the core study.  The ranking penalizes rail-heavy candidates.
calibration = []
for raw_isat in drive_candidates:
    net, states = collect_for(TOPOLOGIES[0], core_seeds[0], raw_isat)
    hidden = states[washout:, :HIDDEN_DIM]
    row = {
        "boundary_isat_raw_mean": raw_isat,
        "mean_abs_over_rail": float(hidden.abs().mean() / net.core.stages[0].x_max),
        "rail_frac": float((hidden.abs() > .9 * net.core.stages[0].x_max).float().mean()),
    }
    calibration.append(row)
    print("DRIVE_CALIBRATION " + json.dumps(row, sort_keys=True), flush=True)


def calibration_score(row):
    # Target the middle of the requested 0.45--0.55 range, but strongly
    # reject a nominally-close candidate that reaches the rails too often.
    return abs(row["mean_abs_over_rail"] - .50) + 2.0 * max(0.0, row["rail_frac"] - .20)


chosen_calibration = min(calibration, key=calibration_score)
chosen_isat = chosen_calibration["boundary_isat_raw_mean"]
print("DRIVE_SELECTED " + json.dumps(chosen_calibration, sort_keys=True), flush=True)

rows = []
for topology in TOPOLOGIES:
    for core_seed in core_seeds:
        net, states = collect_for(topology, core_seed, chosen_isat)
        stage = net.core.stages[0]
        hidden = states[washout:, :HIDDEN_DIM]
        _, mc = memory_capacity_washout(states[:, :HIDDEN_DIM], u_raw, washout=washout)
        nrmse, r2 = ridge_features(states[:, :HIDDEN_DIM], y)
        report = net.spectral_norm_report["stages"][0]
        row = {
            "topology": topology["tag"],
            "topology_family": topology["family"],
            "topology_kwargs": dict(topology["kwargs"]),
            "core_seed": int(core_seed),
            "small_world_graph_seed": (
                None if topology["family"] != "small_world" else 1000 + int(core_seed)
            ),
            "core_edges": int(stage.src.numel()),
            "n_input_taps": N_TAPS,
            "n_boundary_otas": N_TAPS * HIDDEN_DIM,
            "boundary_isat_raw_mean": chosen_isat,
            "mc": float(mc),
            "state_pr": covariance_pr(hidden),
            "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max),
            "rail_frac": float((hidden.abs() > .9 * stage.x_max).float().mean()),
            "knet_ridge_nrmse": nrmse,
            "knet_ridge_r2": r2,
            "spectral_before": float(report["sr_before"]),
            "spectral_after": float(report["sr_after"]),
            "spectral_passes": int(report["passes"]),
        }
        row.update(controllability_metrics(stage))
        row.update(driven_jacobian_metrics(stage, states[:, :HIDDEN_DIM], taps_rail))
        rows.append(row)
        print("CORE_RESULT " + json.dumps(row, sort_keys=True), flush=True)


summary = {}
numeric_keys = (
    "mc", "state_pr", "mean_abs_over_rail", "rail_frac",
    "knet_ridge_nrmse", "knet_ridge_r2", "spectral_after",
    "jac_spectral_radius", "jac_eig_pr", "jac_modes_gt_080",
    "jac_nonnormality", "input_jacobian_pr", "controllability_pr",
    "driven_jac_spectral_radius_mean", "driven_jac_eig_pr_mean",
    "driven_jac_modes_gt_080_mean", "driven_jac_nonnormality_mean",
)
by_topology = defaultdict(list)
for row in rows:
    by_topology[row["topology"]].append(row)
for tag, group in by_topology.items():
    aggregate = {"n_core_seeds": len(group), "boundary_isat_raw_mean": chosen_isat}
    for key in numeric_keys:
        values = [float(row[key]) for row in group if key in row]
        if values:
            vals = torch.tensor(values)
            aggregate[f"{key}_mean"] = float(vals.mean())
            aggregate[f"{key}_std"] = float(vals.std(unbiased=False))
    summary[tag] = aggregate
    print("CORE_SUMMARY " + json.dumps({"topology": tag, **aggregate}, sort_keys=True), flush=True)

result = {
    "experiment": "phase6_real_knet_delay_bank_core_eigenstructure",
    "architecture": "8 causal delayed rail terminals -> 200 LinearOTA boundary currents -> recurrent KCL core",
    "drive_policy": "calibrate once on directed torus, then hold the selected heterogeneous boundary injection fixed for all core topologies",
    "target_mean_abs_over_rail": [0.45, 0.55],
    "drive_candidates_raw": drive_candidates,
    "drive_calibration": calibration,
    "selected_drive": chosen_calibration,
    "core_seeds": core_seeds,
    "topologies": TOPOLOGIES,
    "rows": rows,
    "summary": summary,
}
with open(os.path.join(out_dir, "core_eigen_results.json"), "w") as f:
    json.dump(result, f, indent=2)
with open(os.path.join(out_dir, "core_eigen_rows.jsonl"), "w") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
print("RESULTS_WRITTEN " + os.path.join(out_dir, "core_eigen_results.json"), flush=True)
PY

echo "Results: $OUT/core_eigen_results.json"
