"""Paired Phase-7 leak-timescale experiment for the physical NARMA-10 KNet.

This is deliberately an experiment driver, rather than a model change: the
only condition-specific tensors are ``stage.raw_leak``.  The banded condition
does not apply the legacy global-gain normalizer after those tensors change.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import narma_experiment as ne
from cell_library import make_cell_library
from config import make_narma_preset
from narma_advisor_probes import _one_sample_transition, memory_capacity_washout
from narma_spectral_norm import _jacobian_at_zero
from topology import build_net_from_config

N_TAPS, HIDDEN_DIM, T_SPAN, N_STEPS = 8, 25, 1.0, 8
SR_TARGET, SR_PASSES, GRAMIAN_HORIZON = .95, 30, 20
BAND_LEAKS = {"slow": .051, "medium": .223, "fast": .511}
# Fixed target-independent spatial interleave: 0,3,... slow; 1,4,... medium.
BAND_GROUPS = {"slow": list(range(0, 25, 3)), "medium": list(range(1, 25, 3)), "fast": list(range(2, 25, 3))}


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def delay_bank(u: torch.Tensor) -> torch.Tensor:
    return torch.stack([u if lag == 0 else torch.cat((u.new_zeros(lag), u[:-lag])) for lag in range(N_TAPS)], 1)


def eig_pr(eigs: torch.Tensor) -> float:
    e = eigs.real.clamp_min(0)
    return float(e.sum().square() / e.square().sum().clamp_min(1e-30))


def covariance_eigs(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(0, keepdim=True)
    return torch.linalg.eigvalsh(xc.T @ xc / max(1, xc.shape[0] - 1)).clamp_min(0)


def covariance_metrics(x: torch.Tensor, prefix: str) -> dict:
    eigs = covariance_eigs(x)
    return {f"{prefix}_pr": eig_pr(eigs), f"{prefix}_eigenvalues_desc": [float(v) for v in eigs.flip(0)]}


def corr_pr(x: torch.Tensor) -> float:
    xc = x - x.mean(0, keepdim=True)
    scale = xc.square().mean(0).sqrt().clamp_min(1e-12)
    return eig_pr(covariance_eigs(xc / scale))


def set_leaks(stage, values: torch.Tensor) -> None:
    with torch.no_grad():
        stage.raw_leak.copy_(softplus_inverse(values.to(stage.raw_leak.device)))
        stage.leak_mode = "programmable"


def heterogenize_core(stage, seed: int) -> None:
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(510 + seed)
        lib = stage.cell_lib
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.a_raw.device))
        lib.b_raw.copy_(torch.randn(lib.b_raw.shape, generator=gen).clamp(-1.2, 1.2).to(lib.b_raw.device))
        lib.s_raw.copy_(torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5, 1., -1.).to(lib.s_raw.device))
        lib.gm_raw.copy_((-1. + .45 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2.5, .8).to(lib.gm_raw.device))
        lib.isat_raw.fill_(-4.)
        leaks = torch.empty(stage.raw_leak.shape).uniform_(math.log(.06), math.log(.16), generator=gen).exp()
        set_leaks(stage, leaks)


def configure_boundary(stage) -> None:
    """The exact fixed Phase-6 calibrated -4.4 heterogeneous boundary."""
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(511)
        lib = stage.boundary_cell_lib
        lib.a_raw.copy_(torch.randn(lib.a_raw.shape, generator=gen).clamp(-1., 1.5).to(lib.a_raw.device))
        lib.b_raw.fill_(-6.)
        lib.s_raw.copy_(torch.where(torch.rand(lib.s_raw.shape, generator=gen) > .5, 1., -1.).to(lib.s_raw.device))
        lib.gm_raw.copy_((-1. + .40 * torch.randn(lib.gm_raw.shape, generator=gen)).clamp(-2., .5).to(lib.gm_raw.device))
        lib.isat_raw.copy_((-4.4 + .30 * torch.randn(lib.isat_raw.shape, generator=gen)).clamp(-6., 0.).to(lib.isat_raw.device))


def build_base(seed: int):
    preset = make_narma_preset(order=10, hidden_dim=HIDDEN_DIM, num_steps_per_sample=N_STEPS, t_span=T_SPAN, leak_constant=None)
    preset["stages"] = [dict(preset["stages"][0], num_inputs=N_TAPS, hidden_family="torus", hidden_kwargs={"height": 5, "width": 5, "kernel_size": 3, "bidirectional": False})]
    net = build_net_from_config(cfg=preset, cell_lib=make_cell_library("linear_ota"), boundary_fan_out={i: list(range(HIDDEN_DIM)) for i in range(N_TAPS)}, allow_boundary_target_overlap=True, enable_temporal_readout=False, freeze_read=False, node_activation="tanh", allow_experimental_cells=True)
    stage = net.core.stages[0]
    heterogenize_core(stage, seed)
    configure_boundary(stage)
    # This one-time baseline normalization is part of the established core.
    # Save/restore prevents it from changing any one of the 200 boundary OTAs.
    saved = {name: getattr(stage.boundary_cell_lib, name).detach().clone() for name in ("a_raw", "b_raw", "s_raw", "gm_raw", "isat_raw")}
    ne._apply_spectral_norm(net, SR_TARGET, T_SPAN, N_STEPS, tag=f"phase7-base-seed{seed}", max_passes=SR_PASSES)
    with torch.no_grad():
        for name, value in saved.items():
            getattr(stage.boundary_cell_lib, name).copy_(value)
    return net


def ridge_metrics(features: torch.Tensor, target: torch.Tensor, washout: int) -> tuple[float, float]:
    x, y = features[washout:], target[washout:].to(features)
    xa = torch.cat((x, x.new_ones(x.shape[0], 1)), 1)
    w = torch.linalg.solve(xa.T @ xa + 1e-2 * torch.eye(xa.shape[1], device=x.device), xa.T @ y)
    pred = xa @ w
    return float(ne.nrmse(pred, y)), float(ne.r2(pred, y))


def boundary_net_currents(stage, states: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Per-node net boundary-current vector, recomputed on recorded states."""
    with torch.no_grad():
        device = next(stage.parameters()).device
        x, u = states.to(device), taps.to(device)
        dst_x = stage._node_broadcast(x)[:, stage.boundary_dst] if stage.node_activation == "tanh" else x[:, stage.boundary_dst]
        current = stage.boundary_cell_lib(u[:, stage.boundary_src], dst_x, x_max=stage.x_max)
        current = current * torch.sigmoid(stage.boundary_z_logits).unsqueeze(0)
        out = torch.zeros(x.shape[0], HIDDEN_DIM, device=x.device, dtype=current.dtype)
        out.index_add_(1, stage.boundary_dst, current)
        return out.cpu()


def causal_delay_augmented_map(j: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tangent map for hidden state plus seven causal registers.

    The augmented state is ``[x, u[t], ..., u[t-6]]``.  A scalar innovation
    supplies the new ``u[t+1]``; it affects the first input-tap column of
    ``b``, while the existing registers shift by one place.  Shapes are
    checked here so an implementation change fails clearly instead of inside
    an opaque tensor slice assignment.
    """
    n_registers = N_TAPS - 1
    expected_j, expected_b = (HIDDEN_DIM, HIDDEN_DIM), (HIDDEN_DIM, N_TAPS)
    if tuple(j.shape) != expected_j or tuple(b.shape) != expected_b:
        raise ValueError(
            "unexpected one-sample tangent shapes: "
            f"J={tuple(j.shape)} (expected {expected_j}), "
            f"B={tuple(b.shape)} (expected {expected_b})"
        )
    n_aug = HIDDEN_DIM + n_registers
    a = torch.zeros(n_aug, n_aug, dtype=j.dtype, device=j.device)
    a[:HIDDEN_DIM, :HIDDEN_DIM] = j
    a[:HIDDEN_DIM, HIDDEN_DIM:] = b[:, 1:]
    # r_next[0] is supplied by the innovation; r_next[1:] = r_old[:-1].
    if n_registers > 1:
        a[HIDDEN_DIM + 1:, HIDDEN_DIM:HIDDEN_DIM + n_registers - 1] = torch.eye(
            n_registers - 1, dtype=j.dtype, device=j.device,
        )
    g = torch.cat((b[:, 0], torch.ones(1, dtype=j.dtype, device=j.device),
                   torch.zeros(n_registers - 1, dtype=j.dtype, device=j.device)))
    return a, g


def local_linear_metrics(stage, states: torch.Tensor, taps: torch.Tensor, washout: int, n_local: int) -> dict:
    if n_local <= 0: return {}
    dev = next(stage.parameters()).device; dt = T_SPAN / N_STEPS
    indices = torch.linspace(washout, states.shape[0] - 2, steps=min(n_local, states.shape[0] - washout - 1)).round().long().unique().tolist()
    rows, causal = [], []
    for i in indices:
        def f(x, q): return _one_sample_transition(stage, x.view(1, -1), q.view(1, -1), dt, N_STEPS)
        x = states[i].to(dev).detach(); q = taps[i + 1].to(dev).detach()
        j, b = torch.autograd.functional.jacobian(f, (x, q), create_graph=False)
        vals = torch.linalg.eigvals(j).abs()
        rows.append({"rho": float(vals.max()), "eig_pr": float(vals.sum().square() / vals.square().sum().clamp_min(1e-30)), "b_all_pr": eig_pr(torch.linalg.eigvalsh(b @ b.T)), "b_new_scalar_norm": float(torch.linalg.vector_norm(b[:, 0]))})
        # Actual scalar innovation: q_next=[w,q0,...q6].  Build finite-time
        # augmented tangent Gramian and retain only its hidden-state block.
        gram = torch.zeros(HIDDEN_DIM + N_TAPS - 1, HIDDEN_DIM + N_TAPS - 1,
                           dtype=j.dtype, device=dev)
        for k in range(i, min(i + GRAMIAN_HORIZON, states.shape[0] - 1)):
            def fk(xx, qq): return _one_sample_transition(stage, xx.view(1, -1), qq.view(1, -1), dt, N_STEPS)
            xx, qq = states[k].to(dev).detach(), taps[k + 1].to(dev).detach()
            jk, bk = torch.autograd.functional.jacobian(fk, (xx, qq), create_graph=False)
            a, gg = causal_delay_augmented_map(jk, bk)
            gram = a @ gram @ a.T + torch.outer(gg, gg)
        hidden = (gram[:HIDDEN_DIM, :HIDDEN_DIM] + gram[:HIDDEN_DIM, :HIDDEN_DIM].T) * .5
        causal.append((eig_pr(torch.linalg.eigvalsh(hidden)), float(torch.trace(hidden))))
    return {"n_local_jacobians": len(rows), "driven_jac_spectral_radius_mean": float(torch.tensor([r["rho"] for r in rows]).mean()), "driven_jac_eig_pr_mean": float(torch.tensor([r["eig_pr"] for r in rows]).mean()), "trajectory_input_jacobian_all_taps_pr_mean": float(torch.tensor([r["b_all_pr"] for r in rows]).mean()), "trajectory_input_jacobian_new_scalar_norm_mean": float(torch.tensor([r["b_new_scalar_norm"] for r in rows]).mean()), "causal_trajectory_controllability_hidden_pr_mean": float(torch.tensor([x[0] for x in causal]).mean()), "causal_trajectory_controllability_hidden_trace_mean": float(torch.tensor([x[1] for x in causal]).mean())}


def tangent_surrogate(stage, taps: torch.Tensor, washout: int) -> dict:
    dev = next(stage.parameters()).device
    j = _jacobian_at_zero(stage, t_span=T_SPAN, num_steps=N_STEPS, device=str(dev))
    x0, q0 = torch.zeros(HIDDEN_DIM, device=dev), torch.zeros(N_TAPS, device=dev)
    def f(x, q): return _one_sample_transition(stage, x.view(1, -1), q.view(1, -1), T_SPAN / N_STEPS, N_STEPS)
    b = torch.autograd.functional.jacobian(f, (x0, q0), create_graph=False)[1]
    centered = taps.to(dev) - taps.to(dev).mean(0, keepdim=True)
    z = torch.zeros(centered.shape[0], HIDDEN_DIM, device=dev)
    for t in range(1, len(z)): z[t] = j @ z[t - 1] + b @ centered[t]
    return covariance_metrics(z[washout:].cpu(), "causal_linear_surrogate_state") | {"zero_jacobian_spectral_radius": float(torch.linalg.eigvals(j).abs().max())}


def evaluate(net, condition: str, seed: int, taps: torch.Tensor, u_raw: torch.Tensor, y: torch.Tensor, washout: int, n_local: int) -> dict:
    states = ne.collect_fabric_states(net, taps, device=str(next(net.parameters()).device)).cpu()[:, :HIDDEN_DIM]
    stage, hidden = net.core.stages[0], states[washout:]
    delays, mc = memory_capacity_washout(states, u_raw, washout=washout)
    nr, rtwo = ridge_metrics(states, y, washout)
    bcurr = boundary_net_currents(stage, states, taps)[washout:]
    ones = torch.ones(HIDDEN_DIM)
    centered = hidden - hidden.mean(0, keepdim=True); leading = torch.linalg.eigh(centered.T @ centered)[1][:, -1]
    alignment = float((leading @ ones).abs() / (leading.norm() * ones.norm()))
    ones_fraction = float((centered @ (ones / ones.norm())).var(unbiased=False) / centered.var(0, unbiased=False).sum().clamp_min(1e-30))
    product = u_raw * torch.cat((u_raw.new_zeros(9), u_raw[:-9]))
    _, product_r2 = ridge_metrics(states, product, washout)
    zero_j = _jacobian_at_zero(stage, t_span=T_SPAN, num_steps=N_STEPS,
                               device=str(next(stage.parameters()).device))
    zero_poles = torch.linalg.eigvals(zero_j).abs().sort(descending=True).values
    row = {"condition": condition, "core_seed": seed, "mc": float(mc), "mc_by_delay": [float(v) for v in delays], "mc_lag_8_to_12": [float(v) for v in delays[8:13]], "lag9_product_r2": product_r2, "knet_ridge_nrmse": nr, "knet_ridge_r2": rtwo, "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max), "rail_frac": float((hidden.abs() > .9 * stage.x_max).float().mean()), "state_correlation_pr": corr_pr(hidden), "leading_pc_all_ones_alignment": alignment, "all_ones_variance_fraction": ones_fraction, "zero_jacobian_spectral_radius": float(zero_poles[0]), "zero_jacobian_abs_eigenvalues_desc": [float(v) for v in zero_poles], "net_boundary_current": covariance_metrics(bcurr, "net_boundary_current")}
    row.update(covariance_metrics(hidden, "state")); row.update(local_linear_metrics(stage, states, taps, washout, n_local))
    return row


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--samples", type=int, default=3000); p.add_argument("--washout", type=int, default=200); p.add_argument("--core-seeds", default="0,1,2"); p.add_argument("--n-local-jac", type=int, default=8); p.add_argument("--uniform-leak-grid", default="0.04,0.051,0.08,0.12,0.18,0.223,0.32,0.42,0.511,0.65")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in a.core_seeds.split(",") if v.strip()]; grid = [float(v) for v in a.uniform_leak_grid.split(",") if v.strip()]
    if not seeds or any(v <= 0 for v in grid): raise ValueError("need core seeds and positive uniform leak candidates")
    if a.samples <= a.washout + max(20, GRAMIAN_HORIZON) + 1:
        raise ValueError("samples must exceed washout by at least 21 for MC and local tangent diagnostics")
    # Calibration is a separate, target-free NARMA input realization.  Its
    # scores never choose a band assignment or alter the fixed boundary.
    cal_u, _ = ne.narma(a.samples, order=10, seed=77); cal_taps = delay_bank(ne._scale_drive(cal_u, bipolar=True, order=10, input_scale=1.0))
    u, y = ne.narma(a.samples, order=10, seed=0); taps = delay_bank(ne._scale_drive(u, bipolar=True, order=10, input_scale=1.0))
    rows, calibrations = [], []
    for seed in seeds:
        base = build_base(seed).to(a.device)
        # Primary condition: replace leaks only, and explicitly do not normalize.
        banded = build_base(seed).to(a.device); band_vals = torch.empty(HIDDEN_DIM)
        for name, nodes in BAND_GROUPS.items(): band_vals[nodes] = BAND_LEAKS[name]
        set_leaks(banded.core.stages[0], band_vals)
        band_cal = ne.collect_fabric_states(banded, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
        target_amp = float(band_cal.abs().mean() / banded.core.stages[0].x_max)
        candidates = []
        for leak in grid:
            candidate = build_base(seed).to(a.device); set_leaks(candidate.core.stages[0], torch.full((HIDDEN_DIM,), leak))
            x = ne.collect_fabric_states(candidate, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
            candidates.append({"leak": leak, "mean_abs_over_rail": float(x.abs().mean() / candidate.core.stages[0].x_max), "rail_frac": float((x.abs() > .9 * candidate.core.stages[0].x_max).float().mean())})
        selected = min(candidates, key=lambda r: abs(r["mean_abs_over_rail"] - target_amp))
        uniform = build_base(seed).to(a.device); set_leaks(uniform.core.stages[0], torch.full((HIDDEN_DIM,), selected["leak"]))
        calibrations.append({"core_seed": seed, "banded_calibration_mean_abs_over_rail": target_amp, "uniform_candidates": candidates, "uniform_selected": selected, "amplitude_mismatch": selected["mean_abs_over_rail"] - target_amp})
        for tag, net in (("baseline", base), ("banded_leaks", banded), ("uniform_amplitude_matched", uniform)):
            row = evaluate(net, tag, seed, taps, u, y, a.washout, a.n_local_jac)
            row["leaks"] = [float(v) for v in net.core.stages[0]._effective_leak().detach().cpu()]
            row["spectral_normalization_reapplied_after_leak_change"] = False
            row["uniform_selected_leak"] = selected["leak"] if tag == "uniform_amplitude_matched" else None
            rows.append(row); print("LEAK_BAND_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        # Baseline-only diagnostic; evaluate it after state collection so it
        # never changes model parameters or experimental conditions.
        surrogate = tangent_surrogate(base.core.stages[0], taps, a.washout)
        rows[-3].update(surrogate)
    summary = {}
    for condition in {r["condition"] for r in rows}:
        group = [r for r in rows if r["condition"] == condition]; out = {"n_paired_seeds": len(group)}
        for key in ("mc", "state_pr", "knet_ridge_nrmse", "knet_ridge_r2", "mean_abs_over_rail", "rail_frac", "state_correlation_pr", "lag9_product_r2"):
            v = torch.tensor([r[key] for r in group]); out[key + "_mean"] = float(v.mean()); out[key + "_std"] = float(v.std(unbiased=False))
        summary[condition] = out
    by_seed = {seed: {r["condition"]: r for r in rows if r["core_seed"] == seed} for seed in seeds}
    paired = []
    for seed, conditions in by_seed.items():
        b, band, uniform = conditions["baseline"], conditions["banded_leaks"], conditions["uniform_amplitude_matched"]
        paired.append({"core_seed": seed, "banded_nrmse_improvement": b["knet_ridge_nrmse"] - band["knet_ridge_nrmse"], "banded_mc_gain": band["mc"] - b["mc"], "banded_beats_uniform_nrmse": band["knet_ridge_nrmse"] < uniform["knet_ridge_nrmse"], "screening_win": band["state_pr"] >= 2.5 and .45 <= band["mean_abs_over_rail"] <= .55 and band["rail_frac"] <= .20 and b["knet_ridge_nrmse"] - band["knet_ridge_nrmse"] >= .02 and band["mc"] - b["mc"] >= .3})
    result = {"experiment": "phase7_three_band_leaks_vs_amplitude_matched_uniform", "fixed_architecture": "directed 5x5 torus; 8 causal taps; 200 fixed LinearOTA boundary edges with isat_raw mean -4.4; tanh nodes; eight Heun steps", "leak_assignment": {"groups": BAND_GROUPS, "leaks": BAND_LEAKS}, "no_post_leak_global_gain_normalization": True, "core_seeds": seeds, "calibrations": calibrations, "rows": rows, "summary": summary, "paired_screening": paired, "screening_thresholds": {"state_pr_min": 2.5, "mean_amplitude_range": [.45, .55], "rail_max": .20, "nrmse_absolute_improvement": .02, "mc_gain": .3, "paired_seed_requirement": "all three paired seeds improve NRMSE"}}
    (a.out / "leak_bands_results.json").write_text(json.dumps(result, indent=2)); (a.out / "leak_bands_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print("RESULTS_WRITTEN " + str(a.out / "leak_bands_results.json"))

if __name__ == "__main__": main()
