"""Paired Phase-8 leak-timescale experiment for the physical NARMA-10 KNet.

This is deliberately an experiment driver, rather than a model change: the
only condition-specific tensors are ``stage.raw_leak``.  The banded condition
does not apply the legacy global-gain normalizer after those tensors change.
"""
from __future__ import annotations

import argparse
import copy
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


RIDGE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def ridge_metrics(features: torch.Tensor, target: torch.Tensor, washout: int) -> tuple[float, float]:
    """Legacy in-sample ridge metric retained for the pre-existing headline."""
    x, y = features[washout:], target[washout:].to(features)
    xa = torch.cat((x, x.new_ones(x.shape[0], 1)), 1)
    w = torch.linalg.solve(xa.T @ xa + 1e-2 * torch.eye(xa.shape[1], device=x.device), xa.T @ y)
    pred = xa @ w
    return float(ne.nrmse(pred, y)), float(ne.r2(pred, y))


def validated_ridge_metrics(features: torch.Tensor, target: torch.Tensor, washout: int,
                            standardized: bool) -> dict:
    """Train/validation/test ridge diagnostic; scaling is fit on train only."""
    x, y = features[washout:], target[washout:].to(features)
    n_train, n_val = int(.60 * len(x)), int(.20 * len(x))
    if n_train < 2 or len(x) - n_train - n_val < 1:
        raise ValueError("too few washed samples for validated ridge")
    tr_x, va_x, te_x = x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]
    tr_y, va_y, te_y = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]
    if standardized:
        mean, scale = tr_x.mean(0, keepdim=True), tr_x.std(0, unbiased=False, keepdim=True).clamp_min(1e-8)
        tr_x, va_x, te_x = (tr_x - mean) / scale, (va_x - mean) / scale, (te_x - mean) / scale
    def design(z): return torch.cat((z, z.new_ones(z.shape[0], 1)), 1)
    tr_a, va_a, te_a = design(tr_x), design(va_x), design(te_x)
    ident = torch.eye(tr_a.shape[1], device=x.device); ident[-1, -1] = 0.
    candidates = []
    for l2 in RIDGE_GRID:
        w = torch.linalg.solve(tr_a.T @ tr_a + l2 * ident, tr_a.T @ tr_y)
        candidates.append((float(ne.nrmse(va_a @ w, va_y)), l2, w))
    _, l2, w = min(candidates, key=lambda row: row[0])
    # Refit with the chosen penalty using the validation labels only after selection.
    fit_a, fit_y = torch.cat((tr_a, va_a)), torch.cat((tr_y, va_y))
    w = torch.linalg.solve(fit_a.T @ fit_a + l2 * ident, fit_a.T @ fit_y)
    pred = te_a @ w
    return {"nrmse": float(ne.nrmse(pred, te_y)), "r2": float(ne.r2(pred, te_y)),
            "selected_l2": l2, "n_train": n_train, "n_val": n_val, "n_test": len(te_x)}


def spectrum_metrics(stage) -> dict:
    """Measure the realised zero-point spectrum after each leak intervention."""
    dev = str(next(stage.parameters()).device)
    poles = torch.linalg.eigvals(_jacobian_at_zero(stage, t_span=T_SPAN, num_steps=N_STEPS, device=dev)).abs().sort(descending=True).values.cpu()
    # Band split is hard-coded for the fixed 25-node fabric (9 slow / 8 medium / 8 fast).
    assert poles.numel() == HIDDEN_DIM == 25, f"band split assumes 25 poles, got {poles.numel()}"
    slow, medium, fast = poles[:9], poles[9:17], poles[17:]
    gaps = {"slow_minus_medium": float(slow[-1] - medium[0]), "medium_minus_fast": float(medium[-1] - fast[0])}
    means = {"slow": float(slow.mean()), "medium": float(medium.mean()), "fast": float(fast.mean())}
    return {"zero_jacobian_spectral_radius": float(poles[0]),
            "zero_jacobian_abs_eigenvalues_desc": [float(v) for v in poles],
            "realized_spectrum_band_means": means, "realized_spectrum_band_gaps": gaps,
            "realized_spectrum_bands_distinguishable": bool(
                means["slow"] - means["medium"] >= .02 and means["medium"] - means["fast"] >= .02),}


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
    # collect_fabric_states runs under inference_mode, so its output is an
    # inference tensor; clone to a regular tensor because local_linear_metrics
    # differentiates through _one_sample_transition below (inference tensors
    # cannot require grad, and .detach() alone preserves the flag).
    states = ne.collect_fabric_states(net, taps, device=str(next(net.parameters()).device)).cpu()[:, :HIDDEN_DIM].clone()
    stage, hidden = net.core.stages[0], states[washout:]
    delays, mc = memory_capacity_washout(states, u_raw, washout=washout)
    nr, rtwo = ridge_metrics(states, y, washout)
    raw_ridge = validated_ridge_metrics(states, y, washout, standardized=False)
    standardized_ridge = validated_ridge_metrics(states, y, washout, standardized=True)
    bcurr = boundary_net_currents(stage, states, taps)[washout:]
    ones = torch.ones(HIDDEN_DIM)
    centered = hidden - hidden.mean(0, keepdim=True); leading = torch.linalg.eigh(centered.T @ centered)[1][:, -1]
    alignment = float((leading @ ones).abs() / (leading.norm() * ones.norm()))
    ones_fraction = float((centered @ (ones / ones.norm())).var(unbiased=False) / centered.var(0, unbiased=False).sum().clamp_min(1e-30))
    product = u_raw * torch.cat((u_raw.new_zeros(9), u_raw[:-9]))
    _, product_r2 = ridge_metrics(states, product, washout)
    row = {"condition": condition, "core_seed": seed, "mc": float(mc), "mc_by_delay": [float(v) for v in delays], "mc_lag_8_to_12": [float(v) for v in delays[8:13]], "lag9_product_r2": product_r2, "knet_ridge_nrmse": nr, "knet_ridge_r2": rtwo, "ridge_validated_raw": raw_ridge, "ridge_validated_standardized": standardized_ridge, "mean_abs_over_rail": float(hidden.abs().mean() / stage.x_max), "rail_frac": float((hidden.abs() > .9 * stage.x_max).float().mean()), "state_correlation_pr": corr_pr(hidden), "leading_pc_all_ones_alignment": alignment, "all_ones_variance_fraction": ones_fraction, "net_boundary_current": covariance_metrics(bcurr, "net_boundary_current")}
    row.update(spectrum_metrics(stage))
    row.update(covariance_metrics(hidden, "state")); row.update(local_linear_metrics(stage, states, taps, washout, n_local))
    return row


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--samples", type=int, default=3000); p.add_argument("--washout", type=int, default=200); p.add_argument("--core-seeds", default="0,1,2"); p.add_argument("--n-local-jac", type=int, default=8); p.add_argument("--uniform-leak-grid", default="0.01,0.02,0.03,0.04,0.051,0.08,0.12,0.18,0.223,0.32,0.42,0.511,0.65"); p.add_argument("--band-alpha-grid", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0"); p.add_argument("--activity-target", type=float, default=.49); p.add_argument("--activity-tolerance", type=float, default=.01); p.add_argument("--rail-max", type=float, default=.20); p.add_argument("--zero-rho-max", type=float, default=1.05)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in a.core_seeds.split(",") if v.strip()]; grid = [float(v) for v in a.uniform_leak_grid.split(",") if v.strip()]; alpha_grid = [float(v) for v in a.band_alpha_grid.split(",") if v.strip()]
    if not seeds or not grid or not alpha_grid or any(v <= 0 for v in grid + alpha_grid): raise ValueError("need core seeds and positive calibration candidates")
    if not 0 < a.activity_target < 1 or a.activity_tolerance <= 0: raise ValueError("activity target/tolerance must be positive fractions")
    if a.samples <= a.washout + max(20, GRAMIAN_HORIZON) + 1:
        raise ValueError("samples must exceed washout by at least 21 for MC and local tangent diagnostics")
    # Calibration is a separate, target-free NARMA input realization.  Its
    # scores never choose a band assignment or alter the fixed boundary.
    cal_u, _ = ne.narma(a.samples, order=10, seed=77); cal_taps = delay_bank(ne._scale_drive(cal_u, bipolar=True, order=10, input_scale=1.0))
    u, y = ne.narma(a.samples, order=10, seed=0); taps = delay_bank(ne._scale_drive(u, bipolar=True, order=10, input_scale=1.0))
    rows, calibrations = [], []
    for seed in seeds:
        base = build_base(seed).to(a.device)
        # Every calibration candidate starts from this exact normalized core.
        # Cloning makes leak the sole intervention and avoids redoing the same
        # 30-pass normalization (and its noisy log) for every candidate.
        def clone_base(): return copy.deepcopy(base)
        # The alpha calibration changes all three bands by one scalar only.
        # It cannot alter the core graph, boundary, band ratios, or global gain.
        band_vals = torch.empty(HIDDEN_DIM)
        for name, nodes in BAND_GROUPS.items(): band_vals[nodes] = BAND_LEAKS[name]
        banded = clone_base(); set_leaks(banded.core.stages[0], band_vals)
        band_candidates = []
        for alpha in alpha_grid:
            candidate = clone_base(); set_leaks(candidate.core.stages[0], alpha * band_vals)
            x = ne.collect_fabric_states(candidate, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
            spec = spectrum_metrics(candidate.core.stages[0])
            band_candidates.append({"alpha": alpha, "mean_abs_over_rail": float(x.abs().mean() / candidate.core.stages[0].x_max), "rail_frac": float((x.abs() > .9 * candidate.core.stages[0].x_max).float().mean()), "finite_states": bool(torch.isfinite(x).all()), "spectrum": spec})
        # A coarse grid finds a measured bracket; bisection then calibrates
        # alpha rather than treating the grid itself as the operating point.
        for _ in range(8):
            ordered = sorted(band_candidates, key=lambda r: r["alpha"])
            brackets = [(left, right) for left, right in zip(ordered, ordered[1:])
                        if left["finite_states"] and right["finite_states"]
                        and left["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max
                        and right["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max
                        and left["spectrum"]["realized_spectrum_bands_distinguishable"]
                        and right["spectrum"]["realized_spectrum_bands_distinguishable"]
                        and (left["mean_abs_over_rail"] - a.activity_target) * (right["mean_abs_over_rail"] - a.activity_target) <= 0]
            if not brackets: break
            left, right = min(brackets, key=lambda pair: pair[1]["alpha"] - pair[0]["alpha"])
            alpha = (left["alpha"] + right["alpha"]) / 2
            candidate = clone_base(); set_leaks(candidate.core.stages[0], alpha * band_vals)
            x = ne.collect_fabric_states(candidate, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
            band_candidates.append({"alpha": alpha, "mean_abs_over_rail": float(x.abs().mean() / candidate.core.stages[0].x_max), "rail_frac": float((x.abs() > .9 * candidate.core.stages[0].x_max).float().mean()), "finite_states": bool(torch.isfinite(x).all()), "spectrum": spectrum_metrics(candidate.core.stages[0])})
        acceptable_bands = [r for r in band_candidates if r["finite_states"] and r["rail_frac"] <= a.rail_max and r["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max and r["spectrum"]["realized_spectrum_bands_distinguishable"] and abs(r["mean_abs_over_rail"] - a.activity_target) <= a.activity_tolerance]
        if not acceptable_bands:
            (a.out / "calibration_failure.json").write_text(json.dumps({"core_seed": seed, "condition": "banded_leaks_activity_restored", "reason": "no acceptable one-dimensional band-preserving alpha", "candidates": band_candidates}, indent=2))
            raise RuntimeError(f"no stable band-preserving alpha reaches {a.activity_target:.3f} +/- {a.activity_tolerance:.3f} for seed {seed}; refusing to broaden the intervention")
        selected_band = min(acceptable_bands, key=lambda r: abs(r["mean_abs_over_rail"] - a.activity_target))
        restored = clone_base(); set_leaks(restored.core.stages[0], selected_band["alpha"] * band_vals)
        # Independent uniform calibration targets the requested activity, not
        # Phase-7's old 0.401 banded amplitude or a pre-assumed leak bracket.
        candidates = []
        for leak in grid:
            candidate = clone_base(); set_leaks(candidate.core.stages[0], torch.full((HIDDEN_DIM,), leak))
            x = ne.collect_fabric_states(candidate, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
            candidates.append({"leak": leak, "mean_abs_over_rail": float(x.abs().mean() / candidate.core.stages[0].x_max), "rail_frac": float((x.abs() > .9 * candidate.core.stages[0].x_max).float().mean()), "finite_states": bool(torch.isfinite(x).all()), "spectrum": spectrum_metrics(candidate.core.stages[0])})
        # Use exactly the same measured, input-only calibration procedure for
        # uniform leaks.  This avoids declaring failure merely because the
        # initial supplied grid was too coarse near the requested activity.
        for _ in range(8):
            ordered = sorted(candidates, key=lambda r: r["leak"])
            # A bracket endpoint can be over rail while the target crossing
            # itself is safe (as happened between 0.08 and 0.12).  It is
            # calibration evidence only: the final selection below still
            # requires rails <= rail_max and is the only reported condition.
            brackets = [(left, right) for left, right in zip(ordered, ordered[1:])
                        if left["finite_states"] and right["finite_states"]
                        and left["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max
                        and right["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max
                        and (left["mean_abs_over_rail"] - a.activity_target) * (right["mean_abs_over_rail"] - a.activity_target) <= 0]
            if not brackets: break
            left, right = min(brackets, key=lambda pair: pair[1]["leak"] - pair[0]["leak"])
            leak = (left["leak"] + right["leak"]) / 2
            candidate = clone_base(); set_leaks(candidate.core.stages[0], torch.full((HIDDEN_DIM,), leak))
            x = ne.collect_fabric_states(candidate, cal_taps, device=a.device).cpu()[a.washout:, :HIDDEN_DIM]
            candidates.append({"leak": leak, "mean_abs_over_rail": float(x.abs().mean() / candidate.core.stages[0].x_max), "rail_frac": float((x.abs() > .9 * candidate.core.stages[0].x_max).float().mean()), "finite_states": bool(torch.isfinite(x).all()), "spectrum": spectrum_metrics(candidate.core.stages[0])})
        acceptable_uniform = [r for r in candidates if r["finite_states"] and r["rail_frac"] <= a.rail_max and r["spectrum"]["zero_jacobian_spectral_radius"] <= a.zero_rho_max and abs(r["mean_abs_over_rail"] - a.activity_target) <= a.activity_tolerance]
        if not acceptable_uniform:
            (a.out / "calibration_failure.json").write_text(json.dumps({"core_seed": seed, "condition": "uniform_activity_matched", "reason": "no acceptable scalar uniform leak", "candidates": candidates}, indent=2))
            raise RuntimeError(f"no stable uniform leak reaches {a.activity_target:.3f} +/- {a.activity_tolerance:.3f} for seed {seed}")
        selected = min(acceptable_uniform, key=lambda r: abs(r["mean_abs_over_rail"] - a.activity_target))
        uniform = clone_base(); set_leaks(uniform.core.stages[0], torch.full((HIDDEN_DIM,), selected["leak"]))
        calibrations.append({"core_seed": seed, "method": "input_only; no NARMA targets or scores", "activity_target": a.activity_target, "activity_tolerance": a.activity_tolerance, "rail_max": a.rail_max, "zero_rho_max": a.zero_rho_max, "band_alpha_candidates": band_candidates, "band_alpha_selected": selected_band, "uniform_candidates": candidates, "uniform_selected": selected, "activity_mismatch_uniform_minus_banded": selected["mean_abs_over_rail"] - selected_band["mean_abs_over_rail"]})
        for tag, net in (("baseline", base), ("banded_leaks_alpha1_banked", banded), ("banded_leaks_activity_restored", restored), ("uniform_activity_matched", uniform)):
            row = evaluate(net, tag, seed, taps, u, y, a.washout, a.n_local_jac)
            row["leaks"] = [float(v) for v in net.core.stages[0]._effective_leak().detach().cpu()]
            row["spectral_normalization_reapplied_after_leak_change"] = False
            row["band_leak_alpha"] = selected_band["alpha"] if tag == "banded_leaks_activity_restored" else (1.0 if tag == "banded_leaks_alpha1_banked" else None)
            row["uniform_selected_leak"] = selected["leak"] if tag == "uniform_activity_matched" else None
            rows.append(row); print("LEAK_BAND_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        # Baseline-only diagnostic; evaluate it after state collection so it
        # never changes model parameters or experimental conditions.
        surrogate = tangent_surrogate(base.core.stages[0], taps, a.washout)
        rows[-4].update(surrogate)
    summary = {}
    for condition in {r["condition"] for r in rows}:
        group = [r for r in rows if r["condition"] == condition]; out = {"n_paired_seeds": len(group)}
        for key in ("mc", "state_pr", "knet_ridge_nrmse", "knet_ridge_r2", "mean_abs_over_rail", "rail_frac", "state_correlation_pr", "lag9_product_r2"):
            v = torch.tensor([r[key] for r in group]); out[key + "_mean"] = float(v.mean()); out[key + "_std"] = float(v.std(unbiased=False))
        summary[condition] = out
    by_seed = {seed: {r["condition"]: r for r in rows if r["core_seed"] == seed} for seed in seeds}
    paired = []
    for seed, conditions in by_seed.items():
        b, band, uniform = conditions["baseline"], conditions["banded_leaks_activity_restored"], conditions["uniform_activity_matched"]
        paired.append({"core_seed": seed, "banded_nrmse_improvement": b["knet_ridge_nrmse"] - band["knet_ridge_nrmse"], "banded_mc_gain": band["mc"] - b["mc"], "banded_beats_uniform_nrmse": band["knet_ridge_nrmse"] < uniform["knet_ridge_nrmse"], "banded_beats_uniform_validated_standardized": band["ridge_validated_standardized"]["nrmse"] < uniform["ridge_validated_standardized"]["nrmse"], "screening_win": .45 <= band["mean_abs_over_rail"] <= .55 and band["rail_frac"] <= .20 and b["knet_ridge_nrmse"] - band["knet_ridge_nrmse"] >= .02 and band["knet_ridge_nrmse"] - uniform["knet_ridge_nrmse"] <= -.02 and band["mc"] - b["mc"] >= .3})
    result = {"experiment": "phase8_band_preserving_activity_restoration_vs_activity_matched_uniform", "fixed_architecture": "directed 5x5 torus; 8 causal taps; 200 fixed LinearOTA boundary edges with isat_raw mean -4.4; tanh nodes; eight Heun steps", "leak_assignment": {"groups": BAND_GROUPS, "unscaled_leaks": BAND_LEAKS, "formula": "leak_j(alpha) = alpha * leak_j(banded)"}, "no_post_leak_global_gain_normalization": True, "core_seeds": seeds, "calibrations": calibrations, "rows": rows, "summary": summary, "paired_screening": paired, "screening_thresholds": {"mean_activity_range": [.45, .55], "rail_max": .20, "nrmse_absolute_improvement_vs_baseline": .02, "nrmse_absolute_advantage_vs_uniform": .02, "mc_gain": .3, "raw_pr": "diagnostic only", "paired_seed_requirement": "all paired seeds improve NRMSE"}}
    (a.out / "leak_bands_results.json").write_text(json.dumps(result, indent=2)); (a.out / "leak_bands_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print("RESULTS_WRITTEN " + str(a.out / "leak_bands_results.json"))

if __name__ == "__main__": main()
