"""Post-twin fatal-risk gates (plan post-twin-gates, feature spec flow-sparse-gates).

Three pre-registered fatal-risk gates stand between the confirmed digital
twin (``narma_linear_controls.twin_node_tanh``) and any multistage work.
Each gate is eval-only, zero-training, and carries a pre-registered
verdict that stops its downstream line.  Multistage (frozen baseline, then
training) stays stubbed until all three pass and emit a buildable spec.

Gate 1 — Heun flow-twin (realizability):
    Standalone Heun integrator for
    ``dx/dt = -lambda*x + lambda*tanh(W_mix@x + W_in@d)`` with drive held
    constant within a per-sample window.  Fixed-dt rule: ``t_span=1.0``
    held; ``num_steps`` scaled so ``dt <= 0.125`` at every ``lambda*T``.
    Sweep ``lambda*T in {0.3, 1, 3, 10} x radius x input scale x W-seeds
    x data-seeds``, taps8 main + scalar confirm.  Verdicts: holds at
    ``lambda*T <= 3`` with MC >= 8 -> flow-viable; only ``lambda*T >= 10``
    holds -> speed requirement recorded; collapses everywhere ->
    digital-only redirect.

Gate 2 — sparsity/device-budget + mismatch:
    Sparsity masks on W_mix at matched edge counts
    {~100, 200, 400, 625 dense}, magnitude-pruned-from-dense with
    spectral-radius restoration, plus one low-rank+sparse row (~200
    edges).  Mismatch multiplicative Gaussian sigma {0, 0.02, 0.05, 0.10}
    on W_mix and W_in at the best corner.  Heterogeneous operating-point
    row at the best corner.  Verdicts: graceful to ~200 edges ->
    anneal-prune; cliff below 500 -> hard physical decision; mismatch
    holds at 5-10% -> silicon-plausible; collapses -> precision redirect.

Gate 3 — sparse structured W_in:
    Dense Gaussian vs sparse-random (each tap -> k random nodes, k in
    {3, 6}) vs sparse-structured (each tap -> localized node group
    matching torus geometry), all with dense W_mix distributing.  ~20
    legs at the best corner.  Verdict: structured holds -> boundary spec
    is localized tap groups + dense mixer; only-dense holds -> input
    routing joins the device-budget problem.

On all-pass the plan emits the buildable spec
(``post_twin_gates_buildable_spec.json``) as the sole input to the
frozen-multistage experiment.

The module is *side-effect free at import*; only directly-invoked
functions build nets (none here) or read checkpoints (none here).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import narma_advisor_probes as npr  # noqa: E402
import narma_experiment as ne  # noqa: E402


# ---------------------------------------------------------------------------
# Locked control thresholds (flow-sparse-gates spec).
# ---------------------------------------------------------------------------

PROBE_WASHOUT: int = npr.PROBE_WASHOUT
CANONICAL_HIDDEN: int = 25  # twin node count; matches discrete-twin default
CANONICAL_T_SPAN: float = 1.0  # per-sample integration window (one NARMA step)
CANONICAL_DT_MAX: float = 0.125  # fixed-dt rule: dt <= 0.125 at every lambda*T

# Gate 1 (Heun flow-twin)
G1_LAMBDA_T: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)
G1_RADIUS: tuple[float, ...] = (0.9, 1.1)
G1_INPUT_SCALE: tuple[float, ...] = (0.1, 0.2)
G1_W_SEEDS: tuple[int, ...] = (0, 1, 2)
G1_DATA_SEEDS: tuple[int, ...] = (0, 1, 2)
G1_HARNESSES: tuple[str, ...] = ("taps8", "scalar")
# Scalar confirm leg (locked per spec): lambda*T {1, 3}, r = 1.1,
# input scale 0.2, full W-seed x data-seed product.  Record-only; it
# carries no fatal power (the taps8 main sweep owns the verdict).
G1_CONFIRM_LAMT: tuple[float, ...] = (1.0, 3.0)
G1_CONFIRM_RADIUS: tuple[float, ...] = (1.1,)
G1_CONFIRM_INPUT_SCALE: tuple[float, ...] = (0.2,)
# Per spec: holds at lambda*T<=3 with MC>=8 -> flow-viable proceed.
G1_PASS_MC_ABOVE: float = 8.0
# Per spec: only lambda*T>=10 records speed requirement; below it the
# "speed requirement" flag is dead.
G1_SPEED_REQUIRED_LAMBDA_T: float = 10.0
# Activity / engagement guards (mirrored from the discrete twin).
G1_ACTIVITY_LOW: float = 0.20
G1_ACTIVITY_HIGH: float = 0.80
G1_PRE_LINEAR_REGIME_MEDIAN: float = 0.05

# Gate 2 (sparsity + mismatch)
G2_EDGE_COUNTS: tuple[int, ...] = (100, 200, 400, 625)
G2_DENSE_EDGES: int = 625  # 25x25 dense reference
G2_SPARSE_SEEDS: tuple[int, ...] = (0, 1, 2)
G2_MISMATCH_SIGMA: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
G2_MISMATCH_DRAWS: tuple[int, ...] = (0, 1, 2)
G2_MISMATCH_W_SEEDS: tuple[int, ...] = (0, 1, 2)
# Per spec: graceful degradation to ~200 edges -> anneal-prune target.
G2_PASS_EDGE_TARGET: int = 200
# Per spec: cliff below 500 edges -> hard physical decision.
G2_CLIFF_EDGE: int = 500
# Per spec: mismatch holds at 5-10% -> silicon-plausible; collapses -> redirect.
G2_MISMATCH_PASS_LO: float = 0.05
G2_MISMATCH_FAIL_HI: float = 0.10
# "Holds" bar for Gates 2/3 (MC floor).  Tied to the discrete twin's own
# PASS_ARCH bar (MC >= 5): a corner below it has lost the echo-state
# regime, not merely degraded.  (Gate 1 keeps its stricter MC >= 8 bar.)
G2_G3_HOLDS_MC_ABOVE: float = 5.0

# Gate 3 (sparse structured W_in)
G3_K_PER_TAP: tuple[int, ...] = (3, 6)
G3_TORUS_GEOM: tuple[int, ...] = (5, 5)  # 5x5 grid; 25 nodes
# Gate-3 leg budget (~20 legs at the best corner): dense banked reference
# x 3 W_in seeds + sparse-random {3, 6} x 5 mask seeds + sparse-structured
# x 5 mask seeds = 3 + 10 + 5 = 18 legs.
G3_MASK_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
G3_DENSE_SEEDS: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# Weight / drive / pre-stats helpers (mirror the discrete twin, but as a
# separate symbol set so a future re-tune of one does not silently change
# the other).
# ---------------------------------------------------------------------------


def flow_twin_build_weights(
    *, n_nodes: int, n_in: int, w_seed: int, win_seed: int,
    target_radius: float, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``W_mix`` (Gaussian, spectral-radius normalized) and ``W_in`` (Gaussian)."""
    g_w = torch.Generator(device="cpu").manual_seed(int(w_seed))
    W_raw = torch.randn(n_nodes, n_nodes, generator=g_w, dtype=torch.float32)
    eigs = torch.linalg.eigvals(W_raw.to(torch.float64))
    rho = float(eigs.abs().max().item())
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError(f"W_mix spectral radius invalid: {rho}")
    W = (W_raw * (float(target_radius) / rho)).to(dtype=dtype, device=device)
    g_in = torch.Generator(device="cpu").manual_seed(int(win_seed))
    W_in = torch.randn(n_nodes, n_in, generator=g_in, dtype=torch.float32).to(
        dtype=dtype, device=device
    )
    return W, W_in


def flow_twin_drive(
    u_stream: torch.Tensor, harness: str, *,
    n_taps: int, input_scale: float,
) -> torch.Tensor:
    """Build the ``(T, n_in)`` drive signal for the chosen harness.

    ``taps8``: 8-tap causal delay bank, columns scaled to drive RMS = ``input_scale``.
    ``scalar``: scalar rail-mapped drive, replicated to a 1-column matrix.
    """
    if harness == "taps8":
        taps = nlc_twin_delay_bank(u_stream, n_taps=n_taps)
        rms = float(taps.pow(2).mean().sqrt().item())
        if rms > 0 and input_scale > 0:
            taps = taps * (float(input_scale) / rms)
        return taps
    if harness == "scalar":
        scaled = u_stream * float(input_scale)
        return scaled.unsqueeze(-1)
    raise ValueError(f"unknown flow-twin harness: {harness}")


def nlc_twin_delay_bank(u: torch.Tensor, n_taps: int) -> torch.Tensor:
    """Re-export of the discrete twin's delay-bank for harness parity."""
    from narma_linear_controls import twin_delay_bank
    return twin_delay_bank(u, n_taps=n_taps)


# ---------------------------------------------------------------------------
# Continuous-time Heun integration (drive held constant per sample).
# ---------------------------------------------------------------------------


def _heun_step(
    x: torch.Tensor, drive_row: torch.Tensor,
    *, W: torch.Tensor, W_in: torch.Tensor, lam: float, dt: float,
) -> torch.Tensor:
    """Single Heun substep of ``dx/dt = -lam*x + lam*tanh(W@x + W_in@d)``."""
    pre = W @ x + W_in @ drive_row
    f0 = -lam * x + lam * torch.tanh(pre)
    x_pred = x + dt * f0
    pre_pred = W @ x_pred + W_in @ drive_row
    f1 = -lam * x_pred + lam * torch.tanh(pre_pred)
    return x + 0.5 * dt * (f0 + f1)


def flow_twin_run(
    *, W: torch.Tensor, W_in: torch.Tensor, drive: torch.Tensor,
    lam: float, dt: float, x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Heun flow-twin with per-sample drive-hold and return ``(T, n)`` states.

    Per sample, integrates ``num_steps = round(t_span/dt)`` Heun substeps
    where the per-sample sub-window is implicit (``t_span = 1.0`` and the
    drive is held constant for the full sub-window).  The returned state
    is the state at the END of each sample's sub-window.
    """
    T = drive.shape[0]
    n = W.shape[0]
    dev = W.device
    dtype = W.dtype
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError(f"flow-twin dt must be positive finite, got {dt}")
    if not (0.0 < lam <= 50.0) or not math.isfinite(lam):
        raise ValueError(f"flow-twin lam must be in (0, 50], got {lam}")
    num_steps = max(1, int(round(CANONICAL_T_SPAN / dt)))
    states: list[torch.Tensor] = []
    x = x0 if x0 is not None else torch.zeros(n, dtype=dtype, device=dev)
    for t in range(T):
        d_t = drive[t]
        for _ in range(num_steps):
            x = _heun_step(
                x, d_t, W=W, W_in=W_in, lam=lam, dt=dt,
            )
        states.append(x.detach().clone())
    return torch.stack(states, dim=0)


def flow_twin_jacobian_at(
    *, W: torch.Tensor, W_in: torch.Tensor, x: torch.Tensor,
    drive_row: torch.Tensor, lam: float, dt: float,
) -> torch.Tensor:
    """Analytic per-sample Jacobian (Heun map composition).

    For one Heun substep, the predictor-corrector formula is
    ``x_pred = x + dt * f(x, d)``,
    ``x_new = x + 0.5 * dt * (f(x, d) + f(x_pred, d))``,
    with ``f(x, d) = -lam * x + lam * tanh(W@x + W_in@d)``.  Defining
    ``A(x) = df/dx = -lam*I + lam * diag(sech^2(W@x + W_in@d)) * W``,
    the per-step Jacobian (composition over ``num_steps`` sub-windows)
    is
    ``J_step = I + 0.5 * dt * A(x) + 0.5 * dt * A(x_pred) * (I + dt * A(x))``
    and ``J_total = J_step_{n-1} @ ... @ J_step_0``.
    """
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError(f"flow-twin dt must be positive finite, got {dt}")
    if not (0.0 < lam <= 50.0) or not math.isfinite(lam):
        raise ValueError(f"flow-twin lam must be in (0, 50], got {lam}")
    num_steps = max(1, int(round(CANONICAL_T_SPAN / dt)))
    eye = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    J_total = eye.clone()
    x_cur = x.detach().clone()
    for _ in range(num_steps):
        pre = W @ x_cur + W_in @ drive_row
        tanh_pre = torch.tanh(pre)
        gain = 1.0 - tanh_pre.pow(2)
        f0 = -lam * x_cur + lam * tanh_pre
        x_pred = x_cur + dt * f0
        A_x = -lam * eye + lam * (gain.unsqueeze(1) * W)
        # Per-step Heun Jacobian (predictor-corrector).
        J_step = (eye
                  + 0.5 * dt * A_x
                  + 0.5 * dt * _a_at(
                      W, W_in, x_pred, drive_row, eye, lam,
                  ) @ (eye + dt * A_x))
        J_total = J_step @ J_total
        x_cur = x_pred
    return J_total


def _a_at(
    W: torch.Tensor, W_in: torch.Tensor, x: torch.Tensor,
    drive_row: torch.Tensor, eye: torch.Tensor, lam: float,
) -> torch.Tensor:
    """Helper: compute ``A(x) = -lam*I + lam * diag(sech^2(.)) * W``.

    Used by the Heun Jacobian at the predicted state.
    """
    pre = W @ x + W_in @ drive_row
    gain = 1.0 - torch.tanh(pre).pow(2)
    return -lam * eye + lam * (gain.unsqueeze(1) * W)


def flow_twin_eigs_per_transition(
    *, W: torch.Tensor, W_in: torch.Tensor, states: torch.Tensor,
    drive: torch.Tensor, washout: int, n_samples: int,
    lam: float, dt: float,
) -> tuple[list[dict[str, float]], list[list[float]]]:
    """Eigenvalue spectra at observed transitions, flow-twin map."""
    n_nodes = W.shape[0]
    T = states.shape[0]
    if T - washout - 1 <= 0:
        raise ValueError(
            f"need T > washout+1, got T={T}, washout={washout}"
        )
    n = min(n_samples, T - washout - 1)
    indices = torch.linspace(
        washout, T - 2, steps=n,
    ).round().to(torch.long).unique().tolist()
    rows: list[dict[str, float]] = []
    eig_per: list[list[float]] = []
    for t in indices:
        x = states[t].detach()
        d = drive[t + 1]
        J = flow_twin_jacobian_at(
            W=W, W_in=W_in, x=x, drive_row=d, lam=lam, dt=dt,
        )
        try:
            eig = torch.linalg.eigvals(J)
            abs_e = eig.abs()
            if not torch.isfinite(abs_e).all():
                raise RuntimeError("non-finite flow-twin eigenvalues")
        except Exception:
            try:
                m = J.detach().to(dtype=torch.float64, device="cpu")
                eig = torch.linalg.eigvals(m)
                abs_e = eig.abs().to(dtype=torch.float32)
                if not torch.isfinite(abs_e).all():
                    raise RuntimeError("non-finite after fallback")
            except Exception:
                rows.append({
                    "transition_index": float(t),
                    "state_dim": float(n_nodes),
                    "max_abs": float("nan"),
                    "min_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "rank_proxy": float("nan"),
                })
                eig_per.append([])
                continue
        rows.append({
            "transition_index": float(t),
            "state_dim": float(n_nodes),
            "max_abs": float(abs_e.max().item()),
            "min_abs": float(abs_e.min().item()),
            "mean_abs": float(abs_e.mean().item()),
            "rank_proxy": float(npr.participation_ratio(abs_e.to(torch.float64))),
        })
        eig_per.append([float(v) for v in abs_e.tolist()])
    return rows, eig_per


def flow_twin_pre_stats(
    *, W: torch.Tensor, W_in: torch.Tensor, states: torch.Tensor,
    drive: torch.Tensor, lam: float, dt: float,
) -> dict[str, float]:
    """Stats on ``|W@x + W_in@d|`` over the trajectory (tanh-engagement guard)."""
    T = states.shape[0]
    pre_vals = []
    for t in range(T):
        pre_vals.append((W @ states[t].detach() + W_in @ drive[t]).abs())
    all_pre = torch.stack(pre_vals)
    return {
        "median_abs_pre": float(all_pre.median().item()),
        "p90_abs_pre": float(torch.quantile(all_pre.flatten(), 0.9).item()),
        "mean_abs_pre": float(all_pre.mean().item()),
    }


# ---------------------------------------------------------------------------
# Per-corner Heun flow-twin probe.
# ---------------------------------------------------------------------------


def _ridge_nrmse_r2(
    states: torch.Tensor, y: torch.Tensor, *,
    washout: int, l2: float = 1e-2,
) -> tuple[float, float]:
    X = states[washout:]
    y_w = y[washout:]
    X_aug = torch.cat(
        [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1,
    )
    XtX = X_aug.T @ X_aug + l2 * torch.eye(
        X_aug.shape[1], device=X.device,
    )
    W = torch.linalg.solve(XtX, X_aug.T @ y_w)
    pred = X_aug @ W
    return float(ne.nrmse(pred, y_w)), float(ne.r2(pred, y_w))


def flow_twin_node_tanh(
    *, order: int, seed: int, device: str, hidden_dim: int,
    n_taps: int, harness: str, radius: float, lamT: float,
    input_scale: float, w_seed: int, win_seed: int,
    n_streams: int, train_samples_per_stream: int,
    washout: int, max_delay: int, jacobian_samples: int,
    dt: float = CANONICAL_DT_MAX,
) -> dict[str, Any]:
    """Run one Heun flow-twin corner and return its instrument row + verdict."""
    if order != 10:
        raise ValueError("flow-twin is calibrated for NARMA-10 only")
    if harness not in G1_HARNESSES:
        raise ValueError(
            f"flow-twin harness must be one of {G1_HARNESSES}, got {harness!r}"
        )
    if not (0.0 < float(radius) < 2.0):
        raise ValueError(f"flow-twin radius must be in (0, 2), got {radius}")
    if not (0.0 < float(lamT) <= 50.0):
        raise ValueError(f"flow-twin lambda*T must be in (0, 50], got {lamT}")
    if not (0.0 < float(dt) <= CANONICAL_DT_MAX):
        raise ValueError(
            f"flow-twin dt must satisfy 0 < dt <= {CANONICAL_DT_MAX} "
            f"(fixed-dt rule), got {dt}"
        )
    lam = float(lamT) / float(CANONICAL_T_SPAN)
    dt = float(dt)
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=order, input_scale=1.0,
    )
    drive = flow_twin_drive(
        u_scaled, harness, n_taps=n_taps, input_scale=input_scale,
    )
    drive = drive.to(device)
    y_dev = y_stream.to(device)
    dev = torch.device(device)
    dtype = torch.float32
    n_in = drive.shape[1]
    W, W_in = flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=n_in, w_seed=w_seed, win_seed=win_seed,
        target_radius=radius, dtype=dtype, device=dev,
    )
    states = flow_twin_run(
        W=W, W_in=W_in, drive=drive, lam=lam, dt=dt,
    )
    pre_stats = flow_twin_pre_stats(
        W=W, W_in=W_in, states=states, drive=drive, lam=lam, dt=dt,
    )
    tanh_engaged = pre_stats["median_abs_pre"] >= G1_PRE_LINEAR_REGIME_MEDIAN
    mc_per, mc_total = nlc_safe_per_delay_mc(
        states, drive[:, 0] if harness == "scalar" else u_scaled.to(device),
        washout=washout, max_delay=max_delay,
    )
    ridge_nrmse, ridge_r2 = _ridge_nrmse_r2(
        states, y_dev, washout=washout,
    )
    jac_rows, _eig_per = flow_twin_eigs_per_transition(
        W=W, W_in=W_in, states=states, drive=drive,
        washout=washout, n_samples=jacobian_samples, lam=lam, dt=dt,
    )
    state_pr = _safe_participation_ratio_local(states[washout:])
    activity = float(states[washout:].abs().mean().item())
    activity_in_band = bool(
        G1_ACTIVITY_LOW <= activity <= G1_ACTIVITY_HIGH
    )
    states_finite = bool(torch.isfinite(states).all().item())
    # ESN parity ref on identical stream (no training anywhere).
    esn_in = (u_scaled * float(input_scale)).to("cpu")
    esn = ne.ESN(
        n_reservoir=hidden_dim, spectral_radius=0.9,
        input_scaling=1.0, leak=1.0,
        ridge_l2=1e-2, seed=int(seed),
    )
    esn.fit(esn_in, y_stream.to("cpu"))
    esn_states = esn._run(esn_in).to(device)
    _, esn_mc = nlc_safe_per_delay_mc(
        esn_states, esn_in.squeeze(-1).to(device),
        washout=washout, max_delay=max_delay,
    )
    esn_pred = esn_states @ esn.readout_W.to(device) + esn.readout_b.to(device)
    esn_nrmse = float(ne.nrmse(esn_pred[washout:], y_dev[washout:]))
    # Raw-delay parity diagnostic (mirrors the discrete twin: ridge on the
    # input-only delay line through the identical pipeline, then the
    # matched delta twin-minus-raw).
    from narma_linear_controls import (
        R0_RAW_DELAY_TAPS, _raw_delay_ridge,
    )
    raw = _raw_delay_ridge(
        u_scaled.to(device), y_dev,
        n_taps=R0_RAW_DELAY_TAPS, washout=washout,
    )
    matched_raw_delay_nrmse = float(raw["nrmse"])
    matched_parity_delta = float(ridge_nrmse - raw["nrmse"])
    config_tag = (
        f"flowtwin_{harness}_seed{seed}_{device}_h{hidden_dim}"
        f"_n{n_in}_r{radius:g}_lamT{lamT:g}_is{input_scale:g}"
        f"_ws{w_seed}_wis{win_seed}_washout{washout}_dt{dt:g}"
    )
    return {
        "config_tag": config_tag,
        "harness": harness, "seed": seed, "device": device,
        "hidden_dim": hidden_dim, "n_taps": int(n_taps),
        "n_in": int(n_in),
        "radius": float(radius), "lamT": float(lamT),
        "input_scale": float(input_scale),
        "w_seed": int(w_seed), "win_seed": int(win_seed),
        "dt": float(dt), "num_steps": int(round(CANONICAL_T_SPAN / dt)),
        "washout": washout, "max_delay": max_delay,
        "mc_total": float(mc_total),
        "mc_per_delay": [float(v) for v in mc_per],
        "ridge_nrmse": float(ridge_nrmse),
        "ridge_r2": float(ridge_r2),
        "state_pr": float(state_pr),
        "activity": float(activity),
        "activity_in_band": bool(activity_in_band),
        "tanh_engaged": bool(tanh_engaged),
        "states_finite": bool(states_finite),
        "esn_nrmse": float(esn_nrmse),
        "esn_mc_total": float(esn_mc),
        "matched_raw_delay_nrmse": matched_raw_delay_nrmse,
        "matched_parity_delta": matched_parity_delta,
        "pre_stats": pre_stats,
        "jac_max_abs": float(max(r["max_abs"] for r in jac_rows)),
        "jac_min_abs": float(min(r["min_abs"] for r in jac_rows)),
        "jac_mean_abs": float(
            sum(r["mean_abs"] for r in jac_rows) / max(len(jac_rows), 1)
        ),
        "jac_rank_proxy": float(max(r["rank_proxy"] for r in jac_rows)),
        "note": (
            f"flow-twin Heun, harness={harness}, radius={radius}, "
            f"lambda*T={lamT}, input_scale={input_scale}, dt={dt}, "
            f"num_steps/sample={int(round(CANONICAL_T_SPAN / dt))}; "
            f"states[-inf,inf] rail_frac=N/A (tanh bound N/A); "
            f"pre_median={pre_stats['median_abs_pre']:.4f}, "
            f"engaged={tanh_engaged}"
        ),
    }


def _safe_participation_ratio_local(states: torch.Tensor) -> float:
    """Local PR for eval-only use; mirrors npr.participation_ratio behavior."""
    if states.dim() != 2 or states.shape[0] < 2:
        return float("nan")
    s = states - states.mean(dim=0, keepdim=True)
    cov = (s.T @ s) / max(s.shape[0] - 1, 1)
    try:
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except Exception:
        return float("nan")
    s_val = float(eig.sum().item())
    sq = float(eig.pow(2).sum().item())
    if sq <= 1e-12:
        return float("nan")
    return (s_val * s_val) / sq


def nlc_safe_per_delay_mc(
    states: torch.Tensor, targets: torch.Tensor, *,
    washout: int, max_delay: int, l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Washout-corrected per-delay MC with SVD fallback (call into nlc)."""
    from narma_linear_controls import _per_delay_mc
    return _per_delay_mc(
        states, targets, washout=washout, max_delay=max_delay,
        use_svd_fallback=True, ridge_l2=l2,
    )


# ---------------------------------------------------------------------------
# Gate 1 sweep driver + verdict.
# ---------------------------------------------------------------------------


@dataclass
class Gate1Row:
    config_tag: str
    harness: str
    seed: int
    hidden_dim: int
    n_taps: int
    radius: float
    lamT: float
    input_scale: float
    w_seed: int
    win_seed: int
    dt: float
    num_steps: int
    mc_total: float
    ridge_nrmse: float
    ridge_r2: float
    state_pr: float
    activity: float
    tanh_engaged: bool
    states_finite: bool
    esn_nrmse: float
    esn_mc_total: float
    matched_raw_delay_nrmse: float
    matched_parity_delta: float
    pass_mc: bool
    pre_median: float
    note: str


def _gate1_row(report: dict[str, Any]) -> Gate1Row:
    engaged = bool(report["tanh_engaged"])
    finite = bool(report["states_finite"])
    mc = float(report["mc_total"])
    return Gate1Row(
        config_tag=report["config_tag"],
        harness=report["harness"], seed=report["seed"],
        hidden_dim=report["hidden_dim"],
        n_taps=int(report["n_taps"]),
        radius=report["radius"], lamT=report["lamT"],
        input_scale=report["input_scale"],
        w_seed=report["w_seed"], win_seed=report["win_seed"],
        dt=report["dt"], num_steps=report["num_steps"],
        mc_total=mc,
        ridge_nrmse=report["ridge_nrmse"],
        ridge_r2=report["ridge_r2"],
        state_pr=report["state_pr"],
        activity=report["activity"],
        tanh_engaged=engaged,
        states_finite=finite,
        esn_nrmse=report["esn_nrmse"],
        esn_mc_total=report["esn_mc_total"],
        matched_raw_delay_nrmse=float(report["matched_raw_delay_nrmse"]),
        matched_parity_delta=float(report["matched_parity_delta"]),
        pass_mc=bool(engaged and finite and mc >= G1_PASS_MC_ABOVE),
        pre_median=report["pre_stats"]["median_abs_pre"],
        note=report["note"],
    )


def gate1_verdict(rows: list[Gate1Row]) -> dict[str, Any]:
    """Apply pre-registered Gate-1 fatal criteria to a list of corner rows.

    Verdicts (per spec):
      - ``FLOW_VIABLE``: holds at ``lambda*T <= 3`` with MC >= 8.
      - ``SPEED_REQUIRED``: holds only at ``lambda*T >= 10``.
      - ``DIGITAL_ONLY``: collapses everywhere -> redirect program.

    Aggregation is the *best* (max MC) engaged-and-finite row per
    ``lambda*T`` over the W-seeds, data-seeds, radii, and input scales
    (single-seed eval-only screening numbers; the gate screens, it does
    not certify).  The emitted operating point (``viable_lamT``) is
    always the fastest viable ``lambda*T`` *within the viable regime*:
    a mixed result where both 3 and 10 hold is FLOW_VIABLE at 3.0, never
    at 10.0 (only a holds-exclusively-at->=10 result records the speed
    requirement).
    """
    by_lam: dict[float, list[Gate1Row]] = {lt: [] for lt in G1_LAMBDA_T}
    for r in rows:
        if r.lamT in by_lam:
            by_lam[r.lamT].append(r)
    surviving: dict[float, float] = {}
    for lt, grp in by_lam.items():
        if not grp:
            continue
        engaged_finite = [
            r for r in grp if r.tanh_engaged and r.states_finite
        ]
        if not engaged_finite:
            continue
        best_mc = max(r.mc_total for r in engaged_finite)
        surviving[lt] = best_mc
    if not surviving:
        verdict = "DIGITAL_ONLY"
        viable_lamT = None
    else:
        viable = sorted(lt for lt, mc in surviving.items()
                        if mc >= G1_PASS_MC_ABOVE)
        if viable and max(viable) <= G1_SPEED_REQUIRED_LAMBDA_T - 1e-6:
            verdict = "FLOW_VIABLE"
            viable_lamT = max(viable)
        elif viable and min(viable) >= G1_SPEED_REQUIRED_LAMBDA_T:
            verdict = "SPEED_REQUIRED"
            viable_lamT = min(viable)
        else:
            # Mixed: at least one viable lambda*T below 10 (the
            # all->=10 case is caught above), so this is FLOW_VIABLE
            # with the operating point pinned inside the viable
            # regime, never at 10.0.
            if viable:
                verdict = "FLOW_VIABLE"
                low = [lt for lt in viable
                       if lt <= G1_SPEED_REQUIRED_LAMBDA_T - 1e-6]
                viable_lamT = max(low) if low else max(viable)
            else:
                verdict = "DIGITAL_ONLY"
                viable_lamT = None
    return {
        "verdict": verdict,
        "viable_lamT": (float(viable_lamT)
                        if viable_lamT is not None else None),
        "pass_mc_above": G1_PASS_MC_ABOVE,
        "speed_required_lamT": G1_SPEED_REQUIRED_LAMBDA_T,
        "surviving_by_lamT": {str(lt): float(mc)
                              for lt, mc in surviving.items()},
    }


def _gate1_verdict_meta(
    *, rows: list[Gate1Row], verdict: dict[str, Any], elapsed: float,
    harness: str, n_taps: int, hidden_dim: int, washout: int,
    max_delay: int, dt: float,
    radius_grid: tuple[float, ...], lamT_grid: tuple[float, ...],
    input_scale_grid: tuple[float, ...],
    w_seeds: tuple[int, ...], data_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Attach sweep provenance to a Gate-1 verdict dict (shared by CLI legs)."""
    verdict.update({
        "harness": harness, "n_taps": n_taps,
        "hidden_dim": hidden_dim, "washout": washout,
        "max_delay": max_delay, "dt": float(dt),
        "n_corners": len(rows),
        "n_data_seeds": len(data_seeds),
        "n_w_seeds": len(w_seeds),
        "radius_grid": list(radius_grid),
        "lamT_grid": list(lamT_grid),
        "input_scale_grid": list(input_scale_grid),
        "elapsed_s": float(elapsed),
        "note": (
            f"Gate-1 Heun flow-twin sweep, harness={harness}, "
            f"hidden_dim={hidden_dim}, washout={washout}; "
            f"n_corners={len(rows)}; "
            f"verdict={verdict.get('verdict')}"
        ),
    })
    return verdict


def run_gate1_sweep(
    *, harness: str, n_taps: int, hidden_dim: int,
    n_streams: int, train_samples_per_stream: int,
    washout: int, max_delay: int, jacobian_samples: int,
    device: str = "cpu",
    dt: float = CANONICAL_DT_MAX,
    radius_grid: tuple[float, ...] = G1_RADIUS,
    input_scale_grid: tuple[float, ...] = G1_INPUT_SCALE,
    lamT_grid: tuple[float, ...] = G1_LAMBDA_T,
    w_seeds: tuple[int, ...] = G1_W_SEEDS,
    data_seeds: tuple[int, ...] = G1_DATA_SEEDS,
    max_corners: int | None = None,
) -> tuple[list[Gate1Row], dict[str, Any], float]:
    """Run the Gate-1 Heun flow-twin sweep and return rows + verdict + elapsed.

    ``max_corners`` caps the Cartesian grid *before* running (traversal
    prefix in data-seed-major order), so smoke caps save compute; the
    verdict is always computed on exactly the rows returned.
    """
    t0 = time.time()
    corners: list[tuple[int, int, float, float, float]] = []
    for ds in data_seeds:
        for ws in w_seeds:
            for lt in lamT_grid:
                for r in radius_grid:
                    for is_ in input_scale_grid:
                        corners.append((ds, ws, lt, r, is_))
    if max_corners is not None and max_corners > 0:
        corners = corners[: max_corners]
    rows: list[Gate1Row] = []
    for ds, ws, lt, r, is_ in corners:
        report = flow_twin_node_tanh(
            order=10, seed=ds, device=device,
            hidden_dim=hidden_dim, n_taps=n_taps,
            harness=harness, radius=float(r), lamT=float(lt),
            input_scale=float(is_), w_seed=ws, win_seed=ws,
            n_streams=n_streams,
            train_samples_per_stream=train_samples_per_stream,
            washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            dt=dt,
        )
        rows.append(_gate1_row(report))
    elapsed = float(time.time() - t0)
    verdict = gate1_verdict(rows)
    _gate1_verdict_meta(
        rows=rows, verdict=verdict, elapsed=elapsed,
        harness=harness, n_taps=n_taps, hidden_dim=hidden_dim,
        washout=washout, max_delay=max_delay, dt=dt,
        radius_grid=radius_grid, lamT_grid=lamT_grid,
        input_scale_grid=input_scale_grid,
        w_seeds=w_seeds, data_seeds=data_seeds,
    )
    return rows, verdict, elapsed


# ---------------------------------------------------------------------------
# Gate 2 — sparsity mask builders.
# ---------------------------------------------------------------------------


def random_sparse_mask(
    n: int, m: int, *, target_nnz: int, seed: int,
) -> torch.Tensor:
    """Build a (n, m) bool mask with exactly ``target_nnz`` True entries."""
    if target_nnz < 0 or target_nnz > n * m:
        raise ValueError(
            f"random_sparse_mask: target_nnz={target_nnz} out of range "
            f"for ({n}, {m})"
        )
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n * m, generator=g).tolist()
    flat = torch.zeros(n * m, dtype=torch.bool)
    flat[perm[:target_nnz]] = True
    return flat.view(n, m)


def magnitude_prune_mask(
    W: torch.Tensor, *, target_nnz: int,
) -> torch.Tensor:
    """Build a bool mask keeping the top-``target_nnz`` entries by ``|W|``."""
    n, m = W.shape
    if target_nnz < 0 or target_nnz > n * m:
        raise ValueError(
            f"magnitude_prune_mask: target_nnz={target_nnz} out of range "
            f"for ({n}, {m})"
        )
    abs_W = W.abs().flatten()
    if target_nnz == 0:
        return torch.zeros(n, m, dtype=torch.bool)
    kth = torch.kthvalue(abs_W, n * m - target_nnz + 1).values.item()
    mask = (abs_W >= kth).view(n, m)
    excess = int(mask.sum().item()) - target_nnz
    if excess > 0:
        tied = (abs_W == kth).nonzero(as_tuple=False)
        if tied.shape[0] >= excess:
            drop = tied[-excess:]
            mask.view(-1)[drop[:, 0]] = False
    return mask


def restore_spectral_radius(
    W: torch.Tensor, *, target_radius: float,
) -> torch.Tensor:
    """Rescale ``W`` to a target spectral radius."""
    if target_radius <= 0 or not math.isfinite(target_radius):
        raise ValueError(
            f"restore_spectral_radius: target_radius={target_radius}"
        )
    eigs = torch.linalg.eigvals(W.to(torch.float64))
    rho = float(eigs.abs().max().item())
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError(
            f"restore_spectral_radius: source radius={rho} invalid"
        )
    return W * (float(target_radius) / rho)


def apply_mask_with_radius_restore(
    W: torch.Tensor, mask: torch.Tensor, *, target_radius: float,
) -> torch.Tensor:
    """Zero entries outside the mask, then restore the spectral radius."""
    W_masked = W * mask.to(device=W.device, dtype=W.dtype)
    return restore_spectral_radius(W_masked, target_radius=target_radius)


def low_rank_plus_sparse_W(
    *, n_nodes: int, n_in: int, w_seed: int, win_seed: int,
    rank: int, ring_local_count: int, target_radius: float,
    dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``W = U V^T + R`` where ``U V^T`` is a rank-``rank``
    global-rail term and ``R`` is a local ring with ``ring_local_count``
    off-diagonal entries.

    The *effective* matrix is dense (``U V^T`` fills every entry); the
    device-budget reading is in *stored* parameters:
    ``2 * rank * n_nodes`` rail factors + ``ring_local_count`` sparse
    entries.  Callers must label the row with the stored count, NOT the
    effective nnz, and must NOT pool it with same-nnz sparse buckets.
    """
    if rank < 1:
        raise ValueError(f"low-rank rank must be >= 1, got {rank}")
    g_w = torch.Generator(device="cpu").manual_seed(int(w_seed))
    U = torch.randn(n_nodes, rank, generator=g_w, dtype=torch.float32)
    V = torch.randn(n_nodes, rank, generator=g_w, dtype=torch.float32)
    R = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    g_r = torch.Generator(device="cpu").manual_seed(int(w_seed + 1))
    nnz = min(ring_local_count, n_nodes * (n_nodes - 1))
    if nnz > 0:
        off = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    off.append((i, j))
        perm = torch.randperm(len(off), generator=g_r).tolist()
        for k in perm[:nnz]:
            i, j = off[k]
            R[i, j] = float(torch.randn((), generator=g_r).item())
    W_raw = U @ V.T + R
    eigs = torch.linalg.eigvals(W_raw.to(torch.float64))
    rho = float(eigs.abs().max().item())
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError(f"low-rank+sparse radius invalid: {rho}")
    W = (W_raw * (float(target_radius) / rho)).to(dtype=dtype, device=device)
    g_in = torch.Generator(device="cpu").manual_seed(int(win_seed))
    W_in = torch.randn(n_nodes, n_in, generator=g_in, dtype=torch.float32).to(
        dtype=dtype, device=device,
    )
    return W, W_in


# ---------------------------------------------------------------------------
# Gate 2 — mismatch perturbation + heterogeneous operating-point row.
# ---------------------------------------------------------------------------


def mismatch_perturb(
    W: torch.Tensor, *, sigma: float, seed: int,
) -> torch.Tensor:
    """Multiplicative Gaussian: ``W * (1 + sigma * N(0, 1))``."""
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError(f"mismatch_perturb: sigma={sigma}")
    g = torch.Generator(device=W.device).manual_seed(int(seed))
    noise = torch.randn(W.shape, generator=g, dtype=W.dtype, device=W.device)
    return W * (1.0 + float(sigma) * noise)


def mismatch_perturb_pair(
    W_mix: torch.Tensor, W_in: torch.Tensor, *,
    sigma: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent multiplicative Gaussian on W_mix and W_in sharing one seed."""
    g_mix = torch.Generator(device=W_mix.device).manual_seed(int(seed))
    g_in = torch.Generator(device=W_in.device).manual_seed(int(seed) + 1)
    n_mix = torch.randn(W_mix.shape, generator=g_mix,
                        dtype=W_mix.dtype, device=W_mix.device)
    n_in = torch.randn(W_in.shape, generator=g_in,
                       dtype=W_in.dtype, device=W_in.device)
    return (
        W_mix * (1.0 + float(sigma) * n_mix),
        W_in * (1.0 + float(sigma) * n_in),
    )


def heterogeneous_tanh_jitter(
    *, n_nodes: int, gain_jitter: float, bias_jitter: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node tanh gain and bias jitter (operating-point heterogeneity)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    gain = 1.0 + float(gain_jitter) * torch.randn(
        n_nodes, generator=g, dtype=torch.float32,
    )
    gain = gain.clamp_min(1e-3)
    bias = float(bias_jitter) * torch.randn(
        n_nodes, generator=g, dtype=torch.float32,
    )
    return gain, bias


# ---------------------------------------------------------------------------
# Gate 3 — sparse structured W_in.
# ---------------------------------------------------------------------------


def _node_groups_torus(
    n_nodes: int, geom: tuple[int, ...], taps_per_group: int,
    seed: int,
) -> list[list[int]]:
    """For each tap, build a localized node group on a torus grid.

    The grid is ``geom = (g1, g2, ...)`` and ``prod(geom) == n_nodes``.
    Each tap covers a contiguous block of size ``taps_per_group`` on the
    flattened grid.  When all indices have been consumed, the next group
    wraps around (no infinite loop): the same locality pattern repeats
    over a fixed stride.
    """
    prod = 1
    for d in geom:
        prod *= d
    if prod != n_nodes:
        raise ValueError(
            f"_node_groups_torus: geom={geom} product {prod} != n_nodes "
            f"{n_nodes}"
        )
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # Sample one start offset per tap; group is a contiguous block
    # wrapping the ring with that start.
    starts = torch.randint(0, n_nodes, (n_nodes,), generator=g).tolist()
    groups: list[list[int]] = []
    for s in starts:
        s_int = int(s)
        grp = [(s_int + k) % n_nodes for k in range(taps_per_group)]
        groups.append(grp)
    return groups


def dense_W_in(
    *, n_nodes: int, n_in: int, seed: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Dense Gaussian W_in (banked reference)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(n_nodes, n_in, generator=g, dtype=torch.float32).to(
        dtype=dtype, device=device,
    )


def sparse_random_W_in(
    *, n_nodes: int, n_in: int, k_per_tap: int, seed: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Each tap -> k random nodes; same scale as dense."""
    if not (1 <= int(k_per_tap) <= n_nodes):
        raise ValueError(
            f"sparse_random_W_in: k_per_tap={k_per_tap} must satisfy "
            f"1 <= k <= n_nodes={n_nodes}"
        )
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.zeros(n_nodes, n_in, dtype=torch.float32)
    for j in range(n_in):
        idx = torch.randperm(n_nodes, generator=g)[:k_per_tap]
        vals = torch.randn(k_per_tap, generator=g, dtype=torch.float32)
        W[idx, j] = vals
    # Rescale so per-column RMS == sqrt(n_nodes/k_per_tap) (preserves
    # total drive power under sparsity).
    for j in range(n_in):
        rms = float(W[:, j].pow(2).mean().sqrt().item())
        if rms > 0:
            W[:, j] = W[:, j] * (math.sqrt(n_nodes / k_per_tap) / rms)
    return W.to(dtype=dtype, device=device)


def sparse_structured_W_in(
    *, n_nodes: int, n_in: int, geom: tuple[int, ...], seed: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Each tap -> a localized node group on a torus grid."""
    taps_per_group = max(1, n_nodes // max(n_in, 1))
    if taps_per_group >= n_nodes:
        raise ValueError(
            f"sparse_structured_W_in: taps_per_group={taps_per_group} "
            f"covers all n_nodes={n_nodes} (degenerate: structured == dense; "
            f"need n_in >= 2 for a sparse group)"
        )
    groups = _node_groups_torus(n_nodes, geom, taps_per_group, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.zeros(n_nodes, n_in, dtype=torch.float32)
    for j in range(n_in):
        grp = groups[j % len(groups)]
        vals = torch.randn(len(grp), generator=g, dtype=torch.float32)
        for k, idx in enumerate(grp):
            W[idx, j] = vals[k]
    for j in range(n_in):
        rms = float(W[:, j].pow(2).mean().sqrt().item())
        if rms > 0:
            W[:, j] = W[:, j] * (math.sqrt(n_nodes / taps_per_group) / rms)
    return W.to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Gate 2 / Gate 3 row schemas + verdict.
# ---------------------------------------------------------------------------


@dataclass
class GateRow:
    config_tag: str
    gate: str  # "sparsity" | "mismatch" | "w_in"
    leg: str
    n_edges: int
    n_in: int
    mc_total: float
    ridge_nrmse: float
    state_pr: float
    tanh_engaged: bool
    states_finite: bool
    pass_mc: bool
    note: str = ""


def _gate_row_pass_mc(mc_total: float, engaged: bool, finite: bool) -> bool:
    """Per-row PASS flag for Gate 2/3 rows (twin PASS_ARCH MC bar)."""
    return bool(
        engaged and finite and math.isfinite(mc_total)
        and mc_total >= G2_G3_HOLDS_MC_ABOVE
    )


def _sparsity_row_from_twin(
    *, W: torch.Tensor, W_in: torch.Tensor, hidden_dim: int,
    harness: str, n_taps: int, seed: int, device: str, n_in: int,
    radius: float, lamT: float, input_scale: float,
    w_seed: int, win_seed: int,
    n_streams: int, train_samples_per_stream: int, washout: int,
    max_delay: int, gate: str, leg: str, n_edges: int, note: str,
) -> GateRow:
    """Run the discrete twin on (possibly masked) W and W_in and report.

    The discrete map ``x' = tanh(W@x + W_in@d)`` (leak ``a=1``) is the
    fixed-point iteration of the flow twin's steady-state equation
    ``x = tanh(W@x + W_in@d)``: it screens the *structural* robustness
    (sparsity / mismatch / W_in geometry) of the relaxed per-sample map
    at matched spectral scale, ~10x cheaper than re-integrating every
    structural variant through Heun.  The ``lambda*T`` operating point
    itself is owned by Gate 1; here ``lamT``/``radius`` are provenance
    (they record which Gate-1 corner selected this leg) and the tag.
    """
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=10, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=10, input_scale=1.0,
    )
    drive = flow_twin_drive(
        u_scaled, harness, n_taps=n_taps, input_scale=input_scale,
    )
    drive = drive.to(device)
    y_dev = y_stream.to(device)
    from narma_linear_controls import _twin_run, _twin_pre_stats
    states = _twin_run(W=W, W_in=W_in, drive=drive, a=1.0)
    pre = _twin_pre_stats(W=W, W_in=W_in, states=states, drive=drive)
    engaged = bool(pre["median_abs_pre"] >= G1_PRE_LINEAR_REGIME_MEDIAN)
    finite = bool(torch.isfinite(states).all().item())
    mc_per, mc_total = nlc_safe_per_delay_mc(
        states, drive[:, 0] if harness == "scalar" else u_scaled.to(device),
        washout=washout, max_delay=max_delay,
    )
    ridge_nrmse, _ = _ridge_nrmse_r2(
        states, y_dev, washout=washout,
    )
    state_pr = _safe_participation_ratio_local(states[washout:])
    config_tag = (
        f"g2_{leg}_seed{seed}_h{hidden_dim}_n{n_in}_e{n_edges}"
        f"_r{radius:g}_lamT{lamT:g}_ws{w_seed}"
    )
    return GateRow(
        config_tag=config_tag,
        gate=gate, leg=leg, n_edges=n_edges, n_in=n_in,
        mc_total=float(mc_total), ridge_nrmse=float(ridge_nrmse),
        state_pr=float(state_pr),
        tanh_engaged=engaged, states_finite=finite,
        pass_mc=_gate_row_pass_mc(float(mc_total), engaged, finite),
        note=note,
    )


def run_gate2_sparsity_sweep(
    *, hidden_dim: int, harness: str, n_taps: int, n_streams: int,
    train_samples_per_stream: int, washout: int, max_delay: int,
    lamT_best: float, input_scale: float,
    radius_best: float = 1.0,
    device: str = "cpu",
    edge_counts: tuple[int, ...] = G2_EDGE_COUNTS,
    sparse_seeds: tuple[int, ...] = G2_SPARSE_SEEDS,
) -> tuple[list[GateRow], dict[str, Any]]:
    """Run the Gate-2 sparsity sweep: random-sparse x mask-seeds at each
    edge count, magnitude-prune-from-dense at each edge count, plus one
    low-rank+sparse row.

    All masks restore to ``radius_best`` (the Gate-1 best-corner radius):
    unrescaled pruning is a protocol violation (it confounds sparsity
    with spectral collapse).
    """
    rows: list[GateRow] = []
    dev = torch.device(device)
    n_in = 8 if harness == "taps8" else 1
    # Build the dense reference W_mix once (masked to each count).
    W_dense, W_in = flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=n_in, w_seed=0, win_seed=0,
        target_radius=radius_best, dtype=torch.float32, device=dev,
    )
    for ec in edge_counts:
        # 1) Random-sparse x mask-seeds.
        for ms in sparse_seeds:
            mask = random_sparse_mask(
                hidden_dim, hidden_dim, target_nnz=ec, seed=ms,
            )
            W_r = apply_mask_with_radius_restore(
                W_dense, mask, target_radius=radius_best,
            )
            rows.append(_sparsity_row_from_twin(
                W=W_r, W_in=W_in, hidden_dim=hidden_dim,
                harness=harness, n_taps=n_taps, seed=0,
                device=device, n_in=n_in, radius=radius_best,
                lamT=lamT_best,
                input_scale=input_scale,
                w_seed=0, win_seed=0, n_streams=n_streams,
                train_samples_per_stream=train_samples_per_stream,
                washout=washout, max_delay=max_delay,
                gate="sparsity",
                leg=f"random_sparse_e{ec}_ms{ms}",
                n_edges=int(mask.sum().item()),
                note=f"random-sparse mask, edges={ec}, mask_seed={ms}",
            ))
        # 2) Magnitude-prune-from-dense (one per edge count, no mask seed).
        if ec < G2_DENSE_EDGES:
            mask_mp = magnitude_prune_mask(W_dense, target_nnz=ec)
            W_mp = apply_mask_with_radius_restore(
                W_dense, mask_mp, target_radius=radius_best,
            )
            rows.append(_sparsity_row_from_twin(
                W=W_mp, W_in=W_in, hidden_dim=hidden_dim,
                harness=harness, n_taps=n_taps, seed=0,
                device=device, n_in=n_in, radius=radius_best,
                lamT=lamT_best,
                input_scale=input_scale,
                w_seed=0, win_seed=0, n_streams=n_streams,
                train_samples_per_stream=train_samples_per_stream,
                washout=washout, max_delay=max_delay,
                gate="sparsity",
                leg=f"mag_prune_e{ec}",
                n_edges=int(mask_mp.sum().item()),
                note=f"magnitude-prune-from-dense, edges={ec}, "
                     f"radius restored to {radius_best:g}",
            ))
        else:
            # Dense reference row at the dense count.
            rows.append(_sparsity_row_from_twin(
                W=W_dense, W_in=W_in, hidden_dim=hidden_dim,
                harness=harness, n_taps=n_taps, seed=0,
                device=device, n_in=n_in, radius=radius_best,
                lamT=lamT_best,
                input_scale=input_scale,
                w_seed=0, win_seed=0, n_streams=n_streams,
                train_samples_per_stream=train_samples_per_stream,
                washout=washout, max_delay=max_delay,
                gate="sparsity",
                leg=f"dense_e{ec}",
                n_edges=int(W_dense.numel()),
                note="dense reference row",
            ))
    # 3) Low-rank + sparse row.  The effective matrix is dense, so the row
    # is labeled with its STORED parameter count (rank-2 factors +
    # ring), which forms its own verdict bucket and never pools with the
    # same-nnz sparse rows.
    lr_rank, lr_ring = 2, 200
    W_lr, W_in_lr = low_rank_plus_sparse_W(
        n_nodes=hidden_dim, n_in=n_in, w_seed=0, win_seed=0,
        rank=lr_rank, ring_local_count=lr_ring,
        target_radius=radius_best,
        dtype=torch.float32, device=dev,
    )
    n_edges_lr = 2 * lr_rank * hidden_dim + lr_ring
    rows.append(_sparsity_row_from_twin(
        W=W_lr, W_in=W_in_lr, hidden_dim=hidden_dim,
        harness=harness, n_taps=n_taps, seed=0,
        device=device, n_in=n_in, radius=radius_best,
        lamT=lamT_best,
        input_scale=input_scale,
        w_seed=0, win_seed=0, n_streams=n_streams,
        train_samples_per_stream=train_samples_per_stream,
        washout=washout, max_delay=max_delay,
        gate="sparsity",
        leg="low_rank_sparse_r2_ring200",
        n_edges=n_edges_lr,
        note="rank-2 global rails + 200-edge local ring "
             f"(stored params={n_edges_lr}, effective matrix dense)",
    ))
    verdict = gate2_sparsity_verdict(rows)
    return rows, verdict


def gate2_sparsity_verdict(rows: list[GateRow]) -> dict[str, Any]:
    """Apply pre-registered Gate-2 sparsity fatal criteria.

    Verdict: graceful degradation to ~200 edges -> ANNEAL_PRUNE (target
    edge count returned).  Cliff below 500 edges -> CLIFF (hard physical
    decision before any training code), even when the dense/400 buckets
    still hold.  "Holds" means bucket-mean MC >= G2_G3_HOLDS_MC_ABOVE
    (twin PASS_ARCH bar).
    """
    floor = G2_G3_HOLDS_MC_ABOVE
    by_count: dict[int, list[float]] = {}
    for r in rows:
        by_count.setdefault(r.n_edges, []).append(r.mc_total)
    means = {ec: (sum(v) / len(v) if v else float("nan"))
             for ec, v in by_count.items()}
    finite = {ec: mc for ec, mc in means.items() if math.isfinite(mc)}
    if not finite:
        verdict: str = "CLIFF"
        target: int | None = None
        best_ec = next(iter(means)) if means else 0
    elif (G2_PASS_EDGE_TARGET in finite
            and finite[G2_PASS_EDGE_TARGET] >= floor):
        verdict = "ANNEAL_PRUNE"
        target = G2_PASS_EDGE_TARGET
        best_ec = max(finite, key=lambda ec: finite[ec])
    else:
        sub500 = {ec: mc for ec, mc in finite.items()
                  if ec < G2_CLIFF_EDGE}
        if sub500 and all(mc < floor for mc in sub500.values()):
            # Cliff below 500 edges: every sub-500 bucket lost the
            # regime -> hard physical decision, even if dense holds.
            verdict = "CLIFF"
            target = None
            best_ec = max(finite, key=lambda ec: finite[ec])
        else:
            cleared = [ec for ec, mc in finite.items() if mc >= floor]
            if cleared:
                verdict = "ANNEAL_PRUNE"
                target = min(cleared)
            else:
                verdict = "CLIFF"
                target = None
            best_ec = max(finite, key=lambda ec: finite[ec])
    return {
        "verdict": verdict,
        "edge_target": target,
        "cliff_edge": G2_CLIFF_EDGE,
        "pass_edge_target": G2_PASS_EDGE_TARGET,
        "holds_mc_above": floor,
        "mean_mc_by_edge_count": {str(ec): float(mc)
                                  for ec, mc in means.items()},
        "best_edge_count": int(best_ec),
    }


def run_gate2_mismatch_sweep(
    *, hidden_dim: int, harness: str, n_taps: int, n_streams: int,
    train_samples_per_stream: int, washout: int, max_delay: int,
    lamT_best: float, input_scale: float,
    radius_best: float = 1.0,
    device: str = "cpu",
    sigma_grid: tuple[float, ...] = G2_MISMATCH_SIGMA,
    draws: tuple[int, ...] = G2_MISMATCH_DRAWS,
    w_seeds: tuple[int, ...] = G2_MISMATCH_W_SEEDS,
) -> tuple[list[GateRow], dict[str, Any]]:
    """Run the Gate-2 mismatch sweep: multiplicative Gaussian on W_mix
    AND W_in at the best Gate-1 corner, plus a heterogeneous-ops row.
    """
    rows: list[GateRow] = []
    dev = torch.device(device)
    n_in = 8 if harness == "taps8" else 1
    for ws in w_seeds:
        W_dense, W_in = flow_twin_build_weights(
            n_nodes=hidden_dim, n_in=n_in, w_seed=ws, win_seed=ws,
            target_radius=radius_best, dtype=torch.float32, device=dev,
        )
        for sigma in sigma_grid:
            for dr in draws:
                W_p, W_in_p = mismatch_perturb_pair(
                    W_dense, W_in, sigma=sigma, seed=dr,
                )
                rows.append(_sparsity_row_from_twin(
                    W=W_p, W_in=W_in_p, hidden_dim=hidden_dim,
                    harness=harness, n_taps=n_taps, seed=0,
                    device=device, n_in=n_in, radius=radius_best,
                    lamT=lamT_best,
                    input_scale=input_scale,
                    w_seed=ws, win_seed=ws, n_streams=n_streams,
                    train_samples_per_stream=train_samples_per_stream,
                    washout=washout, max_delay=max_delay,
                    gate="mismatch",
                    leg=f"mismatch_s{sigma:g}_d{dr}",
                    n_edges=int((W_p != 0).sum().item()),
                    note=f"multiplicative Gaussian sigma={sigma}, "
                         f"draw={dr}, ws={ws}",
                ))
    # Heterogeneous operating-point row at the best corner.
    gain_j, bias_j = heterogeneous_tanh_jitter(
        n_nodes=hidden_dim, gain_jitter=0.10, bias_jitter=0.05,
        seed=0,
    )
    rows.append(_heterogeneous_row(
        gain=gain_j, bias=bias_j, hidden_dim=hidden_dim,
        harness=harness, n_taps=n_taps, n_streams=n_streams,
        train_samples_per_stream=train_samples_per_stream,
        washout=washout, max_delay=max_delay,
        lamT_best=lamT_best, input_scale=input_scale,
        radius_best=radius_best, device=device,
    ))
    verdict = gate2_mismatch_verdict(rows)
    return rows, verdict


def _heterogeneous_row(
    *, gain: torch.Tensor, bias: torch.Tensor, hidden_dim: int,
    harness: str, n_taps: int, n_streams: int,
    train_samples_per_stream: int, washout: int, max_delay: int,
    lamT_best: float, input_scale: float,
    radius_best: float = 1.0, device: str = "cpu",
) -> GateRow:
    """Run the discrete twin with per-node gain/bias jitter on tanh."""
    dev = torch.device(device)
    n_in = 8 if harness == "taps8" else 1
    W, W_in = flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=n_in, w_seed=0, win_seed=0,
        target_radius=radius_best, dtype=torch.float32, device=dev,
    )
    gain_d = gain.to(device=dev, dtype=W.dtype)
    bias_d = bias.to(device=dev, dtype=W.dtype)
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=10, seed=0, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream = u_raw[0]
    y_stream = y_raw[0]
    u_scaled = ne._scale_drive(
        u_stream, bipolar=True, order=10, input_scale=1.0,
    )
    drive = flow_twin_drive(
        u_scaled, harness, n_taps=n_taps, input_scale=input_scale,
    ).to(dev)
    y_dev = y_stream.to(dev)
    states = []
    x = torch.zeros(hidden_dim, dtype=W.dtype, device=dev)
    pre_vals = []
    for t in range(drive.shape[0]):
        pre = W @ x + W_in @ drive[t]
        engaged_pre = gain_d * (pre + bias_d)
        x = torch.tanh(engaged_pre)
        states.append(x.detach().clone())
        pre_vals.append(engaged_pre.abs())
    states = torch.stack(states, dim=0)
    all_pre = torch.stack(pre_vals)
    engaged = bool(
        float(all_pre.median().item()) >= G1_PRE_LINEAR_REGIME_MEDIAN
    )
    finite = bool(torch.isfinite(states).all().item())
    mc_per, mc_total = nlc_safe_per_delay_mc(
        states,
        drive[:, 0] if harness == "scalar" else u_scaled.to(dev),
        washout=washout, max_delay=max_delay,
    )
    ridge_nrmse, _ = _ridge_nrmse_r2(states, y_dev, washout=washout)
    state_pr = _safe_participation_ratio_local(states[washout:])
    return GateRow(
        config_tag=(
            f"g2_hetops_seed0_h{hidden_dim}_n{n_in}"
            f"_r{radius_best:g}_lamT{lamT_best:g}"
        ),
        gate="mismatch", leg="hetops_g0.10_b0.05",
        n_edges=int(W.numel()), n_in=n_in,
        mc_total=float(mc_total), ridge_nrmse=float(ridge_nrmse),
        state_pr=float(state_pr),
        tanh_engaged=engaged, states_finite=finite,
        pass_mc=_gate_row_pass_mc(float(mc_total), engaged, finite),
        note="per-node tanh gain/bias jitter (0.10 / 0.05)",
    )


_MISMATCH_LEG_RE = re.compile(r"^mismatch_s([^_]+)_d\d+$")


def gate2_mismatch_verdict(rows: list[GateRow]) -> dict[str, Any]:
    """Apply pre-registered Gate-2 mismatch fatal criteria.

    Verdict: holds at 5-10% mismatch -> SILICON_PLAUSIBLE; collapses
    at 10% -> PRECISION_REDIRECT.  "Holds" means bucket-mean MC >=
    G2_G3_HOLDS_MC_ABOVE (twin PASS_ARCH bar).
    """
    floor = G2_G3_HOLDS_MC_ABOVE
    by_sigma: dict[float, list[float]] = {}
    for r in rows:
        m = _MISMATCH_LEG_RE.match(r.leg)
        if m is None:
            continue
        try:
            sigma = float(m.group(1))
        except ValueError:
            continue
        by_sigma.setdefault(sigma, []).append(r.mc_total)
    if not by_sigma:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "silicon_pass_lo": G2_MISMATCH_PASS_LO,
            "precision_fail_hi": G2_MISMATCH_FAIL_HI,
            "holds_mc_above": floor,
            "mean_mc_by_sigma": {},
        }
    means = {s: (sum(v) / len(v)) for s, v in by_sigma.items()
             if v}
    lo_mc = means.get(G2_MISMATCH_PASS_LO, float("nan"))
    hi_mc = means.get(G2_MISMATCH_FAIL_HI, float("nan"))
    if not math.isfinite(lo_mc) or not math.isfinite(hi_mc):
        verdict = "INSUFFICIENT_DATA"
    elif hi_mc < floor:
        verdict = "PRECISION_REDIRECT"
    elif lo_mc >= floor and hi_mc >= floor:
        verdict = "SILICON_PLAUSIBLE"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "silicon_pass_lo": G2_MISMATCH_PASS_LO,
        "precision_fail_hi": G2_MISMATCH_FAIL_HI,
        "holds_mc_above": floor,
        "mean_mc_by_sigma": {str(s): float(mc)
                             for s, mc in means.items()},
        "lo_mc": float(lo_mc) if math.isfinite(lo_mc) else None,
        "hi_mc": float(hi_mc) if math.isfinite(hi_mc) else None,
    }


def run_gate3_w_in_sweep(
    *, hidden_dim: int, harness: str, n_taps: int, n_streams: int,
    train_samples_per_stream: int, washout: int, max_delay: int,
    lamT_best: float, input_scale: float,
    radius_best: float = 1.0,
    device: str = "cpu",
    k_per_tap_grid: tuple[int, ...] = G3_K_PER_TAP,
    geom: tuple[int, ...] = G3_TORUS_GEOM,
    mask_seeds: tuple[int, ...] = G3_MASK_SEEDS,
    dense_seeds: tuple[int, ...] = G3_DENSE_SEEDS,
) -> tuple[list[GateRow], dict[str, Any]]:
    """Run the Gate-3 sparse W_in sweep at the best corner.

    Leg budget (~20): dense banked reference x ``dense_seeds`` +
    sparse-random {3, 6} x ``mask_seeds`` + sparse-structured x
    ``mask_seeds``, all with the dense W_mix distributing.
    """
    rows: list[GateRow] = []
    dev = torch.device(device)
    n_in = 8 if harness == "taps8" else 1
    W, _W_in_dense = flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=n_in, w_seed=0, win_seed=0,
        target_radius=radius_best, dtype=torch.float32, device=dev,
    )
    # 1) Dense W_in banked reference x dense seeds.
    for ds_ in dense_seeds:
        W_in_dense = dense_W_in(
            n_nodes=hidden_dim, n_in=n_in, seed=ds_,
            dtype=torch.float32, device=dev,
        )
        rows.append(_sparsity_row_from_twin(
            W=W, W_in=W_in_dense, hidden_dim=hidden_dim,
            harness=harness, n_taps=n_taps, seed=0,
            device=device, n_in=n_in, radius=radius_best,
            lamT=lamT_best,
            input_scale=input_scale,
            w_seed=0, win_seed=ds_, n_streams=n_streams,
            train_samples_per_stream=train_samples_per_stream,
            washout=washout, max_delay=max_delay,
            gate="w_in",
            leg=f"w_in_dense_ms{ds_}",
            n_edges=int(W_in_dense.numel()),
            note="dense Gaussian W_in reference",
        ))
    # 2) Sparse-random W_in x k in {3, 6} x mask seeds.
    for k in k_per_tap_grid:
        for ms in mask_seeds:
            W_in_sr = sparse_random_W_in(
                n_nodes=hidden_dim, n_in=n_in, k_per_tap=k,
                seed=ms, dtype=torch.float32, device=dev,
            )
            rows.append(_sparsity_row_from_twin(
                W=W, W_in=W_in_sr, hidden_dim=hidden_dim,
                harness=harness, n_taps=n_taps, seed=0,
                device=device, n_in=n_in, radius=radius_best,
                lamT=lamT_best,
                input_scale=input_scale,
                w_seed=0, win_seed=0, n_streams=n_streams,
                train_samples_per_stream=train_samples_per_stream,
                washout=washout, max_delay=max_delay,
                gate="w_in",
                leg=f"w_in_sparse_random_k{k}_ms{ms}",
                n_edges=int((W_in_sr != 0).sum().item()),
                note=f"sparse-random W_in, k={k}, mask_seed={ms}",
            ))
    # 3) Sparse-structured W_in x mask seeds.
    for ms in mask_seeds:
        W_in_ss = sparse_structured_W_in(
            n_nodes=hidden_dim, n_in=n_in, geom=geom, seed=ms,
            dtype=torch.float32, device=dev,
        )
        rows.append(_sparsity_row_from_twin(
            W=W, W_in=W_in_ss, hidden_dim=hidden_dim,
            harness=harness, n_taps=n_taps, seed=0,
            device=device, n_in=n_in, radius=radius_best,
            lamT=lamT_best,
            input_scale=input_scale,
            w_seed=0, win_seed=0, n_streams=n_streams,
            train_samples_per_stream=train_samples_per_stream,
            washout=washout, max_delay=max_delay,
            gate="w_in",
            leg=f"w_in_sparse_structured_ms{ms}",
            n_edges=int((W_in_ss != 0).sum().item()),
            note=f"sparse-structured W_in on torus geom={geom}, "
                 f"mask_seed={ms}",
        ))
    verdict = gate3_verdict(rows)
    return rows, verdict


def gate3_verdict(rows: list[GateRow]) -> dict[str, Any]:
    """Apply pre-registered Gate-3 fatal criteria.

    Verdict: structured holds -> STRUCTURED (boundary spec is localized
    tap groups + dense mixer); only-dense holds -> DENSE_ONLY (input
    routing joins the device-budget problem).  "Holds" means leg-mean
    MC >= G2_G3_HOLDS_MC_ABOVE (twin PASS_ARCH bar).
    """
    floor = G2_G3_HOLDS_MC_ABOVE
    by_leg: dict[str, list[float]] = {}
    for r in rows:
        leg_key = r.leg.split("_ms")[0]
        by_leg.setdefault(leg_key, []).append(r.mc_total)
    means = {k: (sum(v) / len(v)) for k, v in by_leg.items() if v}
    structured_mc = means.get("w_in_sparse_structured", float("nan"))
    sparse_random_max = max(
        (mc for k, mc in means.items()
         if k.startswith("w_in_sparse_random") and math.isfinite(mc)),
        default=float("nan"),
    )
    dense_mc = means.get("w_in_dense", float("nan"))
    structured_holds = math.isfinite(structured_mc) and structured_mc >= floor
    random_holds = (math.isfinite(sparse_random_max)
                    and sparse_random_max >= floor)
    dense_holds = math.isfinite(dense_mc) and dense_mc >= floor
    if structured_holds and not dense_holds:
        verdict = "STRUCTURED"
    elif dense_holds and not structured_holds and not random_holds:
        verdict = "DENSE_ONLY"
    else:
        # Default: pick the more informative verdict; if structured
        # passes, prefer STRUCTURED.
        if structured_holds:
            verdict = "STRUCTURED"
        elif dense_holds:
            verdict = "DENSE_ONLY"
        else:
            verdict = "MIXED"
    return {
        "verdict": verdict,
        "holds_mc_above": floor,
        "mean_mc_by_leg": {k: float(mc) for k, mc in means.items()},
        "structured_mc": float(structured_mc) if math.isfinite(structured_mc) else None,
        "sparse_random_max_mc": (float(sparse_random_max)
                                 if math.isfinite(sparse_random_max) else None),
        "dense_mc": float(dense_mc) if math.isfinite(dense_mc) else None,
    }


# ---------------------------------------------------------------------------
# Buildable-spec emission (on all-pass).
# ---------------------------------------------------------------------------


def maybe_emit_buildable_spec(
    *, gate1: dict[str, Any], gate2_s: dict[str, Any],
    gate2_m: dict[str, Any], gate3: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any] | None:
    """If all three gates pass, emit a buildable spec; else return None."""
    if (gate1.get("verdict") != "FLOW_VIABLE"
            or gate2_s.get("verdict") != "ANNEAL_PRUNE"
            or gate2_m.get("verdict") != "SILICON_PLAUSIBLE"
            or gate3.get("verdict") != "STRUCTURED"):
        return None
    geom_tag = "x".join(str(int(d)) for d in G3_TORUS_GEOM)
    spec = {
        "schema_version": 1,
        "kind": "post_twin_gates_buildable_spec",
        "lambda_T": float(gate1["viable_lamT"]),
        "edge_target": int(gate2_s["edge_target"]),
        "mismatch_tolerance": float(G2_MISMATCH_PASS_LO),
        "w_in_geometry": f"sparse_structured_torus_{geom_tag}",
        "derived_from": {
            "gate1": gate1,
            "gate2_sparsity": gate2_s,
            "gate2_mismatch": gate2_m,
            "gate3": gate3,
        },
        "downstream_experiment": "frozen_multistage_baseline",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "post_twin_gates_buildable_spec.json").write_text(
        json.dumps(spec, indent=2)
    )
    return spec


# ---------------------------------------------------------------------------
# CSV / TXT / JSON writers.
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def _gate_rows_to_csv(path: Path, rows: list[GateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonable(v) for k, v in asdict(r).items()})


def _gate1_rows_to_csv(path: Path, rows: list[Gate1Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonable(v) for k, v in asdict(r).items()})


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("./output/post_twin_gates"))
    p.add_argument("--hidden-dim", type=int, default=CANONICAL_HIDDEN)
    p.add_argument("--n-streams", type=int, default=1)
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--washout", type=int, default=PROBE_WASHOUT)
    p.add_argument("--max-delay", type=int, default=20)
    p.add_argument("--jacobian-samples", type=int, default=3)


def _resolve_best_corner(g1_rows: list[Gate1Row]) -> dict[str, float]:
    """Find the (lamT, radius, input_scale) with the highest MC across
    taps8 sweep rows, restricted to FLOW_VIABLE-eligible lambda*T."""
    pool = [r for r in g1_rows
            if r.tanh_engaged and r.states_finite
            and r.lamT <= G1_SPEED_REQUIRED_LAMBDA_T - 1e-6]
    if not pool:
        pool = [r for r in g1_rows
                if r.tanh_engaged and r.states_finite]
    if not pool:
        return {"lamT": 1.0, "radius": 0.9, "input_scale": 0.2,
                "mc_total": 0.0}
    best = max(pool, key=lambda r: r.mc_total)
    return {
        "lamT": float(best.lamT),
        "radius": float(best.radius),
        "input_scale": float(best.input_scale),
        "mc_total": float(best.mc_total),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-twin fatal-risk gates (plan post-twin-gates, "
                    "feature spec flow-sparse-gates).",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_heun = sub.add_parser(
        "heun", help="Gate 1: Heun flow-twin sweep (taps8 main or scalar confirm).",
    )
    _add_common(p_heun)
    p_heun.add_argument(
        "--harness", choices=G1_HARNESSES, default="taps8",
    )
    p_heun.add_argument("--n-taps", type=int, default=8)
    p_heun.add_argument("--max-corners", type=int, default=None,
                        help="Cap the Cartesian grid BEFORE running "
                             "(traversal prefix; smoke runs stay cheap "
                             "and the verdict covers exactly the rows "
                             "written).")
    p_heun.add_argument(
        "--dt", type=float, default=CANONICAL_DT_MAX,
        help=f"Heun substep in samples (must satisfy 0 < dt <= "
             f"{CANONICAL_DT_MAX}; fixed-dt rule).",
    )

    p_sparsity = sub.add_parser(
        "sparsity", help="Gate 2: sparsity sweep (random-sparse + "
                          "magnitude-prune + low-rank+sparse).",
    )
    _add_common(p_sparsity)
    p_sparsity.add_argument("--harness", choices=G1_HARNESSES, default="taps8")
    p_sparsity.add_argument("--n-taps", type=int, default=8)
    p_sparsity.add_argument("--lamT", type=float, default=1.0)
    p_sparsity.add_argument("--input-scale", type=float, default=0.2)

    p_mismatch = sub.add_parser(
        "mismatch", help="Gate 2: mismatch sweep (multiplicative Gaussian) "
                          "+ heterogeneous operating-point row.",
    )
    _add_common(p_mismatch)
    p_mismatch.add_argument("--harness", choices=G1_HARNESSES, default="taps8")
    p_mismatch.add_argument("--n-taps", type=int, default=8)
    p_mismatch.add_argument("--lamT", type=float, default=1.0)
    p_mismatch.add_argument("--input-scale", type=float, default=0.2)

    p_w_in = sub.add_parser(
        "w_in", help="Gate 3: sparse structured W_in sweep.",
    )
    _add_common(p_w_in)
    p_w_in.add_argument("--harness", choices=G1_HARNESSES, default="taps8")
    p_w_in.add_argument("--n-taps", type=int, default=8)
    p_w_in.add_argument("--lamT", type=float, default=1.0)
    p_w_in.add_argument("--input-scale", type=float, default=0.2)

    p_all = sub.add_parser(
        "all", help="Run all three gates end-to-end and emit the "
                     "buildable spec on all-pass.",
    )
    _add_common(p_all)
    p_all.add_argument("--harness", choices=G1_HARNESSES, default="taps8")
    p_all.add_argument("--n-taps", type=int, default=8)
    p_all.add_argument("--max-corners-g1", type=int, default=None,
                       help="Cap the Gate-1 Cartesian grid BEFORE running "
                            "(traversal prefix; smoke).")

    args = parser.parse_args(argv)
    if args.order != 10:
        parser.error(
            "post-twin-gate decisions are pre-registered for --order 10 only"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "heun":
        if not (0.0 < float(args.dt) <= CANONICAL_DT_MAX):
            parser.error(
                f"--dt must satisfy 0 < dt <= {CANONICAL_DT_MAX} "
                f"(fixed-dt rule), got {args.dt}"
            )
        rows, verdict, elapsed = run_gate1_sweep(
            harness=args.harness, n_taps=args.n_taps,
            hidden_dim=args.hidden_dim, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            device=args.device, dt=float(args.dt),
            max_corners=args.max_corners,
        )
        _gate1_rows_to_csv(args.output / "gate1_flowtwin.csv", rows)
        (args.output / "gate1_flowtwin.json").write_text(
            json.dumps(_jsonable(verdict), indent=2)
        )
        hdr = (
            f"Gate 1 Heun flow-twin -- harness={args.harness} "
            f"n_taps={args.n_taps} hidden_dim={args.hidden_dim} "
            f"washout={args.washout} n_corners={len(rows)} "
            f"elapsed={elapsed:.1f}s verdict={verdict['verdict']}"
        )
        lines = [
            f"  verdict={verdict['verdict']}  viable_lamT={verdict['viable_lamT']}",
            f"  pass_mc_above={verdict['pass_mc_above']}  "
            f"speed_required_lamT={verdict['speed_required_lamT']}",
            f"  surviving_by_lamT={verdict['surviving_by_lamT']}",
        ]
        (args.output / "gate1_flowtwin.txt").write_text(
            hdr + "\n" + "\n".join(lines) + "\n"
        )
        print(hdr)
        for ln in lines:
            print(ln)
        return 0

    if args.mode == "sparsity":
        rows, verdict = run_gate2_sparsity_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=args.lamT, input_scale=args.input_scale,
            device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate2_sparsity.csv", rows)
        (args.output / "gate2_sparsity.json").write_text(
            json.dumps(_jsonable(verdict), indent=2)
        )
        hdr = (
            f"Gate 2 sparsity -- harness={args.harness} n_taps={args.n_taps} "
            f"hidden_dim={args.hidden_dim} n_corners={len(rows)} "
            f"verdict={verdict['verdict']} edge_target={verdict['edge_target']}"
        )
        print(hdr)
        return 0

    if args.mode == "mismatch":
        rows, verdict = run_gate2_mismatch_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=args.lamT, input_scale=args.input_scale,
            device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate2_mismatch.csv", rows)
        (args.output / "gate2_mismatch.json").write_text(
            json.dumps(_jsonable(verdict), indent=2)
        )
        hdr = (
            f"Gate 2 mismatch -- harness={args.harness} n_taps={args.n_taps} "
            f"hidden_dim={args.hidden_dim} n_corners={len(rows)} "
            f"verdict={verdict['verdict']}"
        )
        print(hdr)
        return 0

    if args.mode == "w_in":
        rows, verdict = run_gate3_w_in_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=args.lamT, input_scale=args.input_scale,
            device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate3_w_in.csv", rows)
        (args.output / "gate3_w_in.json").write_text(
            json.dumps(_jsonable(verdict), indent=2)
        )
        hdr = (
            f"Gate 3 sparse W_in -- harness={args.harness} n_taps={args.n_taps} "
            f"hidden_dim={args.hidden_dim} n_corners={len(rows)} "
            f"verdict={verdict['verdict']}"
        )
        print(hdr)
        return 0

    if args.mode == "all":
        g1_rows, g1_verdict, g1_elapsed = run_gate1_sweep(
            harness=args.harness, n_taps=args.n_taps,
            hidden_dim=args.hidden_dim, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            device=args.device,
            max_corners=args.max_corners_g1,
        )
        _gate1_rows_to_csv(args.output / "gate1_flowtwin.csv", g1_rows)
        (args.output / "gate1_flowtwin.json").write_text(
            json.dumps(_jsonable(g1_verdict), indent=2)
        )
        # Scalar confirm leg (locked grid per spec, record-only: it
        # carries no fatal power; the taps8 main sweep owns the Gate-1
        # verdict).
        g1c_rows, g1c_verdict, _g1c_elapsed = run_gate1_sweep(
            harness="scalar", n_taps=args.n_taps,
            hidden_dim=args.hidden_dim, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            jacobian_samples=args.jacobian_samples,
            device=args.device,
            radius_grid=G1_CONFIRM_RADIUS,
            input_scale_grid=G1_CONFIRM_INPUT_SCALE,
            lamT_grid=G1_CONFIRM_LAMT,
            w_seeds=G1_W_SEEDS, data_seeds=G1_DATA_SEEDS,
        )
        _gate1_rows_to_csv(
            args.output / "gate1_scalar_confirm.csv", g1c_rows,
        )
        (args.output / "gate1_scalar_confirm.json").write_text(
            json.dumps(_jsonable(g1c_verdict), indent=2)
        )
        best = _resolve_best_corner(g1_rows)
        g2s_rows, g2s_verdict = run_gate2_sparsity_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=best["lamT"], input_scale=best["input_scale"],
            radius_best=best["radius"], device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate2_sparsity.csv", g2s_rows)
        (args.output / "gate2_sparsity.json").write_text(
            json.dumps(_jsonable(g2s_verdict), indent=2)
        )
        g2m_rows, g2m_verdict = run_gate2_mismatch_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=best["lamT"], input_scale=best["input_scale"],
            radius_best=best["radius"], device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate2_mismatch.csv", g2m_rows)
        (args.output / "gate2_mismatch.json").write_text(
            json.dumps(_jsonable(g2m_verdict), indent=2)
        )
        g3_rows, g3_verdict = run_gate3_w_in_sweep(
            hidden_dim=args.hidden_dim, harness=args.harness,
            n_taps=args.n_taps, n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            washout=args.washout, max_delay=args.max_delay,
            lamT_best=best["lamT"], input_scale=best["input_scale"],
            radius_best=best["radius"], device=args.device,
        )
        _gate_rows_to_csv(args.output / "gate3_w_in.csv", g3_rows)
        (args.output / "gate3_w_in.json").write_text(
            json.dumps(_jsonable(g3_verdict), indent=2)
        )
        spec = maybe_emit_buildable_spec(
            gate1=g1_verdict, gate2_s=g2s_verdict,
            gate2_m=g2m_verdict, gate3=g3_verdict,
            out_dir=args.output,
        )
        verdict_path = args.output / "post_twin_gates_verdict.json"
        verdict_path.write_text(json.dumps(_jsonable({
            "gate1": g1_verdict, "gate1_scalar_confirm": g1c_verdict,
            "gate2_sparsity": g2s_verdict,
            "gate2_mismatch": g2m_verdict, "gate3": g3_verdict,
            "best_corner": best,
            "buildable_spec_emitted": spec is not None,
        }), indent=2))
        print(
            f"Post-twin gates: gate1={g1_verdict['verdict']} "
            f"(scalar_confirm={g1c_verdict['verdict']}) "
            f"gate2_sparsity={g2s_verdict['verdict']} "
            f"gate2_mismatch={g2m_verdict['verdict']} "
            f"gate3={g3_verdict['verdict']} "
            f"buildable_spec={spec is not None}"
        )
        return 0

    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
