"""Advisor-ordered zero-training probes for the NARMA reservoir plateau.

Implements the four-pillar probe discipline described in the cross-roads
problem statement and ``docs/specs/advisor-probe-suite.md``.  The module
is *side-effect free at import*; only directly-invoked functions build
nets, mutate init, or read checkpoints.

Pillars (called in order, each can be invoked independently):

1. **Baselines calibration** -- delegates to ``narma_experiment.run_baselines``
    via ``run_baselines_calibration``.  The halt/continue decision compares the
    ESN NRMSE to the success band (0.15-0.29 NARMA-10).  Callers should check
    this calibration before interpreting fabric probes.
2. **Frozen-state probe** -- :func:`frozen_state_probe`.  Eval-only fits on
   the existing checkpoint's hidden states: ridge + small MLP on full
    hidden states, ridge on a 5-lag state stack, participation ratio of
    the washed covariance, gate-zero add-on (force ``z_logits`` to a very
    negative value at eval to check if the core edges are doing any work).
3. **Gain-override hook** -- :func:`apply_gain_override`.  Symmetric
   ``gm_raw``/``isat_raw`` overwrite on core + boundary + readout cell
   libraries plus a per-node randomized ``raw_leak`` init, OR a fixed
   ``leak_constant`` (non-programmable mode).  ``cell_library.py`` init
   defaults are NEVER edited; init discipline stays in one place.
4. **E0 sweep** -- :func:`e0_sweep`.  Zero-training 36-point gain x leak x
   drive grid.  Per point reports washout-corrected memory capacity,
   ridge-from-full-states NRMSE, Jacobian eigenvalue instrument, and rail
   fraction; tagged ``PASS``/``FAIL`` against the gates in
   ``advisor-probe-suite``.

Probe outputs land in the leg output dir next to (not inside) training
tables; each row carries config tag plus PASS/FAIL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Reuse the canonical NARMA experiment pieces (build, run, ridge readouts,
# MC, helpers).  No mutation of ``narma_experiment`` itself.
import narma_experiment as ne  # noqa: E402
from config import INIT as _CONFIG_INIT  # noqa: E402

# Locked grid (advisor-probe-suite, E0 sweep).  With gm in [0.01, 10], gains
# {-5,-2,0,1.5} give approximately {0.077, 1.20, 5.01, 8.18}.  Leak modes
# map to {"slow-fixed", "randomized", 0.15}.  Drive {0.25, 0.5, 1.0} uses
# the existing input_scale path.
E0_GAIN_GRID: tuple[float, ...] = (-5.0, -2.0, 0.0, 1.5)
E0_LEAK_GRID: tuple[Any, ...] = ("slow-fixed", "randomized", 0.15)
E0_DRIVE_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)

# NARMA-10 success band (literature-anchored, not in-harness calibrated
# until the baselines calibration leg runs).
NARMA10_BAND: tuple[float, float] = (0.15, 0.29)
# Pre-registered ESN halt point: at or above this value everything stops for
# a task-setup hunt.  The advisor wording is “far above band (~0.6)”.
ESN_HALT_ABOVE: float = 0.6
# Frozen-state decision thresholds (advisor-probe-suite spec).
RIDGE_PROMOTE_READOUT_BELOW: float = 0.45
RIDGE_INIT_REGIME_AT_OR_ABOVE: float = 0.6
# E0 corner decision thresholds (P2 rerun amendment).
E0_PASS_MC_ABOVE: float = 3.0
E0_PASS_RIDGE_BELOW: float = 0.40
E0_PASS_STATE_PR_MIN: float = 6.0
# Rail fraction disqualifier (spec).
RAIL_DISQUALIFY_ABOVE: float = 0.05
# Train-stream washout for frozen-state / E0 probes (matches the
# canonical --ridge-diagnostic washout=200).
PROBE_WASHOUT: int = 200


@dataclass
class FrozenStateRow:
    """One frozen-state probe row."""

    config_tag: str
    nrmse_ridge_full: float
    nrmse_ridge_lag5: float
    nrmse_mlp_full: float
    r2_ridge_full: float
    participation_ratio: float
    nrmse_gate_zero_ridge_full: float
    mc_total: float
    n_params_ridge: int
    n_params_mlp: int
    # PASS/FAIL on the ridge-from-states decision rule.
    readout_fix_promoted: bool  # ridge_full < RIDGE_PROMOTE_READOUT_BELOW
    init_regime_sole_primary: bool  # ridge_full >= RIDGE_INIT_REGIME_AT_OR_ABOVE


@dataclass
class E0SweepRow:
    """One E0 sweep corner, including its boundary-only pass."""

    config_tag: str
    hidden_dim: int
    n_params: int
    gm_init: float
    isat_init: float  # NaN sentinel: same as gm_init (legacy coupled)
    leak_mode: str  # "slow-fixed" | "randomized" | "constant:<val>"
    drive: float
    # Regime-transplant: per-library override dict encoded as JSON, or
    # empty string when the coupled single-fill path was used (default).
    per_lib_overrides_json: str
    # Round-2 §13.1 sparse-drive bookkeeping. Empty string (default) is
    # the canonical full fan-out ``{0: range(hidden_dim)}``; a non-empty
    # value appears when the CLI was passed ``--boundary-fan-out <json>``
    # (JSON-encoded, sorted keys). Kept as a string so CSV dumps stay
    # single-row.
    boundary_fan_out: str
    ridge_nrmse: float
    ridge_r2: float
    mc_total_washout_corrected: float
    state_pr: float  # participation ratio of washed full states
    jac_max_abs: float
    jac_min_abs: float
    jac_mean_abs: float
    jac_rank_proxy: float  # participation ratio of |eig|
    rail_frac: float
    sat_max_ratio: float  # max|x| / x_max
    gate_zero_ridge_nrmse: float
    gate_zero_ridge_r2: float
    gate_zero_mc_total: float
    gate_zero_state_pr: float
    gate_zero_jac_max_abs: float
    gate_zero_jac_min_abs: float
    gate_zero_jac_mean_abs: float
    gate_zero_jac_rank_proxy: float
    gate_zero_rail_frac: float
    gate_zero_sat_max_ratio: float
    rail_disqualified: bool
    pass_init_regime: bool
    pass_rail: bool  # NOT rail_disqualified
    pass_all: bool  # pass_init_regime AND pass_rail
    gate_zero_rail_disqualified: bool
    gate_zero_pass_init_regime: bool
    gate_zero_pass_rail: bool
    gate_zero_pass_all: bool


def participation_ratio(states: torch.Tensor) -> float:
    """Participation ratio of the (washed) state covariance eigenvalues.

    PR = (sum lambda)^2 / sum(lambda^2).  For a uniform spectrum of rank
    r, PR ~ r.  For a rank-1 spectrum PR ~ 1.  Used as a rank-collapse
    diagnostic in :func:`frozen_state_probe`.

    Also called with shape ``(1, D)`` for the Jacobian eigenvalue vector;
    in that case the input is treated as the spectrum directly (no
    centering / covariance).  Returns NaN if the spectrum is degenerate
    or non-finite.
    """
    if states.dim() == 1:
        spec = states.clamp_min(0.0)
        s = float(spec.sum().item())
        sq = float(spec.pow(2).sum().item())
        if sq <= 1e-12:
            return float("nan")
        return (s * s) / sq
    if states.dim() != 2 or states.shape[0] < 2:
        return float("nan")
    states_c = states - states.mean(dim=0, keepdim=True)
    cov = (states_c.T @ states_c) / max(states.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    s = float(eig.sum().item())
    sq = float(eig.pow(2).sum().item())
    if sq <= 1e-12:
        return float("nan")
    return (s * s) / sq


def _ridge_fit_predict(
    X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2,
) -> torch.Tensor:
    """Closed-form ridge fit ``W = (X^T X + l2 I)^-1 X^T y`` with bias."""
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    XtX = X_aug.T @ X_aug + l2 * torch.eye(X_aug.shape[1], device=X.device)
    W = torch.linalg.solve(XtX, X_aug.T @ y)
    return W


def _stack_lag_features(states: torch.Tensor, n_lags: int = 5) -> torch.Tensor:
    """Build a ``(T - n_lags + 1, N * n_lags)`` lag-stacked feature matrix.

    Row ``i`` contains the newest state first:
    ``[x[i+n_lags-1], x[i+n_lags-2], ..., x[i]]`` flattened.  Thus row
    ``i`` must be compared with target ``y[i+n_lags-1]``.
    """
    if states.dim() != 2:
        raise ValueError(f"lag stacking requires a (T, N) matrix, got {tuple(states.shape)}")
    if n_lags < 1:
        raise ValueError(f"n_lags must be positive, got {n_lags}")
    T, N = states.shape
    if T < n_lags:
        raise ValueError(f"need at least {n_lags} states for lag stacking, got {T}")
    rows = []
    for lag in range(n_lags):
        rows.append(states[n_lags - 1 - lag: T - lag])
    return torch.cat(rows, dim=1)


def _huber_loss(pred: torch.Tensor, target: torch.Tensor,
                delta: float = 1.0) -> torch.Tensor:
    return nn.functional.huber_loss(pred, target, delta=delta)


def _train_small_mlp(
    X: torch.Tensor, y: torch.Tensor, *,
    hidden_dim: int = 32, epochs: int = 300, lr: float = 1e-3,
    seed: int = 0, device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a tiny 1-hidden-layer MLP as a probe readout.

    Two-pass protocol: 80% train, 20% val with early-stop on val NRMSE.
    Returns the trained model and a metrics dict.  Robust to ``epochs=0``
    and tiny validation sets (falls back to train-only when val < 2).
    """
    torch.manual_seed(seed)
    N = X.shape[1]
    model = nn.Sequential(
        nn.Linear(N, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 80/20 split with a fixed permutation so the run is reproducible.
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(0.8 * n)
    X_tr, y_tr = X[perm[:cut]].to(device), y[perm[:cut]].to(device)
    X_va, y_va = X[perm[cut:]].to(device), y[perm[cut:]].to(device)
    best_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    patience = max(20, epochs // 10)
    stagnant = 0
    epochs_run = 0
    for ep in range(epochs):
        epochs_run = ep + 1
        model.train()
        # Full-batch (probe, not training): small data, fast.
        optim.zero_grad()
        pred = model(X_tr).squeeze(-1)
        loss = _huber_loss(pred, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        model.eval()
        if X_va.shape[0] >= 2:
            with torch.no_grad():
                v_pred = model(X_va).squeeze(-1)
            val_n = ne.nrmse(v_pred, y_va)
        else:
            # Tiny-val fallback: use train NRMSE as a proxy so the probe
            # still returns a number instead of crashing on an empty split.
            with torch.no_grad():
                v_pred = model(X_tr).squeeze(-1)
            val_n = ne.nrmse(v_pred, y_tr)
        if best_val is None or val_n + 1e-6 < best_val:
            best_val = val_n
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_nrmse": float(best_val if best_val is not None
                                       else float("nan")), "epochs_run": epochs_run}


def _gate_zero_eval_forward(net: nn.Module, u_seq: torch.Tensor) -> torch.Tensor:
    """Run the eval path with core-edge gates forced to a very negative value.

    Returns ``all_states`` of shape ``(T, B, N)``.  Original gates are
    restored on exit, including on multi-stage nets.
    """
    if not hasattr(net, "core"):
        raise ValueError("net has no .core attribute; expected KirchhoffNetWithIO")
    if not net.core.stages:
        raise ValueError("gate-zero probe requires at least one stage")
    saved: list[tuple[nn.Module, torch.Tensor]] = []
    try:
        for stage in net.core.stages:
            if hasattr(stage, "z_logits"):
                saved.append((stage, stage.z_logits.detach().clone()))
                stage.z_logits.data.fill_(-12.0)
        # Build inputs identical to ``_evaluate_fabric_direct``:
        stage = net.core.stages[0]
        t_span = net.core.stage_times[0]
        num_steps = net.core.stage_steps[0]
        stage_width = net.hid_count + net.proj_count + net.output_ode_count
        x0 = u_seq.new_zeros(1, stage_width)
        with torch.no_grad():
            all_states = stage._forward_heun_sequence(
                x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_seq,
            )
        return all_states
    finally:
        for stage, value in saved:
            stage.z_logits.data.copy_(value)


def memory_capacity_washout(
    states: torch.Tensor, targets: torch.Tensor, *,
    washout: int = PROBE_WASHOUT, max_delay: int = 20, ridge_l2: float = 1e-2,
) -> tuple[list[float], float]:
    """Washout-corrected memory capacity (advisor-probe-suite spec).

    Discards the first ``washout`` rows of ``states`` before fitting the
    per-delay ridge regressions; the legacy ``memory_capacity`` keeps
    them (no-washout caveat for slow leaks).
    """
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        return [float("nan")] * max_delay, float("nan")
    if washout >= states.shape[0]:
        return [float("nan")] * max_delay, float("nan")
    s_w = states[washout:]
    t_w = targets[washout:]
    return ne.memory_capacity(s_w, t_w, max_delay=max_delay, ridge_l2=ridge_l2)


def frozen_state_probe(
    net: nn.Module, u_train: torch.Tensor, y_train: torch.Tensor, *,
    washout: int = PROBE_WASHOUT, mlp_hidden: int = 32, mlp_epochs: int = 300,
    mlp_seed: int = 0, device: str = "cpu",
    also_eval_gate_zero: bool = True, config_tag: str = "",
) -> FrozenStateRow:
    """Probe a trained (or untrained) net for information presence.

    Fits (a) ridge + small MLP on full hidden states, (b) ridge on a
    5-lag state stack, (c) participation ratio of the washed covariance,
    (d) optionally ridge with core-edge gates forced to zero at eval.

    All metrics are eval-only; the net weights are not touched (the
    gate-zero add-on saves/restores ``z_logits``).
    """
    stages = list(net.core.stages)
    if len(stages) != 1:
        raise NotImplementedError(
            "frozen_state_probe currently supports single-stage nets only"
        )
    stage = stages[0]
    hid_count = int(net.hid_count)
    proj_count = int(net.proj_count)
    state_width = hid_count + proj_count + int(net.output_ode_count)
    x0 = u_train.new_zeros(1, state_width)
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])

    # Full states on the training stream (single stream, batch B=1).
    # Wrap in ``torch.no_grad`` so the Heun graph does not attach to
    # ``hidden`` and the small-MLP backward pass can run independently.
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_train,
        )
    states_full = all_states[:, 0, :].detach()  # (T, N)
    hidden = states_full[:, :hid_count].detach()  # (T, hid_count)
    if u_train.dim() != 1 or y_train.dim() != 1:
        raise ValueError(
            "frozen_state_probe requires one-dimensional input/target streams, "
            f"got {tuple(u_train.shape)} and {tuple(y_train.shape)}"
        )
    if hidden.shape[0] != y_train.shape[0]:
        raise ValueError(
            "input and target streams must have the same length, got "
            f"{hidden.shape[0]} and {y_train.shape[0]}"
        )
    if hidden.shape[0] <= washout:
        raise ValueError(
            f"need more than {washout} washed-out samples, got {hidden.shape[0]}"
        )
    y = y_train.to(device)
    hidden_w = hidden[washout:]
    y_w = y[washout:]

    # (a) Ridge from full hidden states.
    W = _ridge_fit_predict(hidden_w, y_w, l2=1e-2)
    y_pred = torch.cat([hidden_w, torch.ones(hidden_w.shape[0], 1, device=hidden_w.device)], dim=1) @ W
    nrmse_ridge = ne.nrmse(y_pred, y_w)
    r2_ridge = ne.r2(y_pred, y_w)
    n_params_ridge = hidden_w.shape[1] + 1

    # (b) MLP from full hidden states (small, ~32 hidden units).
    mlp, mlp_metrics = _train_small_mlp(
        hidden_w, y_w, hidden_dim=mlp_hidden, epochs=mlp_epochs,
        lr=1e-3, seed=mlp_seed, device=device,
    )
    with torch.no_grad():
        mlp_pred = mlp(hidden_w).squeeze(-1)
    nrmse_mlp = ne.nrmse(mlp_pred, y_w)
    n_params_mlp = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

    # (c) Ridge from 5-lag state stack.  Stacked row ``i`` ends at state
    # ``i+n_lags-1``, so the first washed row (whose newest state is
    # ``hidden[washout]``) is at stacked index ``washout-(n_lags-1)`` and
    # must be compared with target ``y[washout]``.
    n_lags = 5
    if washout < n_lags - 1:
        raise ValueError(
            f"washout must be at least {n_lags - 1} for a {n_lags}-lag stack, "
            f"got {washout}"
        )
    stacked = _stack_lag_features(hidden, n_lags=n_lags)
    stacked_w = stacked[washout - (n_lags - 1):]
    y_aligned = y[washout: washout + stacked_w.shape[0]]
    if stacked_w.shape[0] != y_aligned.shape[0]:
        raise ValueError(
            "lagged states and aligned targets have different lengths: "
            f"{stacked_w.shape[0]} != {y_aligned.shape[0]}"
        )
    W_lag = _ridge_fit_predict(stacked_w, y_aligned, l2=1e-2)
    pred_lag = torch.cat(
        [stacked_w, torch.ones(stacked_w.shape[0], 1, device=stacked_w.device)], dim=1,
    ) @ W_lag
    nrmse_ridge_lag = ne.nrmse(pred_lag, y_aligned)

    # (d) Participation ratio on washed states (full state, all dims).
    pr = participation_ratio(states_full[washout:])

    # (e) Memory capacity on the full states (washout-corrected).
    _, mc_total = memory_capacity_washout(
        states_full, u_train, washout=washout, max_delay=20, ridge_l2=1e-2,
    )

    # (f) Gate-zero add-on: ridge from hidden states with core-edge gates off.
    nrmse_gate_zero = float("nan")
    if also_eval_gate_zero and hasattr(stage, "z_logits"):
        gate_states = _gate_zero_eval_forward(net, u_train)
        hidden_gz = gate_states[:, 0, :hid_count]
        hidden_gz_w = hidden_gz[washout:]
        W_gz = _ridge_fit_predict(hidden_gz_w, y_w, l2=1e-2)
        pred_gz = torch.cat(
            [hidden_gz_w, torch.ones(hidden_gz_w.shape[0], 1, device=hidden_gz_w.device)],
            dim=1,
        ) @ W_gz
        nrmse_gate_zero = ne.nrmse(pred_gz, y_w)

    return FrozenStateRow(
        config_tag=config_tag,
        nrmse_ridge_full=nrmse_ridge,
        nrmse_ridge_lag5=nrmse_ridge_lag,
        nrmse_mlp_full=nrmse_mlp,
        r2_ridge_full=r2_ridge,
        participation_ratio=pr,
        nrmse_gate_zero_ridge_full=nrmse_gate_zero,
        mc_total=mc_total,
        n_params_ridge=n_params_ridge,
        n_params_mlp=n_params_mlp,
        readout_fix_promoted=nrmse_ridge < RIDGE_PROMOTE_READOUT_BELOW,
        init_regime_sole_primary=nrmse_ridge >= RIDGE_INIT_REGIME_AT_OR_ABOVE,
    )


def apply_gain_override(
    net: nn.Module, *,
    gm_init: float, isat_init: float | None = None,
    leak_mode: str = "slow-fixed", raw_leak_init_seed: int = 0,
    raw_leak_init_mean: float = -3.0, raw_leak_init_std: float = 1.0,
    per_lib_overrides: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Override the gain / leak init of a built fabric net (post-build).

    Args:
        net: A built ``KirchhoffNetWithIO`` (output of
            ``_build_fabric_net`` or ``build_net_from_config``).
        gm_init: Value to fill ``gm_raw`` with on every cell library
            (default scalar fill, used when ``per_lib_overrides`` is
            ``None`` or doesn't cover a library).
        isat_init: Value to fill ``isat_raw`` with. ``None`` (default)
            mirrors ``gm_init`` (symmetric override).
        leak_mode: One of ``"slow-fixed"`` (non-programmable leak
            ``0.0486``), ``"randomized"`` (programmable leak filled
            with mean=``raw_leak_init_mean``, std=``raw_leak_init_std``
            using ``raw_leak_init_seed``), ``"raw:<val>"``
            (regime-transplant: programmable leak with ``raw_leak``
            filled to scalar ``val``), or a numeric value used as a
            fixed scalar leak.
        raw_leak_init_seed / raw_leak_init_mean / raw_leak_init_std:
            Controls the randomized ``raw_leak`` init.
        per_lib_overrides: Optional regime-transplant hook. Dict keyed by
            ``"cell_lib"`` / ``"boundary_cell_lib"`` / ``"output_ode_cell_lib"``;
            values are ``{"gm": float, "isat": float | None}``. A library
            listed here gets its own fill values instead of the scalar
            ``gm_init`` / ``isat_init``; libraries missing from the dict
            (or ``None`` entries) fall back to the scalars. Default
            (``None``) keeps the legacy coupled-fill path byte-identical.

    Returns:
        A small dict with the overrides applied (for the probe row log).
        The ``cell_library.py`` defaults are NEVER modified.
    """
    if isat_init is None:
        isat_init = gm_init

    if not hasattr(net, "core"):
        raise ValueError("apply_gain_override requires a KirchhoffNetWithIO net")

    log: dict[str, Any] = {
        "gm_init": float(gm_init),
        "isat_init": float(isat_init),
        "leak_mode": leak_mode,
    }

    # Per-library override support (regime-transplant plan): a dict
    # ``{lib_name: {"gm": float, "isat": float | None}}`` overrides that
    # one library's fill values; missing libraries fall back to the
    # scalar gm_init / isat_init. ``None`` keeps the old scalar behaviour
    # byte-identical.
    per_lib: dict[str, dict[str, float]] | None = None
    if isinstance(per_lib_overrides, dict):
        per_lib = {}
        for lib_name, ov in per_lib_overrides.items():
            if ov is None:
                continue
            entry: dict[str, float] = {}
            if "gm" in ov and ov["gm"] is not None:
                entry["gm"] = float(ov["gm"])
            if "isat" in ov and ov["isat"] is not None:
                entry["isat"] = float(ov["isat"])
            per_lib[str(lib_name)] = entry

    for stage_idx, stage in enumerate(net.core.stages):
        for lib_name in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
            lib = getattr(stage, lib_name, None)
            if lib is None:
                continue
            ov_gm = gm_init
            ov_isat = isat_init
            if per_lib is not None and lib_name in per_lib:
                e = per_lib[lib_name]
                ov_gm = float(e.get("gm", gm_init))
                ov_isat = float(e.get("isat", isat_init))
            if hasattr(lib, "gm_raw"):
                with torch.no_grad():
                    lib.gm_raw.data.fill_(float(ov_gm))
            if hasattr(lib, "isat_raw"):
                with torch.no_grad():
                    lib.isat_raw.data.fill_(float(ov_isat))

        # Leak dispatch.
        if leak_mode == "randomized":
            if not hasattr(stage, "raw_leak"):
                raise ValueError(
                    f"leak_mode='randomized' requires programmable leak "
                    f"on stage (no raw_leak found)."
                )
            gen = torch.Generator(device="cpu").manual_seed(int(raw_leak_init_seed))
            with torch.no_grad():
                stage.raw_leak.data.copy_(
                    torch.randn(stage.raw_leak.shape, generator=gen)
                    * float(raw_leak_init_std)
                    + float(raw_leak_init_mean)
                )
            stage.leak_mode = "programmable"
        elif leak_mode == "slow-fixed":
            stage.leak_mode = "non-programmable"
            stage.leak_constant = float(_CONFIG_INIT["leak_constant"])
            # ``raw_leak`` (if it exists) is now a no-op parameter that
            # the optimizer still updates; we leave it at the constructor
            # default so existing checkpoints load byte-identically.
            # The forward path picks up ``leak_constant`` instead.
        elif isinstance(leak_mode, str) and leak_mode.startswith("hetero:"):
            # Step 4 hook: hetero-leak init via log-uniform tau.
            # Format: ``hetero:<tau_lo>:<tau_hi>`` (e.g.
            # ``hetero:1.0:40.0``). Colons, NOT commas: the e0 CLI
            # splits --leak-grid on commas, so a comma inside the token
            # would fracture it into bogus modes (Alliance 2026-09-12:
            # "unknown leak_mode 'tau_hi=40.0'"). Colons never appear in
            # floats, so this form survives grid splitting intact.
            from narma_revised_plan import hetero_leak_init as _hetero
            _rest = leak_mode[len("hetero:"):]
            try:
                _lo_s, _hi_s = _rest.split(":")
                tau_lo, tau_hi = float(_lo_s), float(_hi_s)
            except ValueError:
                raise ValueError(
                    f"hetero leak_mode must be 'hetero:<tau_lo>:<tau_hi>', "
                    f"got {leak_mode!r}"
                )
            hl_log = _hetero(
                stage, tau_lo=tau_lo, tau_hi=tau_hi,
                seed=int(raw_leak_init_seed),
            )
            log.update({f"stage{stage_idx}_{k}": v for k, v in hl_log.items()})
        elif isinstance(leak_mode, str) and leak_mode.startswith("raw:"):
            # Regime-transplant hook: programmable scalar raw_leak fill.
            # Format: ``raw:<val>`` (e.g. ``raw:-0.5``). Sets
            # ``leak_mode="programmable"`` and fills every node's
            # ``raw_leak`` with the scalar, giving an effective leak of
            # ``softplus(val)`` (heun path has leak_floor=0.0). This is
            # NOT the same as numeric leak_mode (which sets the fixed
            # ``leak_constant`` directly): ``raw:-0.5`` -> leak ~0.47,
            # while ``-0.5`` as a bare number -> leak_constant=-0.5
            # (negative leak, anti-damping). Colons, never commas, so
            # the token survives --leak-grid splitting intact.
            if not hasattr(stage, "raw_leak"):
                raise ValueError(
                    f"leak_mode={leak_mode!r} requires programmable leak "
                    f"on stage (no raw_leak found)."
                )
            try:
                _raw_val = float(leak_mode[len("raw:"):])
            except (TypeError, ValueError):
                raise ValueError(
                    f"raw leak_mode must be 'raw:<val>', got {leak_mode!r}"
                )
            with torch.no_grad():
                stage.raw_leak.data.fill_(_raw_val)
            stage.leak_mode = "programmable"
        else:
            try:
                leak_val = float(leak_mode)
            except (TypeError, ValueError):
                raise ValueError(
                    f"unknown leak_mode {leak_mode!r}; expected "
                    "'slow-fixed' | 'randomized' | 'hetero:<tau_lo>:<tau_hi>' "
                    "| 'raw:<val>' | <numeric scalar>"
                )
            stage.leak_mode = "non-programmable"
            stage.leak_constant = leak_val
        log[f"stage{stage_idx}_leak_mode"] = stage.leak_mode
        if hasattr(stage, "leak_constant"):
            log[f"stage{stage_idx}_leak_constant"] = float(stage.leak_constant)

    return log


def _select_transition_points(
    states_full: torch.Tensor, u_seq: torch.Tensor, *,
    washout: int, n_samples: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Select deterministic mid-stream state/input transitions.

    Transition ``t`` uses the actually observed state ``states_full[t]``
    and the following input ``u_seq[t+1]``, so the Jacobian is evaluated
    at a genuine operating point of the collected trajectory.
    """
    if n_samples < 1:
        raise ValueError(f"need at least one Jacobian sample, got {n_samples}")
    if u_seq.dim() != 1 or states_full.dim() != 2:
        raise ValueError(
            "transition selection requires (T,) inputs and (T, N) states, got "
            f"{tuple(u_seq.shape)} and {tuple(states_full.shape)}"
        )
    if states_full.shape[0] != u_seq.shape[0]:
        raise ValueError(
            "states and inputs must have the same length, got "
            f"{states_full.shape[0]} and {u_seq.shape[0]}"
        )
    usable = states_full.shape[0] - washout - 1
    if usable < 1:
        raise ValueError(
            f"need at least one post-washout transition, got T={states_full.shape[0]} "
            f"with washout={washout}"
        )
    count = min(n_samples, usable)
    indices = torch.linspace(
        washout, states_full.shape[0] - 2, steps=count,
    ).round().to(torch.long).unique()
    transitions: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for transition_index in indices.tolist():
        x_from = states_full[int(transition_index)].detach().clone().requires_grad_(True)
        u_next = u_seq[int(transition_index) + 1].reshape(1, 1)
        transitions.append((x_from, u_next, int(transition_index)))
    return transitions


def _one_sample_transition(
    stage: nn.Module, x_from: torch.Tensor, u_next: torch.Tensor,
    dt: float, num_steps: int,
) -> torch.Tensor:
    """Evaluate one observed sample transition, exactly as the sequence path.

    This reproduces :meth:`DifferentialStage._forward_heun_sequence` for one
    sample, including VCA gating and frozen-current semantics, so its
    Jacobian is the Jacobian of the actually used discrete map.
    """
    gate_core_cached = None
    if stage._vca_core_enabled and u_next is not None and stage.vca_v_core is not None:
        gate_core_cached = stage._compute_core_gate(u_next)
    # Phase-8 tied tap gate: same once-per-sample convention as the stage
    # sequence path — without this, driven/input Jacobians silently ignore
    # the gates and gate conditions compare identically on those metrics.
    tied_gate_cached = None
    tap_rails = getattr(stage, "tap_rails", None)
    if tap_rails is not None and u_next is not None:
        tied_gate_cached = tap_rails(u_next)
    i_edge_const = None
    if stage.freeze_read or stage.core_refresh_interval > 0:
        i_edge_const = stage._compute_i_edge_const(x_from, gate_core_cached)
    i_boundary_const = (
        stage._compute_frozen_boundary(u_next, x_from)
        if stage.freeze_boundary else None
    )
    i_readout_const = (
        stage._compute_frozen_readout(u_next, x_from)
        if stage.freeze_temporal_read else None
    )
    x_next = stage._heun_steps(
        x_from, u_next, dt, num_steps,
        i_edge_const, i_boundary_const, i_readout_const,
        gate_core_cached, tied_gate_cached,
    )
    return x_next.view(-1)


def jacobian_eigs(
    stage: nn.Module,
    transitions: list[tuple[torch.Tensor, torch.Tensor, int]], *,
    dt: float = 0.125, num_steps: int = 8,
) -> list[dict[str, float]]:
    """Jacobian eigenvalue instrument on observed single-sample transitions.

    Each transition is ``(state[t], input[t+1], t)``.  The closure
    differentiates the same one-sample Heun map that generated
    ``state[t+1]`` and reports its eigenvalues.

    Args:
        stage: A ``DifferentialStage``.
        transitions: One or more observed transitions.
        dt: Heun timestep (``t_span / num_steps``).
        num_steps: Heun steps per sample window.

    Returns:
        List of dicts (one per transition) with keys ``transition_index``,
        ``state_dim``, ``max_abs``, ``min_abs``, ``mean_abs``, and
        ``rank_proxy`` (PR over |eig|).
    """
    if not transitions:
        raise ValueError("jacobian_eigs requires at least one transition")
    stage.eval()
    rows: list[dict[str, float]] = []
    for x_from, u_next, transition_index in transitions:
        if x_from.dim() != 1:
            raise ValueError(
                f"transition states must be one-dimensional, got {tuple(x_from.shape)}"
            )

        def transition_map(x_flat: torch.Tensor) -> torch.Tensor:
            x = x_flat.view(1, -1)
            return _one_sample_transition(stage, x, u_next, dt, num_steps)

        J = torch.autograd.functional.jacobian(
            transition_map, x_from, create_graph=False,
        )
        eig = torch.linalg.eigvals(J)
        abs_eig = eig.abs()
        rows.append({
            "transition_index": float(transition_index),
            "state_dim": float(x_from.numel()),
            "max_abs": float(abs_eig.max().item()),
            "min_abs": float(abs_eig.min().item()),
            "mean_abs": float(abs_eig.mean().item()),
            "rank_proxy": float(participation_ratio(abs_eig)),
        })
    return rows


def _score_state_trajectory(
    *, stage: nn.Module, states_full: torch.Tensor,
    u_seq: torch.Tensor, y_seq: torch.Tensor,
    washout: int, jacobian_samples: int,
    t_span: float, num_steps: int,
) -> dict[str, float]:
    """Score one collected trajectory with the complete E0 instrument set."""
    if states_full.dim() != 2 or u_seq.dim() != 1 or y_seq.dim() != 1:
        raise ValueError(
            "trajectory scoring requires (T, N) states and (T,) input/targets, got "
            f"{tuple(states_full.shape)}, {tuple(u_seq.shape)}, {tuple(y_seq.shape)}"
        )
    if not (
        states_full.shape[0] == u_seq.shape[0] == y_seq.shape[0]
        and states_full.shape[0] > washout
    ):
        raise ValueError(
            "states, inputs, and targets must have the same length above washout, got "
            f"{states_full.shape[0]}, {u_seq.shape[0]}, {y_seq.shape[0]} "
            f"with washout={washout}"
        )
    x_max = float(stage.x_max)
    with torch.no_grad():
        sat_max = float(states_full.abs().max().item())
        rail_frac = float((states_full.abs() > 0.9 * x_max).float().mean().item())
    state_pr = participation_ratio(states_full[washout:])
    dt = t_span / num_steps
    transitions = _select_transition_points(
        states_full, u_seq, washout=washout, n_samples=jacobian_samples,
    )
    j_rows = jacobian_eigs(stage, transitions, dt=dt, num_steps=num_steps)
    X = states_full[washout:]
    y = y_seq[washout:]
    W = _ridge_fit_predict(X, y, l2=1e-2)
    pred = torch.cat(
        [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1,
    ) @ W
    _, mc = memory_capacity_washout(
        states_full, u_seq, washout=washout, max_delay=20,
    )
    return {
        "ridge_nrmse": ne.nrmse(pred, y),
        "ridge_r2": ne.r2(pred, y),
        "mc_total": float(mc),
        "state_pr": float(state_pr),
        "jac_max_abs": float(max(r["max_abs"] for r in j_rows)),
        "jac_min_abs": float(min(r["min_abs"] for r in j_rows)),
        "jac_mean_abs": float(
            sum(r["mean_abs"] for r in j_rows) / len(j_rows)
        ),
        "jac_rank_proxy": float(max(r["rank_proxy"] for r in j_rows)),
        "rail_frac": float(rail_frac),
        "sat_max_ratio": float(sat_max / x_max) if x_max > 0 else float("nan"),
    }


def _select_corner_subset(
    drive_grid: Iterable[float], leak_grid: Iterable[Any],
    gain_grid: Iterable[float], max_corners: int | None,
    isat_grid: Iterable[float | None] | None = None,
) -> list[tuple[float, Any, float, float | None]]:
    """Select the sweep corners without changing their traversal order.

    The full sweep traverses drive, then leak, then gain.  A positive
    ``max_corners`` returns exactly that prefix; otherwise the full grid
    is returned.

    Regime-transplant: ``isat_grid`` pairs with ``gain_grid`` POSITIONALLY
    (``zip``, not a ``gm -> isat`` dict) so duplicate gm entries keep
    distinct Isat values -- e.g. ``gain=(-3.5, -3.5)`` with
    ``isat=(None, -2.0)`` yields one coupled and one decoupled corner.
    A dict keyed by gm value would collapse the duplicates and silently
    drop a corner. ``None`` entries mean coupled (``isat == gm``).
    """
    gain_list = [float(g) for g in gain_grid]
    if isat_grid is None:
        pairs = [(g, None) for g in gain_list]
    else:
        isat_list = [None if v is None else float(v) for v in isat_grid]
        if len(isat_list) != len(gain_list):
            raise ValueError(
                f"isat_grid length ({len(isat_list)}) must match gain_grid "
                f"length ({len(gain_list)}) when provided"
            )
        pairs = list(zip(gain_list, isat_list))
    full = [
        (float(drive), leak_mode, gm_init, isat_init)
        for drive in drive_grid
        for leak_mode in leak_grid
        for gm_init, isat_init in pairs
    ]
    if not full:
        raise ValueError("gain, leak, and drive grids must all be non-empty")
    if max_corners is None or max_corners <= 0 or max_corners >= len(full):
        return full
    return full[:max_corners]


def _e0_config_tag(
    *, order: int, seed: int, device: str, hidden_dim: int, refresh: int,
    t_span: float, num_steps: int, gm_init: float, leak_mode: Any,
    drive: float, washout: int, jacobian_samples: int,
    boundary_fan_out: dict[int, list[int]] | None = None,
    isat_init: float | None = None,
    per_lib_overrides: dict[str, dict[str, float | None]] | None = None,
) -> str:
    """Corner identity shared by :func:`_e0_row` and the resume matcher.

    Round-2 §13.1: the canonical full fan-out (``{0: range(hidden_dim)}``,
    or ``None``) keeps the historical tag format **byte-identical** so
    progress files written by older runs still match. A non-canonical
    sparse fan-out appends a stable ``_fan<tgt>-<tgt>-...`` suffix built
    from the sorted unique target set. Add new axes here as suffixes only.

    Regime-transplant: when ``isat_init`` is set or ``per_lib_overrides``
    is non-None, append ``_isat<...>`` and/or ``_plib<...>`` suffixes so
    two corners with the same gm/leak/drive but different fills don't
    collide on resume. Empty / None values leave the tag unchanged
    (legacy byte-identity preserved).
    """
    base = (
        f"order{order}_seed{seed}_{device}_tanhfree"
        f"_h{int(hidden_dim)}_k{int(refresh)}"
        f"_tspan{float(t_span):g}_steps{int(num_steps)}"
        f"_gm{float(gm_init):g}_leak{leak_mode}_drive{float(drive):g}"
        f"_washout{int(washout)}_jac{int(jacobian_samples)}"
    )
    if boundary_fan_out is None:
        _tag_with_fan = base
    else:
        canon_targets = list(range(int(hidden_dim)))
        sparse_targets = sorted(
            {t for tgts in boundary_fan_out.values() for t in tgts}
        )
        if sparse_targets == canon_targets:
            _tag_with_fan = base
        else:
            _short = "-".join(str(t) for t in sparse_targets[:6])
            if len(sparse_targets) > 6:
                _short += "-etc"
            _tag_with_fan = f"{base}_fan{_short}"
    if isat_init is not None:
        _tag_with_fan = f"{_tag_with_fan}_isat{float(isat_init):g}"
    if per_lib_overrides:
        # Stable encoding: sorted library names, gm and isat rounded to
        # ``:g`` to avoid float drift making two equivalent dicts
        # produce different tags.
        _plibs = sorted(per_lib_overrides.items())
        _parts = []
        for _lib, _ov in _plibs:
            _g = _ov.get("gm") if isinstance(_ov, dict) else None
            _i = _ov.get("isat") if isinstance(_ov, dict) else None
            _g_str = f"g{float(_g):g}" if _g is not None else "g-"
            _i_str = f"i{float(_i):g}" if _i is not None else "i-"
            _parts.append(f"{_lib[0]}{_g_str}{_i_str}")
        _tag_with_fan = f"{_tag_with_fan}_plib{'-'.join(_parts)}"
    return _tag_with_fan


def _bfo_json(boundary_fan_out: dict[Any, Any] | None) -> str:
    """Canonical JSON encoding for a boundary fan-out map (Round-2 §13.1).

    ``None`` -> ``""`` (canonical full fan-out; keeps CSV cells and resume
    keys readable). Otherwise ``{int(k): [int, ...]}`` with sorted keys so
    ``{"0": [...]}`` and ``{0: [...]}`` encode identically and resume
    matching is order-stable.
    """
    if boundary_fan_out is None:
        return ""
    return json.dumps(
        {int(k): [int(v) for v in vs] for k, vs in sorted(boundary_fan_out.items())},
        sort_keys=True,
    )


def _write_e0_progress(path: Path, payload: dict[str, Any]) -> None:
    """Write the e0 progress file atomically (temp + os.replace).

    A kill can never leave a truncated progress file behind: readers see
    either the previous complete write or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read_e0_progress(path: Path) -> dict[str, Any] | None:
    """Load a progress file, or None when missing/corrupt.

    Corrupt (killed-mid-write without atomic replace, hand-edited) files
    fail safe: the sweep reruns everything instead of trusting garbage.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _e0_row(
    net: nn.Module, u_train: torch.Tensor, y_train: torch.Tensor, *,
    order: int, seed: int, device: str,
    gm_init: float, leak_mode: Any, drive: float,
    raw_leak_init_seed: int = 0, washout: int = PROBE_WASHOUT,
    jacobian_samples: int = 3,
    boundary_fan_out: dict[int, list[int]] | None = None,
    isat_init: float | None = None,
    per_lib_overrides: dict[str, dict[str, float | None]] | None = None,
    skip_override: bool = False,
) -> E0SweepRow:
    """Score one E0 sweep corner from normal and boundary-only passes."""
    if len(list(net.core.stages)) != 1:
        raise NotImplementedError(
            "E0 corners currently support single-stage nets only"
        )
    if u_train.dim() != 1 or y_train.dim() != 1:
        raise ValueError(
            "E0 corners require one-dimensional input/target streams, got "
            f"{tuple(u_train.shape)} and {tuple(y_train.shape)}"
        )
    if u_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "E0 input and target streams must have the same length, got "
            f"{u_train.shape[0]} and {y_train.shape[0]}"
        )
    if u_train.shape[0] <= washout:
        raise ValueError(
            f"need more than {washout} washed-out samples, got {u_train.shape[0]}"
        )
    net.eval()
    # Gain + leak override (cell_library defaults untouched).  Drive is
    # already represented by the scaled input sequence. ``isat_init``
    # decouples Isat from gm when set; ``per_lib_overrides`` overrides
    # individual cell libraries (regime-transplant hook). Both default to
    # the legacy coupled/symmetric path byte-identically.
    # ``skip_override=True`` is used by the exact-tensor anchor corner
    # so the canonical override does NOT overwrite the pre-filled tensors.
    if not skip_override:
        apply_gain_override(
            net, gm_init=gm_init, isat_init=isat_init, leak_mode=leak_mode,
            raw_leak_init_seed=raw_leak_init_seed,
            per_lib_overrides=per_lib_overrides,
        )
    stage = net.core.stages[0]
    state_width = net.hid_count + net.proj_count + net.output_ode_count
    x0 = u_train.new_zeros(1, state_width)
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_train,
        )
    states_full = all_states[:, 0, :].detach()
    normal = _score_state_trajectory(
        stage=stage, states_full=states_full,
        u_seq=u_train, y_seq=y_train,
        washout=washout, jacobian_samples=jacobian_samples,
        t_span=t_span, num_steps=num_steps,
    )

    # Boundary-only pass: suppress core-edge gates transiently, then score
    # the same instruments on the resulting trajectory.
    gate_states_full = _gate_zero_eval_forward(net, u_train)[:, 0, :].detach()
    boundary_only = _score_state_trajectory(
        stage=stage, states_full=gate_states_full,
        u_seq=u_train, y_seq=y_train,
        washout=washout, jacobian_samples=jacobian_samples,
        t_span=t_span, num_steps=num_steps,
    )

    rail_disqualified = normal["rail_frac"] > RAIL_DISQUALIFY_ABOVE
    pass_init_regime = (
        normal["mc_total"] > E0_PASS_MC_ABOVE
        and normal["ridge_nrmse"] <= E0_PASS_RIDGE_BELOW
        and normal["state_pr"] >= E0_PASS_STATE_PR_MIN
    )
    gate_zero_rail_disqualified = (
        boundary_only["rail_frac"] > RAIL_DISQUALIFY_ABOVE
    )
    gate_zero_pass_init_regime = (
        boundary_only["mc_total"] > E0_PASS_MC_ABOVE
        and boundary_only["ridge_nrmse"] <= E0_PASS_RIDGE_BELOW
        and boundary_only["state_pr"] >= E0_PASS_STATE_PR_MIN
    )
    hidden_dim = int(net.hid_count)
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    config_tag = _e0_config_tag(
        order=order, seed=seed, device=device, hidden_dim=hidden_dim,
        refresh=int(stage.core_refresh_interval),
        t_span=t_span, num_steps=num_steps,
        gm_init=gm_init, leak_mode=leak_mode, drive=drive,
        washout=washout, jacobian_samples=jacobian_samples,
        boundary_fan_out=boundary_fan_out,
        # Regime-transplant: the row tag must carry the same isat /
        # per-lib suffixes as the resume key in e0_sweep, or resume
        # never matches and every rerun recomputes decoupled corners.
        isat_init=isat_init,
        per_lib_overrides=per_lib_overrides,
    )
    return E0SweepRow(
        config_tag=config_tag,
        hidden_dim=hidden_dim,
        n_params=n_params,
        gm_init=float(gm_init),
        # NaN sentinel: caller didn't pass a decoupled isat_init, so the
        # legacy coupled fill was used. CSV/log readers can detect this
        # via float('nan') to mean "isat == gm for this corner".
        isat_init=(
            float(isat_init) if isat_init is not None else float("nan")
        ),
        leak_mode=str(leak_mode),
        drive=float(drive),
        per_lib_overrides_json=(
            json.dumps(per_lib_overrides, sort_keys=True)
            if per_lib_overrides else ""
        ),
        # Round-2 §13.1: store the JSON-encoded fan-out so CSV dumps
        # carry the build flag. The e0_sweep caller overrides this
        # with the canonical-form string on the way out; defaulting to
        # an empty string keeps legacy code that constructs E0SweepRow
        # directly (resume path) working.
        boundary_fan_out="",
        ridge_nrmse=normal["ridge_nrmse"],
        ridge_r2=normal["ridge_r2"],
        mc_total_washout_corrected=normal["mc_total"],
        state_pr=normal["state_pr"],
        jac_max_abs=normal["jac_max_abs"],
        jac_min_abs=normal["jac_min_abs"],
        jac_mean_abs=normal["jac_mean_abs"],
        jac_rank_proxy=normal["jac_rank_proxy"],
        rail_frac=normal["rail_frac"],
        sat_max_ratio=normal["sat_max_ratio"],
        gate_zero_ridge_nrmse=boundary_only["ridge_nrmse"],
        gate_zero_ridge_r2=boundary_only["ridge_r2"],
        gate_zero_mc_total=boundary_only["mc_total"],
        gate_zero_state_pr=boundary_only["state_pr"],
        gate_zero_jac_max_abs=boundary_only["jac_max_abs"],
        gate_zero_jac_min_abs=boundary_only["jac_min_abs"],
        gate_zero_jac_mean_abs=boundary_only["jac_mean_abs"],
        gate_zero_jac_rank_proxy=boundary_only["jac_rank_proxy"],
        gate_zero_rail_frac=boundary_only["rail_frac"],
        gate_zero_sat_max_ratio=boundary_only["sat_max_ratio"],
        rail_disqualified=rail_disqualified,
        pass_init_regime=pass_init_regime,
        pass_rail=not rail_disqualified,
        pass_all=pass_init_regime and not rail_disqualified,
        gate_zero_rail_disqualified=gate_zero_rail_disqualified,
        gate_zero_pass_init_regime=gate_zero_pass_init_regime,
        gate_zero_pass_rail=not gate_zero_rail_disqualified,
        gate_zero_pass_all=(
            gate_zero_pass_init_regime and not gate_zero_rail_disqualified
        ),
    )


def _make_anchor_pre_fill(ckpt_path: str | None, device: str) -> Any:
    """Return a ``(net) -> None`` hook that pre-fills dynamics tensors.

    Loads ``ckpt_path`` once and returns a closure suitable for
    ``e0_sweep(..., pre_fill=...)``. The closure copies the checkpoint's
    dynamics tensors (per-library ``gm_raw`` / ``isat_raw``, per-stage
    ``raw_leak``, gate logits ``z_logits`` / ``boundary_z_logits`` /
    ``output_ode_z_logits`` / ``u_logits``, and the remaining trainable
    cell params ``a/b/s/theta/g_resistive_raw``) from the checkpoint into
    the freshly built net. Readout tensors (``output_mapper.*``,
    ``post_readout_transfer.*``) are NEVER copied -- E0 refits the
    readout via ridge, so copying them would only pretend the anchor
    carries a trained readout. Index buffers (``src``/``dst`` maps) are
    skipped: they describe topology, not regime, and must come from the
    fresh build.

    Strict load: every expected dynamics key must be present in the
    checkpoint with a matching shape (e.g. same hidden_dim / cell
    library); a missing or mismatched key hard-fails instead of scoring
    a half-filled net. The caller is responsible for ensuring the ckpt
    and the build are compatible.

    Returns ``None`` when ``ckpt_path`` is ``None`` (no pre-fill).
    """
    if ckpt_path is None:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    if not isinstance(state, dict):
        raise ValueError(
            f"--anchor-ckpt: not a state_dict container, got {type(state).__name__}"
        )

    # Readout prefixes are never copied (ridge refit owns the readout).
    # Index buffers are topology, not regime (fresh build owns them).
    _SKIP_PREFIXES = ("output_mapper.", "post_readout_transfer.")
    _SKIP_SUFFIXES = ("_src", "_dst", "_drive_idx", "_drive_mask")
    _REQUIRED_SUBSTRINGS = ("gm_raw", "isat_raw", "raw_leak")

    def _pre_fill(net: nn.Module) -> None:
        net.to("cpu")
        own = dict(net.named_parameters())
        own_bufs = dict(net.named_buffers())
        # Expected dynamics keys: every gm_raw/isat_raw/raw_leak tensor
        # the fresh build owns must be covered by the checkpoint.
        expected = {
            k for k in list(own) + list(own_bufs)
            if any(s in k for s in _REQUIRED_SUBSTRINGS)
        }
        missing = [k for k in sorted(expected) if k not in state]
        if missing:
            raise ValueError(
                f"--anchor-ckpt: checkpoint is missing {len(missing)} "
                f"dynamics tensor(s), e.g. {missing[0]!r}; build and ckpt "
                f"are incompatible (different hidden_dim / cell library?)"
            )
        copied = 0
        for k, src in state.items():
            if k.startswith(_SKIP_PREFIXES) or k.endswith(_SKIP_SUFFIXES):
                continue
            src_t = src.detach().to("cpu") if torch.is_tensor(src) else src
            if not torch.is_tensor(src_t):
                continue
            if k in own:
                tgt = own[k]
                if tuple(tgt.shape) != tuple(src_t.shape):
                    raise ValueError(
                        f"--anchor-ckpt: shape mismatch on {k!r} "
                        f"(ckpt {tuple(src_t.shape)} vs net "
                        f"{tuple(tgt.shape)}); build and ckpt are "
                        f"incompatible"
                    )
                with torch.no_grad():
                    tgt.data.copy_(src_t.to(tgt.dtype))
                copied += 1
            elif k in own_bufs:
                tgt = own_bufs[k]
                if tuple(tgt.shape) != tuple(src_t.shape):
                    raise ValueError(
                        f"--anchor-ckpt: shape mismatch on buffer {k!r}"
                    )
                tgt.data.copy_(src_t.to(tgt.dtype))
                copied += 1
            # Silently skip checkpoint-only keys the fresh build doesn't
            # own (e.g. optimizer-adjacent extras); the strict gate above
            # already guaranteed every expected dynamics key was covered.
        if copied == 0:
            raise ValueError(
                "--anchor-ckpt: no dynamics tensors copied; ckpt is "
                "incompatible with the build (different cell library / "
                "topology?)"
            )
        net.to(device)

    return _pre_fill



def e0_sweep(
    order: int, seed: int, *, device: str = "cpu",
    gain_grid: Iterable[float] = E0_GAIN_GRID,
    leak_grid: Iterable[Any] = E0_LEAK_GRID,
    drive_grid: Iterable[float] = E0_DRIVE_GRID,
    n_streams: int = 4, train_samples_per_stream: int = 2500,
    washout: int = PROBE_WASHOUT, jacobian_samples: int = 3,
    t_span: float = 1.0, num_steps: int = 8, hidden_dim: int = 25,
    net_factory: Any | None = None,
    selected_corners: Iterable[tuple] | None = None,
    isat_grid: Iterable[float | None] | None = None,
    per_lib_overrides: dict[str, dict[str, float | None]] | None = None,
    pre_fill: Any | None = None,
    pre_fill_overrides_after: bool = False,
    refresh: int = 0,
    use_small_world: bool = False,
    small_world_k: int = 4,
    small_world_p: float = 0.2,
    small_world_seed: int = 1,
    progress_json: Path | str | None = None,
    boundary_fan_out: dict[int, list[int]] | None = None,
) -> list[E0SweepRow]:
    """Run the locked E0 (zero-training) gain/leak/drive sweep.

    Per-corner: build a fresh net, override gains/leak, scale drive,
    collect full states once on the first train stream, fit ridge, compute
    washout-corrected MC, evaluate Jacobian eigenvalues at observed
    mid-stream transitions, and measure rail fraction.  Returns a list of
    :class:`E0SweepRow`.

    The canonical training pipeline generates ``n_streams`` streams, but an
    E0 probe is eval-only and uses only stream zero, matching the legacy
    MC input.  Stream zero is seed-identical whether the other streams are
    generated or not.

    Args:
        net_factory: Optional callable ``() -> nn.Module``.  If omitted,
            uses the canonical NARMA fabric built via
            ``narma_experiment._build_fabric_net``.
        selected_corners: Optional explicit corner list. Entries are
            ``(drive, leak, gm)`` or ``(drive, leak, gm, isat)``
            (3-tuples pad ``isat=None`` = coupled). Used by
            ``--max-corners`` so smoke runs take an exact prefix of
            the traversal order.
        isat_grid: Optional per-gain Isat values, paired with
            ``gain_grid`` POSITIONALLY (see :func:`_select_corner_subset`).
            ``None`` entries restore the legacy coupled-isat behaviour
            for that gm. Duplicate gm entries keep distinct Isat values.
        refresh: ``core_refresh_interval`` applied to every built stage
            (default 0 = legacy frozen). The exact-tensor anchor corner
            must use the checkpoint's own ``k`` (k2 ckpt -> ``refresh=2``)
            or the forward won't reproduce the trained trajectory.
        pre_fill: Optional ``(net) -> None`` hook (see
            :func:`_make_anchor_pre_fill`). Runs BEFORE the gm/Isat/leak
            override pass when ``pre_fill_overrides_after`` is True, or
            replaces it (``skip_override``) when False (pure anchor).
        use_small_world: If True, build the NARMA net with
            ``hidden_family='small_world'`` (Step 4 grid axis). Replaces
            the canonical 5x5 torus with a Watts-Strogatz graph of the
            same node count. Random-sign fan-out is NOT yet wired (plan
            sec. 7 notes this can be added once a corner passes).
        small_world_k / p / seed: Parameters of the Watts-Strogatz
            graph. See ``narma_revised_plan.build_small_world_narma_preset``.
        progress_json: Optional path to a progress file. After every
            corner the rows-so-far are flushed there (``complete: false``);
            on entry, corners whose ``config_tag`` is already present are
            skipped, so a killed sweep resumes instead of restarting. A
            build-flag mismatch (or a custom ``net_factory``) disables
            resume -- rerunning is always the safe direction. ``None``
            (default) disables progress tracking entirely.
    """
    if n_streams < 1:
        raise ValueError(f"n_streams must be positive, got {n_streams}")
    if train_samples_per_stream <= washout:
        raise ValueError(
            f"need more than {washout} samples, got {train_samples_per_stream}"
        )
    if jacobian_samples < 1:
        raise ValueError(f"jacobian_samples must be positive, got {jacobian_samples}")
    if t_span <= 0:
        raise ValueError(f"t_span must be positive, got {t_span}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if refresh < 0:
        raise ValueError(f"refresh must be non-negative, got {refresh}")
    # Regime-transplant: isat pairs with gain POSITIONALLY via
    # _select_corner_subset (duplicate gm entries keep distinct Isat).
    if selected_corners is None:
        corners = _select_corner_subset(
            drive_grid, leak_grid, gain_grid, None,
            isat_grid=isat_grid,
        )
    else:
        corners = []
        for _c in selected_corners:
            if len(_c) == 3:
                _d, _l, _g = _c
                corners.append((float(_d), _l, float(_g), None))
            elif len(_c) == 4:
                _d, _l, _g, _i = _c
                corners.append((
                    float(_d), _l, float(_g),
                    None if _i is None else float(_i),
                ))
            else:
                raise ValueError(
                    "selected_corners entries must be (drive, leak, gm) or "
                    f"(drive, leak, gm, isat), got {tuple(_c)!r}"
                )
        if not corners:
            raise ValueError("selected_corners must be non-empty")

    # Stream zero is the probe stream; other generated streams preserve the
    # canonical training-stream construction.
    u_train_raw, y_train_raw = ne._gen_narma_train_streams(
        order=order, seed=seed, n_streams=n_streams,
        n=train_samples_per_stream,
    )
    u_stream_raw = u_train_raw[0]
    y_stream_raw = y_train_raw[0]

    # Resolve the default net factory: small_world preset when
    # ``use_small_world`` is True, canonical torus otherwise. Both
    # factories ignore their arguments and always rebuild via the seeded
    # RNG so corner reproducibility holds (corners share the same seed).
    # Capture custom-factory status BEFORE resolution: a caller-supplied
    # factory has an opaque build, so it disables progress resume below.
    # Round-2 §13.1: ``boundary_fan_out`` is forwarded into the build so
    # sparse-drive E0 corners use the same plumbing as the training path.
    # ``None`` keeps the canonical full fan-out untouched.
    _allow_resume = net_factory is None
    if net_factory is None:
        if use_small_world:
            from topology import build_net_from_config
            from cell_library import make_cell_library
            from config import make_narma_preset as _make_narma_preset
            from narma_revised_plan import build_small_world_narma_preset

            def _small_world_factory() -> nn.Module:
                preset = build_small_world_narma_preset(
                    order=order, hidden_dim=hidden_dim,
                    num_steps_per_sample=num_steps,
                    t_span=t_span, small_world_k=small_world_k,
                    small_world_p=small_world_p,
                    small_world_seed=small_world_seed,
                    core_refresh_interval=refresh,
                    leak_constant=None,
                )
                torch.manual_seed(seed)
                cell_lib = make_cell_library("tanh_free")
                _bfo = (
                    preset["boundary_fan_out"]
                    if boundary_fan_out is None else boundary_fan_out
                )
                return build_net_from_config(
                    cfg=preset, cell_lib=cell_lib,
                    boundary_fan_out=_bfo,
                    enable_temporal_readout=True,
                    freeze_read=False,
                )

            net_factory = _small_world_factory
        else:
            def _torus_factory() -> nn.Module:
                net, _t_span, _num_steps = ne._build_fabric_net(
                    order=order, seed=seed, freeze_read=False,
                    t_span=t_span, num_steps=num_steps,
                    cell_library="tanh_free",
                    hidden_dim=hidden_dim,
                    core_refresh_interval=refresh,
                    leak_constant=None,
                    compile_sequence=False,
                    boundary_fan_out=boundary_fan_out,
                )
                return net

            net_factory = _torus_factory

    # Corner-level resume. Corners are pure functions of (seed, grid,
    # build flags), matched by config_tag, so skipping completed corners
    # is exact. Resume is keyed on the build flags below: any mismatch
    # (different topology, geometry, or protocol) ignores the progress
    # file and reruns everything. A custom net_factory also disables
    # resume -- its build is opaque, so rerunning is the safe direction.
    _progress_path = Path(progress_json) if progress_json is not None else None
    _build_key: dict[str, Any] = {
        "cell_library": "tanh_free",
        "freeze_read": False,
        "core_refresh_interval": int(refresh),
        "use_small_world": bool(use_small_world),
        "small_world_k": int(small_world_k),
        "small_world_p": float(small_world_p),
        "small_world_seed": int(small_world_seed),
        "hidden_dim": int(hidden_dim),
        "t_span": float(t_span),
        "num_steps": int(num_steps),
        "washout": int(washout),
        "jacobian_samples": int(jacobian_samples),
        # Round-2 §13.1: include the boundary-fan-out map so resume can
        # distinguish sparse-drive corners from full-fan-out corners.
        # ``None`` encodes as the "canonical-full" marker; an explicit map
        # (even a canonical-equivalent one) encodes as JSON.
        "boundary_fan_out": (
            "canonical-full" if boundary_fan_out is None
            else _bfo_json(boundary_fan_out)
        ),
    }
    _done: dict[str, dict[str, Any]] = {}
    if _progress_path is not None and _allow_resume:
        _saved = _read_e0_progress(_progress_path)
        if (
            _saved is not None
            and _saved.get("build") == _build_key
            and isinstance(_saved.get("rows"), list)
        ):
            for _rd in _saved["rows"]:
                if isinstance(_rd, dict) and "config_tag" in _rd:
                    _done[str(_rd["config_tag"])] = _rd

    def _progress_payload(
        _rows: list[E0SweepRow], *, complete: bool,
    ) -> dict[str, Any]:
        def _leak_json(v: Any) -> Any:
            return v if isinstance(v, (int, float, str)) else str(v)

        return {
            "order": order,
            "seed": seed,
            "device": device,
            "build": _build_key,
            "grids": {
                "gain": [float(g) for _, _, g, _ in corners],
                "leak": [_leak_json(m) for _, m, _, _ in corners],
                "drive": [float(d) for d, _, _, _ in corners],
                "isat": [
                    (None if i is None else float(i))
                    for _, _, _, i in corners
                ],
            },
            "n_corners_total": len(corners),
            "n_corners_done": len(_rows),
            "complete": complete,
            "rows": [asdict(r) for r in _rows],
        }

    rows: list[E0SweepRow] = []
    scaled_cache: dict[float, torch.Tensor] = {}
    for drive, leak_mode, gm_init, isat_for_corner in corners:
        _tag = _e0_config_tag(
            order=order, seed=seed, device=device, hidden_dim=hidden_dim,
            refresh=refresh, t_span=t_span, num_steps=num_steps,
            gm_init=float(gm_init), leak_mode=leak_mode,
            drive=float(drive), washout=washout,
            jacobian_samples=jacobian_samples,
            boundary_fan_out=boundary_fan_out,
            isat_init=(
                isat_for_corner if isat_for_corner is not None else None
            ),
            per_lib_overrides=per_lib_overrides,
        )
        if _allow_resume and _tag in _done:
            try:
                rows.append(E0SweepRow(**_done[_tag]))
            except TypeError:
                # Schema drift between the run that wrote the progress
                # file and this code: rerun the corner instead of
                # trusting a row we cannot reconstruct.
                pass
            else:
                continue
        drive_value = float(drive)
        if drive_value not in scaled_cache:
            scaled_cache[drive_value] = ne._scale_drive(
                u_stream_raw, bipolar=True, order=order,
                input_scale=drive_value,
            )
        u_drive = scaled_cache[drive_value].to(device)
        y_drive = y_stream_raw.to(device)
        # Build a fresh net for each corner so the overridden init is
        # reproducible and no corner inherits another corner's override.
        net = net_factory()
        net.to(device)
        # Regime-transplant: optional exact-tensor pre-fill. When set,
        # overwrite the freshly-built net's dynamics tensors with values
        # from a reference checkpoint (e.g. the k2 epoch-54 artifact)
        # before the gm/Isat/leak override pass. With
        # ``pre_fill_overrides_after=False`` (pure anchor) the override
        # pass is skipped via ``skip_override`` so the exact tensors
        # survive; with True the canonical override runs on top of the
        # pre-fill (partial pre-fill + override handles the rest).
        _skip = not (pre_fill is None or pre_fill_overrides_after)
        if pre_fill is not None:
            pre_fill(net)
        row = _e0_row(
            net, u_drive, y_drive,
            order=order, seed=seed, device=device,
            gm_init=float(gm_init), leak_mode=leak_mode,
            drive=drive_value, raw_leak_init_seed=seed,
            washout=washout, jacobian_samples=jacobian_samples,
            boundary_fan_out=boundary_fan_out,
            isat_init=isat_for_corner,
            per_lib_overrides=per_lib_overrides,
            skip_override=_skip,
        )
        # Stamp the JSON-encoded boundary_fan_out so the CSV row joins
        # against the build flag (canonical runs keep an empty string so
        # the column reads naturally). Direct assignment: the dataclass is
        # mutable, so no reconstruction is needed.
        row.boundary_fan_out = _bfo_json(boundary_fan_out)
        rows.append(row)
        if _progress_path is not None:
            _write_e0_progress(
                _progress_path, _progress_payload(rows, complete=False),
            )
    if _progress_path is not None:
        _write_e0_progress(
            _progress_path, _progress_payload(rows, complete=True),
        )
    return rows


# ---------------------------------------------------------------------------
# Regime-transplant grid + canary (plan narma-regime-transplant)
# ---------------------------------------------------------------------------

# Full transplant grid (feature spec regime-transplant-grid): 2 gm x 3 Isat
# x 3 raw-leak x 3 swing = 54 corners. Gain/Isat pair POSITIONALLY (duplicate
# gm entries keep distinct Isat values -- see _select_corner_subset).
# ``None`` Isat = coupled (isat == gm). Raw leaks map to effective leaks
# softplus(raw) = 0.20 / 0.47 / 0.97 (heun leak_floor=0.0); swings below the
# historical 0.25 floor ride the existing input-scale path.
TRANSPLANT_GM_GRID: tuple[float, ...] = (-3.5, -3.5, -3.5, -3.0, -3.0, -3.0)
TRANSPLANT_ISAT_GRID: tuple[float | None, ...] = (
    None, -2.0, -6.0, None, -2.0, -6.0,
)
TRANSPLANT_LEAK_GRID: tuple[str, ...] = ("raw:-1.5", "raw:-0.5", "raw:0.5")
TRANSPLANT_DRIVE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3)

# Advisor priority canary corner: high core Isat at moderate core gm, fast
# leak, swing 0.2. The core library differs (per-lib override); boundary
# and output libraries use the coupled global fill.
TRANSPLANT_CANARY_GM: float = -3.5
TRANSPLANT_CANARY_LEAK: str = "raw:-0.5"
TRANSPLANT_CANARY_DRIVE: float = 0.2
TRANSPLANT_CANARY_PER_LIB: dict[str, dict[str, float | None]] = {
    "cell_lib": {"gm": -3.5, "isat": -2.0},
}

# Canary verdict gates (spec; deliberately looser than the E0 sweep's
# pass_all, which also requires state-PR >= 6).
CANARY_PASS_MC_ABOVE: float = 4.0
CANARY_PASS_RIDGE_BELOW: float = 0.6
CANARY_ANCHOR_BAND: tuple[float, float] = (0.60, 0.68)
CANARY_BLIND_AT_OR_ABOVE: float = 0.72
# Gate-zero "flip": core edges carry information when forcing them off
# degrades the ridge by more than this (and MC drops too).
CANARY_GZ_FLIP_DELTA: float = 0.05


def transplant_corner_count() -> int:
    """Number of corners in the full transplant grid (spec: ~54)."""
    return (
        len(TRANSPLANT_GM_GRID)
        * len(TRANSPLANT_LEAK_GRID)
        * len(TRANSPLANT_DRIVE_GRID)
    )


def canary_verdict(
    anchor_ridge: float, canary: E0SweepRow,
) -> dict[str, Any]:
    """Apply the pre-registered 4-outcome canary table.

    Returns a dict with ``outcome`` (one of ``EXPAND-TO-FULL-GRID``,
    ``TRANSPLANT-WOUNDED``, ``INSTRUMENT-BLIND``,
    ``CORE-USEFUL-NO-PASS``, ``ANCHOR-OFF-NOMINAL``), ``anchor_ridge``,
    ``canary_pass`` (spec gates MC>=4, rails<=5%, ridge<0.6),
    ``gate_zero_flip`` (core load-bearing without gate pass), and a
    human-readable ``summary``.
    """
    lo, hi = CANARY_ANCHOR_BAND
    canary_pass = (
        canary.mc_total_washout_corrected >= CANARY_PASS_MC_ABOVE
        and canary.ridge_nrmse < CANARY_PASS_RIDGE_BELOW
        and canary.rail_frac <= RAIL_DISQUALIFY_ABOVE
    )
    gate_zero_flip = (
        (canary.gate_zero_ridge_nrmse - canary.ridge_nrmse)
        > CANARY_GZ_FLIP_DELTA
        and canary.gate_zero_mc_total < canary.mc_total_washout_corrected
    )
    if anchor_ridge >= CANARY_BLIND_AT_OR_ABOVE:
        outcome = "INSTRUMENT-BLIND"
        summary = (
            f"anchor ridge {anchor_ridge:.4f} >= {CANARY_BLIND_AT_OR_ABOVE}: "
            f"instrument blind, all historical FAILs voided -- stop and "
            f"fix the meter, do not expand."
        )
    elif lo <= anchor_ridge <= hi and canary_pass:
        outcome = "EXPAND-TO-FULL-GRID"
        summary = (
            f"anchor {anchor_ridge:.4f} in [{lo}, {hi}] (calibrated) + "
            f"canary passes (MC={canary.mc_total_washout_corrected:.2f}, "
            f"ridge={canary.ridge_nrmse:.4f}, "
            f"rail={100.0 * canary.rail_frac:.1f}%) -- expand to the full "
            f"{transplant_corner_count()}-corner grid, then a short k=2 "
            f"train from the winning init."
        )
    elif lo <= anchor_ridge <= hi and gate_zero_flip:
        outcome = "CORE-USEFUL-NO-PASS"
        summary = (
            f"anchor {anchor_ridge:.4f} calibrated but canary fails the "
            f"gates (MC={canary.mc_total_washout_corrected:.2f}, "
            f"ridge={canary.ridge_nrmse:.4f}); gate-zero flip "
            f"(gz ridge={canary.gate_zero_ridge_nrmse:.4f}) confirms the "
            f"core is useful -- full grid with core-Isat as lead axis."
        )
    elif lo <= anchor_ridge <= hi:
        outcome = "TRANSPLANT-WOUNDED"
        summary = (
            f"anchor {anchor_ridge:.4f} calibrated but canary fails "
            f"(MC={canary.mc_total_washout_corrected:.2f}, "
            f"ridge={canary.ridge_nrmse:.4f}, no gate-zero flip) -- "
            f"transplant wounded; re-examine the ESN->fabric translation, "
            f"do not burn GPU on the full grid."
        )
    else:
        outcome = "ANCHOR-OFF-NOMINAL"
        summary = (
            f"anchor ridge {anchor_ridge:.4f} outside both the calibrated "
            f"band [{lo}, {hi}] and the blind threshold "
            f">={CANARY_BLIND_AT_OR_ABOVE}: unregistered zone -- inspect "
            f"before any expand/stop decision."
        )
    return {
        "outcome": outcome,
        "summary": summary,
        "anchor_ridge": float(anchor_ridge),
        "anchor_band": [lo, hi],
        "blind_at_or_above": CANARY_BLIND_AT_OR_ABOVE,
        "canary_pass": bool(canary_pass),
        "canary_gates": {
            "mc_above_or_equal": CANARY_PASS_MC_ABOVE,
            "ridge_below": CANARY_PASS_RIDGE_BELOW,
            "rail_at_or_below": RAIL_DISQUALIFY_ABOVE,
        },
        "gate_zero_flip": bool(gate_zero_flip),
        "gate_zero_flip_delta": CANARY_GZ_FLIP_DELTA,
    }


# ---------------------------------------------------------------------------
# Split leg: per-node diagnostics + shunt-vs-tanh arms (plan canary-core-split)
# ---------------------------------------------------------------------------

# Arm fills (spec core-decomp-legs). Boundary/output libraries are never
# touched by the arms: the drive path stays identical across arms.
SPLIT_ARM_SHUNT_FILL: float = -20.0  # g_resistive_raw -> G ~= 2e-9
SPLIT_ARM_TANH_FILL: float = -20.0  # gm_raw/isat_raw -> tanh current ~= 0


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors; NaN when either is constant."""
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    af = a.detach().float()
    bf = b.detach().float()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = torch.sqrt((af ** 2).sum() * (bf ** 2).sum())
    if not torch.isfinite(denom) or float(denom) == 0.0:
        return float("nan")
    return float(((af * bf).sum() / denom).item())


def _single_stage(net: nn.Module) -> nn.Module:
    """Return the single stage of a fabric net (split legs are 1-stage)."""
    stages = list(net.core.stages)
    if len(stages) != 1:
        raise NotImplementedError(
            "split legs currently support single-stage nets only"
        )
    return stages[0]


def apply_arm_fill(net: nn.Module, arm: str) -> dict[str, Any]:
    """Apply a split-leg arm fill to the core cell library (post-build).

    Arm ``"A"`` (shunt-only kill) fills core ``g_resistive_raw``; arm
    ``"B"`` (tanh-only kill) fills core ``gm_raw``/``isat_raw``. Boundary
    and output libraries are untouched. No-op parameters that a library
    lacks are skipped (defensive; the ``tanh_free`` core has all three).
    """
    if arm not in ("A", "B"):
        raise ValueError(f"split arm must be 'A' or 'B', got {arm!r}")
    stage = _single_stage(net)
    lib = stage.cell_lib
    log: dict[str, Any] = {"arm": arm}
    with torch.no_grad():
        if arm == "A":
            if hasattr(lib, "g_resistive_raw"):
                lib.g_resistive_raw.data.fill_(SPLIT_ARM_SHUNT_FILL)
                g_eff = torch.nn.functional.softplus(
                    torch.tensor(SPLIT_ARM_SHUNT_FILL)).item()
                log["g_resistive_fill"] = float(SPLIT_ARM_SHUNT_FILL)
                log["g_effective"] = float(g_eff)
        else:
            if hasattr(lib, "gm_raw"):
                lib.gm_raw.data.fill_(SPLIT_ARM_TANH_FILL)
                log["gm_fill"] = float(SPLIT_ARM_TANH_FILL)
            if hasattr(lib, "isat_raw"):
                lib.isat_raw.data.fill_(SPLIT_ARM_TANH_FILL)
                log["isat_fill"] = float(SPLIT_ARM_TANH_FILL)
    return log


def _forward_states(net: nn.Module, u_seq: torch.Tensor) -> torch.Tensor:
    """Collect full states ``(T, N)`` on a 1-D drive stream (eval-only)."""
    stage = _single_stage(net)
    state_width = (int(net.hid_count) + int(net.proj_count)
                   + int(net.output_ode_count))
    x0 = u_seq.new_zeros(1, state_width)
    with torch.no_grad():
        all_states = stage._forward_heun_sequence(
            x0=x0, t_span=float(net.core.stage_times[0]),
            num_steps=int(net.core.stage_steps[0]), u_seq=u_seq,
        )
    return all_states[:, 0, :].detach()


def node_diagnostics(
    net: nn.Module, states_full: torch.Tensor, u_seq: torch.Tensor, *,
    hidden_dim: int | None = None,
) -> dict[str, Any]:
    """Per-node decomposition of a collected trajectory (spec
    per-node-diagnostics).

    Returns per-hidden-node ``max_abs_ratio`` (the bimodality histogram),
    ``shunt_G_sum`` (incident shunt conductance), ``sign_balance``
    (incoming sign sum), ``boundary_current`` (per-node boundary OTA
    current at the zero-state/first-sample operating point), the dumped
    ``s_raw`` sign pattern, and Pearson correlations of the three
    structural vectors against ``max_abs_ratio``. Eval-only; weights are
    never mutated. All outputs are JSON-safe Python floats/lists.
    """
    stage = _single_stage(net)
    hid = int(net.hid_count) if hidden_dim is None else int(hidden_dim)
    if states_full.dim() != 2 or u_seq.dim() != 1:
        raise ValueError(
            "node_diagnostics requires (T, N) states and (T,) inputs, got "
            f"{tuple(states_full.shape)} and {tuple(u_seq.shape)}"
        )
    if states_full.shape[0] != u_seq.shape[0]:
        raise ValueError(
            "states and inputs must have the same length, got "
            f"{states_full.shape[0]} and {u_seq.shape[0]}"
        )
    hidden = states_full[:, :hid]
    x_max = float(stage.x_max)
    max_abs = hidden.abs().amax(dim=0).cpu()
    max_abs_ratio = (max_abs / x_max).tolist()

    src = stage.src.detach().cpu().long()
    dst = stage.dst.detach().cpu().long()
    n_edges = int(src.numel())
    mask = (src < hid) | (dst < hid)

    lib = stage.cell_lib
    if hasattr(lib, "g_resistive_raw"):
        g_all = torch.nn.functional.softplus(
            lib.g_resistive_raw.detach().cpu().float())
    else:
        g_all = torch.zeros(n_edges)
    shunt_sums = torch.zeros(hid)
    shunt_sums.index_add_(0, dst[mask & (dst < hid)],
                          g_all[mask & (dst < hid)])
    shunt_sums.index_add_(0, src[mask & (src < hid)],
                          g_all[mask & (src < hid)])
    shunt_list = shunt_sums.tolist()

    if hasattr(lib, "s_raw"):
        s_all = torch.sign(lib.s_raw.detach().cpu().float())
        s_dump = s_all.tolist()
    else:
        s_all = torch.zeros(n_edges)
        s_dump = [0.0] * n_edges
    sign_bal = torch.zeros(hid)
    sign_bal.index_add_(0, dst[mask & (dst < hid)],
                        s_all[mask & (dst < hid)])
    sign_list = sign_bal.tolist()

    n_nodes = int(stage.num_nodes)
    with torch.no_grad():
        u_first = u_seq[:1].reshape(1, 1).to(states_full.device)
        x0 = u_seq.new_zeros(1, n_nodes)
        acc_b = stage._compute_frozen_boundary(u_first, x0)
    if acc_b is None:
        bnd = [0.0] * hid
    else:
        bnd = acc_b[0, :hid].abs().detach().cpu().float().tolist()

    max_t = torch.tensor(max_abs_ratio)
    out: dict[str, Any] = {
        "hidden_dim": hid,
        "x_max": x_max,
        "max_abs_ratio": [float(v) for v in max_abs_ratio],
        "shunt_G_sum": [float(v) for v in shunt_list],
        "sign_balance": [float(v) for v in sign_list],
        "boundary_current": [float(v) for v in bnd],
        "s_raw_sign": [float(v) for v in s_dump],
        "corr_shunt_vs_max": _pearson(max_t, torch.tensor(shunt_list)),
        "corr_sign_vs_max": _pearson(max_t, torch.tensor(sign_list)),
        "corr_boundary_vs_max": _pearson(max_t, torch.tensor(bnd)),
    }
    return out


def _build_split_base_net(
    order: int, seed: int, t_span: float, num_steps: int, hidden_dim: int,
) -> nn.Module:
    """Rebuild the exact canary priority-corner net (base for all arms).

    Same factory + seed + override as the ``canary`` priority leg
    (``e0_sweep`` torus path with ``TRANSPLANT_CANARY_*``), so the base
    row must reproduce ``output/canary/canary.json`` priority numbers.
    """
    net, _, _ = ne._build_fabric_net(
        order=order, seed=seed, freeze_read=False,
        t_span=t_span, num_steps=num_steps,
        cell_library="tanh_free",
        hidden_dim=hidden_dim,
        core_refresh_interval=0,
        leak_constant=None,
        compile_sequence=False,
        boundary_fan_out=None,
    )
    apply_gain_override(
        net, gm_init=TRANSPLANT_CANARY_GM, isat_init=None,
        leak_mode=TRANSPLANT_CANARY_LEAK,
        raw_leak_init_seed=int(seed),
        per_lib_overrides=TRANSPLANT_CANARY_PER_LIB,
    )
    return net


def run_split_leg(
    order: int = 10, seed: int = 0, device: str = "cpu",
    t_span: float = 1.0, num_steps: int = 8, hidden_dim: int = 25,
    n_streams: int = 4, train_samples: int = 2500,
    jacobian_samples: int = 3,
) -> dict[str, Any]:
    """Run the split leg: base corner + arms A/B/C with node diagnostics.

    The base build reuses stream zero and the ``canary`` priority override,
    so ``base.score`` must match the recorded canary priority row. Each arm
    rebuilds the identical net (same seed), applies its fill, and re-scores
    the full E0 instrument set plus per-node diagnostics. Returns a
    JSON-safe dict with ``base``/``arm_A``/``arm_B``/``arm_C`` entries.
    """
    u_raw, y_raw = ne._gen_narma_train_streams(
        order, seed, n_streams, train_samples)
    drive = TRANSPLANT_CANARY_DRIVE
    u = ne._scale_drive(u_raw[0], bipolar=True, order=order,
                        input_scale=drive).to(device)
    y = y_raw[0].to(device)
    washout = PROBE_WASHOUT

    def _score(net: nn.Module, states: torch.Tensor) -> dict[str, float]:
        stage = _single_stage(net)
        return _score_state_trajectory(
            stage=stage, states_full=states,
            u_seq=u, y_seq=y,
            washout=washout, jacobian_samples=jacobian_samples,
            t_span=float(net.core.stage_times[0]),
            num_steps=int(net.core.stage_steps[0]),
        )

    result: dict[str, Any] = {
        "order": order, "seed": seed, "device": device,
        "t_span": t_span, "num_steps": num_steps,
        "hidden_dim": hidden_dim, "drive": drive,
        "leak": TRANSPLANT_CANARY_LEAK,
        "per_lib_overrides": TRANSPLANT_CANARY_PER_LIB,
    }
    base_net = _build_split_base_net(order, seed, t_span, num_steps,
                                     hidden_dim)
    base_net.to(device)
    base_states = _forward_states(base_net, u)
    result["base"] = {
        "score": _score(base_net, base_states),
        "node": node_diagnostics(base_net, base_states, u,
                                 hidden_dim=hidden_dim),
    }
    for arm in ("A", "B"):
        arm_net = _build_split_base_net(order, seed, t_span, num_steps,
                                        hidden_dim)
        arm_net.to(device)
        fill_log = apply_arm_fill(arm_net, arm)
        arm_states = _forward_states(arm_net, u)
        result[f"arm_{arm}"] = {
            "fill": fill_log,
            "score": _score(arm_net, arm_states),
            "node": node_diagnostics(arm_net, arm_states, u,
                                     hidden_dim=hidden_dim),
        }
    gz_states = _gate_zero_eval_forward(base_net, u)[:, 0, :].detach()
    result["arm_C"] = {
        "score": _score(base_net, gz_states),
        "node": node_diagnostics(base_net, gz_states, u,
                                 hidden_dim=hidden_dim),
    }
    return result


# ---------------------------------------------------------------------------
# Baselines calibration
# ---------------------------------------------------------------------------

def _classify_esn(
    esn_nrmse: float, halt_above: float, band: tuple[float, float],
) -> str:
    """Apply the pre-registered ESN calibration decision without running data."""
    if esn_nrmse >= halt_above:
        return "halt-task-setup"
    if band[0] <= esn_nrmse <= band[1]:
        return "fabric-guilty"
    return "band-unknown"


def run_baselines_calibration(
    order: int, seed: int, *, device: str = "cpu",
    halt_above: float = ESN_HALT_ABOVE, band: tuple[float, float] = NARMA10_BAND,
) -> dict[str, Any]:
    """Run ``run_baselines`` and apply the halt-or-continue decision.

    The supplied band is calibrated for NARMA-10; callers must not use this
    helper for another order unless they also supply that order's band.

    Returns:
        Dict with ``baselines`` (raw ``run_baselines`` output),
        ``esn_nrmse`` (the ESN figure), and ``decision`` in
        ``{"halt-task-setup", "fabric-guilty", "band-unknown"}``.
    """
    if order != 10 and band == NARMA10_BAND:
        raise ValueError(
            "the default calibration band is NARMA-10-specific; supply an "
            f"explicit band for order {order}"
        )
    bls = ne.run_baselines(order=order, seed=seed, device=device)
    esn_nrmse = float(bls["esn"]["nrmse"])
    decision = _classify_esn(esn_nrmse, halt_above, band)
    return {
        "order": order,
        "seed": seed,
        "device": device,
        "baselines": bls,
        "esn_nrmse": esn_nrmse,
        "decision": decision,
        "halt_above": halt_above,
        "band": list(band),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_probe_csv(path: Path, rows: list[dict[str, Any]],
                    fieldnames: list[str] | None = None) -> None:
    """Write probe rows to CSV with the given (or inferred) fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        # Stable order: keys of first row, then any new keys in order seen.
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_probe_txt(path: Path, header: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("./output/advisor_probes"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advisor-ordered zero-training probes for the NARMA reservoir.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- baselines calibration ---
    p_bl = sub.add_parser("baselines", help="Run in-harness baselines calibration.")
    _add_common_args(p_bl)

    # --- frozen-state probe ---
    p_fr = sub.add_parser(
        "frozen",
        help="Eval-only probe on a trained/native fabric checkpoint.",
    )
    _add_common_args(p_fr)
    p_fr.add_argument("--ckpt", type=Path, required=True,
                      help="Path to a fabric .pt checkpoint.")
    p_fr.add_argument("--cell-library", default="tanh_free")
    p_fr.add_argument("--core-refresh-interval", type=int, default=0)
    p_fr.add_argument("--freeze-read", action="store_true")
    p_fr.add_argument("--leak-constant", type=float, default=None)
    p_fr.add_argument("--input-scale", type=float, default=1.0)
    p_fr.add_argument("--n-streams", type=int, default=4)
    p_fr.add_argument("--train-samples", type=int, default=2500)
    p_fr.add_argument("--mlp-hidden", type=int, default=32)
    p_fr.add_argument("--mlp-epochs", type=int, default=300)
    p_fr.add_argument("--no-gate-zero", action="store_true")
    p_fr.add_argument("--t-span", type=float, default=1.0)
    p_fr.add_argument("--num-steps", type=int, default=8)

    # --- E0 sweep (caller can shrink the grid) ---
    p_e0 = sub.add_parser("e0", help="Run the zero-training 36-point gain/leak/drive sweep.")
    _add_common_args(p_e0)
    p_e0.add_argument("--n-streams", type=int, default=4)
    p_e0.add_argument("--train-samples", type=int, default=2500)
    p_e0.add_argument("--jacobian-samples", type=int, default=3)
    p_e0.add_argument("--t-span", type=float, default=1.0)
    p_e0.add_argument("--num-steps", type=int, default=8)
    p_e0.add_argument("--hidden-dim", type=int, default=25)
    p_e0.add_argument("--gain-grid", type=str, default="-5,-2,0,1.5",
                      help="Comma-separated gm_init values (default -5,-2,0,1.5). "
                           "Use the --flag=value form when the value starts "
                           "with '-' (argparse rejects a bare space-separated "
                           "negative).")
    p_e0.add_argument(
        "--leak-grid", type=str, default="slow-fixed,randomized,0.15",
        help="Comma-separated leak modes; numeric -> fixed scalar leak. "
             "Step 4 hetero-leak token: 'hetero:1.0:40.0' (per-node "
             "log-uniform time constants; colons, never commas, so the "
             "token survives grid splitting). Regime-transplant raw token: "
             "'raw:-0.5' (programmable leak, raw_leak filled to scalar; "
             "effective leak softplus(-0.5)~0.47 -- NOT the same as bare "
             "'-0.5', which sets a negative leak_constant).",
    )
    p_e0.add_argument("--drive-grid", type=str, default="0.25,0.5,1.0",
                      help="Comma-separated input scales. Regime-transplant "
                           "runs below the historical 0.25 floor "
                           "(e.g. '0.1,0.2,0.3').")
    p_e0.add_argument(
        "--isat-grid", type=str, default=None,
        help="Regime-transplant: comma-separated isat_init values, one per "
             "gain-grid entry (length must match --gain-grid). Use 'nan' "
             "or leave blank for a position to mean 'coupled (==gm_init)'. "
             "Example: '--gain-grid -3.5,-3.5 --isat-grid -2,-6' decouples "
             "Isat into both directions at moderate core gm.",
    )
    p_e0.add_argument(
        "--per-lib-override", type=str, default=None,
        help="Regime-transplant: per-cell-library override JSON, e.g. "
             "'{\"boundary_cell_lib\":{\"gm\":-3.5,\"isat\":-2}}'. Libraries "
             "not listed fall back to --gain-grid values. Mutually "
             "compatible with --isat-grid (per-lib wins for listed libs).",
    )
    p_e0.add_argument(
        "--anchor-ckpt", type=str, default=None,
        help="Regime-transplant: exact-tensor anchor corner. Path to a "
             "fabric .pt checkpoint whose dynamics tensors (gm_raw, "
             "isat_raw, raw_leak) are copied into every freshly built "
             "net BEFORE the gm/Isat/leak override pass is suppressed. "
             "Used to calibrate the E0 instrument against a known-good "
             "trained artifact. Mutually exclusive with --isat-grid and "
             "--per-lib-override (the override pass is suppressed).",
    )
    p_e0.add_argument("--refresh", type=int, default=0,
                      help="core_refresh_interval for every built stage "
                           "(default 0 = legacy frozen). The exact-tensor "
                           "anchor must use the checkpoint's own k "
                           "(k2 ckpt -> --refresh 2).")
    p_e0.add_argument("--max-corners", type=int, default=0,
                      help="If >0, run exactly the first N corners in "
                           "drive/leak/gain traversal order (smoke).")
    p_e0.add_argument("--use-small-world", action="store_true",
                      help="Step 4 grid axis: build the NARMA net with "
                           "hidden_family='small_world' instead of the "
                           "canonical torus (same node count).")
    p_e0.add_argument("--small-world-k", type=int, default=4)
    p_e0.add_argument("--small-world-p", type=float, default=0.2)
    p_e0.add_argument("--small-world-seed", type=int, default=1)
    p_e0.add_argument(
        "--boundary-fan-out", type=str, default=None,
        dest="boundary_fan_out",
        help="Round-2 §13.1 sparse-drive E0: JSON dict overriding the "
             "canonical full boundary fan-out {0: range(hidden_dim)}. "
             "Targets must be unique, in [0, hidden_dim), and the input "
             "key must be in [0, in_dim). 'None' (default) preserves the "
             "canonical full fan-out exactly. Sparse runs get a "
             "_fan<tgt>... config-tag suffix and a separate resume key, "
             "so the canonical progress file is untouched.",
    )
    # --- regime-transplant canary: anchor + advisor priority corner ---
    p_can = sub.add_parser(
        "canary",
        help="Regime-transplant canary leg: exact-tensor anchor corner + "
             "advisor priority corner, then the pre-registered 4-outcome "
             "verdict. CPU-runnable (2 corners).",
    )
    _add_common_args(p_can)
    p_can.add_argument("--anchor-ckpt", type=str, required=True,
                       help="Path to the k2 epoch-54 fabric .pt checkpoint "
                            "whose dynamics tensors calibrate the E0 "
                            "instrument (expect ridge 0.60-0.68).")
    p_can.add_argument("--anchor-refresh", type=int, default=2,
                       help="core_refresh_interval for the anchor build "
                            "ONLY (must match the ckpt's own k; default 2).")
    p_can.add_argument("--n-streams", type=int, default=4)
    p_can.add_argument("--train-samples", type=int, default=2500)
    p_can.add_argument("--jacobian-samples", type=int, default=3)
    p_can.add_argument("--t-span", type=float, default=1.0)
    p_can.add_argument("--num-steps", type=int, default=8)
    p_can.add_argument("--hidden-dim", type=int, default=25)

    # --- regime-transplant full grid (~54 corners, GPU-intended) ---
    p_tr = sub.add_parser(
        "transplant",
        help="Regime-transplant full grid: 2 gm x 3 Isat x 3 raw-leak x "
             "3 swing = 54 corners (spec regime-transplant-grid). "
             "GPU-intended; per-corner progress flush + resume included.",
    )
    _add_common_args(p_tr)
    p_tr.add_argument("--n-streams", type=int, default=4)
    p_tr.add_argument("--train-samples", type=int, default=2500)
    p_tr.add_argument("--jacobian-samples", type=int, default=3)
    p_tr.add_argument("--t-span", type=float, default=1.0)
    p_tr.add_argument("--num-steps", type=int, default=8)
    p_tr.add_argument("--hidden-dim", type=int, default=25)
    p_tr.add_argument("--refresh", type=int, default=0)
    p_tr.add_argument("--max-corners", type=int, default=0,
                      help="If >0, run exactly the first N corners in "
                           "drive/leak/gain traversal order (smoke).")

    # --- split leg: per-node diagnostics + shunt-vs-tanh arms ---
    p_sp = sub.add_parser(
        "split",
        help="Split leg (plan canary-core-split): rebuild the canary "
             "priority corner, run per-node diagnostics, then arms A/B/C "
             "with the full E0 instrument set. GPU-intended; writes "
             "split.json.",
    )
    _add_common_args(p_sp)
    p_sp.add_argument("--n-streams", type=int, default=4)
    p_sp.add_argument("--train-samples", type=int, default=2500)
    p_sp.add_argument("--jacobian-samples", type=int, default=3)
    p_sp.add_argument("--t-span", type=float, default=1.0)
    p_sp.add_argument("--num-steps", type=int, default=8)
    p_sp.add_argument("--hidden-dim", type=int, default=25)

    args = parser.parse_args(argv)
    # Pre-registered thresholds and locked grids are NARMA-10-specific.
    # Library functions remain order-agnostic for future calibration work.
    if args.order != 10:
        parser.error("advisor-probe CLI decisions are pre-registered for --order 10 only")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "baselines":
        res = run_baselines_calibration(args.order, args.seed, device=args.device)
        esn = res["esn_nrmse"]
        print(f"  baselines seed={args.seed}: ESN NRMSE={esn:.4f}")
        for name, v in res["baselines"].items():
            print(f"    {name:>12}  NRMSE={v['nrmse']:.4f}  R^2={v['r2']:.4f}  "
                  f"params={v.get('n_params', '')}")
        print(f"  decision: {res['decision']} (band {res['band']}, halt>={res['halt_above']})")
        (args.output / "baselines.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "esn_nrmse": esn,
            "decision": res["decision"],
            "thresholds": {
                "band": res["band"],
                "halt_above": res["halt_above"],
            },
            "baselines": {
                k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
                    for kk, vv in v.items()}
                for k, v in res["baselines"].items()
            },
        }, indent=2))
        return 0

    if args.mode == "frozen":
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        net, t_span_used, num_steps_used = ne._build_fabric_net(
            order=args.order, seed=args.seed, freeze_read=args.freeze_read,
            t_span=args.t_span, num_steps=args.num_steps,
            cell_library=args.cell_library,
            core_refresh_interval=args.core_refresh_interval,
            leak_constant=args.leak_constant,
            compile_sequence=False,
        )
        net.load_state_dict(ckpt["model_state"], strict=True)
        net.to(args.device)
        u_train_raw, _ = ne._gen_narma_train_streams(
            args.order, args.seed, args.n_streams, args.train_samples,
        )
        u_train = ne._scale_drive(
            u_train_raw[0], bipolar=True, order=args.order,
            input_scale=args.input_scale,
        )
        u_train, y_train = u_train.to(args.device), _[0].to(args.device)
        config_tag = (
            f"ckpt{Path(args.ckpt).name}_order{args.order}_seed{args.seed}_"
            f"{args.cell_library}_k{args.core_refresh_interval}_"
            f"freeze{int(args.freeze_read)}_leak{args.leak_constant}_"
            f"drive{args.input_scale:g}_washout{PROBE_WASHOUT}_"
            f"mlp{args.mlp_hidden}x{args.mlp_epochs}"
        )
        row = frozen_state_probe(
            net, u_train, y_train,
            mlp_hidden=args.mlp_hidden, mlp_epochs=args.mlp_epochs,
            mlp_seed=args.seed, device=args.device,
            also_eval_gate_zero=not args.no_gate_zero,
            config_tag=config_tag,
        )
        row_dict = asdict(row)
        print(f"  FROZEN_STATE probe on {args.ckpt}:")
        for k, v in row_dict.items():
            print(f"    {k}={v}")
        # Decision.
        if row.readout_fix_promoted:
            decision = "READOUT-FIX-LEADS"
        elif row.init_regime_sole_primary:
            decision = "INIT-REGIME-SOLE-PRIMARY"
        else:
            decision = "AMBIGUOUS"
        print(f"  DECISION: {decision} "
              f"(thresholds: promote<{RIDGE_PROMOTE_READOUT_BELOW}, "
              f"init>={RIDGE_INIT_REGIME_AT_OR_ABOVE})")
        out = {
            "ckpt": str(args.ckpt),
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "cell_library": args.cell_library,
            "core_refresh_interval": args.core_refresh_interval,
            "freeze_read": args.freeze_read,
            "leak_constant": args.leak_constant,
            "input_scale": args.input_scale,
            "n_streams": args.n_streams,
            "train_samples": args.train_samples,
            "washout": PROBE_WASHOUT,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
            "gate_zero": not args.no_gate_zero,
            "row": row_dict,
            "decision": decision,
            "thresholds": {
                "promote_readout_below": RIDGE_PROMOTE_READOUT_BELOW,
                "init_regime_at_or_above": RIDGE_INIT_REGIME_AT_OR_ABOVE,
            },
        }
        (args.output / "frozen_state.json").write_text(json.dumps(out, indent=2))
        write_probe_csv(args.output / "frozen_state.csv", [row_dict])
        return 0

    if args.mode == "e0":
        gain_grid = tuple(float(s) for s in args.gain_grid.split(",") if s.strip())
        leak_grid: list[Any] = []
        for tok in args.leak_grid.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                leak_grid.append(float(tok))
            except ValueError:
                leak_grid.append(tok)
        drive_grid = tuple(float(s) for s in args.drive_grid.split(",") if s.strip())
        # Regime-transplant: parse --isat-grid (length must match gain_grid;
        # 'nan' / empty / 'none' tokens mean "coupled (==gm_init)").
        isat_grid: list[float | None] | None = None
        if args.isat_grid is not None:
            isat_grid = []
            for tok in args.isat_grid.split(","):
                tok = tok.strip()
                if not tok or tok.lower() in ("nan", "none", "coupled"):
                    isat_grid.append(None)
                else:
                    isat_grid.append(float(tok))
            if len(isat_grid) != len(gain_grid):
                raise SystemExit(
                    f"--isat-grid length ({len(isat_grid)}) must match "
                    f"--gain-grid length ({len(gain_grid)})"
                )
        # Regime-transplant: parse --per-lib-override (JSON dict).
        per_lib_parsed: dict[str, dict[str, float | None]] | None = None
        if args.per_lib_override is not None:
            try:
                pl_raw = json.loads(args.per_lib_override)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--per-lib-override: invalid JSON: {e}")
            if not isinstance(pl_raw, dict):
                raise SystemExit(
                    "--per-lib-override must be a JSON object, got "
                    f"{type(pl_raw).__name__}"
                )
            valid_libs = {"cell_lib", "boundary_cell_lib", "output_ode_cell_lib"}
            per_lib_parsed = {}
            for lib_name, ov in pl_raw.items():
                if lib_name not in valid_libs:
                    raise SystemExit(
                        f"--per-lib-override: unknown library {lib_name!r}; "
                        f"valid: {sorted(valid_libs)}"
                    )
                if not isinstance(ov, dict):
                    raise SystemExit(
                        f"--per-lib-override[{lib_name!r}] must be a dict, "
                        f"got {type(ov).__name__}"
                    )
                entry: dict[str, float | None] = {}
                for k in ("gm", "isat"):
                    if k in ov:
                        v = ov[k]
                        entry[k] = None if v is None else float(v)
                per_lib_parsed[lib_name] = entry
        # Regime-transplant: the anchor suppresses the override pass, so
        # combining it with override flags is almost certainly a mistake.
        if args.anchor_ckpt is not None and (
            args.isat_grid is not None or args.per_lib_override is not None
        ):
            raise SystemExit(
                "--anchor-ckpt is mutually exclusive with --isat-grid and "
                "--per-lib-override (the anchor suppresses the gm/Isat/leak "
                "override pass; exact tensors come from the checkpoint)."
            )
        selected_corners = _select_corner_subset(
            drive_grid, leak_grid, gain_grid,
            args.max_corners if args.max_corners > 0 else None,
            isat_grid=isat_grid,
        )
        # Round-2 §13.1 sparse-drive E0: parse the JSON boundary-fan-out
        # spec. ``None`` keeps the canonical full fan-out exactly (zero
        # behavior change vs the legacy 48-corner legs).
        bfo_parsed: dict[int, list[int]] | None = None
        if args.boundary_fan_out is not None:
            try:
                bfo_raw = json.loads(args.boundary_fan_out)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--boundary-fan-out: invalid JSON: {e}")
            if not isinstance(bfo_raw, dict):
                raise SystemExit(
                    "--boundary-fan-out must be a JSON object, got "
                    f"{type(bfo_raw).__name__}"
                )
            bfo_parsed = {}
            for k, v in bfo_raw.items():
                try:
                    ik = int(k)
                except (TypeError, ValueError):
                    raise SystemExit(
                        f"--boundary-fan-out: input keys must be ints, "
                        f"got {k!r}"
                    )
                if ik < 0:
                    raise SystemExit(
                        f"--boundary-fan-out: input keys must be "
                        f"non-negative, got {k!r}"
                    )
                if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
                    raise SystemExit(
                        f"--boundary-fan-out: target list for input {ik} "
                        f"must be int[], got {v!r}"
                    )
                if any(x < 0 for x in v):
                    raise SystemExit(
                        f"--boundary-fan-out: targets must be non-negative, "
                        f"got {v!r}"
                    )
                bfo_parsed[ik] = [int(x) for x in v]
            # Quick out-of-range guard before the sweep starts. Full
            # validation (uniqueness, in_dim coverage) lives in
            # build_net_from_config; this catches the cheap failures
            # up-front so the loop isn't wasted on a corner that won't
            # build.
            for k, v in bfo_parsed.items():
                if any(t < 0 or t >= args.hidden_dim for t in v):
                    raise SystemExit(
                        f"--boundary-fan-out: target {v} for input {k} "
                        f"out of range [0, {args.hidden_dim})"
                    )
        t0 = time.time()
        rows = e0_sweep(
            order=args.order, seed=args.seed, device=args.device,
            gain_grid=gain_grid, leak_grid=leak_grid, drive_grid=drive_grid,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            selected_corners=selected_corners,
            isat_grid=isat_grid,
            per_lib_overrides=per_lib_parsed,
            pre_fill=_make_anchor_pre_fill(args.anchor_ckpt, args.device)
                if args.anchor_ckpt else None,
            pre_fill_overrides_after=False,
            refresh=args.refresh,
            use_small_world=args.use_small_world,
            small_world_k=args.small_world_k,
            small_world_p=args.small_world_p,
            small_world_seed=args.small_world_seed,
            progress_json=args.output / "e0_progress.json",
            boundary_fan_out=bfo_parsed,
        )
        elapsed = time.time() - t0
        row_dicts = [asdict(r) for r in rows]
        # CSV
        write_probe_csv(args.output / "e0_sweep.csv", row_dicts)
        # Plain-text summary.
        hdr = (f"E0 sweep -- order={args.order} seed={args.seed} "
               f"{len(rows)} corners in {elapsed:.1f}s")
        if bfo_parsed is not None:
            hdr += f"  (sparse fan-out: {bfo_parsed})"
        lines = [
            f"{'gm':>6} {'leak':>28} {'drive':>5} {'ridge':>7} "
            f"{'mc':>6} {'sPR':>6} {'gzR':>7} {'gzMC':>6} "
            f"{'gzPR':>6} {'rail%':>6} {'pass':>4} {'gzpass':>6}",
        ]
        for r in rows:
            lines.append(
                f"{r.gm_init:>6.2f} {r.leak_mode:>28} {r.drive:>5.2f} "
                f"{r.ridge_nrmse:>7.4f} {r.mc_total_washout_corrected:>6.2f} "
                f"{r.state_pr:>6.2f} {r.gate_zero_ridge_nrmse:>7.4f} "
                f"{r.gate_zero_mc_total:>6.2f} {r.gate_zero_state_pr:>6.2f} "
                f"{100.0 * r.rail_frac:>5.1f}% "
                f"{'PASS' if r.pass_all else 'FAIL':>4} "
                f"{'PASS' if r.gate_zero_pass_all else 'FAIL':>6}"
            )
        write_probe_txt(args.output / "e0_sweep.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        # Pre-registered sweep decision.
        passing = [r for r in rows if r.pass_all]
        rail_pass = [r for r in rows if r.pass_rail]
        print()
        if passing:
            best = min(passing, key=lambda r: r.ridge_nrmse)
            print(
                f"  DECISION: INIT-SEARCH-WARRANTED. {len(passing)} corner(s) "
                f"with MC>{E0_PASS_MC_ABOVE}, ridge<={E0_PASS_RIDGE_BELOW}, "
                f"state-PR>={E0_PASS_STATE_PR_MIN}, and rail pass; "
                f"best ridge={best.ridge_nrmse:.4f} at "
                f"gm={best.gm_init}, leak={best.leak_mode}, drive={best.drive}. "
                f"Next: init search + 1 confirmation train."
            )
        elif rail_pass and not passing:
            print(
                f"  DECISION: NO-CORNER-BREAKS-RIDGE. {len(rail_pass)} corner(s) "
                f"pass rail but none reach MC>{E0_PASS_MC_ABOVE}, "
                f"ridge<={E0_PASS_RIDGE_BELOW}, and "
                f"state-PR>={E0_PASS_STATE_PR_MIN}. Next: structural talk "
                f"(systolic delay-line candidate; depth-per-sample stays paused)."
            )
        else:
            print(
                f"  DECISION: ALL-CORNERS-RAIL-DISQUALIFIED. Init regime "
                f"shape does not break the failure mode; treat as "
                f"structural suppression."
            )
        # Also dump a structured summary.
        (args.output / "e0_sweep.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "canonical_net": {
                "cell_library": "tanh_free",
                "freeze_read": False,
                "core_refresh_interval": args.refresh,
                "t_span": args.t_span,
                "num_steps": args.num_steps,
                "hidden_dim": args.hidden_dim,
                "use_small_world": args.use_small_world,
                "small_world_k": args.small_world_k,
                "small_world_p": args.small_world_p,
                "small_world_seed": args.small_world_seed,
                # Round-2 §13.1: record the boundary-fan-out override used
                # for this sweep so JSON consumers can join against the
                # canonical full-fan-out legs without re-reading the CLI.
                "boundary_fan_out": bfo_parsed,
            },
            "n_streams": args.n_streams,
            "train_samples": args.train_samples,
            "washout": PROBE_WASHOUT,
            "jacobian_samples": args.jacobian_samples,
            "grids": {
                "gain": list(gain_grid),
                "leak": [str(value) for value in leak_grid],
                "drive": list(drive_grid),
                # Regime-transplant: only present when caller actually
                # decoupled. Keeps the JSON schema clean for legacy legs.
                **({"isat": [v for v in (isat_grid or [])]}
                   if isat_grid is not None else {}),
                **({"per_lib_overrides": per_lib_parsed}
                   if per_lib_parsed is not None else {}),
            },
            "corner_selection": (
                f"first {args.max_corners}" if args.max_corners > 0 else "full grid"
            ),
            "n_corners": len(rows),
            "complete": True,
            "elapsed_s": elapsed,
            "thresholds": {
                "mc_above": E0_PASS_MC_ABOVE,
                "ridge_below_or_equal": E0_PASS_RIDGE_BELOW,
                "state_pr_min": E0_PASS_STATE_PR_MIN,
                "rail_disqualify_above": RAIL_DISQUALIFY_ABOVE,
            },
            "rows": row_dicts,
        }, indent=2))
        return 0

    if args.mode == "canary":
        # Canary leg (plan narma-regime-transplant): exact-tensor anchor
        # corner (instrument calibration) + advisor priority corner
        # (high core Isat, moderate core gm, fast raw leak, swing 0.2),
        # then the pre-registered 4-outcome verdict. Two corners,
        # CPU-runnable.
        t0 = time.time()
        anchor_rows = e0_sweep(
            order=args.order, seed=args.seed, device=args.device,
            gain_grid=(0.0,), leak_grid=("slow-fixed",), drive_grid=(1.0,),
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            pre_fill=_make_anchor_pre_fill(args.anchor_ckpt, args.device),
            pre_fill_overrides_after=False,
            refresh=args.anchor_refresh,
            progress_json=args.output / "canary_anchor_progress.json",
        )
        canary_rows = e0_sweep(
            order=args.order, seed=args.seed, device=args.device,
            gain_grid=(TRANSPLANT_CANARY_GM,),
            leak_grid=(TRANSPLANT_CANARY_LEAK,),
            drive_grid=(TRANSPLANT_CANARY_DRIVE,),
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            per_lib_overrides=TRANSPLANT_CANARY_PER_LIB,
            refresh=0,
            progress_json=args.output / "canary_priority_progress.json",
        )
        elapsed = time.time() - t0
        anchor, canary = anchor_rows[0], canary_rows[0]
        verdict = canary_verdict(anchor.ridge_nrmse, canary)
        print(f"  ANCHOR (exact-tensor k2 ckpt, refresh={args.anchor_refresh}):")
        print(f"    ridge={anchor.ridge_nrmse:.4f} R^2={anchor.ridge_r2:.4f} "
              f"MC={anchor.mc_total_washout_corrected:.2f} PR={anchor.state_pr:.2f} "
              f"rail={100.0 * anchor.rail_frac:.1f}% "
              f"(expect ridge 0.60-0.68, MC ~2.6)")
        print(f"  PRIORITY (core gm={TRANSPLANT_CANARY_GM}, core "
              f"isat=-2.0, leak={TRANSPLANT_CANARY_LEAK}, "
              f"drive={TRANSPLANT_CANARY_DRIVE}):")
        print(f"    ridge={canary.ridge_nrmse:.4f} R^2={canary.ridge_r2:.4f} "
              f"MC={canary.mc_total_washout_corrected:.2f} PR={canary.state_pr:.2f} "
              f"rail={100.0 * canary.rail_frac:.1f}% "
              f"gz_ridge={canary.gate_zero_ridge_nrmse:.4f} "
              f"gz_MC={canary.gate_zero_mc_total:.2f}")
        print(f"  VERDICT: {verdict['outcome']}")
        print(f"  {verdict['summary']}")
        (args.output / "canary.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "anchor_ckpt": args.anchor_ckpt,
            "anchor_refresh": args.anchor_refresh,
            "anchor": asdict(anchor),
            "priority": asdict(canary),
            "priority_spec": {
                "gm": TRANSPLANT_CANARY_GM,
                "leak": TRANSPLANT_CANARY_LEAK,
                "drive": TRANSPLANT_CANARY_DRIVE,
                "per_lib_overrides": TRANSPLANT_CANARY_PER_LIB,
            },
            "verdict": verdict,
            "elapsed_s": elapsed,
        }, indent=2))
        write_probe_csv(
            args.output / "canary.csv", [asdict(anchor), asdict(canary)],
        )
        return 0

    if args.mode == "transplant":
        # Full transplant grid (spec regime-transplant-grid): 2 gm x 3 Isat
        # x 3 raw-leak x 3 swing = 54 corners. GPU-intended; per-corner
        # progress flush + resume via transplant_progress.json (kept
        # separate from the e0 progress file so legacy legs never match).
        assert len(TRANSPLANT_GM_GRID) == len(TRANSPLANT_ISAT_GRID)
        t0 = time.time()
        rows = e0_sweep(
            order=args.order, seed=args.seed, device=args.device,
            gain_grid=TRANSPLANT_GM_GRID, leak_grid=TRANSPLANT_LEAK_GRID,
            drive_grid=TRANSPLANT_DRIVE_GRID,
            n_streams=args.n_streams,
            train_samples_per_stream=args.train_samples,
            jacobian_samples=args.jacobian_samples,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            selected_corners=_select_corner_subset(
                TRANSPLANT_DRIVE_GRID, TRANSPLANT_LEAK_GRID,
                TRANSPLANT_GM_GRID,
                args.max_corners if args.max_corners > 0 else None,
                isat_grid=TRANSPLANT_ISAT_GRID,
            ),
            isat_grid=TRANSPLANT_ISAT_GRID,
            refresh=args.refresh,
            progress_json=args.output / "transplant_progress.json",
        )
        elapsed = time.time() - t0
        row_dicts = [asdict(r) for r in rows]
        write_probe_csv(args.output / "transplant_sweep.csv", row_dicts)
        hdr = (f"Transplant sweep -- order={args.order} seed={args.seed} "
               f"{len(rows)} corners in {elapsed:.1f}s")
        lines = [
            f"{'gm':>6} {'isat':>6} {'leak':>10} {'drive':>5} {'ridge':>7} "
            f"{'mc':>6} {'sPR':>6} {'gzR':>7} {'gzMC':>6} "
            f"{'rail%':>6} {'pass':>4}",
        ]
        for r in rows:
            _isat = (f"{r.isat_init:.1f}" if r.isat_init == r.isat_init
                     else "coup")
            lines.append(
                f"{r.gm_init:>6.2f} {_isat:>6} {r.leak_mode:>10} "
                f"{r.drive:>5.2f} {r.ridge_nrmse:>7.4f} "
                f"{r.mc_total_washout_corrected:>6.2f} {r.state_pr:>6.2f} "
                f"{r.gate_zero_ridge_nrmse:>7.4f} "
                f"{r.gate_zero_mc_total:>6.2f} "
                f"{100.0 * r.rail_frac:>5.1f}% "
                f"{'PASS' if r.pass_all else 'FAIL':>4}"
            )
        write_probe_txt(args.output / "transplant_sweep.txt", hdr, lines)
        print("\n".join([hdr] + lines))
        passing = [r for r in rows if r.pass_all]
        print()
        if passing:
            best = min(passing, key=lambda r: r.ridge_nrmse)
            print(
                f"  DECISION: TRANSPLANT-LIVES. {len(passing)} corner(s) "
                f"pass; best ridge={best.ridge_nrmse:.4f} at gm={best.gm_init}, "
                f"isat={best.isat_init}, leak={best.leak_mode}, "
                f"drive={best.drive}. Next: short k=2 train from the "
                f"winning init."
            )
        else:
            print(
                f"  DECISION: TRANSPLANT-DEAD. No corner passes "
                f"(MC>{E0_PASS_MC_ABOVE}, ridge<={E0_PASS_RIDGE_BELOW}, "
                f"state-PR>={E0_PASS_STATE_PR_MIN}, rail<=5%). Next: "
                f"re-examine the ESN->fabric translation."
            )
        (args.output / "transplant_sweep.json").write_text(json.dumps({
            "order": args.order,
            "seed": args.seed,
            "device": args.device,
            "grids": {
                "gain": list(TRANSPLANT_GM_GRID),
                "isat": list(TRANSPLANT_ISAT_GRID),
                "leak": list(TRANSPLANT_LEAK_GRID),
                "drive": list(TRANSPLANT_DRIVE_GRID),
            },
            "corner_selection": (
                f"first {args.max_corners}" if args.max_corners > 0 else "full grid"
            ),
            "n_corners": len(rows),
            "complete": True,
            "elapsed_s": elapsed,
            "thresholds": {
                "mc_above": E0_PASS_MC_ABOVE,
                "ridge_below_or_equal": E0_PASS_RIDGE_BELOW,
                "state_pr_min": E0_PASS_STATE_PR_MIN,
                "rail_disqualify_above": RAIL_DISQUALIFY_ABOVE,
            },
            "rows": row_dicts,
        }, indent=2))
        return 0

    if args.mode == "split":
        # Split leg (plan canary-core-split): base corner + arms A/B/C.
        t0 = time.time()
        split = run_split_leg(
            order=args.order, seed=args.seed, device=args.device,
            t_span=args.t_span, num_steps=args.num_steps,
            hidden_dim=args.hidden_dim,
            n_streams=args.n_streams,
            train_samples=args.train_samples,
            jacobian_samples=args.jacobian_samples,
        )
        split["elapsed_s"] = time.time() - t0
        b = split["base"]["score"]
        print(f"  BASE (priority corner rebuild):")
        print(f"    ridge={b['ridge_nrmse']:.4f} MC={b['mc_total']:.2f} "
              f"PR={b['state_pr']:.2f} rail={100.0 * b['rail_frac']:.1f}% "
              f"(expect 0.7611 / 1.68 / 15.2%)")
        bn = split["base"]["node"]
        print(f"    corr shunt/max={bn['corr_shunt_vs_max']:.3f} "
              f"sign/max={bn['corr_sign_vs_max']:.3f} "
              f"bnd/max={bn['corr_boundary_vs_max']:.3f}")
        for arm in ("A", "B", "C"):
            s = split[f"arm_{arm}"]["score"]
            n = split[f"arm_{arm}"]["node"]
            hot = sum(1 for v in n["max_abs_ratio"] if v > 0.9)
            print(f"  ARM {arm}: ridge={s['ridge_nrmse']:.4f} "
                  f"MC={s['mc_total']:.2f} PR={s['state_pr']:.2f} "
                  f"rail={100.0 * s['rail_frac']:.1f}% hot_nodes={hot}")
        (args.output / "split.json").write_text(json.dumps(split, indent=2))
        return 0

    parser.error(f"unknown mode {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
