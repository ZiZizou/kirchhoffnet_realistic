"""Hoisted, zero-graph-break Heun ODE engine for DifferentialStage.

The CTLE KirchhoffNet has small node/edge dimensions (18 nodes, ~52 OTAs per
stage, 7 stages) and runs 10-40 Heun steps (20-80 RHS evaluations) per stage
per batch. Under eager PyTorch this is Python-dispatch and kernel-launch bound.

This module factors out all stage-invariant calculations (parameter transforms,
cap-divider / VCA / GLN modulations, frozen-leg KCL scatters, core resistive
Laplacian + leak matrix, and Vref rail compliance) once per stage entry, and
runs the inner Heun loop through pure-tensor, scatter-free step functions that
compile cleanly with ``torch.compile(..., fullgraph=True)`` on supported CUDA
devices (sm_70+) with automatic eager fallback everywhere else.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hw.constraints import (
    COMPLIANCE_K,
    LAMBDA_V,
    THETA_MAX,
    IsatScale,
    ab_mode,
    theta_bound,
)

__all__ = [
    "configure_compiled_heun",
    "compiled_heun_enabled",
    "disable_compiled_heun",
    "fast_heun_forward",
]

_COMPILE_STATE: dict = {
    "mode": "off",
    "enabled": False,
    "failed": False,
    "standard": None,
    "compliance": None,
    "logger": None,
}


def _tanh_ste_inline(u: torch.Tensor, alpha_bwd: float) -> torch.Tensor:
    t = torch.tanh(u)
    if alpha_bwd != 0.0:
        t = t + alpha_bwd * (u - u.detach())
    return t


def _leg_factors_fast(
    x_n: torch.Tensor,
    lam: float,
    scale: float,
    x_hard: float,
    v_off: float,
    k: float,
    c0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_top = (1.0 - lam * x_n).clamp_min(0.0)
    f_bot = (1.0 + lam * x_n).clamp_min(0.0)
    y_top = (x_hard - x_n) * scale - v_off
    y_bot = (x_hard + x_n) * scale - v_off
    f_top = f_top * (1.0 - torch.exp(-F.softplus(y_top, beta=k))) / c0
    f_bot = f_bot * (1.0 - torch.exp(-F.softplus(y_bot, beta=k))) / c0
    return f_top, f_bot


def _heun_step_standard(
    x: torch.Tensor,
    dt: float,
    i_edge_const: torch.Tensor,
    M_lin: torch.Tensor,
    b_dst: torch.Tensor,
    b_slope_x: torch.Tensor,
    b_bias: torch.Tensor,
    b_isat_g: torch.Tensor,
    b_gate_src: torch.Tensor,
    b_G_g: torch.Tensor,
    b_res_const: torch.Tensor,
    Inc_b: torch.Tensor,
    s_src: torch.Tensor,
    s_slope_x: torch.Tensor,
    s_bias: torch.Tensor,
    s_isat_g: torch.Tensor,
    s_gate_dst: torch.Tensor,
    s_G_g: torch.Tensor,
    s_vref: torch.Tensor,
    W_T: torch.Tensor,
    x_max: float,
    clip_current: float,
    clip_s: torch.Tensor,
    c_div: torch.Tensor,
    dst_start: int,
    n_tail: int,
    b_alpha: float,
    b_alpha_bwd: float,
    b_use_ste: bool,
    b_beta: float,
    s_alpha: float,
    s_alpha_bwd: float,
    s_use_ste: bool,
    s_beta: float,
) -> torch.Tensor:
    def rhs(state: torch.Tensor) -> torch.Tensor:
        acc = i_edge_const + (state @ M_lin)

        # Boundary-terminal OTAs
        x_dst_b = state.index_select(1, b_dst)
        u_b = b_bias + b_slope_x * x_dst_b
        if b_use_ste:
            t_b = _tanh_ste_inline(u_b, b_alpha_bwd)
        else:
            t_b = torch.tanh(u_b) + b_alpha * u_b
        gate_dst_b = torch.sigmoid((x_max - x_dst_b.abs()) / b_beta)
        i_b = (b_gate_src * gate_dst_b) * (b_isat_g * t_b) + (b_res_const - b_G_g * x_dst_b)
        acc = acc + (i_b @ Inc_b)

        # Shared-sense + crossbar accumulator readout
        x_j = state.index_select(1, s_src)
        u_s = s_bias + s_slope_x * x_j
        if s_use_ste:
            t_s = _tanh_ste_inline(u_s, s_alpha_bwd)
        else:
            t_s = torch.tanh(u_s) + s_alpha * u_s
        gate_src_s = torch.sigmoid((x_max - x_j.abs()) / s_beta)
        i_s = (gate_src_s * s_gate_dst) * (s_isat_g * t_s) + s_G_g * (x_j - s_vref)
        i_acc = i_s @ W_T
        acc_pre, acc_tail = torch.split(acc, [dst_start, n_tail], dim=1)
        acc = torch.cat([acc_pre, acc_tail + i_acc], dim=1)

        clip = clip_current * (
            torch.sigmoid((state - x_max) / clip_s)
            - torch.sigmoid((-state - x_max) / clip_s)
        )
        return (acc - clip) / c_div

    k1 = rhs(x)
    k2 = rhs(x + dt * k1)
    return x + (0.5 * dt) * (k1 + k2)


def _heun_step_compliance(
    x: torch.Tensor,
    dt: float,
    i_pos_node: torch.Tensor,
    i_neg_node: torch.Tensor,
    M_lin: torch.Tensor,
    b_dst: torch.Tensor,
    b_slope_x: torch.Tensor,
    b_bias: torch.Tensor,
    b_half_isat: torch.Tensor,
    b_G_g: torch.Tensor,
    b_res_const: torch.Tensor,
    Inc_b: torch.Tensor,
    s_src: torch.Tensor,
    s_slope_x: torch.Tensor,
    s_bias: torch.Tensor,
    s_const: torch.Tensor,
    s_amp: torch.Tensor,
    s_G_g: torch.Tensor,
    W_T: torch.Tensor,
    c_div: torch.Tensor,
    dst_start: int,
    n_tail: int,
    b_alpha_bwd: float,
    s_alpha_bwd: float,
    lam: float,
    scale: float,
    x_hard: float,
    v_off: float,
    k: float,
    c0: float,
) -> torch.Tensor:
    def rhs(state: torch.Tensor) -> torch.Tensor:
        f_top, f_bot = _leg_factors_fast(state, lam, scale, x_hard, v_off, k, c0)
        acc = i_pos_node * f_top - i_neg_node * f_bot + (state @ M_lin)

        # Boundary-terminal OTAs with leg compliance on destination nodes
        x_dst_b = state.index_select(1, b_dst)
        u_b = b_bias + b_slope_x * x_dst_b
        t_b = _tanh_ste_inline(u_b, b_alpha_bwd)
        ftb = f_top.index_select(1, b_dst)
        fbb = f_bot.index_select(1, b_dst)
        i_b = b_half_isat * ((ftb - fbb) + t_b * (ftb + fbb)) + (b_res_const - b_G_g * x_dst_b)
        acc = acc + (i_b @ Inc_b)

        # Shared-sense OTAs (Vref compliance hoisted) + accumulator leg compliance
        x_j = state.index_select(1, s_src)
        u_s = s_bias + s_slope_x * x_j
        t_s = _tanh_ste_inline(u_s, s_alpha_bwd)
        i_s = s_const + s_amp * t_s + s_G_g * x_j
        i_acc = i_s @ W_T
        fta = f_top[:, dst_start:]
        fba = f_bot[:, dst_start:]
        i_acc = torch.relu(i_acc) * fta - torch.relu(-i_acc) * fba
        acc_pre, acc_tail = torch.split(acc, [dst_start, n_tail], dim=1)
        acc = torch.cat([acc_pre, acc_tail + i_acc], dim=1)

        return acc / c_div

    k1 = rhs(x)
    k2 = rhs(x + dt * k1)
    return x + (0.5 * dt) * (k1 + k2)


def configure_compiled_heun(mode: str = "auto", device=None, logger=None) -> bool:
    """Enable or disable ``torch.compile`` on the stateless Heun step functions."""
    mode = str(mode or "off").lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"compile_mode must be 'auto', 'on', or 'off', got {mode!r}")
    _COMPILE_STATE["mode"] = mode
    _COMPILE_STATE["logger"] = logger
    if mode == "off":
        _COMPILE_STATE["enabled"] = False
        return False

    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type != "cuda" or not torch.cuda.is_available():
        _COMPILE_STATE["enabled"] = False
        if mode == "on" and logger is not None:
            logger.warning("[fast_heun] --compile-mode=on requested without CUDA; using eager fast path")
        return False

    cap = torch.cuda.get_device_capability(dev)
    if mode == "auto" and cap[0] < 7:
        _COMPILE_STATE["enabled"] = False
        if logger is not None:
            logger.info(
                f"[fast_heun] GPU compute capability {cap[0]}.{cap[1]} < 7.0 "
                f"(Triton unsupported); using eager hoisted fast path"
            )
        return False

    if _COMPILE_STATE["failed"] and mode == "auto":
        _COMPILE_STATE["enabled"] = False
        return False

    try:
        import torch._dynamo as _dynamo
        if _dynamo.config.cache_size_limit < 64:
            _dynamo.config.cache_size_limit = 64
        if _COMPILE_STATE["standard"] is None:
            _COMPILE_STATE["standard"] = torch.compile(
                _heun_step_standard, backend="inductor", fullgraph=True, dynamic=True,
            )
        if _COMPILE_STATE["compliance"] is None:
            _COMPILE_STATE["compliance"] = torch.compile(
                _heun_step_compliance, backend="inductor", fullgraph=True, dynamic=True,
            )
        _COMPILE_STATE["enabled"] = True
        if logger is not None:
            logger.info(
                f"[fast_heun] torch.compile enabled (sm_{cap[0]}{cap[1]}, fullgraph=True)"
            )
        return True
    except Exception as exc:
        _COMPILE_STATE["failed"] = True
        _COMPILE_STATE["enabled"] = False
        if logger is not None:
            logger.warning(f"[fast_heun] torch.compile setup failed ({exc}); using eager hoisted fast path")
        return False


def compiled_heun_enabled() -> bool:
    """True when inductor-compiled Heun steps are currently in use."""
    return bool(_COMPILE_STATE.get("enabled"))


def disable_compiled_heun(reason: str = "") -> None:
    """Turn compiled Heun steps off for the rest of the process.

    Sets ``_COMPILE_STATE["enabled"]=False`` and ``["failed"]=True`` so later
    forwards use the eager hoisted step. The forward try/except in
    ``fast_heun_forward`` only covers the step call; a backward failure inside
    ``loss.backward()`` must call this and recompute the batch eagerly.
    """
    _COMPILE_STATE["enabled"] = False
    _COMPILE_STATE["failed"] = True
    log = _COMPILE_STATE.get("logger")
    if log is not None and reason:
        log.warning(f"[fast_heun] compiled Heun disabled ({reason})")


def _get_stage_geom_cache(stage, device: torch.device, dtype: torch.dtype):
    """Cache constant topology incidence matrices and identity matrix in stage.__dict__."""
    key = (device, dtype, int(stage.src.numel()), int(stage.boundary_dst.numel()), float(stage.c_eff), bool(stage.read_only_source))
    cache = stage.__dict__.get("_fast_heun_geom")
    if cache is not None and cache[0] == key:
        return cache[1]

    n_nodes = stage.num_nodes
    eye_n = torch.eye(n_nodes, device=device, dtype=dtype)

    n_core = int(stage.src.numel())
    e_src = torch.zeros((n_core, n_nodes), device=device, dtype=dtype)
    e_dst = torch.zeros((n_core, n_nodes), device=device, dtype=dtype)
    if n_core > 0:
        ar_c = torch.arange(n_core, device=device)
        e_src[ar_c, stage.src] = 1.0
        e_dst[ar_c, stage.dst] = 1.0
    s_mat_t = (e_src - e_dst).T.contiguous()  # [N, E_core]
    d_mat = (e_dst if stage.read_only_source else (e_dst - e_src)).contiguous()  # [E_core, N]

    n_b = int(stage.boundary_dst.numel())
    inc_b = torch.zeros((n_b, n_nodes), device=device, dtype=dtype)
    if n_b > 0:
        inc_b[torch.arange(n_b, device=device), stage.boundary_dst] = 1.0

    c_eff_vec = torch.full((1, n_nodes), float(stage.c_eff), device=device, dtype=dtype)

    payload = {
        "eye_n": eye_n,
        "e_src_f32": e_src.float(),
        "e_dst_f32": e_dst.float(),
        "s_mat_t": s_mat_t,
        "d_mat": d_mat,
        "inc_b": inc_b,
        "c_eff_vec": c_eff_vec,
        "clip_s_cache": {},
    }
    object.__setattr__(stage, "_fast_heun_geom", (key, payload))
    return payload


def _precompute_cell_params(lib, gm_override, has_res: bool, dtype: torch.dtype):
    """Evaluate all stage-invariant FreeTanhLibrary parameter transforms once."""
    c = lib.hw_constraints
    isat_scale = None
    if isinstance(gm_override, IsatScale):
        isat_scale = gm_override.m
        gm_override = None

    if ab_mode(c) is not None:
        a_val, b_val = lib.ab_values()
        a_val = a_val.unsqueeze(0)
        b_val = b_val.unsqueeze(0)
    else:
        a_val = F.softplus(lib.a_raw).unsqueeze(0)
        b_val = F.softplus(lib.b_raw).unsqueeze(0)

    if lib.mismatch is not None:
        n_edges = a_val.shape[-1]
        f_a = lib._mismatch_factor("a", n_edges, a_val)
        if f_a is not None:
            a_val = a_val * f_a
        f_b = lib._mismatch_factor("b", n_edges, b_val)
        if f_b is not None:
            b_val = b_val * f_b

    s = torch.sign(lib.s_raw)
    s_ste = (s + lib.s_raw - lib.s_raw.detach()).unsqueeze(0)

    if c is not None and (c.hw_slope_bound or c.hw_ic_coupling):
        gm = lib._gm_hw(c, gm_override)
        if gm.dim() == 1:
            gm = gm.unsqueeze(0)
    elif gm_override is not None:
        gm = gm_override
    else:
        gm = (lib.gm_min + (lib.gm_max - lib.gm_min) * torch.sigmoid(lib.gm_raw)).unsqueeze(0)

    sig_isat = torch.sigmoid(lib.isat_raw)
    isat = (lib.isat_min + (lib.isat_max - lib.isat_min) * sig_isat).clamp_min(1e-6)
    if lib.mismatch is not None:
        f_i = lib._mismatch_factor("isat", isat.shape[-1], isat)
        if f_i is not None:
            isat = isat * f_i

    if c is not None and c.hw_theta_bound and lib._bias_enabled:
        theta = theta_bound(lib.theta_raw, THETA_MAX).unsqueeze(0)
    else:
        theta = lib.theta_raw.unsqueeze(0) if lib._bias_enabled else None
    if lib.mismatch is not None and lib._bias_enabled:
        f_t = lib._mismatch_factor("theta", theta.shape[-1], theta)
        if f_t is not None:
            theta = theta + f_t

    isat_f = isat.unsqueeze(0)
    if isat_scale is not None:
        isat_f = isat_f * isat_scale

    if has_res:
        g_res = F.softplus(lib.g_resistive_raw).unsqueeze(0)
        if lib.mismatch is not None:
            f_g = lib._mismatch_factor("g", g_res.shape[-1], g_res)
            if f_g is not None:
                g_res = g_res * f_g
    else:
        g_res = torch.zeros_like(isat_f)

    use_ste = bool(c is not None and (c.hw_no_alpha or c.hw_alpha_bwd > 0.0))
    alpha_bwd = 0.0 if c is None else float(c.hw_alpha_bwd)
    return (
        a_val.to(dtype),
        b_val.to(dtype),
        s_ste.to(dtype),
        gm.to(dtype),
        isat_f.to(dtype),
        None if theta is None else theta.to(dtype),
        g_res.to(dtype),
        use_ste,
        alpha_bwd,
        float(lib.ALPHA),
        float(lib._beta_softness),
    )


def fast_heun_forward(
    stage,
    x0: torch.Tensor,
    t_span: float,
    num_steps: int,
    store_trajectory: bool,
    u: torch.Tensor,
    x_hold: torch.Tensor | None,
    gate_core_cached: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run ``num_steps`` Heun steps with all stage-invariant math hoisted."""
    dt = float(t_span) / float(num_steps)
    batch_size = int(x0.shape[0])
    dtype = x0.dtype
    device = x0.device
    comp = stage._hw_compliance_on()
    geom = _get_stage_geom_cache(stage, device, dtype)

    # 1. Core frozen nonlinear contribution
    x_core = x0 if x_hold is None else x_hold
    i_edge_const = stage._compute_i_edge_const(x_core, gate_core_cached)

    # 2. Core linear matrix M_lin = M_resistive - diag(leak)
    if stage._has_resistive:
        g_core = F.softplus(stage.cell_lib.g_resistive_raw)
        if stage.cell_lib.mismatch is not None:
            f_g = stage.cell_lib._mismatch_factor("g", g_core.shape[-1], g_core)
            if f_g is not None:
                g_core = g_core * f_g
        g_eff = (g_core * torch.sigmoid(stage.z_logits)).to(dtype)
        m_lin = (geom["s_mat_t"] * g_eff) @ geom["d_mat"]
    else:
        m_lin = torch.zeros_like(geom["eye_n"])

    if stage.leak_mode == "non-programmable":
        m_lin = m_lin - stage.leak_constant * geom["eye_n"]
    else:
        leak_vec = F.softplus(stage.raw_leak).to(dtype)
        m_lin = m_lin - torch.diag(leak_vec)

    # 3. Shared GLN rails for this stage entry
    gln_z = stage._gln_rails_z(u)

    # 4. Boundary family precomputation
    u_src = u[:, stage.boundary_src].to(dtype)
    b_mask = torch.sigmoid(stage.boundary_z_logits).unsqueeze(0).to(dtype)
    b_vca = stage.edge_gain("boundary", u)
    if b_vca is not None:
        b_vca = b_vca.to(dtype)
    b_gm_ov = stage.edge_gm_scale("boundary", stage.boundary_cell_lib, gln_z)
    b_a, b_b, b_s, b_gm, b_isat, b_th, b_g, b_ste, b_abwd, b_alpha, b_beta = _precompute_cell_params(
        stage.boundary_cell_lib, b_gm_ov, hasattr(stage.boundary_cell_lib, "resistive_current"), dtype,
    )
    b_gate = b_mask if b_vca is None else (b_mask * b_vca)
    b_scale = stage._isat_eff_scale(
        stage.boundary_cell_lib, b_mask.squeeze(0), b_vca,
    )
    if b_scale is not None:
        b_scale = b_scale.to(dtype)
    b_slope_x = (-(b_s * b_b * b_gm)).expand(batch_size, -1)
    b_arg = b_s * (b_a * u_src)
    if b_th is not None:
        b_arg = b_arg + b_th
    b_bias = (b_arg * b_gm).expand(batch_size, -1)
    b_isat_g = (b_isat * b_gate).expand(batch_size, -1)
    b_g_g = (b_g * b_gate).expand(batch_size, -1)

    # 5. Shared-sense + crossbar readout precomputation
    w_t = stage.crossbar_weights().to(dtype).T.contiguous()
    s_mask = torch.sigmoid(stage.readout_sense_z_logits).unsqueeze(0).to(dtype)
    s_vca = stage.edge_gain("readout", u)
    if s_vca is not None:
        s_vca = s_vca.to(dtype)
    s_gm_ov = stage.edge_gm_scale("readout", stage.readout_sense_cell_lib, gln_z)
    s_a, s_b, s_s, s_gm, s_isat, s_th, s_g, s_ste, s_abwd, s_alpha, s_beta = _precompute_cell_params(
        stage.readout_sense_cell_lib, s_gm_ov, stage._sense_has_resistive, dtype,
    )
    s_gate = s_mask if s_vca is None else (s_mask * s_vca)
    s_scale = stage._isat_eff_scale(
        stage.readout_sense_cell_lib, s_mask.squeeze(0), s_vca,
    )
    if s_scale is not None:
        s_scale = s_scale.to(dtype)
    vref = stage.sense_vref().to(dtype)
    s_slope_x = (s_s * s_a * s_gm).expand(batch_size, -1)
    s_arg = -(s_s * (s_b * vref))
    if s_th is not None:
        s_arg = s_arg + s_th
    s_bias = (s_arg * s_gm).expand(batch_size, -1)
    s_isat_g = (s_isat * s_gate).expand(batch_size, -1)
    s_g_g = (s_g * s_gate).expand(batch_size, -1)

    c = stage.hw_constraints
    if c is None or c.hw_c_node is None:
        c_div = geom["c_eff_vec"]
    else:
        c_div = stage.node_capacitance(x0)

    dst_start = int(stage._readout_dst_start)
    n_tail = int(stage.num_nodes) - dst_start
    inc_b = geom["inc_b"]

    if comp:
        # Pre-scatter core FrozenLegs once per stage into [B, N] node tensors.
        i_pos_f32 = i_edge_const.i_pos.float()
        i_neg_f32 = i_edge_const.i_neg.float()
        e_dst_f32 = geom["e_dst_f32"]
        i_pos_node = i_pos_f32 @ e_dst_f32
        i_neg_node = i_neg_f32 @ e_dst_f32
        if not stage.read_only_source:
            e_src_f32 = geom["e_src_f32"]
            i_pos_node = i_pos_node + (i_neg_f32 @ e_src_f32)
            i_neg_node = i_neg_node + (i_pos_f32 @ e_src_f32)
        i_pos_node = i_pos_node.to(dtype)
        i_neg_node = i_neg_node.to(dtype)

        # In hw_compliance mode, _isat_eff_scale scales the OTA legs only (not the shunt).
        if b_scale is not None:
            b_isat_g = b_isat_g * b_scale
        b_half_isat = 0.5 * b_isat_g
        b_res_const = b_g_g * u_src

        if s_scale is not None:
            s_isat_g = s_isat_g * s_scale

        units = stage._hw_units()
        lam = float(LAMBDA_V) * float(units.vu)
        scale = float(units.vu) / float(units.u_t)
        x_hard = float(units.x_hard)
        v_off = 0.0
        k = float(COMPLIANCE_K)
        y0 = x_hard * scale - v_off
        ky0 = k * y0
        sp0 = y0 if ky0 > 20.0 else (math.log1p(math.exp(ky0)) / k if ky0 > -700.0 else 0.0)
        c0 = 1.0 - math.exp(-sp0)

        ft_vref, fb_vref = _leg_factors_fast(vref, lam, scale, x_hard, v_off, k, c0)
        s_const = (0.5 * s_isat_g * (ft_vref - fb_vref) - s_g_g * vref).expand(batch_size, -1)
        s_amp = (0.5 * s_isat_g * (ft_vref + fb_vref)).expand(batch_size, -1)

        step_args = (
            dt, i_pos_node, i_neg_node, m_lin,
            stage.boundary_dst, b_slope_x, b_bias, b_half_isat, b_g_g, b_res_const, inc_b,
            stage.readout_sense_src, s_slope_x, s_bias, s_const, s_amp, s_g_g, w_t,
            c_div, dst_start, n_tail, b_abwd, s_abwd,
            lam, scale, x_hard, v_off, k, c0,
        )
        eager_fn = _heun_step_compliance
        step_fn = _COMPILE_STATE["compliance"] if _COMPILE_STATE["enabled"] else eager_fn
    else:
        # In standard mode, _clamp_isat_eff scales the full dynamic family current (tanh + shunt).
        if b_scale is not None:
            b_isat_g = b_isat_g * b_scale
            b_g_g = b_g_g * b_scale
        b_res_const = b_g_g * u_src
        b_gate_src = torch.sigmoid((stage.x_max - u_src.abs()) / b_beta)

        if s_scale is not None:
            s_isat_g = s_isat_g * s_scale
            s_g_g = s_g_g * s_scale
        s_gate_dst = torch.sigmoid((stage.x_max - vref.abs()) / s_beta).view(1, 1)
        s_vref = vref.view(1, 1)

        clip_s_raw = stage.clip_sharpness()
        if isinstance(clip_s_raw, torch.Tensor):
            clip_s = clip_s_raw.to(dtype)
        else:
            cs_f = float(clip_s_raw)
            clip_s = geom["clip_s_cache"].get(cs_f)
            if clip_s is None:
                clip_s = torch.tensor(cs_f, device=device, dtype=dtype)
                geom["clip_s_cache"][cs_f] = clip_s

        step_args = (
            dt, i_edge_const, m_lin,
            stage.boundary_dst, b_slope_x, b_bias, b_isat_g, b_gate_src, b_g_g, b_res_const, inc_b,
            stage.readout_sense_src, s_slope_x, s_bias, s_isat_g, s_gate_dst, s_g_g, s_vref, w_t,
            float(stage.x_max), float(stage.clip_current), clip_s, c_div, dst_start, n_tail,
            b_alpha, b_abwd, b_ste, b_beta, s_alpha, s_abwd, s_ste, s_beta,
        )
        eager_fn = _heun_step_standard
        step_fn = _COMPILE_STATE["standard"] if _COMPILE_STATE["enabled"] else eager_fn

    x = x0
    traj_chunks = [x] if store_trajectory else None
    for step_idx in range(int(num_steps)):
        if step_fn is not eager_fn:
            try:
                x = step_fn(x, *step_args)
            except Exception as exc:
                log = _COMPILE_STATE.get("logger")
                if log is not None:
                    log.warning(f"[fast_heun] compiled step failed ({exc}); falling back to eager fast path")
                _COMPILE_STATE["failed"] = True
                _COMPILE_STATE["enabled"] = False
                step_fn = eager_fn
                x = eager_fn(x, *step_args)
        else:
            x = eager_fn(x, *step_args)
        if store_trajectory:
            traj_chunks.append(x)

    traj = torch.stack(traj_chunks, dim=2) if store_trajectory else None
    return x, traj
