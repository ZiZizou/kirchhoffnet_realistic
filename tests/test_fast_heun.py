"""Unit & regression tests for fast_heun.py and Phase-A training optimizations."""

from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

import hw as hw_mod
from cell_library import make_cell_library
from ctle_dagger_common import (
    _phase_a_decode_and_forward,
    _phase_a_decode_and_forward_with_validity,
    _phase_a_validity_nll,
    run_phase_a_training,
)
from differential_stage import DifferentialStage
from fast_heun import (
    _heun_step_compliance,
    _heun_step_standard,
    configure_compiled_heun,
    fast_heun_forward,
)
from hw import HWConfig
from hw.constraints import HWConstraints, MISMATCH_DEFAULT
from topology import build_net_from_config


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def _build_ctle_net(
    preset_name: str | None = None,
    steps_per_stage: int = 8,
    num_stages: int = 2,
    leak_mode: str = "non-programmable",
    extra_constraints: dict | None = None,
    dtype=torch.float32,
):
    hw_cfg = hw_mod.HWConfig.from_dict(hw_mod.load_preset(preset_name)) if preset_name else None
    if extra_constraints:
        if hw_cfg is None:
            hw_cfg = HWConfig(constraints=HWConstraints(**extra_constraints))
        else:
            c_dict = hw_cfg.constraints.to_dict() if hasattr(hw_cfg.constraints, "to_dict") else vars(hw_cfg.constraints).copy()
            c_dict.update(extra_constraints)
            hw_cfg = HWConfig(
                units=hw_cfg.units,
                constraints=HWConstraints(**c_dict),
                timemux=hw_cfg.timemux,
            )
    cfg = {
        "stages": [
            {
                "num_inputs": 4,
                "num_hidden": 11,
                "num_proj": 0,
                "num_outputs": 0,
                "hidden_family": "small_world",
                "hidden_kwargs": {"k": 2, "p": 0.2, "seed": 1, "bidirectional": False},
                "input_pattern": "all_to_all",
                "output_pattern": "all_to_all",
                "proj_pattern": "all_to_all",
                "edge_repeats": 2,
                "t_span": 4.660583 / 7.0,
                "num_steps": steps_per_stage,
            }
            for _ in range(num_stages)
        ],
        "out_dim": 7,
        "write_mode": "sparse_proj",
        "read_mode": "dense",
        "use_robust_input": False,
    }
    cell_lib = make_cell_library(
        "tanh_free",
        gm_max=4.241970,
        isat_max=16.0 if preset_name else 28.68855,
    )
    net = build_net_from_config(
        cfg,
        cell_lib=cell_lib,
        leak_mode=leak_mode,
        freeze_read=True,
        interstage_activation="residual-relu-tanh",
        boundary_fan_out={0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]},
        enable_temporal_readout=True,
        readout_mode="shared_sense",
        readout_senses_per_node=2,
        learnable_clip_sharpness=True,
        gln_rails=True,
        gln_B=4,
        gln_rank=2,
        gln_families="boundary,readout",
        x_max=4.0,
        vca_enabled=True,
        vca_rank=3,
        vca_core_enabled=True,
        vca_separate_core_bus=True,
        hw=hw_cfg,
    ).to(dtype=dtype)
    # Perturb zero-init VCA/GLN/xbar weights slightly so all branches are non-trivial.
    with torch.no_grad():
        g = torch.Generator().manual_seed(42)
        for p in net.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g, dtype=dtype) * 0.1)
    return net


def _run_generic_rhs_stage(stage, x0, t_span, num_steps, store_trajectory, u, x_hold=None):
    """Force the generic rhs() loop by placing rhs into stage.__dict__."""
    stage.rhs = type(stage).rhs.__get__(stage)
    try:
        return stage._forward_heun(
            x0=x0, t_span=t_span, num_steps=num_steps,
            store_trajectory=store_trajectory, u=u,
            **({} if x_hold is None else {"x_hold": x_hold}),
        )
    finally:
        if "rhs" in stage.__dict__:
            del stage.rhs


@pytest.mark.parametrize(
    "preset,steps,extra",
    [
        (None, 8, None),
        ("phase3a_core", 10, None),
        ("phase3a_core_diff", 10, None),
        ("phase3a_full", 10, None),
        ("phase3b_tm", 10, None),
        ("phase3a_core", 6, {"hw_isat_eff_clamp": True, "hw_alpha_bwd": 0.02}),
        ("phase3a_full", 6, {"hw_isat_eff_clamp": True, "hw_alpha_bwd": 0.02}),
    ],
)
@pytest.mark.parametrize("store_traj", [False, True])
def test_fast_heun_matches_generic_rhs_float32_and_float64(preset, steps, extra, store_traj):
    torch.manual_seed(7)
    # 1. float32 through stage(...) directly (normal training/eval dispatch)
    net32 = _build_ctle_net(preset, steps, num_stages=2, extra_constraints=extra, dtype=torch.float32)
    net32.eval()
    st32 = net32.core.stages[0]
    x0_32 = torch.randn(12, st32.num_nodes, dtype=torch.float32)
    u_32 = torch.randn(12, 4, dtype=torch.float32)
    x_hold_32 = torch.randn(12, st32.num_nodes, dtype=torch.float32) if preset == "phase3b_tm" else None

    st32_ref = copy.deepcopy(st32)
    y_fast, traj_fast = st32(x0_32, t_span=0.6658, num_steps=steps, store_trajectory=store_traj, u=u_32,
                             **({} if x_hold_32 is None else {"x_hold": x_hold_32}))
    loss_fast = (y_fast ** 2).mean() + (0.0 if traj_fast is None else 0.1 * (traj_fast ** 2).mean())
    loss_fast.backward()

    y_ref, traj_ref = _run_generic_rhs_stage(
        st32_ref, x0_32, 0.6658, steps, store_traj, u_32, x_hold=x_hold_32,
    )
    loss_ref = (y_ref ** 2).mean() + (0.0 if traj_ref is None else 0.1 * (traj_ref ** 2).mean())
    loss_ref.backward()

    torch.testing.assert_close(y_fast, y_ref, rtol=1e-4, atol=1e-5)
    if store_traj:
        torch.testing.assert_close(traj_fast, traj_ref, rtol=1e-4, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(st32.named_parameters(), st32_ref.named_parameters()):
        assert n1 == n2
        if p1.grad is None and p2.grad is None:
            continue
        assert p1.grad is not None and p2.grad is not None, n1
        torch.testing.assert_close(p1.grad, p2.grad, rtol=2e-4, atol=2e-5, msg=lambda m, n=n1: f"f32 {n}: {m}")

    # 2. float64 direct fast_heun_forward vs generic rhs
    net64 = _build_ctle_net(preset, steps, num_stages=2, extra_constraints=extra, dtype=torch.float64)
    net64.eval()
    st64 = net64.core.stages[0]
    st64_ref = copy.deepcopy(st64)
    x0_64 = x0_32.double()
    u_64 = u_32.double()
    x_hold_64 = None if x_hold_32 is None else x_hold_32.double()

    y64_fast, traj64_fast = fast_heun_forward(
        st64, x0_64, 0.6658, steps, store_traj, u_64, x_hold_64, st64.edge_gain("core", u_64),
    )
    l64_fast = (y64_fast ** 2).mean() + (0.0 if traj64_fast is None else 0.1 * (traj64_fast ** 2).mean())
    l64_fast.backward()

    y64_ref, traj64_ref = _run_generic_rhs_stage(
        st64_ref, x0_64, 0.6658, steps, store_traj, u_64, x_hold=x_hold_64,
    )
    l64_ref = (y64_ref ** 2).mean() + (0.0 if traj64_ref is None else 0.1 * (traj64_ref ** 2).mean())
    l64_ref.backward()

    torch.testing.assert_close(y64_fast, y64_ref, rtol=1e-6, atol=1e-7)
    if store_traj:
        torch.testing.assert_close(traj64_fast, traj64_ref, rtol=1e-6, atol=1e-7)
    for (n1, p1), (n2, p2) in zip(st64.named_parameters(), st64_ref.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6, msg=lambda m, n=n1: f"f64 {n}: {m}")


def test_fast_heun_programmable_leak_and_mismatch_and_c_uniform():
    # Programmable leak
    net_prog = _build_ctle_net("phase3a_full", steps_per_stage=6, leak_mode="programmable")
    st_p = net_prog.core.stages[0]
    st_p_ref = copy.deepcopy(st_p)
    x = torch.randn(8, st_p.num_nodes)
    u = torch.randn(8, 4)
    y1, _ = st_p(x, t_span=0.6658, num_steps=6, store_trajectory=False, u=u)
    y1.sum().backward()
    y2, _ = _run_generic_rhs_stage(st_p_ref, x, 0.6658, 6, False, u)
    y2.sum().backward()
    torch.testing.assert_close(y1, y2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(st_p.raw_leak.grad, st_p_ref.raw_leak.grad, rtol=1e-4, atol=1e-5)

    # Seeded mismatch in train mode (deterministic across identical seed)
    net_mm = _build_ctle_net(
        "phase3a_full", steps_per_stage=6,
        extra_constraints={"hw_mismatch_sigma": dict(MISMATCH_DEFAULT)},
    )
    net_mm.train()
    y_mm, _ = net_mm.core(torch.randn(8, 18), store_trajectory=False, u=u)
    assert torch.isfinite(y_mm).all()

    # hw_c_node="uniform" matches legacy float32 bit-for-bit
    plain = _build_ctle_net(None, steps_per_stage=6)
    unif = _build_ctle_net(None, steps_per_stage=6, extra_constraints={"hw_c_node": "uniform"})
    unif.load_state_dict(plain.state_dict())
    plain.eval()
    unif.eval()
    with torch.no_grad():
        yp, _ = plain(u)
        yu, _ = unif(u)
    torch.testing.assert_close(yu, yp, rtol=0, atol=0)


def test_dynamo_explain_zero_graph_breaks():
    b, n, eb, es, d_out = 8, 18, 8, 22, 7
    x = torch.randn(b, n, requires_grad=True)
    i_edge_const = torch.randn(b, n, requires_grad=True)
    i_pos_node = torch.randn(b, n, requires_grad=True)
    i_neg_node = torch.randn(b, n, requires_grad=True)
    m_lin = torch.randn(n, n, requires_grad=True)
    b_dst = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.long)
    b_slope_x = torch.randn(b, eb, requires_grad=True)
    b_bias = torch.randn(b, eb, requires_grad=True)
    b_isat_g = torch.randn(b, eb, requires_grad=True)
    b_gate_src = torch.randn(b, eb, requires_grad=True)
    b_g_g = torch.randn(b, eb, requires_grad=True)
    b_res_const = torch.randn(b, eb, requires_grad=True)
    inc_b = torch.zeros(eb, n)
    inc_b[torch.arange(eb), b_dst] = 1.0
    s_src = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], dtype=torch.long)
    s_slope_x = torch.randn(b, es, requires_grad=True)
    s_bias = torch.randn(b, es, requires_grad=True)
    s_isat_g = torch.randn(b, es, requires_grad=True)
    s_gate_dst = torch.randn(1, 1, requires_grad=True)
    s_g_g = torch.randn(b, es, requires_grad=True)
    s_vref = torch.randn(1, 1, requires_grad=True)
    w_t = torch.randn(es, d_out, requires_grad=True)
    clip_s = torch.tensor(0.02, requires_grad=True)
    c_div = torch.ones(1, n)

    exp_std = torch._dynamo.explain(_heun_step_standard)(
        x, 0.0166, i_edge_const, m_lin,
        b_dst, b_slope_x, b_bias, b_isat_g, b_gate_src, b_g_g, b_res_const, inc_b,
        s_src, s_slope_x, s_bias, s_isat_g, s_gate_dst, s_g_g, s_vref, w_t,
        4.0, 0.05, clip_s, c_div, 11, 7,
        0.01, 0.0, True, 0.02, 0.01, 0.0, True, 0.02,
    )
    assert exp_std.graph_break_count == 0, exp_std.break_reasons

    exp_cmp = torch._dynamo.explain(_heun_step_compliance)(
        x, 0.0166, i_pos_node, i_neg_node, m_lin,
        b_dst, b_slope_x, b_bias, b_isat_g, b_g_g, b_res_const, inc_b,
        s_src, s_slope_x, s_bias, b_g_g.new_zeros(b, es), s_isat_g, s_g_g, w_t,
        c_div, 11, 7, 0.0, 0.0,
        0.0105, 2.708, 5.714, 0.0, 2000.0, 1.0,
    )
    assert exp_cmp.graph_break_count == 0, exp_cmp.break_reasons


@pytest.mark.parametrize(
    "preset,min_reduction",
    [
        ("phase3a_core", 3.0),
        ("phase3a_full", 4.0),
    ],
)
def test_aten_op_reduction(preset, min_reduction):
    net = _build_ctle_net(preset, steps_per_stage=40, num_stages=1)
    st = net.core.stages[0]
    x0 = torch.randn(4, st.num_nodes)
    u = torch.randn(4, 4)

    c_new = _OpCounter()
    with c_new:
        y_new, _ = st(x0, t_span=0.6658, num_steps=40, store_trajectory=False, u=u)
        y_new.sum().backward()
    st.zero_grad(set_to_none=True)

    c_old = _OpCounter()
    with c_old:
        y_old, _ = _run_generic_rhs_stage(st, x0, 0.6658, 40, False, u)
        y_old.sum().backward()
    st.zero_grad(set_to_none=True)

    ops_old = sum(c_old.counts.values())
    ops_new = sum(c_new.counts.values())
    assert ops_old / ops_new >= min_reduction, (ops_old, ops_new)


class _DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)


class _DummyZig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(7, 5)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_scaled):
        z = self.net(x_scaled)
        return {
            "pred_h_soft": torch.exp(0.1 * z[:, 0]).clamp(min=1e-3, max=1e3),
            "pred_w_soft": torch.exp(0.1 * z[:, 1]).clamp(min=1e-3, max=1e3),
            "pred_j_soft": torch.exp(0.1 * z[:, 2]).clamp(min=1e-3, max=1e3),
            "mu_power": 0.1 * z[:, 3],
            "p_valid": torch.sigmoid(z[:, 4]),
        }


class _TinyPhaseAStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 7)
        self.register_buffer("log_lo", torch.full((7,), -6.0))
        self.register_buffer("log_hi", torch.full((7,), 1.0))

    def forward(self, x):
        return self.fc(x)


def test_single_zig_pass_matches_separate_calls():
    torch.manual_seed(0)
    student = _TinyPhaseAStudent()
    zig = _DummyZig()
    scaler_x = _DummyScaler(np.zeros(7), np.ones(7) * 2.0)
    scaler_yp = _DummyScaler([-3.0], [0.5])
    x_mean_t = torch.from_numpy(scaler_x.mean_)
    x_scale_t = torch.from_numpy(scaler_x.scale_)

    logits1 = torch.randn(16, 7, requires_grad=True)
    logits2 = logits1.detach().clone().requires_grad_(True)

    p1, j1, h1, w1, b1 = _phase_a_decode_and_forward(student, logits1, scaler_x, scaler_yp, zig)
    nll1 = _phase_a_validity_nll(student, logits1, scaler_x, zig)
    loss1 = p1.mean() + j1.mean() + h1.mean() + w1.mean() + b1.mean() + 0.3 * nll1
    loss1.backward()

    p2, j2, h2, w2, b2, nll2 = _phase_a_decode_and_forward_with_validity(
        student, logits2, x_mean_t, x_scale_t, float(scaler_yp.scale_[0]), float(scaler_yp.mean_[0]), zig,
    )
    loss2 = p2.mean() + j2.mean() + h2.mean() + w2.mean() + b2.mean() + 0.3 * nll2
    loss2.backward()

    for a, b in ((p1, p2), (j1, j2), (h1, h2), (w1, w2), (b1, b2), (nll1, nll2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(logits1.grad, logits2.grad, rtol=1e-6, atol=1e-7)


def test_max_hours_stops_cleanly_and_resumes(tmp_path):
    torch.manual_seed(0)
    n = 36
    rng = np.random.default_rng(123)
    specs = rng.uniform(0.05, 0.5, size=(n, 4)).astype(np.float32)
    params = rng.uniform(1e-4, 1.0, size=(n, 7)).astype(np.float32)
    canonical = {
        "specs": specs,
        "params": params,
        "mlp_logits_trial0019": np.zeros((n, 7), dtype=np.float32),
        "power_norm_const": np.array(1e-3, dtype=np.float32),
        "schema_rev": np.array(2, dtype=np.int64),
        "train_idx": np.arange(0, 20, dtype=np.int64),
        "val_idx": np.arange(20, 28, dtype=np.int64),
        "test_idx": np.arange(28, 36, dtype=np.int64),
    }
    student = _TinyPhaseAStudent()
    zig = _DummyZig()
    scaler_x = _DummyScaler(np.full(7, -2.5), np.full(7, 2.0))
    scaler_yp = _DummyScaler([-3.0], [0.5])
    out_dir = tmp_path / "run"

    ctx = {
        "DEVICE": torch.device("cpu"),
        "scaler_X": scaler_x,
        "scaler_y_p": scaler_yp,
        "zig_model": zig,
        "eye_scale_h": 0.1,
        "eye_scale_w": 0.1,
        "eye_scale_j": 0.1,
        "canonical_dataset": canonical,
        "epochs": 4,
        "batch_size": 10,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "output_dir": str(out_dir),
        "val_eval_every": 2,
        "earlystop_patience": 10,
        "compile_mode": "off",
        "max_hours": 1e-7,  # triggers after epoch 1
    }
    res1 = run_phase_a_training(student, ctx, model_name="tiny_budget")
    assert res1["stopped_by_time_budget"] is True
    assert res1["completed_epochs"] == 1
    assert (out_dir / "phase_a_last.pt").is_file()
    assert (out_dir / "phase_a_best.pt").is_file()
    hist1 = json.loads((out_dir / "phase_a_history.json").read_text())
    assert hist1["stopped_by_time_budget"] is True
    assert hist1["completed_epochs"] == 1
    assert hist1["history"]["epoch"] == [1]

    # Resume with generous budget to completion (epochs 2..4)
    ctx_resume = dict(ctx)
    ctx_resume["resume"] = str(out_dir / "phase_a_last.pt")
    ctx_resume["max_hours"] = None
    student2 = _TinyPhaseAStudent()
    res2 = run_phase_a_training(student2, ctx_resume, model_name="tiny_budget")
    assert res2["stopped_by_time_budget"] is False
    assert res2["completed_epochs"] == 4
    hist2 = json.loads((out_dir / "phase_a_history.json").read_text())
    assert hist2["history"]["epoch"] == [1, 2, 3, 4]
    assert configure_compiled_heun("off") is False


def test_compiled_backward_failure_falls_back_to_eager_gradients():
    """A compiled step whose backward raises must not kill the batch.

    The forward try/except in fast_heun only covers the step call. This test
    installs a step whose forward succeeds and whose backward raises, then
    checks phase_a_fwd_bwd_guard disables compile and matches an eager run.
    """
    import fast_heun
    from ctle_dagger_common import phase_a_fwd_bwd_guard
    from fast_heun import compiled_heun_enabled

    saved = dict(fast_heun._COMPILE_STATE)
    calls = {"fwd": 0, "bwd": 0}

    class _RaiseBwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y):
            calls["fwd"] += 1
            return y

        @staticmethod
        def backward(ctx, grad):
            calls["bwd"] += 1
            raise RuntimeError("compiled heun backward failed")

    def bad_step(*args, **kwargs):
        y = fast_heun._heun_step_standard(*args, **kwargs)
        return _RaiseBwd.apply(y)

    try:
        torch.manual_seed(0)
        net = _build_ctle_net(None, steps_per_stage=4, num_stages=1, dtype=torch.float32)
        stage = net.core.stages[0]
        x0 = torch.randn(4, stage.num_nodes)
        u = torch.randn(4, 4)
        t_span = 0.6658

        fast_heun._COMPILE_STATE["enabled"] = False
        fast_heun._COMPILE_STATE["failed"] = False
        fast_heun._COMPILE_STATE["standard"] = None
        net_ref = copy.deepcopy(net)
        stage_ref = net_ref.core.stages[0]
        y_ref, _ = stage_ref(x0, t_span=t_span, num_steps=4, store_trajectory=False, u=u)
        (y_ref ** 2).mean().backward()
        ref_grads = {
            n: p.grad.detach().clone()
            for n, p in stage_ref.named_parameters()
            if p.grad is not None
        }

        fast_heun._COMPILE_STATE["enabled"] = True
        fast_heun._COMPILE_STATE["failed"] = False
        fast_heun._COMPILE_STATE["standard"] = bad_step
        opt = torch.optim.SGD(stage.parameters(), lr=0.0)

        def run_batch():
            opt.zero_grad(set_to_none=True)
            y, _ = stage(x0, t_span=t_span, num_steps=4, store_trajectory=False, u=u)
            loss = (y ** 2).mean()
            loss.backward()
            return loss

        phase_a_fwd_bwd_guard(opt, run_batch, student=stage)
        assert calls["fwd"] >= 1 and calls["bwd"] >= 1
        assert compiled_heun_enabled() is False
        assert fast_heun._COMPILE_STATE["failed"] is True
        compared = 0
        for n, p in stage.named_parameters():
            if n not in ref_grads:
                assert p.grad is None
                continue
            assert p.grad is not None, n
            diff = (p.grad - ref_grads[n]).abs().max().item()
            assert diff <= 1e-7, (n, diff)
            compared += 1
        assert compared > 0

        # Compile off: the same backward error must propagate.
        fast_heun._COMPILE_STATE["enabled"] = False
        fast_heun._COMPILE_STATE["failed"] = False
        fast_heun._COMPILE_STATE["standard"] = bad_step

        def run_eager_raise():
            raise RuntimeError("not a compile failure")

        with pytest.raises(RuntimeError, match="not a compile failure"):
            phase_a_fwd_bwd_guard(opt, run_eager_raise, student=stage)
        assert compiled_heun_enabled() is False
    finally:
        fast_heun._COMPILE_STATE.clear()
        fast_heun._COMPILE_STATE.update(saved)


def test_phase_a_rng_round_trip_and_partial_restore():
    import random
    from ctle_dagger_common import capture_phase_a_rng, restore_phase_a_rng

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    state = capture_phase_a_rng()
    assert set(state) == {"torch", "cuda", "numpy", "python"}
    assert state["cuda"] is None or isinstance(state["cuda"], list)
    a_t = torch.rand(4)
    a_n = np.random.rand(4)
    a_p = random.random()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    restore_phase_a_rng(state)
    torch.testing.assert_close(torch.rand(4), a_t)
    np.testing.assert_array_equal(np.random.rand(4), a_n)
    assert random.random() == a_p

    # Missing fields and a missing blob are tolerated.
    restore_phase_a_rng({})
    restore_phase_a_rng(None)
    restore_phase_a_rng({"numpy": state["numpy"]})


def test_resume_pads_missing_history_and_warns_when_rng_absent(tmp_path, caplog):
    import logging
    from ctle_dagger_common import run_phase_a_training

    torch.manual_seed(0)
    n = 24
    rng = np.random.default_rng(7)
    specs = rng.uniform(0.05, 0.5, size=(n, 4)).astype(np.float32)
    params = rng.uniform(1e-4, 1.0, size=(n, 7)).astype(np.float32)
    canonical = {
        "specs": specs,
        "params": params,
        "mlp_logits_trial0019": np.zeros((n, 7), dtype=np.float32),
        "power_norm_const": np.array(1e-3, dtype=np.float32),
        "schema_rev": np.array(2, dtype=np.int64),
        "train_idx": np.arange(0, 12, dtype=np.int64),
        "val_idx": np.arange(12, 18, dtype=np.int64),
        "test_idx": np.arange(18, 24, dtype=np.int64),
    }
    student = _TinyPhaseAStudent()
    zig = _DummyZig()
    scaler_x = _DummyScaler(np.full(7, -2.5), np.full(7, 2.0))
    scaler_yp = _DummyScaler([-3.0], [0.5])
    out_dir = tmp_path / "run"
    ctx = {
        "DEVICE": torch.device("cpu"),
        "scaler_X": scaler_x,
        "scaler_y_p": scaler_yp,
        "zig_model": zig,
        "eye_scale_h": 0.1,
        "eye_scale_w": 0.1,
        "eye_scale_j": 0.1,
        "canonical_dataset": canonical,
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "output_dir": str(out_dir),
        "val_eval_every": 5,
        "earlystop_patience": 10,
        "compile_mode": "off",
    }
    run_phase_a_training(student, ctx, model_name="tiny_pad")
    ckpt = out_dir / "phase_a_last.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "rng" in state
    state.pop("rng")
    state["history"].pop("val_invalid_rate")
    torch.save(state, ckpt)

    ctx2 = dict(ctx)
    ctx2["epochs"] = 2
    ctx2["resume"] = str(ckpt)
    with caplog.at_level(logging.INFO, logger="ctle_dagger"):
        res = run_phase_a_training(_TinyPhaseAStudent(), ctx2, model_name="tiny_pad")
    assert "checkpoint has no RNG state" in caplog.text
    assert "val_invalid_rate" in caplog.text
    assert res["completed_epochs"] == 2
    hist = json.loads((out_dir / "phase_a_history.json").read_text())
    assert len(hist["history"]["val_invalid_rate"]) == 2
    assert hist["history"]["val_invalid_rate"][0] != hist["history"]["val_invalid_rate"][0]  # NaN pad
    assert len(hist["history"]["epoch"]) == 2


def test_phase_a_rng_torch_save_load_round_trip(tmp_path):
    """torch.save -> torch.load must reproduce torch/numpy/python streams."""
    import random
    from ctle_dagger_common import capture_phase_a_rng, restore_phase_a_rng

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.rand(2)
    np.random.rand(2)
    random.random()
    state = capture_phase_a_rng()
    expected_t = torch.rand(4)
    expected_n = np.random.rand(4)
    expected_p = random.random()

    path = tmp_path / "rng.pt"
    torch.save({"rng": state}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    restore_phase_a_rng(loaded["rng"])
    torch.testing.assert_close(torch.rand(4), expected_t)
    np.testing.assert_array_equal(np.random.rand(4), expected_n)
    assert random.random() == expected_p


def test_restore_cuda_rng_slices_to_device_count(monkeypatch, caplog):
    """A 2-GPU CUDA RNG blob resumed on 1 GPU must not IndexError."""
    import logging
    from ctle_dagger_common import restore_phase_a_rng

    calls = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fake_set_all(states):
        calls["states"] = list(states)

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake_set_all)
    saved = [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]
    with caplog.at_level(logging.WARNING, logger="ctle_dagger"):
        restore_phase_a_rng({"cuda": saved})
    assert len(calls["states"]) == 1
    assert torch.equal(calls["states"][0], saved[0])
    assert "device_count" in caplog.text

    caplog.clear()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with caplog.at_level(logging.WARNING, logger="ctle_dagger"):
        restore_phase_a_rng({"cuda": saved})
    assert len(calls["states"]) == 2
    assert "device_count" not in caplog.text


def test_guard_restores_rng_before_eager_retry():
    import fast_heun
    from ctle_dagger_common import phase_a_fwd_bwd_guard

    saved = dict(fast_heun._COMPILE_STATE)
    try:
        fast_heun._COMPILE_STATE["enabled"] = True
        fast_heun._COMPILE_STATE["failed"] = False
        torch.manual_seed(0)
        seen = []
        calls = {"n": 0}

        def run_batch():
            calls["n"] += 1
            seen.append(torch.rand(1).item())
            if calls["n"] == 1:
                raise RuntimeError("compiled heun failed")
            return "ok"

        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
        assert phase_a_fwd_bwd_guard(opt, run_batch) == "ok"
        assert calls["n"] == 2
        assert seen[0] == seen[1]
        assert fast_heun.compiled_heun_enabled() is False
    finally:
        fast_heun._COMPILE_STATE.clear()
        fast_heun._COMPILE_STATE.update(saved)


def test_guard_reraises_cuda_oom_without_disabling_compile():
    import fast_heun
    from ctle_dagger_common import phase_a_fwd_bwd_guard

    saved = dict(fast_heun._COMPILE_STATE)
    try:
        fast_heun._COMPILE_STATE["enabled"] = True
        fast_heun._COMPILE_STATE["failed"] = False
        calls = {"n": 0}

        def run_batch():
            calls["n"] += 1
            raise torch.cuda.OutOfMemoryError("oom")

        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            phase_a_fwd_bwd_guard(opt, run_batch)
        assert calls["n"] == 1
        assert fast_heun.compiled_heun_enabled() is True
        assert fast_heun._COMPILE_STATE["failed"] is False
    finally:
        fast_heun._COMPILE_STATE.clear()
        fast_heun._COMPILE_STATE.update(saved)


def test_check_resume_hw_hash_warns_when_absent_and_raises_on_mismatch(caplog):
    import logging
    from ctle_dagger_common import check_resume_hw_hash

    with caplog.at_level(logging.WARNING, logger="ctle_dagger"):
        check_resume_hw_hash({"epoch": 1}, None, current_preset=None)
    assert "no hw_hash" in caplog.text

    check_resume_hw_hash(
        {"hw_hash": "9c9fe3fe9ec3", "hw_preset": "phase3a_full"},
        "9c9fe3fe9ec3",
        current_preset="phase3a_full",
    )
    with pytest.raises(RuntimeError, match="eda59b5b6e0c"):
        check_resume_hw_hash(
            {"hw_hash": "9c9fe3fe9ec3", "hw_preset": "phase3a_full"},
            "eda59b5b6e0c",
            current_preset="phase3a_core",
        )
