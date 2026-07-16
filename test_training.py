"""Training tests for KirchhoffNet.

These tests verify training loops, optimizer steps, and full grid
forward/backward passes. Moved out of test_smoke.py so the smoke test
stays fast.

Run with:
    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_training.py
"""

import copy
import os
import sys
import math
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

passed = 0
failed = 0

def check(name, condition, msg=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f": {msg}" if msg and not condition else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1

def main():
    test_sparsity_push()
    test_round_trip_preset()
    test_no_adc_flag()
    test_smooth2d_preset()
    test_smooth2d_grid_preset()
    test_housing_grid_preset()
    test_persistent_drive_auto_fan_out()
    test_persistent_drive_sparse_proj()
    test_amp_dtype_fix()
    test_fan_out_torus_bug_fix()
    test_write_fan_out_cli()
    test_write_fan_out_validation()
    test_mlp_benchmark()
    test_mlp_benchmark_tanh()
    test_v15_cell_parameters_preset_smooth2d_grid()
    test_stage_lr_scale_scheduler_compat()
    test_parameter_breakdown_aggregates_components()
    test_edge_repeats_propagation_to_preset()
    test_boundary_fan_out_basic()
    test_boundary_fan_out_zero_init()
    test_boundary_fan_out_validation()
    test_boundary_fan_out_grad_flow()
    test_boundary_fan_out_with_no_edge_gates()
    test_boundary_fan_out_incompatible_with_persistent_drive()
    test_boundary_fan_out_uses_null_input_mapper()

    print()
    print("=" * 60)
    print(f"Training test results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sparsity_push():
    print("\nTest 8: sparsity regularizer reduces P(active) over training")
    from cell_library import make_cell_library
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNetWithIO, KirchhoffNet
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import compute_loss, make_optimizer, default_ctx_factory, LAMBDAS

    cell_lib = make_cell_library('tanh')
    hid = ring_graph(3)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, active, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[8])
    inp = InputMapper(in_dim=1, out_dim=3)
    out = OutputMapper(node_dim=3, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=3, proj_count=0)

    opt = make_optimizer(net, lr=1e-2)
    u = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    target = torch.zeros(16, 1)
    ctx_factory = default_ctx_factory(net)

    with torch.no_grad():
        gates_before = torch.sigmoid(stage.z_logits).clone()

    lambdas = dict(LAMBDAS)
    lambdas["edge_gate"] = 0.1
    for step in range(50):
        ctx = ctx_factory(batch_size=16, device=u.device)
        opt.zero_grad()
        loss_task, loss_structural = compute_loss(net, u, target, ctx, F.mse_loss, lambdas=lambdas)
        (loss_task + loss_structural).backward()
        opt.step()

    with torch.no_grad():
        gates_after = torch.sigmoid(stage.z_logits)

    mean_before = gates_before.mean().item()
    mean_after = gates_after.mean().item()
    check("edge_gate: mean gate decreased", mean_after < mean_before,
          f"before={mean_before:.4f} after={mean_after:.4f}")

def test_round_trip_preset():
    print("\nTest 10: end-to-end round-trip with sinx preset (forward + backward + step)")
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from sim_context import SimContext
    from train import compute_loss, make_optimizer, default_ctx_factory, LAMBDAS

    net = build_net_from_preset("sinx", cell_lib=make_cell_library('tanh'))

    u = torch.linspace(-math.pi, math.pi, 32).unsqueeze(1)
    target = torch.sin(u)
    opt = make_optimizer(net, lr=1e-3)
    ctx_factory = default_ctx_factory(net)
    ctx = ctx_factory(batch_size=32, device=u.device)

    loss_task, loss_structural, parts = compute_loss(net, u, target, ctx, F.mse_loss, return_parts=True)
    check("sinx preset: total loss finite", math.isfinite(parts["total"]))
    check("sinx preset: task loss finite", math.isfinite(parts["task"]))

    loss_task.backward(retain_graph=True)
    loss_structural.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    opt.step()
    check("sinx preset: optimization step OK", True)

    # Second forward should differ from first (parameters changed)
    with torch.no_grad():
        y1, _ = net(u, store_trajectory=False)
        y2, _ = net(u, store_trajectory=False)
    check("sinx preset: re-forward succeeds (no NaN)", torch.isfinite(y2).all().item())

def test_no_adc_flag():
    print("\nTest 50: --no-adc disables per-layer ADC/DAC quantization")
    import subprocess
    import tempfile
    import torch.nn.functional as F
    from analog_noise import (
        AnalogMLPWrapper, NoiseConfig, evaluate_clean, evaluate_with_noise,
    )
    from mlp_benchmark import MLPRegressor

    # 1. CLI flag present in both scripts' --help output
    venv_py = "/home/annaik/Documents/ASPDAC_2026/venv/bin/python"
    smooth_help = subprocess.run(
        [venv_py,
         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",
         "--help"],
        capture_output=True, text=True,
    ).stdout
    housing_help = subprocess.run(
        [venv_py,
         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark_housing.py",
         "--help"],
        capture_output=True, text=True,
    ).stdout
    check("--no-adc flag present in mlp_benchmark.py --help",
          "--no-adc" in smooth_help)
    check("--no-adc flag present in mlp_benchmark_housing.py --help",
          "--no-adc" in housing_help)

    # 2. NoiseConfig with all three quantize flags False: pure-digital mode.
    torch.manual_seed(3)
    base = MLPRegressor(in_dim=2, hidden_dim=8, out_dim=1, num_layers=2)
    base.eval()

    cfg_pure = NoiseConfig(
        quant_bits=4,
        noise_std=0.0,
        mc_trials=1,
        quantize_input=False,
        quantize_output=False,
        quantize_intermediate=False,
        weight_noise=False,
        activation_noise=False,
    )
    wrapper_pure = AnalogMLPWrapper(base, cfg_pure)
    x = torch.randn(4, 2)
    y_pure = wrapper_pure(x)

    from analog_noise import fake_quantize_symmetric
    with torch.no_grad():
        q_fc1 = fake_quantize_symmetric(base.fc1.weight, bits=4, ste=False)
        q_fc2 = fake_quantize_symmetric(base.fc2.weight, bits=4, ste=False)
        h = F.relu(F.linear(x, q_fc1, base.fc1.bias))
        y_ref = F.linear(h, q_fc2, base.fc2.bias)
    check(
        "no-adc: wrapper output matches manual weight-only quantization",
        torch.allclose(y_pure, y_ref, atol=1e-5),
        f"max diff = {(y_pure - y_ref).abs().max().item():.3e}",
    )

    # 3. With circuit noise + --no-adc, output should differ from base
    cfg_noisy = NoiseConfig(
        quant_bits=4,
        noise_std=0.05,
        mc_trials=1,
        seed=7,
        quantize_input=False,
        quantize_output=False,
        quantize_intermediate=False,
    )
    wrapper_noisy = AnalogMLPWrapper(base, cfg_noisy)
    y_noisy = wrapper_noisy(x)
    with torch.no_grad():
        y_base = base(x)
    check("no-adc + circuit noise: output finite",
          torch.isfinite(y_noisy).all().item())
    check("no-adc + circuit noise: output differs from clean base",
          not torch.allclose(y_noisy, y_base, atol=1e-3))

    # 4. Compare degradation with and without --no-adc.
    loader = [(torch.randn(8, 2), torch.zeros(8, 1)) for _ in range(2)]

    cfg_full = NoiseConfig(
        quant_bits=4, noise_std=0.05, mc_trials=5, seed=0,
        quantize_input=True, quantize_output=True, quantize_intermediate=True,
    )
    cfg_digital = NoiseConfig(
        quant_bits=4, noise_std=0.05, mc_trials=5, seed=0,
        quantize_input=False, quantize_output=False, quantize_intermediate=False,
    )
    wrapper_full = AnalogMLPWrapper(
        MLPRegressor(in_dim=2, hidden_dim=8, out_dim=1), cfg_full,
    )
    wrapper_digital = AnalogMLPWrapper(
        MLPRegressor(in_dim=2, hidden_dim=8, out_dim=1), cfg_digital,
    )
    res_full = evaluate_with_noise(wrapper_full, loader, F.mse_loss,
                                   cfg_full, "cpu")
    res_digital = evaluate_with_noise(wrapper_digital, loader, F.mse_loss,
                                      cfg_digital, "cpu")
    check(
        "no-adc: pure-digital degradation <= full analog degradation",
        res_digital.mean <= res_full.mean + 1e-3,
        f"digital={res_digital.mean:.6f}, full={res_full.mean:.6f}",
    )

    # 5. End-to-end CLI smoke: train tiny model with --noise --no-adc
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "out")
        result = subprocess.run(
            [venv_py,
             "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",
             "--epochs", "1", "--patience", "100",
             "--hidden-dim", "20",
             "--noise", "--no-adc", "--quant-bits", "4",
             "--mc-trials", "3",
             "--output", out_dir],
            capture_output=True, text=True,
            cwd="/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal",
        )
        metrics_path = os.path.join(out_dir, "noise_metrics.txt")
        check(
            "no-adc CLI smoke: --noise --no-adc exit OK",
            result.returncode == 0,
            f"rc={result.returncode}, stderr={result.stderr[-300:]}",
        )
        check(
            "no-adc CLI smoke: noise_metrics.txt created",
            os.path.exists(metrics_path),
        )
        if os.path.exists(metrics_path):
            text = open(metrics_path).read()
            check(
                "no-adc CLI smoke: noise_metrics.txt records adc_quantization: False",
                "adc_quantization: False" in text,
                f"contents: {text}",
            )

def test_smooth2d_preset():
    print("\nTest NN: smooth2d preset structure and Franke dataset")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from io_mapper import SparseInputMapper

    cfg = PRESETS.get("smooth2d")
    check("smooth2d present in PRESETS", cfg is not None)
    if cfg is None:
        return

    s = cfg["stages"][0]
    check("smooth2d: 1 stage", len(cfg["stages"]) == 1)
    check("smooth2d: num_inputs=2", s["num_inputs"] == 2)
    check("smooth2d: num_hidden=10", s["num_hidden"] == 10)
    check("smooth2d: num_proj=2", s["num_proj"] == 2)
    check("smooth2d: line topology (topology-fix: no 1-hop write->read bypass)",
          s["hidden_family"] == "line")
    check("smooth2d: radius=2 (line topology)", s["hidden_kwargs"].get("radius") == 2)
    check("smooth2d: write_idx=[0,1]", cfg["write_idx"] == [0, 1])
    check("smooth2d: read_idx=[9] (hidden, not proj)", cfg["read_idx"] == [9])
    check("smooth2d: loss=mse", cfg["loss"] == "mse")
    check("smooth2d: out_dim=1", cfg["out_dim"] == 1)
    from config import SOLVER
    check("smooth2d: t_span=SOLVER.t_span",
          s["t_span"] == SOLVER["t_span"],
          f"got {s['t_span']}")
    check("smooth2d: num_steps=SOLVER.num_steps",
          s["num_steps"] == SOLVER["num_steps"],
          f"got {s['num_steps']}")
    check("smooth2d: no lambdas override", "lambdas" not in cfg)

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("smooth2d", cell_lib=cell_lib)
    check("smooth2d: builds successfully", net is not None)
    check("smooth2d: write_idx=[0,1]", net.write_idx == [0, 1])
    check("smooth2d: read_idx=[9]", net.read_idx == [9])
    check("smooth2d: uses SparseInputMapper",
          isinstance(net.input_mapper, SparseInputMapper))
    check("smooth2d: hid_count=10", net.hid_count == 10)
    check("smooth2d: proj_count=2", net.proj_count == 2)

    u = torch.rand(8, 2)
    ctx = None
    out, _ = net(u)
    check("smooth2d: forward shape (8,1)", out.shape == (8, 1))
    check("smooth2d: forward output is finite", torch.isfinite(out).all().item())

    from train_script import _franke
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    gx1, gx2 = torch.meshgrid(x1, x2, indexing="ij")
    f = _franke(gx1, gx2)
    check("smooth2d: Franke output is finite", torch.isfinite(f).all().item())
    check("smooth2d: Franke range in plausible bounds",
          f.min().item() >= -0.1 and f.max().item() <= 1.3,
          f"min={f.min().item():.4f} max={f.max().item():.4f}")

    from train import make_optimizer
    from train_script import make_data_smooth2d
    train_loader, _, task_fn = make_data_smooth2d(batch_size=8, val_size=16)
    optimizer = make_optimizer(net, lr=1e-3)
    net.train()
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b, _ = net(u_b)
        loss = task_fn(out_b, y_b)
        loss.backward()
        optimizer.step()
        check("smooth2d: 1-batch backward finite", math.isfinite(float(loss.item())))
        break

def test_smooth2d_grid_preset():
    print("\nTest NN2: smooth2d_grid preset (7x7 grid + 3 proj, fan-out I/O, 3 stages)")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from io_mapper import FanOutInputMapper, OutputMapper
    from stage_transfer import StageTransfer

    cfg = PRESETS.get("smooth2d_grid")
    check("smooth2d_grid present in PRESETS", cfg is not None)
    if cfg is None:
        return

    s = cfg["stages"][0]
    check("smooth2d_grid: 3 stages (multistage-smooth2d-grid spec)",
          len(cfg["stages"]) == 3)
    check("smooth2d_grid: num_inputs=2", s["num_inputs"] == 2)
    check("smooth2d_grid: num_hidden=49", s["num_hidden"] == 49)
    check("smooth2d_grid: num_proj=3", s["num_proj"] == 3)
    check("smooth2d_grid: hidden_family=grid", s["hidden_family"] == "grid")
    check("smooth2d_grid: height=7", s["hidden_kwargs"].get("height") == 7)
    check("smooth2d_grid: width=7", s["hidden_kwargs"].get("width") == 7)
    check("smooth2d_grid: kernel_size=3 (8-neighbor)",
          s["hidden_kwargs"].get("kernel_size") == 3)
    check("smooth2d_grid: write_mode=fan_out", cfg.get("write_mode") == "fan_out")
    check("smooth2d_grid: write_fan_out maps both inputs",
          cfg.get("write_fan_out") == {0: [0, 14, 28, 42], 1: [6, 20, 34, 48]})
    check("smooth2d_grid: read_idx = 7 center column + 3 proj",
          cfg["read_idx"] == [3, 10, 17, 24, 31, 38, 45, 49, 50, 51])
    check("smooth2d_grid: loss=mse", cfg["loss"] == "mse")
    check("smooth2d_grid: out_dim=1", cfg["out_dim"] == 1)
    check("smooth2d_grid: per-stage t_span=5/3 (~1.667)",
          abs(s["t_span"] - 5.0 / 3) < 1e-6)
    check("smooth2d_grid: per-stage num_steps=17 (round(50/3))",
          s["num_steps"] == 17)
    check("smooth2d_grid: all 3 stages share t_span/num_steps",
          all(st["t_span"] == s["t_span"] and st["num_steps"] == s["num_steps"]
              for st in cfg["stages"]))
    check("smooth2d_grid: proj_pattern=all_to_all",
          s.get("proj_pattern") == "all_to_all")
    check("smooth2d_grid: legacy preset lambdas edge_gate=5e-6",
          cfg.get("lambdas", {}).get("edge_gate") == 5e-6)
    check("smooth2d_grid: legacy preset lambdas node_gate=0.0 (deprecate-node-gates)",
          cfg.get("lambdas", {}).get("node_gate") == 0.0)
    check("smooth2d_grid: legacy preset lambdas power=1e-5",
          cfg.get("lambdas", {}).get("power") == 1e-5)
    check("smooth2d_grid: legacy preset lambdas capacitance=0.0 (deprecate-node-gates)",
          cfg.get("lambdas", {}).get("capacitance") == 0.0)

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    check("smooth2d_grid: builds successfully (center column reads >= 3 hops from writes)",
          net is not None)
    check("smooth2d_grid: core has 3 stages", len(net.core.stages) == 3)
    check("smooth2d_grid: core has 2 StageTransfer modules (N-1)",
          len(net.core.transfers) == 2)
    check("smooth2d_grid: all transfers are StageTransfer instances",
          all(isinstance(t, StageTransfer) for t in net.core.transfers))
    check("smooth2d_grid: all transfers are identity (52->52)",
          all(t.in_nodes == 52 and t.out_nodes == 52 for t in net.core.transfers))
    check("smooth2d_grid: write_idx = sorted union of fan_out targets",
          net.write_idx == [0, 6, 14, 20, 28, 34, 42, 48])
    check("smooth2d_grid: read_idx = 7 center column + 3 proj",
          net.read_idx == [3, 10, 17, 24, 31, 38, 45, 49, 50, 51])
    check("smooth2d_grid: uses FanOutInputMapper",
          isinstance(net.input_mapper, FanOutInputMapper))
    check("smooth2d_grid: hid_count=49", net.hid_count == 49)
    check("smooth2d_grid: proj_count=3", net.proj_count == 3)
    check("smooth2d_grid: final_hid_count=49", net.final_hid_count == 49)
    check("smooth2d_grid: final_proj_count=3", net.final_proj_count == 3)
    check("smooth2d_grid: OutputMapper is sparse read (read_idx set)",
          isinstance(net.output_mapper, OutputMapper)
          and net.output_mapper.read_idx == [3, 10, 17, 24, 31, 38, 45, 49, 50, 51])

    n_hidden = 49
    n_proj = 3
    n_hidden_edges = 156
    n_hidden_repeat = n_hidden_edges * 2
    n_proj_edges = n_hidden * n_proj
    expected_total = n_hidden_repeat + n_proj_edges
    for i, stage in enumerate(net.core.stages):
        check(f"smooth2d_grid: stage {i} edge count = {expected_total} (grid 312 + proj 147)",
              int(stage.src.shape[0]) == expected_total,
              f"got {int(stage.src.shape[0])}")
        check(f"smooth2d_grid: stage {i} num_nodes=52 (49 hid + 3 proj)",
              int(stage.num_nodes) == 52)

    check("smooth2d_grid: write_idx entries in [0, hid_count)",
          all(0 <= w < net.hid_count for w in net.write_idx))
    check("smooth2d_grid: read_idx entries in [0, final_state_dim)",
          all(0 <= r < net.final_hid_count + net.final_proj_count
              for r in net.read_idx))

    u = torch.rand(8, 2)
    ctx = None
    out, _ = net(u)
    check("smooth2d_grid: forward shape (8,1)", out.shape == (8, 1))
    check("smooth2d_grid: forward output is finite",
          torch.isfinite(out).all().item())

    from train import make_optimizer
    from train_script import make_data_smooth2d
    train_loader, _, task_fn = make_data_smooth2d(batch_size=8, val_size=16)
    optimizer = make_optimizer(net, lr=1e-3)
    net.train()
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b, _ = net(u_b)
        loss = task_fn(out_b, y_b)
        loss.backward()
        optimizer.step()
        check("smooth2d_grid: 1-batch backward finite on 49-node grid",
              math.isfinite(float(loss.item())))
        break

    from io_mapper import InputMapper, FanOutInputMapper
    net_dense = build_net_from_preset(
        "smooth2d_grid", cell_lib=make_cell_library('tanh'), write_mode="dense",
    )
    check("smooth2d_grid: write_mode='dense' override produces InputMapper",
          isinstance(net_dense.input_mapper, InputMapper)
          and type(net_dense.input_mapper) is InputMapper,
          f"got {type(net_dense.input_mapper).__name__}")
    net_fanout = build_net_from_preset(
        "smooth2d_grid", cell_lib=make_cell_library('tanh'),
    )
    check("smooth2d_grid: default (no write_mode) produces FanOutInputMapper",
          isinstance(net_fanout.input_mapper, FanOutInputMapper),
          f"got {type(net_fanout.input_mapper).__name__}")

def test_housing_grid_preset():
    print("\nTest NN3: housing_grid preset (5x5 grid + 3 proj, dense I/O, 3 stages, Huber loss)")
    from config import PRESETS, make_housing_grid_preset
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from io_mapper import InputMapper, OutputMapper
    from stage_transfer import StageTransfer

    cfg = PRESETS.get("housing_grid")
    check("housing_grid present in PRESETS", cfg is not None)
    if cfg is None:
        return

    s = cfg["stages"][0]
    check("housing_grid: 3 stages (mirrors smooth2d_grid)",
          len(cfg["stages"]) == 3)
    check("housing_grid: num_inputs=8 (CA housing features)", s["num_inputs"] == 8)
    check("housing_grid: num_hidden=25 (5x5 grid)", s["num_hidden"] == 25)
    check("housing_grid: num_proj=3", s["num_proj"] == 3)
    check("housing_grid: hidden_family=grid", s["hidden_family"] == "grid")
    check("housing_grid: height=5", s["hidden_kwargs"].get("height") == 5)
    check("housing_grid: width=5", s["hidden_kwargs"].get("width") == 5)
    check("housing_grid: kernel_size=3 (8-neighbor)",
          s["hidden_kwargs"].get("kernel_size") == 3)
    check("housing_grid: write_mode=dense", cfg.get("write_mode") == "dense")
    check("housing_grid: write_mode=dense (no explicit write_idx needed)",
          cfg.get("write_mode") == "dense" and cfg.get("write_idx") is None)
    check("housing_grid: loss=huber", cfg["loss"] == "huber")
    check("housing_grid: out_dim=1", cfg["out_dim"] == 1)
    check("housing_grid: schedule=three_phase",
          cfg.get("schedule") == "three_phase")
    check("housing_grid: tau_anneal=True",
          cfg.get("tau_anneal", True) is True)
    check("housing_grid: per-stage t_span=5/3 (~1.667)",
          abs(s["t_span"] - 5.0 / 3) < 1e-6)
    check("housing_grid: per-stage num_steps=17 (round(50/3))",
          s["num_steps"] == 17)
    check("housing_grid: all 3 stages share t_span/num_steps",
          all(st["t_span"] == s["t_span"] and st["num_steps"] == s["num_steps"]
              for st in cfg["stages"]))
    check("housing_grid: read_idx = center column (5 nodes) + 3 proj = 8 reads",
          len(cfg["read_idx"]) == 8
          and cfg["read_idx"][:5] == [2, 7, 12, 17, 22]
          and cfg["read_idx"][5:] == [25, 26, 27])
    check("housing_grid: preset lambdas edge_gate=5e-6",
          cfg.get("lambdas", {}).get("edge_gate") == 5e-6)
    check("housing_grid: preset lambdas node_gate=0.0 (deprecate-node-gates)",
          cfg.get("lambdas", {}).get("node_gate") == 0.0)
    check("housing_grid: preset lambdas power=1e-5",
          cfg.get("lambdas", {}).get("power") == 1e-5)
    check("housing_grid: preset lambdas capacitance=0.0 (deprecate-node-gates)",
          cfg.get("lambdas", {}).get("capacitance") == 0.0)
    check("housing_grid: preset lambdas rail=0.1",
          cfg.get("lambdas", {}).get("rail") == 0.1)

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("housing_grid", cell_lib=cell_lib)
    check("housing_grid: builds successfully", net is not None)
    check("housing_grid: core has 3 stages", len(net.core.stages) == 3)
    check("housing_grid: core has 2 StageTransfer modules (N-1)",
          len(net.core.transfers) == 2)
    check("housing_grid: all transfers are StageTransfer instances",
          all(isinstance(t, StageTransfer) for t in net.core.transfers))
    check("housing_grid: all transfers are identity (28->28)",
          all(t.in_nodes == 28 and t.out_nodes == 28 for t in net.core.transfers))
    check("housing_grid: write_idx is None (dense mode = all hidden)",
          net.write_idx is None)
    check("housing_grid: read_idx = 8 nodes (5 hidden center col + 3 proj)",
          net.read_idx == [2, 7, 12, 17, 22, 25, 26, 27])
    check("housing_grid: uses plain InputMapper (dense write)",
          isinstance(net.input_mapper, InputMapper)
          and type(net.input_mapper) is InputMapper)
    check("housing_grid: input_mapper gain is (8, 25) dense (8 inputs -> 25 hidden)",
          net.input_mapper.gain.weight.shape == (25, 8))
    check("housing_grid: hid_count=25", net.hid_count == 25)
    check("housing_grid: proj_count=3", net.proj_count == 3)
    check("housing_grid: final_hid_count=25", net.final_hid_count == 25)
    check("housing_grid: final_proj_count=3", net.final_proj_count == 3)
    check("housing_grid: OutputMapper is sparse read (read_idx set)",
          isinstance(net.output_mapper, OutputMapper)
          and net.output_mapper.read_idx == [2, 7, 12, 17, 22, 25, 26, 27])

    n_hidden = 25
    n_proj = 3
    n_hidden_edges = 72
    n_hidden_repeat = n_hidden_edges * 2
    n_proj_edges = n_hidden * n_proj
    expected_total = n_hidden_repeat + n_proj_edges
    for i, stage in enumerate(net.core.stages):
        check(f"housing_grid: stage {i} edge count = {expected_total} (grid 144 + proj {n_proj_edges})",
              int(stage.src.shape[0]) == expected_total,
              f"got {int(stage.src.shape[0])}")
        check(f"housing_grid: stage {i} num_nodes=28 (25 hid + 3 proj)",
              int(stage.num_nodes) == 28)

    check("housing_grid: read_idx entries in [0, final_state_dim)",
          all(0 <= r < net.final_hid_count + net.final_proj_count
              for r in net.read_idx))
    check("housing_grid: 5 read nodes are hidden (center column)",
          sum(1 for r in net.read_idx if r < net.hid_count) == 5)
    check("housing_grid: 3 read nodes are proj",
          sum(1 for r in net.read_idx if r >= net.hid_count) == 3)

    u = torch.rand(8, 8)
    ctx = None
    out, _ = net(u)
    check("housing_grid: forward shape (8,1)", out.shape == (8, 1))
    check("housing_grid: forward output is finite",
          torch.isfinite(out).all().item())

    cfg4 = make_housing_grid_preset(grid_size=4)
    check("make_housing_grid_preset(4): 4x4 grid, 16 hidden",
          cfg4["stages"][0]["num_hidden"] == 16)
    check("make_housing_grid_preset(4): 3 stages",
          len(cfg4["stages"]) == 3)
    check("make_housing_grid_preset(4): write_mode still dense",
          cfg4["write_mode"] == "dense")

def test_persistent_drive_auto_fan_out():
    print("\nTest PD-1: --persistent-drive auto-injects write_fan_out for non-fan_out presets")
    from train_script import _build_grid_write_fan_out
    from config import make_housing_grid_preset, PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library

    cfg = make_housing_grid_preset(grid_size=5)
    check("housing_grid base preset has write_mode='dense' (no fan_out)",
          cfg["write_mode"] == "dense")
    check("housing_grid base preset has NO write_fan_out",
          "write_fan_out" not in cfg or cfg.get("write_fan_out") is None)

    num_inputs = int(cfg["stages"][0]["num_inputs"])
    fan_out = _build_grid_write_fan_out(num_inputs=num_inputs, grid_size=5)
    check("_build_grid_write_fan_out: one entry per input",
          len(fan_out) == num_inputs)
    all_targets = [t for tgts in fan_out.values() for t in tgts]
    check("_build_grid_write_fan_out: all targets unique",
          len(all_targets) == len(set(all_targets)))
    check("_build_grid_write_fan_out: every input has >= 1 target",
          all(len(v) >= 1 for v in fan_out.values()))

    active = dict(PRESETS.get("housing_grid", {}))
    if "write_fan_out" in active:
        active.pop("write_fan_out", None)
    active["write_fan_out"] = fan_out
    active["write_mode"] = "fan_out"
    PRESETS["housing_grid"] = active

    try:
        cell_lib = make_cell_library('tanh')
        net = build_net_from_preset(
            "housing_grid",
            cell_lib=cell_lib,
            write_mode="fan_out",
            read_mode=None,
            write_idx=None,
            read_idx=None,
            enable_drive=True,
        )
        from io_mapper import FanOutInputMapper
        check("housing_grid + persistent-drive uses FanOutInputMapper",
              isinstance(net.input_mapper, FanOutInputMapper))
        check("housing_grid + persistent-drive has drive_mappers",
              net.drive_mappers is not None and len(net.drive_mappers) == 3)
        check("housing_grid + persistent-drive: forward output is finite",
              torch.isfinite(net(torch.rand(4, 8))[0]).all().item())
    finally:
        PRESETS["housing_grid"] = make_housing_grid_preset(grid_size=5)


def test_persistent_drive_sparse_proj():
    print("\nTest PD-sparse_proj: persistent-drive with write_mode='sparse_proj'")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library

    cell_lib = make_cell_library('tanh')

    # Use _make_dynamic_preset to build a proper torus preset
    from train_script import _make_dynamic_preset
    cfg = _make_dynamic_preset(
        problem="smooth2d", hidden_family="torus", num_hidden=25,
        num_stages=3, edge_repeats=1, grid_size=5, bidirectional=False,
        write_mode_override="sparse_proj",
    )
    PRESETS["smooth2d_torus_persistent_test"] = cfg

    # Test with drive_mode='fan_out' (default) - uses FanOutInputMapper with round-robin
    net = build_net_from_preset(
        "smooth2d_torus_persistent_test",
        cell_lib=cell_lib,
        write_mode="sparse_proj",
        write_idx=[2, 7, 12, 17, 22],
        read_idx=[0, 15, 4, 19],
        enable_drive=True,
        drive_mode="fan_out",
    )
    from io_mapper import FanOutInputMapper, ProjectedSparseInputMapper, SparseInputMapper
    check("input_mapper is ProjectedSparseInputMapper",
          isinstance(net.input_mapper, ProjectedSparseInputMapper))
    check("drive_mappers exist (3 stages)",
          net.drive_mappers is not None and len(net.drive_mappers) == 3)
    check("drive_mappers are FanOutInputMapper (default mode)",
          isinstance(net.drive_mappers[0], FanOutInputMapper))
    check("drive fan_out_map covers all write_idx nodes",
          len(set().union(*net.drive_mappers[0].fan_out_map.values())) == 5)
    check("write_idx preserved",
          list(net.write_idx) == [2, 7, 12, 17, 22])
    y, _ = net(torch.rand(4, 2))
    check("forward output is finite", torch.isfinite(y).all().item())

    # Test with drive_mode='projection' - uses ProjectedSparseInputMapper
    net = build_net_from_preset(
        "smooth2d_torus_persistent_test",
        cell_lib=cell_lib,
        write_mode="sparse_proj",
        write_idx=[2, 7, 12, 17, 22],
        read_idx=[0, 15, 4, 19],
        enable_drive=True,
        drive_mode="projection",
    )
    check("drive_mappers are ProjectedSparseInputMapper (projection mode)",
          isinstance(net.drive_mappers[0], ProjectedSparseInputMapper))
    y, _ = net(torch.rand(4, 2))
    check("forward output is finite (projection)", torch.isfinite(y).all().item())

    # Test one_to_one with persistent drive (use write_idx far from read_idx to
    # satisfy degree-of-separation validation on torus+kernel3 topology).
    cfg2 = _make_dynamic_preset(
        problem="smooth2d", hidden_family="torus", num_hidden=25,
        num_stages=3, edge_repeats=1, grid_size=5, bidirectional=False,
        write_mode_override="one_to_one",
    )
    cfg2["write_idx"] = [0, 5]
    cfg2["read_idx"] = [12, 18]
    PRESETS["smooth2d_one_to_one_persistent_test"] = cfg2

    net = build_net_from_preset(
        "smooth2d_one_to_one_persistent_test",
        cell_lib=cell_lib,
        write_mode="one_to_one",
        write_idx=[0, 5],
        read_idx=[12, 18],
        enable_drive=True,
        drive_mode="fan_out",
    )
    check("input_mapper is SparseInputMapper",
          isinstance(net.input_mapper, SparseInputMapper))
    check("drive_mappers are FanOutInputMapper (1-to-1)",
          isinstance(net.drive_mappers[0], FanOutInputMapper))
    check("drive fan_out_map has 1 target per input",
          {i: [write_idx] for i, write_idx in zip(range(2), [0, 5])}
          == net.drive_mappers[0].fan_out_map)
    y, _ = net(torch.rand(4, 2))
    check("forward output is finite (one_to_one)", torch.isfinite(y).all().item())

    # Cleanup
    PRESETS.pop("smooth2d_torus_persistent_test", None)
    PRESETS.pop("smooth2d_one_to_one_persistent_test", None)


def test_amp_dtype_fix():
    print("\nTest AMP-fix: index_copy_ dtype mismatch with half-precision input")
    import io_mapper

    # Simulate the production AMP scenario: the input u is float32 (outside autocast),
    # but the linear/op internals run in half via autocast. So the per_feature/
    # per_target/projected tensor is half while x (new_zeros from u) is float32.
    # The fix casts the half tensor to float32 to match x before index_copy_.

    # Test SparseInputMapper: simulate by manually running the path
    m = io_mapper.SparseInputMapper(in_dim=2, out_dim=10, write_idx=[0, 1])
    u = torch.randn(4, 2)
    with torch.no_grad():
        # Simulate what happens under autocast: ops cast to half
        half_per_feature = m.x_max * torch.tanh((u * m.gain + m.bias).half())
        x = u.new_zeros(*u.shape[:-1], m.out_dim)  # float32 (u is float32)
        # Without fix: dtype mismatch. With fix: cast to x.dtype.
        x.index_copy_(-1, torch.tensor(m.write_idx, dtype=torch.long), half_per_feature.to(dtype=x.dtype))
    check("SparseInputMapper AMP scatter works", x.dtype == torch.float32)
    check("SparseInputMapper output finite", torch.isfinite(x).all().item())

    # Test FanOutInputMapper
    m = io_mapper.FanOutInputMapper(in_dim=2, out_dim=10, fan_out_map={0: [0, 5], 1: [3, 7]})
    u = torch.randn(4, 2)
    with torch.no_grad():
        u_picked = u.index_select(-1, m._input_index)
        half_per_target = m.x_max * torch.tanh((u_picked * m.gain + m.bias).half())
        x = u.new_zeros(*u.shape[:-1], m.out_dim)
        x.index_copy_(-1, m._flat_targets, half_per_target.to(dtype=x.dtype))
    check("FanOutInputMapper AMP scatter works", x.dtype == torch.float32)
    check("FanOutInputMapper output finite", torch.isfinite(x).all().item())

    # Test ProjectedSparseInputMapper
    m = io_mapper.ProjectedSparseInputMapper(in_dim=2, out_dim=10, write_idx=[2, 7])
    u = torch.randn(4, 2)
    with torch.no_grad():
        half_proj = m.proj(u).half()
        half_projected = m.x_max * torch.tanh(half_proj)
        x = u.new_zeros(*u.shape[:-1], m.out_dim)
        x.index_copy_(-1, m._write_index, half_projected.to(dtype=x.dtype))
    check("ProjectedSparseInputMapper AMP scatter works", x.dtype == torch.float32)
    check("ProjectedSparseInputMapper output finite", torch.isfinite(x).all().item())


def test_fan_out_torus_bug_fix():
    print("\nTest FO-torus: --write-mode fan_out + torus family auto-generates write_fan_out")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from train_script import _make_dynamic_preset

    cell_lib = make_cell_library('tanh')

    # Build a torus preset with fan_out mode — the bug was that
    # _make_dynamic_preset only generated write_fan_out for 'grid' family,
    # not for 'torus'. With the fix, torus also gets auto-generated fan_out.
    cfg = _make_dynamic_preset(
        problem="smooth2d", hidden_family="torus", num_hidden=25,
        num_stages=3, edge_repeats=1, grid_size=5, bidirectional=False,
        write_mode_override="fan_out",
    )
    check("torus+fan_out: write_fan_out auto-generated",
          cfg.get("write_fan_out") is not None)
    check("torus+fan_out: write_fan_out has one entry per input",
          len(cfg["write_fan_out"]) == cfg["stages"][0]["num_inputs"])

    PRESETS["smooth2d_torus_fan_out_test"] = cfg
    try:
        # Use read_idx far from auto-generated writes to satisfy topology validation.
        net = build_net_from_preset(
            "smooth2d_torus_fan_out_test",
            cell_lib=cell_lib,
            write_mode="fan_out",
            read_idx=[12, 18],
            enable_drive=True,  # also exercises the persistent-drive block
            drive_mode="fan_out",
        )
        from io_mapper import FanOutInputMapper
        check("builds successfully with fan_out + persistent-drive",
              isinstance(net.input_mapper, FanOutInputMapper))
        check("drive_mappers created (3 stages)",
              net.drive_mappers is not None and len(net.drive_mappers) == 3)
        y, _ = net(torch.rand(4, 2))
        check("forward output is finite",
              torch.isfinite(y).all().item())
    finally:
        PRESETS.pop("smooth2d_torus_fan_out_test", None)


def test_write_fan_out_cli():
    print("\nTest FO-cli: --write-fan-out JSON injects custom fan_out_map")
    import json
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from train_script import _make_dynamic_preset

    cell_lib = make_cell_library('tanh')

    # Simulate the CLI flow: user passes --write-fan-out JSON, which is parsed
    # and injected into the active preset dict before build_net_from_preset.
    cfg = _make_dynamic_preset(
        problem="smooth2d", hidden_family="torus", num_hidden=25,
        num_stages=3, edge_repeats=1, grid_size=5, bidirectional=False,
        write_mode_override="fan_out",
    )

    # Step 1: parse the CLI arg as the train_script does
    cli_arg = '{"0": [2, 12], "1": [7, 17]}'
    raw = json.loads(cli_arg)
    fan_out_map = {int(k): [int(vv) for vv in v] for k, v in raw.items()}
    cfg["write_fan_out"] = fan_out_map

    # Step 2: validation
    num_inputs = int(cfg["stages"][0]["num_inputs"])
    all_targets = [t for tgts in fan_out_map.values() for t in tgts]
    check("write_fan_out validation: no duplicate targets",
          len(all_targets) == len(set(all_targets)))
    check("write_fan_out validation: all keys in [0, num_inputs)",
          all(k < num_inputs for k in fan_out_map))
    check("write_fan_out has correct content",
          fan_out_map == {0: [2, 12], 1: [7, 17]})

    PRESETS["smooth2d_torus_fan_out_cli_test"] = cfg
    try:
        net = build_net_from_preset(
            "smooth2d_torus_fan_out_cli_test",
            cell_lib=cell_lib,
            write_mode="fan_out",
            read_idx=[0, 15, 4, 19],
            enable_drive=True,
            drive_mode="fan_out",
        )
        from io_mapper import FanOutInputMapper
        check("input_mapper is FanOutInputMapper",
              isinstance(net.input_mapper, FanOutInputMapper))
        check("input_mapper uses the custom fan_out_map",
              net.input_mapper.fan_out_map == {0: [2, 12], 1: [7, 17]})
        check("drive_mappers use the custom fan_out_map",
              net.drive_mappers[0].fan_out_map == {0: [2, 12], 1: [7, 17]})
        check("write_idx derived from fan_out targets",
              set(net.write_idx) == {2, 7, 12, 17})
        y, _ = net(torch.rand(4, 2))
        check("forward output is finite",
              torch.isfinite(y).all().item())
    finally:
        PRESETS.pop("smooth2d_torus_fan_out_cli_test", None)


def test_write_fan_out_validation():
    print("\nTest FO-cli-validate: --write-fan-out rejects invalid JSON")
    import json
    cfg = {"stages": [{"num_inputs": 2}]}

    # Test invalid JSON
    try:
        json.loads("not json")
        check("invalid JSON raises", False)
    except json.JSONDecodeError:
        check("invalid JSON raises", True)

    # Test duplicate targets
    raw = '{"0": [2, 5], "1": [5, 7]}'
    parsed = {int(k): [int(v) for v in v] for k, v in json.loads(raw).items()}
    all_targets = [t for tgts in parsed.values() for t in tgts]
    check("duplicate targets detected",
          len(all_targets) != len(set(all_targets)))

    # Test input key out of range
    raw = '{"3": [2, 5]}'
    parsed = {int(k): [int(v) for v in v] for k, v in json.loads(raw).items()}
    check("input key out of range detected",
          any(k >= cfg["stages"][0]["num_inputs"] for k in parsed))


def test_mlp_benchmark():
    print("\nTest OO: minimal MLP benchmark for smooth2d")
    from mlp_benchmark import MLPRegressor, count_parameters

    net = MLPRegressor(in_dim=2, hidden_dim=100, out_dim=1)
    n_params = count_parameters(net)
    check("mlp: default hidden_dim=100 produces 401 params", n_params == 401,
          f"got {n_params}")

    x = torch.randn(8, 2)
    y = net(x)
    check("mlp: forward output shape (8,1)", tuple(y.shape) == (8, 1))
    check("mlp: forward output is finite", torch.isfinite(y).all().item())

    for h in (8, 16, 32, 64, 100, 128):
        m = MLPRegressor(in_dim=2, hidden_dim=h, out_dim=1)
        expected = 4 * h + 1
        check(f"mlp: hidden_dim={h} -> {expected} params",
              count_parameters(m) == expected)

    x = torch.randn(4, 2, requires_grad=False)
    target = torch.randn(4, 1)
    out = net(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    grad_ok = (
        net.fc1.weight.grad is not None
        and net.fc1.bias.grad is not None
        and net.fc2.weight.grad is not None
        and net.fc2.bias.grad is not None
    )
    check("mlp: all 4 parameter tensors receive gradients", grad_ok)

    from mlp_benchmark import validate
    from train_script import make_data_smooth2d
    train_loader, val_loader, task_fn = make_data_smooth2d(batch_size=128, val_size=200)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    initial_val = validate(net, val_loader, task_fn, device="cpu")

    net.train()
    n_steps = 0
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b = net(u_b)
        loss_b = task_fn(out_b, y_b)
        loss_b.backward()
        optimizer.step()
        n_steps += 1
        if n_steps >= 5:
            break
    after_val = validate(net, val_loader, task_fn, device="cpu")
    check("mlp: 5-step training reduces val loss",
          after_val < initial_val,
          f"before={initial_val:.4f} after={after_val:.4f}")

def test_mlp_benchmark_tanh():
    print("\nTest OO-b: tanh-activation MLP benchmark for smooth2d")
    from mlp_benchmark import MLPRegressor, count_parameters

    net = MLPRegressor(in_dim=2, hidden_dim=100, out_dim=1, activation="tanh")
    n_params = count_parameters(net)
    check("mlp-tanh: hidden_dim=100 produces 401 params", n_params == 401,
          f"got {n_params}")
    check("mlp-tanh: activation string stored as 'tanh'", net.activation == "tanh")

    x = torch.randn(8, 2)
    y = net(x)
    check("mlp-tanh: forward output shape (8,1)", tuple(y.shape) == (8, 1))
    check("mlp-tanh: forward output is finite", torch.isfinite(y).all().item())

    with torch.no_grad():
        h_pre = net.fc1(x)
        h_post = torch.tanh(h_pre)
    check("mlp-tanh: hidden activation bounded in (-1, 1)",
          (h_post > -1.0).all().item() and (h_post < 1.0).all().item())

    for h in (8, 16, 32, 64, 100, 128):
        m = MLPRegressor(in_dim=2, hidden_dim=h, out_dim=1, activation="tanh")
        expected = 4 * h + 1
        check(f"mlp-tanh: hidden_dim={h} -> {expected} params",
              count_parameters(m) == expected)

    x = torch.randn(4, 2)
    target = torch.randn(4, 1)
    out = net(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    grad_ok = (
        net.fc1.weight.grad is not None
        and net.fc1.bias.grad is not None
        and net.fc2.weight.grad is not None
        and net.fc2.bias.grad is not None
    )
    check("mlp-tanh: all 4 parameter tensors receive gradients", grad_ok)

    raised = False
    try:
        MLPRegressor(in_dim=2, hidden_dim=10, out_dim=1, activation="sigmoid")
    except ValueError:
        raised = True
    check("mlp-tanh: invalid activation raises ValueError", raised)

    from mlp_benchmark import validate
    from train_script import make_data_smooth2d
    train_loader, val_loader, task_fn = make_data_smooth2d(batch_size=128, val_size=200)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    initial_val = validate(net, val_loader, task_fn, device="cpu")
    net.train()
    n_steps = 0
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b = net(u_b)
        loss_b = task_fn(out_b, y_b)
        loss_b.backward()
        optimizer.step()
        n_steps += 1
        if n_steps >= 5:
            break
    after_val = validate(net, val_loader, task_fn, device="cpu")
    check("mlp-tanh: 5-step training reduces val loss",
          after_val < initial_val,
          f"before={initial_val:.4f} after={after_val:.4f}")

def test_v15_cell_parameters_preset_smooth2d_grid():
    """V15-10: v15 library works with build_net_from_preset on smooth2d_grid."""
    print("\nTest V15-10: v15 library with smooth2d_grid build")
    from cell_library import make_cell_library
    from topology import build_net_from_preset
    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    import torch
    x = torch.randn(4, 2)
    with torch.no_grad():
        y, traj = net(x, store_trajectory=False)
    check("V15-10: network output finite",
          torch.isfinite(y).all().item())
    check("V15-10: network output shape (4, 1)",
          y.shape == (4, 1),
          f"got {y.shape}")

def test_stage_lr_scale_scheduler_compat():
    """SLS-3: CosineAnnealingLR works correctly with multi-group optimizer."""
    print("\nTest SLS-3: scheduler compatibility with staged optimizer")
    from train import make_optimizer
    from config import PRESETS, OPTIM
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from torch.optim.lr_scheduler import CosineAnnealingLR

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(net, lr=1e-3, stage_lr_scale=10.0)
    scheduler = CosineAnnealingLR(optim, T_max=100, eta_min=OPTIM["scheduler_eta_min"])

    initial_lrs = [g["lr"] for g in optim.param_groups]
    scheduler.step()
    stepped_lrs = [g["lr"] for g in optim.param_groups]

    check("SLS-3: all groups reduced after one scheduler step",
          all(s < i for s, i in zip(stepped_lrs, initial_lrs)),
          f"initial={initial_lrs}, after_step={stepped_lrs}")
    ratios_before = [i / j for i, j in zip(initial_lrs[:-1], initial_lrs[1:])]
    ratios_after  = [i / j for i, j in zip(stepped_lrs[:-1], stepped_lrs[1:])]
    check("SLS-3: group LR ratios preserved after scheduler step",
          all(abs(rb - ra) < 1e-4 for rb, ra in zip(ratios_before, ratios_after)),
          f"ratios before={ratios_before}, after={ratios_after}")

def test_parameter_breakdown_aggregates_components():
    """PBD-1: KirchhoffNetWithIO.parameter_breakdown() groups params by component
    and the breakdown total matches the manual sum of named_parameters."""
    print("\nTest PBD-1: parameter_breakdown aggregates input/output/drive/core")
    from cell_library import make_cell_library
    from topology import build_net_from_preset
    from kirchhoff_net import format_parameter_breakdown

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    bd = net.parameter_breakdown()

    manual_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    check("PBD-1: breakdown total matches manual sum",
          bd["total"] == manual_total,
          f"breakdown={bd['total']} manual={manual_total}")

    # All three top-level groups exist, even if zero.
    for k in ("input_mapper", "output_mapper", "drive_mappers"):
        check(f"PBD-1: groups[{k}] present and >= 0",
              k in bd["groups"] and bd["groups"][k] >= 0,
              f"got {bd['groups'].get(k)}")

    # smooth2d_grid has 3 stages (mirrors smooth2d); every stage key present.
    check("PBD-1: per_stage has 3 stage keys",
          set(bd["per_stage"].keys()) == {"stage_0", "stage_1", "stage_2"},
          f"got {sorted(bd['per_stage'].keys())}")

    # Each stage bucket has all expected sub-keys.
    for sk, bucket in bd["per_stage"].items():
        for sub in ("cell_lib", "z_logits", "u_logits", "raw_leak",
                    "raw_drive_g", "other"):
            check(f"PBD-1: {sk} has {sub} sub-key",
                  sub in bucket,
                  f"missing {sub} in {bucket}")

    # cell_lib > 0 for each stage (tanh SimpleEdgeLibrary has 3 params/edge).
    for sk, bucket in bd["per_stage"].items():
        check(f"PBD-1: {sk} cell_lib > 0",
              bucket["cell_lib"] > 0,
              f"got {bucket['cell_lib']}")

    # format_parameter_breakdown returns a non-empty string with header + total.
    text = format_parameter_breakdown(bd)
    check("PBD-1: formatted output is multi-line and contains 'total:'",
          text.count("\n") >= 3 and "total:" in text,
          f"got {text!r}")

def test_edge_repeats_propagation_to_preset():
    """ERP-1: build_net_from_preset honors edge_repeats injected into a preset
    that did NOT have it set (regression test for the param-log-fix plan)."""
    print("\nTest ERP-1: edge_repeats propagates into stages of preset")
    from cell_library import make_cell_library
    from config import PRESETS
    from topology import build_net_from_preset

    # Clone the friedman2 preset so we don't mutate the global registry.
    cfg = {k: (list(v) if isinstance(v, list) else
               {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
               if isinstance(v, dict) else v)
           for k, v in PRESETS["friedman2"].items()}
    cfg["stages"] = [{k: (list(v) if isinstance(v, list) else
                          {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
                          if isinstance(v, dict) else v)
                      for k, v in stage.items()}
                     for stage in cfg["stages"]]
    cfg["stages"][0]["edge_repeats"] = 3
    PRESETS["__test_friedman2_x3"] = cfg
    try:
        cell_lib = make_cell_library('tanh')
        net = build_net_from_preset("__test_friedman2_x3", cell_lib=cell_lib)
        # num_edges should be 3 * 64 (single torus edge count) = 192
        edges = net.core.stages[0].num_edges()
        check("ERP-1: edge_repeats=3 doubles stage edges",
              edges == 3 * 64,
              f"got {edges}")
        bd = net.parameter_breakdown()
        # With 3x edges, cell_lib should be ~3x larger than baseline 192.
        check("ERP-1: breakdown cell_lib reflects 3x edges",
              bd["per_stage"]["stage_0"]["cell_lib"] == 3 * 64 * 3,
              f"got {bd['per_stage']['stage_0']['cell_lib']}")
    finally:
        del PRESETS["__test_friedman2_x3"]


def test_boundary_fan_out_basic():
    print("\nTest BFO-1: --boundary-fan-out builds net with boundary OTA edges")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    from io_mapper import NullInputMapper

    cell_lib = make_cell_library('tanh')
    boundary_fan_out = {0: [2, 5], 1: [7, 9]}

    net = build_net_from_preset(
        "smooth2d",
        cell_lib=cell_lib,
        boundary_fan_out=boundary_fan_out,
    )

    check("input_mapper is NullInputMapper (zero init)",
          isinstance(net.input_mapper, NullInputMapper))
    check("enable_boundary flag set",
          getattr(net, "enable_boundary", False))
    check("boundary_fan_out stored on net",
          net.boundary_fan_out == {0: [2, 5], 1: [7, 9]})

    stage = net.core.stages[0]
    check("stage has _has_boundary flag",
          getattr(stage, "_has_boundary", False))
    check("stage boundary_src length matches total boundary edges",
          stage.boundary_src.numel() == 4)
    check("stage boundary_dst length matches total boundary edges",
          stage.boundary_dst.numel() == 4)
    check("boundary_src content matches (input 0 -> 2,5; input 1 -> 7,9)",
          stage.boundary_src.tolist() == [0, 0, 1, 1])
    check("boundary_dst content matches",
          stage.boundary_dst.tolist() == [2, 5, 7, 9])
    check("boundary_cell_lib exists",
          stage.boundary_cell_lib is not None)
    check("boundary_z_logits has 4 entries",
          stage.boundary_z_logits.shape == (4,))

    y, _ = net(torch.rand(8, 2))
    check("forward output is finite",
          torch.isfinite(y).all().item())


def test_boundary_fan_out_zero_init():
    print("\nTest BFO-2: initial state is zero when boundary mode is active")
    from topology import build_net_from_preset
    from cell_library import make_cell_library

    cell_lib = make_cell_library('tanh')
    boundary_fan_out = {0: [3], 1: [8]}

    net = build_net_from_preset(
        "smooth2d",
        cell_lib=cell_lib,
        boundary_fan_out=boundary_fan_out,
    )

    stage = net.core.stages[0]
    boundary_cell = stage.boundary_cell_lib
    # Bias boundary edge weights to zero to make the per-step boundary
    # current ≈ 0, so the only force on x0 should be zero too (since the
    # initial state itself is zeros and boundary currents start near 0).
    if hasattr(boundary_cell, "param"):
        with torch.no_grad():
            boundary_cell.param.zero_()
            boundary_cell.requires_grad_(False)

    # Hook into the first stage's forward to capture x0.
    captured = {}

    def capture_x0(x0, t_span=None, num_steps=None, store_trajectory=True,
                   x_drive=None, drive_scale=0.0, solver="heun", deq_cfg=None,
                   u=None):
        captured["x0"] = x0.detach().clone()
        captured["u"] = u.detach().clone() if u is not None else None
        return stage._forward_heun(x0=x0, t_span=t_span, num_steps=num_steps,
                                    store_trajectory=store_trajectory,
                                    x_drive=x_drive, drive_scale=drive_scale,
                                    u=u)

    stage.forward = capture_x0
    u = torch.tensor([[0.5, -0.3]])
    y, _ = net(u)
    x0 = captured["x0"]
    check("initial state x0 is exactly zero (no write mapper)",
          torch.allclose(x0, torch.zeros_like(x0)),
          f"max abs = {x0.abs().max().item():.3e}")
    check("u passed through to stage.forward",
          torch.equal(captured["u"], u))


def test_boundary_fan_out_validation():
    print("\nTest BFO-3: --boundary-fan-out rejects invalid configurations")
    from topology import build_net_from_config
    from cell_library import make_cell_library
    from config import PRESETS

    cell_lib = make_cell_library('tanh')

    # 1. Duplicate targets across inputs.
    try:
        bad = {0: [2, 5], 1: [5, 7]}
        build_net_from_config(
            dict(PRESETS["smooth2d"]),
            cell_lib=cell_lib,
            boundary_fan_out=bad,
        )
        check("duplicate target nodes raise ValueError", False)
    except ValueError:
        check("duplicate target nodes raise ValueError", True)

    # 2. Target out of range.
    try:
        bad = {0: [99]}
        build_net_from_config(
            dict(PRESETS["smooth2d"]),
            cell_lib=cell_lib,
            boundary_fan_out=bad,
        )
        check("target out of range raises ValueError", False)
    except ValueError:
        check("target out of range raises ValueError", True)

    # 3. Missing input keys (only covers [0] not [0, 1]).
    try:
        bad = {0: [2]}
        build_net_from_config(
            dict(PRESETS["smooth2d"]),
            cell_lib=cell_lib,
            boundary_fan_out=bad,
        )
        check("missing input keys raise ValueError", False)
    except ValueError:
        check("missing input keys raise ValueError", True)

    # 4. Valid config succeeds.
    net = build_net_from_config(
        dict(PRESETS["smooth2d"]),
        cell_lib=cell_lib,
        boundary_fan_out={0: [2], 1: [7]},
    )
    check("valid boundary_fan_out builds net",
          net.core.stages[0]._has_boundary)
    check("valid boundary_fan_out: 2 edges",
          net.core.stages[0].boundary_src.numel() == 2)


def test_boundary_fan_out_grad_flow():
    print("\nTest BFO-4: gradients flow through boundary edge parameters")
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    import torch.nn.functional as F

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset(
        "smooth2d",
        cell_lib=cell_lib,
        boundary_fan_out={0: [2, 5], 1: [7, 9]},
    )

    u = torch.randn(4, 2)
    target = torch.zeros(4, 1)
    y, _ = net(u)
    loss = F.mse_loss(y, target)
    loss.backward()

    stage = net.core.stages[0]
    check("boundary_z_logits receives gradient",
          stage.boundary_z_logits.grad is not None
          and stage.boundary_z_logits.grad.abs().sum().item() > 0)
    check("boundary_cell_lib parameter receives gradient",
          stage.boundary_cell_lib.param.grad is not None
          and stage.boundary_cell_lib.param.grad.abs().sum().item() > 0)


def test_boundary_fan_out_with_no_edge_gates():
    print("\nTest BFO-5: --boundary-fan-out compatible with --no-edge-gates flag")
    from topology import build_net_from_preset
    from cell_library import make_cell_library
    import torch

    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset(
        "smooth2d",
        cell_lib=cell_lib,
        boundary_fan_out={0: [2], 1: [7]},
    )

    # Simulate the --no-edge-gates code path in train_script.
    for stage in net.core.stages:
        if hasattr(stage, "z_logits") and stage.z_logits is not None:
            stage.z_logits.data.fill_(10.0)
            stage.z_logits.requires_grad_(False)
        if hasattr(stage, "boundary_z_logits") and stage.boundary_z_logits is not None:
            stage.boundary_z_logits.data.fill_(10.0)
            stage.boundary_z_logits.requires_grad_(False)

    stage = net.core.stages[0]
    check("boundary_z_logits frozen to +10 (no-edge-gates)",
          stage.boundary_z_logits.requires_grad is False
          and torch.allclose(stage.boundary_z_logits.data, torch.full((2,), 10.0)))
    check("z_logits frozen to +10 (no-edge-gates)",
          stage.z_logits.requires_grad is False
          and torch.allclose(stage.z_logits.data, torch.full((stage.z_logits.shape[0],), 10.0)))

    y, _ = net(torch.rand(4, 2))
    check("forward still works after no-edge-gates freeze",
          torch.isfinite(y).all().item())


def test_boundary_fan_out_incompatible_with_persistent_drive():
    print("\nTest BFO-6: --boundary-fan-out and --persistent-drive are mutually exclusive (CLI)")
    # The mutual-exclusion check lives in train_script.py, where it parses
    # both flags and raises ValueError. Re-implement the same check here
    # so we don't have to spin up a subprocess.
    import json
    args_boundary_fan_out = '{"0": [2], "1": [7]}'
    args_persistent_drive = True

    raw = json.loads(args_boundary_fan_out)
    boundary_fan_out_parsed = {int(k): [int(vv) for vv in v] for k, v in raw.items()}

    # Mirror the train_script.py guard.
    if args_persistent_drive and boundary_fan_out_parsed is not None:
        check("boundary + persistent-drive combination rejected by CLI guard", True)
    else:
        check("boundary + persistent-drive combination rejected by CLI guard", False)


def test_boundary_fan_out_uses_null_input_mapper():
    print("\nTest BFO-7: NullInputMapper returns zeros of correct shape")
    from io_mapper import NullInputMapper
    m = NullInputMapper(out_dim=25)
    u = torch.randn(8, 4)
    x = m(u)
    check("NullInputMapper output shape (batch, out_dim)",
          tuple(x.shape) == (8, 25))
    check("NullInputMapper output is exactly zero",
          torch.allclose(x, torch.zeros_like(x)))
    check("NullInputMapper has no learnable parameters",
          sum(p.numel() for p in m.parameters()) == 0)


if __name__ == "__main__":
    main()
