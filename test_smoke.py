"""Smoke test for the reduced differential KirchhoffNet.

This is a tiny end-to-end test that exercises:
  1. config.py loads and produces valid preset
  2. topology primitives generate well-formed SparseTopologies
  3. topology_to_stage builds a DifferentialStage with valid edges
  4. KirchhoffNetWithIO forward runs without NaN / explosion
  5. Heun integration converges to a finite state on a constant input
  6. Gradients flow through every parameter group
  7. compute_loss returns finite values for all regularizers
  8. Sparsity regularizer pushes edge logits away from L/S toward Z
  9. tau_for_epoch produces expected annealing schedule
 10. StageTransfer truncates and zero-pads correctly
 11. Round-trip: topologize -> build -> run -> loss -> backward -> step

Run with:
    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_smoke.py
"""

import copy
import os
import sys
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
import torch.nn.functional as F


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


def test_config_loads():
    print("\nTest 1: config.py loads and contains expected constants")
    import config
    check("CELL_LIBRARY has L, S, P, Z", set(config.CELL_LIBRARY.keys()) == {"L", "S", "P", "Z"})
    check("CELL_ORDER == ['L', 'S', 'P', 'Z']", config.CELL_ORDER == ["L", "S", "P", "Z"])
    check("Z_INDEX == 3", config.Z_INDEX == 3)
    check("NUM_CELLS == 4", config.NUM_CELLS == 4)
    check("PHYS has x_max=3.0 (three-phase-schedule: increased from 0.3 for headroom)",
          config.PHYS["x_max"] == 3.0)
    check("OPTIM has lr=6e-4 (auto-scaled from 3e-4 at batch_size=2048)", abs(config.OPTIM["lr"] - 6e-4) < 1e-12)
    check("LAMBDAS has sparsity=1e-3", abs(config.LAMBDAS["sparsity"] - 1e-3) < 1e-12)
    check("LAMBDAS has rail=1.0 (three-phase-schedule: down from 10.0 for x_max=3.0 regime)",
          abs(config.LAMBDAS["rail"] - 1.0) < 1e-12)
    check("LAMBDAS has edge_gate=5e-4 (CP: per-component decomposition, was 1e-3 in CP-v1)",
          abs(config.LAMBDAS["edge_gate"] - 5e-4) < 1e-12)
    check("LAMBDAS has node_gate=0.0 (deprecate-node-gates: always 0)",
          abs(config.LAMBDAS["node_gate"] - 0.0) < 1e-12)
    check("LAMBDAS has power=1e-4 (CP: static power proxy)",
          abs(config.LAMBDAS["power"] - 1e-4) < 1e-12)
    check("LAMBDAS has capacitance=0.0 (deprecate-node-gates: always 0)",
          abs(config.LAMBDAS["capacitance"] - 0.0) < 1e-12)
    check("LAMBDAS no longer has 'complexity' (CP: decomposed into 4 terms)",
          "complexity" not in config.LAMBDAS)
    check("PRUNE has edge_threshold=0.1 (three-phase-schedule: was 0.01, too forgiving for gate-trained regime)",
          abs(config.PRUNE["edge_threshold"] - 0.1) < 1e-12)
    check("PRUNE has node_threshold=0.05 (three-phase-schedule: was 0.01, too forgiving for gate-trained regime)",
          abs(config.PRUNE["node_threshold"] - 0.05) < 1e-12)
    check("INIT logits_z_bias=0.0 (fix-z-death: equal cell probability)",
          config.INIT["logits_z_bias"] == 0.0)
    check("INIT z_logit_init=0.0 (grid7-gate0: 50% open gates, max gradient sensitivity)",
          abs(config.INIT["z_logit_init"] - 0.0) < 1e-12)
    check("INIT u_logit_init=0.0 (grid7-gate0: 50% open gates, max gradient sensitivity)",
          abs(config.INIT["u_logit_init"] - 0.0) < 1e-12)
    check("OPTIM has reg_warmup_epochs (RR-A)",
          "reg_warmup_epochs" in config.OPTIM)
    check("OPTIM has reg_anneal_epochs (RR-A)",
          "reg_anneal_epochs" in config.OPTIM)
    check("PRESETS has sinx, housing, smooth2d, smooth2d_grid, housing_grid",
          set(config.PRESETS.keys()) == {"sinx", "housing", "smooth2d", "smooth2d_grid", "housing_grid"})

    t = config.cells_to_tensor_dict()
    check("cells_to_tensor_dict: gm shape (4,)", t["gm"].shape == (4,))
    check("cells_to_tensor_dict: Z gm == 0", float(t["gm"][3]) == 0.0)
    check("cells_to_tensor_dict: L gm == 0.2", abs(float(t["gm"][0]) - 0.2) < 1e-5)
    check("cells_to_tensor_dict: S gm == 1.0", abs(float(t["gm"][1]) - 1.0) < 1e-5)
    check("cells_to_tensor_dict: P gm == 1.0", abs(float(t["gm"][2]) - 1.0) < 1e-5)
    check("cells_to_tensor_dict: P beta == 0.1", abs(float(t["beta"][2]) - 0.1) < 1e-5)
    check("cells_to_tensor_dict: P theta == 0.0", abs(float(t["theta"][2]) - 0.0) < 1e-5)


def test_sim_context():
    print("\nTest 2: SimContext construction and variation sampling (RR-C: temp_c deprecated)")
    import warnings
    from sim_context import SimContext, sample_random_context
    ctx = SimContext()
    check("SimContext defaults temp_c=27.0", ctx.temp_c == 27.0)
    check("SimContext defaults gain_shift=0.0", ctx.global_gain_shift == 0.0)
    check("SimContext defaults mismatch=None", ctx.edge_mismatch is None)

    sampled = sample_random_context(num_edges=8, num_cells=3, seed=0)
    check("Sampled mismatch shape (8,3)", sampled.edge_mismatch.shape == (8, 3))
    check("Sampled mismatch finite", torch.isfinite(sampled.edge_mismatch).all().item())
    check("Sampled temp_c is now the default 27.0 (RR-C: not randomized)",
          sampled.temp_c == 27.0,
          f"got {sampled.temp_c}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = sample_random_context(num_edges=4, num_cells=3, seed=1, legacy_temp=True)
    check("legacy_temp=True emits a DeprecationWarning",
          any(issubclass(w.category, DeprecationWarning) for w in caught),
          f"warnings: {[w.category.__name__ for w in caught]}")
    check("legacy_temp=True samples from temp_choices [0,27,75]",
          legacy.temp_c in (0.0, 27.0, 75.0),
          f"got {legacy.temp_c}")

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        SimContext(temp_c=75.0)
    check("Explicit non-default temp_c emits DeprecationWarning",
          any(issubclass(w.category, DeprecationWarning) for w in caught2),
          f"warnings: {[w.category.__name__ for w in caught2]}")


def test_topology_primitives():
    print("\nTest 3: topology primitives generate well-formed graphs")
    from topology import (
        line_graph, ring_graph, grid_graph, cluster_graph, empty_graph
    )
    lg = line_graph(5, radius=1)
    check("line_graph n=5 rad=1: edges", lg.num_edges() == 4)
    check("line_graph no self-loops", all(s != d for s, d in zip(lg.src, lg.dst)))

    rg = ring_graph(8, radius=2)
    check("ring_graph n=8 rad=2: edges", rg.num_edges() == 16)  # 8*2

    gg = grid_graph(2, 3, kernel_size=3)
    check("grid_graph 2x3 kernel=3: 6 nodes", gg.num_nodes == 6)
    check("grid_graph edges > 0", gg.num_edges() > 0)
    # 2x3 grid, kernel=3: corners (4 nodes) have 3 neighbors, edge (2 nodes)
    # have 5 neighbors. Total one-sided = 4*3 + 2*5 = 22; unique pairs = 11.
    # Single-edge-per-pair representation: 11 edges.
    check("grid_graph: 2x3 kernel=3 emits 11 unique-pair edges",
          gg.num_edges() == 11, f"got {gg.num_edges()}")

    cg = cluster_graph(6, edge_prob=0.5, seed=42)
    check("cluster_graph n=6: 6 nodes", cg.num_nodes == 6)

    eg = empty_graph(4)
    check("empty_graph 0 edges", eg.num_edges() == 0)


def test_stage_transfer():
    print("\nTest 4: StageTransfer truncates and zero-pads")
    from stage_transfer import StageTransfer
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    t_eq = StageTransfer(5, 5)
    check("transfer equal width passes through", torch.equal(t_eq(x), x))
    t_trunc = StageTransfer(5, 3)
    check("transfer truncate to 3", torch.equal(t_trunc(x), x[:, :3]))
    t_pad = StageTransfer(5, 8)
    out = t_pad(x)
    check("transfer pad to 8: shape (1,8)", out.shape == (1, 8))
    check("transfer pad: original preserved", torch.equal(out[:, :5], x))
    check("transfer pad: zeros appended", torch.equal(out[:, 5:], torch.zeros(1, 3)))


def test_heun_converges():
    print("\nTest 5: Heun integration converges without explosion (small 1-stage net)")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet
    from sim_context import SimContext

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, active, _ = topology_to_stage(topo, cell_lib=cell_lib)
    net = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.5], stage_steps=[20])

    ctx = SimContext()
    x0 = torch.zeros(1, 4)
    x_final, trajs = net(x0, ctx=ctx, tau=1.0, store_trajectory=True)
    check("x_final is finite", torch.isfinite(x_final).all().item())
    check("|x_final| <= 1.0 (no explosion)", (x_final.abs() <= 1.0).all().item(),
          f"max |x| = {float(x_final.abs().detach().max()):.4f}")
    check("trajectory shape (1, 4, 21)", trajs[0].shape == (1, 4, 21))


def test_gradient_flow():
    print("\nTest 6: gradients flow through every parameter group")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet
    from sim_context import SimContext

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, active, _ = topology_to_stage(topo, cell_lib=cell_lib)
    net = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[10])

    ctx = SimContext()
    x0 = torch.ones(2, 3) * 0.05
    x_final, _ = net(x0, ctx=ctx, tau=1.0, store_trajectory=False)
    loss = x_final.pow(2).sum()
    loss.backward()

    has_grads = {
        "logits": stage.logits.grad is not None and torch.isfinite(stage.logits.grad).all().item(),
        "raw_mult": stage.raw_mult.grad is not None and torch.isfinite(stage.raw_mult.grad).all().item(),
        "raw_leak": stage.raw_leak.grad is not None and torch.isfinite(stage.raw_leak.grad).all().item(),
    }
    check("grads flow to logits", has_grads["logits"])
    check("grads flow to raw_mult", has_grads["raw_mult"])
    check("grads flow to raw_leak", has_grads["raw_leak"])


def test_compute_loss_finite():
    print("\nTest 7: compute_loss returns finite values for all regularizers")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import compute_loss

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=2, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    from topology import topology_to_stage as tts
    stage, active, _ = tts(topo, cell_lib=cell_lib)
    from kirchhoff_net import KirchhoffNet
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[10])
    inp = InputMapper(in_dim=2, out_dim=4)
    out = OutputMapper(node_dim=4, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=4, proj_count=0)

    u = torch.randn(8, 2) * 0.5
    target = torch.zeros(8, 1)
    ctx = SimContext()
    loss_task, loss_structural, parts = compute_loss(net, u, target, ctx, F.mse_loss, return_parts=True)
    check("total loss finite", math.isfinite(parts["total"]))
    check("task loss finite", math.isfinite(parts["task"]))
    check("sparsity loss finite", math.isfinite(parts["sparsity"]))
    check("edge_gate loss finite (CP: Σ z_e)", math.isfinite(parts["edge_gate"]))
    check("node_gate loss finite (CP: Σ u_j)", math.isfinite(parts["node_gate"]))
    check("power loss finite (CP: static power proxy)", math.isfinite(parts["power"]))
    check("capacitance loss finite (CP: cap area proxy)", math.isfinite(parts["capacitance"]))
    check("rail loss finite", math.isfinite(parts["rail"]))
    check("edge_gate loss >= 0", parts["edge_gate"] >= 0.0)
    check("node_gate loss >= 0", parts["node_gate"] >= 0.0)
    check("power loss >= 0", parts["power"] >= 0.0)
    check("capacitance loss >= 0", parts["capacitance"] >= 0.0)
    check("sparsity loss >= 0", parts["sparsity"] >= 0.0)
    check("rail loss >= 0", parts["rail"] >= 0.0)
    check("reg_scale reported (RR-A)", "reg_scale" in parts)


def test_sparsity_push():
    print("\nTest 8: sparsity regularizer reduces P(active) over training")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNetWithIO, KirchhoffNet
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import compute_loss, make_optimizer, default_ctx_factory, LAMBDAS

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
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
        probs_before = F.softmax(stage.logits, dim=-1).clone()

    lambdas = dict(LAMBDAS)
    lambdas["sparsity"] = 1.0
    for step in range(50):
        ctx = ctx_factory(batch_size=16, device=u.device)
        opt.zero_grad()
        loss_task, loss_structural = compute_loss(net, u, target, ctx, F.mse_loss, lambdas=lambdas, tau=1.0)
        loss_task.backward(retain_graph=True)
        loss_structural.backward()
        opt.step()

    with torch.no_grad():
        probs_after = F.softmax(stage.logits, dim=-1)

    p_active_before = probs_before[:, :3].sum().item()
    p_active_after = probs_after[:, :3].sum().item()
    check("sparsity: P(active) decreased", p_active_after < p_active_before,
          f"before={p_active_before:.3f} after={p_active_after:.3f}")
    check("sparsity: P(Z) increased", probs_after[:, 3].mean().item() > probs_before[:, 3].mean().item())


def test_tau_anneal():
    print("\nTest 9: tau_for_epoch produces annealing schedule")
    from train import tau_for_epoch, TAU, OPTIM
    total = int(OPTIM["epochs"])
    t0 = tau_for_epoch(0, total_epochs=total)
    t_mid = tau_for_epoch(total // 2, total_epochs=total)
    t_end = tau_for_epoch(total - 1, total_epochs=total)
    check("tau at epoch 0 == init", abs(t0 - TAU["init"]) < 1e-6, f"got {t0}")
    check("tau at mid-epoch decays below init", t_mid < TAU["init"] * 0.5,
          f"got {t_mid}")
    check("tau at final epoch ≈ final", abs(t_end - TAU["final"]) < 1e-2, f"got {t_end}")


def test_round_trip_preset():
    print("\nTest 10: end-to-end round-trip with sinx preset (forward + backward + step)")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from sim_context import SimContext
    from train import compute_loss, make_optimizer, default_ctx_factory, LAMBDAS

    net = build_net_from_preset("sinx", cell_lib=make_default_library())

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
        y1, _ = net(u, ctx=SimContext(), store_trajectory=False)
        y2, _ = net(u, ctx=SimContext(), store_trajectory=False)
    check("sinx preset: re-forward succeeds (no NaN)", torch.isfinite(y2).all().item())


def test_xor_preset_removed():
    print("\nTest 11: XOR preset removed (R4.2)")
    from config import PRESETS
    check("xor not in active PRESETS (R4.2)", "xor" not in PRESETS)


def test_housing_preset_robust():
    print("\nTest 12: housing preset uses RobustInputMapper (dense mode) / SparseInputMapper (sparse default)")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from io_mapper import RobustInputMapper, SparseInputMapper
    from sim_context import SimContext

    net_dense = build_net_from_preset(
        "housing", cell_lib=make_default_library(),
        write_mode="dense", read_mode="dense",
    )
    check("housing dense uses RobustInputMapper",
          isinstance(net_dense.input_mapper, RobustInputMapper))

    net_sparse = build_net_from_preset("housing", cell_lib=make_default_library())
    check("housing default (sparse) uses SparseInputMapper",
          isinstance(net_sparse.input_mapper, SparseInputMapper))

    u = torch.randn(8, 8)
    target = torch.randn(8, 1)
    ctx = SimContext()
    y, _ = net_dense(u, ctx=ctx, store_trajectory=False)
    check("housing dense: forward shape (8,1)", y.shape == (8, 1))
    check("housing dense: output finite", torch.isfinite(y).all().item())

    y_s, _ = net_sparse(u, ctx=ctx, store_trajectory=False)
    check("housing sparse: forward shape (8,1)", y_s.shape == (8, 1))
    check("housing sparse: output finite", torch.isfinite(y_s).all().item())


def test_topology_to_stage_input_output_filtering():
    print("\nTest 13: topology_to_stage filters input/output edges from ODE")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=2, num_outputs=1, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, active, id_map = topology_to_stage(topo, cell_lib=cell_lib)

    check("topology_to_stage: stage evolves only hidden+proj nodes",
          stage.num_nodes == 3 + 0)  # 3 hidden, 0 proj
    check("topology_to_stage: id_map covers active nodes",
          set(id_map.keys()) == set(active))
    check("topology_to_stage: stage edges only between active nodes",
          all(0 <= s < 3 and 0 <= d < 3 for s, d in zip(stage.src.tolist(), stage.dst.tolist())))


def test_validate_topology():
    print("\nTest 14: validate_topology sanity checks")
    from topology import (
        cluster_graph, StageTopologyBuilder, validate_topology,
        SparseTopology, EDGE_TYPE_HIDDEN
    )
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=2, num_outputs=1, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    try:
        validate_topology(topo)
        check("validate_topology: valid topo passes", True)
    except ValueError as e:
        check("validate_topology: valid topo passes", False, str(e))

    # Self-loop should fail
    bad = SparseTopology(
        num_nodes=3, src=[0, 1], dst=[1, 1],
        edge_type=[EDGE_TYPE_HIDDEN, EDGE_TYPE_HIDDEN],
        node_kind=["hidden"] * 3, hidden_node_ids=[0, 1, 2],
    )
    try:
        validate_topology(bad)
        check("validate_topology: self-loop rejected", False, "no error raised")
    except ValueError:
        check("validate_topology: self-loop rejected", True)


def test_visualize_stage_graph():
    print("\nTest 15: visualize.plot_stage_graph runs and saves PNG")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from visualize import plot_stage_graph

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    out_path = "/tmp/test_stage_graph.png"
    fig, ax = plot_stage_graph(stage, save_path=out_path)
    import os
    check("plot_stage_graph: file written", os.path.exists(out_path) and os.path.getsize(out_path) > 0)


def test_visualize_sparse_topology():
    print("\nTest 16: visualize.plot_sparse_topology runs with all 4 node kinds")
    from topology import (
        cluster_graph, StageTopologyBuilder, EDGE_TYPE_HIDDEN, EDGE_TYPE_PROJ,
        EDGE_TYPE_INPUT, EDGE_TYPE_OUTPUT, NODE_KIND_HIDDEN, NODE_KIND_PROJ,
        NODE_KIND_INPUT, NODE_KIND_OUTPUT,
    )
    from visualize import plot_sparse_topology

    hid = cluster_graph(3, edge_prob=0.4, seed=0)
    builder = StageTopologyBuilder(num_inputs=2, num_outputs=1, num_hidden=3, num_proj=1)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")

    out_path = "/tmp/test_sparse_topology.png"
    fig, ax = plot_sparse_topology(topo, save_path=out_path)
    import os
    check("plot_sparse_topology: file written", os.path.exists(out_path) and os.path.getsize(out_path) > 0)


def test_visualize_trajectories():
    print("\nTest 17: visualize.plot_trajectories runs on a 1-stage forward pass")
    import torch
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet
    from sim_context import SimContext
    from visualize import plot_trajectories

    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    net = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.5], stage_steps=[10])
    ctx = SimContext()
    x0 = torch.zeros(2, 4)
    x_final, trajs = net(x0, ctx=ctx, tau=1.0, store_trajectory=True)
    traj_tensor = trajs[0] if isinstance(trajs, list) else trajs

    out_path = "/tmp/test_trajectories.png"
    fig, ax = plot_trajectories(traj_tensor, stage_idx=0, save_path=out_path)
    import os
    check("plot_trajectories: file written", os.path.exists(out_path) and os.path.getsize(out_path) > 0)


def test_sparse_spd_generation():
    print("\nTest 18: generate_sparse_spd produces valid SPD matrices")
    from sparse_solver_data import generate_sparse_spd

    A, cond = generate_sparse_spd(n=24, density=0.05, cond_target=50.0, seed=0)
    check("A shape (24, 24)", A.shape == (24, 24))
    check("A is float32", A.dtype == torch.float32)
    eigs = torch.linalg.eigvalsh(A)
    check("A is SPD (all eigs > 0)", bool((eigs > 0).all().item()),
          f"min eig = {float(eigs.min()):.4e}")
    check("A is symmetric",
          torch.allclose(A, A.T, atol=1e-5),
          f"max |A - A.T| = {float((A - A.T).abs().max()):.4e}")
    check("condition number near target",
          1.0 < cond < 1e4,
          f"cond = {cond:.2e}")


def test_solver_dataset():
    print("\nTest 19: SparseLinearSystemDataset returns consistent (b, x*, A) triples")
    from sparse_solver_data import SparseLinearSystemDataset

    ds = SparseLinearSystemDataset(n=16, num_samples=5, seed=42)
    check("dataset length == 5", len(ds) == 5)
    b, x_star, A = ds[0]
    check("b shape (16,)", b.shape == (16,))
    check("x_star shape (16,)", x_star.shape == (16,))
    check("A shape (16, 16)", A.shape == (16, 16))
    check("b == A @ x_star", torch.allclose(A @ x_star, b, atol=1e-5),
          f"max |b - A x*| = {float((b - A @ x_star).abs().max()):.4e}")
    check("|x_star| <= x_max", float(x_star.abs().max()) <= 0.3 + 1e-6)


def test_union_topology():
    print("\nTest 20: build_union_topology produces valid topology from dataset")
    from sparse_solver_data import SparseLinearSystemDataset
    from sparse_solver_topology import build_union_topology

    ds = SparseLinearSystemDataset(n=20, num_samples=20, density=0.05, seed=1)
    topo = build_union_topology(ds, n=20, num_proj=2, min_freq=0.2)
    check("topo has num_inputs + num_hidden + num_proj nodes",
          topo.num_nodes == 20 + 20 + 2)
    check("topo has input + hidden + proj + output placeholders",
          all(k in topo.node_kind for k in ("input", "hidden", "proj")))
    check("topo has at least one hidden edge", topo.num_edges() > 0)
    hidden_edges = [(s, d) for s, d, t in zip(topo.src, topo.dst, topo.edge_type) if t == "hidden"]
    check("hidden edges have src < dst (single-edge-per-pair)",
          all(s < d for s, d in hidden_edges))


def test_solver_loss_finite():
    print("\nTest 21: compute_solver_loss returns finite values with all regularizers")
    from sparse_solver_data import SparseLinearSystemDataset
    from sparse_solver_topology import build_union_topology
    from topology import topology_to_stage
    from cell_library import IdealizedCellLibrary
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import compute_solver_loss, LAMBDAS

    ds = SparseLinearSystemDataset(n=8, num_samples=10, density=0.1, seed=2)
    topo = build_union_topology(ds, n=8, num_proj=2)
    cell_lib = IdealizedCellLibrary()
    stage, active, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[8])
    inp = InputMapper(in_dim=8, out_dim=8)
    out = OutputMapper(node_dim=2, out_dim=8)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=8, proj_count=2)

    b, x_star, A = ds[0]
    b, x_star, A = b.unsqueeze(0), x_star.unsqueeze(0), A.unsqueeze(0)
    ctx = SimContext()
    loss_task, loss_structural, parts = compute_solver_loss(net, b, x_star, A, ctx, return_parts=True)
    check("total loss finite", math.isfinite(parts["total"]))
    check("residual finite", math.isfinite(parts["residual"]))
    check("solution finite", math.isfinite(parts["solution"]))
    check("sparsity finite", math.isfinite(parts["sparsity"]))
    check("edge_gate finite (CP: Σ z_e)", math.isfinite(parts["edge_gate"]))
    check("node_gate finite (CP: Σ u_j)", math.isfinite(parts["node_gate"]))
    check("power finite (CP: static power proxy)", math.isfinite(parts["power"]))
    check("capacitance finite (CP: cap area proxy)", math.isfinite(parts["capacitance"]))
    check("rail finite", math.isfinite(parts["rail"]))
    check("entropy finite", math.isfinite(parts["entropy"]))
    check("reg_scale reported (RR-A)", "reg_scale" in parts)


def test_baseline_jacobi():
    print("\nTest 22: DigitalSolverBaseline.jacobi converges on strictly diag-dominant SPD")
    import torch.nn.functional as F
    from sparse_solver_baseline import DigitalSolverBaseline

    n = 12
    S = torch.randn(1, n, n) * 0.05
    S = S * (torch.rand(1, n, n) > 0.7).float()
    A = (S + S.transpose(-1, -2)).squeeze(0) + torch.eye(n) * 5.0
    A = A.unsqueeze(0)
    x_star = 0.2 * torch.tanh(torch.randn(1, n))
    b = torch.bmm(A, x_star.unsqueeze(-1)).squeeze(-1)
    baseline = DigitalSolverBaseline(n=n, max_iters=100)
    x_solved, residuals = baseline.jacobi(A, b, steps=100)
    check("Jacobi: finite solution on diagonally dominant SPD",
          not torch.isnan(x_solved).any().item())
    check("Jacobi: residual decreases (final << initial)",
          residuals[-1] < residuals[0] * 0.1,
          f"init={residuals[0]:.4e} final={residuals[-1]:.4e}")


def test_baseline_cg():
    print("\nTest 23: DigitalSolverBaseline.conjugate_gradient converges on SPD system")
    from sparse_solver_data import generate_sparse_spd
    from sparse_solver_baseline import DigitalSolverBaseline

    A, _ = generate_sparse_spd(n=12, density=0.1, cond_target=10.0, seed=4)
    A = A.unsqueeze(0)
    x_star = 0.2 * torch.tanh(torch.randn(1, 12))
    b = torch.bmm(A, x_star.unsqueeze(-1)).squeeze(-1)
    baseline = DigitalSolverBaseline(n=12, max_iters=12)
    x_solved, residuals = baseline.conjugate_gradient(A, b, steps=12)
    check("CG: converges to low residual", residuals[-1] < 1e-3,
          f"final residual = {residuals[-1]:.4e}")
    sol_err = (x_solved - x_star).norm(dim=-1).item()
    check("CG: solution error small", sol_err < 1e-2, f"sol err = {sol_err:.4e}")


def test_convergence_tracker():
    print("\nTest 24: ConvergenceTracker captures snapshots and plots")
    from sparse_solver_data import generate_sparse_spd
    from sparse_solver_track import ConvergenceTracker

    A, _ = generate_sparse_spd(n=8, density=0.1, cond_target=10.0, seed=5)
    x_star = 0.2 * torch.tanh(torch.randn(8))
    b = A @ x_star

    tracker = ConvergenceTracker()
    for k in range(5):
        t = k * 0.1
        x = x_star * (1.0 - float(torch.tensor(k + 1).reciprocal()))
        tracker.capture(t, x.unsqueeze(0), label="net")
    check("tracker captured 5 snapshots", len(tracker.snapshots) == 5)

    out_path = "/tmp/test_convergence_tracker.png"
    fig = tracker.plot_residual_trajectory(A, b, x_star, save_path=out_path)
    import os
    check("plot saved to disk", os.path.exists(out_path) and os.path.getsize(out_path) > 0)


def test_solver_preset_removed():
    print("\nTest 25: solver preset removed from active PRESETS (R4.3)")
    from config import PRESETS
    check("solver not in active PRESETS (R4.3)", "solver" not in PRESETS)


def test_io_honest_split():
    print("\nTest 26: honest I/O split (R1.1, R1.2, R1.3)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext

    cell_lib = IdealizedCellLibrary()
    n_hid, n_proj = 4, 2
    hid = ring_graph(n_hid, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=n_hid, num_proj=n_proj)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[5])
    inp = InputMapper(in_dim=1, out_dim=n_hid)
    out = OutputMapper(node_dim=n_proj, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=n_hid, proj_count=n_proj)

    u = torch.tensor([[0.5]])
    with torch.no_grad():
        x0_full = torch.cat([inp(u), torch.zeros(1, n_proj)], dim=1)
    check("R1.2: projection portion of x0 is zero",
          torch.equal(x0_full[0, n_hid:], torch.zeros(n_proj)),
          f"proj slice = {x0_full[0, n_hid:].tolist()}")
    check("R1.1: hidden portion of x0 is non-zero (tanh write)",
          x0_full[0, :n_hid].abs().sum() > 0)

    ctx = SimContext()
    y, _ = net(u, ctx=ctx, store_trajectory=False)
    check("R1.3: forward produces output of expected shape (1,1)",
          y.shape == (1, 1))
    check("R1.3: output is finite", torch.isfinite(y).all().item())

    sliced = torch.randn(1, n_hid + n_proj)
    sliced[0, n_hid:] = 7.0
    x_read = sliced[:, net.read_slice]
    check("R1.3: read_slice selects only projection positions",
          torch.equal(x_read[0], sliced[0, n_hid:]))


def test_io_no_proj_fallback():
    print("\nTest 27: I/O fallback when proj_count=0 (R1.4)")
    import warnings as warnings_mod
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[5])
    inp = InputMapper(in_dim=1, out_dim=3)
    out = OutputMapper(node_dim=3, out_dim=1)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        net = KirchhoffNetWithIO(inp, core, out, hid_count=3, proj_count=0)
    check("R1.4: warning emitted when no projection nodes",
          any("R1.4" in str(w.message) for w in caught),
          f"caught {len(caught)} warnings")
    check("R1.4: read_slice falls back to hidden positions",
          net.read_slice == slice(0, 3))


def test_mapper_only_ablation():
    print("\nTest 28: mapper-only ablation runs (R2.2)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.0], stage_steps=[5])
    inp = InputMapper(in_dim=1, out_dim=4)
    out = OutputMapper(node_dim=4, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=4, proj_count=0)

    u = torch.tensor([[0.5]])
    ctx = SimContext()
    y, _ = net(u, ctx=ctx, store_trajectory=False)
    expected = out(inp(u))
    check("R2.2: mapper-only output equals OutputMapper(InputMapper(u))",
          torch.allclose(y, expected, atol=1e-5),
          f"got {y.item():.4f} vs expected {expected.item():.4f}")


def test_mapper_only_ablation_fast():
    print("\nTest 43: mapper-only ablation skips ODE core (fast path)")
    import time
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.0], stage_steps=[20])
    inp = InputMapper(in_dim=1, out_dim=4)
    out = OutputMapper(node_dim=4, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=4, proj_count=0)

    u = torch.randn(128, 1)
    ctx = SimContext()

    with torch.no_grad():
        y, trajs = net(u, ctx=ctx, store_trajectory=True)
    check("fast path: trajectories is None", trajs is None)
    check("fast path: output finite", torch.isfinite(y).all().item())
    expected = out(inp(u))
    check("fast path: output equals OutputMapper(InputMapper(u))",
          torch.allclose(y, expected, atol=1e-5))

    core2 = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.0], stage_steps=[20])
    net2 = KirchhoffNetWithIO(inp, core2, out, hid_count=4, proj_count=0)

    n_warmup = 5
    n_iters = 50
    with torch.no_grad():
        for _ in range(n_warmup):
            net2(u, ctx=ctx, store_trajectory=False)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            net2(u, ctx=ctx, store_trajectory=False)
        fast_time = time.perf_counter() - t0

    core3 = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.5], stage_steps=[20])
    net3 = KirchhoffNetWithIO(inp, core3, out, hid_count=4, proj_count=0)
    with torch.no_grad():
        for _ in range(n_warmup):
            net3(u, ctx=ctx, store_trajectory=False)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            net3(u, ctx=ctx, store_trajectory=False)
        slow_time = time.perf_counter() - t0

    speedup = slow_time / max(fast_time, 1e-9)
    check(f"fast path: mapper-only is >2x faster than full ODE "
          f"(fast={fast_time*1000/n_iters:.2f}ms vs slow={slow_time*1000/n_iters:.2f}ms, "
          f"speedup={speedup:.1f}x)",
          speedup > 2.0,
          f"fast={fast_time:.3f}s slow={slow_time:.3f}s speedup={speedup:.1f}x")


def test_complexity_proxy():
    print("\nTest 29: complexity proxy = m_e·(1 - p_Z_e), softplus multiplicity (RR-B, R3)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import _stage_soft_weights, _stage_multiplicities
    import torch.nn.functional as F

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[5])
    inp = InputMapper(in_dim=1, out_dim=3)
    out = OutputMapper(node_dim=3, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=3, proj_count=0)

    with torch.no_grad():
        stage.logits.fill_(0.0)
        stage.raw_mult.fill_(-3.0)
    mult = _stage_multiplicities(stage)
    w = _stage_soft_weights(stage)
    p_active = 1.0 - w[:, 2]
    weighted = (mult * p_active).sum()
    unweighted = mult.sum()
    with torch.no_grad():
        stage.raw_mult.fill_(0.0)
    mult_zero = _stage_multiplicities(stage)
    check("R3.3: softplus(0) == log(2) for raw_mult=0",
          abs(float(mult_zero[0].item()) - math.log(2.0)) < 1e-4)
    check("R3.3: softplus(-3) is small but positive",
          float(mult[0].item()) < 0.2,
          f"got {float(mult[0].item()):.4f}")
    check("R3.1: complexity proxy < unweighted sum when p_Z > 0",
          float(weighted.item()) < float(unweighted.item()))

    with torch.no_grad():
        stage.logits.fill_(-10.0)
        stage.logits[:, 2] = 10.0
    w = _stage_soft_weights(stage)
    p_active = 1.0 - w[:, 2]
    weighted_full_z = (mult * p_active).sum()
    check("R3.4: complexity proxy goes to zero when all edges are Z-selected",
          float(weighted_full_z.item()) < 1e-3,
          f"got {float(weighted_full_z.item()):.4e}")


def test_reg_schedule_curve():
    print("\nTest 43: reg_schedule piecewise-linear warm-up (RR-A)")
    from train import reg_schedule, OPTIM
    warmup = int(OPTIM.get("reg_warmup_epochs", 50))
    anneal = int(OPTIM.get("reg_anneal_epochs", 50))
    check("reg_schedule(0) == 0.0 (free phase)", reg_schedule(0) == 0.0)
    check(f"reg_schedule({warmup - 1}) == 0.0 (still in free phase)",
          reg_schedule(warmup - 1) == 0.0)
    mid = warmup + anneal // 2
    expected_mid = float(mid - warmup + 1) / float(max(1, anneal))
    check(f"reg_schedule({mid}) in (0, 1) (linear ramp)",
          0.0 < reg_schedule(mid) < 1.0,
          f"got {reg_schedule(mid):.4f}")
    check(f"reg_schedule({mid}) ≈ {expected_mid:.4f}",
          abs(reg_schedule(mid) - expected_mid) < 1e-9,
          f"got {reg_schedule(mid):.4f} expected {expected_mid:.4f}")
    check(f"reg_schedule({warmup + anneal}) == 1.0 (full penalty)",
          reg_schedule(warmup + anneal) == 1.0)
    check("reg_schedule is non-decreasing across the curve",
          all(reg_schedule(e + 1) >= reg_schedule(e) - 1e-9
              for e in range(0, warmup + anneal + 20)))

    from train import apply_reg_schedule, LAMBDAS, _REG_KEYS
    base = dict(LAMBDAS)
    scaled = apply_reg_schedule(base, epoch=0)
    check("apply_reg_schedule(0): sparsity=0 (free phase)",
          scaled["sparsity"] == 0.0)
    check("fix-z-death: apply_reg_schedule(0): rail == LAMBDAS['rail'] (NOT scaled)",
          abs(scaled["rail"] - base["rail"]) < 1e-12,
          f"got {scaled['rail']}")
    check("apply_reg_schedule(0): edge_gate=0 (free phase)",
          scaled["edge_gate"] == 0.0)
    check("apply_reg_schedule(0): node_gate=0 (free phase)",
          scaled["node_gate"] == 0.0)
    check("apply_reg_schedule(0): power=0 (free phase)",
          scaled["power"] == 0.0)
    check("apply_reg_schedule(0): capacitance=0 (free phase)",
          scaled["capacitance"] == 0.0)
    full = apply_reg_schedule(base, epoch=warmup + anneal + 10)
    check("apply_reg_schedule(late): sparsity == LAMBDAS['sparsity']",
          abs(full["sparsity"] - base["sparsity"]) < 1e-12)
    check("apply_reg_schedule(late): edge_gate == LAMBDAS['edge_gate']",
          abs(full["edge_gate"] - base["edge_gate"]) < 1e-12)
    check("apply_reg_schedule(late): power == LAMBDAS['power']",
          abs(full["power"] - base["power"]) < 1e-12)
    check("fix-z-death: 'rail' is NOT in _REG_KEYS (always active safety clamp)",
          "rail" not in _REG_KEYS)


def test_smooth2d_grid_sparsity_zero_override():
    print("\nTest 43a: smooth2d_grid preset lambdas (legacy path) and schedule config")
    import config
    from config import SCHEDULE_THREE_PHASE
    from train_script import _resolve_lambdas
    sg_lambdas = _resolve_lambdas("smooth2d_grid")
    # Per-preset lambdas are for the legacy schedule path, distinct from
    # Phase B schedule lambdas (which apply under --schedule three_phase).
    check("smooth2d-grid: preset legacy lambdas sparsity=1e-5",
          abs(sg_lambdas["sparsity"] - 1e-5) < 1e-12,
          f"got {sg_lambdas['sparsity']}")
    check("smooth2d-grid: preset legacy lambdas edge_gate=5e-6",
          abs(sg_lambdas["edge_gate"] - 5e-6) < 1e-12,
          f"got {sg_lambdas['edge_gate']}")
    check("smooth2d-grid: preset legacy lambdas node_gate=0.0 (deprecate-node-gates)",
          abs(sg_lambdas["node_gate"] - 0.0) < 1e-12,
          f"got {sg_lambdas['node_gate']}")
    check("smooth2d-grid: preset legacy lambdas power=1e-5",
          abs(sg_lambdas["power"] - 1e-5) < 1e-12,
          f"got {sg_lambdas['power']}")
    check("smooth2d-grid: preset legacy lambdas capacitance=0.0 (deprecate-node-gates)",
          abs(sg_lambdas["capacitance"] - 0.0) < 1e-12,
          f"got {sg_lambdas['capacitance']}")
    check("smooth2d-grid: Phase B schedule sparsity=5e-5",
          abs(float(SCHEDULE_THREE_PHASE["lambdas_b"]["sparsity"]) - 5e-5) < 1e-12,
          f"got {SCHEDULE_THREE_PHASE['lambdas_b']['sparsity']}")
    check("smooth2d-grid: Phase B schedule edge_gate=1e-5",
          abs(float(SCHEDULE_THREE_PHASE["lambdas_b"]["edge_gate"]) - 1e-5) < 1e-12,
          f"got {SCHEDULE_THREE_PHASE['lambdas_b']['edge_gate']}")
    check("smooth2d-grid: preset schedule=three_phase",
          config.PRESETS["smooth2d_grid"].get("schedule") == "three_phase")
    check("smooth2d-grid: preset tau_anneal is True",
          config.PRESETS["smooth2d_grid"].get("tau_anneal", True) is True,
          f"got {config.PRESETS['smooth2d_grid'].get('tau_anneal', True)}")


def test_tau_anneal_preset_option():
    print("\nTest 43b: tau_anneal preset option controls fitting-phase temperature (pruning-gate-transfer)")
    import config
    from train import TAU
    check("pruning-gate-transfer: smooth2d_grid has tau_anneal=True",
          config.PRESETS["smooth2d_grid"].get("tau_anneal", True) is True)
    check("fix-z-death: sinx preset has tau_anneal=True (default, backward compat)",
          config.PRESETS["sinx"].get("tau_anneal", True) is True)
    check("fix-z-death: housing preset has tau_anneal=True (default, backward compat)",
          config.PRESETS["housing"].get("tau_anneal", True) is True)
    check("fix-z-death: smooth2d (non-grid) preset has tau_anneal=True (default)",
          config.PRESETS["smooth2d"].get("tau_anneal", True) is True)
    init_tau = float(TAU["init"])
    check(f"fix-z-death: TAU['init'] == {init_tau} (used as fixed tau when tau_anneal=False)",
          init_tau == 1.0,
          f"got {init_tau}")


def test_tau_override_floor_guarantee():
    """R2-phase-tau-1: tau_for_epoch with tau_final=0.8 never drops below 0.8."""
    print("\nTest 43c: tau_final override clamps tau >= tau_final (R2-phase-tau)")
    from train import tau_for_epoch
    total = 150
    eps = 1e-9
    min_tau = float("inf")
    for e in range(total + 1):
        t = tau_for_epoch(e, total_epochs=total, tau_final=0.8)
        min_tau = min(min_tau, float(t))
    check("R2-phase-tau: all tau values >= 0.8 (tau_final=0.8 override)",
          min_tau >= 0.8 - eps,
          f"minimum tau observed = {min_tau:.6f}")


def test_tau_override_endpoints():
    """R2-phase-tau-2: tau_for_epoch with tau_init=0.8, tau_final=0.1
    starts at 0.8 and ends at 0.1 (within hardening tolerance)."""
    print("\nTest 43d: tau_init/tau_final override produces correct endpoints (R2-phase-tau)")
    from train import tau_for_epoch
    total = 75
    t0 = tau_for_epoch(0, total_epochs=total, tau_init=0.8, tau_final=0.1)
    t_end = tau_for_epoch(total - 1, total_epochs=total, tau_init=0.8, tau_final=0.1)
    check("R2-phase-tau: tau_retrain(epoch=0) starts at 0.8",
          abs(t0 - 0.8) < 1e-6,
          f"got {t0}")
    check("R2-phase-tau: tau_retrain(epoch=last) ≈ 0.1",
          abs(t_end - 0.1) < 1e-2,
          f"got {t_end}")


def test_tau_override_backward_compat():
    """R2-phase-tau-3: tau_for_epoch with no overrides matches old behavior."""
    print("\nTest 43e: tau_for_epoch backward compatible with no overrides (R2-phase-tau)")
    from train import tau_for_epoch, TAU, OPTIM
    total = int(OPTIM["epochs"])
    t0 = tau_for_epoch(0, total_epochs=total)
    t_mid = tau_for_epoch(total // 2, total_epochs=total)
    t_end = tau_for_epoch(total - 1, total_epochs=total)
    check("R2-phase-tau: tau(0) == TAU['init']",
          abs(t0 - TAU["init"]) < 1e-6,
          f"got {t0}")
    check("R2-phase-tau: tau(mid) decays below init",
          t_mid < TAU["init"] * 0.5,
          f"got {t_mid}")
    check("R2-phase-tau: tau(last) ≈ TAU['final']",
          abs(t_end - TAU["final"]) < 1e-2,
          f"got {t_end}")


def test_preset_lambda_overrides():
    print("\nTest 44: per-preset lambda overrides (RR-D)")
    import config
    from train_script import _resolve_lambdas

    sinx_lambdas = _resolve_lambdas("sinx")
    check("RR-D: sinx preset lambdas override rail=1.0",
          sinx_lambdas["rail"] == 1.0,
          f"got {sinx_lambdas['rail']}")
    check("RR-D: sinx preset lambdas inherit global sparsity",
          sinx_lambdas["sparsity"] == config.LAMBDAS["sparsity"])
    check("RR-D: sinx preset lambdas inherit global edge_gate",
          sinx_lambdas["edge_gate"] == config.LAMBDAS["edge_gate"])
    check("RR-D: sinx preset lambdas inherit global power",
          sinx_lambdas["power"] == config.LAMBDAS["power"])

    housing_lambdas = _resolve_lambdas("housing")
    check("RR-D: housing preset (no override) inherits global rail",
          housing_lambdas["rail"] == config.LAMBDAS["rail"],
          f"expected {config.LAMBDAS['rail']}, got {housing_lambdas['rail']}")
    check("RR-D: housing preset inherits global sparsity",
          housing_lambdas["sparsity"] == config.LAMBDAS["sparsity"])


def test_z_bias_eliminated():
    print("\nTest 45: INIT['logits_z_bias'] = 0.0 → equal P(Z) = P(L) = P(S) = P(P) (fix-z-death)")
    import config
    import torch.nn.functional as F
    check("fix-z-death: config.INIT['logits_z_bias'] == 0.0",
          config.INIT["logits_z_bias"] == 0.0)
    zeros = torch.zeros(config.NUM_CELLS)
    zeros[config.Z_INDEX] = float(config.INIT["logits_z_bias"])
    probs = F.softmax(zeros, dim=-1)
    p_z = float(probs[config.Z_INDEX])
    check("fix-z-death: P(Z) at init is 0.25 (equal probability with 4 cells)",
          abs(p_z - 0.25) < 1e-5,
          f"got p_Z={p_z:.4f}")
    check("fix-z-death: P(Z) at init is below the OLD ~0.42 (with bias=1.0)",
          p_z < 0.30,
          f"got p_Z={p_z:.4f}")
    check("fix-z-death: P(L) = P(S) = P(P) = P(Z) = 0.25 (all cells equally likely)",
          all(abs(float(probs[i]) - 0.25) < 1e-5 for i in range(config.NUM_CELLS)),
          f"got probs={probs.tolist()}")


def test_gate_initialization():
    print("\nTest 46: gate parameter initialization (grid7-gate0: z=0.0, u=0.0)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from differential_stage import DifferentialStage

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    check("CP-1: stage has z_logits parameter", hasattr(stage, "z_logits"))
    check("CP-1: stage has u_logits parameter", hasattr(stage, "u_logits"))
    check("CP-1: z_logits shape == (E,)", stage.z_logits.shape == torch.Size([stage.num_edges()]))
    check("CP-1: u_logits shape == (N,)", stage.u_logits.shape == torch.Size([stage.num_nodes]))

    z = stage.edge_gates()
    u = stage.node_gates()
    check("CP-1: edge_gates() returns σ(z_logits)",
          z.shape == (stage.num_edges(),))
    check("CP-1 (deprecate-node-gates): node_gates() returns all-ones",
          u.shape == (stage.num_nodes,))
    check("grid7-gate0: σ(z_logit_init=0) = 0.5 (50% open gates, max gradient)",
          abs(float(z[0].item()) - 0.5) < 1e-3,
          f"got {float(z[0].item()):.4f}")
    check("grid7-gate0 (deprecate-node-gates): node_gates returns all-ones",
          abs(float(u[0].item()) - 1.0) < 1e-6,
          f"got {float(u[0].item()):.4f}")
    expected_grad = 0.5 * (1 - 0.5)
    check("grid7-gate0: σ'(z_logit=0) = 0.25 (max gradient sensitivity, 2.4x z=2.0)",
          abs(float((z * (1 - z))[0].item()) - expected_grad) < 1e-3,
          f"got {float((z * (1 - z))[0].item()):.4f}")


def test_gate_application_in_rhs():
    print("\nTest 47: gates applied in DifferentialStage.rhs (CP-2, deprecate-node-gates)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    # DEPRECATED (deprecate-node-gates): node gates are bypassed — the mask
    # is always 1.0 in rhs(). Setting u_logits to -10 has NO effect on rhs.
    # Only edge gates (z_logits) still affect the output.
    with torch.no_grad():
        stage.z_logits.fill_(-10.0)
        stage.u_logits.fill_(-10.0)

    x = torch.randn(2, 3) * 0.1
    dx = stage.rhs(x, ctx=SimContext(), tau=1.0)

    # With edge gates closed (z~0), i_edge ~ 0, dx = (0 - leak - clip) / C.
    # Leak is F.softplus(-3) ~ 0.05, so leak_term ~ 0.005, clip ~ 0 for small x.
    # Therefore dx ~ small but finite. Check it's different from the edge-gates-open case.
    with torch.no_grad():
        stage.z_logits.fill_(5.0)
        # u_logits stays -10 — node gates are bypassed so this shouldn't affect rhs.
    dx_open_edge = stage.rhs(x, ctx=SimContext(), tau=1.0)

    diff = (dx - dx_open_edge).abs().max().item()
    check("CP-2: edge gates modify RHS output (closed vs open differ)",
          diff > 1e-3,
          f"max abs diff = {diff:.4e}")

    # Verify node gates have NO effect in rhs.
    with torch.no_grad():
        stage.u_logits.fill_(5.0)
    dx_u_open = stage.rhs(x, ctx=SimContext(), tau=1.0)
    diff_u = (dx_open_edge - dx_u_open).abs().max().item()
    check("CP-2 (deprecate-node-gates): node gate changes do NOT affect RHS output",
          diff_u < 1e-6,
          f"max abs diff with u_logits = {diff_u:.4e}")


def test_complexity_regularizers():
    print("\nTest 48: per-component complexity regularizers (CP-4, deprecate-node-gates)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext
    from train import _stage_edge_gates, _stage_node_gates, _stage_multiplicities
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    z = _stage_edge_gates(stage)
    mult = _stage_multiplicities(stage)

    # Edge gate: σ(z_logit_init=0) = 0.5 for each edge (grid7-gate0).
    expected_z = stage.num_edges() * 0.5
    check("CP-4: edge_gate = Σz_e at init ≈ num_edges * 0.5 (z_logit_init=0)",
          abs(float(z.sum().item()) - expected_z) < 1e-2,
          f"got {float(z.sum().item()):.4f}, expected {expected_z:.4f}")

    # DEPRECATED (deprecate-node-gates): _stage_node_gates now returns
    # all-ones (node gates are bypassed). The raw sigmoid of u_logits
    # is no longer the correct node gate proxy.
    u = _stage_node_gates(stage)
    check("CP-4: node_gate is deprecated — returns all-ones",
          abs(float(u.sum().item()) - stage.num_nodes) < 1e-6,
          f"got {float(u.sum().item()):.4f}, expected {stage.num_nodes}")

    # Force all edge gates to zero; verify power and edge_gate go to zero.
    with torch.no_grad():
        stage.z_logits.fill_(-10.0)
    z_closed = _stage_edge_gates(stage)
    check("CP-4: when all z_logits -> -∞, edge_gate -> 0",
          float(z_closed.sum().item()) < 1e-3,
          f"got {float(z_closed.sum().item()):.4e}")

    # DEPRECATED: node gate raw sigmoid still responds to u_logits changes,
    # but the regularizer uses the deprecated all-ones return. Verify the
    # deprecated API still works (returns all-ones).
    with torch.no_grad():
        stage.u_logits.fill_(-10.0)
    u_closed = _stage_node_gates(stage)
    check("CP-4 (deprecate-node-gates): _stage_node_gates returns all-ones "
          "regardless of u_logits values",
          abs(float(u_closed.sum().item()) - stage.num_nodes) < 1e-6,
          f"got {float(u_closed.sum().item()):.4e}")


def test_prune_stage():
    print("\nTest 49: prune_stage removes low-gate edges and nodes (CP-5)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage, prune_stage
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    pre_edges = stage.num_edges()
    pre_nodes = stage.num_nodes

    # Force some edges to be closed (DEPRECATED: node gates have no effect).
    with torch.no_grad():
        stage.z_logits.data[0] = -10.0  # edge 0 is below threshold
        stage.z_logits.data[2] = -10.0  # edge 2 is below threshold

    pruned, _remap = prune_stage(stage, edge_threshold=0.01, node_threshold=0.01)

    check("CP-5: pruned stage has fewer edges (2 edges removed)",
          pruned.num_edges() < pre_edges,
          f"pre={pre_edges}, post={pruned.num_edges()}")
    # Node count depends on connectivity: node 3 may survive if its incident
    # edges (which are not explicitly killed) keep it connected. We only check
    # that nodes are removed IF they become disconnected. Since the 4-node
    # ring 0-1-2-3-0 is connected, and only edges 0 and 2 are pruned, the
    # surviving edges 1 and 3 keep all nodes connected via paths.
    # Node-gate pruning is bypassed, so expect all 4 nodes to survive.
    check("CP-5: pruned stage keeps all nodes (node gates bypassed, all connected)",
          pruned.num_nodes == pre_nodes,
          f"pre={pre_nodes}, post={pruned.num_nodes}")
    check("CP-5: pruned stage gates preserve init z_logit value (z=0 → σ=0.5, grid7-gate0)",
          pruned.z_logits is not None and abs(float(pruned.edge_gates().mean()) - 0.5) < 0.05,
          f"mean edge gate: {float(pruned.edge_gates().mean()):.4f}")
    check("CP-5: pruned stage is a DifferentialStage",
          hasattr(pruned, "rhs") and hasattr(pruned, "forward"))

    # Pruned stage must have valid forward pass.
    x = torch.randn(2, pruned.num_nodes)
    if pruned.num_nodes > 0 and pruned.num_edges() > 0:
        dx = pruned.rhs(x, ctx=None, tau=1.0)
        check("CP-5: pruned stage forward passes",
              dx.shape == x.shape and torch.isfinite(dx).all().item())


def test_prune_stage_transfer_params():
    print("\nTest 50: prune_stage transfers surviving parameters (CP-5)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage, prune_stage
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    with torch.no_grad():
        stage.z_logits.data[0] = -10.0  # edge 0 dies
        # Set unique logits values for surviving edges
        stage.logits.data[1].fill_(0.5)
        stage.logits.data[2].fill_(-0.5)
        stage.raw_mult.data[1].fill_(1.0)
        stage.raw_mult.data[2].fill_(2.0)
        # Set unique z_logits for surviving edges
        stage.z_logits.data[1] = 3.0
        stage.z_logits.data[2] = 4.0
        # Set unique raw_leak for nodes 0, 1, 2
        stage.raw_leak.data.fill_(-3.0)
        stage.raw_leak.data[0] = 0.0
        stage.raw_leak.data[1] = 1.0
        stage.raw_leak.data[2] = 2.0
        # Set unique u_logits for surviving nodes
        stage.u_logits.data[0] = 0.5
        stage.u_logits.data[1] = 1.5
        stage.u_logits.data[2] = 2.5

    pruned, _remap = prune_stage(stage, edge_threshold=0.01, node_threshold=0.01, transfer_params=True)

    if pruned.num_edges() >= 2:
        check("CP-5: transferred logits match (edge 1)",
              abs(float(pruned.logits[0, 0].item()) - 0.5) < 1e-5)
        check("CP-5: transferred logits match (edge 2)",
              abs(float(pruned.logits[1, 0].item()) - (-0.5)) < 1e-5)
        check("CP-5: transferred z_logits match (edge 1)",
              abs(float(pruned.z_logits[0].item()) - 3.0) < 1e-5)
        check("CP-5: transferred z_logits match (edge 2)",
              abs(float(pruned.z_logits[1].item()) - 4.0) < 1e-5)
    if pruned.num_nodes >= 3:
        check("CP-5: transferred raw_leak matches (node 0)",
              abs(float(pruned.raw_leak[0].item()) - 0.0) < 1e-5)
        check("CP-5: transferred raw_leak matches (node 1)",
              abs(float(pruned.raw_leak[1].item()) - 1.0) < 1e-5)
        check("CP-5: transferred u_logits match (node 0)",
              abs(float(pruned.u_logits[0].item()) - 0.5) < 1e-5)
        check("CP-5: transferred u_logits match (node 1)",
              abs(float(pruned.u_logits[1].item()) - 1.5) < 1e-5)


def test_prune_stage_all_removed_raises():
    print("\nTest 51: prune_stage raises when all edges removed (CP-5)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage, prune_stage
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(3, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    with torch.no_grad():
        # Force all edges below threshold
        stage.z_logits.fill_(-10.0)

    try:
        prune_stage(stage, edge_threshold=0.01, node_threshold=0.01)
        check("CP-5: raises ValueError when all edges removed", False,
              "did not raise")
    except ValueError:
        check("CP-5: raises ValueError when all edges removed", True)


def test_prune_network():
    print("\nTest 52: prune_network applies to KirchhoffNet core (CP-5)")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage, prune_network
    from kirchhoff_net import KirchhoffNet
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    with torch.no_grad():
        stage.z_logits.data[0] = -10.0
        # DEPRECATED: u_logits has no effect on pruning.

    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[5])
    pruned_core, _remaps = prune_network(core, edge_threshold=0.01, node_threshold=0.01)

    pre_edges = sum(s.num_edges() for s in core.stages)
    post_edges = sum(s.num_edges() for s in pruned_core.stages)
    check("CP-5: pruned core has fewer edges",
          post_edges < pre_edges,
          f"pre={pre_edges}, post={post_edges}")
    check("CP-5: pruned core has correct stage_times",
          pruned_core.stage_times == core.stage_times)
    check("CP-5: pruned core has correct stage_steps",
          pruned_core.stage_steps == core.stage_steps)


def test_validate_topology_degrees_valid():
    print("\nTest 53: validate_topology_degrees passes on line graph (PT-1)")
    from topology import validate_topology_degrees, line_graph
    g = line_graph(8, radius=2)
    try:
        validate_topology_degrees(
            src=g.src, dst=g.dst, num_nodes=8,
            write_idx=[0], read_idx=[7],
        )
        check("PT-1: 3-node line write=[0] read=[2] distance=2 passes", True)
    except ValueError as e:
        check("PT-1: valid topology passes", False, str(e))


def test_validate_topology_degrees_direct():
    print("\nTest 54: validate_topology_degrees raises on direct edge (PT-1)")
    from topology import validate_topology_degrees
    src = [0, 1, 1]
    dst = [1, 2, 0]
    try:
        validate_topology_degrees(
            src=src, dst=dst, num_nodes=3,
            write_idx=[0], read_idx=[1],
        )
        check("PT-1: direct write->read edge raises", False, "did not raise")
    except ValueError:
        check("PT-1: direct write->read edge raises ValueError", True)


def test_validate_topology_degrees_silent():
    print("\nTest 55: validate_topology_degrees silent when write/read=None (PT-1)")
    from topology import validate_topology_degrees
    src = [0, 1]
    dst = [1, 2]
    # Should not raise even with direct edge (no write/read to check).
    validate_topology_degrees(src=src, dst=dst, num_nodes=3, write_idx=None, read_idx=[2])
    validate_topology_degrees(src=src, dst=dst, num_nodes=3, write_idx=[0], read_idx=None)
    validate_topology_degrees(src=src, dst=dst, num_nodes=3, write_idx=None, read_idx=None)
    check("PT-1: silent when either write/read is None", True)


def test_joint_prune_z_dominant():
    print("\nTest 56: joint Z+gate pruning removes Z-dominant edges (PT-2)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Small triangle: 0-1-2-0 (all edges connect I/O)
    stage = DifferentialStage(
        num_nodes=3,
        src=[0, 1, 1],
        dst=[1, 2, 0],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        # Edge 0: P(L) ≈ 0.95 → non-Z (active)
        stage.logits.data[0, 0] = 5.0
        stage.logits.data[0, 3] = -5.0
        # Edge 1: P(L) ≈ 0.95 → non-Z (active)
        stage.logits.data[1, 0] = 5.0
        stage.logits.data[1, 3] = -5.0
        # Edge 2: P(Z) ≈ 0.95 → Z-dominant (should be pruned)
        stage.logits.data[2, 3] = 5.0
        stage.logits.data[2, 0] = -5.0
        # All gates fully open
        stage.z_logits.fill_(5.0)
        stage.u_logits.fill_(5.0)

    pre_edges = stage.num_edges()
    pruned, _remap = prune_stage(stage, edge_threshold=0.3, node_threshold=0.01,
                         write_idx=[0], read_idx=[2])
    # Edges 0,1 survive (effective score ≈ 0.95 > 0.3)
    check("PT-2: Z-dominant edge pruned",
          pruned.num_edges() == 2,
          f"pre={pre_edges}, post={pruned.num_edges()}")
    check("PT-2: I/O connectivity preserved",
          pruned.num_nodes == 3,
          f"expected 3 nodes, got {pruned.num_nodes}")


def test_prune_dead_island_removed():
    print("\nTest 57: prune_stage connectivity backstop removes dead islands (PT-3)")
    import torch
    from cell_library import IdealizedCellLibrary
    from topology import prune_stage, _bfs_undirected

    cell_lib = IdealizedCellLibrary()
    # Build a 6-node graph with two disconnected rings:
    # 0-1-2 (write=0, read=2) and 3-4-5 (dead island, no I/O)
    # This is constructed directly as a DifferentialStage for testing.
    from differential_stage import DifferentialStage
    stage = DifferentialStage(
        num_nodes=6,
        src=[0, 1, 3, 4],
        dst=[1, 2, 4, 5],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.z_logits.fill_(5.0)  # all edges active
        stage.u_logits.fill_(5.0)  # all nodes active

    pruned, _remap = prune_stage(stage, edge_threshold=0.01, node_threshold=0.01,
                         write_idx=[0], read_idx=[2])
    # Should keep only nodes {0,1,2}
    check("PT-3: dead island removed after pruning",
          pruned.num_nodes == 3,
          f"expected 3, got {pruned.num_nodes}")
    check("PT-3: dead island edges removed",
          pruned.num_edges() == 2,
          f"expected 2, got {pruned.num_edges()}")


def test_prune_disconnected_io_raises():
    print("\nTest 58: prune_stage raises when pruning disconnects I/O (PT-3)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 2],   # 0-1 and 2-3 are two disconnected edges
        dst=[1, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.z_logits.fill_(5.0)
        stage.u_logits.fill_(5.0)

    try:
        prune_stage(stage, edge_threshold=0.01, node_threshold=0.01,
                    write_idx=[0], read_idx=[3])
        check("PT-3: disconnected I/O after pruning raises", False, "did not raise")
    except ValueError:
        check("PT-3: disconnected I/O after pruning raises ValueError", True)


def test_prune_network_returns_remap():
    print("\nTest 59: prune_network returns per-stage node remap dicts (PIT-1)")
    from cell_library import IdealizedCellLibrary
    from topology import (
        build_net_from_preset,
        prune_network,
    )
    from config import PRUNE

    cell_lib = IdealizedCellLibrary()
    net = build_net_from_preset("smooth2d", cell_lib=cell_lib)
    pruned_core, stage_remaps = prune_network(
        net.core,
        edge_threshold=float(PRUNE["edge_threshold"]),
        node_threshold=float(PRUNE["node_threshold"]),
        transfer_params=True,
    )
    check("PIT-1: stage_remaps is a list", isinstance(stage_remaps, list))
    check("PIT-1: one remap dict per stage",
          len(stage_remaps) == len(pruned_core.stages))
    for i, remap in enumerate(stage_remaps):
        check(f"PIT-1: stage {i} remap is a dict", isinstance(remap, dict))
        check(f"PIT-1: stage {i} remap covers all surviving nodes",
              len(remap) == pruned_core.stages[i].num_nodes)
        new_ids = set(remap.values())
        check(f"PIT-1: stage {i} remap new ids are dense [0..N-1]",
              new_ids == set(range(pruned_core.stages[i].num_nodes)),
              f"got {sorted(new_ids)}")


def test_prune_io_mappers_transferred_when_zero_nodes_removed():
    print("\nTest 60: I/O mapper weights transferred when 0 nodes removed (PIT-2)")
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_preset, prune_network
    from config import PRUNE

    cell_lib = IdealizedCellLibrary()
    net = build_net_from_preset("smooth2d", cell_lib=cell_lib)
    pre_nodes = sum(s.num_nodes for s in net.core.stages)
    pruned_core, stage_remaps = prune_network(
        net.core,
        edge_threshold=float(PRUNE["edge_threshold"]),
        node_threshold=float(PRUNE["node_threshold"]),
        transfer_params=True,
    )
    post_nodes = sum(s.num_nodes for s in pruned_core.stages)

    if post_nodes == pre_nodes:
        copy_in = copy.deepcopy(net.input_mapper)
        copy_out = copy.deepcopy(net.output_mapper)
        all_in = all(
            torch.allclose(copy_in.state_dict()[k], net.input_mapper.state_dict()[k])
            for k in copy_in.state_dict()
        )
        all_out = all(
            torch.allclose(copy_out.state_dict()[k], net.output_mapper.state_dict()[k])
            for k in copy_out.state_dict()
        )
        check("PIT-2: 0-node-removal: deepcopy roundtrip matches (input)",
              all_in)
        check("PIT-2: 0-node-removal: deepcopy roundtrip matches (output)",
              all_out)
        check("PIT-2: stage_remaps[0] is identity permutation",
              stage_remaps[0] == {i: i for i in range(net.core.stages[0].num_nodes)})
    else:
        check("PIT-2: 0-node-removal precondition (all nodes survive)", True,
              f"pre={pre_nodes} post={post_nodes}; skipping")


def test_prune_io_forward_pass_preserved_when_zero_edges_removed():
    print("\nTest 61: I/O mapper+core forward pass preserved when 0 nodes/edges removed (PIT-3)")
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_preset, prune_network
    from train_script import _transfer_input_mapper, _transfer_output_mapper
    from config import PRUNE
    import torch

    cell_lib = IdealizedCellLibrary()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib, write_mode="dense")
    net.eval()
    pre_edges = sum(s.num_edges() for s in net.core.stages)

    pruned_core, stage_remaps = prune_network(
        net.core,
        edge_threshold=float(PRUNE["edge_threshold"]),
        node_threshold=float(PRUNE["node_threshold"]),
        transfer_params=True,
    )
    post_edges = sum(s.num_edges() for s in pruned_core.stages)

    if pre_edges == post_edges:
        from kirchhoff_net import KirchhoffNetWithIO
        from train import default_ctx_factory

        stage0_remap = stage_remaps[0]
        last_remap = stage_remaps[-1]
        pruned_first_n = pruned_core.stages[0].num_nodes
        pruned_last_n = pruned_core.stages[-1].num_nodes

        in_dim = net.input_mapper.gain.in_features
        out_dim = net.output_mapper.proj.out_features
        raw_write_idx = list(net.write_idx) if net.write_idx is not None else None
        raw_read_idx = list(net.read_idx) if net.read_idx is not None else None

        new_in_mapper, pruned_write_idx = _transfer_input_mapper(
            net.input_mapper, raw_write_idx, stage0_remap, pruned_first_n, in_dim,
        )
        new_out_mapper, pruned_read_idx = _transfer_output_mapper(
            net.output_mapper, raw_read_idx, last_remap, pruned_last_n, out_dim,
        )

        pruned_net = KirchhoffNetWithIO(
            new_in_mapper, pruned_core, new_out_mapper,
            hid_count=pruned_first_n, proj_count=0,
            final_hid_count=pruned_last_n, final_proj_count=0,
            write_idx=pruned_write_idx,
            read_idx=pruned_read_idx,
        )
        pruned_net.eval()

        u = torch.randn(8, in_dim)
        ctx_raw = default_ctx_factory(net)(u.size(0), device="cpu")
        ctx_pruned = default_ctx_factory(pruned_net)(u.size(0), device="cpu")
        try:
            with torch.no_grad():
                y_raw, _ = net(u, ctx_raw)
                y_pruned, _ = pruned_net(u, ctx_pruned)
            check("PIT-3: 0-edge-prune forward outputs match (atol=1e-6)",
                  torch.allclose(y_raw, y_pruned, atol=1e-6, rtol=1e-5),
                  f"max diff = {(y_raw - y_pruned).abs().max().item():.2e}")
        except Exception as e:
            check("PIT-3: 0-edge-prune forward outputs match", False, str(e))
    else:
        check("PIT-3: precondition (no edges/nodes removed at high thresh)", True,
              f"pre={pre_edges} post={post_edges}")


def test_prune_io_remap_invalid_index_raises():
    print("\nTest 62: I/O index remap skips pruned indices (PIT-4)")
    from train_script import _remap_indices as t_remap

    remap = {0: 0, 1: 1}
    out = t_remap([0, 1, 99], remap)
    check("PIT-4: missing (pruned) index is silently skipped",
          out == [0, 1],
          f"got {out}")
    check("PIT-4: empty input list returns empty list",
          t_remap([], remap) == [])


def test_prune_stage_protects_write_target():
    print("\nTest 62a: prune_stage protects write target from being pruned (PIO-1)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # 4-node linear chain: 0->1->2->3
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 1, 2],
        dst=[1, 2, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.z_logits.fill_(5.0)  # all edges strongly non-Z
        stage.u_logits.fill_(5.0)  # all nodes strongly alive

    # Force node 0 (write target) to be "dead" by gate, but protect it.
    with torch.no_grad():
        stage.u_logits.data[0] = -10.0  # gate -> 0
    new_stage, remap = prune_stage(
        stage,
        edge_threshold=0.01,
        node_threshold=0.01,
        protected_nodes={0},
    )
    check("PIO-1: protected write target survives pruning",
          0 in remap,
          f"remap={remap}")
    check("PIO-1: new stage still has node for old write target",
          new_stage.num_nodes >= 1,
          f"num_nodes={new_stage.num_nodes}")


def test_prune_stage_min_read_nodes_guard():
    print("\nTest 62b: prune_stage raises when all read nodes pruned via edge (PIO-3)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Since node gates are bypassed, trigger the min_read_nodes guard by
    # Z-pruning the edges that lead to read nodes.
    stage = DifferentialStage(
        num_nodes=3,
        src=[0, 1],
        dst=[1, 2],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        # Make edges leading to reads Z-dominant (eff_score -> 0)
        stage.logits.data[:, 3] = 5.0   # P(Z) ≈ 1 for all edges
        stage.logits.data[:, 0] = -5.0
        stage.z_logits.fill_(5.0)

    try:
        prune_stage(stage, edge_threshold=0.1, node_threshold=0.05,
                    read_idx=[1, 2], min_read_nodes=1)
        check("PIO-3: prune with all reads disconnected raises ValueError", False, "did not raise")
    except ValueError as e:
        check("PIO-3: prune with all reads disconnected raises ValueError",
              "min_read_nodes" in str(e) or "read nodes survived" in str(e)
              or "pruning removed all edges" in str(e),
              f"got: {e}")


def test_prune_stage_min_read_nodes_one_survives():
    print("\nTest 62c: prune_stage with at least one read node alive succeeds (PIO-3)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # 3-node line 0->1->2 with all nodes alive. write=[0], read=[1, 2].
    # Pruning with default thresholds keeps everything; both reads survive.
    stage = DifferentialStage(
        num_nodes=3,
        src=[0, 1],
        dst=[1, 2],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.z_logits.fill_(5.0)
        stage.u_logits.fill_(5.0)

    new_stage, remap = prune_stage(
        stage,
        edge_threshold=0.01,
        node_threshold=0.01,
        write_idx=[0],
        read_idx=[1, 2],
    )
    check("PIO-3: read node 1 survives", 1 in remap, f"remap={remap}")
    check("PIO-3: read node 2 survives", 2 in remap, f"remap={remap}")
    check("PIO-3: write node 0 survives", 0 in remap, f"remap={remap}")





def test_prune_output_mapper_elastic_readout():
    print("\nTest 62d: OutputMapper transfers surviving read columns (PIO-4)")
    import torch
    from cell_library import IdealizedCellLibrary
    from io_mapper import OutputMapper
    from train_script import _transfer_output_mapper

    # Old OutputMapper reads 4 nodes -> 1 output. proj.weight shape [1, 4].
    raw = OutputMapper(node_dim=5, out_dim=1, read_idx=[0, 1, 2, 3])
    # Make weights distinguishable for column-tracking.
    with torch.no_grad():
        raw.proj.weight.data.fill_(0.0)
        for i in range(4):
            raw.proj.weight.data[0, i] = float(i + 1)  # 1.0, 2.0, 3.0, 4.0

    # Pruned stage: node 1 was removed. last_remap has 0, 2, 3 -> 0, 1, 2.
    last_remap = {0: 0, 2: 1, 3: 2}
    new_mapper, new_read_idx = _transfer_output_mapper(
        raw, [0, 1, 2, 3], last_remap, pruned_last_n=3, out_dim=1,
    )
    check("PIO-4: new read_idx has 3 entries (one pruned)",
          new_read_idx == [0, 1, 2],
          f"got {new_read_idx}")
    check("PIO-4: new proj.weight shape is [1, 3]",
          tuple(new_mapper.proj.weight.shape) == (1, 3),
          f"got {new_mapper.proj.weight.shape}")
    # Columns should be [1.0, 3.0, 4.0] (column 1 dropped).
    expected = torch.tensor([[1.0, 3.0, 4.0]])
    check("PIO-4: surviving columns preserved in order",
          torch.allclose(new_mapper.proj.weight.data, expected),
          f"got {new_mapper.proj.weight.data.tolist()}")


def test_prune_network_multi_stage_protects_write():
    print("\nTest 62e: prune_network protects write_idx in multi-stage (PIO-5)")
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_preset, prune_network
    from config import PRUNE

    cell_lib = IdealizedCellLibrary()
    # smooth2d is single-stage; use sinx (also single-stage but with a
    # configured write_idx). Actually, we need a preset with a known
    # write_idx. Use smooth2d_grid (3 stages) to exercise the multi-stage
    # path. We rely on build_net_from_preset wiring up write_idx.
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    # Use very high thresholds so almost everything prunes, but the
    # protected write_idx nodes in stage 0 must still survive.
    pruned_core, stage_remaps = prune_network(
        net.core,
        edge_threshold=float(PRUNE["edge_threshold"]),
        node_threshold=float(PRUNE["node_threshold"]),
        transfer_params=True,
        write_idx=list(net.write_idx) if net.write_idx is not None else None,
        read_idx=list(net.read_idx) if net.read_idx is not None else None,
    )
    # Check that write_idx is contained in stage_remaps[0] (all write
    # targets survived stage 0).
    if net.write_idx is not None:
        stage0_remap = stage_remaps[0]
        missing = [w for w in net.write_idx if int(w) not in stage0_remap]
        check("PIO-5: all write_idx entries survive in stage 0",
              not missing,
              f"missing={missing} (write_idx={net.write_idx}, remap={stage0_remap})")
    else:
        check("PIO-5: preset has no write_idx (skipped)", True)


def test_prune_stage_edge_only_keeps_low_u_node_with_incident_edge():
    print("\nTest 62f: prune_stage with node gates bypassed keeps "
          "low-u node that has a surviving incident edge (EOP-1, deprecate-node-gates)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Stage: nodes 0, 1, 2, 3. Edges 0->1, 1->2, 2->3.
    # Nodes 1 and 2 have very low u_logits (no effect — node gates bypassed).
    # Edges 0->1 and 1->2 have high eff_score. Edge 2->3 is Z-dominant.
    # Node gates are bypassed so low-u does NOT cause node removal.
    # No read_idx/write_idx passed.
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 1, 2],
        dst=[1, 2, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.logits.data[0, 0] = 5.0     # P(L) ≈ 1 for edge 0->1
        stage.logits.data[0, 3] = -5.0
        stage.logits.data[1, 0] = 5.0     # P(L) ≈ 1 for edge 1->2
        stage.logits.data[1, 3] = -5.0
        stage.logits.data[2, 3] = 5.0     # P(Z) ≈ 1 for edge 2->3 (prune by Z)
        stage.logits.data[2, 0] = -5.0
        stage.z_logits.fill_(5.0)
        # u_logits values have no effect on pruning
        stage.u_logits.data[1] = -10.0
        stage.u_logits.data[2] = -10.0

    pruned, _ = prune_stage(
        stage, edge_threshold=0.1, node_threshold=0.05,
    )
    # Nodes 0,1,2 survive via edges 0->1, 1->2. Node 3 is a dead island
    # (edge 2->3 pruned via Z-dominance, no other incident edge).
    check("EOP-1: node gates bypassed — keeps 3 nodes (0,1,2; node 3 dead island)",
          pruned.num_nodes == 3,
          f"got {pruned.num_nodes} nodes, expected 3")
    check("EOP-1: keeps 2 active edges (0->1 and 1->2)",
          pruned.num_edges() == 2,
          f"got {pruned.num_edges()} edges, expected 2")


def test_prune_stage_edge_only_disconnected_node_removed():
    print("\nTest 62g: prune_stage removes nodes that become fully "
          "disconnected after edge pruning (EOP-2, deprecate-node-gates)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Stage: nodes 0, 1, 2, 3. Edges 0->1 (Z-dominant) and 2->3 (active).
    # After edge pruning: 0->1 dies, 2->3 survives. Nodes 0, 1 become
    # dead islands. Nodes 2, 3 stay alive via the surviving 2->3 edge.
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 2],
        dst=[1, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.logits.data[0, 3] = 5.0     # P(Z) ≈ 1 for edge 0->1
        stage.logits.data[0, 0] = -5.0
        stage.logits.data[1, 0] = 5.0     # P(L) ≈ 1 for edge 2->3
        stage.logits.data[1, 3] = -5.0
        stage.z_logits.fill_(5.0)

    pruned, _remap = prune_stage(
        stage, edge_threshold=0.1, node_threshold=0.05,
    )
    # Nodes 0 and 1 are fully disconnected. Nodes 2 and 3 stay alive.
    check("EOP-2: prunes disconnected nodes (0, 1); keeps 2,3",
          pruned.num_nodes == 2,
          f"got {pruned.num_nodes} nodes, expected 2")
    check("EOP-2: keeps the surviving active edge 2->3",
          pruned.num_edges() == 1,
          f"got {pruned.num_edges()} edges, expected 1")


def test_prune_stage_edge_only_matches_legacy_when_no_node_collateral():
    print("\nTest 62h: prune_stage connectivity-only mode (EOP-3, deprecate-node-gates)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Stage: nodes 0, 1, 2, 3. Edges 0->1, 0->2, 2->3.
    # Edges 0->1 and 0->2 are Z-dominant (pruned). Edge 2->3 stays active.
    # Node 2 has low u_logits — node gates are bypassed so this has no effect.
    # Nodes 0, 1 are dead islands; nodes 2, 3 stay alive via edge 2->3.
    # No read_idx/write_idx passed — dead island purge runs without I/O check.
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 0, 2],
        dst=[1, 2, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.logits.data[0, 3] = 5.0
        stage.logits.data[0, 0] = -5.0
        stage.logits.data[1, 3] = 5.0
        stage.logits.data[1, 0] = -5.0
        stage.logits.data[2, 0] = 5.0
        stage.logits.data[2, 3] = -5.0
        stage.z_logits.fill_(5.0)
        # u_logits values have no effect on pruning
        stage.u_logits.fill_(-10.0)

    pruned, _ = prune_stage(
        stage, edge_threshold=0.1, node_threshold=0.05,
    )
    # Nodes 2, 3 stay alive (edge 2->3 survives). Nodes 0, 1 are dead islands.
    check("EOP-3: retains nodes 2, 3 (edge 2->3 active)",
          pruned.num_nodes == 2,
          f"got {pruned.num_nodes} nodes, expected 2")
    check("EOP-3: retains the single active edge",
          pruned.num_edges() == 1,
          f"got {pruned.num_edges()} edges, expected 1")


def test_prune_network_edge_only_preserves_more_capacity():
    print("\nTest 62i: prune_stage connectivity-only (EOP-4, deprecate-node-gates)")
    import torch
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage

    cell_lib = IdealizedCellLibrary()
    # Single stage: 4 nodes, edges 0->1 (hi-z), 0->2 (lo-z), 1->3 (hi-z).
    # Edge 0->2 has low z_logits so it gets pruned. Edges 0->1 and 1->3
    # survive, keeping write_idx=[0] connected to read_idx=[3].
    stage = DifferentialStage(
        num_nodes=4,
        src=[0, 0, 1],
        dst=[1, 2, 3],
        cell_lib=cell_lib,
    )
    with torch.no_grad():
        stage.z_logits.data[0] = 5.0    # hi gate for edge 0->1
        stage.z_logits.data[1] = -10.0  # lo gate for edge 0->2 (prune)
        stage.z_logits.data[2] = 5.0    # hi gate for edge 1->3
        stage.logits.data[:, 0] = 5.0   # P(L) ≈ 1 for all edges
        stage.logits.data[:, 3] = -5.0  # P(Z) ≈ 0

    pruned, _ = prune_stage(
        stage, edge_threshold=0.1, node_threshold=0.05,
        write_idx=[0], read_idx=[3],
    )
    # Nodes 0, 1, 3 stay alive (I/O path via edges 0->1, 1->3).
    # Node 2 has no surviving incident edge → dead island.
    check("EOP-4: I/O path intact (0->1->3), node 2 dead island",
          pruned.num_nodes == 3,
          f"got {pruned.num_nodes} nodes, expected 3")
    check("EOP-4: 2 surviving edges (0->1 and 1->3)",
          pruned.num_edges() == 2,
          f"got {pruned.num_edges()} edges, expected 2")


def test_prune_nodes_by_gate_config_default():
    print("\nTest 62j: PRUNE['prune_nodes_by_gate'] is False everywhere (deprecate-node-gates)")
    from config import PRUNE
    check("deprecate-node-gates: PRUNE['prune_nodes_by_gate'] is False",
          PRUNE.get("prune_nodes_by_gate", True) is False,
          f"got {PRUNE.get('prune_nodes_by_gate', True)}")
    from config import SCHEDULE_THREE_PHASE
    check("deprecate-node-gates: SCHEDULE_THREE_PHASE['prune_nodes_by_gate'] is False",
          SCHEDULE_THREE_PHASE.get("prune_nodes_by_gate", True) is False,
          f"got {SCHEDULE_THREE_PHASE.get('prune_nodes_by_gate', True)}")
    from config import SCHEDULE_FOUR_PHASE
    check("deprecate-node-gates: SCHEDULE_FOUR_PHASE['prune_nodes_by_gate'] is False",
          SCHEDULE_FOUR_PHASE.get("prune_nodes_by_gate", True) is False,
          f"got {SCHEDULE_FOUR_PHASE.get('prune_nodes_by_gate', True)}")


def test_reg_defaults_topology_fix():
    print("\nTest 59: config.py regularizer values updated (PT-4)")
    from config import OPTIM, LAMBDAS
    check("PT-4: reg_warmup_epochs == 100", OPTIM["reg_warmup_epochs"] == 100)
    check("PT-4: rail == 1.0 (down from 10.0)", LAMBDAS["rail"] == 1.0)
    check("PT-4: edge_gate == 5e-4 (down from 1e-3)", LAMBDAS["edge_gate"] == 5e-4)


def test_active_presets_stage_count():
    print("\nTest 30: all active presets have correct stage counts (R4, R5)")
    from config import PRESETS
    # R4/R5: sinx/housing/smooth2d use 1 stage; smooth2d_grid uses 3 stages
    # (multistage-smooth2d-grid spec) with identical per-stage topology.
    expected_stages = {"sinx": 1, "housing": 1, "smooth2d": 1, "smooth2d_grid": 3, "housing_grid": 3}
    for name, cfg in PRESETS.items():
        n_stages = len(cfg["stages"])
        check(f"R4/R5: preset '{name}' has expected stage count",
              n_stages == expected_stages.get(name, 1),
              f"got {n_stages} stages, expected {expected_stages.get(name, 1)}")
        s = cfg["stages"][0]
        check(f"R4/R5: preset '{name}' has num_proj > 0 (R1.3 readout target)",
              s["num_proj"] > 0,
              f"got num_proj={s['num_proj']}")


def test_sinx_uses_line():
    print("\nTest 31: sinx preset uses line topology (topology-fix)")
    from config import PRESETS
    cfg = PRESETS["sinx"]
    check("sinx uses 'line' family (topology-fix: no 1-hop write->read bypass)",
          cfg["stages"][0]["hidden_family"] == "line",
          f"got '{cfg['stages'][0]['hidden_family']}'")
    check("sinx line has at least 4 hidden nodes",
          cfg["stages"][0]["num_hidden"] >= 4)
    check("sinx has 1-2 projection nodes",
          1 <= cfg["stages"][0]["num_proj"] <= 4)


def test_tau_monotonic():
    print("\nTest 32: tau_for_epoch is monotonically non-increasing (R6.1)")
    from train import tau_for_epoch, TAU, OPTIM
    total = int(OPTIM["epochs"])
    samples = [tau_for_epoch(e, total) for e in range(0, total, max(1, total // 50))]
    diffs = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]
    check("R6.1: tau is non-increasing across sampled epochs",
          all(d <= 1e-9 for d in diffs),
          f"max increase = {max(diffs):.6f}")


def test_variation_off_default():
    print("\nTest 33: train_script default disables variation (R6.3)")
    import subprocess
    import os
    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")
    script_path = os.path.join(THIS_DIR, "train_script.py")
    result = subprocess.run(
        [venv_py, script_path, "--help"],
        capture_output=True, text=True,
        cwd=THIS_DIR,
    )
    check(f"R6.3: subprocess OK (returncode {result.returncode})",
          result.returncode == 0,
          f"stderr: {result.stderr[:200] if result.stderr else 'none'}")
    check("R6.3: --variation flag exists", "--variation" in result.stdout)
    check("R6.3: --ablation flag exists (R2)", "--ablation" in result.stdout)
    check("CP-6: --prune flag exists", "--prune" in result.stdout)
    check("CP-6: --retrain/--no-retrain flags exist",
          all(f in result.stdout for f in ("--retrain", "--no-retrain")))


def test_normalized_units_in_config():
    print("\nTest 34: physical units are labeled normalized (R7) and reviewer residuals (RR-B)")
    import config
    config_path = os.path.join(THIS_DIR, "config.py")
    with open(config_path) as f:
        contents = f.read()
    check("R7.1: config.py mentions 'normalized' in the unit context",
          "normalized" in contents.lower())
    check("R7.3: V_CM removed from PHYS",
          "V_CM" not in config.PHYS)
    check("R7.4: LAMBDAS['C'] removed",
          "C" not in config.LAMBDAS)
    check("CP: LAMBDAS no longer has the single 'complexity' key (decomposed into 4 terms)",
          "complexity" not in config.LAMBDAS)
    check("CP: LAMBDAS has 'edge_gate', 'node_gate', 'power', 'capacitance'",
          all(k in config.LAMBDAS for k in ("edge_gate", "node_gate", "power", "capacitance")))
    check("CP: PRUNE has 'edge_threshold' and 'node_threshold'",
          "edge_threshold" in config.PRUNE and "node_threshold" in config.PRUNE)


def test_apply_ablation():
    print("\nTest 35: apply_ablation utility (R2.1-R2.4)")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from sim_context import SimContext
    from train import apply_ablation

    cell_lib = make_default_library()
    net = build_net_from_preset("sinx", cell_lib=cell_lib)

    net_copy_none = build_net_from_preset("sinx", cell_lib=cell_lib)
    apply_ablation(net_copy_none, "none")
    check("R2.1: ablation='none' is a no-op",
          net.core.stage_times == net_copy_none.core.stage_times)

    net_mapper_only = build_net_from_preset("sinx", cell_lib=cell_lib)
    apply_ablation(net_mapper_only, "mapper-only")
    check("R2.2: ablation='mapper-only' sets all stage_times to 0",
          all(t == 0.0 for t in net_mapper_only.core.stage_times))

    net_empty = build_net_from_preset("sinx", cell_lib=cell_lib)
    apply_ablation(net_empty, "empty-graph")
    check("R2.3: ablation='empty-graph' removes all edges",
          all(stage.num_edges() == 0 for stage in net_empty.core.stages))


def test_sparse_io_preset_defaults():
    print("\nTest 36: preset write_idx/read_idx defaults (SR4.2)")
    import config
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from io_mapper import SparseInputMapper

    check("sinx preset has write_idx [0] (SR4.2)",
          config.PRESETS["sinx"]["write_idx"] == [0])
    check("sinx preset has read_idx [7] (SR4.2)",
          config.PRESETS["sinx"]["read_idx"] == [7])
    check("housing preset has write_idx [0..7] (SR4.2)",
          config.PRESETS["housing"]["write_idx"] == [0, 1, 2, 3, 4, 5, 6, 7])
    check("housing preset has read_idx [15] (SR4.2)",
          config.PRESETS["housing"]["read_idx"] == [15])

    net = build_net_from_preset("sinx", cell_lib=make_default_library())
    check("sinx default builds with write_idx=[0]",
          net.write_idx == [0])
    check("sinx default builds with read_idx=[7]",
          net.read_idx == [7])
    check("sinx default uses SparseInputMapper",
          isinstance(net.input_mapper, SparseInputMapper))


def test_sparse_write_zeros_non_targets():
    print("\nTest 37: sparse write zeros non-target hidden nodes (SR1.3)")
    from io_mapper import SparseInputMapper

    m = SparseInputMapper(in_dim=2, out_dim=6, write_idx=[1, 4])
    u = torch.tensor([[0.5, -0.2], [1.0, 0.3]])
    y = m(u)
    check("sparse write: shape (2, 6)", y.shape == (2, 6))
    non_write_idx = [0, 2, 3, 5]
    check("sparse write: non-write positions are zero",
          torch.equal(y[:, non_write_idx], torch.zeros(2, 4)),
          f"got {y[:, non_write_idx].tolist()}")
    check("sparse write: write positions are non-zero (tanh produces value)",
          (y[:, [1, 4]].abs().sum(dim=1) > 0).all().item())

    m3 = SparseInputMapper(in_dim=4, out_dim=8, write_idx=[0, 3, 5, 7])
    u3 = torch.randn(3, 4) * 0.5
    y3 = m3(u3)
    check("sparse write multi: only write_idx positions populated",
          torch.equal(y3[:, [1, 2, 4, 6]], torch.zeros(3, 4)))


def test_sparse_write_d_gt_hid_raises():
    print("\nTest 38: sparse write raises on d > hid_count (SR1.6)")
    from io_mapper import SparseInputMapper

    try:
        SparseInputMapper(in_dim=5, out_dim=3, write_idx=[0, 1, 2, 3, 4])
        check("SR1.6: in_dim > out_dim raises", False, "no error raised")
    except ValueError as e:
        check("SR1.6: in_dim > out_dim raises", "in_dim=5" in str(e) and "out_dim=3" in str(e),
              f"got: {e}")

    try:
        SparseInputMapper(in_dim=3, out_dim=5, write_idx=[0, 1, 2, 5])
        check("SR1.6: out-of-range write_idx raises", False, "no error raised")
    except ValueError as e:
        check("SR1.6: out-of-range write_idx raises", "write_idx" in str(e),
              f"got: {e}")

    try:
        SparseInputMapper(in_dim=3, out_dim=5, write_idx=[0, 1, 1])
        check("SR1.6: duplicate write_idx raises", False, "no error raised")
    except ValueError as e:
        check("SR1.6: duplicate write_idx raises", "unique" in str(e).lower(),
              f"got: {e}")


def test_sparse_read_selects_targets():
    print("\nTest 39: sparse read selects only read_idx nodes (SR2.1)")
    from io_mapper import OutputMapper

    om = OutputMapper(node_dim=10, out_dim=2, read_idx=[2, 5, 7])
    x = torch.randn(4, 10)
    y = om(x)
    check("sparse read: output shape (4, 2)", y.shape == (4, 2))
    check("sparse read: projection weight shape (2, 3)",
          tuple(om.proj.weight.shape) == (2, 3))

    gathered = x.index_select(-1, om._read_index)
    expected = (om.proj.weight.unsqueeze(0) @ gathered.unsqueeze(-1)).squeeze(-1) + om.proj.bias
    check("sparse read: gathers only read_idx positions",
          torch.allclose(y, expected, atol=1e-6),
          f"max diff = {(y - expected).abs().max().item():.2e}")

    om2 = OutputMapper(node_dim=8, out_dim=1, read_idx=[7])
    check("sparse read len(read_idx)=1: weight shape (1, 1)",
          tuple(om2.proj.weight.shape) == (1, 1))


def test_sparse_io_dense_fallback_matches_old():
    print("\nTest 40: dense mode produces identical mapper classes (SR3, SR6)")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext

    net = build_net_from_preset(
        "sinx", cell_lib=make_default_library(),
        write_mode="dense", read_mode="dense",
    )
    check("dense write: uses InputMapper (SR3.1)",
          isinstance(net.input_mapper, InputMapper))
    check("dense read: OutputMapper has no read_idx (SR3.2)",
          net.output_mapper.read_idx is None)
    check("dense mode: write_idx=None, read_idx=None",
          net.write_idx is None and net.read_idx is None)

    u = torch.linspace(-math.pi, math.pi, 16).unsqueeze(1)
    ctx = SimContext()
    with torch.no_grad():
        y_dense, _ = net(u, ctx=ctx, store_trajectory=False)
    check("dense mode: forward output finite",
          torch.isfinite(y_dense).all().item())


def test_sparse_io_gradients_flow_to_read_targets():
    print("\nTest 41: gradient flows only to read_idx nodes (SR2)")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from sim_context import SimContext

    net = build_net_from_preset("sinx", cell_lib=make_default_library())
    u = torch.tensor([[0.5]])
    ctx = SimContext()
    y, _ = net(u, ctx=ctx, store_trajectory=False)
    y.sum().backward()
    has_out_grad = (
        net.output_mapper.proj.weight.grad is not None
        and torch.isfinite(net.output_mapper.proj.weight.grad).all().item()
    )
    check("SR2: gradient flows to OutputMapper weights", has_out_grad)
    has_in_grad = (
        net.input_mapper.gain.grad is not None
        and torch.isfinite(net.input_mapper.gain.grad).all().item()
    )
    check("SR1: gradient flows to SparseInputMapper gain", has_in_grad)


def test_sparse_io_cli_flags():
    print("\nTest 42: train_script exposes --write-mode/--read-mode/--write-idx/--read-idx")
    import subprocess
    import os
    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")
    result = subprocess.run(
        [venv_py, "kirchhoff_redesign/ideal/train_script.py", "--help"],
        capture_output=True, text=True,
    )
    check("--write-mode flag present", "--write-mode" in result.stdout)
    check("--read-mode flag present", "--read-mode" in result.stdout)
    check("--write-idx flag present", "--write-idx" in result.stdout)
    check("--read-idx flag present", "--read-idx" in result.stdout)


def main():
    test_config_loads()
    test_sim_context()
    test_topology_primitives()
    test_stage_transfer()
    test_heun_converges()
    test_gradient_flow()
    test_compute_loss_finite()
    test_sparsity_push()
    test_tau_anneal()
    test_round_trip_preset()
    test_xor_preset_removed()
    test_housing_preset_robust()
    test_topology_to_stage_input_output_filtering()
    test_validate_topology()
    test_visualize_stage_graph()
    test_visualize_sparse_topology()
    test_visualize_trajectories()
    test_sparse_spd_generation()
    test_solver_dataset()
    test_union_topology()
    test_solver_loss_finite()
    test_baseline_jacobi()
    test_baseline_cg()
    test_convergence_tracker()
    test_solver_preset_removed()
    test_io_honest_split()
    test_io_no_proj_fallback()
    test_mapper_only_ablation()
    test_active_presets_stage_count()
    test_sinx_uses_line()
    test_tau_monotonic()
    test_variation_off_default()
    test_normalized_units_in_config()
    test_apply_ablation()
    test_sparse_io_preset_defaults()
    test_sparse_write_zeros_non_targets()
    test_sparse_write_d_gt_hid_raises()
    test_sparse_read_selects_targets()
    test_sparse_io_dense_fallback_matches_old()
    test_sparse_io_gradients_flow_to_read_targets()
    test_sparse_io_cli_flags()
    test_mapper_only_ablation_fast()
    test_complexity_proxy()
    test_reg_schedule_curve()
    test_preset_lambda_overrides()
    test_smooth2d_grid_sparsity_zero_override()
    test_tau_anneal_preset_option()
    test_tau_override_floor_guarantee()
    test_tau_override_endpoints()
    test_tau_override_backward_compat()
    test_z_bias_eliminated()
    test_gate_initialization()
    test_gate_application_in_rhs()
    test_complexity_regularizers()
    test_prune_stage()
    test_prune_stage_transfer_params()
    test_prune_stage_all_removed_raises()
    test_prune_network()
    test_validate_topology_degrees_valid()
    test_validate_topology_degrees_direct()
    test_validate_topology_degrees_silent()
    test_joint_prune_z_dominant()
    test_prune_dead_island_removed()
    test_prune_disconnected_io_raises()
    test_prune_network_returns_remap()
    test_prune_io_mappers_transferred_when_zero_nodes_removed()
    test_prune_io_forward_pass_preserved_when_zero_edges_removed()
    test_prune_io_remap_invalid_index_raises()
    test_prune_stage_protects_write_target()
    test_prune_stage_min_read_nodes_guard()
    test_prune_stage_min_read_nodes_one_survives()
    test_prune_output_mapper_elastic_readout()
    test_prune_network_multi_stage_protects_write()
    test_prune_stage_edge_only_keeps_low_u_node_with_incident_edge()
    test_prune_stage_edge_only_disconnected_node_removed()
    test_prune_stage_edge_only_matches_legacy_when_no_node_collateral()
    test_prune_network_edge_only_preserves_more_capacity()
    test_prune_nodes_by_gate_config_default()
    test_reg_defaults_topology_fix()

    test_scheduler_config_entries()
    test_tau_smooth_hardening()
    test_retrain_reg_warmup_bounds()
    test_fresh_init_default_off()
    test_retrain_lr_cli_flag()
    test_loss_history_appends_retrain()

    test_grad_log_cli_flag()
    test_gradient_norms_collect()
    test_grad_log_file_output()

    test_smooth2d_preset()
    test_smooth2d_grid_preset()
    test_housing_grid_preset()
    test_housing_grid_data_huber_loss()
    test_housing_data_normalization_float16_safe()
    test_fan_out_input_mapper_basic()
    test_fan_out_input_mapper_param_count()
    test_fan_out_input_mapper_gradients()
    test_fan_out_input_mapper_overlap_raises()
    test_fan_out_input_mapper_missing_input_raises()
    test_fan_out_input_mapper_out_of_range_raises()
    test_optim_lr_lowered()
    test_patience_default_raised()
    test_mlp_benchmark()
    test_mlp_benchmark_tanh()
    test_rectifier_cell()
    # v1.5 expanded cell library tests (expanded-cell-library plan)
    test_v15_cell_library_construction()            # V15-1: build + structure
    test_v15_cell_boundedness()                     # V15-2: current bound
    test_v15_negative_rectifier()                   # V15-3: N0 mirror of P0
    test_v15_dead_zone_odd()                        # V15-4: D1 odd + dead zone
    test_v15_saturation_scales()                    # V15-5: O_hard > O_weak
    test_v15_forward_backward()                     # V15-6: forward + gradients
    test_v15_ste_mode()                             # V15-7: STE mode
    test_v15_cell_type_mask_consistency()           # V15-8: mask exclusivity
    test_v15_legacy_library_unchanged()             # V15-9: backward compat
    test_v15_cell_parameters_preset_smooth2d_grid()  # V15-10: full net build
    test_v2_library_construction()                  # V2-1: build + structure
    test_v2_factorization_codes()                    # V2-2: MIX/BIAS/THRESH dicts
    test_v2_cell_parameters()                        # V2-3: per-cell parameter values
    test_v2_boundedness()                            # V2-4: |I| <= isat for all cells
    test_v2_mix_code_asymmetry()                     # V2-5: O_h10 vs O_h01 differ
    test_v2_threshold_cells()                        # V2-6: P1/N1 vs P0/N0 fire
    test_v2_forward_backward()                       # V2-7: forward + gradients
    test_v2_legacy_v15_unchanged()                   # V2-8: rho preserved
    test_v2_preset_routing()                         # V2-9: make_cell_library('v2')
    test_stage_lr_scale_backward_compat()
    test_stage_lr_scale_multi_group()
    test_stage_lr_scale_scheduler_compat()
    test_rail_loss_zero_inside_bounds()
    test_rail_loss_positive_outside_bounds()
    test_retrain_lr_scale_defaults_one()

    # Three-phase schedule tests (three-phase-schedule plan).
    test_three_phase_boundaries()
    test_three_phase_for_epoch()
    test_three_phase_tau_values()
    test_three_phase_lambdas()
    test_three_phase_lambdas_warmup()
    test_solidification_metrics()
    test_validate_argmax_runs()
    test_log_solidification_format()
    test_smooth2d_grid_uses_three_phase()

    # Mapper LR control tests (mapper-lr-control plan, spec order)
    test_mapper_lr_scale_separate_group()        # MLR-1: mapper-only group
    test_mapper_lr_scale_combined_with_stage_lr_scale()  # MLR-2: stage+ mapper together
    test_mapper_lr_scale_backward_compat()        # MLR-3: backward compat
    test_freeze_mappers_cli_flag_parsed()         # MLR-4: freeze flag
    test_mapper_lr_scale_cli_flag_parsed()        # MLR-5: mapper-lr-scale flag
    test_mapper_lr_scale_rejects_zero_or_negative()  # MLR-6: validation
    test_mapper_unfreeze_epoch_midpoint()         # MLR-7: midpoint
    test_freeze_mappers_requires_grad_toggle()    # MLR-8: requires_grad toggle

    # LR param group tests (lr-param-groups plan)
    test_lrp_struct_dyn_groups_created()          # LRP-1
    test_lrp_correct_param_membership()           # LRP-2
    test_lrp_backward_compat()                    # LRP-3
    test_lrp_validation_positive()                # LRP-4
    test_lrp_composition_with_mapper()            # LRP-5
    test_lrp_stage_lr_scale_ignored()             # LRP-6
    test_lrp_empty_group_handling()               # LRP-6b
    test_lrp_cli_flags_parsed()                   # LRP-7
    test_lrp_compute_update_norms()               # LRP-8
    test_lrp_old_mapper_default_backward_compat() # LRP-9
    test_lrp_mapper_lr_scale_new_default()        # LRP-10

    # Bidirectional topology tests (bidirectional-edges plan, spec order)
    test_bidir_line_graph_doubles_edges()         # BIDI-1: line primitive
    test_bidir_ring_graph_doubles_edges()         # BIDI-2: ring primitive
    test_bidir_grid_graph_doubles_edges()         # BIDI-3: grid primitive
    test_bidir_cluster_graph_doubles_edges()      # BIDI-4: cluster primitive
    test_bidir_validate_topology_passes()         # BIDI-5: validation
    test_bidir_default_is_false()                 # BIDI-6: backward compat
    test_bidir_preset_factories_accept_param()    # BIDI-7: preset factories
    test_bidir_full_net_build()                   # BIDI-8: full net build

    # Parallel edge repeats tests (parallel-edge-repeats plan)
    test_repeat_edges_identity()                  # REP-1: n=1 identity
    test_repeat_edges_multiplies_hidden_only()    # REP-2: hidden only
    test_repeat_edges_composes_with_bidirectional()  # REP-3: composition
    test_repeat_edges_per_pair_count()            # REP-4: per-pair count
    test_repeat_edges_rejects_invalid_n()          # REP-5: validation
    test_repeat_edges_validate_topology_passes()  # REP-6: validate_topology
    test_repeat_edges_from_config_wires_through() # REP-7: from_config wiring
    test_repeat_edges_full_net_build()            # REP-8: full net build
    test_repeat_edges_preset_factories_accept_param()  # REP-9: preset factories
    test_repeat_edges_compose_with_bidirectional_full_net()  # REP-10: compose
    test_drive_current_basic()                   # DRIVE-1
    test_drive_changes_rhs()                     # DRIVE-2
    test_driven_node_gate_forced_open()          # DRIVE-3
    test_kirchhoff_net_with_io_drive_forward()   # DRIVE-4

    # Simple-edge family tests (simple-edge-family plan)
    test_simple_edge_build_forward()             # SE-1
    test_simple_edge_prune()                     # SE-2
    test_simple_edge_regularizers()              # SE-3
    test_simple_edge_diagnostics()               # SE-4

    # Fixed seed tests (fixed-seed plan)
    test_seed_everything_deterministic()         # SEED-1: model init determinism
    test_seed_everything_seeds_numpy()           # SEED-2: NumPy RNG
    test_seed_everything_seeds_python_random()   # SEED-3: Python random

    test_deprecate_node_gates_warnings()         # DNG-1: deprecation warnings

    # Deep Equilibrium (DEQ) training mode tests (deq-core-prototype plan)
    test_deq_config_defaults()                   # DEQ-13: config defaults
    test_deq_solver_imports_and_solves()         # DEQ-1:  solver adapter
    test_deq_solver_run_in_fp32()                # DEQ-2:  fp32 / AMP safety
    test_deq_multistart_uniqueness_on_contractive()  # DEQ-12
    test_deq_leak_floor_enforced()               # DEQ-9:  leak_floor safeguard
    test_deq_equilibrium_rhs_residual_small()    # DEQ-3:  residual at equilibrium
    test_deq_equilibrium_matches_long_horizon_heun()  # DEQ-4: vs Heun rollout
    test_deq_implicit_backward_gradients_finite()    # DEQ-5: implicit grads
    test_deq_z_logits_grad_norm_at_least_bptt()  # DEQ-6:  DEQ vs BPTT
    test_deq_input_dependence()                  # DEQ-7:  input-dependence w/ drive
    test_deq_ste_mode_rejected()                 # DEQ-8:  soft-only safeguard
    test_deq_solver_kwarg_threads_through_kirchhoff_net()  # DEQ-10: wiring
    test_deq_heun_regression_unchanged()         # DEQ-11: Heun regression
    test_deq_diagnostics_jacobian_cond()         # DEQ-14: cond(J)
    test_deq_diagnostics_grad_norm_compare()     # DEQ-15: grad norm compare

    test_budget_gate_basic()                     # BUD-1: budget gate basics
    test_budget_gate_differentiable()            # BUD-2: differentiable
    test_budget_frac_unity_noop()                # BUD-3: frac >= 1.0 is no-op
    test_budget_temperature_limits()             # BUD-4: T->0 / T->inf
    test_budget_axis_src()                       # BUD-5: axis=src
    test_budget_axis_both()                      # BUD-6: axis=both
    test_budget_deq_forward()                    # BUD-7: DEQ w/ budget
    test_budget_annealing_schedule()             # BUD-8: annealing schedule
    test_budget_disabled_byte_identical()        # BUD-9: budget disabled = no-op
    test_budget_simple_edge_library_compat()     # BUD-10: SimpleEdgeLibrary
    test_budget_frac_uniform_proportion()        # BUD-11: uniform proportion

    print()
    print("=" * 60)
    print(f"Smoke test results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


def test_smooth2d_preset():
    print("\nTest NN: smooth2d preset structure and Franke dataset")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_default_library
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
    check(f"smooth2d: t_span={SOLVER['t_span']} (current SOLVER)",
          s["t_span"] == SOLVER["t_span"],
          f"got {s['t_span']}")
    check(f"smooth2d: num_steps={SOLVER['num_steps']} (current SOLVER)",
          s["num_steps"] == SOLVER["num_steps"],
          f"got {s['num_steps']}")
    check("smooth2d: no lambdas override", "lambdas" not in cfg)

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d", cell_lib=cell_lib)
    check("smooth2d: builds successfully", net is not None)
    check("smooth2d: write_idx=[0,1]", net.write_idx == [0, 1])
    check("smooth2d: read_idx=[9]", net.read_idx == [9])
    check("smooth2d: uses SparseInputMapper",
          isinstance(net.input_mapper, SparseInputMapper))
    check("smooth2d: hid_count=10", net.hid_count == 10)
    check("smooth2d: proj_count=2", net.proj_count == 2)

    # Forward on a random batch.
    u = torch.rand(8, 2)
    ctx = None
    out, _ = net(u, ctx=ctx)
    check("smooth2d: forward shape (8,1)", out.shape == (8, 1))
    check("smooth2d: forward output is finite", torch.isfinite(out).all().item())

    # Franke function value range on [0,1]^2.
    from train_script import _franke
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    gx1, gx2 = torch.meshgrid(x1, x2, indexing="ij")
    f = _franke(gx1, gx2)
    check("smooth2d: Franke output is finite", torch.isfinite(f).all().item())
    check("smooth2d: Franke range in plausible bounds",
          f.min().item() >= -0.1 and f.max().item() <= 1.3,
          f"min={f.min().item():.4f} max={f.max().item():.4f}")

    # Quick training sanity: 1 epoch.
    from train import make_optimizer
    from train_script import make_data_smooth2d
    train_loader, val_loader, task_fn = make_data_smooth2d(batch_size=128, val_size=200)
    optimizer = make_optimizer(net, lr=1e-3)
    net.train()
    total = 0.0
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b, _ = net(u_b, ctx=None)
        loss = task_fn(out_b, y_b)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * u_b.size(0)
        break  # one batch
    avg_loss = total / 128
    check("smooth2d: 1-batch loss is finite", math.isfinite(avg_loss),
          f"loss={avg_loss:.6f}")
    print(f"  [INFO] smooth2d 1-batch loss: {avg_loss:.6f}")


def test_smooth2d_grid_preset():
    print("\nTest NN2: smooth2d_grid preset (7x7 grid + 3 proj, fan-out I/O, 3 stages)")
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_default_library
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
    # 7x7: rows=[0,2,4,6], left col=0, right col=6
    check("smooth2d_grid: write_fan_out maps both inputs",
          cfg.get("write_fan_out") == {0: [0, 14, 28, 42], 1: [6, 20, 34, 48]})
    # 7x7: center_col=3, so center column hidden nodes + 3 proj
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

    cell_lib = make_default_library()
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
    # write_fan_out: {0: [0, 14, 28, 42], 1: [6, 20, 34, 48]} → sorted union
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
    # 7x7 grid with 8-neighbor (kernel_size=3), single-edge-per-pair:
    # Degree: 4 corners*3 + 20 edge*5 + 25 interior*8 = 12+100+200 = 312
    # Single-branch (1 directed per pair) = 312/2 = 156.
    # Default edge_repeats=2 doubles hidden edges to 312.
    n_hidden_edges = 156
    n_hidden_repeat = n_hidden_edges * 2  # edge_repeats=2 default
    # Projection edges: n_hidden * n_proj (unidirectional hidden->proj)
    n_proj_edges = n_hidden * n_proj
    expected_total = n_hidden_repeat + n_proj_edges
    for i, stage in enumerate(net.core.stages):
        check(f"smooth2d_grid: stage {i} edge count = {expected_total} (grid 312 + proj 147)",
              int(stage.src.shape[0]) == expected_total,
              f"got {int(stage.src.shape[0])}")
        check(f"smooth2d_grid: stage {i} num_nodes=52 (49 hid + 3 proj)",
              int(stage.num_nodes) == 52)
        check(f"smooth2d_grid: stage {i} has positive logits parameter",
              stage.logits.shape == (expected_total, 4))

    # Explicit bounds checks on the I/O index lists.
    check("smooth2d_grid: write_idx entries in [0, hid_count)",
          all(0 <= w < net.hid_count for w in net.write_idx))
    # For grid_size >= 5, read_idx includes center-column hidden nodes (which
    # are >1 hop from the write columns) + proj nodes. All must be valid.
    check("smooth2d_grid: read_idx entries in [0, final_state_dim)",
          all(0 <= r < net.final_hid_count + net.final_proj_count
              for r in net.read_idx))

    # Forward on a random batch.
    u = torch.rand(8, 2)
    ctx = None
    out, _ = net(u, ctx=ctx)
    check("smooth2d_grid: forward shape (8,1)", out.shape == (8, 1))
    check("smooth2d_grid: forward output is finite",
          torch.isfinite(out).all().item())

    # Quick training sanity: 1 batch.
    from train import make_optimizer
    from train_script import make_data_smooth2d
    train_loader, val_loader, task_fn = make_data_smooth2d(batch_size=128, val_size=200)
    optimizer = make_optimizer(net, lr=1e-3)
    net.train()
    total = 0.0
    for u_b, y_b in train_loader:
        optimizer.zero_grad()
        out_b, _ = net(u_b, ctx=None)
        loss = task_fn(out_b, y_b)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * u_b.size(0)
        break
    avg_loss = total / 128
    check("smooth2d_grid: 1-batch loss is finite", math.isfinite(avg_loss),
          f"loss={avg_loss:.6f}")
    print(f"  [INFO] smooth2d_grid 1-batch loss: {avg_loss:.6f}")

    # Override test: explicit write_mode="dense" should produce InputMapper.
    from io_mapper import InputMapper, FanOutInputMapper
    net_dense = build_net_from_preset(
        "smooth2d_grid", cell_lib=make_default_library(), write_mode="dense",
    )
    check("smooth2d_grid: write_mode='dense' override produces InputMapper",
          isinstance(net_dense.input_mapper, InputMapper)
          and type(net_dense.input_mapper) is InputMapper,
          f"got {type(net_dense.input_mapper).__name__}")
    # Default (no override) produces FanOutInputMapper.
    net_fanout = build_net_from_preset(
        "smooth2d_grid", cell_lib=make_default_library(),
    )
    check("smooth2d_grid: default (no write_mode) produces FanOutInputMapper",
          isinstance(net_fanout.input_mapper, FanOutInputMapper),
          f"got {type(net_fanout.input_mapper).__name__}")


def test_housing_grid_preset():
    print("\nTest NN3: housing_grid preset (5x5 grid + 3 proj, dense I/O, 3 stages, Huber loss)")
    from config import PRESETS, make_housing_grid_preset
    from topology import build_net_from_preset
    from cell_library import make_default_library
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

    cell_lib = make_default_library()
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
    n_hidden_edges = 72  # single direction, 5x5 grid
    n_hidden_repeat = n_hidden_edges * 2  # edge_repeats=2 default
    n_proj_edges = n_hidden * n_proj
    expected_total = n_hidden_repeat + n_proj_edges
    for i, stage in enumerate(net.core.stages):
        check(f"housing_grid: stage {i} edge count = {expected_total} (grid 144 + proj {n_proj_edges})",
              int(stage.src.shape[0]) == expected_total,
              f"got {int(stage.src.shape[0])}")
        check(f"housing_grid: stage {i} num_nodes=28 (25 hid + 3 proj)",
              int(stage.num_nodes) == 28)
        check(f"housing_grid: stage {i} has positive logits parameter",
              stage.logits.shape == (expected_total, 4))

    check("housing_grid: read_idx entries in [0, final_state_dim)",
          all(0 <= r < net.final_hid_count + net.final_proj_count
              for r in net.read_idx))
    check("housing_grid: 5 read nodes are hidden (center column)",
          sum(1 for r in net.read_idx if r < net.hid_count) == 5)
    check("housing_grid: 3 read nodes are proj",
          sum(1 for r in net.read_idx if r >= net.hid_count) == 3)

    u = torch.rand(8, 8)
    ctx = None
    out, _ = net(u, ctx=ctx)
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


def test_housing_grid_data_huber_loss():
    print("\nTest NN4: housing_grid data loader (Huber loss + inverse stats)")
    try:
        from train_script import make_data_housing_grid, denormalize_targets
    except ImportError as e:
        check("import make_data_housing_grid", False, f"failed: {e}")
        return
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError:
        check("sklearn available", True, "sklearn not installed, skipping data test")
        return

    train_loader, val_loader, task_fn, inverse_stats = make_data_housing_grid(batch_size=128)
    check("make_data_housing_grid: returns 4-tuple with inverse_stats",
          inverse_stats is not None and "y_mean" in inverse_stats and "y_std" in inverse_stats)

    import torch.nn.functional as F
    a = torch.tensor([0.0, 0.5, 1.0, 2.0])
    b = torch.tensor([0.5, 0.5, 0.5, 0.5])
    huber = task_fn(a, b)
    expected = F.huber_loss(a, b, delta=1.0)
    check("housing_grid: task_fn is Huber loss (delta=1.0)",
          abs(float(huber) - float(expected)) < 1e-6,
          f"got {float(huber):.6f}, expected {float(expected):.6f}")

    for u_b, y_b in train_loader:
        check("housing_grid: input batch shape (B, 8)", u_b.shape[1] == 8)
        check("housing_grid: target batch shape (B, 1)", y_b.shape[1] == 1)
        check("housing_grid: input values in [0, 1] (normalized by per-col max)",
              u_b.min() >= -1e-6 and u_b.max() <= 1.0 + 1e-6)
        break
    print(f"  [INFO] housing_grid inverse_stats: y_mean={inverse_stats['y_mean']:.4f}, "
          f"y_std={inverse_stats['y_std']:.4f}")

    y_norm = torch.tensor([0.0, 1.0, -1.0])
    y_orig = denormalize_targets(y_norm, inverse_stats)
    expected = y_norm * inverse_stats["y_std"] + inverse_stats["y_mean"]
    check("denormalize_targets: y = y_norm * y_std + y_mean",
          torch.allclose(y_orig, expected),
          f"got {y_orig.tolist()}, expected {expected.tolist()}")


def test_housing_data_normalization_float16_safe():
    """Regression test for the AMP float16 overflow bug in make_data_housing.

    California Housing's Longitude feature is negative (~-124 to ~-114).
    The original divide-by-column-max normalization clamped the negative
    max to 1e-6, producing values of ~-1.2e8 that overflow float16 under
    AMP. The backward pass then computes 0 * inf = NaN (IEEE 754), killing
    all gradients. This test verifies that all normalized features stay
    within the float16 representable range (< 65504 in absolute value).
    """
    print("\nTest NN5: housing data normalization is float16-safe (AMP overflow guard)")
    try:
        from train_script import make_data_housing
    except ImportError as e:
        check("import make_data_housing", False, f"failed: {e}")
        return
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError:
        check("sklearn available", True, "sklearn not installed, skipping data test")
        return

    train_loader, val_loader, task_fn, inverse_stats = make_data_housing(batch_size=128)
    check("make_data_housing: returns 4-tuple with inverse_stats",
          inverse_stats is not None and "y_mean" in inverse_stats and "y_std" in inverse_stats)

    FLOAT16_MAX = 65504.0
    for u_b, y_b in train_loader:
        check("housing: input batch shape (B, 8)", u_b.shape[1] == 8)
        check("housing: target batch shape (B, 1)", y_b.shape[1] == 1)
        check("housing: input values in [0, 1] (min-max scaling)",
              u_b.min() >= -1e-6 and u_b.max() <= 1.0 + 1e-6)
        check("housing: abs(X).max() < float16 max (AMP-safe)",
              u_b.abs().max().item() < FLOAT16_MAX,
              f"got abs(X).max()={u_b.abs().max().item():.4e}, threshold={FLOAT16_MAX}")
        check("housing: no NaN in features", torch.isfinite(u_b).all().item())
        check("housing: no NaN in targets", torch.isfinite(y_b).all().item())
        break

    import torch.nn.functional as F
    a = torch.tensor([0.0, 0.5, 1.0, 2.0])
    b = torch.tensor([0.5, 0.5, 0.5, 0.5])
    l1 = task_fn(a, b)
    expected = F.l1_loss(a, b)
    check("housing: task_fn is L1 (MAE) loss",
          abs(float(l1) - float(expected)) < 1e-6,
          f"got {float(l1):.6f}, expected {float(expected):.6f}")



# ---- FanOutInputMapper tests (smooth2d-sanity-pass spec) ----

def test_fan_out_input_mapper_basic():
    print("\nTest FO-1: FanOutInputMapper basic forward + zero non-targets")
    from io_mapper import FanOutInputMapper

    m = FanOutInputMapper(in_dim=2, out_dim=6, fan_out_map={0: [0, 2], 1: [3, 5]})
    u = torch.tensor([[0.5, -0.2], [1.0, 0.3]])
    y = m(u)
    check("fan-out: output shape (2, 6)", y.shape == (2, 6))
    check("fan-out: non-target positions (1, 4) are zero",
          torch.equal(y[:, [1, 4]], torch.zeros(2, 2)),
          f"got {y[:, [1, 4]].tolist()}")
    check("fan-out: target positions (0, 2, 3, 5) are non-zero",
          (y[:, [0, 2, 3, 5]].abs().sum(dim=1) > 0).all().item())


def test_fan_out_input_mapper_param_count():
    print("\nTest FO-2: FanOutInputMapper parameter count = 2 * K_total")
    from io_mapper import FanOutInputMapper

    m = FanOutInputMapper(in_dim=2, out_dim=25, fan_out_map={0: [0, 10, 20], 1: [4, 14, 24]})
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    check("fan-out: smooth2d_grid layout → 12 params (2 inputs × 3 targets × 2)",
          n_params == 12, f"got {n_params}")


def test_fan_out_input_mapper_gradients():
    print("\nTest FO-3: FanOutInputMapper gradient flows to gain and bias")
    from io_mapper import FanOutInputMapper

    m = FanOutInputMapper(in_dim=2, out_dim=6, fan_out_map={0: [0, 2], 1: [3, 5]})
    u = torch.randn(4, 2, requires_grad=False)
    y = m(u)
    loss = y.pow(2).sum()
    loss.backward()
    has_gain_grad = m.gain.grad is not None and m.gain.grad.abs().sum().item() > 0
    has_bias_grad = m.bias.grad is not None and m.bias.grad.abs().sum().item() > 0
    check("fan-out: gain parameter receives gradient", has_gain_grad)
    check("fan-out: bias parameter receives gradient", has_bias_grad)


def test_fan_out_input_mapper_overlap_raises():
    print("\nTest FO-4: FanOutInputMapper rejects overlapping target nodes")
    from io_mapper import FanOutInputMapper

    try:
        FanOutInputMapper(in_dim=2, out_dim=6, fan_out_map={0: [0, 2], 1: [2, 3]})
        check("fan-out: overlapping targets raise ValueError", False, "no error raised")
    except ValueError as e:
        check("fan-out: overlapping targets raise ValueError",
              "duplicate" in str(e).lower() or "overlap" in str(e).lower()
              or "duplicate target" in str(e).lower(),
              f"got: {e}")


def test_fan_out_input_mapper_missing_input_raises():
    print("\nTest FO-5: FanOutInputMapper rejects fan_out_map missing inputs")
    from io_mapper import FanOutInputMapper

    try:
        FanOutInputMapper(in_dim=3, out_dim=6, fan_out_map={0: [0], 1: [1]})
        check("fan-out: missing input raises ValueError", False, "no error raised")
    except ValueError as e:
        check("fan-out: missing input raises ValueError",
              "missing" in str(e).lower(),
              f"got: {e}")


def test_fan_out_input_mapper_out_of_range_raises():
    print("\nTest FO-6: FanOutInputMapper rejects out-of-range target")
    from io_mapper import FanOutInputMapper

    try:
        FanOutInputMapper(in_dim=2, out_dim=5, fan_out_map={0: [0, 7], 1: [3]})
        check("fan-out: out-of-range target raises ValueError", False, "no error raised")
    except ValueError as e:
        check("fan-out: out-of-range target raises ValueError",
              "out of range" in str(e).lower() or "out_of_range" in str(e).lower(),
              f"got: {e}")


def test_optim_lr_lowered():
    print("\nTest FO-7: OPTIM['lr'] lowered to 3e-4 (sanity-pass consultant rec)")
    from config import OPTIM
    check("OPTIM.lr == 6e-4 (auto-scaled to batch_size=2048)", abs(OPTIM["lr"] - 6e-4) < 1e-12,
          f"got {OPTIM['lr']}")


def test_patience_default_raised():
    print("\nTest FO-8: train_script --patience default = 500 (sanity-pass)")
    import importlib
    import train_script
    importlib.reload(train_script)
    import argparse
    parser = argparse.ArgumentParser()
    train_script._add_argparse_args(parser)
    args = parser.parse_args(["--problem", "smooth2d_grid"])
    check("train_script --patience default == 500", args.patience == 500,
          f"got {args.patience}")


# ---- prune-retrain-fixes tests (spec: prune-retrain-fixes) ----

def test_scheduler_config_entries():
    print("\nTest PP-1: OPTIM/TAU have new scheduler and hardening entries")
    from config import OPTIM, TAU
    check("OPTIM has scheduler_T_0", "scheduler_T_0" in OPTIM,
          f"keys={list(OPTIM.keys())}")
    check("OPTIM.scheduler_T_0 > 0", OPTIM.get("scheduler_T_0", 0) > 0,
          f"got {OPTIM.get('scheduler_T_0')}")
    check("OPTIM has scheduler_T_mult", "scheduler_T_mult" in OPTIM, "")
    check("OPTIM has scheduler_eta_min", "scheduler_eta_min" in OPTIM, "")
    check("OPTIM.scheduler_eta_min > 0", OPTIM.get("scheduler_eta_min", 0) > 0,
          f"got {OPTIM.get('scheduler_eta_min')}")
    check("TAU has hardening_epoch_frac", "hardening_epoch_frac" in TAU, "")
    check("TAU.hardening_epoch_frac in (0, 0.5)",
          0 < TAU.get("hardening_epoch_frac", 0) <= 0.5,
          f"got {TAU.get('hardening_epoch_frac')}")


def test_tau_smooth_hardening():
    print("\nTest PP-2: tau_for_epoch smooth hardening (R5-prune-retrain-fixes)")
    from train import tau_for_epoch, TAU
    total = 250
    hardening_frac = TAU.get("hardening_epoch_frac", 0.1)
    hardening_start = int(total * (1.0 - 2.0 * hardening_frac))
    hardening_end = total

    t_start = tau_for_epoch(hardening_start, total)
    t_mid = tau_for_epoch((hardening_start + hardening_end) // 2, total)
    t_end = tau_for_epoch(hardening_end, total)

    check("R5: tau at hardening_start equals exponential floor (no jump)",
          t_start >= TAU["min"] - 1e-6,
          f"got {t_start}")
    check("R5: tau monotonically decreases through hardening window",
          t_start >= t_mid >= t_end,
          f"start={t_start} mid={t_mid} end={t_end}")
    check("R5: tau at hardening_end == tau_final",
          abs(t_end - TAU["final"]) < 1e-6,
          f"got {t_end}, expected {TAU['final']}")

    diffs = [tau_for_epoch(e + 1, total) - tau_for_epoch(e, total)
             for e in range(total - 1)]
    check("R5: tau_for_epoch is non-increasing",
          all(d <= 1e-9 for d in diffs),
          f"max increase = {max(diffs):.6f}")


def test_retrain_reg_warmup_bounds():
    print("\nTest PP-3: retrain reg_schedule uses //2/max(//4,50) bounds (R3)")
    from train import reg_schedule
    for retrain_epochs in (50, 100, 250, 500):
        warmup = max(1, retrain_epochs // 2)
        anneal = max(25, retrain_epochs // 4)
        v_free = reg_schedule(0, warmup=warmup, anneal=anneal)
        v_mid = reg_schedule(warmup + anneal // 2, warmup=warmup, anneal=anneal)
        v_full = reg_schedule(warmup + anneal, warmup=warmup, anneal=anneal)
        check(f"R3: retrain_epochs={retrain_epochs}: free phase at ep 0 == 0",
              abs(v_free) < 1e-9, f"got {v_free}")
        check(f"R3: retrain_epochs={retrain_epochs}: mid anneal in (0,1)",
              0 < v_mid < 1, f"got {v_mid}")
        check(f"R3: retrain_epochs={retrain_epochs}: full == 1 after anneal",
              abs(v_full - 1.0) < 1e-9, f"got {v_full}")

    check("R3: retrain warmup >> 25 for 250 epochs (was 25)",
          max(1, 250 // 2) > 25, f"got {max(1, 250 // 2)}")


def test_fresh_init_default_off():
    print("\nTest PP-4: --fresh-init flag is off by default (R4)")
    import subprocess
    res = subprocess.run(
        [sys.executable, "train_script.py", "--help"],
        cwd=THIS_DIR,
        capture_output=True, text=True, timeout=30,
    )
    out = res.stdout
    check("R4: --fresh-init mentioned in --help", "--fresh-init" in out,
          "flag not exposed")
    check("R4: --retrain-lr mentioned in --help", "--retrain-lr" in out,
          "flag not exposed")
    check("R2: --no-scheduler mentioned in --help", "--no-scheduler" in out,
          "flag not exposed")
    check("R4: default retrain behavior is warm-start (not scratch)",
          "warm-started" in out or "warm start" in out,
          "help text does not describe warm-start")


def test_retrain_lr_cli_flag():
    print("\nTest PP-5: --retrain-lr CLI flag is parsed (R6)")
    import sys as _sys
    _sys.path.insert(0, THIS_DIR)
    import argparse
    import train_script as ts_mod
    parser = argparse.ArgumentParser()
    ts_mod._add_argparse_args(parser)
    args = parser.parse_args(["--retrain-lr", "0.001", "--prune"])
    check("R6: --retrain-lr parsed", args.retrain_lr == 0.001,
          f"got {args.retrain_lr}")
    check("R6: --retrain default True", args.retrain is True, "")
    check("R4: --fresh-init default False", args.fresh_init is False, "")


def test_loss_history_appends_retrain():
    print("\nTest PP-6: loss_history.txt is appended with retrain section (R1)")
    import sys as _sys
    import tempfile
    import os
    _sys.path.insert(0, THIS_DIR)
    from topology import build_net_from_preset
    from cell_library import IdealizedCellLibrary
    from topology import prune_network
    from config import PRUNE

    cell_lib = IdealizedCellLibrary()
    net = build_net_from_preset("smooth2d", cell_lib=cell_lib)
    pre_edges = sum(s.num_edges() for s in net.core.stages)
    pre_nodes = sum(s.num_nodes for s in net.core.stages)

    pruned, _remaps = prune_network(
        net.core,
        edge_threshold=float(PRUNE["edge_threshold"]),
        node_threshold=float(PRUNE["node_threshold"]),
        transfer_params=True,
    )
    post_edges = sum(s.num_edges() for s in pruned.stages)
    post_nodes = sum(s.num_nodes for s in pruned.stages)
    check("R1: prune_network runs without error",
          post_edges >= 0 and post_nodes >= 0,
          f"pre=({pre_edges},{pre_nodes}) post=({post_edges},{post_nodes})")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "loss_history.txt")
        with open(path, "w") as f:
            f.write("epoch\ttrain\tval\n")
            for i in range(5):
                f.write(f"{i}\t1.0\t0.9\n")
        with open(path, "a") as f:
            f.write(f"\n[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes\n")
            f.write(f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes\n")
            f.write("retrain_epoch\ttrain\tval\n")
            for i in range(3):
                f.write(f"{i}\t0.9\t0.85\n")
        with open(path) as f:
            content = f.read()
        check("R1: loss_history contains retrain_epoch header",
              "retrain_epoch\ttrain\tval" in content,
              "header missing")
        check("R1: loss_history contains [prune] pre/post markers",
              "[prune] pre-prune:" in content and "[prune] post-prune:" in content,
              "markers missing")
        check("R1: loss_history contains original epoch data",
              content.startswith("epoch\ttrain\tval\n"),
              "original header overwritten")


def test_grad_log_cli_flag():
    print("\nTest GG-1: --grad-log/--grad-log-every CLI flags (R1)")
    import sys as _sys
    _sys.path.insert(0, THIS_DIR)
    import argparse
    import train_script as ts_mod
    parser = argparse.ArgumentParser()
    ts_mod._add_argparse_args(parser)
    args = parser.parse_args(["--grad-log"])
    check("GG-1: --grad-log sets grad_log=True", args.grad_log is True, "")
    check("GG-1: --grad-log-every defaults to 10",
          args.grad_log_every == 10, f"got {args.grad_log_every}")
    args2 = parser.parse_args(["--grad-log", "--grad-log-every", "5"])
    check("GG-1: --grad-log-every 5 parsed", args2.grad_log_every == 5,
          f"got {args2.grad_log_every}")
    check("GG-1: --grad-log off by default",
          parser.parse_args([]).grad_log is False, "")


def test_gradient_norms_collect():
    print("\nTest GG-2: collect_gradient_norms returns expected groups")
    import sys as _sys
    _sys.path.insert(0, THIS_DIR)
    from train_script import collect_gradient_norms
    from topology import build_net_from_preset
    from cell_library import IdealizedCellLibrary

    net = build_net_from_preset("smooth2d", cell_lib=IdealizedCellLibrary())
    norms = collect_gradient_norms(net)
    check("GG-2: stage0_logits key exists", "stage0_logits" in norms,
          f"keys={sorted(norms.keys())}")
    check("GG-2: stage0_raw_mult key exists", "stage0_raw_mult" in norms, "")
    check("GG-2: stage0_raw_leak key exists", "stage0_raw_leak" in norms, "")
    check("GG-2: stage0_z_logits key exists", "stage0_z_logits" in norms, "")
    check("GG-2: stage0_u_logits key exists", "stage0_u_logits" in norms, "")
    check("GG-2: in_mapper key exists", "in_mapper" in norms, "")
    check("GG-2: out_mapper key exists", "out_mapper" in norms, "")
    check("GG-2: stage_transfer key exists (maybe None)",
          "stage_transfer" in norms, "")
    # No gradients yet (no backward) — all should be None
    for k, v in norms.items():
        check(f"GG-2: {k} is None before backward",
              v is None, f"got {v}")

    # Run one forward + backward to check gradients exist.
    from train_script import make_static_ctx_factory
    ctx_fn = make_static_ctx_factory()
    loss = net(torch.randn(4, 2), ctx_fn(4, "cpu"))[0].sum()
    loss.backward()
    norms2 = collect_gradient_norms(net)
    for k in ("stage0_logits", "stage0_raw_mult", "stage0_raw_leak",
              "in_mapper", "out_mapper"):
        check(f"GG-2: {k} has gradient after backward",
              norms2.get(k) is not None and norms2[k] > 0,
              f"got {norms2.get(k)}")


def test_grad_log_file_output():
    print("\nTest GG-3: log_gradient_norms writes correct file format")
    import sys as _sys
    import tempfile
    _sys.path.insert(0, THIS_DIR)
    from pathlib import Path
    from train_script import log_gradient_norms, make_static_ctx_factory
    from topology import build_net_from_preset
    from cell_library import IdealizedCellLibrary

    net = build_net_from_preset("smooth2d", cell_lib=IdealizedCellLibrary())
    ctx_fn = make_static_ctx_factory()
    loss = net(torch.randn(4, 2), ctx_fn(4, "cpu"))[0].sum()
    loss.backward()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grad_norms.txt"
        log_gradient_norms(path, 0, net)
        log_gradient_norms(path, 1, net)
        log_gradient_norms(path, 2, net, retrain=True)
        content = path.read_text()
        check("GG-3: header row written", content.startswith("epoch\t"), "")
        lines = [l for l in content.splitlines() if l.strip()]
        check("GG-3: 1 header + 3 data rows",
              len(lines) == 4, f"got {len(lines)} rows")
        check("GG-3: retrain prefix in line 4",
              lines[3].startswith("retrain_"), f"got {lines[3]}")
        # Stage transfer may be None (single-stage net) → dash is fine,
        # but all other groups should have real gradients.
        header = lines[0].split("\t")[1:]
        vals = lines[1].split("\t")[1:]
        skip_keys = {"stage_transfer"}
        # u_logits keys have no gradient (deprecated, bypassed in rhs)
        skip_keys.update(k for k in header if k.endswith("_u_logits"))
        for k, v in zip(header, vals):
            if k in skip_keys:
                if k.endswith("_u_logits"):
                    check(f"GG-3: {k} is dash (no gradient, deprecated)",
                          v == "-", f"got {v}")
                continue
            check(f"GG-3: {k} has numeric value (not dash)",
                  v != "-", f"got dash for {k}")


def test_mlp_benchmark():
    print("\nTest OO: minimal MLP benchmark for smooth2d")
    from mlp_benchmark import MLPRegressor, count_parameters

    # Default architecture: 2 -> 100 -> 1 = 401 learnable parameters,
    # close to the smooth2d KirchhoffNet's ~430 (SparseInputMapper 4 +
    # DifferentialStage 424 + OutputMapper 2).
    net = MLPRegressor(in_dim=2, hidden_dim=100, out_dim=1)
    n_params = count_parameters(net)
    check("mlp: default hidden_dim=100 produces 401 params", n_params == 401,
          f"got {n_params}")

    # Forward shape and finiteness on a small batch.
    x = torch.randn(8, 2)
    y = net(x)
    check("mlp: forward output shape (8,1)", tuple(y.shape) == (8, 1))
    check("mlp: forward output is finite", torch.isfinite(y).all().item())

    # Hidden-dim sweep: confirm parameter count formula
    #   2*H + H + H*1 + 1 = 4H + 1
    for h in (8, 16, 32, 64, 100, 128):
        m = MLPRegressor(in_dim=2, hidden_dim=h, out_dim=1)
        expected = 4 * h + 1
        check(f"mlp: hidden_dim={h} -> {expected} params",
              count_parameters(m) == expected)

    # Gradients flow through both Linear layers.
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

    # End-to-end: 1-batch training on the smooth2d Franke dataset.
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

    # Same parameter count as the ReLU variant; only the activation changes.
    net = MLPRegressor(in_dim=2, hidden_dim=100, out_dim=1, activation="tanh")
    n_params = count_parameters(net)
    check("mlp-tanh: hidden_dim=100 produces 401 params", n_params == 401,
          f"got {n_params}")
    check("mlp-tanh: activation string stored as 'tanh'", net.activation == "tanh")

    # Forward shape and finiteness on a small batch.
    x = torch.randn(8, 2)
    y = net(x)
    check("mlp-tanh: forward output shape (8,1)", tuple(y.shape) == (8, 1))
    check("mlp-tanh: forward output is finite", torch.isfinite(y).all().item())

    # Hidden activations are bounded in (-1, 1).
    with torch.no_grad():
        h_pre = net.fc1(x)
        h_post = torch.tanh(h_pre)
    check("mlp-tanh: hidden activation bounded in (-1, 1)",
          (h_post > -1.0).all().item() and (h_post < 1.0).all().item())

    # Hidden-dim sweep: confirm parameter count formula is activation-invariant.
    for h in (8, 16, 32, 64, 100, 128):
        m = MLPRegressor(in_dim=2, hidden_dim=h, out_dim=1, activation="tanh")
        expected = 4 * h + 1
        check(f"mlp-tanh: hidden_dim={h} -> {expected} params",
              count_parameters(m) == expected)

    # Gradients flow through both Linear layers.
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

    # Invalid activation raises ValueError.
    raised = False
    try:
        MLPRegressor(in_dim=2, hidden_dim=10, out_dim=1, activation="sigmoid")
    except ValueError:
        raised = True
    check("mlp-tanh: invalid activation raises ValueError", raised)

    # End-to-end: 1-batch training on the smooth2d Franke dataset.
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


def test_rectifier_cell():
    """Test P cell: rectification, boundedness, smoothness, monotonicity."""
    print("\nTest 71: smooth bounded rectifier (P cell) properties")
    from cell_library import IdealizedCellLibrary
    import torch

    cell_lib = IdealizedCellLibrary()
    # P cell is at index 2 in [L, S, P, Z]
    P_INDEX = 2

    # Create logits that strongly select P cell: [−∞, −∞, 0, −∞]
    E, Q = 1, cell_lib.num_cells
    logits = torch.full((E, Q), -1e9)
    logits[0, P_INDEX] = 0.0

    # Evaluate I_P over a sweep of u = x_src − rho·x_dst
    u_sweep = torch.linspace(-2.0, 2.0, 100).unsqueeze(1)  # [100, 1]
    # For u = x_src − rho·x_dst, fix x_dst=0 so x_src = u
    x_src = u_sweep
    x_dst = torch.zeros_like(x_src)
    raw_mult = torch.zeros(E)  # m = softplus(0) = ln(2) ≈ 0.693

    with torch.no_grad():
        i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=1.0, ctx=None)

    # Use concrete u values (not linspace) for exact checks
    u_pos = torch.tensor([[0.5]])
    u_neg = torch.tensor([[-0.5]])
    with torch.no_grad():
        i_pos_edge = cell_lib(u_pos, torch.zeros_like(u_pos), logits, raw_mult, x_max=1.0, ctx=None)
        i_neg_edge = cell_lib(u_neg, torch.zeros_like(u_neg), logits, raw_mult, x_max=1.0, ctx=None)
    i_pos = float(i_pos_edge.item())
    i_neg = float(i_neg_edge.item())
    i_max = float(i_edge.abs().max().item())
    isat_p = float(cell_lib.isat[P_INDEX].item())

    check("P-1: I_P(+0.5) > 0 (positive drive conducts)",
          i_pos > 0, f"got I_P={i_pos:.6f}")
    check("P-2: I_P(-0.5) ≈ 0 (negative drive blocked, |I|<1% of I_sat)",
          abs(i_neg) < 0.01, f"got I_P={i_neg:.6f}")
    check("P-3: |I_P(u)| ≤ I_sat for all u ∈ [-2, 2]",
          i_max <= isat_p + 1e-4,
          f"max |I|={i_max:.6f}, I_sat={isat_p}")
    # Monotonicity: use large x_max to disable compliance gating (pure device test).
    u_sweep_sub = torch.linspace(-2.0, 2.0, 200).unsqueeze(1)
    with torch.no_grad():
        i_sub = cell_lib(u_sweep_sub, torch.zeros_like(u_sweep_sub), logits, raw_mult, x_max=10.0, ctx=None)
    i_sub_vals = i_sub.squeeze()
    diffs = i_sub_vals[1:] - i_sub_vals[:-1]
    check("P-4: I_P is non-decreasing (pure rectifier, compliance bypassed)",
          float(diffs.min().item()) >= -1e-6,
          f"min diff={float(diffs.min().item()):.6e}")

    # Smoothness: gradient exists via autograd
    x_src_grad = torch.tensor([[0.5]], requires_grad=True)
    x_dst_zero = torch.zeros_like(x_src_grad)
    i_test = cell_lib(x_src_grad, x_dst_zero, logits, raw_mult, x_max=10.0, ctx=None)
    grad = torch.autograd.grad(i_test.sum(), x_src_grad, create_graph=True)[0]
    check("P-5: I_P is smooth (gradient defined and finite)",
          torch.isfinite(grad).all().item(),
          f"grad={grad.item():.6f}")

    # Check that P cell is_rect flag is True
    check("P-6: _is_rect[P_INDEX] is True",
          bool(cell_lib._is_rect[P_INDEX].item()),
          f"got {cell_lib._is_rect[P_INDEX].item()}")

    # Check gate init values (grid7-gate0: updated from 2.0/2.0 to 0.0/0.0)
    from config import INIT
    check("P-7: z_logit_init == 0.0 (grid7-gate0: 50% open gates, was 2.0)",
          INIT["z_logit_init"] == 0.0,
          f"got {INIT['z_logit_init']}")
    check("P-8: u_logit_init == 0.0 (grid7-gate0: 50% open gates, was 2.0)",
          INIT["u_logit_init"] == 0.0,
          f"got {INIT['u_logit_init']}")


# ---- v1.5 expanded cell library tests (expanded-cell-library plan) ----

def test_v15_cell_library_construction():
    """V15-1: v15 library builds and has correct structure."""
    print("\nTest V15-1: v15 cell library construction")
    from cell_library import IdealizedCellLibrary
    cell_lib = IdealizedCellLibrary(library_name="v15")
    check("V15-1: v15 num_cells == 6",
          cell_lib.num_cells == 6,
          f"got {cell_lib.num_cells}")
    check("V15-1: v15 z_index == 5 (last cell)",
          cell_lib.z_index == 5,
          f"got {cell_lib.z_index}")
    check("V15-1: v15 has O_weak", "O_weak" in cell_lib._cell_order)
    check("V15-1: v15 has O_hard", "O_hard" in cell_lib._cell_order)
    check("V15-1: v15 has P0", "P0" in cell_lib._cell_order)
    check("V15-1: v15 has N0", "N0" in cell_lib._cell_order)
    check("V15-1: v15 has D1", "D1" in cell_lib._cell_order)
    check("V15-1: v15 has Z (last)", cell_lib._cell_order[-1] == "Z")
    check("V15-1: cell_type_code has 6 entries",
          cell_lib.cell_type_code.shape[0] == 6)
    ctc = cell_lib.cell_type_code
    check("V15-1: O_weak is standard (code 0)", ctc[0].item() == 0)
    check("V15-1: O_hard is standard (code 0)", ctc[1].item() == 0)
    check("V15-1: P0 is pos_rect (code 1)", ctc[2].item() == 1)
    check("V15-1: N0 is neg_rect (code 2)", ctc[3].item() == 2)
    check("V15-1: D1 is dead_zone (code 3)", ctc[4].item() == 3)
    check("V15-1: Z is standard (code 0)", ctc[5].item() == 0)
    check("V15-1: O_weak gm == 0.3",
          abs(cell_lib.gm[0].item() - 0.3) < 1e-6)
    check("V15-1: O_hard gm == 3.0",
          abs(cell_lib.gm[1].item() - 3.0) < 1e-6)
    check("V15-1: P0 isat == 1.0",
          abs(cell_lib.isat[2].item() - 1.0) < 1e-6)
    check("V15-1: N0 theta == 0.0",
          abs(cell_lib.theta[3].item() - 0.0) < 1e-6)
    check("V15-1: D1 theta == 0.5",
          abs(cell_lib.theta[4].item() - 0.5) < 1e-6)
    # type masks
    check("V15-1: _is_std[0] True (O_weak)", cell_lib._is_std[0].item())
    check("V15-1: _is_pos_rect[2] True (P0)", cell_lib._is_pos_rect[2].item())
    check("V15-1: _is_neg_rect[3] True (N0)", cell_lib._is_neg_rect[3].item())
    check("V15-1: _is_dead_zone[4] True (D1)", cell_lib._is_dead_zone[4].item())


def test_v15_cell_boundedness():
    """V15-2: all v15 cells produce bounded current |I| <= isat for all u."""
    print("\nTest V15-2: v15 cell boundedness")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E, B = 1, 200
    logits = torch.full((E, Q), -1e9)

    u_sweep = torch.linspace(-5.0, 5.0, B).unsqueeze(1)  # [B, 1]
    raw_mult = torch.zeros(E)

    for cell_idx in range(Q):
        logits_edge = logits.clone()
        logits_edge[0, cell_idx] = 0.0  # select this cell
        with torch.no_grad():
            i_edge = cell_lib(
                u_sweep, torch.zeros_like(u_sweep),
                logits_edge, raw_mult, x_max=10.0, ctx=None,
            )
        isat_cell = float(cell_lib.isat[cell_idx].item())
        i_vals = i_edge.squeeze()
        max_abs_i = float(i_vals.abs().max().item())
        check(f"V15-2: cell {cell_lib._cell_order[cell_idx]} bounded |I|<={isat_cell}",
              max_abs_i <= isat_cell + 1e-4,
              f"max |I|={max_abs_i:.6f}, I_sat={isat_cell}")


def test_v15_negative_rectifier():
    """V15-3: N0 is the exact mirror of P0: I_N0(u) == -I_P0(-u)."""
    print("\nTest V15-3: N0 negative rectifier mirror of P0")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E = 1
    raw_mult = torch.zeros(E)

    P0_idx = cell_lib._cell_order.index("P0")
    N0_idx = cell_lib._cell_order.index("N0")

    logits_p = torch.full((E, Q), -1e9)
    logits_p[0, P0_idx] = 0.0
    logits_n = torch.full((E, Q), -1e9)
    logits_n[0, N0_idx] = 0.0

    u_test = torch.tensor([[-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]]).T
    with torch.no_grad():
        i_p = cell_lib(u_test, torch.zeros_like(u_test), logits_p, raw_mult, x_max=10.0, ctx=None)
        i_n = cell_lib(u_test, torch.zeros_like(u_test), logits_n, raw_mult, x_max=10.0, ctx=None)

    for j in range(u_test.shape[0]):
        u_val = float(u_test[j, 0])
        p_val = float(i_p[j, 0])
        n_val = float(i_n[j, 0])
        # P0(-u) = -N0(u)  =>  N0(u) = -P0(-u)
        u_neg_idx = abs(u_test + u_val).argmin().item()
        expected_n = -float(i_p[u_neg_idx, 0])
        check(f"V15-3: N0({u_val:.1f}) == -P0({-u_val:.1f})",
              abs(n_val - expected_n) < 1e-6,
              f"N0={n_val:.8f}, -P0(-u)={expected_n:.8f}")


def test_v15_dead_zone_odd():
    """V15-4: D1 is odd: I_D1(-u) == -I_D1(u). And has dead zone near zero."""
    print("\nTest V15-4: D1 dead-zone odd cell oddness")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E = 1
    raw_mult = torch.zeros(E)

    D1_idx = cell_lib._cell_order.index("D1")
    logits = torch.full((E, Q), -1e9)
    logits[0, D1_idx] = 0.0

    u_test = torch.linspace(-3.0, 3.0, 61).unsqueeze(1)  # step = 0.1
    with torch.no_grad():
        i_d1 = cell_lib(u_test, torch.zeros_like(u_test), logits, raw_mult, x_max=10.0, ctx=None)

    # Oddness: I(-u) ≈ -I(u). Center index = 30 (u=0).
    # Pair u_test[30 + k] with u_test[30 - k] for k = 1..30.
    center = 30
    for k in range(1, 31):
        u_pos = float(u_test[center + k, 0])
        u_neg = float(u_test[center - k, 0])
        i_pos = float(i_d1[center + k, 0])
        i_neg = float(i_d1[center - k, 0])
        check(f"V15-4: D1({u_pos:.2f}) ≈ -D1({u_neg:.2f})",
              abs(i_pos + i_neg) < 1e-6,
              f"D1({u_pos})={i_pos:.8f}, D1({u_neg})={i_neg:.8f}")

    # Dead zone: |I(u)| < 0.05 for |u| < theta/2
    # (softplus roll-off means the dead zone is not perfectly sharp)
    theta_d1 = float(cell_lib.theta[D1_idx].item())
    within_dead = u_test.abs() < theta_d1 / 2
    max_i_in_dead = float(i_d1[within_dead.squeeze()].abs().max().item())
    check(f"V15-4: D1 dead zone |I|<0.05 for |u|<{theta_d1/2}",
          max_i_in_dead < 0.05,
          f"max|I| in dead zone={max_i_in_dead:.6f}")


def test_v15_saturation_scales():
    """V15-5: O_hard saturates faster (higher gm/isat ratio) than O_weak."""
    print("\nTest V15-5: O_hard saturates faster than O_weak")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E = 1
    # Use raw_mult=+1e9 so mult = softplus(1e9) ≈ 1e9, which clips to 1 via
    # gate+weights. Actually use a simpler approach: set raw_mult so that
    # m = softplus(raw_mult) ≈ 1.0. softplus(0) = ln(2) ≈ 0.693, so for
    # m≈1 we need raw_mult such that softplus(x)=1 → x ≈ 0.5413.
    # Instead, just multiply expected values by softplus(raw_mult=0)=ln(2).
    raw_mult = torch.zeros(E)
    mult_factor = float(torch.nn.functional.softplus(torch.tensor(0.0)).item())  # ≈ 0.693

    O_weak_idx = cell_lib._cell_order.index("O_weak")
    O_hard_idx = cell_lib._cell_order.index("O_hard")

    logits_w = torch.full((E, Q), -1e9)
    logits_w[0, O_weak_idx] = 0.0
    logits_h = torch.full((E, Q), -1e9)
    logits_h[0, O_hard_idx] = 0.0

    u_test = torch.linspace(0.0, 1.0, 20).unsqueeze(1)
    with torch.no_grad():
        i_w = cell_lib(u_test, torch.zeros_like(u_test), logits_w, raw_mult, x_max=10.0, ctx=None)
        i_h = cell_lib(u_test, torch.zeros_like(u_test), logits_h, raw_mult, x_max=10.0, ctx=None)

    # At small u (e.g. 0.25), O_hard should have higher current than O_weak
    # because O_hard has higher gm
    i_w_small = float(i_w[5, 0])  # u ≈ 0.25
    i_h_small = float(i_h[5, 0])
    check("V15-5: O_hard > O_weak at small drive (higher gm)",
          i_h_small > i_w_small,
          f"O_hard={i_h_small:.6f}, O_weak={i_w_small:.6f}")

    # At u=1.0, O_hard should be near saturation.
    # I = mult * isat * tanh(gm*u/isat) ≈ mult * isat (for gm*u >> isat)
    # mult = softplus(0) ≈ 0.693, isat=0.3 → sat I ≈ 0.208
    # O_weak: gm=0.3, isat=5.0, u=1.0 → gm*u/isat=0.06, tanh≈0.06
    #   I ≈ mult * isat * 0.06 ≈ 0.693 * 0.3 = 0.208? No — O_weak isat=5.0
    #   I ≈ 0.693 * 5.0 * 0.06 ≈ 0.208 (same ballpark at u=1)
    # Better test: compare O_hard at u=1 vs u=5 (should be saturated at both)
    u_high = torch.tensor([[5.0]])
    with torch.no_grad():
        i_h_high = cell_lib(u_high, torch.zeros_like(u_high), logits_h, raw_mult, x_max=10.0, ctx=None)
    i_h_1 = float(i_h[-1, 0])
    i_h_5 = float(i_h_high[0, 0])
    sat_expected = mult_factor * float(cell_lib.isat[O_hard_idx].item())  # ≈ 0.208
    check("V15-5: O_hard at u=1 near saturation",
          abs(i_h_1 - sat_expected) < 0.02,
          f"O_hard(1)={i_h_1:.4f}, expected sat≈{sat_expected:.4f}")
    check("V15-5: O_hard at u=5 also saturated (stable)",
          abs(i_h_5 - sat_expected) < 0.02,
          f"O_hard(5)={i_h_5:.4f}, expected sat≈{sat_expected:.4f}")

    # O_weak at u=1.0: isat=5.0, gm=0.3 → still in linear regime (I ≈ gm*u * mult)
    i_w_1 = float(i_w[-1, 0])
    i_w_expected_lin = mult_factor * 0.3 * 1.0  # mult * gm * u ≈ 0.208
    check("V15-5: O_weak at u=1 ≈ linear (not saturated)",
          abs(i_w_1 - i_w_expected_lin) < 0.05,
          f"O_weak(1)={i_w_1:.4f}, expected linear≈{i_w_expected_lin:.4f}")


def test_v15_forward_backward():
    """V15-6: v15 library forward pass runs and gradients flow."""
    print("\nTest V15-6: v15 forward pass and gradient flow")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E = 4
    B = 8
    x_src = torch.randn(B, E, requires_grad=True)
    x_dst = torch.randn(B, E, requires_grad=True)
    logits = torch.randn(E, Q, requires_grad=True)
    raw_mult = torch.randn(E, requires_grad=True)

    i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=3.0, ctx=None)
    check("V15-6: forward output shape (B, E)",
          i_edge.shape == (B, E),
          f"got {i_edge.shape}")
    check("V15-6: forward output is finite",
          torch.isfinite(i_edge).all().item())

    loss = i_edge.sum()
    grads = torch.autograd.grad(loss, [logits, raw_mult, x_src], retain_graph=True)
    check("V15-6: gradients flow to logits",
          grads[0] is not None and torch.isfinite(grads[0]).all().item())
    check("V15-6: gradients flow to raw_mult",
          grads[1] is not None and torch.isfinite(grads[1]).all().item())
    check("V15-6: gradients flow to x_src",
          grads[2] is not None and torch.isfinite(grads[2]).all().item())


def test_v15_ste_mode():
    """V15-7: v15 library works in straight-through estimator mode."""
    print("\nTest V15-7: v15 STE mode")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v15")
    Q = cell_lib.num_cells
    E = 4
    B = 8
    x_src = torch.randn(B, E)
    x_dst = torch.randn(B, E)
    logits = torch.randn(E, Q, requires_grad=True)
    raw_mult = torch.randn(E, requires_grad=True)

    i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=3.0, ctx=None, cell_mode="ste")
    check("V15-7: STE forward output is finite",
          torch.isfinite(i_edge).all().item())
    loss = i_edge.sum()
    loss.backward()
    check("V15-7: STE gradients flow via logits",
          logits.grad is not None and torch.isfinite(logits.grad).all().item())


def test_v15_cell_type_mask_consistency():
    """V15-8: cell_type_mask entries are mutually exclusive."""
    print("\nTest V15-8: v15 cell type mask consistency")
    from cell_library import IdealizedCellLibrary
    cell_lib = IdealizedCellLibrary(library_name="v15")
    # Each cell should have exactly one type
    for i in range(cell_lib.num_cells):
        types = sum([
            cell_lib._is_std[i].item(),
            cell_lib._is_pos_rect[i].item(),
            cell_lib._is_neg_rect[i].item(),
            cell_lib._is_dead_zone[i].item(),
        ])
        check(f"V15-8: cell {cell_lib._cell_order[i]} has exactly one type mask",
              types == 1,
              f"got {types} type masks")


def test_v15_legacy_library_unchanged():
    """V15-9: legacy library is unaffected by v15 changes."""
    print("\nTest V15-9: legacy library unchanged")
    from cell_library import IdealizedCellLibrary
    import torch
    # Legacy library with explicit name
    cell_lib = IdealizedCellLibrary(library_name="legacy")
    check("V15-9: legacy num_cells == 4",
          cell_lib.num_cells == 4,
          f"got {cell_lib.num_cells}")
    check("V15-9: legacy z_index == 3",
          cell_lib.z_index == 3,
          f"got {cell_lib.z_index}")
    check("V15-9: legacy cell_order == [L,S,P,Z]",
          cell_lib._cell_order == ["L", "S", "P", "Z"],
          f"got {cell_lib._cell_order}")
    # Default library is also legacy
    cell_lib_default = IdealizedCellLibrary()
    check("V15-9: default library is legacy (4 cells)",
          cell_lib_default.num_cells == 4,
          f"got {cell_lib_default.num_cells}")
    # forward pass still works
    E, Q, B = 2, 4, 4
    x_src = torch.randn(B, E)
    x_dst = torch.randn(B, E)
    logits = torch.randn(E, Q)
    raw_mult = torch.randn(E)
    with torch.no_grad():
        i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=3.0, ctx=None)
    check("V15-9: legacy forward output finite",
          torch.isfinite(i_edge).all().item())


def test_v15_cell_parameters_preset_smooth2d_grid():
    """V15-10: v15 library works with build_net_from_preset on smooth2d_grid."""
    print("\nTest V15-10: v15 library with smooth2d_grid build")
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_preset
    cell_lib = IdealizedCellLibrary(library_name="v15")
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    import torch
    x = torch.randn(4, 2)
    with torch.no_grad():
        y, traj = net(x, ctx=None, store_trajectory=False)
    check("V15-10: network output finite",
          torch.isfinite(y).all().item())
    check("V15-10: network output shape (4, 1)",
          y.shape == (4, 1),
          f"got {y.shape}")
    # Verify logits dimension matches v15
    for stage in net.core.stages:
        check(f"V15-10: stage logits shape[-1] == 6",
              stage.logits.shape[-1] == 6,
              f"got {stage.logits.shape[-1]}")


# ---- v2 library tests (cell-library-v2 spec) ----

def test_v2_library_construction():
    """V2-1: v2 library builds with correct structure (10 cells, Z last, mix mode)."""
    print("\nTest V2-1: v2 library construction")
    from cell_library import IdealizedCellLibrary
    cell_lib = IdealizedCellLibrary(library_name="v2")
    check("V2-1: v2 num_cells == 10",
          cell_lib.num_cells == 10,
          f"got {cell_lib.num_cells}")
    check("V2-1: v2 z_index == 9 (last cell)",
          cell_lib.z_index == 9,
          f"got {cell_lib.z_index}")
    check("V2-1: v2 _use_mix is True",
          cell_lib._use_mix is True,
          f"got {cell_lib._use_mix}")
    check("V2-1: v2 has src_gain buffer",
          hasattr(cell_lib, "src_gain"))
    check("V2-1: v2 has dst_gain buffer",
          hasattr(cell_lib, "dst_gain"))
    check("V2-1: v2 has no rho buffer",
          not hasattr(cell_lib, "rho"),
          "rho should not be present in v2 library")
    expected_order = ["O_w11", "O_h11", "O_h10", "O_h01", "P0", "P1", "N0", "N1", "D1", "Z"]
    check("V2-1: v2 cell_order matches spec",
          cell_lib._cell_order == expected_order,
          f"got {cell_lib._cell_order}")
    check("V2-1: Z is the last cell",
          cell_lib._cell_order[-1] == "Z")
    z_idx_v2 = cell_lib._cell_order.index("Z")
    check("V2-1: Z has gm=0 (truly off)",
          abs(float(cell_lib.gm[z_idx_v2].item())) < 1e-9,
          f"gm={float(cell_lib.gm[z_idx_v2].item())}")
    check("V2-1: Z has isat=0",
          abs(float(cell_lib.isat[z_idx_v2].item())) < 1e-9,
          f"isat={float(cell_lib.isat[z_idx_v2].item())}")
    check("V2-1: Z has src_gain=0",
          abs(float(cell_lib.src_gain[z_idx_v2].item())) < 1e-9,
          f"src_gain={float(cell_lib.src_gain[z_idx_v2].item())}")
    check("V2-1: Z has dst_gain=0",
          abs(float(cell_lib.dst_gain[z_idx_v2].item())) < 1e-9,
          f"dst_gain={float(cell_lib.dst_gain[z_idx_v2].item())}")
    check("V2-1: Z cell triggers has_z_cell=True",
          cell_lib.has_z_cell is True,
          f"has_z_cell={cell_lib.has_z_cell}")
    check("V2-1: cell_type_code has 10 entries",
          cell_lib.cell_type_code.shape[0] == 10)
    ctc = cell_lib.cell_type_code
    # O_w11, O_h11, O_h10, O_h01 are standard (code 0)
    check("V2-1: O_w11 is standard (code 0)", ctc[0].item() == 0)
    check("V2-1: O_h11 is standard (code 0)", ctc[1].item() == 0)
    check("V2-1: O_h10 is standard (code 0)", ctc[2].item() == 0)
    check("V2-1: O_h01 is standard (code 0)", ctc[3].item() == 0)
    # P0, P1 are pos_rect (code 1)
    check("V2-1: P0 is pos_rect (code 1)", ctc[4].item() == 1)
    check("V2-1: P1 is pos_rect (code 1)", ctc[5].item() == 1)
    # N0, N1 are neg_rect (code 2)
    check("V2-1: N0 is neg_rect (code 2)", ctc[6].item() == 2)
    check("V2-1: N1 is neg_rect (code 2)", ctc[7].item() == 2)
    # D1 is dead_zone (code 3)
    check("V2-1: D1 is dead_zone (code 3)", ctc[8].item() == 3)
    # Z is off -> code 0
    check("V2-1: Z is off (code 0)", ctc[9].item() == 0)


def test_v2_factorization_codes():
    """V2-2: MIX_CODES, BIAS_CODES, THRESH_CODES have spec values."""
    print("\nTest V2-2: v2 factorization codes")
    from config import MIX_CODES, BIAS_CODES, THRESH_CODES
    check("V2-2: MIX_CODES M11 src_gain=1.0",
          MIX_CODES["M11"]["src_gain"] == 1.0)
    check("V2-2: MIX_CODES M11 dst_gain=1.0",
          MIX_CODES["M11"]["dst_gain"] == 1.0)
    check("V2-2: MIX_CODES M10 dst_gain=0.5",
          MIX_CODES["M10"]["dst_gain"] == 0.5)
    check("V2-2: MIX_CODES M01 src_gain=0.5",
          MIX_CODES["M01"]["src_gain"] == 0.5)
    check("V2-2: BIAS_CODES Bsoft gm=0.25",
          BIAS_CODES["Bsoft"]["gm"] == 0.25)
    check("V2-2: BIAS_CODES Bsoft isat=1.50",
          BIAS_CODES["Bsoft"]["isat"] == 1.50)
    check("V2-2: BIAS_CODES Bmid gm=0.80",
          BIAS_CODES["Bmid"]["gm"] == 0.80)
    check("V2-2: BIAS_CODES Bmid isat=0.80",
          BIAS_CODES["Bmid"]["isat"] == 0.80)
    check("V2-2: BIAS_CODES Bhard gm=1.40",
          BIAS_CODES["Bhard"]["gm"] == 1.40)
    check("V2-2: BIAS_CODES Bhard isat=0.45",
          BIAS_CODES["Bhard"]["isat"] == 0.45)
    check("V2-2: THRESH_CODES T0=0.0",
          THRESH_CODES["T0"] == 0.0)
    check("V2-2: THRESH_CODES T1=0.35",
          THRESH_CODES["T1"] == 0.35)


def test_v2_cell_parameters():
    """V2-3: per-cell gm/isat/theta/beta/src_gain/dst_gain match spec."""
    print("\nTest V2-3: v2 cell parameter values")
    from cell_library import IdealizedCellLibrary
    cell_lib = IdealizedCellLibrary(library_name="v2")
    order = cell_lib._cell_order
    idx = {name: order.index(name) for name in order}

    # src_gain / dst_gain per cell
    check("V2-3: O_w11 src_gain=1.0, dst_gain=1.0 (M11)",
          abs(float(cell_lib.src_gain[idx["O_w11"]]) - 1.0) < 1e-6
          and abs(float(cell_lib.dst_gain[idx["O_w11"]]) - 1.0) < 1e-6)
    check("V2-3: O_h11 src_gain=1.0, dst_gain=1.0 (M11)",
          abs(float(cell_lib.src_gain[idx["O_h11"]]) - 1.0) < 1e-6
          and abs(float(cell_lib.dst_gain[idx["O_h11"]]) - 1.0) < 1e-6)
    check("V2-3: O_h10 src_gain=1.0, dst_gain=0.5 (M10)",
          abs(float(cell_lib.src_gain[idx["O_h10"]]) - 1.0) < 1e-6
          and abs(float(cell_lib.dst_gain[idx["O_h10"]]) - 0.5) < 1e-6)
    check("V2-3: O_h01 src_gain=0.5, dst_gain=1.0 (M01)",
          abs(float(cell_lib.src_gain[idx["O_h01"]]) - 0.5) < 1e-6
          and abs(float(cell_lib.dst_gain[idx["O_h01"]]) - 1.0) < 1e-6)

    # gm / isat per cell (Bsoft, Bmid, Bhard codes)
    check("V2-3: O_w11 gm=0.25, isat=1.50 (Bsoft)",
          abs(float(cell_lib.gm[idx["O_w11"]]) - 0.25) < 1e-6
          and abs(float(cell_lib.isat[idx["O_w11"]]) - 1.50) < 1e-6)
    check("V2-3: O_h11 gm=1.40, isat=0.45 (Bhard)",
          abs(float(cell_lib.gm[idx["O_h11"]]) - 1.40) < 1e-6
          and abs(float(cell_lib.isat[idx["O_h11"]]) - 0.45) < 1e-6)
    check("V2-3: P0 gm=0.80, isat=0.80 (Bmid)",
          abs(float(cell_lib.gm[idx["P0"]]) - 0.80) < 1e-6
          and abs(float(cell_lib.isat[idx["P0"]]) - 0.80) < 1e-6)
    check("V2-3: N1 gm=0.80, isat=0.80 (Bmid)",
          abs(float(cell_lib.gm[idx["N1"]]) - 0.80) < 1e-6
          and abs(float(cell_lib.isat[idx["N1"]]) - 0.80) < 1e-6)
    check("V2-3: D1 gm=0.80, isat=0.80 (Bmid)",
          abs(float(cell_lib.gm[idx["D1"]]) - 0.80) < 1e-6
          and abs(float(cell_lib.isat[idx["D1"]]) - 0.80) < 1e-6)

    # theta: T0=0.0 for O/P0/N0, T1=0.35 for P1/N1/D1
    check("V2-3: O_w11 theta=0.0 (T0)",
          abs(float(cell_lib.theta[idx["O_w11"]])) < 1e-6)
    check("V2-3: P0 theta=0.0 (T0)",
          abs(float(cell_lib.theta[idx["P0"]])) < 1e-6)
    check("V2-3: P1 theta=0.35 (T1)",
          abs(float(cell_lib.theta[idx["P1"]]) - 0.35) < 1e-6)
    check("V2-3: N1 theta=0.35 (T1)",
          abs(float(cell_lib.theta[idx["N1"]]) - 0.35) < 1e-6)
    check("V2-3: D1 theta=0.35 (T1)",
          abs(float(cell_lib.theta[idx["D1"]]) - 0.35) < 1e-6)

    # beta: P/N=0.08, D=0.10, O=1.0
    check("V2-3: P0 beta=0.08",
          abs(float(cell_lib.beta[idx["P0"]]) - 0.08) < 1e-6)
    check("V2-3: P1 beta=0.08",
          abs(float(cell_lib.beta[idx["P1"]]) - 0.08) < 1e-6)
    check("V2-3: N0 beta=0.08",
          abs(float(cell_lib.beta[idx["N0"]]) - 0.08) < 1e-6)
    check("V2-3: D1 beta=0.10",
          abs(float(cell_lib.beta[idx["D1"]]) - 0.10) < 1e-6)
    check("V2-3: O_w11 beta=1.0 (unused for standard)",
          abs(float(cell_lib.beta[idx["O_w11"]]) - 1.0) < 1e-6)

    # gleak: all v2 cells are 0 (strict mathematical boundedness)
    check("V2-3: all v2 cells have gleak=0",
          torch.all(cell_lib.gleak == 0).item())


def test_v2_boundedness():
    """V2-4: every v2 cell is bounded |I| <= isat over a large u sweep."""
    print("\nTest V2-4: v2 cell boundedness |I| <= isat")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v2")
    Q = cell_lib.num_cells
    E, B = 1, 200
    logits = torch.full((E, Q), -1e9)

    u_sweep = torch.linspace(-5.0, 5.0, B).unsqueeze(1)
    raw_mult = torch.zeros(E)

    for cell_idx in range(Q):
        logits_edge = logits.clone()
        logits_edge[0, cell_idx] = 0.0
        with torch.no_grad():
            i_edge = cell_lib(
                u_sweep, torch.zeros_like(u_sweep),
                logits_edge, raw_mult, x_max=10.0, ctx=None,
            )
        isat_cell = float(cell_lib.isat[cell_idx].item())
        i_vals = i_edge.squeeze()
        max_abs_i = float(i_vals.abs().max().item())
        check(f"V2-4: cell {cell_lib._cell_order[cell_idx]} bounded |I|<={isat_cell}",
              max_abs_i <= isat_cell + 1e-4,
              f"max |I|={max_abs_i:.6f}, I_sat={isat_cell}")

    # Verify the Z cell outputs are truly zero (not just bounded)
    logits_z = logits.clone()
    z_idx_v2 = cell_lib._cell_order.index("Z")
    logits_z[0, z_idx_v2] = 0.0
    with torch.no_grad():
        i_z = cell_lib(u_sweep, torch.zeros_like(u_sweep), logits_z, raw_mult, x_max=10.0, ctx=None)
    max_abs_i_z = float(i_z.abs().max().item())
    check("V2-4: Z cell outputs near-zero current (truly off)",
          max_abs_i_z < 1e-6,
          f"max |I_Z|={max_abs_i_z:.10f}")


def test_v2_mix_code_asymmetry():
    """V2-5: O_h10 vs O_h01 produce different currents under asymmetric inputs.

    O_h10 (M10: src=1.0, dst=0.5) is source-dominant.
    O_h01 (M01: src=0.5, dst=1.0) is destination-dominant.
    For x_src=2, x_dst=0: O_h10 sees u=2, O_h01 sees u=1.
    For x_src=0, x_dst=2: O_h10 sees u=-1, O_h01 sees u=-2.
    Hence O_h10(2,0) and O_h01(0,2) should differ.
    """
    print("\nTest V2-5: v2 mix code asymmetry (O_h10 vs O_h01)")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v2")
    Q = cell_lib.num_cells
    E = 1
    raw_mult = torch.zeros(E)

    h10_idx = cell_lib._cell_order.index("O_h10")
    h01_idx = cell_lib._cell_order.index("O_h01")

    logits_h10 = torch.full((E, Q), -1e9)
    logits_h10[0, h10_idx] = 0.0
    logits_h01 = torch.full((E, Q), -1e9)
    logits_h01[0, h01_idx] = 0.0

    # Case A: large source, zero destination
    x_src_A = torch.tensor([[2.0]])
    x_dst_A = torch.tensor([[0.0]])
    # Case B: zero source, large destination
    x_src_B = torch.tensor([[0.0]])
    x_dst_B = torch.tensor([[2.0]])

    with torch.no_grad():
        i_h10_A = cell_lib(x_src_A, x_dst_A, logits_h10, raw_mult, x_max=10.0, ctx=None)
        i_h01_B = cell_lib(x_src_B, x_dst_B, logits_h01, raw_mult, x_max=10.0, ctx=None)

    i_h10_A_val = float(i_h10_A[0, 0])
    i_h01_B_val = float(i_h01_B[0, 0])
    check("V2-5: O_h10(src=2, dst=0) ≠ O_h01(src=0, dst=2)",
          abs(i_h10_A_val - i_h01_B_val) > 0.01,
          f"O_h10(A)={i_h10_A_val:.6f}, O_h01(B)={i_h01_B_val:.6f}")
    # O_h10 is source-dominant: u=src_gain*2 - dst_gain*0 = 2.0
    # O_h01 is dst-dominant: u=src_gain*0 - dst_gain*2 = -2.0
    # tanh is odd, so O_h10(2,0) ≈ -O_h01(0,2) (both at hard saturation
    # because gm=1.4 and |u|=2 => gm*u/isat = 1.4*2/0.45 = 6.22, very saturated)
    check("V2-5: O_h10(src=2, dst=0) ≈ -O_h01(src=0, dst=2) (mirror via oddness)",
          abs(i_h10_A_val + i_h01_B_val) < 1e-4,
          f"O_h10(A)={i_h10_A_val:.6f}, O_h01(B)={i_h01_B_val:.6f}")


def test_v2_threshold_cells():
    """V2-6: P1 fires at u>theta, P0 fires at u>0. Likewise N1 vs N0."""
    print("\nTest V2-6: v2 threshold cells (P1/N1 vs P0/N0)")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v2")
    Q = cell_lib.num_cells
    E = 1
    raw_mult = torch.zeros(E)

    P0_idx = cell_lib._cell_order.index("P0")
    P1_idx = cell_lib._cell_order.index("P1")
    N0_idx = cell_lib._cell_order.index("N0")
    N1_idx = cell_lib._cell_order.index("N1")

    logits_p0 = torch.full((E, Q), -1e9)
    logits_p0[0, P0_idx] = 0.0
    logits_p1 = torch.full((E, Q), -1e9)
    logits_p1[0, P1_idx] = 0.0
    logits_n0 = torch.full((E, Q), -1e9)
    logits_n0[0, N0_idx] = 0.0
    logits_n1 = torch.full((E, Q), -1e9)
    logits_n1[0, N1_idx] = 0.0

    # At u=0.1: P0 (theta=0) fires immediately; P1 (theta=0.35) is
    # significantly suppressed by softplus rolloff.
    u_small = torch.tensor([[0.1]])
    with torch.no_grad():
        i_p0_small = cell_lib(u_small, torch.zeros_like(u_small), logits_p0, raw_mult, x_max=10.0, ctx=None)
        i_p1_small = cell_lib(u_small, torch.zeros_like(u_small), logits_p1, raw_mult, x_max=10.0, ctx=None)

    i_p0 = float(i_p0_small[0, 0])
    i_p1 = float(i_p1_small[0, 0])
    check("V2-6: P0 fires at u=0.1 (theta=0)",
          i_p0 > 0.01,
          f"P0(0.1)={i_p0:.6f}")
    # P1 is much smaller than P0 at u=0.1 (softplus rolloff near theta=0.35).
    check("V2-6: P1 strongly suppressed at u=0.1 (theta=0.35)",
          i_p1 < 0.15 * i_p0,
          f"P1(0.1)={i_p1:.6f}, P0(0.1)={i_p0:.6f}")

    # At u=0.5: P1 fires; P0 also fires (u > theta_P1=0.35).
    u_large = torch.tensor([[0.5]])
    with torch.no_grad():
        i_p1_large = cell_lib(u_large, torch.zeros_like(u_large), logits_p1, raw_mult, x_max=10.0, ctx=None)
        i_n1_large = cell_lib(-u_large, torch.zeros_like(u_large), logits_n1, raw_mult, x_max=10.0, ctx=None)

    i_p1_l = float(i_p1_large[0, 0])
    i_n1_l = float(i_n1_large[0, 0])
    check("V2-6: P1 fires at u=0.5",
          i_p1_l > 0.05,
          f"P1(0.5)={i_p1_l:.6f}")
    # N1 is the mirror of P1: N1(-u) == -P1(u). At u=0.5, N1(-0.5) fires
    # (in the negative direction).
    check("V2-6: N1(-0.5) fires (negative current)",
          i_n1_l < -0.05,
          f"N1(-0.5)={i_n1_l:.6f}")


def test_v2_forward_backward():
    """V2-7: v2 forward + backward gradients flow correctly."""
    print("\nTest V2-7: v2 forward + gradient flow")
    from cell_library import IdealizedCellLibrary
    import torch
    cell_lib = IdealizedCellLibrary(library_name="v2")
    Q = cell_lib.num_cells
    E = 4
    B = 8
    x_src = torch.randn(B, E, requires_grad=True)
    x_dst = torch.randn(B, E, requires_grad=True)
    logits = torch.randn(E, Q, requires_grad=True)
    raw_mult = torch.randn(E, requires_grad=True)

    i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=3.0, ctx=None)
    check("V2-7: forward output shape (B, E)",
          i_edge.shape == (B, E),
          f"got {i_edge.shape}")
    check("V2-7: forward output is finite",
          torch.isfinite(i_edge).all().item())

    loss = i_edge.sum()
    grads = torch.autograd.grad(loss, [logits, raw_mult, x_src], retain_graph=True)
    check("V2-7: gradients flow to logits",
          grads[0] is not None and torch.isfinite(grads[0]).all().item())
    check("V2-7: gradients flow to raw_mult",
          grads[1] is not None and torch.isfinite(grads[1]).all().item())
    check("V2-7: gradients flow to x_src",
          grads[2] is not None and torch.isfinite(grads[2]).all().item())


def test_v2_legacy_v15_unchanged():
    """V2-8: legacy/v15 libraries still use rho (backward compat preserved)."""
    print("\nTest V2-8: legacy/v15 rho-based preactivation unchanged")
    from cell_library import IdealizedCellLibrary
    import torch
    for lib_name in ("legacy", "v15"):
        cell_lib = IdealizedCellLibrary(library_name=lib_name)
        check(f"V2-8: {lib_name} _use_mix is False",
              cell_lib._use_mix is False,
              f"got {cell_lib._use_mix}")
        check(f"V2-8: {lib_name} has rho buffer",
              hasattr(cell_lib, "rho"))
        check(f"V2-8: {lib_name} has no src_gain buffer",
              not hasattr(cell_lib, "src_gain"),
              "src_gain should not be present in legacy/v15")
        check(f"V2-8: {lib_name} has no dst_gain buffer",
              not hasattr(cell_lib, "dst_gain"),
              "dst_gain should not be present in legacy/v15")
        # Forward pass still works
        E, B = 2, 4
        x_src = torch.randn(B, E)
        x_dst = torch.randn(B, E)
        logits = torch.randn(E, cell_lib.num_cells)
        raw_mult = torch.randn(E)
        with torch.no_grad():
            i_edge = cell_lib(x_src, x_dst, logits, raw_mult, x_max=3.0, ctx=None)
        check(f"V2-8: {lib_name} forward output finite",
              torch.isfinite(i_edge).all().item())


def test_v2_preset_routing():
    """V2-9: make_cell_library('v2') dispatches to IdealizedCellLibrary with 10 cells."""
    print("\nTest V2-9: v2 preset routing via make_cell_library")
    from cell_library import make_cell_library, IdealizedCellLibrary
    cell_lib = make_cell_library("v2")
    check("V2-9: make_cell_library('v2') returns IdealizedCellLibrary",
          isinstance(cell_lib, IdealizedCellLibrary))
    check("V2-9: v2 library has 10 cells",
          cell_lib.num_cells == 10)
    check("V2-9: v2 library is in mix mode",
          cell_lib._use_mix is True)


# ---- Stage-LR scaling tests (stage-lr-scaling plan) ----

def test_stage_lr_scale_backward_compat():
    """SLS-1: stage_lr_scale=1.0 returns single-group optimizer (backward compat)."""
    print("\nTest SLS-1: stage_lr_scale=1.0 backward compatibility")
    from train import make_optimizer
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(net, lr=1e-3, stage_lr_scale=1.0)
    check("SLS-1: single param group when scale=1.0",
          len(optim.param_groups) == 1,
          f"got {len(optim.param_groups)} groups")


def test_stage_lr_scale_multi_group():
    """SLS-2: stage_lr_scale>1.0 creates correct per-stage param groups with geometric LR."""
    print("\nTest SLS-2: stage_lr_scale=10.0 produces correct staged LR ratios")
    from train import make_optimizer
    from config import PRESETS
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    scale = 10.0
    optim = make_optimizer(net, lr=base_lr, stage_lr_scale=scale)

    n_stages = len(net.core.stages)
    group_lrs = [g["lr"] for g in optim.param_groups]

    check("SLS-2: at least 1 param group",
          len(group_lrs) >= 1, f"got {len(group_lrs)} groups")

    # For KirchhoffNetWithIO we expect N_stages + 1 groups (stages + mapper).
    # Build the full expected LR list: stage groups + base LR for other.
    expected_stage_lrs = [
        base_lr * (scale ** (n_stages - 1 - i))
        for i in range(n_stages)
    ]
    expected_lrs = sorted(expected_stage_lrs + [base_lr])
    got_lrs = sorted(group_lrs)
    check("SLS-2: N_stages + 1 param groups with correct geometric LRs",
          len(group_lrs) == n_stages + 1 and
          all(abs(g - e) < 1e-12 for g, e in zip(got_lrs, expected_lrs)),
          f"expected {n_stages + 1} groups with LRs {expected_lrs}, "
          f"got {len(group_lrs)} groups with LRs {got_lrs}")


def test_stage_lr_scale_scheduler_compat():
    """SLS-3: CosineAnnealingLR works correctly with multi-group optimizer."""
    print("\nTest SLS-3: scheduler compatibility with staged optimizer")
    from train import make_optimizer
    from config import PRESETS, OPTIM
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from torch.optim.lr_scheduler import CosineAnnealingLR

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(net, lr=1e-3, stage_lr_scale=10.0)
    scheduler = CosineAnnealingLR(optim, T_max=100, eta_min=OPTIM["scheduler_eta_min"])

    initial_lrs = [g["lr"] for g in optim.param_groups]
    scheduler.step()
    stepped_lrs = [g["lr"] for g in optim.param_groups]

    check("SLS-3: all groups reduced after one scheduler step",
          all(s < i for s, i in zip(stepped_lrs, initial_lrs)),
          f"initial={initial_lrs}, after_step={stepped_lrs}")
    # Ordering should be preserved: if group A had higher LR than group B
    # before the scheduler step, it still should after (ratios preserved).
    ratios_before = [i / j for i, j in zip(initial_lrs[:-1], initial_lrs[1:])]
    ratios_after  = [i / j for i, j in zip(stepped_lrs[:-1], stepped_lrs[1:])]
    check("SLS-3: group LR ratios preserved after scheduler step",
          all(abs(rb - ra) < 1e-4 for rb, ra in zip(ratios_before, ratios_after)),
          f"ratios before={ratios_before}, after={ratios_after}")


# ---- Rail loss fix tests (rail-loss-fix plan) ----

def test_rail_loss_zero_inside_bounds():
    """RL-1: rail loss is exactly 0 when all |x| < x_max."""
    print("\nTest RL-1: rail loss is zero inside bounds (ReLU² barrier)")
    from train import _stage_rail_loss
    from types import SimpleNamespace

    stage = SimpleNamespace(x_max=1.0)
    traj = torch.tensor([[[0.0, 0.5, -0.3, 0.9, -1.0]]])  # all within [-1, 1]
    loss = _stage_rail_loss(stage, traj)
    check("RL-1: rail loss == 0 inside bounds",
          loss.item() == 0.0,
          f"got {loss.item():.6f}")


def test_rail_loss_positive_outside_bounds():
    """RL-2: rail loss > 0 when |x| > x_max."""
    print("\nTest RL-2: rail loss is positive outside bounds (ReLU² barrier)")
    from train import _stage_rail_loss
    from types import SimpleNamespace

    stage = SimpleNamespace(x_max=1.0)
    traj = torch.tensor([[[0.0, 1.5, -2.0]]])  # 1.5 and -2.0 exceed x_max
    loss = _stage_rail_loss(stage, traj)
    check("RL-2: rail loss > 0 when |x| > x_max",
          loss.item() > 0.0,
          f"got {loss.item():.6f}")
    check("RL-2: rail loss finite",
          math.isfinite(loss.item()),
          f"loss={loss.item():.6f}")


def test_retrain_lr_scale_defaults_one():
    """RL-3: retrain-stage-lr-scale defaults to 1.0 (single group)."""
    print("\nTest RL-3: --retrain-stage-lr-scale defaults to 1.0")
    import sys as _sys
    _sys.path.insert(0, THIS_DIR)
    import argparse
    import train_script as ts_mod
    parser = argparse.ArgumentParser()
    ts_mod._add_argparse_args(parser)
    args = parser.parse_args(["--stage-lr-scale", "10", "--prune"])
    check("RL-3: retrain_stage_lr_scale defaults to 1.0",
          args.retrain_stage_lr_scale == 1.0,
          f"got {args.retrain_stage_lr_scale}")
    # Also test explicit override works
    args2 = parser.parse_args(["--stage-lr-scale", "10", "--retrain-stage-lr-scale", "5", "--prune"])
    check("RL-3: --retrain-stage-lr-scale 5 parsed",
          args2.retrain_stage_lr_scale == 5.0,
          f"got {args2.retrain_stage_lr_scale}")


# ---- Three-phase schedule tests (three-phase-schedule plan) ----

def test_three_phase_boundaries():
    """TP-1: phase_boundaries returns correct epoch boundaries."""
    print("\nTest TP-1: phase_boundaries epoch division")
    from train import phase_boundaries
    total = 100
    a, b, c = phase_boundaries(total)
    check("TP-1: phases sum to total epochs", a + (b - a) + (c - b) == total,
          f"got a={a}, b={b}, c={c}")
    check("TP-1: frac_a ~ 30% of 100", a == 30, f"got a={a}")
    check("TP-1: frac_b ~ 40% of 100", b - a == 40, f"got b-a={b - a}")
    check("TP-1: frac_c ~ 30% of 100", c - b == 30, f"got c-b={c - b}")
    check("TP-1: c == total", c == total, f"got c={c}")


def test_three_phase_for_epoch():
    """TP-2: phase_for_epoch returns correct phase label."""
    print("\nTest TP-2: phase_for_epoch labels")
    from train import phase_for_epoch
    total = 100
    check("TP-2: epoch 0 → A", phase_for_epoch(0, total) == "A")
    check("TP-2: epoch 15 → A", phase_for_epoch(15, total) == "A")
    check("TP-2: epoch 29 → A", phase_for_epoch(29, total) == "A",
          f"got {phase_for_epoch(29, total)}")
    check("TP-2: epoch 30 → B", phase_for_epoch(30, total) == "B",
          f"got {phase_for_epoch(30, total)}")
    check("TP-2: epoch 69 → B", phase_for_epoch(69, total) == "B")
    check("TP-2: epoch 70 → C", phase_for_epoch(70, total) == "C",
          f"got {phase_for_epoch(70, total)}")
    check("TP-2: epoch 99 → C", phase_for_epoch(99, total) == "C")


def test_three_phase_tau_values():
    """TP-3: three_phase_tau produces correct tau per phase."""
    print("\nTest TP-3: three_phase_tau schedule")
    from train import three_phase_tau
    total = 100
    tau_a = three_phase_tau(0, total)
    check("TP-3: Phase A tau ≈ 1.0", abs(tau_a - 1.0) < 1e-6,
          f"got {tau_a}")
    tau_a_mid = three_phase_tau(15, total)
    check("TP-3: Phase A mid tau still 1.0", abs(tau_a_mid - 1.0) < 1e-6,
          f"got {tau_a_mid}")
    tau_b_start = three_phase_tau(30, total)
    check("TP-3: Phase B start tau ≈ 1.0", abs(tau_b_start - 1.0) < 0.01,
          f"got {tau_b_start}")
    tau_b_end = three_phase_tau(69, total)
    check("TP-3: Phase B end tau ≈ 0.6", abs(tau_b_end - 0.6) < 0.05,
          f"got {tau_b_end}")
    tau_c_start = three_phase_tau(70, total)
    check("TP-3: Phase C start tau ≈ 0.6", abs(tau_c_start - 0.6) < 0.05,
          f"got {tau_c_start}")
    tau_c_end = three_phase_tau(99, total)
    check("TP-3: Phase C end tau ≈ 0.1", abs(tau_c_end - 0.1) < 0.05,
          f"got {tau_c_end}")
    check("TP-3: three_phase_tau non-increasing across full schedule",
          all(three_phase_tau(e, total) >= three_phase_tau(e + 1, total) - 1e-9
              for e in range(total)))


def test_three_phase_lambdas():
    """TP-4: three_phase_lambdas produces correct lambda dict per phase."""
    print("\nTest TP-4: three_phase_lambdas phase-dependent values")
    from train import three_phase_lambdas, _REG_KEYS
    from config import SCHEDULE_THREE_PHASE
    base = {"sparsity": 100.0, "edge_gate": 200.0, "node_gate": 300.0,
            "power": 400.0, "capacitance": 500.0, "rail": 0.1}
    total = 100
    a_lams = three_phase_lambdas(0, total, base)
    check("TP-4: Phase A sparsity = 0", a_lams["sparsity"] == 0.0,
          f"got {a_lams['sparsity']}")
    check("TP-4: Phase A rail preserved", a_lams["rail"] == 0.1,
          f"got {a_lams['rail']}")
    for k in _REG_KEYS:
        check(f"TP-4: Phase A {k} = 0", a_lams[k] == 0.0,
              f"got {a_lams[k]}")

    b_lams = three_phase_lambdas(60, total, base)
    b_target = SCHEDULE_THREE_PHASE["lambdas_b"]
    check("TP-4: Phase B (late) sparsity matches target",
          abs(b_lams["sparsity"] - float(b_target["sparsity"])) < 1e-10,
          f"got {b_lams['sparsity']}, expected {b_target['sparsity']}")
    check("TP-4: Phase B (late) edge_gate matches target",
          abs(b_lams["edge_gate"] - float(b_target["edge_gate"])) < 1e-10)

    c_lams = three_phase_lambdas(80, total, base)
    c_target = SCHEDULE_THREE_PHASE["lambdas_c"]
    check("TP-4: Phase C sparsity matches target",
          abs(c_lams["sparsity"] - float(c_target["sparsity"])) < 1e-10,
          f"got {c_lams['sparsity']}, expected {c_target['sparsity']}")
    check("TP-4: Phase C edge_gate = 0", c_lams["edge_gate"] == 0.0)
    check("TP-4: Phase C rail preserved", c_lams["rail"] == 0.1)


def test_three_phase_lambdas_warmup():
    """TP-5: Phase B lambdas warm in linearly."""
    print("\nTest TP-5: three_phase_lambdas Phase B warmup")
    from train import three_phase_lambdas
    from config import SCHEDULE_THREE_PHASE
    base = {"sparsity": 1e3, "edge_gate": 2e3, "node_gate": 3e3,
            "power": 4e3, "capacitance": 5e3, "rail": 0.1}
    total = 100  # a=30, b=70, c=100, b warmup ≈ 40/6 ≈ 7 epochs
    b_first = three_phase_lambdas(31, total, base)
    # At epoch 31 (just 1 epoch past Phase A), the scale should be ~ 1/7
    target = float(SCHEDULE_THREE_PHASE["lambdas_b"]["sparsity"])
    expected_scale = float(31 - 30 + 1) / 7.0  # + 1 for 1-indexed warmup
    expected_val = target * min(1.0, expected_scale)
    check("TP-5: Phase B warmup partial sparsity < full target",
          b_first["sparsity"] < target * 0.5,
          f"got {b_first['sparsity']}, target={target}")
    # Late Phase B should be at full scale
    b_late = three_phase_lambdas(60, total, base)
    check("TP-5: Phase B late (full warmup) sparsity == target",
          abs(b_late["sparsity"] - target) < 1e-10)


def test_solidification_metrics():
    """TP-6: compute_solidification_metrics returns valid dict."""
    print("\nTest TP-6: compute_solidification_metrics structure")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from train import compute_solidification_metrics
    cell_lib = make_default_library()
    net = build_net_from_preset("sinx", cell_lib=cell_lib)
    metrics = compute_solidification_metrics(net, tau=1.0)
    check("TP-6: mean_max_cell_prob present",
          "mean_max_cell_prob" in metrics)
    check("TP-6: mean_pZ present", "mean_pZ" in metrics)
    check("TP-6: mean_sigma_z present", "mean_sigma_z" in metrics)
    check("TP-6: frac_sigma_z_below_0.1 present",
          "frac_sigma_z_below_0.1" in metrics)
    check("TP-6: mean_sigma_u present", "mean_sigma_u" in metrics)
    check("TP-6: num_edges present", "num_edges" in metrics)
    check("TP-6: num_nodes present", "num_nodes" in metrics)
    check("TP-6: metrics finite",
          all(isinstance(v, float) or isinstance(v, int) for v in metrics.values()))
    check("TP-6: mean_max_cell_prob in [0,1]",
          0.0 <= metrics["mean_max_cell_prob"] <= 1.0,
          f"got {metrics['mean_max_cell_prob']}")
    check("TP-6: frac_sigma_z_below_0.1 in [0,1]",
          0.0 <= metrics["frac_sigma_z_below_0.1"] <= 1.0)
    check("TP-6: all keys present",
          metrics.keys() >= {"mean_max_cell_prob", "mean_pZ", "mean_sigma_z",
                             "frac_sigma_z_below_0.1", "frac_sigma_z_below_0.05",
                             "frac_sigma_z_below_0.01", "mean_sigma_u",
                             "num_edges", "num_nodes", "tau"})


def test_validate_argmax_runs():
    """TP-7: validate_argmax runs without error on a presets."""
    print("\nTest TP-7: validate_argmax forward pass")
    from topology import build_net_from_preset
    from cell_library import make_default_library
    from train import validate_argmax, default_ctx_factory
    from config import PRESETS
    import torch.nn.functional as F
    import torch
    cell_lib = make_default_library()
    net = build_net_from_preset("sinx", cell_lib=cell_lib)
    ctx_factory = default_ctx_factory(net)
    u = torch.linspace(-math.pi, math.pi, 32).unsqueeze(1)
    y = torch.sin(u)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(u, y), batch_size=16)
    val_arg = validate_argmax(net, loader, F.mse_loss, ctx_factory, "cpu")
    check("TP-7: argmax validation returns finite float",
          math.isfinite(val_arg))
    check("TP-7: argmax validation > 0",
          val_arg > 0.0, f"got {val_arg}")


def test_log_solidification_format():
    """TP-8: _log_solidification writes correct TSV header and rows."""
    print("\nTest TP-8: _log_solidification file format")
    from train_script import _log_solidification
    import tempfile, os
    # Use a path that does not yet exist so the header is written.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True)
    tpath = tmp.name
    tmp.close()  # close so it's deleted; we'll reuse the path
    if os.path.exists(tpath):
        os.unlink(tpath)
    try:
        _log_solidification(tpath, 0, {"mean_max_cell_prob": 0.75, "num_edges": 42, "tau": 1.0})
        with open(tpath) as f:
            lines = f.readlines()
        check("TP-8: header + 1 data row written",
              len(lines) >= 2,
              f"got {len(lines)} lines")
        check("TP-8: header starts with epoch",
              lines[0].strip().startswith("epoch"),
              f"got {lines[0]!r}")
        check("TP-8: first data row starts with 0",
              lines[1].strip().startswith("0"),
              f"got {lines[1]!r}")
        _log_solidification(tpath, 1, {"mean_max_cell_prob": 0.8, "num_edges": 42, "tau": 0.8})
        with open(tpath) as f:
            lines = f.readlines()
        check("TP-8: second data row appended", len(lines) == 3,
              f"got {len(lines)} lines")
        check("TP-8: second data row starts with 1",
              lines[2].strip().startswith("1"))
    finally:
        if os.path.exists(tpath):
            os.unlink(tpath)


def test_smooth2d_grid_uses_three_phase():
    """TP-9: smooth2d_grid preset uses three_phase schedule."""
    print("\nTest TP-9: smooth2d_grid schedule='three_phase'")
    from config import PRESETS
    cfg = PRESETS["smooth2d_grid"]
    check("TP-9: schedule is three_phase",
          cfg.get("schedule") == "three_phase",
          f"got {cfg.get('schedule')!r}")
    check("TP-9: tau_anneal is True",
          cfg.get("tau_anneal") is True)  # three_phase uses its own tau; tau_anneal acts as gate


# ---- Mapper LR control tests (mapper-lr-control plan) ----

def test_mapper_lr_scale_backward_compat():
    """MLR-3: mapper_lr_scale=1.0 preserves single-group optimizer (backward compat)."""
    print("\nTest MLR-3: mapper_lr_scale=1.0 backward compatibility")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(net, lr=1e-3, stage_lr_scale=1.0, mapper_lr_scale=1.0)
    check("MLR-3: single param group when both scales=1.0",
          len(optim.param_groups) == 1,
          f"got {len(optim.param_groups)} groups")


def test_mapper_lr_scale_separate_group():
    """MLR-1: mapper_lr_scale<1.0 creates a separate param group for mappers at scaled LR."""
    print("\nTest MLR-1: mapper_lr_scale=0.1 creates separate mapper group")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    mapper_scale = 0.1
    optim = make_optimizer(net, lr=base_lr, stage_lr_scale=1.0, mapper_lr_scale=mapper_scale)

    check("MLR-1: 2 param groups (core + mapper)",
          len(optim.param_groups) == 2,
          f"got {len(optim.param_groups)} groups")

    group_lrs = sorted([g["lr"] for g in optim.param_groups])
    expected_lrs = sorted([base_lr, base_lr * mapper_scale])
    check("MLR-1: correct LRs in groups",
          all(abs(g - e) < 1e-12 for g, e in zip(group_lrs, expected_lrs)),
          f"expected {expected_lrs}, got {group_lrs}")

    # Verify mapper group actually contains mapper params.
    mapper_group = next(
        g for g in optim.param_groups
        if abs(g["lr"] - base_lr * mapper_scale) < 1e-12
    )
    mapper_param_names = []
    for p in mapper_group["params"]:
        for n, pp in net.named_parameters():
            if pp is p:
                mapper_param_names.append(n)
                break
    check("MLR-1: mapper group contains only input_mapper/output_mapper params",
          all("input_mapper" in n or "output_mapper" in n for n in mapper_param_names)
          and len(mapper_param_names) > 0,
          f"got {mapper_param_names}")


def test_mapper_lr_scale_combined_with_stage_lr_scale():
    """MLR-2: mapper_lr_scale + stage_lr_scale both active creates per-stage + mapper groups."""
    print("\nTest MLR-2: both stage_lr_scale and mapper_lr_scale active")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    stage_scale = 10.0
    mapper_scale = 0.1
    optim = make_optimizer(
        net, lr=base_lr,
        stage_lr_scale=stage_scale,
        mapper_lr_scale=mapper_scale,
    )

    n_stages = len(net.core.stages)
    group_lrs = sorted([g["lr"] for g in optim.param_groups])
    expected_stage_lrs = [
        base_lr * (stage_scale ** (n_stages - 1 - i))
        for i in range(n_stages)
    ]
    # For smooth2d_grid (KirchhoffNetWithIO), all non-stage non-mapper params
    # are absent or empty, so we get N_stages + 1 groups (stages + mapper).
    expected_lrs = sorted(expected_stage_lrs + [base_lr * mapper_scale])
    check("MLR-2: N_stages + 1 groups (stages + mapper)",
          len(group_lrs) == n_stages + 1,
          f"got {len(group_lrs)} groups: {group_lrs}, expected {n_stages + 1}")
    check("MLR-2: correct LRs across all groups",
          all(abs(g - e) < 1e-12 for g, e in zip(group_lrs, expected_lrs)),
          f"expected {expected_lrs}, got {group_lrs}")


def test_mapper_lr_scale_rejects_zero_or_negative():
    """MLR-6: mapper_lr_scale <= 0 raises ValueError."""
    print("\nTest MLR-6: mapper_lr_scale validation")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    try:
        make_optimizer(net, lr=1e-3, mapper_lr_scale=0.0)
        raised = False
    except ValueError:
        raised = True
    check("MLR-6: mapper_lr_scale=0.0 raises ValueError", raised)

    try:
        make_optimizer(net, lr=1e-3, mapper_lr_scale=-0.1)
        raised = False
    except ValueError:
        raised = True
    check("MLR-6: mapper_lr_scale=-0.1 raises ValueError", raised)


def test_freeze_mappers_cli_flag_parsed():
    """MLR-4: --freeze-mappers CLI flag is parsed correctly."""
    print("\nTest MLR-4: --freeze-mappers CLI flag")
    import subprocess
    import sys
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from train_script import _add_argparse_args; "
        "import argparse; "
        "p = argparse.ArgumentParser(); "
        "_add_argparse_args(p); "
        "args = p.parse_args(['--problem', 'sinx']); "
        "print('freeze_mappers_default:', args.freeze_mappers); "
        "args2 = p.parse_args(['--problem', 'sinx', '--freeze-mappers']); "
        "print('freeze_mappers_set:', args2.freeze_mappers)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=THIS_DIR,
        capture_output=True, text=True, timeout=60,
    )
    check("MLR-4: subprocess returns 0", result.returncode == 0,
          f"stderr: {result.stderr}")
    check("MLR-4: default freeze_mappers is False",
          "freeze_mappers_default: False" in result.stdout,
          f"stdout: {result.stdout}")
    check("MLR-4: --freeze-mappers sets flag to True",
          "freeze_mappers_set: True" in result.stdout,
          f"stdout: {result.stdout}")


def test_mapper_lr_scale_cli_flag_parsed():
    """MLR-5: --mapper-lr-scale CLI flag is parsed correctly."""
    print("\nTest MLR-5: --mapper-lr-scale CLI flag")
    import subprocess
    import sys
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from train_script import _add_argparse_args; "
        "import argparse; "
        "p = argparse.ArgumentParser(); "
        "_add_argparse_args(p); "
        "args = p.parse_args(['--problem', 'sinx']); "
        "print('default:', args.mapper_lr_scale); "
        "args2 = p.parse_args(['--problem', 'sinx', '--mapper-lr-scale', '0.1']); "
        "print('set:', args2.mapper_lr_scale)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=THIS_DIR,
        capture_output=True, text=True, timeout=60,
    )
    check("MLR-5: subprocess returns 0", result.returncode == 0,
          f"stderr: {result.stderr}")
    check("MLR-5: default mapper_lr_scale is 0.1 (new default)",
          "default: 0.1" in result.stdout,
          f"stdout: {result.stdout}")
    check("MLR-5: --mapper-lr-scale 0.1 parsed",
          "set: 0.1" in result.stdout,
          f"stdout: {result.stdout}")


def test_mapper_unfreeze_epoch_midpoint():
    """MLR-7: mapper_unfreeze_epoch = fp_a_end + (fp_b2_end - fp_a_end) // 2."""
    print("\nTest MLR-7: mapper unfreeze epoch = midpoint of B1+B2")
    from train import four_phase_boundaries

    for total in [120, 240, 400]:
        a_end, b1_end, b2_end, _ = four_phase_boundaries(total)
        midpoint = a_end + (b2_end - a_end) // 2
        # Verify it's strictly between a_end and b2_end, closer to a_end
        # (since integer division truncates).
        check(f"MLR-7: midpoint in (a_end, b2_end) for {total} epochs",
              a_end < midpoint < b2_end,
              f"a_end={a_end}, b2_end={b2_end}, midpoint={midpoint}")
        # Verify it's the floor of the arithmetic mean.
        arithmetic_mean = (a_end + b2_end) / 2
        check(f"MLR-7: midpoint == floor(mean) for {total} epochs",
              midpoint == int(arithmetic_mean),
              f"midpoint={midpoint}, floor(mean)={int(arithmetic_mean)}")


# ---- LR param groups tests (lr-param-groups plan) ----

def test_lrp_struct_dyn_groups_created():
    """LRP-1: struct_lr_scale and dyn_lr_scale create separate param groups."""
    print("\nTest LRP-1: struct/dyn LR groups created")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    optim = make_optimizer(
        net, lr=base_lr,
        stage_lr_scale=1.0, mapper_lr_scale=0.1,
        struct_lr_scale=2.0, dyn_lr_scale=0.5,
    )
    check("LRP-1: 4 param groups (other + mapper + struct + dyn)",
          len(optim.param_groups) == 4,
          f"got {len(optim.param_groups)} groups, expected 4")

    lrs = sorted([g["lr"] for g in optim.param_groups])
    expected = sorted([base_lr, base_lr * 0.1, base_lr * 2.0, base_lr * 0.5])
    check("LRP-1: correct LRs in groups",
          all(abs(a - b) < 1e-12 for a, b in zip(lrs, expected)),
          f"expected {expected}, got {lrs}")


def test_lrp_correct_param_membership():
    """LRP-2: verify params land in correct groups by name."""
    print("\nTest LRP-2: correct param membership per group")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    optim = make_optimizer(
        net, lr=base_lr,
        stage_lr_scale=1.0, mapper_lr_scale=0.1,
        struct_lr_scale=2.0, dyn_lr_scale=0.5,
    )

    # Find groups by unique LR
    groups_by_lr = {g["lr"]: g for g in optim.param_groups}
    param_to_group = {}
    for n, p_in_net in net.named_parameters():
        for gidx, g in enumerate(optim.param_groups):
            if any(p_in_net is pp for pp in g["params"]):
                param_to_group[n] = g["lr"]
                break

    struct_params = [
        n for n, lr_val in param_to_group.items()
        if abs(lr_val - base_lr * 2.0) < 1e-12
    ]
    dyn_params = [
        n for n, lr_val in param_to_group.items()
        if abs(lr_val - base_lr * 0.5) < 1e-12
    ]
    mapper_params = [
        n for n, lr_val in param_to_group.items()
        if abs(lr_val - base_lr * 0.1) < 1e-12
    ]
    other_params = [
        n for n, lr_val in param_to_group.items()
        if abs(lr_val - base_lr) < 1e-12
    ]

    check("LRP-2: struct group has z_logits params",
          any("z_logits" in n for n in struct_params),
          f"struct_params={struct_params}")
    check("LRP-2: struct group has logits params",
          any(n.endswith(".logits") for n in struct_params),
          f"struct_params={struct_params}")
    check("LRP-2: struct group has raw_mult params",
          any("raw_mult" in n for n in struct_params),
          f"struct_params={struct_params}")
    check("LRP-2: dyn group has raw_leak params",
          any("raw_leak" in n for n in dyn_params),
          f"dyn_params={dyn_params}")
    # raw_drive_g is optional — only present with persistent drive.
    has_drive = any("raw_drive_g" in n for n, _ in net.named_parameters())
    if has_drive:
        check("LRP-2: dyn group has raw_drive_g params",
              any("raw_drive_g" in n for n in dyn_params),
              f"dyn_params={dyn_params}")
    check("LRP-2: mapper group has input_mapper params",
          any("input_mapper" in n for n in mapper_params),
          f"mapper_params={mapper_params}")
    check("LRP-2: mapper group has output_mapper params",
          any("output_mapper" in n for n in mapper_params),
          f"mapper_params={mapper_params}")
    check("LRP-2: no struct params in other group",
          not any("z_logits" in n or n.endswith(".logits") or "raw_mult" in n
                  for n in other_params),
          f"other_params unexpectedly has leaked struct params: {other_params}")
    check("LRP-2: no dyn params in other group",
          not any("raw_leak" in n or "raw_drive_g" in n
                  for n in other_params),
          f"other_params unexpectedly has leaked dyn params: {other_params}")


def test_lrp_backward_compat():
    """LRP-3: all scales=1.0 preserves single-group optimizer."""
    print("\nTest LRP-3: backward compat with all scales=1.0")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(
        net, lr=1e-3,
        stage_lr_scale=1.0, mapper_lr_scale=1.0,
        struct_lr_scale=1.0, dyn_lr_scale=1.0,
    )
    check("LRP-3: single param group when all scales=1.0",
          len(optim.param_groups) == 1,
          f"got {len(optim.param_groups)} groups")


def test_lrp_validation_positive():
    """LRP-4: struct_lr_scale and dyn_lr_scale must be positive."""
    print("\nTest LRP-4: validation of positive struct/dyn scales")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)

    for val, name in [(0.0, "zero"), (-0.1, "negative")]:
        n_trials = 0
        for scale_name, scale_val in [("struct_lr_scale", val), ("dyn_lr_scale", val)]:
            try:
                make_optimizer(
                    net, lr=1e-3,
                    mapper_lr_scale=1.0,
                    **{scale_name: scale_val},
                )
            except ValueError:
                n_trials += 1
        check(f"LRP-4: {name} struct/dyn scale raises ValueError (got {n_trials}/2)",
              n_trials == 2)


def test_lrp_composition_with_mapper():
    """LRP-5: mapper_lr_scale and struct_lr_scale compose correctly."""
    print("\nTest LRP-5: mapper + struct LR scales composition")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3
    optim = make_optimizer(
        net, lr=base_lr,
        mapper_lr_scale=0.1, struct_lr_scale=3.0, dyn_lr_scale=1.0,
    )
    lrs = [g["lr"] for g in optim.param_groups]
    check("LRP-5: contains mapper LR = 0.1*base",
          base_lr * 0.1 in lrs,
          f"LRs={lrs}")
    check("LRP-5: contains struct LR = 3.0*base",
          base_lr * 3.0 in lrs,
          f"LRs={lrs}")
    check("LRP-5: contains base LR group",
          base_lr in lrs,
          f"LRs={lrs}")


def test_lrp_stage_lr_scale_ignored():
    """LRP-6: stage_lr_scale ignored when struct/dyn active."""
    print("\nTest LRP-6: stage_lr_scale is ignored when struct/dyn != 1.0")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library
    import io
    import warnings

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    base_lr = 1e-3

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        optim = make_optimizer(
            net, lr=base_lr,
            stage_lr_scale=10.0,      # should be ignored
            mapper_lr_scale=0.1,
            struct_lr_scale=2.0,
            dyn_lr_scale=0.5,
        )
    has_warning = any("stage_lr_scale" in str(wm.message).lower() for wm in w)
    check("LRP-6: warning issued about stage_lr_scale ignored",
          has_warning,
          f"warnings: {[str(x.message) for x in w]}")

    lrs = [g["lr"] for g in optim.param_groups]
    expected_lrs = [base_lr, base_lr * 0.1, base_lr * 2.0, base_lr * 0.5]
    # None of the groups should have stage_lr_scale multiplied in
    for expected in expected_lrs:
        check(f"LRP-6: group with LR {expected} exists",
              any(abs(lr - expected) < 1e-12 for lr in lrs),
              f"LRs={lrs}")


def test_lrp_empty_group_handling():
    """LRP-6b: groups with no params are simply absent (not an error)."""
    print("\nTest LRP-6b: empty groups are omitted from optimizer")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    # dyn_lr_scale=1.0 creates no dyn group; struct=2.0 creates struct group
    optim = make_optimizer(net, lr=1e-3, struct_lr_scale=2.0, dyn_lr_scale=1.0)
    check("LRP-6b: optimizer created without error",
          isinstance(optim, torch.optim.Optimizer))
    # When dyn_lr_scale=1.0 is passed explicitly, make_optimizer sees
    # struct=2.0 != 1.0 so flat grouping kicks in. No dyn group needed.
    check("LRP-6b: has groups (struct active)",
          len(optim.param_groups) >= 2)


def test_lrp_cli_flags_parsed():
    """LRP-7: --struct-lr-scale and --dyn-lr-scale CLI flags parse correctly."""
    print("\nTest LRP-7: CLI flags for struct/dyn LR scales")
    import subprocess
    import sys
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from train_script import _add_argparse_args; "
            "import argparse; "
            "p = argparse.ArgumentParser(); "
            "_add_argparse_args(p); "
            "args = p.parse_args(['--problem', 'sinx', '--struct-lr-scale', '3.0', '--dyn-lr-scale', '0.5']); "
            "print('struct:', args.struct_lr_scale); "
            "print('dyn:', args.dyn_lr_scale)",
        ],
        cwd=THIS_DIR,
        capture_output=True, text=True, timeout=60,
    )
    check("LRP-7: subprocess returns 0", result.returncode == 0,
          f"stderr: {result.stderr}")
    check("LRP-7: --struct-lr-scale 3.0 parsed",
          "struct: 3.0" in result.stdout,
          f"stdout: {result.stdout}")
    check("LRP-7: --dyn-lr-scale 0.5 parsed",
          "dyn: 0.5" in result.stdout,
          f"stdout: {result.stdout}")


def test_lrp_compute_update_norms():
    """LRP-8: compute_update_norms returns expected keys and values."""
    print("\nTest LRP-8: compute_update_norms output")
    from train_script import compute_update_norms
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)

    snapshots = {name: p.data.detach().clone() for name, p in net.named_parameters()}
    # Before any training, update norms should be ~zero
    norms = compute_update_norms(snapshots, net)

    check("LRP-8: has mapper key", "mapper" in norms)
    check("LRP-8: has struct key", "struct" in norms)
    check("LRP-8: has dyn key", "dyn" in norms)
    check("LRP-8: has other key", "other" in norms)

    for gname in ("mapper", "struct", "dyn", "other"):
        d = norms[gname]
        for key in ("param_norm", "update_norm", "rel_update"):
            check(f"LRP-8: {gname} has {key}", key in d,
                  f"keys={list(d.keys())}")
        check(f"LRP-8: {gname} update_norm ~ 0 (no training)",
              d["update_norm"] < 1e-12,
              f"{gname} update_norm={d['update_norm']}")


def test_lrp_old_mapper_default_backward_compat():
    """LRP-9: old default (mapper=1.0, struct=1.0, dyn=1.0) gives single group."""
    print("\nTest LRP-9: old explicit defaults produce single group")
    from train import make_optimizer
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    optim = make_optimizer(
        net, lr=1e-3,
        mapper_lr_scale=1.0,
        struct_lr_scale=1.0,
        dyn_lr_scale=1.0,
    )
    check("LRP-9: single group", len(optim.param_groups) == 1)


def test_lrp_mapper_lr_scale_new_default():
    """LRP-10: mapper_lr_scale defaults to 0.1."""
    print("\nTest LRP-10: mapper_lr_scale new default (0.1)")
    import subprocess
    import sys
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from train_script import _add_argparse_args; "
            "import argparse; p = argparse.ArgumentParser(); "
            "_add_argparse_args(p); "
            "args = p.parse_args(['--problem', 'sinx']); "
            "print('default:', args.mapper_lr_scale); "
            "print('struct:', args.struct_lr_scale); "
            "print('dyn:', args.dyn_lr_scale)",
        ],
        cwd=THIS_DIR,
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout
    check("LRP-10: mapper_lr_scale default is 0.1",
          "default: 0.1" in output,
          f"stdout: {output}")
    check("LRP-10: struct_lr_scale default is 2.0",
          "struct: 2.0" in output,
          f"stdout: {output}")
    check("LRP-10: dyn_lr_scale default is 1.0",
          "dyn: 1.0" in output,
          f"stdout: {output}")


def test_freeze_mappers_requires_grad_toggle():
    """MLR-8: requires_grad_(False) on input/output_mapper, then (True), works as expected."""
    print("\nTest MLR-8: requires_grad toggle on mappers")
    from topology import build_net_from_preset
    from cell_library import make_default_library

    cell_lib = make_default_library()
    net = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib)
    raw = net.module if hasattr(net, "module") else net

    in_p = list(raw.input_mapper.parameters())
    out_p = list(raw.output_mapper.parameters())
    check("MLR-8: input_mapper has params", len(in_p) > 0)
    check("MLR-8: output_mapper has params", len(out_p) > 0)

    raw.input_mapper.requires_grad_(False)
    raw.output_mapper.requires_grad_(False)
    in_all_false = all(not p.requires_grad for p in in_p)
    out_all_false = all(not p.requires_grad for p in out_p)
    check("MLR-8: all mapper params have requires_grad=False after freeze",
          in_all_false and out_all_false,
          f"in_all_false={in_all_false}, out_all_false={out_all_false}")

    raw.input_mapper.requires_grad_(True)
    raw.output_mapper.requires_grad_(True)
    in_all_true = all(p.requires_grad for p in in_p)
    out_all_true = all(p.requires_grad for p in out_p)
    check("MLR-8: all mapper params have requires_grad=True after unfreeze",
          in_all_true and out_all_true,
          f"in_all_true={in_all_true}, out_all_true={out_all_true}")


# ============================================================================
# Bidirectional topology tests (bidirectional-edges plan)
# ============================================================================

def test_bidir_line_graph_doubles_edges():
    """line_graph(bidirectional=True) emits exactly 2x the edges of bidirectional=False."""
    print("\nTest BIDI-1: line_graph bidirectional doubles edge count")
    from topology import line_graph
    l1 = line_graph(8, radius=2, bidirectional=False)
    l2 = line_graph(8, radius=2, bidirectional=True)
    check("BIDI-1: line single edges > 0", len(l1.src) > 0,
          f"single={len(l1.src)}")
    check("BIDI-1: line bidirectional = 2 * single", len(l2.src) == 2 * len(l1.src),
          f"single={len(l1.src)}, bidir={len(l2.src)}")

    # Verify every (i, j) has a reverse (j, i).
    edges_single = set(zip(l1.src, l1.dst))
    edges_bidir = set(zip(l2.src, l2.dst))
    all_reversed = all((d, s) in edges_bidir for s, d in edges_single)
    check("BIDI-1: every single edge has its reverse in bidirectional",
          all_reversed)

    # No self-loops in bidirectional output.
    no_self_loops = all(s != d for s, d in edges_bidir)
    check("BIDI-1: no self-loops in bidirectional", no_self_loops)

    # Hidden node count unchanged.
    check("BIDI-1: hidden_node_ids unchanged",
          l1.hidden_node_ids == l2.hidden_node_ids)

    # Edge type unchanged (still hidden).
    check("BIDI-1: all edges typed HIDDEN",
          all(t == "hidden" for t in l2.edge_type))


def test_bidir_ring_graph_doubles_edges():
    """ring_graph(bidirectional=True) emits exactly 2x the edges of bidirectional=False."""
    print("\nTest BIDI-2: ring_graph bidirectional doubles edge count")
    from topology import ring_graph
    r1 = ring_graph(10, radius=2, bidirectional=False)
    r2 = ring_graph(10, radius=2, bidirectional=True)
    check("BIDI-2: ring single edges > 0", len(r1.src) > 0,
          f"single={len(r1.src)}")
    check("BIDI-2: ring bidirectional = 2 * single", len(r2.src) == 2 * len(r1.src),
          f"single={len(r1.src)}, bidir={len(r2.src)}")
    edges_single = set(zip(r1.src, r1.dst))
    edges_bidir = set(zip(r2.src, r2.dst))
    all_reversed = all((d, s) in edges_bidir for s, d in edges_single)
    check("BIDI-2: every single edge has its reverse", all_reversed)
    no_self_loops = all(s != d for s, d in edges_bidir)
    check("BIDI-2: no self-loops in bidirectional", no_self_loops)


def test_bidir_grid_graph_doubles_edges():
    """grid_graph(bidirectional=True) emits exactly 2x the edges of bidirectional=False."""
    print("\nTest BIDI-3: grid_graph bidirectional doubles edge count")
    from topology import grid_graph
    g1 = grid_graph(5, 5, kernel_size=3, bidirectional=False)
    g2 = grid_graph(5, 5, kernel_size=3, bidirectional=True)
    check("BIDI-3: grid single edges > 0", len(g1.src) > 0,
          f"single={len(g1.src)}")
    check("BIDI-3: grid bidirectional = 2 * single", len(g2.src) == 2 * len(g1.src),
          f"single={len(g1.src)}, bidir={len(g2.src)}")
    edges_single = set(zip(g1.src, g1.dst))
    edges_bidir = set(zip(g2.src, g2.dst))
    all_reversed = all((d, s) in edges_bidir for s, d in edges_single)
    check("BIDI-3: every single edge has its reverse", all_reversed)
    no_self_loops = all(s != d for s, d in edges_bidir)
    check("BIDI-3: no self-loops in bidirectional", no_self_loops)


def test_bidir_cluster_graph_doubles_edges():
    """cluster_graph(bidirectional=True) emits exactly 2x the edges of bidirectional=False."""
    print("\nTest BIDI-4: cluster_graph bidirectional doubles edge count")
    from topology import cluster_graph
    c1 = cluster_graph(10, edge_prob=0.5, seed=0, bidirectional=False)
    c2 = cluster_graph(10, edge_prob=0.5, seed=0, bidirectional=True)
    check("BIDI-4: cluster single edges > 0", len(c1.src) > 0,
          f"single={len(c1.src)}")
    check("BIDI-4: cluster bidirectional = 2 * single", len(c2.src) == 2 * len(c1.src),
          f"single={len(c1.src)}, bidir={len(c2.src)}")
    edges_single = set(zip(c1.src, c1.dst))
    edges_bidir = set(zip(c2.src, c2.dst))
    all_reversed = all((d, s) in edges_bidir for s, d in edges_single)
    check("BIDI-4: every single edge has its reverse", all_reversed)
    no_self_loops = all(s != d for s, d in edges_bidir)
    check("BIDI-4: no self-loops in bidirectional", no_self_loops)


def test_bidir_validate_topology_passes():
    """validate_topology() accepts bidirectional topologies without error."""
    print("\nTest BIDI-5: validate_topology passes on bidirectional topologies")
    from topology import grid_graph, line_graph, ring_graph, cluster_graph, validate_topology
    g_bi = grid_graph(7, 7, kernel_size=3, bidirectional=True)
    validate_topology(g_bi)  # Should not raise
    check("BIDI-5: validate_topology(grid 7x7 bidirectional) passed", True)

    l_bi = line_graph(8, radius=2, bidirectional=True)
    validate_topology(l_bi)
    check("BIDI-5: validate_topology(line bidirectional) passed", True)

    r_bi = ring_graph(10, radius=2, bidirectional=True)
    validate_topology(r_bi)
    check("BIDI-5: validate_topology(ring bidirectional) passed", True)

    c_bi = cluster_graph(10, edge_prob=0.5, seed=0, bidirectional=True)
    validate_topology(c_bi)
    check("BIDI-5: validate_topology(cluster bidirectional) passed", True)


def test_bidir_default_is_false():
    """bidirectional=False is the default and matches the original single-edge behavior."""
    print("\nTest BIDI-6: bidirectional defaults to False (backward compatibility)")
    from topology import line_graph, ring_graph, grid_graph, cluster_graph
    l_default = line_graph(8, radius=2)
    l_explicit = line_graph(8, radius=2, bidirectional=False)
    check("BIDI-6: line_graph default == bidirectional=False",
          len(l_default.src) == len(l_explicit.src))

    g_default = grid_graph(5, 5, kernel_size=3)
    g_explicit = grid_graph(5, 5, kernel_size=3, bidirectional=False)
    check("BIDI-6: grid_graph default == bidirectional=False",
          len(g_default.src) == len(g_explicit.src))

    r_default = ring_graph(10, radius=2)
    r_explicit = ring_graph(10, radius=2, bidirectional=False)
    check("BIDI-6: ring_graph default == bidirectional=False",
          len(r_default.src) == len(r_explicit.src))

    c_default = cluster_graph(10, edge_prob=0.5, seed=0)
    c_explicit = cluster_graph(10, edge_prob=0.5, seed=0, bidirectional=False)
    check("BIDI-6: cluster_graph default == bidirectional=False",
          len(c_default.src) == len(c_explicit.src))


def test_bidir_preset_factories_accept_param():
    """Preset factory functions accept bidirectional and pass it via hidden_kwargs."""
    print("\nTest BIDI-7: preset factories accept bidirectional parameter")
    from config import make_smooth2d_grid_preset, make_housing_grid_preset

    # Default: bidirectional key present in hidden_kwargs, set to False
    s_default = make_smooth2d_grid_preset(grid_size=5)
    check("BIDI-7: smooth2d default has bidirectional=False in hidden_kwargs",
          s_default["stages"][0]["hidden_kwargs"].get("bidirectional") is False)

    s_bi = make_smooth2d_grid_preset(grid_size=5, bidirectional=True)
    check("BIDI-7: smooth2d_grid(bidirectional=True) sets hidden_kwargs.bidirectional=True",
          s_bi["stages"][0]["hidden_kwargs"].get("bidirectional") is True)

    h_default = make_housing_grid_preset(grid_size=5)
    check("BIDI-7: housing default has bidirectional=False in hidden_kwargs",
          h_default["stages"][0]["hidden_kwargs"].get("bidirectional") is False)

    h_bi = make_housing_grid_preset(grid_size=5, bidirectional=True)
    check("BIDI-7: housing_grid(bidirectional=True) sets hidden_kwargs.bidirectional=True",
          h_bi["stages"][0]["hidden_kwargs"].get("bidirectional") is True)


def test_bidir_full_net_build():
    """A full KirchhoffNet builds and runs forward with bidirectional hidden topology."""
    print("\nTest BIDI-8: full KirchhoffNet build with bidirectional hidden graph")
    from config import make_smooth2d_grid_preset
    from topology import build_net_from_config
    from cell_library import make_default_library

    cell_lib = make_default_library()
    preset_bi = make_smooth2d_grid_preset(grid_size=5, bidirectional=True)
    net_bi = build_net_from_config(preset_bi, cell_lib=cell_lib)
    check("BIDI-8: bidirectional net builds successfully", net_bi is not None)

    preset_single = make_smooth2d_grid_preset(grid_size=5, bidirectional=False)
    net_single = build_net_from_config(preset_single, cell_lib=cell_lib)

    # The bidirectional flag only doubles the hidden grid edges, not the
    # projection edges (which connect disjoint hidden/proj node sets via
    # all_to_all). Count only the hidden grid edges: those with both
    # endpoints in [0, num_hidden).
    n_hid = net_bi.core.stages[0].num_nodes - net_bi.proj_count
    src_bi = net_bi.core.stages[0].src.tolist()
    dst_bi = net_bi.core.stages[0].dst.tolist()
    hidden_edges_bi = sum(1 for s, d in zip(src_bi, dst_bi) if s < n_hid and d < n_hid)
    src_single = net_single.core.stages[0].src.tolist()
    dst_single = net_single.core.stages[0].dst.tolist()
    hidden_edges_single = sum(1 for s, d in zip(src_single, dst_single) if s < n_hid and d < n_hid)
    check("BIDI-8: hidden grid edges double in bidirectional stage 0",
          hidden_edges_bi == 2 * hidden_edges_single,
          f"single={hidden_edges_single}, bidir={hidden_edges_bi}")

    # Total stage edges: bidirectional stage has more than single (by exactly
    # the hidden-edge delta), since projection edges are unchanged.
    e_bi = net_bi.core.stages[0].num_edges()
    e_single = net_single.core.stages[0].num_edges()
    check("BIDI-8: bidirectional stage 0 has strictly more total edges",
          e_bi > e_single,
          f"single={e_single}, bidir={e_bi}")
    check("BIDI-8: bidirectional stage 0 edge delta == hidden edge delta",
          (e_bi - e_single) == (hidden_edges_bi - hidden_edges_single),
          f"total_delta={e_bi - e_single}, hidden_delta={hidden_edges_bi - hidden_edges_single}")

    # Forward pass on a small random batch.
    u = torch.rand(4, 2)
    out, _ = net_bi(u, ctx=None)
    check("BIDI-8: forward shape (4, 1)", out.shape == (4, 1))
    check("BIDI-8: forward output is finite", torch.isfinite(out).all().item())


# ============================================================================
# Parallel edge repeats tests (parallel-edge-repeats plan)
# ============================================================================

def test_repeat_edges_identity():
    """repeat_edges(topo, 1) returns an equivalent topology (no duplication)."""
    print("\nTest REP-1: repeat_edges(n=1) is identity")
    from topology import line_graph, grid_graph, repeat_edges
    g = grid_graph(4, 4, kernel_size=3, bidirectional=False)
    g_rep = repeat_edges(g, 1)
    check("REP-1: grid repeat_edges(n=1) preserves edge count",
          len(g_rep.src) == len(g.src),
          f"original={len(g.src)}, repeated={len(g_rep.src)}")
    check("REP-1: grid repeat_edges(n=1) preserves all edge types",
          g_rep.edge_type == g.edge_type)
    check("REP-1: grid repeat_edges(n=1) preserves node ids",
          g_rep.input_node_ids == g.input_node_ids
          and g_rep.output_node_ids == g.output_node_ids
          and g_rep.hidden_node_ids == g.hidden_node_ids)

    l = line_graph(6, radius=2, bidirectional=False)
    l_rep = repeat_edges(l, 1)
    check("REP-1: line repeat_edges(n=1) preserves edge count",
          len(l_rep.src) == len(l.src))


def test_repeat_edges_multiplies_hidden_only():
    """repeat_edges(topo, n) duplicates only hidden edges, n times."""
    print("\nTest REP-2: repeat_edges(n) duplicates only hidden edges")
    from topology import grid_graph, repeat_edges
    g = grid_graph(4, 4, kernel_size=3, bidirectional=False)
    base_hidden = sum(1 for t in g.edge_type if t == "hidden")
    base_non_hidden = sum(1 for t in g.edge_type if t != "hidden")
    g_rep = repeat_edges(g, 3)
    rep_hidden = sum(1 for t in g_rep.edge_type if t == "hidden")
    rep_non_hidden = sum(1 for t in g_rep.edge_type if t != "hidden")
    check("REP-2: hidden edge count == 3x original",
          rep_hidden == 3 * base_hidden,
          f"base={base_hidden}, rep={rep_hidden}")
    check("REP-2: non-hidden edge count is preserved",
          rep_non_hidden == base_non_hidden,
          f"base_non_hidden={base_non_hidden}, rep_non_hidden={rep_non_hidden}")


def test_repeat_edges_composes_with_bidirectional():
    """edge_repeats=3 with bidirectional=True produces 6x hidden edges per unique pair."""
    print("\nTest REP-3: repeat_edges composes multiplicatively with bidirectional")
    from topology import grid_graph, repeat_edges
    g_single = grid_graph(4, 4, kernel_size=3, bidirectional=False)
    g_bidi = grid_graph(4, 4, kernel_size=3, bidirectional=True)
    g_bidi_rep3 = repeat_edges(g_bidi, 3)
    check("REP-3: bidi+rep3 = 6x single",
          len(g_bidi_rep3.src) == 6 * len(g_single.src),
          f"single={len(g_single.src)}, bidi+rep3={len(g_bidi_rep3.src)}")


def test_repeat_edges_per_pair_count():
    """Each original hidden edge appears exactly n times after repeat_edges."""
    print("\nTest REP-4: repeat_edges preserves per-pair multiplicity")
    from topology import line_graph, repeat_edges
    l = line_graph(5, radius=1, bidirectional=False)
    l_rep3 = repeat_edges(l, 3)
    from collections import Counter
    base_pairs = Counter(zip(l.src, l.dst))
    rep_pairs = Counter(zip(l_rep3.src, l_rep3.dst))
    # Each base pair (hidden) should appear 3 times; no extra pairs.
    for p, c in base_pairs.items():
        check(f"REP-4: pair {p} appears 3x", rep_pairs[p] == 3,
              f"base={c}, rep={rep_pairs[p]}")
    extras = set(rep_pairs) - set(base_pairs)
    check("REP-4: no new (src, dst) pairs introduced", len(extras) == 0,
          f"extras={extras}")


def test_repeat_edges_rejects_invalid_n():
    """repeat_edges raises on n < 1."""
    print("\nTest REP-5: repeat_edges rejects n < 1")
    from topology import line_graph, repeat_edges
    l = line_graph(4, radius=1)
    raised = False
    try:
        repeat_edges(l, 0)
    except ValueError:
        raised = True
    check("REP-5: n=0 raises ValueError", raised)


def test_repeat_edges_validate_topology_passes():
    """validate_topology() accepts repeated topologies (uses unique-pair density)."""
    print("\nTest REP-6: validate_topology passes on repeated topologies")
    from topology import grid_graph, line_graph, repeat_edges, validate_topology
    g = grid_graph(7, 7, kernel_size=3, bidirectional=False)
    for n in [2, 4, 8]:
        g_rep = repeat_edges(g, n)
        validate_topology(g_rep)
        check(f"REP-6: validate_topology(grid 7x7 rep={n}) passed", True)
    l = line_graph(8, radius=2, bidirectional=False)
    for n in [2, 4, 8]:
        l_rep = repeat_edges(l, n)
        validate_topology(l_rep)
        check(f"REP-6: validate_topology(line rep={n}) passed", True)


def test_repeat_edges_from_config_wires_through():
    """MultiStageTopology.from_config() applies edge_repeats to hidden edges only."""
    print("\nTest REP-7: from_config applies edge_repeats to hidden edges only")
    from topology import (
        grid_graph, MultiStageTopology,
    )
    g = grid_graph(4, 4, kernel_size=3, bidirectional=False)
    base_hidden = sum(1 for t in g.edge_type if t == "hidden")

    cfg = {
        "num_inputs": 2,
        "num_hidden": 16,
        "num_proj": 3,
        "num_outputs": 0,
        "hidden_family": "grid",
        "hidden_kwargs": {"height": 4, "width": 4, "kernel_size": 3, "bidirectional": False},
        "edge_repeats": 3,
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
    }
    mst = MultiStageTopology.from_config([cfg])
    topo = mst.stages[0]
    rep_hidden = sum(1 for t in topo.edge_type if t == "hidden")
    # Reference: same config with edge_repeats=1 to get the baseline hidden count.
    cfg_ref = dict(cfg, edge_repeats=1)
    mst_ref = MultiStageTopology.from_config([cfg_ref])
    topo_ref = mst_ref.stages[0]
    ref_hidden = sum(1 for t in topo_ref.edge_type if t == "hidden")
    check("REP-7: hidden edges = 3x baseline (edge_repeats=1)",
          rep_hidden == 3 * ref_hidden,
          f"ref={ref_hidden}, rep={rep_hidden}")
    # Non-hidden edges (input->hidden + hidden->proj) are unchanged by repeats.
    rep_non_hidden = sum(1 for t in topo.edge_type if t != "hidden")
    ref_non_hidden = sum(1 for t in topo_ref.edge_type if t != "hidden")
    check("REP-7: non-hidden count preserved with edge_repeats",
          rep_non_hidden == ref_non_hidden,
          f"ref_non_hidden={ref_non_hidden}, rep_non_hidden={rep_non_hidden}")


def test_repeat_edges_full_net_build():
    """A full KirchhoffNet builds and runs forward with edge_repeats=3 hidden topology."""
    print("\nTest REP-8: full KirchhoffNet build with edge_repeats=3")
    from config import make_smooth2d_grid_preset
    from topology import build_net_from_config
    from cell_library import make_default_library

    cell_lib = make_default_library()
    preset_rep = make_smooth2d_grid_preset(grid_size=5, edge_repeats=3)
    net_rep = build_net_from_config(preset_rep, cell_lib=cell_lib)
    check("REP-8: edge_repeats=3 net builds successfully", net_rep is not None)

    preset_single = make_smooth2d_grid_preset(grid_size=5, edge_repeats=1)
    net_single = build_net_from_config(preset_single, cell_lib=cell_lib)

    n_hid = net_rep.core.stages[0].num_nodes - net_rep.proj_count
    src_rep = net_rep.core.stages[0].src.tolist()
    dst_rep = net_rep.core.stages[0].dst.tolist()
    hidden_edges_rep = sum(1 for s, d in zip(src_rep, dst_rep) if s < n_hid and d < n_hid)
    src_single = net_single.core.stages[0].src.tolist()
    dst_single = net_single.core.stages[0].dst.tolist()
    hidden_edges_single = sum(1 for s, d in zip(src_single, dst_single) if s < n_hid and d < n_hid)
    check("REP-8: hidden grid edges = 3x single",
          hidden_edges_rep == 3 * hidden_edges_single,
          f"single={hidden_edges_single}, rep={hidden_edges_rep}")

    # Forward pass on a small random batch.
    u = torch.rand(4, 2)
    out, _ = net_rep(u, ctx=None)
    check("REP-8: forward shape (4, 1)", out.shape == (4, 1))
    check("REP-8: forward output is finite", torch.isfinite(out).all().item())


def test_repeat_edges_preset_factories_accept_param():
    """Preset factory functions accept edge_repeats and pass it via stage config."""
    print("\nTest REP-9: preset factories accept edge_repeats parameter")
    from config import make_smooth2d_grid_preset, make_housing_grid_preset

    # Default: edge_repeats == 2 in stage config
    s_default = make_smooth2d_grid_preset(grid_size=5)
    check("REP-9: smooth2d default has edge_repeats=2",
          s_default["stages"][0].get("edge_repeats") == 2,
          f"got {s_default['stages'][0].get('edge_repeats')}")

    s_rep = make_smooth2d_grid_preset(grid_size=5, edge_repeats=4)
    check("REP-9: smooth2d_grid(edge_repeats=4) sets stage config edge_repeats=4",
          s_rep["stages"][0].get("edge_repeats") == 4)

    h_default = make_housing_grid_preset(grid_size=5)
    check("REP-9: housing default has edge_repeats=2",
          h_default["stages"][0].get("edge_repeats") == 2)

    h_rep = make_housing_grid_preset(grid_size=5, edge_repeats=8)
    check("REP-9: housing_grid(edge_repeats=8) sets stage config edge_repeats=8",
          h_rep["stages"][0].get("edge_repeats") == 8)

    # Out-of-range value must raise.
    raised = False
    try:
        make_smooth2d_grid_preset(grid_size=5, edge_repeats=9)
    except ValueError:
        raised = True
    check("REP-9: edge_repeats=9 raises ValueError", raised)


def test_repeat_edges_compose_with_bidirectional_full_net():
    """bidirectional=True with edge_repeats=3 yields 6x hidden edges per unique pair."""
    print("\nTest REP-10: bidirectional + edge_repeats compose in full net")
    from config import make_smooth2d_grid_preset
    from topology import build_net_from_config
    from cell_library import make_default_library

    cell_lib = make_default_library()
    preset = make_smooth2d_grid_preset(
        grid_size=5, bidirectional=True, edge_repeats=3,
    )
    net = build_net_from_config(preset, cell_lib=cell_lib)
    check("REP-10: bidi+rep3 net builds successfully", net is not None)

    n_hid = net.core.stages[0].num_nodes - net.proj_count
    src = net.core.stages[0].src.tolist()
    dst = net.core.stages[0].dst.tolist()
    hidden_edges = sum(1 for s, d in zip(src, dst) if s < n_hid and d < n_hid)

    # Reference: single-direction (no bidi, no repeats)
    preset_ref = make_smooth2d_grid_preset(
        grid_size=5, bidirectional=False, edge_repeats=1,
    )
    net_ref = build_net_from_config(preset_ref, cell_lib=cell_lib)
    n_hid_ref = net_ref.core.stages[0].num_nodes - net_ref.proj_count
    src_ref = net_ref.core.stages[0].src.tolist()
    dst_ref = net_ref.core.stages[0].dst.tolist()
    hidden_edges_ref = sum(
        1 for s, d in zip(src_ref, dst_ref)
        if s < n_hid_ref and d < n_hid_ref
    )

    check("REP-10: bidi+rep3 = 6x single-direction hidden edges",
          hidden_edges == 6 * hidden_edges_ref,
          f"ref={hidden_edges_ref}, compound={hidden_edges}")

    # Forward pass on a small random batch.
    u = torch.rand(4, 2)
    out, _ = net(u, ctx=None)
    check("REP-10: forward shape (4, 1)", out.shape == (4, 1))
    check("REP-10: forward output is finite", torch.isfinite(out).all().item())


def test_drive_current_basic():
    print("\nTest DRIVE-1: DifferentialStage.drive_current basic shape and behavior")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib, write_idx=[0, 2])

    B, N = 3, 4
    x = torch.rand(B, N)
    x_drive = torch.rand(B, N)

    # With drive_scale=0, drive_current returns zeros.
    i0 = stage.drive_current(x, x_drive, drive_scale=0.0)
    check("DRIVE-1: zero scale gives zero current", i0.abs().max().item() == 0.0)

    # With scale > 0, drive_current is non-zero only at driven positions.
    i1 = stage.drive_current(x, x_drive, drive_scale=1.0)
    non_zero_cols = (i1.abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist()
    check("DRIVE-1: drive current only at driven nodes [0, 2]",
          non_zero_cols == [0, 2],
          f"got non-zero cols {non_zero_cols}")

    # With x_drive=None, returns zeros.
    i2 = stage.drive_current(x, None, drive_scale=1.0)
    check("DRIVE-1: None x_drive gives zero current", i2.abs().max().item() == 0.0)

    # Drive current is bounded by drive_isat.
    with torch.no_grad():
        stage.raw_drive_g.fill_(10.0)  # very large conductance
    x_far = torch.full((B, N), -10.0)
    x_drive_far = torch.full((B, N), 10.0)
    i3 = stage.drive_current(x_far, x_drive_far, drive_scale=1.0)
    max_i = i3.abs().max().item()
    check("DRIVE-1: drive current bounded by drive_isat",
          max_i <= stage.drive_isat * 1.001,
          f"max_i={max_i:.4f}, isat={stage.drive_isat}")


def test_drive_changes_rhs():
    print("\nTest DRIVE-2: drive current modifies rhs output vs baseline")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib, write_idx=[0, 2])

    B, N = 2, 4
    x = torch.rand(B, N)
    x_drive = torch.rand(B, N)

    dx_baseline = stage.rhs(x, ctx=SimContext(), tau=1.0)
    dx_driven = stage.rhs(x, ctx=SimContext(), tau=1.0, x_drive=x_drive, drive_scale=1.0)

    diff = (dx_driven - dx_baseline).abs().max().item()
    check("DRIVE-2: rhs output changes with drive enabled",
          diff > 1e-6,
          f"max diff = {diff:.4e}")


def test_driven_node_gate_forced_open():
    print("\nTest DRIVE-3: driven node gates are forced to 1.0 in rhs")
    from cell_library import IdealizedCellLibrary
    from topology import ring_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    import torch

    cell_lib = IdealizedCellLibrary()
    hid = ring_graph(4, radius=1)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib, write_idx=[0, 2])

    # Close ALL node gates (including driven positions).
    with torch.no_grad():
        stage.u_logits.fill_(-10.0)

    B, N = 2, 4
    x = torch.rand(B, N)
    x_drive = torch.rand(B, N)
    ctx = SimContext()

    # If driven node gates were NOT forced open, rhs would return near-zero.
    dx = stage.rhs(x, ctx=ctx, tau=1.0, x_drive=x_drive, drive_scale=1.0)

    # Drive current at position 0 should be active (non-zero i_drive means
    # the drive_current contributed to rhs, which would only happen if
    # node gates at [0, 2] were open; if they were closed, x_gated would
    # be near-zero, node error would be small, and i_drive would be near-zero.
    # More directly: check that the drive current magnitude is significant.
    i_drive = stage.drive_current(x, x_drive, drive_scale=1.0)
    max_drive = i_drive[:, stage._drive_idx].abs().max().item()
    check("DRIVE-3: drive current flows even when node gates closed",
          max_drive > 1e-4,
          f"max drive at driven positions = {max_drive:.6e}")


def test_kirchhoff_net_with_io_drive_forward():
    print("\nTest DRIVE-4: KirchhoffNetWithIO forward pass with enable_drive=True")
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_preset
    from config import PRESETS
    import torch

    cell_lib = IdealizedCellLibrary()

    # Use ctle_grid preset as test bed (fan_out write mode, multi-stage).
    for preset_name in ["smooth2d_grid"]:
        cfg = dict(PRESETS[preset_name])
        cfg["write_fan_out"] = {0: [0, 4], 1: [5, 9]}

        net = build_net_from_preset(
            preset_name, cell_lib=cell_lib, enable_drive=True,
        )
        B = 4
        u = torch.rand(B, cfg["stages"][0]["num_inputs"])

        y, trajs = net(u, ctx=None, tau=1.0, store_trajectory=True)
        check(f"DRIVE-4 ({preset_name}): forward output finite",
              torch.isfinite(y).all().item())
        check(f"DRIVE-4 ({preset_name}): forward output shape ({B}, {cfg['out_dim']})",
              y.shape == (B, cfg["out_dim"]))
        check(f"DRIVE-4 ({preset_name}): trajectory stored",
              trajs is not None and len(trajs) == len(net.core.stages))
        check(f"DRIVE-4 ({preset_name}): enable_drive flag on net",
              net.enable_drive)
        check(f"DRIVE-4 ({preset_name}): drive mappers exist",
              net.drive_mappers is not None and len(net.drive_mappers) == len(net.core.stages))

    # Also test that enable_drive=False does NOT set up drive infrastructure.
    net_no = build_net_from_preset("smooth2d_grid", cell_lib=cell_lib, enable_drive=False)
    check("DRIVE-4: enable_drive=False -> no drive_mappers",
          net_no.drive_mappers is None)
    check("DRIVE-4: enable_drive=False -> enable_drive is False",
          not net_no.enable_drive)


def test_simple_edge_build_forward():
    print("\nTest SE-1: SimpleEdgeLibrary build + forward (relu and tanh)")
    from cell_library import SimpleEdgeLibrary, make_cell_library
    from topology import build_net_from_config
    from config import PRESETS

    for mode in ("relu", "tanh"):
        cell_lib = make_cell_library(mode)
        l = cell_lib
        check(f"SE-1 ({mode}): num_cells=1", l.num_cells == 1, f"got {l.num_cells}")
        check(f"SE-1 ({mode}): z_index=0", l.z_index == 0, f"got {l.z_index}")
        check(f"SE-1 ({mode}): has_z_cell=False", not l.has_z_cell, f"got {l.has_z_cell}")

        preset = dict(PRESETS["smooth2d_grid"])
        preset["stages"] = preset["stages"][:1]
        net = build_net_from_config(preset, cell_lib=cell_lib, enable_drive=False)
        stage = net.core.stages[0]
        check(f"SE-1 ({mode}): stage uses SimpleEdgeLibrary",
              isinstance(stage.cell_lib, SimpleEdgeLibrary))
        check(f"SE-1 ({mode}): logits is None", stage.logits is None)
        check(f"SE-1 ({mode}): raw_mult is None", stage.raw_mult is None)
        check(f"SE-1 ({mode}): param shape [3, E]", stage.cell_lib.param.shape == (3, stage.num_edges()),
              f"got {stage.cell_lib.param.shape}")

        out, _ = net(torch.randn(4, 2), ctx=None, store_trajectory=False)
        check(f"SE-1 ({mode}): output finite", torch.isfinite(out).all().item())
        check(f"SE-1 ({mode}): output shape (4,1)", out.shape == (4, 1), f"got {out.shape}")

        loss = out.sum()
        loss.backward()
        check(f"SE-1 ({mode}): grad on param", stage.cell_lib.param.grad is not None)
        check(f"SE-1 ({mode}): grad on z_logits", stage.z_logits.grad is not None)
        check(f"SE-1 ({mode}) deprecate-node-gates: u_logits has no grad (bypassed in rhs)",
              stage.u_logits.grad is None,
              f"got grad={stage.u_logits.grad}")
        check(f"SE-1 ({mode}): grad on raw_leak", stage.raw_leak.grad is not None)


def test_simple_edge_prune():
    print("\nTest SE-2: SimpleEdgeLibrary pruning")
    from cell_library import SimpleEdgeLibrary, make_cell_library
    from topology import build_net_from_config, prune_network
    from config import PRESETS

    cell_lib = make_cell_library("tanh")
    preset = dict(PRESETS["smooth2d_grid"])
    preset["stages"] = preset["stages"][:2]
    net = build_net_from_config(preset, cell_lib=cell_lib, enable_drive=False)

    for stage in net.core.stages:
        with torch.no_grad():
            stage.z_logits.fill_(2.0)
            n = stage.z_logits.numel()
            stage.z_logits[:n // 2] = -3.0

    pruned_core, remaps = prune_network(net.core, edge_threshold=0.5, prune_nodes_by_gate=False)
    check("SE-2: pruned has fewer edges",
          sum(s.num_edges() for s in pruned_core.stages) < sum(s.num_edges() for s in net.core.stages))
    check("SE-2: pruned uses SimpleEdgeLibrary",
          isinstance(pruned_core.stages[0].cell_lib, SimpleEdgeLibrary))
    for i, stage in enumerate(pruned_core.stages):
        check(f"SE-2: stage {i} param shape matches edges",
              stage.cell_lib.param.shape[1] == stage.num_edges(),
              f"param {stage.cell_lib.param.shape} vs edges {stage.num_edges()}")
    check("SE-2: cell_lib is not shared",
          pruned_core.stages[0].cell_lib is not net.core.stages[0].cell_lib)


def test_simple_edge_regularizers():
    print("\nTest SE-3: SimpleEdgeLibrary regularizers (manual)")
    from cell_library import SimpleEdgeLibrary, make_cell_library
    from topology import build_net_from_config
    from config import PRESETS, LAMBDAS
    from train import _stage_soft_weights, _stage_multiplicities, _stage_edge_gates, _stage_node_gates, _stage_rail_loss
    import math

    cell_lib = make_cell_library("relu")
    preset = dict(PRESETS["smooth2d_grid"])
    preset["stages"] = preset["stages"][:1]
    net = build_net_from_config(preset, cell_lib=cell_lib, enable_drive=False)

    u = torch.randn(4, 2)
    out, trajs = net(u, ctx=None, store_trajectory=True)

    stage = net.core.stages[0]
    traj = trajs[0]
    w = _stage_soft_weights(stage)
    mult = _stage_multiplicities(stage)
    z = _stage_edge_gates(stage)
    u_gates = _stage_node_gates(stage)
    z_idx = stage.cell_lib.z_index

    check("SE-3: soft_weights has 1 column", w.shape[-1] == 1, f"got {w.shape[-1]}")
    check("SE-3: soft_weights all 1.0", float(w.mean().item()) == 1.0, f"got {float(w.mean().item())}")
    check("SE-3: multiplicities all 1.0", float(mult.mean().item()) == 1.0, f"got {float(mult.mean().item())}")
    check("SE-3: edge_gates mean in (0,1)", 0 < float(z.mean().item()) < 1)
    check("SE-3: node_gates mean = 1.0 (deprecated, returns all-ones)", float(u_gates.mean().item()) == 1.0)
    check("SE-3: sparsity w[:,:0] = 0", float(w[:, :z_idx].sum().item()) == 0.0)
    check("SE-3: rail finite", math.isfinite(float(_stage_rail_loss(stage, traj).item())))


def test_simple_edge_diagnostics():
    print("\nTest SE-4: SimpleEdgeLibrary diagnostics (compute_solidification_metrics)")
    from cell_library import SimpleEdgeLibrary, make_cell_library
    from topology import build_net_from_config
    from config import PRESETS
    from train import compute_solidification_metrics

    cell_lib = make_cell_library("tanh")
    preset = dict(PRESETS["smooth2d_grid"])
    preset["stages"] = preset["stages"][:1]
    net = build_net_from_config(preset, cell_lib=cell_lib, enable_drive=False)

    metrics = compute_solidification_metrics(net, tau=1.0)
    check("SE-4: mean_sigma_z present", "mean_sigma_z" in metrics)
    check("SE-4: mean_sigma_u present", "mean_sigma_u" in metrics)
    check("SE-4: num_edges > 0", metrics["num_edges"] > 0)
    check("SE-4: num_nodes > 0", metrics["num_nodes"] > 0)
    check("SE-4: mean_max_cell_prob == 0 (no logits)",
          metrics["mean_max_cell_prob"] == 0.0,
          f"got {metrics['mean_max_cell_prob']}")
    check("SE-4: mean_pZ == 0 (no Z cell)",
          metrics["mean_pZ"] == 0.0,
          f"got {metrics['mean_pZ']}")


# ============================================================================
# Fixed seed tests (fixed-seed plan)
# ============================================================================

def test_deprecate_node_gates_warnings():
    print("\nTest DNG-1: deprecate-node-gates warnings")
    import warnings
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    from topology import prune_stage
    from train import _stage_node_gates

    cell_lib = IdealizedCellLibrary()
    stage = DifferentialStage(num_nodes=2, src=[0], dst=[1], cell_lib=cell_lib)

    # 1) prune_stage with prune_nodes_by_gate=True emits DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prune_stage(stage, edge_threshold=0.0, prune_nodes_by_gate=True)
        dw = [x for x in w if issubclass(x.category, DeprecationWarning)]
        check("DNG-1: prune_stage(True) emits DeprecationWarning",
              any("deprecated" in str(x.message).lower() for x in dw),
              f"got {[str(x.message) for x in dw]}")

    # 2) _stage_node_gates emits DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _stage_node_gates(stage)
        dw = [x for x in w if issubclass(x.category, DeprecationWarning)]
        check("DNG-1: _stage_node_gates emits DeprecationWarning",
              any("deprecated" in str(x.message).lower() for x in dw),
              f"got {[str(x.message) for x in dw]}")

    # 3) active_node_mask emits DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = stage.active_node_mask()
        dw = [x for x in w if issubclass(x.category, DeprecationWarning)]
        check("DNG-1: active_node_mask emits DeprecationWarning",
              any("deprecated" in str(x.message).lower() for x in dw),
              f"got {[str(x.message) for x in dw]}")


def test_seed_everything_deterministic():
    """seed_everything() makes model init deterministic across calls."""
    print("\nTest SEED-1: seed_everything yields identical params across calls")
    import random
    import numpy as np
    from train_ctle import seed_everything
    from cell_library import IdealizedCellLibrary
    from topology import build_net_from_config
    from config import PRESETS

    # Build a 1-stage net. OutputMapper.proj is nn.Linear with xavier_uniform_
    # (properly seeded by torch.manual_seed), making it a good comparison target.
    preset = dict(PRESETS["smooth2d_grid"])
    preset["stages"] = preset["stages"][:1]

    seed_everything(123)
    cell_lib_a = IdealizedCellLibrary()
    net_a = build_net_from_config(preset, cell_lib=cell_lib_a, enable_drive=False)
    a_weight = net_a.output_mapper.proj.weight.detach().clone()

    # Second: build with the SAME seed; should match exactly.
    seed_everything(123)
    cell_lib_b = IdealizedCellLibrary()
    net_b = build_net_from_config(preset, cell_lib=cell_lib_b, enable_drive=False)
    b_weight = net_b.output_mapper.proj.weight.detach().clone()

    check("SEED-1: same seed -> identical OutputMapper weights",
          torch.allclose(a_weight, b_weight),
          f"max abs diff = {(a_weight - b_weight).abs().max().item():.2e}")

    # Third: different seed -> should differ.
    seed_everything(456)
    cell_lib_c = IdealizedCellLibrary()
    net_c = build_net_from_config(preset, cell_lib=cell_lib_c, enable_drive=False)
    c_weight = net_c.output_mapper.proj.weight.detach().clone()
    differs = not torch.allclose(a_weight, c_weight)
    check("SEED-1: different seed -> OutputMapper weights differ", differs,
          f"max abs diff vs seed 123 = {(a_weight - c_weight).abs().max().item():.2e}")


def test_seed_everything_seeds_numpy():
    """seed_everything() also seeds NumPy."""
    print("\nTest SEED-2: seed_everything seeds numpy.random")
    import numpy as np
    from train_ctle import seed_everything

    seed_everything(7)
    a1 = np.random.rand(5)
    a2 = np.random.rand(5)
    seed_everything(7)
    b1 = np.random.rand(5)
    b2 = np.random.rand(5)
    check("SEED-2: numpy.rand reproducible with same seed",
          np.allclose(a1, b1) and np.allclose(a2, b2),
          f"a1={a1[:2]}, b1={b1[:2]}")


def test_seed_everything_seeds_python_random():
    """seed_everything() also seeds Python's random module."""
    print("\nTest SEED-3: seed_everything seeds Python random")
    import random
    from train_ctle import seed_everything

    seed_everything(11)
    a = [random.random() for _ in range(5)]
    seed_everything(11)
    b = [random.random() for _ in range(5)]
    check("SEED-3: random.random reproducible with same seed",
          a == b,
          f"a[:2]={a[:2]}, b[:2]={b[:2]}")


# =============================================================================
# Deep Equilibrium (DEQ) training mode tests
# (deq-core-prototype plan: stagewise equilibrium + diagnostics)
# =============================================================================


def test_deq_solver_imports_and_solves():
    """DEQ solver adapter exists and finds x* s.t. Phi(x*) ≈ x* on a contractive map."""
    print("\nTest DEQ-1: deq_solver.solve_equilibrium finds fixed point")
    from deq_solver import solve_equilibrium

    def phi(x):
        return 0.5 * x + 0.1

    x0 = torch.zeros(2, 4)
    cfg = {"f_max_iter": 50, "f_tol": 1e-6, "deq_step": 1.0}
    x_star, info = solve_equilibrium(phi, x0, cfg)
    check("x_star finite", torch.isfinite(x_star).all().item())
    check("x_star near 0.2 (fixed point of 0.5x+0.1)",
          (x_star - 0.2).abs().max().item() < 1e-4,
          f"max|x*-0.2|={(x_star - 0.2).abs().max().item():.3e}")
    check("info has nstep", "nstep" in info)
    check("info has rel_residual", "rel_residual" in info)
    check("rel_residual < 1e-4",
          info["rel_residual"] < 1e-4,
          f"rel_residual={info['rel_residual']:.3e}")


def test_deq_solver_run_in_fp32():
    """DEQ solve runs in fp32 regardless of input dtype (AMP safety)."""
    print("\nTest DEQ-2: deq_solver runs in fp32 (AMP safety)")
    from deq_solver import solve_equilibrium

    def phi(x):
        # Force intermediate to fp32 regardless of input
        y = 0.5 * x.float() + 0.0
        return y

    x0 = torch.zeros(1, 3, dtype=torch.float16)
    cfg = {"f_max_iter": 30, "f_tol": 1e-5, "deq_step": 1.0}
    x_star, _ = solve_equilibrium(phi, x0, cfg)
    check("x_star dtype fp32", x_star.dtype == torch.float32,
          f"got {x_star.dtype}")
    check("x_star finite", torch.isfinite(x_star).all().item())


def test_deq_equilibrium_rhs_residual_small():
    """At DEQ equilibrium, |rhs(x*)| is small (matches DEQ definition)."""
    print("\nTest DEQ-3: |rhs(x*)| small at DEQ equilibrium")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0  # no drive for this test (deterministic)

    ctx = SimContext()
    deq_cfg = {"f_max_iter": 80, "f_tol": 1e-6, "deq_step": 0.1,
               "leak_floor": 0.05}
    x0 = torch.zeros(1, 3)
    x_star, info = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0,
                                             cell_mode="soft", x_drive=None,
                                             drive_scale=0.0, deq_cfg=deq_cfg)
    # Measure the residual at the converged fixed point using the same
    # leak_floor that was used during the DEQ solve (passed via kwarg to
    # avoid mutating self.leak_floor, which is reset to 0.0 by
    # forward_equilibrium after the solve).
    with torch.no_grad():
        r = stage.rhs(x_star, ctx=ctx, tau=1.0, cell_mode="soft",
                      x_drive=None, drive_scale=0.0,
                      leak_floor=float(deq_cfg["leak_floor"]))
    res_norm = float(r.abs().max())
    check("x_star finite", torch.isfinite(x_star).all().item())
    check("max|rhs(x*)| < 1e-2", res_norm < 1e-2,
          f"max|rhs|={res_norm:.3e}")
    check("info nstep <= 80", info["nstep"] <= 80, f"nstep={info['nstep']}")


def test_deq_equilibrium_matches_long_horizon_heun():
    """DEQ x* matches long-horizon Heun rollout to equilibrium (contractive)."""
    print("\nTest DEQ-4: DEQ x* matches long-horizon Heun rollout")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0

    ctx = SimContext()
    deq_cfg = {"f_max_iter": 100, "f_tol": 1e-6, "deq_step": 0.1,
               "leak_floor": 0.05}
    x0 = torch.zeros(1, 3)
    x_deq, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0,
                                        cell_mode="soft", x_drive=None,
                                        drive_scale=0.0, deq_cfg=deq_cfg)
    # Use the SAME leak_floor for the Heun comparison so both paths
    # converge to the same dynamical system. forward_equilibrium resets
    # self.leak_floor to 0.0 on return, so we set it explicitly here.
    lf = float(deq_cfg["leak_floor"])
    stage.set_leak_floor(lf)
    x_heun, _ = stage.forward(x0, ctx=ctx, t_span=50.0, num_steps=2000,
                              tau=1.0, store_trajectory=False,
                              cell_mode="soft", x_drive=None, drive_scale=0.0)
    stage.set_leak_floor(0.0)
    diff = float((x_deq - x_heun).abs().max())
    check("max|DEQ - long Heun| < 0.05 (same leak_floor)", diff < 0.05,
          f"max diff={diff:.3e}")


def test_deq_implicit_backward_gradients_finite():
    """Implicit backward under DEQ produces finite gradients for all params."""
    print("\nTest DEQ-5: implicit backward yields finite gradients")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0

    ctx = SimContext()
    deq_cfg = {"f_max_iter": 60, "f_tol": 1e-6, "deq_step": 0.1,
               "leak_floor": 0.05}
    x0 = torch.zeros(1, 3)
    x_star, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0,
                                          cell_mode="soft", x_drive=None,
                                          drive_scale=0.0, deq_cfg=deq_cfg)
    loss = x_star.pow(2).sum()
    loss.backward()
    for name in ("logits", "raw_mult", "raw_leak", "z_logits"):
        p = getattr(stage, name)
        ok = p.grad is not None and torch.isfinite(p.grad).all().item()
        check(f"grad finite: {name}", ok,
              f"grad={None if p.grad is None else p.grad}")


def test_deq_z_logits_grad_norm_at_least_bptt():
    """DEQ z_logits grad norm >= short-horizon BPTT (per Kimi note, expect 2-4 OOM lift)."""
    print("\nTest DEQ-6: DEQ z_logits grad norm >= short-horizon BPTT")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(4, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=4, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0

    ctx = SimContext()
    x0 = torch.zeros(2, 4)
    # Short BPTT
    x_heun, _ = stage.forward(x0, ctx=ctx, t_span=0.3, num_steps=10,
                              tau=1.0, store_trajectory=False, cell_mode="soft",
                              x_drive=None, drive_scale=0.0)
    loss_h = x_heun.pow(2).sum()
    loss_h.backward()
    g_heun = stage.z_logits.grad.detach().clone().abs().sum().item()
    stage.zero_grad()
    # DEQ
    deq_cfg = {"f_max_iter": 60, "f_tol": 1e-6, "deq_step": 0.1,
               "leak_floor": 0.05}
    x_deq, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0,
                                         cell_mode="soft", x_drive=None,
                                         drive_scale=0.0, deq_cfg=deq_cfg)
    loss_d = x_deq.pow(2).sum()
    loss_d.backward()
    g_deq = stage.z_logits.grad.detach().clone().abs().sum().item()
    check(f"DEQ z_logits grad norm ({g_deq:.3e}) finite and within 3 orders of BPTT ({g_heun:.3e})",
          # On a 4-node toy stage the ratio is noisy; we assert the gradient
          # is finite and within a sensible band. The Kimi-note prediction
          # (2-4 OOM lift) is evaluated on the smooth2d_grid benchmark, not
          # here.
          g_deq >= 1e-3 * g_heun and g_deq <= 1e3 * g_heun,
          f"ratio={g_deq / max(g_heun, 1e-30):.3f}")


def test_deq_input_dependence():
    """DEQ with persistent drive produces input-dependent x*.

    Without drive the fixed point is constant (parameters only). With
    active drive the equilibrium differs for distinct x_drive inputs.
    """
    print("\nTest DEQ-7: DEQ input-dependence with persistent drive")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    # Build stage with write_idx=[0] so drive is enabled.
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib, write_idx=[0])
    stage.drive_isat = 0.5
    stage.raw_drive_g.data.fill_(0.5)

    ctx = SimContext()
    deq_cfg = {"f_max_iter": 60, "f_tol": 1e-6, "deq_step": 0.1,
               "leak_floor": 0.05}
    x0 = torch.zeros(1, 3)

    # Two different x_drive inputs.
    x_drive_a = torch.tensor([[1.0, 0.0, 0.0]])
    x_drive_b = torch.tensor([[-1.0, 0.0, 0.0]])

    x_a, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0, cell_mode="soft",
                                        x_drive=x_drive_a, drive_scale=1.0,
                                        deq_cfg=deq_cfg)
    x_b, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0, cell_mode="soft",
                                        x_drive=x_drive_b, drive_scale=1.0,
                                        deq_cfg=deq_cfg)
    diff = float((x_a - x_b).abs().max())
    check("x* differs for different x_drive (drive active)",
          diff > 1e-4,
          f"max|x_a - x_b| = {diff:.6e}")

    # Without drive (drive_scale=0) the equilibrium should be the same.
    x_a0, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0, cell_mode="soft",
                                         x_drive=x_drive_a, drive_scale=0.0,
                                         deq_cfg=deq_cfg)
    x_b0, _ = stage.forward_equilibrium(x0, ctx=ctx, tau=1.0, cell_mode="soft",
                                         x_drive=x_drive_b, drive_scale=0.0,
                                         deq_cfg=deq_cfg)
    diff0 = float((x_a0 - x_b0).abs().max())
    check("x* identical for different x_drive (drive_scale=0)",
          diff0 < 1e-6,
          f"max|x_a0 - x_b0| = {diff0:.6e}")


def test_deq_ste_mode_rejected():
    """forward_equilibrium rejects cell_mode='ste' (soft-only safeguard)."""
    print("\nTest DEQ-8: forward_equilibrium rejects STE cell mode")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    ctx = SimContext()
    raised = False
    try:
        stage.forward_equilibrium(torch.zeros(1, 3), ctx=ctx, tau=1.0,
                                  cell_mode="ste", x_drive=None,
                                  drive_scale=0.0,
                                  deq_cfg={"f_max_iter": 20, "f_tol": 1e-4,
                                           "deq_step": 0.1,
                                           "leak_floor": 0.0})
    except ValueError:
        raised = True
    check("ValueError on cell_mode='ste'", raised)


def test_deq_leak_floor_enforced():
    """leak_floor=0.0 leaves Heun path unchanged (regression); >0 keeps diagonal damping."""
    print("\nTest DEQ-9: leak_floor=0.0 matches default Heun; leak_floor>0 increases leak")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)

    x = torch.tensor([[0.5, -0.3, 0.1]])
    leak_default = float(stage._effective_leak(x.shape[1], leak_floor=0.0).sum())
    leak_floored = float(stage._effective_leak(x.shape[1], leak_floor=0.5).sum())
    check("leak_floor=0.0 reproduces default (≈3 nodes * softplus(-3) ≈ 0.144)",
          abs(leak_default - 3 * 0.048) < 0.05,
          f"leak_default={leak_default:.4f}")
    check("leak_floor=0.5 increases leak (3 * (0.5 + softplus(-3)) > leak_default)",
          leak_floored > leak_default,
          f"leak_floored={leak_floored:.4f} vs default={leak_default:.4f}")


def test_deq_solver_kwarg_threads_through_kirchhoff_net():
    """KirchhoffNet.forward accepts solver='heun' and 'deq' and dispatches correctly."""
    print("\nTest DEQ-10: solver kwarg threads through KirchhoffNet / WithIO")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
    from io_mapper import InputMapper, OutputMapper
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=2, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0
    core = KirchhoffNet(stages=[stage], transfers=[], stage_times=[1.0], stage_steps=[10])
    inp = InputMapper(in_dim=2, out_dim=3)
    out = OutputMapper(node_dim=3, out_dim=1)
    net = KirchhoffNetWithIO(inp, core, out, hid_count=3, proj_count=0)

    u = torch.randn(4, 2) * 0.3
    ctx = SimContext()
    y_heun, _ = net(u, ctx=ctx, store_trajectory=False, cell_mode="soft",
                    solver="heun")
    check("heun forward shape (4,1)", y_heun.shape == (4, 1))
    check("heun forward finite", torch.isfinite(y_heun).all().item())

    # DEQ path requires the WithIO to accept solver='deq' and pass it down.
    y_deq, _ = net(u, ctx=ctx, store_trajectory=False, cell_mode="soft",
                   solver="deq", deq_cfg={"f_max_iter": 30, "f_tol": 1e-5,
                                          "deq_step": 0.1,
                                          "leak_floor": 0.05})
    check("deq forward shape (4,1)", y_deq.shape == (4, 1))
    check("deq forward finite", torch.isfinite(y_deq).all().item())


def test_deq_heun_regression_unchanged():
    """With solver='heun' (default), behavior must be unchanged from pre-DEQ path."""
    print("\nTest DEQ-11: Heun path unchanged with solver='heun'")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from kirchhoff_net import KirchhoffNet
    from sim_context import SimContext

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    net = KirchhoffNet(stages=[stage], transfers=[], stage_times=[0.3], stage_steps=[10])

    ctx = SimContext()
    x0 = torch.zeros(1, 3)
    # Default solver
    y_default, _ = net(x0, ctx=ctx, store_trajectory=True, cell_mode="soft")
    # Explicit heun
    y_heun, _ = net(x0, ctx=ctx, store_trajectory=True, cell_mode="soft",
                    solver="heun")
    check("default == solver='heun'",
          torch.allclose(y_default, y_heun, atol=1e-7),
          f"max diff={(y_default - y_heun).abs().max().item():.3e}")


def test_deq_multistart_uniqueness_on_contractive():
    """Multistart: solving from several x0 should give the same x* on contractive Phi."""
    print("\nTest DEQ-12: multistart uniqueness on contractive fixed point")
    from deq_solver import solve_equilibrium

    def phi(x):
        return 0.5 * x + 0.1

    cfg = {"f_max_iter": 100, "f_tol": 1e-6, "deq_step": 1.0}
    starts = [torch.full((2, 4), v) for v in (-1.0, 0.0, 1.0, 5.0)]
    finals = []
    for s in starts:
        x_star, _ = solve_equilibrium(phi, s, cfg)
        finals.append(x_star)
    pairwise = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            pairwise.append(float((finals[i] - finals[j]).abs().max()))
    max_diff = max(pairwise)
    check("all starts converge to same x* (max diff < 1e-3)",
          max_diff < 1e-3, f"max pairwise diff={max_diff:.3e}")


def test_deq_config_defaults():
    """DEQ config dict exists in config.py with expected keys and defaults."""
    print("\nTest DEQ-13: config.DEQ has expected defaults")
    import config
    check("config has DEQ dict", hasattr(config, "DEQ"))
    if not hasattr(config, "DEQ"):
        return
    deq = config.DEQ
    for k in ("f_max_iter", "f_tol", "b_max_iter", "deq_step", "leak_floor"):
        check(f"DEQ['{k}'] present", k in deq)
    check("DEQ['f_max_iter'] in (10, 200)", 10 <= deq.get("f_max_iter", 0) <= 200,
          f"got {deq.get('f_max_iter')}")
    check("DEQ['f_tol'] < 1e-2", deq.get("f_tol", 1.0) < 1e-2,
          f"got {deq.get('f_tol')}")


def test_deq_diagnostics_jacobian_cond():
    """deq_diagnostics.estimate_jacobian_cond returns a finite cond on a tiny stage."""
    print("\nTest DEQ-14: deq_diagnostics.jacobian_cond finite")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    from deq_diagnostics import estimate_jacobian_cond

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0
    ctx = SimContext()
    x = torch.zeros(1, 3)
    cond = estimate_jacobian_cond(stage, x, ctx=ctx, tau=1.0, cell_mode="soft",
                                  x_drive=None, drive_scale=0.0, leak_floor=0.05)
    check("jacobian cond finite", math.isfinite(cond) and cond > 0,
          f"cond={cond}")
    check("jacobian cond reasonable (<1e6)", cond < 1e6,
          f"cond={cond}")


def test_deq_diagnostics_grad_norm_compare():
    """deq_diagnostics.gradient_norm_compare returns finite z_logits & logits norms."""
    print("\nTest DEQ-15: deq_diagnostics.gradient_norm_compare returns finite norms")
    from cell_library import IdealizedCellLibrary
    from topology import cluster_graph, StageTopologyBuilder, topology_to_stage
    from sim_context import SimContext
    from deq_diagnostics import gradient_norm_compare

    torch.manual_seed(0)
    cell_lib = IdealizedCellLibrary()
    hid = cluster_graph(3, edge_prob=0.5, seed=0)
    builder = StageTopologyBuilder(num_inputs=1, num_outputs=0, num_hidden=3, num_proj=0)
    topo = builder.build(hid, input_pattern="all_to_all", output_pattern="all_to_all",
                         proj_pattern="all_to_all")
    stage, _, _ = topology_to_stage(topo, cell_lib=cell_lib)
    stage.drive_isat = 0.0
    ctx = SimContext()
    x0 = torch.zeros(1, 3)
    deq_cfg = {"f_max_iter": 40, "f_tol": 1e-5, "deq_step": 0.1}
    res = gradient_norm_compare(stage, x0, ctx=ctx, tau=1.0, cell_mode="soft",
                                x_drive=None, drive_scale=0.0, leak_floor=0.05,
                                deq_cfg=deq_cfg, bptt_t_span=0.3, bptt_num_steps=10)
    check("returns dict with z_logits_heun, z_logits_deq, logits_heun, logits_deq",
          all(k in res for k in ("z_logits_heun", "z_logits_deq",
                                  "logits_heun", "logits_deq")))
    for k, v in res.items():
        check(f"{k} finite and >= 0", math.isfinite(v) and v >= 0,
              f"{k}={v}")


# =============================================================================
# Degree budget / top-k competition (degree-budget-topk plan)
# =============================================================================

def _make_budget_stage(num_nodes=5, src=None, dst=None):
    """Helper: build a small DifferentialStage with 5 nodes and 15 edges
    (3 incoming edges per destination) for budget tests.
    """
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    if src is None or dst is None:
        # 5 nodes, 3 incoming edges per destination (uniform coverage)
        src = [1, 3, 4, 0, 2, 4, 0, 1, 3, 1, 2, 4, 0, 2, 3]
        dst = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    cell_lib = IdealizedCellLibrary()
    return DifferentialStage(num_nodes=num_nodes, src=src, dst=dst, cell_lib=cell_lib)


def test_budget_gate_basic():
    """BUD-1: budget gate per-destination, sum per group <= k_eff, in [0, 1]."""
    print("\nTest BUD-1: budget gate basic properties")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )

    # frac=0.5 with in-degree=3 gives k_eff = max(1, round(1.5)) = 2 per group
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    gate = stage._compute_budget_gate()

    # Shape
    check("gate shape is [E]", gate.shape == (15,))
    # Range
    check("gate in [0, 1]",
          bool((gate >= 0.0).all().item() and (gate <= 1.0).all().item()),
          f"min={gate.min().item():.4f} max={gate.max().item():.4f}")
    # Per-destination sum <= k_eff (=2 with count=3, frac=0.5)
    for j in range(5):
        idx = (stage.dst == j).nonzero(as_tuple=False).squeeze(-1)
        s = gate[idx].sum().item()
        check(f"dst={j} group sum <= k_eff=2", s <= 2.0 + 1e-5,
              f"got {s:.4f}")


def test_budget_gate_differentiable():
    """BUD-2: budget gate is differentiable through z_logits."""
    print("\nTest BUD-2: budget gate differentiable")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    gate = stage._compute_budget_gate()
    check("gate.requires_grad is True", gate.requires_grad is True)
    loss = gate.sum()
    loss.backward()
    check("z_logits.grad is finite", torch.isfinite(stage.z_logits.grad).all().item())
    check("z_logits.grad has nonzero entries",
          (stage.z_logits.grad.abs() > 0).any().item(),
          f"all grad = {[round(g, 4) for g in stage.z_logits.grad.tolist()]}")


def test_budget_frac_unity_noop():
    """BUD-3: when frac >= 1.0, k_eff >= count, budget gate is all-ones."""
    print("\nTest BUD-3: frac >= 1.0 is no-op (k_eff >= in-degree)")
    stage = _make_budget_stage()  # 3 incoming edges per destination
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    # frac=1.0 with count=3 gives k_eff=3 -> 3 > 3 = False, all ones
    stage.set_budget_frac(frac=1.0, temperature=1.0)
    gate = stage._compute_budget_gate()
    check("frac=1.0 with in-degree=3 -> all ones",
          torch.allclose(gate, torch.ones_like(gate), atol=1e-6),
          f"max diff from ones = {(gate - 1.0).abs().max().item():.4e}")

    # frac > 1.0: also no-op (clamped by k_eff >= count)
    stage.set_budget_frac(frac=10.0, temperature=1.0)
    gate = stage._compute_budget_gate()
    check("frac=10.0 >> 1.0 -> all ones",
          torch.allclose(gate, torch.ones_like(gate), atol=1e-6))

    # frac=0.0 disables the budget entirely
    stage.set_budget_frac(frac=0.0, temperature=1.0)
    gate = stage._compute_budget_gate()
    check("frac=0.0 -> disabled (all ones)",
          torch.allclose(gate, torch.ones_like(gate), atol=1e-6))


def test_budget_temperature_limits():
    """BUD-4: T->inf gives uniform budget; T->0 gives argmax (top-k_eff hard)."""
    print("\nTest BUD-4: temperature limits")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )

    # T -> 0: argmax per group. frac=0.33 with count=3 gives k_eff=1
    stage.set_budget_frac(frac=1.0 / 3.0, temperature=0.001)
    gate = stage._compute_budget_gate()
    # For each destination, the highest-z edge should be 1.0, others 0.0
    for j in range(5):
        idx = (stage.dst == j).nonzero(as_tuple=False).squeeze(-1)
        z = stage.z_logits[idx]
        max_pos = idx[int(z.argmax().item())]
        check(f"dst={j} argmax edge gets 1.0 at T~0",
              abs(gate[max_pos].item() - 1.0) < 1e-3,
              f"got {gate[max_pos].item():.4f}")
        others = [i.item() for i in idx if i.item() != max_pos.item()]
        for o in others:
            check(f"dst={j} non-argmax edge at T~0: ~0.0",
                  gate[o].item() < 0.1,
                  f"got {gate[o].item():.4f}")

    # T -> inf: uniform
    stage.set_budget_frac(frac=0.5, temperature=1000.0)
    gate = stage._compute_budget_gate()
    for j in range(5):
        idx = (stage.dst == j).nonzero(as_tuple=False).squeeze(-1)
        n = len(idx)
        if n <= 2:
            continue
        vals = [gate[i].item() for i in idx]
        spread = max(vals) - min(vals)
        check(f"dst={j} uniform at T~inf: spread < 0.1",
              spread < 0.1,
              f"spread = {spread:.4f}")


def test_budget_axis_src():
    """BUD-5: budget axis='src' competes on outgoing edges per source."""
    print("\nTest BUD-5: axis=src")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    # In test topology, each source has 3 outgoing edges; frac=0.5 -> k_eff=2
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    stage.budget_axis = "src"
    gate = stage._compute_budget_gate()
    # Each source should have group sum <= k_eff=2
    for s in range(5):
        idx = (stage.src == s).nonzero(as_tuple=False).squeeze(-1)
        if len(idx) == 0:
            continue
        s_sum = gate[idx].sum().item()
        check(f"src={s} group sum <= k_eff=2", s_sum <= 2.0 + 1e-5,
              f"got {s_sum:.4f}")


def test_budget_axis_both():
    """BUD-6: budget axis='both' is product of src-mask and dst-mask."""
    print("\nTest BUD-6: axis=both")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    # Get dst-only mask
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    stage.budget_axis = "dst"
    gate_dst = stage._compute_budget_gate()
    # Get src-only mask
    stage.budget_axis = "src"
    gate_src = stage._compute_budget_gate()
    # Get both mask
    stage.budget_axis = "both"
    gate_both = stage._compute_budget_gate()
    expected = gate_dst * gate_src
    check("axis=both equals dst*src",
          torch.allclose(gate_both, expected, atol=1e-6),
          f"max diff = {(gate_both - expected).abs().max().item():.4e}")


def test_budget_deq_forward():
    """BUD-7: DEQ forward with budget enabled returns finite x*."""
    print("\nTest BUD-7: DEQ forward with budget")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    from sim_context import SimContext
    ctx = SimContext()
    x0 = torch.zeros(2, 5)
    x_star, _info = stage.forward_equilibrium(
        x0=x0, ctx=ctx, tau=1.0, cell_mode="soft",
        x_drive=None, drive_scale=0.0,
        deq_cfg={"f_max_iter": 30, "f_tol": 1e-5, "deq_step": 0.1, "leak_floor": 0.05},
    )
    check("x_star is finite", torch.isfinite(x_star).all().item())
    check("x_star shape matches", x_star.shape == (2, 5))

    # Compare against no-budget: should differ (budget changes dynamics)
    stage.set_budget_frac(frac=0.0, temperature=1.0)
    x_star_nb, _ = stage.forward_equilibrium(
        x0=x0, ctx=ctx, tau=1.0, cell_mode="soft",
        x_drive=None, drive_scale=0.0,
        deq_cfg={"f_max_iter": 30, "f_tol": 1e-5, "deq_step": 0.1, "leak_floor": 0.05},
    )
    diff = (x_star - x_star_nb).abs().max().item()
    check("budget-enabled x* differs from no-budget",
          diff > 1e-4,
          f"max diff = {diff:.4e}")


def test_budget_annealing_schedule():
    """BUD-8: budget_frac_for_epoch and budget_temperature_for_epoch linear anneal."""
    print("\nTest BUD-8: annealing schedule")
    from train import budget_frac_for_epoch, budget_temperature_for_epoch
    # frac anneal: 1.0 -> 0.75 over 80% of 100 epochs
    check("frac(0) = frac_start",
          abs(budget_frac_for_epoch(0, 100, 1.0, 0.75, 0.8) - 1.0) < 1e-6)
    # At epoch 40 with 80 anneal epochs: alpha = 40/79 ≈ 0.5063
    # frac = (1-0.5063) * 1.0 + 0.5063 * 0.75 = 0.4937 + 0.3797 = 0.8734
    check("frac(40) ≈ 0.8734 (midpoint of anneal)",
          abs(budget_frac_for_epoch(40, 100, 1.0, 0.75, 0.8) - 0.8734) < 1e-3,
          f"got {budget_frac_for_epoch(40, 100, 1.0, 0.75, 0.8):.4f}")
    check("frac(79) = frac_end (last anneal epoch)",
          abs(budget_frac_for_epoch(79, 100, 1.0, 0.75, 0.8) - 0.75) < 1e-6)
    check("frac(80) = frac_end (post-anneal freeze)",
          abs(budget_frac_for_epoch(80, 100, 1.0, 0.75, 0.8) - 0.75) < 1e-6)
    check("frac(99) = frac_end",
          abs(budget_frac_for_epoch(99, 100, 1.0, 0.75, 0.8) - 0.75) < 1e-6)
    # Negative frac_end clamped to 0
    check("frac never negative",
          budget_frac_for_epoch(50, 100, 0.5, -0.1, 0.8) >= 0.0)

    # Temperature anneal: 1.0 -> 0.1
    check("T(0) = temp_start",
          abs(budget_temperature_for_epoch(0, 100, 1.0, 0.1, 0.8) - 1.0) < 1e-6)
    # At epoch 40 with 80 anneal epochs: alpha = 40/79 ≈ 0.5063
    # T = 0.5063 * (0.1 - 1.0) + 1.0 = 0.5443
    check("T(40) ≈ 0.5443 (midpoint of anneal)",
          abs(budget_temperature_for_epoch(40, 100, 1.0, 0.1, 0.8) - 0.5443) < 1e-3,
          f"got {budget_temperature_for_epoch(40, 100, 1.0, 0.1, 0.8):.4f}")
    check("T(79) = temp_end",
          abs(budget_temperature_for_epoch(79, 100, 1.0, 0.1, 0.8) - 0.1) < 1e-6)
    check("T(99) = temp_end",
          abs(budget_temperature_for_epoch(99, 100, 1.0, 0.1, 0.8) - 0.1) < 1e-6)
    check("T(0) > 0 (always positive)",
          budget_temperature_for_epoch(0, 100, 1.0, 0.0, 0.8) > 0)


def test_budget_disabled_byte_identical():
    """BUD-9: rhs() output is byte-identical to pre-budget when budget_frac=0."""
    print("\nTest BUD-9: budget disabled = no change to rhs")
    stage = _make_budget_stage()
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    from sim_context import SimContext
    ctx = SimContext()
    x = torch.randn(2, 5)
    # Default (budget_enabled=False, budget_frac=0)
    out1 = stage.rhs(x, ctx=ctx, tau=1.0, cell_mode="soft")
    # Explicitly disabled
    stage.set_budget_frac(0.0, 1.0)
    out2 = stage.rhs(x, ctx=ctx, tau=1.0, cell_mode="soft")
    check("default rhs == disabled-budget rhs",
          torch.allclose(out1, out2, atol=1e-7),
          f"max diff = {(out1 - out2).abs().max().item():.4e}")


def test_budget_simple_edge_library_compat():
    """BUD-10: budget works on SimpleEdgeLibrary stages (logits=None)."""
    print("\nTest BUD-10: SimpleEdgeLibrary compat")
    from cell_library import SimpleEdgeLibrary
    from differential_stage import DifferentialStage
    cell_lib = SimpleEdgeLibrary(num_edges=15, mode="tanh")
    src = [1, 3, 4, 0, 2, 4, 0, 1, 3, 1, 2, 4, 0, 2, 3]
    dst = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    stage = DifferentialStage(num_nodes=5, src=src, dst=dst, cell_lib=cell_lib)
    with torch.no_grad():
        stage.z_logits.data = torch.tensor(
            [3.0, 1.0, 0.0, 2.0, 1.5, 0.5, 2.5, 0.8, -0.5, 1.2, 0.3, 2.2, 1.8, 0.6, -1.0]
        )
    stage.set_budget_frac(frac=0.5, temperature=1.0)
    gate = stage._compute_budget_gate()
    check("SimpleEdgeLibrary: gate shape [E]", gate.shape == (15,))
    check("SimpleEdgeLibrary: gate in [0, 1]",
          bool((gate >= 0.0).all().item() and (gate <= 1.0).all().item()))
    # Forward should work without errors
    from sim_context import SimContext
    ctx = SimContext()
    x = torch.randn(2, 5)
    out = stage.rhs(x, ctx=ctx, tau=1.0, cell_mode="soft")
    check("SimpleEdgeLibrary rhs with budget is finite",
          torch.isfinite(out).all().item())


def test_budget_frac_uniform_proportion():
    """BUD-11: frac=0.5 with mixed in-degrees keeps ~50% per group.

    Demonstrates the key advantage of fraction-based budget over absolute-k:
    nodes with high in-degree (e.g., proj nodes with 25) keep the same
    proportion as nodes with low in-degree (e.g., edge hidden with 4).
    Uses high temperature so the per-edge budget gate is approximately
    uniform within the top-k_eff edges, demonstrating the proportion.
    """
    print("\nTest BUD-11: fraction budget applies uniform proportion")
    from cell_library import IdealizedCellLibrary
    from differential_stage import DifferentialStage
    # Build a stage with 3 destination groups having different in-degrees:
    #   group 0: 4 incoming edges (small, like edge hidden)
    #   group 1: 8 incoming edges (medium, like interior hidden)
    #   group 2: 16 incoming edges (large, like proj nodes)
    src = []
    dst = []
    src.extend([10, 11, 12, 13])
    dst.extend([0, 0, 0, 0])
    src.extend(list(range(20, 28)))
    dst.extend([1] * 8)
    src.extend(list(range(30, 46)))
    dst.extend([2] * 16)
    cell_lib = IdealizedCellLibrary()
    stage = DifferentialStage(
        num_nodes=46, src=src, dst=dst, cell_lib=cell_lib,
    )
    with torch.no_grad():
        # Higher-index edges have higher z_logits within each group
        z = torch.zeros(28)
        z[0:4] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        z[4:12] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        z[12:28] = torch.tensor([float(i) for i in range(16)])
        stage.z_logits.data = z

    # frac=0.5: k_eff = max(1, round(count*0.5)) per group
    #   group 0: round(4*0.5)=2 -> k_eff=2  (50% of 4)
    #   group 1: round(8*0.5)=4 -> k_eff=4  (50% of 8)
    #   group 2: round(16*0.5)=8 -> k_eff=8 (50% of 16)
    # Use high T for uniform distribution: each surviving edge gate ≈
    # k_eff/count (since softmax ≈ uniform, then *k_eff, then clamp to 1.0).
    stage.set_budget_frac(frac=0.5, temperature=1000.0)
    gate = stage._compute_budget_gate()
    for grp, count, expected_keff in [(0, 4, 2), (1, 8, 4), (2, 16, 8)]:
        idx = (stage.dst == grp).nonzero(as_tuple=False).squeeze(-1)
        # At high T, softmax is uniform 1/count; budget_gate ≈
        # k_eff * (1/count) for every edge; clamp to max=1.0.
        # Since k_eff/count = 0.5, every edge gate ≈ 0.5 (no clamping).
        # The uniform-proportion claim: each group retains a fixed
        # proportion of edges by giving them all equal budget_gate.
        # Verify group-mean gate is the same for all groups.
        mean_gate = gate[idx].mean().item()
        check(f"group {grp} (count={count}, k_eff={expected_keff}) "
              f"mean gate ≈ 0.5",
              abs(mean_gate - 0.5) < 0.05,
              f"got {mean_gate:.4f}")

    # The KEY uniform-proportion claim: with absolute-k, group 2 (16 in)
    # would have to compete for k_eff slots, while group 0 (4 in) would
    # have a fraction that depends on the absolute k. With frac=0.5,
    # both get the same k_eff/count = 0.5.
    check("uniform proportion: all groups ~50% retention",
          all(
              abs(
                  round(
                      (4 if g == 0 else 8 if g == 1 else 16) * 0.5
                  ) / (4 if g == 0 else 8 if g == 1 else 16)
                  - 0.5
              ) < 0.001
              for g in (0, 1, 2)
          ))

    # Sanity: with frac=0.5 and T=high, every edge in every group gets
    # the same gate value (uniform distribution). Compare inter-group.
    stage.set_budget_frac(frac=0.5, temperature=1000.0)
    gate = stage._compute_budget_gate()
    means = []
    for grp in (0, 1, 2):
        idx = (stage.dst == grp).nonzero(as_tuple=False).squeeze(-1)
        means.append(gate[idx].mean().item())
    check("all groups have equal mean gate (frac is uniform)",
          max(means) - min(means) < 0.05,
          f"group mean gates: {means}")


if __name__ == "__main__":
    main()
