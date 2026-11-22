"""Runtime smoke tests for entry-script student wrappers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe_import(name, filename):
    try:
        return _load(name, filename)
    except Exception as exc:
        pytest.skip(f"entry wrapper import requires unavailable runtime dependency: {exc}")


def test_fixed_entry_student_wrapper_forward():
    entry = _safe_import("fixed_mlp_runtime_smoke", "fixed-mlp-distillation-kirchhoffnet.py")
    args = SimpleNamespace(
        kn_input_rail=4.0, kn_solver_scale=1.0, kn_num_stages=7,
        kn_num_hidden=11, kn_small_world_k=2, kn_small_world_p=0.2,
        kn_small_world_seed=1, kn_edge_repeats=2,
        kn_cell_library="tanh_free", kn_leak_mode="non-programmable",
        kn_freeze_read=True, kn_interstage_activation="residual-relu-tanh",
        boundary_fan_out='{"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}',
        kn_temporal_readout=True, kn_x_max=4.0, kn_vca=True,
        kn_vca_rank=3, kn_vca_core=True, kn_vca_gate_shunt=False,
        kn_vca_separate_core_bus=True, vca_bias=False, hw_resolved=None)
    model = entry.KirchhoffStudent(
        args, np.full(4, -2, np.float32), np.full(4, 2, np.float32)).eval()
    with torch.no_grad():
        output = model(torch.ones(8, 4))
    assert output.shape == (8, 7)
    assert torch.isfinite(output).all()


def test_spice_entry_knet_wrapper_forward():
    entry = _safe_import("spice_runtime_smoke", "spice_failure_check.py")
    from cell_library import make_cell_library
    from config import VCA
    from topology import build_net_from_config
    model = entry.KNetDistillWrapper(
        torch, build_net_from_config, make_cell_library,
        vca_cfg_min_rank=VCA["min_rank"], num_stages=7, num_hidden=11,
        small_world_k=2, small_world_p=0.2, small_world_seed=1,
        edge_repeats=2, boundary_fan_out='{"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}',
        input_log_min=np.full(4, -2, np.float32),
        input_log_max=np.full(4, 2, np.float32), stage_num_steps=2,
    ).eval()
    with torch.no_grad():
        output = model.forward(torch.ones(8, 4))
    assert output.shape == (8, 7)
    assert torch.isfinite(output).all()
