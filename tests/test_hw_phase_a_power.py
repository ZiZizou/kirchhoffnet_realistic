"""hw_power_penalty is wired into the Phase-A loss (Phase 3b W5 step 3).

``ctle_dagger_common.run_phase_a_training`` adds
``hw_power_penalty(student.net, hw, u=u_batch)`` (= hw_power_lambda * P_avg /
P_ref) when the student's KNet carries an HWConfig with hw_power_lambda > 0.
Off (no HWConfig, or lambda = 0) the penalty is never evaluated, so the
training is bit-identical to the pre-hw path.

One epoch on a 48/32/32-row subset of the real canonical Phase-A dataset with
the real ZIG surrogate and a small 2-stage KNet student (~10 s on CPU).
"""

import hashlib
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import eval_paper_checkpoint as ev  # noqa: E402
from checkpoints import PARAM_LOG_BOUNDS  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.constraints import HWConstraints  # noqa: E402

HAVE_DATA = os.path.isfile(ev.DEFAULT_CANONICAL) and os.path.isdir(ev.DEFAULT_ZIG_DIR)
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="canonical npz / ZIG surrogate not available")


class TinyStudent(nn.Module):
    """A 2-stage x 8-hidden version of the paper student (checkpoints.PaperStudent) with ``hw``."""

    def __init__(self, hw=None):
        super().__init__()
        from cell_library import make_cell_library
        from hw import hw_or_none
        from topology import build_net_from_config
        self.hw = hw_or_none(hw)
        self.input_rail = 4.0
        self.register_buffer("input_log_min", torch.zeros(4))
        self.register_buffer("input_log_max", torch.ones(4))
        self.log_lo = nn.Parameter(torch.tensor([b[0] for b in PARAM_LOG_BOUNDS], dtype=torch.float32),
                                   requires_grad=False)
        self.log_hi = nn.Parameter(torch.tensor([b[1] for b in PARAM_LOG_BOUNDS], dtype=torch.float32),
                                   requires_grad=False)
        stage = {"num_inputs": 4, "num_hidden": 8, "num_proj": 0, "num_outputs": 0,
                 "hidden_family": "small_world",
                 "hidden_kwargs": {"k": 2, "p": 0.2, "seed": 1, "bidirectional": False},
                 "input_pattern": "all_to_all", "output_pattern": "all_to_all",
                 "proj_pattern": "all_to_all", "edge_repeats": 2, "t_span": 1.0, "num_steps": 4}
        cfg = {"stages": [dict(stage) for _ in range(2)], "out_dim": 7,
               "write_mode": "sparse_proj", "read_mode": "dense", "use_robust_input": False}
        self.net = build_net_from_config(
            cfg, cell_lib=make_cell_library("tanh_free", hw=self.hw),
            boundary_fan_out={0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}, x_max=4.0,
            leak_mode="non-programmable", freeze_read=True,
            interstage_activation="residual-relu-tanh", enable_temporal_readout=True,
            readout_mode="shared_sense", readout_senses_per_node=1,
            vca_enabled=True, vca_rank=2, vca_core_enabled=True, vca_gate_shunt=False,
            vca_separate_core_bus=True, vca_bias=False, hw=self.hw)
        self._clip_elements = 0
        self._input_elements = 0

    def scale_input(self, x):
        lo, hi = self.input_log_min.to(x), self.input_log_max.to(x)
        u = 2.0 * (torch.log10(x.clamp(min=1e-12)) - lo) / (hi - lo).clamp(min=1e-8) - 1.0
        with torch.no_grad():
            self._clip_elements += int((u.abs() >= self.input_rail).sum().item())
            self._input_elements += int(u.numel())
        return u.clamp(min=-self.input_rail, max=self.input_rail)

    def forward(self, x):
        out = self.net(self.scale_input(x), store_trajectory=False)
        return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def assets():
    return load_assets()


def load_assets():
    from ctle_dagger_common import load_canonical_flow_dataset
    can = dict(load_canonical_flow_dataset(ev.DEFAULT_CANONICAL))
    for split, n in (("train", 48), ("val", 32), ("test", 32)):
        can[f"{split}_idx"] = np.asarray(can[f"{split}_idx"])[:n]
    zig, sx, sy, cfg = ev.load_zig(ev.DEFAULT_ZIG_DIR)
    return can, zig, sx, sy, cfg


def _train(assets, tmp_path, hw, monkeypatch=None):
    from ctle_dagger_common import run_phase_a_training
    can, zig, sx, sy, cfg = assets
    torch.manual_seed(0)
    student = TinyStudent(hw)
    with torch.no_grad():
        student.input_log_min.copy_(torch.as_tensor(np.asarray(can["input_log_min"], dtype=np.float32)))
        student.input_log_max.copy_(torch.as_tensor(np.asarray(can["input_log_max"], dtype=np.float32)))
    ctx = {"DEVICE": torch.device("cpu"), "scaler_X": sx, "scaler_y_p": sy, "zig_model": zig,
           "eye_scale_h": cfg["eye_scale_h"], "eye_scale_w": cfg["eye_scale_w"],
           "eye_scale_j": cfg["eye_scale_j"], "canonical_dataset": can, "epochs": 1,
           "batch_size": 16, "lr": 1e-2, "weight_decay": 0.0, "output_dir": str(tmp_path),
           "earlystop_patience": 5}
    torch.manual_seed(1)
    res = run_phase_a_training(student, ctx, model_name="tiny")
    return student, res


def _digest(student):
    h = hashlib.sha256()
    for k, v in sorted(student.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _spy(monkeypatch):
    import hw.constraints as C
    calls = []
    real = C.hw_power_penalty

    def spy(net, hw=None, u=None, **kw):
        out = real(net, hw, u=u, **kw)
        calls.append((u is not None and tuple(u.shape), float(out.detach())))
        return out
    monkeypatch.setattr(C, "hw_power_penalty", spy)
    return calls


def test_off_path_never_evaluates_the_penalty_and_is_deterministic(assets, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    s0, r0 = _train(assets, tmp_path / "a", None)
    assert calls == []
    # lambda = 0 with another flag on: the hw path is active but the penalty is not
    s_l0, _ = _train(assets, tmp_path / "b",
                     HWConfig(constraints=HWConstraints(hw_power_lambda=0.0, hw_theta_bound=True)))
    assert calls == []
    s1, r1 = _train(assets, tmp_path / "c", None)
    assert _digest(s0) == _digest(s1)
    assert r0["history"]["train_loss"] == r1["history"]["train_loss"]


def test_power_penalty_enters_the_phase_a_loss(assets, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    hw_on = HWConfig(constraints=HWConstraints(hw_power_lambda=1.0))
    s_on, r_on = _train(assets, tmp_path / "on", hw_on)
    assert len(calls) == 3                                  # 48 rows / batch 16
    assert all(shape == (16, 4) for shape, _ in calls)      # u of the batch was passed
    assert all(np.isfinite(v) and v > 0 for _, v in calls)
    # the student's input-rail clipping counters only count the forward passes
    assert s_on._input_elements > 0 and s_on._input_elements % 4 == 0
    s_off, r_off = _train(assets, tmp_path / "off", None)
    assert r_on["history"]["train_loss"][0] > r_off["history"]["train_loss"][0]
    assert _digest(s_on) != _digest(s_off)
    # the clipping counters are the same as without the penalty (no extra scale_input calls)
    assert s_on._input_elements == s_off._input_elements


def test_hw_package_reexports_the_loss_and_diagnostic_entry_points():
    import hw
    import hw.constraints as C
    for name in ("hw_power_penalty", "net_hw", "stiffness_report", "stage_small_signal_lambda"):
        assert getattr(hw, name) is getattr(C, name) and name in hw.__all__
