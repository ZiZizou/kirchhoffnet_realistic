"""Load the trained PAPER HR-KNet checkpoint (trial_0011) at the current HEAD.

The paper model is the ``LocalKirchhoffStudentWrapper`` of
``dagger-nuance-distillation-kirchhoffnet.py`` (git 5c6a4f4) built with the
flags in ``assets/baselines/paper_trial_0011/run_command.txt``.  That script
cannot be imported (it runs its whole training prologue at import), so
:class:`PaperStudent` mirrors the wrapper's state layout and forward:

    input_log_min / input_log_max   buffers  (4,)
    log_lo / log_hi                 frozen parameters (7,)
    net.*                           the fabric (``topology.build_net_from_config``)

``net`` contains the input/output mappers (``net.output_mapper`` is the
OutputAffine readout) and the GLN rails, so loading the whole wrapper state
loads those too.

The refactor surgery removed state that the checkpoint still carries:
``u_logits`` (node gates, never received a gradient) and the empty int64
buffers ``_drive_idx`` / ``_drive_mask`` / ``ref_dst``.  :func:`load_paper_checkpoint`
drops exactly those keys and loads everything else with ``strict=True``.
"""

from __future__ import annotations

import fnmatch
import json
import os

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(REPO, "assets", "baselines", "paper_trial_0011", "phase_a_best.pt")

# State-dict keys present in the checkpoint but removed by the surgery.
DROPPED_KEY_PATTERNS = (
    "net.core.stages.*.u_logits",
    "net.core.stages.*._drive_idx",
    "net.core.stages.*.ref_dst",
    "net.core.transfers.*._drive_mask",
    "net.post_readout_transfer._drive_mask",
)

# Phase-A paper run (run_command.txt): 7 stages x 11 hidden, small-world k=2 p=0.2,
# VCA rank 3 + core, shared-x2 readout, learnable clip sharpness, GLN rails.
PAPER_CONFIG = {
    "num_stages": 7,
    "num_hidden": 11,
    "small_world_k": 2,
    "small_world_p": 0.2,
    "small_world_seed": 1,
    "edge_repeats": 2,
    "t_span_total": 4.660583,
    "num_steps_per_stage": 10,          # round(SOLVER num_steps 70 / 7)
    "num_targets": 7,
    "input_rail": 4.0,
    "x_max": 4.0,
    "gm_max": 4.241970,
    "isat_max": 28.68855,
    "boundary_fan_out": {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]},
    "build": {
        "leak_mode": "non-programmable",
        "freeze_read": True,
        "interstage_activation": "residual-relu-tanh",
        "enable_temporal_readout": True,
        "readout_mode": "shared_sense",          # --kn-readout shared-x2
        "readout_senses_per_node": 2,
        "learnable_clip_sharpness": True,
        "gln_rails": True, "gln_B": 4, "gln_rank": 2, "gln_alpha_init": 1.0,
        "gln_families": "boundary,readout",
        "vca_enabled": True, "vca_rank": 3, "vca_core_enabled": True,
        "vca_gate_shunt": False, "vca_separate_core_bus": True, "vca_bias": False,
    },
}

# Physical log10 bounds of the 7 ZIG inputs (ctle_dagger_common.PARAM_LOG_BOUNDS order).
PARAM_LOG_BOUNDS = (
    (-6.0, 1.0), (np.log10(5e-4), np.log10(2.5)), (-12.0, np.log10(3.0)),
    (1.0, np.log10(1500)), (-15.0, -9.0), (1.0, np.log10(1500)), (np.log10(0.6), np.log10(1.2)),
)


class PaperStudent(nn.Module):
    """Same state layout and forward as ``LocalKirchhoffStudentWrapper``."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        from cell_library import make_cell_library
        from topology import build_net_from_config

        c = dict(PAPER_CONFIG if config is None else config)
        self.input_rail = float(c["input_rail"])
        self.register_buffer("input_log_min", torch.zeros(4))
        self.register_buffer("input_log_max", torch.ones(4))
        self.log_lo = nn.Parameter(torch.tensor([b[0] for b in PARAM_LOG_BOUNDS], dtype=torch.float32),
                                   requires_grad=False)
        self.log_hi = nn.Parameter(torch.tensor([b[1] for b in PARAM_LOG_BOUNDS], dtype=torch.float32),
                                   requires_grad=False)
        stage = {
            "num_inputs": 4, "num_hidden": int(c["num_hidden"]), "num_proj": 0, "num_outputs": 0,
            "hidden_family": "small_world",
            "hidden_kwargs": {"k": int(c["small_world_k"]), "p": float(c["small_world_p"]),
                              "seed": int(c["small_world_seed"]), "bidirectional": False},
            "input_pattern": "all_to_all", "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all", "edge_repeats": int(c["edge_repeats"]),
            "t_span": float(c["t_span_total"]) / int(c["num_stages"]),
            "num_steps": int(c["num_steps_per_stage"]),
        }
        cfg = {"stages": [dict(stage) for _ in range(int(c["num_stages"]))],
               "out_dim": int(c["num_targets"]), "write_mode": "sparse_proj",
               "read_mode": "dense", "use_robust_input": False}
        cell_lib = make_cell_library("tanh_free", gm_max=c["gm_max"], isat_max=c["isat_max"])
        fan = {int(k): list(v) for k, v in c["boundary_fan_out"].items()}
        self.net = build_net_from_config(cfg, cell_lib=cell_lib, boundary_fan_out=fan,
                                         x_max=float(c["x_max"]), **c["build"])

    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.input_log_min.to(x), self.input_log_max.to(x)
        u = 2.0 * (torch.log10(x.clamp(min=1e-12)) - lo) / (hi - lo).clamp(min=1e-8) - 1.0
        return u.clamp(min=-self.input_rail, max=self.input_rail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(self.scale_input(x), store_trajectory=False)
        return out[0] if isinstance(out, tuple) else out


def _is_dropped(key: str) -> bool:
    return any(fnmatch.fnmatch(key, p) for p in DROPPED_KEY_PATTERNS)


def load_paper_checkpoint(path: str | os.PathLike = DEFAULT_CKPT, device="cpu",
                          dtype=torch.float32, config: dict | None = None) -> PaperStudent:
    """Rebuild the paper model at HEAD and load ``path`` (phase_a_best.pt).

    Drops exactly the surgery-removed keys (``DROPPED_KEY_PATTERNS``), asserts
    that they are the only difference, and loads the rest with ``strict=True``.
    Returns the model in eval mode.
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    dropped = sorted(k for k in sd if _is_dropped(k))
    kept = {k: v for k, v in sd.items() if not _is_dropped(k)}

    model = PaperStudent(config)
    own = set(model.state_dict().keys())
    missing, unexpected = sorted(own - set(kept)), sorted(set(kept) - own)
    assert not missing and not unexpected, (
        f"checkpoint/model mismatch beyond the surgery-removed keys: "
        f"missing={missing[:10]} unexpected={unexpected[:10]}")
    assert all(k.endswith(("u_logits", "_drive_idx", "ref_dst", "_drive_mask")) for k in dropped)
    result = model.load_state_dict(kept, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.dropped_keys = dropped
    return model.to(device=device, dtype=dtype).eval()


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = load_paper_checkpoint()
    print(f"loaded; dropped {len(m.dropped_keys)} keys; trainable params = {trainable_param_count(m)}")
