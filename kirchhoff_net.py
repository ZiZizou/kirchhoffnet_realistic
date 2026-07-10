"""KirchhoffNet: multi-stage ODE core and KirchhoffNetWithIO pipeline.

KirchhoffNet sequences DifferentialStage modules and StageTransfer modules.
KirchhoffNetWithIO wraps the core with an InputMapper and OutputMapper
that implement the write/evolve/read phases.

Honest I/O (R1): the write/evolve/read pipeline enforces a strict split
between hidden and projection nodes:
  - InputMapper writes ONLY to the hidden-node positions of the
    differential state vector.
  - Projection nodes start at zero (and remain zero until evolution moves
    them).
  - OutputMapper reads ONLY from projection-node positions; if a stage
    has no projection nodes, it falls back to hidden-node positions
    (with a warning) so backward compatibility is preserved.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from config import DRIVE
from differential_stage import DifferentialStage
from stage_transfer import StageTransfer


__all__ = ["KirchhoffNet", "KirchhoffNetWithIO", "format_parameter_breakdown"]


def format_parameter_breakdown(breakdown: dict) -> str:
    """Render a KirchhoffNetWithIO.parameter_breakdown() dict as aligned text.

    Output example (single-stage friedman2):

        input_mapper:           20
        output_mapper:        1169
        drive_mappers:           8
        stage_0:
          cell_lib:            320
          z_logits:             64
          u_logits:             16
          raw_leak:             16
          raw_drive_g:           4
          ── stage total:      420
        ───────────────────────
        total:               1605
    """
    groups = breakdown.get("groups", {})
    per_stage = breakdown.get("per_stage", {})
    total = breakdown.get("total", 0)
    label_w = max((len(k) for k in groups), default=0)
    label_w = max(label_w, 10)
    stage_label_w = max(
        (len(k) for k in per_stage), default=0,
    )
    stage_label_w = max(stage_label_w, 8)
    width = max(label_w + 18, stage_label_w + 18, 30)
    lines: list[str] = []
    for name in ("input_mapper", "output_mapper", "drive_mappers"):
        lines.append(f"  {name:<{label_w}}: {groups.get(name, 0):>{width - label_w - 4}}")
    for stage_key in sorted(per_stage.keys()):
        lines.append(f"  {stage_key}:")
        bucket = per_stage[stage_key]
        for sub_name in ("cell_lib", "z_logits", "u_logits", "raw_leak", "raw_drive_g", "other"):
            if bucket.get(sub_name, 0):
                lines.append(f"    {sub_name:<{stage_label_w}}: {bucket[sub_name]:>{width - stage_label_w - 6}}")
        stage_total = sum(bucket.values())
        lines.append(f"    {'── stage total:':<{stage_label_w + 2}} {stage_total}")
    lines.append("  " + "─" * (width - 2))
    lines.append(f"  {'total:':<{label_w + 2}} {total}")
    return "\n".join(lines)


class KirchhoffNet(nn.Module):
    """Multi-stage reduced differential KirchhoffNet ODE core.

    Each stage integrates its internal state for stage_times[i] over
    stage_steps[i] Heun steps. Between stages, StageTransfer truncates
    or zero-pads the state vector to match the next stage's width.

    Handles multi-stage edge_mismatch slicing internally: when ``ctx``
    carries an ``edge_mismatch`` tensor sized for the *sum* of all stages,
    ``forward()`` slices it into per-stage chunks so each stage receives
    mismatch matching its own edge count.
    """

    def __init__(
        self,
        stages: list[DifferentialStage],
        transfers: list[StageTransfer],
        stage_times: list[float],
        stage_steps: list[int],
    ) -> None:
        super().__init__()
        if len(stages) != len(stage_times) or len(stages) != len(stage_steps):
            raise ValueError(
                f"stages/stage_times/stage_steps length mismatch: "
                f"{len(stages)}, {len(stage_times)}, {len(stage_steps)}"
            )
        if len(transfers) != max(0, len(stages) - 1):
            raise ValueError(
                f"transfers length {len(transfers)} should be len(stages)-1 = {len(stages)-1}"
            )

        self.stages = nn.ModuleList(stages)
        self.transfers = nn.ModuleList(transfers)
        self.stage_times = [float(t) for t in stage_times]
        self.stage_steps = [int(n) for n in stage_steps]

        offset = 0
        self._edge_offsets = [offset]
        for s in self.stages:
            offset += s.num_edges()
            self._edge_offsets.append(offset)

    def forward(
        self,
        x0: torch.Tensor,
        store_trajectory: bool = False,
        drive_targets: list[torch.Tensor] | None = None,
        drive_scales: list[float] | None = None,
        solver: str = "heun",
        deq_cfg: dict | None = None,
        stage_noise_std: float = 0.0,
        stage_noise_generator: torch.Generator | None = None,
    ):
        x = x0
        all_trajs = []
        stage_outputs = []
        stage_infos = []
        for i, stage in enumerate(self.stages):
            x_drive_i = None if drive_targets is None else drive_targets[i]
            drive_scale_i = 0.0 if drive_scales is None else drive_scales[i]
            x, traj = stage(
                x0=x,
                t_span=self.stage_times[i],
                num_steps=self.stage_steps[i],
                store_trajectory=store_trajectory,
                x_drive=x_drive_i,
                drive_scale=drive_scale_i,
                solver=solver,
                deq_cfg=deq_cfg,
            )
            stage_outputs.append(x.detach())
            stage_infos.append(dict(getattr(stage, "last_deq_info", {}) or {}))
            if store_trajectory and traj is not None:
                all_trajs.append(traj)
            # kirchhoff-noise: per-stage additive Gaussian noise on the state
            # vector (thermal/IR-drop modeling on the analog voltage rails).
            if stage_noise_std > 0.0:
                if stage_noise_generator is not None:
                    noise = torch.empty_like(x)
                    noise.normal_(mean=0.0, std=stage_noise_std,
                                  generator=stage_noise_generator)
                    x = x + noise
                else:
                    x = x + torch.randn_like(x) * stage_noise_std
            if i < len(self.transfers):
                x = self.transfers[i](x)
        self.last_stage_outputs = stage_outputs
        self.last_stage_infos = stage_infos
        self.last_drive_targets = drive_targets
        self.last_drive_scales = list(drive_scales) if drive_scales is not None else None
        self.last_solver = solver
        return x, all_trajs if store_trajectory else None

    def parameter_breakdown(self) -> dict:
        """Return parameter counts per component for the regularizer / loss."""
        out = {
            "raw_leak_per_stage": [],
            "total_raw_leak": 0,
        }
        for s in self.stages:
            n = int(s.raw_leak.numel()) if hasattr(s, "raw_leak") else 0
            out["raw_leak_per_stage"].append(n)
            out["total_raw_leak"] += n
        return out


class KirchhoffNetWithIO(nn.Module):
    """Write/evolve/read pipeline: input_mapper -> core -> output_mapper.

    The pipeline enforces a hidden/projection split:
      - ``hid_count`` is the number of hidden nodes in the FIRST stage.
      - ``proj_count`` is the number of projection nodes in the FIRST stage.
      - The first-stage differential state vector is the concatenation
        ``[hidden(0..hid_count-1); projection(hid_count..hid_count+proj_count-1)]``.
      - InputMapper writes only to the hidden portion.
      - Projection portion is zero-initialized.
      - ``final_hid_count`` / ``final_proj_count`` describe the LAST stage's
        split; OutputMapper reads from the final projection portion (or
        final hidden portion if ``final_proj_count == 0``, with a warning,
        R1.4).

    For multi-stage networks with intermediate width changes, the read
    happens at the LAST stage's state width (after all StageTransfer
    operations). When all stages have the same width (the recommended
    case for paper v1, R5), first and final counts match.

    For mapper-only ablations, set ``t_span=0`` at the core level; the
    pipeline is unchanged.

    Sparse I/O (sparse-io-mapping spec):
      - ``write_idx``: list of hidden-node indices (length = in_dim).
        Each input feature `u_i` writes only to `h_{write_idx[i]}`.
        Hidden nodes NOT in `write_idx` are zero-initialized at t=0.
        When `None` (default), all hidden positions are written (dense
        mode, original behavior).
      - ``read_idx``: list of full-state indices to read from. When
        `None` (default), the read slice is the final projection
        portion (or final hidden portion, with the R1.4 warning). When
        provided, the OutputMapper must already be configured with the
        same `read_idx` (it will gather from the full state itself) and
        the pipeline hands the entire final state to the mapper.
    """

    def __init__(
        self,
        input_mapper,
        core: KirchhoffNet,
        output_mapper,
        hid_count: int,
        proj_count: int,
        final_hid_count: int | None = None,
        final_proj_count: int | None = None,
        write_idx: list[int] | None = None,
        read_idx: list[int] | None = None,
        enable_drive: bool = False,
        drive_mappers: list | None = None,
        drive_scales: list[float] | None = None,
    ) -> None:
        super().__init__()
        if hid_count < 0 or proj_count < 0:
            raise ValueError(
                f"hid_count/proj_count must be non-negative, got {hid_count}/{proj_count}"
            )
        if hid_count + proj_count == 0:
            raise ValueError("KirchhoffNetWithIO requires hid_count + proj_count > 0")
        self.input_mapper = input_mapper
        self.core = core
        self.output_mapper = output_mapper
        self.hid_count = int(hid_count)
        self.proj_count = int(proj_count)
        self.final_hid_count = int(hid_count if final_hid_count is None else final_hid_count)
        self.final_proj_count = int(proj_count if final_proj_count is None else final_proj_count)
        self.write_idx = list(write_idx) if write_idx is not None else None
        self.read_idx = list(read_idx) if read_idx is not None else None
        self.enable_drive = bool(enable_drive)
        if self.enable_drive:
            if drive_mappers is None:
                raise ValueError("enable_drive=True requires drive_mappers list")
            if len(drive_mappers) != len(core.stages):
                raise ValueError(
                    f"drive_mappers length {len(drive_mappers)} must equal "
                    f"num_stages {len(core.stages)}"
                )
            self.drive_mappers = nn.ModuleList(drive_mappers)
            # Validate all stages have the same width for drive target shape match.
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_drive=True requires all stages to have the same width, "
                    f"got {stage_widths}"
                )
            self.drive_scales = [float(s) for s in (drive_scales or DRIVE["drive_scales"])]
            if len(self.drive_scales) != len(core.stages):
                self.drive_scales = self.drive_scales[:len(core.stages)]
                if len(self.drive_scales) < len(core.stages):
                    self.drive_scales = self.drive_scales + [0.0] * (
                        len(core.stages) - len(self.drive_scales)
                    )
        else:
            self.drive_mappers = None
            self.drive_scales = []
        if self.write_idx is not None:
            if any(i < 0 or i >= self.hid_count for i in self.write_idx):
                raise ValueError(
                    f"write_idx entries must be in [0, hid_count)={self.hid_count}, "
                    f"got {self.write_idx}"
                )
            if len(set(self.write_idx)) != len(self.write_idx):
                raise ValueError(
                    f"write_idx entries must be unique, got {self.write_idx}"
                )
        if self.read_idx is not None:
            max_full = self.final_hid_count + self.final_proj_count
            if any(i < 0 or i >= max_full for i in self.read_idx):
                raise ValueError(
                    f"read_idx entries must be in [0, {max_full}), got {self.read_idx}"
                )
            self.read_start = 0
            self.read_dim = max_full
        elif self.final_proj_count == 0:
            warnings.warn(
                "KirchhoffNetWithIO: no projection nodes in final stage; "
                "OutputMapper falls back to reading from final hidden nodes (R1.4).",
                stacklevel=2,
            )
            self.read_start = 0
            self.read_dim = self.final_hid_count
        else:
            self.read_start = self.final_hid_count
            self.read_dim = self.final_proj_count
        self.read_slice = slice(self.read_start, self.read_start + self.read_dim)

    def _make_full_drive(self, hidden_drive: torch.Tensor) -> torch.Tensor:
        B = hidden_drive.shape[0]
        if self.proj_count > 0:
            proj_zeros = hidden_drive.new_zeros(B, self.proj_count)
            return torch.cat([hidden_drive, proj_zeros], dim=1)
        return hidden_drive

    def forward(
        self,
        u: torch.Tensor,
        store_trajectory: bool = False,
        solver: str = "heun",
        deq_cfg: dict | None = None,
        stage_noise_std: float = 0.0,
        stage_noise_generator: torch.Generator | None = None,
    ):
        x0 = self.input_mapper(u)
        if x0.size(1) != self.hid_count:
            raise ValueError(
                f"InputMapper output has {x0.size(1)} dims, expected hid_count={self.hid_count}"
            )
        if self.proj_count > 0:
            pad = x0.new_zeros(x0.size(0), self.proj_count)
            x0_full = torch.cat([x0, pad], dim=1)
        else:
            x0_full = x0

        # Build per-stage drive targets when persistent drive is enabled.
        if self.enable_drive and self.drive_mappers is not None:
            drive_targets = []
            for dm in self.drive_mappers:
                hidden_drive = dm(u)
                drive_targets.append(self._make_full_drive(hidden_drive))
        else:
            drive_targets = None

        if all(t == 0.0 for t in self.core.stage_times):
            x_final, trajs = x0_full, None
        else:
            x_final, trajs = self.core(
                x0_full, store_trajectory=store_trajectory,
                drive_targets=drive_targets,
                drive_scales=self.drive_scales if self.enable_drive else None,
                solver=solver,
                deq_cfg=deq_cfg,
                stage_noise_std=stage_noise_std,
                stage_noise_generator=stage_noise_generator,
            )
        if self.read_idx is not None:
            x_read = x_final
        else:
            x_read = x_final[:, self.read_slice]
            if x_read.size(1) != self.read_dim:
                raise ValueError(
                    f"Final state has {x_final.size(1)} dims; read_slice gives "
                    f"{x_read.size(1)} dims; expected {self.read_dim}"
                )
        y = self.output_mapper(x_read)
        return y, trajs

    def parameter_breakdown(self) -> dict:
        """Return trainable-parameter counts grouped by component.

        Walks ``self.named_parameters()`` and buckets each tensor by its
        fully-qualified name prefix:

          - ``input_mapper.*``
          - ``output_mapper.*``
          - ``drive_mappers.*``
          - ``core.stages.N.z_logits``  (structural)
          - ``core.stages.N.u_logits``  (structural, deprecated)
          - ``core.stages.N.raw_leak``  (dynamical)
          - ``core.stages.N.raw_drive_g``  (dynamical)
          - ``core.stages.N.cell_lib.*``  (per-edge device params)

        Returns a dict with per-group subtotals, per-stage ``stage_N``
        subtotals, and a ``total``. Use :func:`format_parameter_breakdown`
        for a human-readable rendering.
        """
        groups = {
            "input_mapper": 0,
            "output_mapper": 0,
            "drive_mappers": 0,
        }
        per_stage: dict[str, dict[str, int]] = {}
        total = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            n = int(p.numel())
            total += n
            if name.startswith("input_mapper"):
                groups["input_mapper"] += n
            elif name.startswith("output_mapper"):
                groups["output_mapper"] += n
            elif name.startswith("drive_mappers"):
                groups["drive_mappers"] += n
            elif name.startswith("core.stages."):
                parts = name.split(".")
                if len(parts) < 4:
                    continue
                stage_idx = parts[2]
                tail = ".".join(parts[3:])
                stage_key = f"stage_{stage_idx}"
                stage_bucket = per_stage.setdefault(stage_key, {
                    "cell_lib": 0,
                    "z_logits": 0,
                    "u_logits": 0,
                    "raw_leak": 0,
                    "raw_drive_g": 0,
                    "other": 0,
                })
                matched = False
                if tail == "z_logits":
                    stage_bucket["z_logits"] += n; matched = True
                elif tail == "u_logits":
                    stage_bucket["u_logits"] += n; matched = True
                elif tail == "raw_leak":
                    stage_bucket["raw_leak"] += n; matched = True
                elif tail == "raw_drive_g":
                    stage_bucket["raw_drive_g"] += n; matched = True
                elif tail.startswith("cell_lib."):
                    stage_bucket["cell_lib"] += n; matched = True
                if not matched:
                    stage_bucket["other"] += n
        return {
            "groups": groups,
            "per_stage": per_stage,
            "total": total,
        }

    def extra_repr(self) -> str:
        return (
            f"hid_count={self.hid_count}, proj_count={self.proj_count}, "
            f"final_hid_count={self.final_hid_count}, final_proj_count={self.final_proj_count}, "
            f"write_idx={self.write_idx}, read_idx={self.read_idx}"
        )
