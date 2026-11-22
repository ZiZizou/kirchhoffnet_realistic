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

from differential_stage import DifferentialStage
from stage_transfer import StageTransfer


__all__ = ["KirchhoffNet", "KirchhoffNetWithIO", "format_parameter_breakdown"]


def format_parameter_breakdown(breakdown: dict) -> str:
    """Render a KirchhoffNetWithIO.parameter_breakdown() dict as aligned text.

    Output example (single-stage friedman2):

        input_mapper:           20
        output_mapper:        1169
        stage_0:
          cell_lib:            320
          z_logits:             64
          raw_leak:             16
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
    for name in ("input_mapper", "output_mapper", "gln_rails"):
        if groups.get(name, 0) or name in ("input_mapper", "output_mapper"):
            lines.append(f"  {name:<{label_w}}: {groups.get(name, 0):>{width - label_w - 4}}")
    for stage_key in sorted(per_stage.keys()):
        lines.append(f"  {stage_key}:")
        bucket = per_stage[stage_key]
        for sub_name in (
            "cell_lib", "z_logits", "raw_leak",
            "boundary_cell_lib", "boundary_z_logits",
            "output_ode_cell_lib", "output_ode_z_logits",
            "readout_sense_z_logits", "readout_sense_cell_lib",
            "readout_crossbar", "raw_vref_sense",
            "vca_W", "vca_v", "other",
        ):
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
        u: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ):
        x = initial_state if initial_state is not None else x0
        all_trajs = []
        stage_outputs = []
        for i, stage in enumerate(self.stages):
            x, traj = stage(
                x0=x,
                t_span=self.stage_times[i],
                num_steps=self.stage_steps[i],
                store_trajectory=store_trajectory,
                u=u,
            )
            stage_outputs.append(x.detach())
            if store_trajectory and traj is not None:
                all_trajs.append(traj)
            if i < len(self.transfers):
                x = self.transfers[i](x)
        self.last_stage_outputs = stage_outputs
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
        enable_boundary: bool = False,
        boundary_fan_out: dict[int, list[int]] | None = None,
        enable_temporal_readout: bool = False,
        output_ode_count: int = 0,
        enable_vca: bool = False,
        vca_rank: int | None = None,
        vca_in_dim: int | None = None,
        vca_bias: bool | None = None,
        gln_rails: "GLNRails | None" = None,
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
        self.enable_temporal_readout = bool(enable_temporal_readout)
        self.output_ode_count = int(output_ode_count)
        if self.enable_temporal_readout:
            if self.output_ode_count <= 0:
                raise ValueError(
                    f"enable_temporal_readout=True requires output_ode_count > 0, "
                    f"got {self.output_ode_count}"
                )
            # Require all stages to have the same width so StageTransfer
            # doesn't need to remap the output ODE accumulator region
            # (which lives outside the topology's compact space).
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_temporal_readout=True requires all stages to "
                    f"have the same width, got {stage_widths}"
                )
            if (
                self.final_hid_count + self.final_proj_count + self.output_ode_count
                != stage_widths[0]
            ):
                raise ValueError(
                    f"enable_temporal_readout: final stage width "
                    f"{stage_widths[0]} must equal "
                    f"final_hid_count + final_proj_count + output_ode_count "
                    f"= {self.final_hid_count} + {self.final_proj_count} + "
                    f"{self.output_ode_count}"
                )
            # Read slice targets the output ODE accumulator region at the
            # tail of the ODE state vector.
            self.read_start = self.final_hid_count + self.final_proj_count
            self.read_dim = self.output_ode_count
        elif self.read_idx is not None:
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

        self.enable_boundary = bool(enable_boundary)
        self.boundary_fan_out = (
            None if boundary_fan_out is None
            else {int(k): [int(v) for v in tgts] for k, tgts in boundary_fan_out.items()}
        )
        # Post-readout residual-relu-tanh transfer (always-on, dataset-agnostic).
        # Placed after OutputMapper/OutputAffine so the settled temporal
        # accumulator readout gets one final learned nonlinearity before the
        # bounded-output sigmoid. Identity at init (w1=1, w2=w3=vth=0) so
        # epoch-0 behaviour is bit-identical to the pre-transfer baseline.
        # Hard-wired in KirchhoffNetWithIO so every future preset / problem
        # inherits it regardless of config/CLI.
        _out_dim_post = int(self.output_mapper.out_dim) if hasattr(self.output_mapper, "out_dim") else int(self.read_dim)
        self.post_readout_transfer = StageTransfer(
            in_nodes=_out_dim_post,
            out_nodes=_out_dim_post,
            activation="residual-relu-tanh",
        )
        # VCA (low-rank input-driven attention gating) lives on unfrozen
        # edges. Stored at the IO level so the forward pass knows to pass
        # ``u`` to every stage (boundary / temporal-readout edge currents
        # already need ``u``; VCA needs the raw input features for its
        # shared input projection ``W``).
        self.enable_vca = bool(enable_vca)
        self.vca_rank = int(vca_rank) if vca_rank is not None else 0
        self.vca_in_dim = int(vca_in_dim) if vca_in_dim is not None else 0
        # GLN rails (F2): one shared module for the whole net, registered
        # here (single ownership). Stages hold a plain (non-registered)
        # reference so the shared tensors are counted exactly once by
        # ``net.parameters()`` / ``net.state_dict()``.
        self.gln_rails = gln_rails
        # Stored for topology/API compatibility; per-edge VCA biases are
        # owned and applied by DifferentialStage instances.
        self.vca_bias = vca_bias
        if self.enable_boundary and self.boundary_fan_out is not None:
            stage_widths = [s.num_nodes for s in core.stages]
            if len(set(stage_widths)) != 1:
                raise ValueError(
                    f"enable_boundary=True requires all stages to have the same width, "
                    f"got {stage_widths}"
                )
    def forward(
        self,
        u: torch.Tensor,
        store_trajectory: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ):
        """Run the write/evolve/read pipeline on input ``u``.

        When ``initial_state`` is provided, it overrides the input-driven
        initial ODE state (input_mapper output, or zero-init under
        boundary mode). The shape must be ``[batch, hid_count + proj_count
        + output_ode_count]`` (i.e. the full stage width). This is the
        entry point for sequential/streaming workloads (CTLE
        symbol-by-symbol, etc.) where the fabric state must carry across
        calls.

        When ``return_final_state`` is True the final ODE state (after
        integration) is returned as a third tuple element so the caller
        can pass it back on the next call. Trajectory storage is forced
        off when ``return_final_state`` is set (trajectory tensors are
        batch-internal and not designed to be cached across calls).
        """
        if initial_state is not None:
            # Carry-over path: caller owns the initial state. Validate shape
            # matches the full stage width we expect.
            expected_width = self.hid_count + self.proj_count + self.output_ode_count
            if initial_state.size(1) != expected_width:
                raise ValueError(
                    f"initial_state has {initial_state.size(1)} dims; expected "
                    f"hid_count + proj_count + output_ode_count = {expected_width}"
                )
            x0_full = initial_state
        else:
            if self.enable_boundary:
                # Boundary-fan-out mode: all dynamic nodes start at zero.
                # The input signal enters only through boundary-terminal OTA
                # edges, computed per-step in each stage's RHS.
                x0 = u.new_zeros(u.size(0), self.hid_count)
            else:
                x0 = self.input_mapper(u)
            if x0.size(1) != self.hid_count:
                raise ValueError(
                    f"InputMapper output has {x0.size(1)} dims, expected hid_count={self.hid_count}"
                )
            # Append zero padding for projection and (when enabled) output
            # ODE accumulator nodes so the initial ODE state has the full
            # stage width. Output ODE nodes are always zero-initialized;
            # their current comes from the temporal-readout OTA edges only.
            parts = [x0]
            if self.proj_count > 0:
                parts.append(x0.new_zeros(x0.size(0), self.proj_count))
            if self.output_ode_count > 0:
                parts.append(x0.new_zeros(x0.size(0), self.output_ode_count))
            if len(parts) == 1:
                x0_full = x0
            else:
                x0_full = torch.cat(parts, dim=1)

        if all(t == 0.0 for t in self.core.stage_times):
            x_final, trajs = x0_full, None
        else:
            x_final, trajs = self.core(
                x0_full, store_trajectory=store_trajectory and not return_final_state,
                u=u if (
                    self.enable_boundary or self.enable_vca
                    or self.gln_rails is not None
                ) else None,
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
        y = self.post_readout_transfer(y)
        if return_final_state:
            return y, None, x_final
        return y, trajs

    def parameter_breakdown(self) -> dict:
        """Return trainable-parameter counts grouped by component.

        Walks ``self.named_parameters()`` and buckets each tensor by its
        fully-qualified name prefix:

          - ``input_mapper.*``
          - ``output_mapper.*``
          - ``post_readout_transfer.*``  (post-OutputMapper residual-relu-tanh)
          - ``core.stages.N.z_logits``  (structural)
          - ``core.stages.N.raw_leak``  (dynamical)
          - ``core.stages.N.cell_lib.*``  (per-edge device params)

        Returns a dict with per-group subtotals, per-stage ``stage_N``
        subtotals, and a ``total``. Use :func:`format_parameter_breakdown`
        for a human-readable rendering.
        """
        groups = {
            "input_mapper": 0,
            "output_mapper": 0,
            "post_readout_transfer": 0,
            "gln_rails": 0,
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
            elif name.startswith("post_readout_transfer"):
                groups["post_readout_transfer"] += n
            elif name.startswith("output_mapper"):
                groups["output_mapper"] += n
            elif name.startswith("gln_rails"):
                groups["gln_rails"] += n
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
                    "raw_leak": 0,
                    "boundary_z_logits": 0,
                    "boundary_cell_lib": 0,
                    "output_ode_z_logits": 0,
                    "output_ode_cell_lib": 0,
                    "readout_sense_z_logits": 0,
                    "readout_sense_cell_lib": 0,
                    "readout_crossbar": 0,
                    "raw_vref_sense": 0,
                    "vca_W": 0,
                    "vca_v": 0,
                    "other": 0,
                })
                matched = False
                if tail == "z_logits":
                    stage_bucket["z_logits"] += n; matched = True
                elif tail == "raw_leak":
                    stage_bucket["raw_leak"] += n; matched = True
                elif tail == "boundary_z_logits":
                    stage_bucket["boundary_z_logits"] += n; matched = True
                elif tail.startswith("boundary_cell_lib."):
                    stage_bucket["boundary_cell_lib"] += n; matched = True
                elif tail.startswith("cell_lib."):
                    stage_bucket["cell_lib"] += n; matched = True
                elif tail == "output_ode_z_logits":
                    stage_bucket["output_ode_z_logits"] += n; matched = True
                elif tail.startswith("output_ode_cell_lib."):
                    stage_bucket["output_ode_cell_lib"] += n; matched = True
                elif tail == "readout_sense_z_logits":
                    stage_bucket["readout_sense_z_logits"] += n; matched = True
                elif tail.startswith("readout_sense_cell_lib."):
                    stage_bucket["readout_sense_cell_lib"] += n; matched = True
                elif tail == "readout_crossbar_W":
                    stage_bucket["readout_crossbar"] += n; matched = True
                elif tail == "raw_vref_sense":
                    stage_bucket["raw_vref_sense"] += n; matched = True
                elif tail == "vca_W" or tail == "vca_W_core":
                    stage_bucket["vca_W"] += n; matched = True
                elif tail.startswith("vca_v_"):
                    stage_bucket["vca_v"] += n; matched = True
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
