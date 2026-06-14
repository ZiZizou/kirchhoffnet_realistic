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


__all__ = ["KirchhoffNet", "KirchhoffNetWithIO"]


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
        ctx,
        tau: float | None = None,
        store_trajectory: bool = False,
    ):
        from sim_context import SimContext

        x = x0
        all_trajs = []
        for i, stage in enumerate(self.stages):
            tau_i = 1.0 if tau is None else tau
            stage_ctx = ctx
            if ctx is not None and ctx.edge_mismatch is not None:
                start = self._edge_offsets[i]
                end = self._edge_offsets[i + 1]
                stage_ctx = SimContext(
                    temp_c=ctx.temp_c,
                    global_gain_shift=ctx.global_gain_shift,
                    edge_mismatch=ctx.edge_mismatch[start:end],
                )
            x, traj = stage(
                x0=x,
                ctx=stage_ctx,
                t_span=self.stage_times[i],
                num_steps=self.stage_steps[i],
                tau=tau_i,
                store_trajectory=store_trajectory,
            )
            if store_trajectory and traj is not None:
                all_trajs.append(traj)
            if i < len(self.transfers):
                x = self.transfers[i](x)
        return x, all_trajs if store_trajectory else None

    def parameter_breakdown(self) -> dict:
        """Return parameter counts per component for the regularizer / loss."""
        out = {
            "logits_per_stage": [],
            "raw_mult_per_stage": [],
            "raw_leak_per_stage": [],
            "total_logits": 0,
            "total_raw_mult": 0,
            "total_raw_leak": 0,
        }
        for s in self.stages:
            out["logits_per_stage"].append(int(s.logits.numel()))
            out["raw_mult_per_stage"].append(int(s.raw_mult.numel()))
            out["raw_leak_per_stage"].append(int(s.raw_leak.numel()))
            out["total_logits"] += int(s.logits.numel())
            out["total_raw_mult"] += int(s.raw_mult.numel())
            out["total_raw_leak"] += int(s.raw_leak.numel())
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

    def forward(
        self,
        u: torch.Tensor,
        ctx,
        tau: float | None = None,
        store_trajectory: bool = False,
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
        if all(t == 0.0 for t in self.core.stage_times):
            x_final, trajs = x0_full, None
        else:
            x_final, trajs = self.core(
                x0_full, ctx=ctx, tau=tau, store_trajectory=store_trajectory
            )
        if self.read_idx is not None:
            x_read = x_final
        else:
            x_read = x_final[:, self.read_slice]
            if x_read.size(1) != self.read_dim:
                raise ValueError(
                    f"Final state has {x_final.size(1)} dims; read_slice gives "
                    f"{x_read.size(1)} dims, expected {self.read_dim}"
                )
        y = self.output_mapper(x_read)
        return y, trajs

    def extra_repr(self) -> str:
        return (
            f"hid_count={self.hid_count}, proj_count={self.proj_count}, "
            f"final_hid_count={self.final_hid_count}, final_proj_count={self.final_proj_count}, "
            f"write_idx={self.write_idx}, read_idx={self.read_idx}"
        )
