"""DifferentialStage: sparse COO graph + KCL scatter-add + Heun integration.

Per-node dynamics:
    C_eff * dx_j/dt = sum_in I_edge - sum_out I_edge - leak_j * x_j - clip_j(x_j)

Edge currents are computed by an IdealizedCellLibrary.
Heun integration (predictor-corrector, 2nd order) is used for fixed-step
BPTT. The stage returns both the final state and the full trajectory so
that regularizers can be evaluated along the path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DRIVE,
    INIT,
    PHYS,
    SOLVER,
)
from cell_library import IdealizedCellLibrary


__all__ = ["DifferentialStage"]


class DifferentialStage(nn.Module):
    """A single stage of the reduced differential KirchhoffNet.

    Args:
        num_nodes: Number of differential nodes in this stage's internal state.
        src: List of source node ids (length E).
        dst: List of destination node ids (length E).
        cell_lib: IdealizedCellLibrary used to compute edge currents.
        c_eff: Effective node capacitance (default from config).
        x_max: Differential rail limit (default from config).
        clip_current: Soft rail clip current magnitude (default from config).
        clip_softness: Soft rail clip transition width (default from config).
        logits_z_bias: Initial bias toward Z (disabled) cell, applied to
            logits[:, z_index] in __init__.
        write_idx: Indices of hidden nodes that receive persistent bounded
            drive current. When provided, a drive source is created with
            learnable per-node conductance ``raw_drive_g``. ``None`` disables
            drive for this stage.
        drive_isat: Saturation current for the bounded drive source. When
            ``None``, uses ``config.DRIVE["drive_isat"]``.
    """

    def __init__(
        self,
        num_nodes: int,
        src: list[int],
        dst: list[int],
        cell_lib: IdealizedCellLibrary,
        c_eff: float | None = None,
        x_max: float | None = None,
        clip_current: float | None = None,
        clip_softness: float | None = None,
        logits_z_bias: float | None = None,
        write_idx: list[int] | None = None,
        drive_isat: float | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cell_lib = cell_lib

        self.c_eff = float(c_eff if c_eff is not None else PHYS["C_eff"])
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.clip_current = float(clip_current if clip_current is not None else PHYS["clip_current"])
        self.clip_softness = float(clip_softness if clip_softness is not None else PHYS["clip_softness"])

        if len(src) != len(dst):
            raise ValueError(f"src/dst length mismatch: {len(src)} vs {len(dst)}")
        if any(s == d for s, d in zip(src, dst)):
            raise ValueError("Self-loops are not allowed in DifferentialStage edges.")
        if any(s < 0 or s >= num_nodes or d < 0 or d >= num_nodes for s, d in zip(src, dst)):
            raise ValueError("Edge endpoint out of range for num_nodes.")

        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))

        # Persistent drive: optional bounded input-current source.
        if write_idx is not None:
            if any(i < 0 or i >= self.num_nodes for i in write_idx):
                raise ValueError(f"write_idx entries must be in [0, {self.num_nodes}), got {write_idx}")
            if len(set(write_idx)) != len(write_idx):
                raise ValueError(f"write_idx entries must be unique, got {write_idx}")
            self.register_buffer("_drive_idx", torch.tensor(write_idx, dtype=torch.long))
            self.raw_drive_g = nn.Parameter(
                torch.full((len(write_idx),), float(DRIVE["raw_drive_g_init"]))
            )
            self.drive_isat = float(drive_isat if drive_isat is not None else DRIVE["drive_isat"])
            self._has_drive = True
        else:
            self.register_buffer("_drive_idx", torch.empty(0, dtype=torch.long))
            self._has_drive = False
            self.drive_isat = 0.0

        E = len(src)
        Q = cell_lib.num_cells

        self.logits = nn.Parameter(torch.zeros(E, Q))
        with torch.no_grad():
            self.logits[:, cell_lib.z_index] = (
                INIT["logits_z_bias"] if logits_z_bias is None else float(logits_z_bias)
            )

        self.raw_mult = nn.Parameter(torch.full((E,), float(INIT["raw_mult_init"])))
        self.raw_leak = nn.Parameter(torch.full((num_nodes,), float(INIT["raw_leak_init"])))

        # Gate parameters for complexity-regularized pruning (CP-1, CP-2).
        # z_e = sigmoid(z_logits) is the edge gate: multiplies the edge current.
        # u_j = sigmoid(u_logits) is the node gate: gates the node voltage.
        # Initialized to a large positive value so all edges/nodes are active at start.
        z_init = float(INIT.get("z_logit_init", 5.0))
        u_init = float(INIT.get("u_logit_init", 5.0))
        self.z_logits = nn.Parameter(torch.full((E,), z_init))
        self.u_logits = nn.Parameter(torch.full((num_nodes,), u_init))

    def num_edges(self) -> int:
        return int(self.src.numel())

    def drive_current(
        self, x: torch.Tensor, x_drive: torch.Tensor | None, drive_scale: float
    ) -> torch.Tensor:
        if x_drive is None or not self._has_drive or drive_scale == 0.0:
            return torch.zeros_like(x)
        g_in = F.softplus(self.raw_drive_g).unsqueeze(0)
        err = x_drive[:, self._drive_idx] - x[:, self._drive_idx]
        i = self.drive_isat * torch.tanh(g_in * err / self.drive_isat)
        i = float(drive_scale) * i
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        out[:, self._drive_idx] = i
        return out.to(dtype=x.dtype)

    def rhs(self, x: torch.Tensor, ctx, tau: float = 1.0, cell_mode: str = "soft",
            x_drive: torch.Tensor | None = None, drive_scale: float = 0.0) -> torch.Tensor:
        """Compute dx/dt at state x. x: [batch, num_nodes].

        Gate application (CP-2):
        - Node gate: x_gated = x * sigmoid(u_logits) — gates node voltage before
          being used as edge input. When u_j -> 0 the node is effectively
          disconnected from all edges.
        - Edge gate: i_edge *= sigmoid(z_logits) — multiplies the edge current
          after cell-library evaluation. When z_e -> 0 the edge contributes
          zero current regardless of cell type.

        ``cell_mode`` (four-phase-redesign/Phase 2a): forwarded to
        ``cell_lib.forward`` to control the cell-selection mode. ``'soft'``
        uses a softmax mixture (default). ``'ste'`` uses one cell per edge
        in the forward pass with straight-through soft gradients.
        """
        # Node gate: scale node voltages before computing edge inputs.
        # Driven nodes are forced open so persistent input is not gated away.
        node_mask = torch.sigmoid(self.u_logits)  # [N]
        if self._has_drive:
            node_mask = node_mask.clone()
            node_mask[self._drive_idx] = 1.0
        x_gated = x * node_mask.unsqueeze(0)  # [B, N]

        x_src = x_gated[:, self.src]
        x_dst = x_gated[:, self.dst]

        i_edge = self.cell_lib(
            x_src=x_src,
            x_dst=x_dst,
            logits=self.logits,
            raw_mult=self.raw_mult,
            x_max=self.x_max,
            ctx=ctx,
            tau=tau,
            cell_mode=cell_mode,
        )

        # Edge gate: multiply each edge's current by its gate.
        edge_mask = torch.sigmoid(self.z_logits)  # [E]
        i_edge = i_edge * edge_mask.unsqueeze(0)  # [B, E]

        # KCL scatter-add: accumulate in float32 for AMP robustness.
        # Under torch.autocast the node/edge gate multiplications promote
        # i_edge to fp32 even when x is fp16, so x.new_zeros() creates a
        # Half accumulator while the source is Float → index_add_ error.
        # Accumulating in float32 then casting back to x.dtype is safe and
        # numerically preferable for scatter operations.
        i_edge_f32 = i_edge.float()
        acc = torch.zeros_like(x, dtype=torch.float32)
        acc.index_add_(1, self.dst, i_edge_f32)
        acc.index_add_(1, self.src, -i_edge_f32)
        acc = acc.to(dtype=x.dtype)

        leak = F.softplus(self.raw_leak).unsqueeze(0)  # [1, N]
        leak_term = leak * x

        clip = torch.sigmoid((x - self.x_max) / self.clip_softness)
        clip = clip - torch.sigmoid((-x - self.x_max) / self.clip_softness)
        clip_term = self.clip_current * clip

        i_drive = self.drive_current(x, x_drive, drive_scale)
        return (acc + i_drive - leak_term - clip_term) / self.c_eff

    def compile_rhs(self, backend: str = "inductor"):
        """Compile `rhs` with `torch.compile` for kernel fusion.

        Note: `index_add_` on `acc` is a scatter op that may force a graph
        break in some PyTorch versions. The scatter itself remains in eager
        mode; the `cell_lib` call and post-scatter math are fused.
        """
        self.rhs = torch.compile(self.rhs, backend=backend)

    def forward(
        self,
        x0: torch.Tensor,
        ctx,
        t_span: float | None = None,
        num_steps: int | None = None,
        tau: float | None = None,
        store_trajectory: bool = True,
        cell_mode: str = "soft",
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Integrate stage with fixed-step Heun. Returns (x_final, traj).

        traj: [batch, num_nodes, num_steps+1] if store_trajectory else None.

        ``cell_mode`` (four-phase-redesign/Phase 2a): forwarded to
        ``rhs`` to control cell selection. ``'soft'`` is the standard
        mixture; ``'ste'`` uses one cell per edge in the forward pass
        with straight-through soft gradients.
        """
        t_span = float(t_span if t_span is not None else SOLVER["t_span"])
        num_steps = int(num_steps if num_steps is not None else SOLVER["num_steps"])
        tau = float(tau if tau is not None else 1.0)
        dt = t_span / float(num_steps)

        x = x0
        traj_chunks = [x] if store_trajectory else None

        for _ in range(num_steps):
            k1 = self.rhs(x, ctx=ctx, tau=tau, cell_mode=cell_mode,
                          x_drive=x_drive, drive_scale=drive_scale)
            x_pred = x + dt * k1
            k2 = self.rhs(x_pred, ctx=ctx, tau=tau, cell_mode=cell_mode,
                          x_drive=x_drive, drive_scale=drive_scale)
            x = x + 0.5 * dt * (k1 + k2)
            if store_trajectory:
                traj_chunks.append(x)

        traj = torch.stack(traj_chunks, dim=2) if store_trajectory else None
        return x, traj

    def edge_gates(self) -> torch.Tensor:
        """Return edge gate values z_e = σ(z_logits), shape [E]."""
        return torch.sigmoid(self.z_logits)

    def node_gates(self) -> torch.Tensor:
        """Return node gate values u_j = σ(u_logits), shape [N]."""
        return torch.sigmoid(self.u_logits)

    def active_edge_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask of edges that survive pruning at the given threshold."""
        return self.edge_gates() > threshold

    def active_node_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask of nodes that survive pruning at the given threshold."""
        return self.node_gates() > threshold

    def parameter_breakdown(self) -> dict:
        """Return parameter counts including gate parameters (for diagnostics)."""
        return {
            "logits": int(self.logits.numel()),
            "raw_mult": int(self.raw_mult.numel()),
            "raw_leak": int(self.raw_leak.numel()),
            "z_logits": int(self.z_logits.numel()),
            "u_logits": int(self.u_logits.numel()),
            "total": (
                int(self.logits.numel())
                + int(self.raw_mult.numel())
                + int(self.raw_leak.numel())
                + int(self.z_logits.numel())
                + int(self.u_logits.numel())
            ),
        }
