"""StageTransfer: width-changing layer between DifferentialStages.

Truncation or zero-padding. New nodes (when out_nodes > in_nodes) are
initialized to zero differential state. An optional pointwise transform
applied after the width adjustment:

* ``activation="none"`` (default): pure identity / truncate / pad.
* ``activation="relu"``: ``F.relu`` on the state vector.
* ``activation="residual"``: per-node learnable residual
  ``W1 * x + W2 * tanh(x)``. W1
  is initialized to 1 and W2 to 0, so the network starts as identity.
* ``activation="residual_mixing"``: like ``"residual"`` plus an additive
  zero-initialized mixing term that can mix signals across nodes.
  ``residual_rank=-1`` (or >= out_nodes) gives a full N×N matrix,
  ``residual_rank=0`` reduces to the diagonal ``"residual"`` form, and
  ``residual_rank=r`` with ``0 < r < N`` uses a low-rank factorization
  ``U [N×r] @ V [r×N]``. Mixing params are zero-initialized so the
  network starts as identity regardless of rank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["StageTransfer", "STAGE_TRANSFER_VALID_ACTIVATIONS"]

STAGE_TRANSFER_VALID_ACTIVATIONS = ("none", "relu", "residual", "residual_mixing", "residual-relu-tanh")


class StageTransfer(nn.Module):
    """Truncation/zero-padding between stages of differing width.

    Args:
        in_nodes: Number of input nodes (state width of the upstream stage).
        out_nodes: Number of output nodes (state width of the downstream
            stage).
        activation: Optional transform applied after the width adjustment.
            ``"none"`` (default): identity / truncate / pad.
            ``"relu"``: ``F.relu`` on the state vector.
            ``"residual"``: per-node learnable
            ``residual_w1 * x + residual_w2 * tanh(x)``. Adds
            ``out_nodes`` learnable parameters per side (W1 init=1, W2
            init=0).
            ``"residual_mixing"``: same as ``"residual"`` plus an
            additive zero-initialized mixing term (full matrix or
            low-rank depending on ``residual_rank``).
        residual_rank: Rank of the additive mixing term when
            ``activation="residual_mixing"``. ``-1`` (default) or any
            value ``>= out_nodes``: full N×N matrix. ``0``: pure
            diagonal (mixing term is the zero matrix, equivalent to
            ``"residual"``). ``1..out_nodes-1``: low-rank factorization
            with N×r and r×N factors. Ignored for other activation
            modes.
    """

    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        activation: str = "none",
        residual_rank: int = -1,
        hw=None,
    ) -> None:
        super().__init__()
        # Hardware configuration (``hw.HWConfig`` or ``None``).  A plain
        # attribute, not a submodule: it adds nothing to ``state_dict``.
        self.hw = hw
        if in_nodes < 0 or out_nodes < 0:
            raise ValueError(f"Negative node count: in={in_nodes}, out={out_nodes}")
        if activation not in STAGE_TRANSFER_VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {STAGE_TRANSFER_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        self.in_nodes = int(in_nodes)
        self.out_nodes = int(out_nodes)
        self.activation = activation

        # Diagonal residual params are shared by "residual" and
        # "residual_mixing". For "residual_mixing" the mixing term is
        # added on top via zero-initialized full/low-rank matrices.
        if activation in ("residual", "residual_mixing"):
            self.residual_w1 = nn.Parameter(torch.full((self.out_nodes,), 1.0))
            self.residual_w2 = nn.Parameter(torch.full((self.out_nodes,), 0.0))
        elif activation == "residual-relu-tanh":
            self.residual_w1 = nn.Parameter(torch.full((self.out_nodes,), 1.0))
            self.residual_w2 = nn.Parameter(torch.full((self.out_nodes,), 0.0))
            self.residual_w3 = nn.Parameter(torch.zeros(self.out_nodes))
            self.residual_vth = nn.Parameter(torch.zeros(self.out_nodes))
        else:
            self.register_parameter("residual_w1", None)
            self.register_parameter("residual_w2", None)
            self.register_parameter("residual_w3", None)
            self.register_parameter("residual_vth", None)

        # Mixing-term parameters and rank flag (only meaningful for
        # "residual_mixing").
        self.residual_rank = 0
        if activation == "residual_mixing":
            r = int(residual_rank)
            if r == 0:
                # Pure diagonal, no mixing params.
                self.residual_rank = 0
            elif r == -1 or r >= self.out_nodes:
                # Full N×N matrix.
                self.residual_rank = -1
                self.mix_w1 = nn.Parameter(
                    torch.zeros(self.out_nodes, self.out_nodes)
                )
                self.mix_w2 = nn.Parameter(
                    torch.zeros(self.out_nodes, self.out_nodes)
                )
            else:
                # Low-rank factorization U [N×r] @ V [r×N].
                if r < 1:
                    raise ValueError(
                        f"residual_rank must be -1, 0, or a positive int, "
                        f"got {residual_rank!r}"
                    )
                self.residual_rank = r
                self.mix_u1 = nn.Parameter(torch.zeros(self.out_nodes, r))
                self.mix_v1 = nn.Parameter(torch.zeros(r, self.out_nodes))
                self.mix_u2 = nn.Parameter(torch.zeros(self.out_nodes, r))
                self.mix_v2 = nn.Parameter(torch.zeros(r, self.out_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_nodes:
            raise ValueError(
                f"Expected x shape (batch, {self.in_nodes}), got {tuple(x.shape)}"
            )
        if self.out_nodes == self.in_nodes:
            out = x
        elif self.out_nodes < self.in_nodes:
            out = x[:, : self.out_nodes]
        else:
            pad = x.new_zeros(x.size(0), self.out_nodes - self.in_nodes)
            out = torch.cat([x, pad], dim=1)

        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "residual-relu-tanh":
            transformed = (
                self.residual_w1 * out
                + self.residual_w2 * torch.tanh(out)
                + self.residual_w3 * F.relu(out - self.residual_vth)
            )
            out = transformed
        elif self.activation in ("residual", "residual_mixing"):
            transformed = self.residual_w1 * out + self.residual_w2 * torch.tanh(out)

            if self.activation == "residual_mixing":
                r = self.residual_rank
                if r == -1:
                    transformed = (
                        transformed
                        + (out @ self.mix_w1.T)
                        + (torch.tanh(out) @ self.mix_w2.T)
                    )
                elif r > 0:
                    transformed = transformed + (out @ self.mix_v1.T) @ self.mix_u1.T
                    transformed = (
                        transformed
                        + (torch.tanh(out) @ self.mix_v2.T) @ self.mix_u2.T
                    )
                # r == 0: pure diagonal, no mixing term to add.

            out = transformed

        c = None if self.hw is None else self.hw.constraints
        if c is None or not c.hw_clamp_transfer:
            return out
        return self.clamp_to_rails(out)

    def clamp_to_rails(self, out: torch.Tensor) -> torch.Tensor:
        """Phase 3a 3.4 (``hw_clamp_transfer``): the transfer writes a node voltage,
        so its output cannot leave the supply rails ``[-x_hard, x_hard]``
        (``x_hard = VDD/(2 Vu)``, 5.714 units at Vu = 0.07 V).

        Hard clamp in the forward pass with a straight-through (identity)
        gradient: ``clamp(out.detach()) + (out - out.detach())``.  The forward
        value is exactly the clamped write (the second term is an exact 0); the
        backward pass is that of the unclamped transfer, so a saturated write
        still passes gradient to the transfer parameters and to the upstream
        stage (a plain ``torch.clamp`` would zero it).

        Only the interstage transfers get this hook: the post-readout transfer
        of ``KirchhoffNetWithIO`` acts on the ``OutputAffine`` output (a
        digital/post-processing quantity, not a node voltage) and is left alone.
        """
        x_hard = float(self.hw.units.x_hard)
        return torch.clamp(out.detach(), -x_hard, x_hard) + (out - out.detach())

    def extra_repr(self) -> str:
        parts = [
            f"in_nodes={self.in_nodes}",
            f"out_nodes={self.out_nodes}",
            f"activation={self.activation}",
        ]
        if self.activation == "residual_mixing":
            parts.append(f"residual_rank={self.residual_rank}")
        return ", ".join(parts)