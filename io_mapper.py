"""Input/output mappers for the differential KirchhoffNet.

InputMapper:  x_j(0) = x_max * tanh(Linear(u))
SparseInputMapper: x_{write_idx[i]}(0) = x_max * tanh(a_i * u_i + b_i)
ProjectedSparseInputMapper: x_{write_idx[k]}(0) = x_max * tanh(Linear(u))[k]
                   (length(write_idx) >= in_dim; learned projection lets
                   one input feature feed multiple target hidden nodes)
FanOutInputMapper: x_j(0) = x_max * tanh(gain_{i,k} * u_i + bias_{i,k})
                   for j in fan_out_map[i], 0 otherwise.
OutputMapper: y_hat = Linear(x_final)
OutputMapper(read_idx=...):  y_hat = Linear(x[:, read_idx])

No MLPs, no hidden layers, no LayerNorm. The ODE core is the compute engine.

Honest I/O (R1): In the default pipeline, the InputMapper writes ONLY to the
hidden-node portion of the differential state vector; the projection-node
positions are forced to zero. The OutputMapper reads ONLY from the
projection-node portion (or from hidden nodes when no projections exist).

Sparse I/O (sparse-io-mapping spec): With write_idx and read_idx lists,
input features are written one-to-one to designated hidden nodes, and the
readout is taken only from designated full-state indices (favoring hidden
nodes). Non-write hidden nodes are zero-initialized at t=0.

Projected sparse I/O (sparse-proj-write spec): with write_mode='sparse_proj',
in_dim inputs are mapped through a learned nn.Linear to len(write_idx)
targets and then scattered. Each input feature may influence multiple target
hidden nodes (when len(write_idx) > in_dim). All non-target hidden nodes
remain zero-initialized at t=0.

Fan-out I/O (smooth2d-sanity-pass spec): With fan_out_map, each input
feature writes to K>1 designated hidden nodes via per-target (gain, bias)
pairs. This widens the input injection footprint beyond 1-to-1 while
remaining sparse.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import INIT, PHYS


__all__ = [
    "InputMapper",
    "RobustInputMapper",
    "SparseInputMapper",
    "ProjectedSparseInputMapper",
    "FanOutInputMapper",
    "OutputMapper",
    "GroupedOutputMapper",
]



def _init_linear_small(linear: nn.Linear, gain_scale: float) -> None:
    """Xavier init scaled by gain_scale; bias zeros."""
    nn.init.xavier_uniform_(linear.weight)
    with torch.no_grad():
        linear.weight.mul_(gain_scale)
        if linear.bias is not None:
            linear.bias.zero_()


class InputMapper(nn.Module):
    """x_j(0) = x_max * tanh(W u + b)."""

    def __init__(self, in_dim: int, out_dim: int, x_max: float | None = None) -> None:
        super().__init__()
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.gain = nn.Linear(in_dim, out_dim, bias=True)
        _init_linear_small(self.gain, gain_scale=INIT["gain_scale"])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.x_max * torch.tanh(self.gain(u))


class RobustInputMapper(nn.Module):
    """Like InputMapper with per-feature log-scale preconditioner.

    Use for tasks with heterogeneous feature scales (e.g. California housing).
    """

    def __init__(self, in_dim: int, out_dim: int, x_max: float | None = None) -> None:
        super().__init__()
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.log_scale = nn.Parameter(torch.zeros(in_dim))
        self.gain = nn.Linear(in_dim, out_dim, bias=True)
        _init_linear_small(self.gain, gain_scale=INIT["gain_scale"])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_scaled = u * torch.exp(self.log_scale)
        return self.x_max * torch.tanh(self.gain(u_scaled))


class SparseInputMapper(nn.Module):
    """One-to-one input writer: x_{write_idx[i]}(0) = x_max * tanh(a_i u_i + b_i).

    Each input feature `u_i` is mapped independently through its own
    (scalar gain, scalar bias) pair and written to exactly one designated
    hidden node `h_{write_idx[i]}`. Hidden nodes NOT in ``write_idx`` are
    left at 0 in the mapper's output; the caller (KirchhoffNetWithIO) is
    responsible for assembling the full state vector.

    Parameter count = 2 * in_dim (vs InputMapper = in_dim * out_dim + out_dim).
    For d << hid_count this is dramatically smaller.

    SR1.6: raises ``ValueError`` if in_dim > out_dim.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        write_idx: list[int],
        x_max: float | None = None,
    ) -> None:
        super().__init__()
        if in_dim > out_dim:
            raise ValueError(
                f"SparseInputMapper: in_dim={in_dim} > out_dim={out_dim}; "
                f"cannot assign one-to-one (SR1.6)"
            )
        if len(write_idx) != in_dim:
            raise ValueError(
                f"SparseInputMapper: len(write_idx)={len(write_idx)} must equal "
                f"in_dim={in_dim}"
            )
        if any(i < 0 or i >= out_dim for i in write_idx):
            raise ValueError(
                f"SparseInputMapper: write_idx entries must be in [0, out_dim)={out_dim}, "
                f"got {write_idx}"
            )
        if len(set(write_idx)) != len(write_idx):
            raise ValueError(
                f"SparseInputMapper: write_idx entries must be unique, got {write_idx}"
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.write_idx = list(write_idx)
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.gain = nn.Parameter(torch.full((in_dim,), INIT["gain_scale"]))
        self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.size(-1) != self.in_dim:
            raise ValueError(
                f"SparseInputMapper: input has {u.size(-1)} features, "
                f"expected in_dim={self.in_dim}"
            )
        per_feature = self.x_max * torch.tanh(u * self.gain + self.bias)
        x = u.new_zeros(*u.shape[:-1], self.out_dim)
        idx = torch.tensor(self.write_idx, dtype=torch.long, device=u.device)
        x.index_copy_(-1, idx, per_feature.to(dtype=x.dtype))
        return x


class FanOutInputMapper(nn.Module):
    """Sparse fan-out input writer.

    Each input feature ``u_i`` is mapped to multiple designated hidden-node
    targets via per-target learnable (gain, bias) pairs:

        x_{fan_out_map[i][k]}(0) = x_max * tanh(gain_{i,k} * u_i + bias_{i,k})

    Hidden nodes NOT in the union of all ``fan_out_map`` values are left
    at 0 in the mapper's output; the caller (KirchhoffNetWithIO) is
    responsible for assembling the full state vector.

    Parameter count = 2 * K_total, where K_total = sum of len(targets)
    over all inputs. For d=2 inputs with 3 targets each this is 12 params,
    versus Linear(2, 25) = 50+25 = 75 params for the dense InputMapper.

    Args:
        in_dim: Number of input features.
        out_dim: Number of hidden nodes in the target stage.
        fan_out_map: dict mapping input index to list of target hidden-node
            indices. Must cover all input indices ``0..in_dim-1``.
        x_max: Rail limit; defaults to ``PHYS["x_max"]``.

    Validation:
        - All input indices 0..in_dim-1 must appear as keys.
        - All target indices must be in [0, out_dim).
        - No target index may appear in more than one input's list.

    Raises:
        ValueError: on any of the above violations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fan_out_map: dict[int, list[int]],
        x_max: float | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.fan_out_map = {int(k): [int(v) for v in targets] for k, targets in fan_out_map.items()}

        missing = [i for i in range(self.in_dim) if i not in self.fan_out_map]
        if missing:
            raise ValueError(
                f"FanOutInputMapper: fan_out_map missing input indices {missing}; "
                f"must cover [0, {self.in_dim})"
            )
        extra = [k for k in self.fan_out_map if k < 0 or k >= self.in_dim]
        if extra:
            raise ValueError(
                f"FanOutInputMapper: fan_out_map has out-of-range keys {extra}; "
                f"keys must be in [0, {self.in_dim})"
            )

        all_targets: list[int] = []
        for i, targets in self.fan_out_map.items():
            for t in targets:
                if t < 0 or t >= self.out_dim:
                    raise ValueError(
                        f"FanOutInputMapper: input {i} target {t} out of range [0, {self.out_dim})"
                    )
                all_targets.append(t)
        if len(set(all_targets)) != len(all_targets):
            seen: set[int] = set()
            dupes: list[int] = []
            for t in all_targets:
                if t in seen and t not in dupes:
                    dupes.append(t)
                seen.add(t)
            raise ValueError(
                f"FanOutInputMapper: duplicate target nodes {dupes}; "
                f"each hidden node can be written by at most one input"
            )

        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])

        flat_targets: list[int] = []
        for i in range(self.in_dim):
            flat_targets.extend(self.fan_out_map[i])
        self.register_buffer(
            "_flat_targets",
            torch.tensor(flat_targets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_input_index",
            torch.tensor(
                [i for i in range(self.in_dim) for _ in self.fan_out_map[i]],
                dtype=torch.long,
            ),
            persistent=False,
        )

        K_total = sum(len(v) for v in self.fan_out_map.values())
        self.gain = nn.Parameter(torch.full((K_total,), INIT["gain_scale"]))
        self.bias = nn.Parameter(torch.zeros(K_total))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.size(-1) != self.in_dim:
            raise ValueError(
                f"FanOutInputMapper: input has {u.size(-1)} features, "
                f"expected in_dim={self.in_dim}"
            )
        u_picked = u.index_select(-1, self._input_index.to(u.device))
        per_target = self.x_max * torch.tanh(u_picked * self.gain + self.bias)
        x = u.new_zeros(*u.shape[:-1], self.out_dim)
        x.index_copy_(-1, self._flat_targets.to(u.device), per_target.to(dtype=x.dtype))
        return x


class ProjectedSparseInputMapper(nn.Module):
    """Learned-projection sparse input writer.

    Maps ``in_dim`` input features through a single ``nn.Linear`` to
    ``len(write_idx)`` projected channels, then scatters those channels to
    the designated hidden-node targets. Each input feature contributes to
    ALL target channels via the learned projection weights, so
    ``len(write_idx) >= in_dim`` is required for the projection to be a
    proper lift (not a compression).

        x_{write_idx[k]}(0) = x_max * tanh(Linear(u))[k]
        x_j(0) = 0  for j not in write_idx

    Hidden nodes NOT in ``write_idx`` are left at 0 in the mapper's output;
    the caller (KirchhoffNetWithIO) is responsible for assembling the full
    state vector.

    Parameter count = in_dim * len(write_idx) + len(write_idx). For
    in_dim=2, write_idx=[0, 1, 3] (out_dim=6): Linear(2, 3) = 6+3 = 9 params.

    Validation:
        - len(write_idx) must be >= in_dim (otherwise raise ValueError).
        - All write_idx entries must be in [0, out_dim).
        - write_idx entries must be unique.

    Raises:
        ValueError: on any of the above violations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        write_idx: list[int],
        x_max: float | None = None,
    ) -> None:
        super().__init__()
        if len(write_idx) < in_dim:
            raise ValueError(
                f"ProjectedSparseInputMapper: len(write_idx)={len(write_idx)} must be "
                f">= in_dim={in_dim} (sparse-proj-write/SPW1.5)"
            )
        if any(i < 0 or i >= out_dim for i in write_idx):
            raise ValueError(
                f"ProjectedSparseInputMapper: write_idx entries must be in "
                f"[0, out_dim)={out_dim}, got {write_idx}"
            )
        if len(set(write_idx)) != len(write_idx):
            raise ValueError(
                f"ProjectedSparseInputMapper: write_idx entries must be unique, "
                f"got {write_idx}"
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.write_idx = list(write_idx)
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.proj = nn.Linear(in_dim, len(write_idx), bias=True)
        _init_linear_small(self.proj, gain_scale=INIT["gain_scale"])
        self.register_buffer(
            "_write_index",
            torch.tensor(self.write_idx, dtype=torch.long),
            persistent=False,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.size(-1) != self.in_dim:
            raise ValueError(
                f"ProjectedSparseInputMapper: input has {u.size(-1)} features, "
                f"expected in_dim={self.in_dim}"
            )
        projected = self.x_max * torch.tanh(self.proj(u))
        x = u.new_zeros(*u.shape[:-1], self.out_dim)
        x.index_copy_(-1, self._write_index.to(u.device), projected.to(dtype=x.dtype))
        return x


class OutputMapper(nn.Module):
    """y_hat = Linear(x_final). Linear only, no activation.

    When ``read_idx`` is provided, the mapper applies a learnable linear
    projection of size ``len(read_idx) -> out_dim`` and an explicit Index
    gather over the full state, so the parameter count depends on
    ``len(read_idx)`` rather than the full state width. The caller's
    full-state ``x_final`` is sliced by ``read_idx`` (a list of full-state
    indices) before the linear layer runs.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        read_idx: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.out_dim = int(out_dim)
        if read_idx is None:
            self.read_idx: list[int] | None = None
            in_features = self.node_dim
        else:
            if any(i < 0 or i >= self.node_dim for i in read_idx):
                raise ValueError(
                    f"OutputMapper: read_idx entries must be in [0, node_dim)={self.node_dim}, "
                    f"got {read_idx}"
                )
            self.read_idx = list(read_idx)
            in_features = len(self.read_idx)
            self.register_buffer(
                "_read_index",
                torch.tensor(self.read_idx, dtype=torch.long),
                persistent=False,
            )
        self.proj = nn.Linear(in_features, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            with torch.no_grad():
                self.proj.bias.zero_()

    def forward(self, x_final: torch.Tensor) -> torch.Tensor:
        if self.read_idx is None:
            return self.proj(x_final)
        if x_final.size(-1) != self.node_dim:
            raise ValueError(
                f"OutputMapper: input last dim={x_final.size(-1)}, "
                f"expected node_dim={self.node_dim}"
            )
        gathered = x_final.index_select(-1, self._read_index.to(x_final.device))
        return self.proj(gathered)


class GroupedOutputMapper(nn.Module):
    """Per-target Linear heads reading from disjoint state-node windows.

    Replaces the monolithic ``Linear(read_dim, num_targets)`` OutputMapper with
    ``num_targets`` independent ``Linear(nodes_per_target, 1)`` layers. Target
    ``i`` reads from ``state[..., offset + i*nodes_per_target :
                              offset + (i+1)*nodes_per_target]``.

    Unlike ``OutputMapper``, this mapper ignores any ``read_idx`` list passed
    to ``KirchhoffNetWithIO`` — it does its own contiguous slicing on the full
    state vector. The caller's full state must be at least
    ``offset + num_targets * nodes_per_target`` wide; the caller should set
    ``read_idx = list(range(node_dim))`` on ``KirchhoffNetWithIO`` so the full
    state is forwarded.
    """

    def __init__(
        self,
        nodes_per_target: int,
        num_targets: int,
        node_dim: int,
        offset: int = 0,
    ) -> None:
        super().__init__()
        if nodes_per_target <= 0:
            raise ValueError(
                f"GroupedOutputMapper: nodes_per_target must be > 0, got {nodes_per_target}"
            )
        if num_targets <= 0:
            raise ValueError(
                f"GroupedOutputMapper: num_targets must be > 0, got {num_targets}"
            )
        if offset < 0:
            raise ValueError(
                f"GroupedOutputMapper: offset must be >= 0, got {offset}"
            )
        self.nodes_per_target = int(nodes_per_target)
        self.num_targets = int(num_targets)
        self.offset = int(offset)
        self.node_dim = int(node_dim)
        required = self.offset + self.num_targets * self.nodes_per_target
        if self.node_dim < required:
            raise ValueError(
                f"GroupedOutputMapper: node_dim must be >= offset + num_targets * "
                f"nodes_per_target = {required} (got node_dim={self.node_dim}, "
                f"offset={self.offset}, num_targets={self.num_targets}, "
                f"nodes_per_target={self.nodes_per_target})"
            )
        self.heads = nn.ModuleList([
            nn.Linear(self.nodes_per_target, 1, bias=True) for _ in range(self.num_targets)
        ])
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                with torch.no_grad():
                    head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        end = self.offset + self.num_targets * self.nodes_per_target
        if x.size(-1) < end:
            raise RuntimeError(
                f"GroupedOutputMapper needs state dim >= {end} "
                f"(offset={self.offset}, num_targets={self.num_targets}, "
                f"nodes_per_target={self.nodes_per_target}); got {x.size(-1)}. "
                f"Increase --num-hidden/--grid-size."
            )
        out = []
        for i, head in enumerate(self.heads):
            start = self.offset + i * self.nodes_per_target
            out.append(head(x[..., start:start + self.nodes_per_target]))
        return torch.cat(out, dim=-1)
