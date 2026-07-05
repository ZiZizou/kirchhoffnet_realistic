"""DifferentialStage: sparse COO graph + KCL scatter-add + Heun integration.

Per-node dynamics:
    C_eff * dx_j/dt = sum_in I_edge - sum_out I_edge - leak_j * x_j - clip_j(x_j)

Edge currents are computed by a per-edge device library.
Heun integration (predictor-corrector, 2nd order) is used for fixed-step
BPTT. The stage returns both the final state and the full trajectory so
that regularizers can be evaluated along the path.

Deep Equilibrium (DEQ) forward path (deq-core-prototype plan): the stage
exposes ``forward_equilibrium`` which solves ``rhs(x*)=0`` via the
:mod:`deq_solver` adapter and returns implicit gradients. Selected by
passing ``solver='deq'`` to ``forward``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DEQ,
    DRIVE,
    INIT,
    PHYS,
    SOLVER,
)
from cell_library import (
    FreeTanhLibrary,
    RealisticTanhLibrary,
    RealisticTanhUpgradeLibrary,
    SimpleEdgeLibrary,
)


__all__ = ["DifferentialStage"]


class DifferentialStage(nn.Module):
    """A single stage of the reduced differential KirchhoffNet.

    Args:
        num_nodes: Number of differential nodes in this stage's internal state.
        src: List of source node ids (length E).
        dst: List of destination node ids (length E).
        cell_lib: Edge device library used to compute edge currents.
        c_eff: Effective node capacitance (default from config).
        x_max: Differential rail limit (default from config).
        clip_current: Soft rail clip current magnitude (default from config).
        clip_softness: Soft rail clip transition width (default from config).
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
        cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary,
        c_eff: float | None = None,
        x_max: float | None = None,
        clip_current: float | None = None,
        clip_softness: float | None = None,
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

        self.raw_leak = nn.Parameter(torch.full((num_nodes,), float(INIT["raw_leak_init"])))

        # Minimum effective leak (deq-core-prototype plan). Defaults to 0.0 so
        # the Heun path is byte-for-byte unchanged. Under DEQ this is set to a
        # positive value (config DEQ['leak_floor']) via :meth:`set_leak_floor`
        # to keep the fixed-point map contractive (diagonal damping).
        self.leak_floor = 0.0

        # Gate parameters for complexity-regularized pruning (CP-1, CP-2).
        # z_e = sigmoid(z_logits) is the edge gate: multiplies the edge current.
        # u_j = sigmoid(u_logits) is the node gate: gates the node voltage.
        # Initialized to a large positive value so all edges/nodes are active at start.
        z_init = float(INIT.get("z_logit_init", 5.0))
        u_init = float(INIT.get("u_logit_init", 5.0))
        self.z_logits = nn.Parameter(torch.full((len(src),), z_init))
        # DEPRECATED (deprecate-node-gates): u_logits is no longer used in the
        # forward pass (see ``rhs``) or in any regularizer; nodes are pruned
        # only by connectivity. The parameter is retained for backward
        # compatibility with existing checkpoints — the optimizer still has
        # it as a no-op parameter and its state_dict entry persists.
        self.u_logits = nn.Parameter(torch.full((num_nodes,), u_init))

        # Degree budget / top-k competition (degree-budget-topk plan).
        # Each destination (or source) keeps a fraction ``budget_frac`` of
        # its incoming edges open via temperature-scaled softmax
        # renormalization of z_logits scores (per-group k_eff).
        # budget_frac=0 disables the budget entirely (byte-identical rhs).
        # budget_axis: "dst" (per-destination), "src" (per-source), "both".
        self.budget_frac: float = 0.0
        self.budget_temperature: float = 1.0
        self.budget_axis: str = "dst"
        self.budget_enabled: bool = False

    def num_edges(self) -> int:
        return int(self.src.numel())

    def set_leak_floor(self, leak_floor: float) -> None:
        """Set the minimum effective leak per node.

        Used by the DEQ solver path to enforce a positive diagonal damping so
        the fixed-point map Phi(x)=x+dt*rhs(x) is contractive. Has no effect
        on the Heun path beyond the explicit addend.
        """
        self.leak_floor = float(leak_floor)

    def set_budget_frac(self, frac: float, temperature: float) -> None:
        """Set the degree-budget fraction for this stage.

        Each destination (or source) node keeps a fraction ``frac`` of its
        incoming edges open via temperature-scaled softmax renormalization of
        z_logits scores. The effective per-group budget is computed as
        ``k_eff = max(1, round(count * frac))`` where ``count`` is the
        number of incident edges for that group.

        - ``frac <= 0`` disables the budget entirely (no overhead in
          ``rhs``). Used in Phase C retrain.
        - ``frac >= 1.0`` means no restriction (every group keeps all its
          incident edges; budget_gate = 1.0).
        - ``0 < frac < 1`` activates per-group competition proportional to
          each group's size — nodes with many incoming edges (e.g. proj
          nodes with 25) keep the same fraction as nodes with few (e.g.
          edge hidden with 4), unlike the prior absolute-``k`` mechanism
          which over-pruned high-degree nodes.

        ``temperature`` controls sharpness of competition (smaller = sharper,
        approaches hard top-``k_eff``).

        The budget gate is layered on top of the existing sigmoid gate:
        ``edge_gate = sigmoid(z_logits) * budget_gate``.

        Called once per epoch by the training loop. Captured by attribute
        (not closure) so DEQ IFT re-evaluates with the same values.
        """
        self.budget_frac = float(frac)
        self.budget_temperature = float(temperature)
        self.budget_enabled = (self.budget_frac > 0.0)
        if self.budget_frac < 0.0 or self.budget_frac > 1.0:
            warnings.warn(
                f"set_budget_frac: frac={self.budget_frac} is outside [0, 1]. "
                f"frac<0 disables budget; frac>1 is a no-op (all ones).",
            )

    def _effective_leak(self, num_nodes: int | None = None,
                        leak_floor: float | None = None) -> torch.Tensor:
        """Return the per-node effective leak (leak_floor + softplus(raw_leak))."""
        if num_nodes is None:
            num_nodes = self.num_nodes
        lf = self.leak_floor if leak_floor is None else float(leak_floor)
        base = F.softplus(self.raw_leak)
        if lf == 0.0:
            return base
        return lf + base

    def _compute_budget_gate(self) -> torch.Tensor:
        """Compute the per-destination (or per-source) competitive budget gate.

        For each group node (destination by default), gather the indices of
        incident edges and apply temperature-scaled softmax over their
        z_logits scores. The effective per-group budget is
        ``k_eff = max(1, round(count * frac))`` (a fraction of the group's
        actual incident edge count), so every node type receives a uniform
        proportion of its incoming connections regardless of absolute
        degree. The softmax is scaled to a total budget of ``k_eff`` and
        clamped per-edge to [0, 1]. Groups with ``count <= k_eff`` incident
        edges receive an all-ones mask (no competition needed).

        Fully differentiable (C-infinity) so it is compatible with DEQ
        implicit differentiation. No STE, no hard threshold.

        Returns a tensor of shape ``[E]`` with values in ``[0, 1]``.

        For ``budget_axis="both"`` the per-destination and per-source masks
        are multiplied. Empty groups (no edges) produce a 1.0 contribution
        that does not affect the product.
        """
        if not self.budget_enabled or self.budget_frac <= 0.0:
            return torch.ones(
                self.z_logits.shape, device=self.z_logits.device,
                dtype=self.z_logits.dtype,
            )
        scores = self.z_logits
        frac = float(self.budget_frac)
        T = float(self.budget_temperature)
        if T <= 0.0:
            T = 1e-6  # avoid div-by-zero; effectively hard

        gate = torch.ones(
            scores.shape, device=scores.device, dtype=scores.dtype,
        )

        if self.budget_axis in ("dst", "both"):
            gate = gate * self._budget_group_mask(scores, self.dst, frac, T)

        if self.budget_axis in ("src", "both"):
            gate = gate * self._budget_group_mask(scores, self.src, frac, T)

        return gate

    def _budget_group_mask(
        self,
        scores: torch.Tensor,
        group: torch.Tensor,
        frac: float,
        T: float,
    ) -> torch.Tensor:
        """Build a [E] gate for a single axis (dst or src).

        For each unique group value, compute the per-group effective budget
        ``k_eff = max(1, round(count * frac))`` and the per-edge gate
        ``clamp(softmax(scores / T) * k_eff, max=1.0)``. Groups with
        ``count <= k_eff`` incident edges are all 1.0 (no competition).

        Fully vectorized via scatter operations (no Python loop) for
        performance under torch.compile and large graphs.
        """
        n_groups = self.num_nodes  # group is always self.src/self.dst, max = N-1
        logits = scores.float() / T

        # Per-group max for numerical stability (like F.softmax internally)
        group_max = torch.full(
            (n_groups,), -torch.inf, device=logits.device, dtype=logits.dtype,
        )
        group_max.scatter_reduce_(
            0, group.long().contiguous(), logits.contiguous(),
            reduce='amax', include_self=True,
        )
        shifted = logits - group_max[group]

        exp_shifted = torch.exp(shifted)  # [E], stable

        # Per-group denominator
        group_sum = torch.zeros(
            n_groups, device=logits.device, dtype=logits.dtype,
        )
        group_sum.index_add_(0, group.long(), exp_shifted)

        softmaxed = exp_shifted / group_sum[group].clamp(min=1e-30)

        # Per-group edge count
        count = torch.zeros(
            n_groups, dtype=torch.long, device=logits.device,
        )
        count.index_add_(
            0, group.long(),
            torch.ones_like(group, dtype=torch.long),
        )

        # Per-group effective budget: max(1, round(count * frac))
        # frac is in [0, 1] so count*frac is in [0, count].
        # max(1, ...) ensures isolated single-edge groups still get a
        # competitive budget of 1 even when frac=0.
        k_per_group = torch.clamp(
            (count.float() * frac).round(), min=1.0,
        ).long()
        k_eff = k_per_group[group].float()  # [E]

        needs_budget = count[group].float() > k_eff
        gate = torch.where(
            needs_budget,
            torch.clamp(softmaxed * k_eff, max=1.0),
            torch.ones_like(softmaxed),
        )
        return gate

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

    def rhs(self, x: torch.Tensor,
            x_drive: torch.Tensor | None = None, drive_scale: float = 0.0,
            leak_floor: float | None = None) -> torch.Tensor:
        """Compute dx/dt at state x. x: [batch, num_nodes].

        Gate application:
        - Edge gate: i_edge *= sigmoid(z_logits) — multiplies the edge current
          after cell-library evaluation. When z_e -> 0 the edge contributes
          zero current.
        """
        x_src = x[:, self.src]
        x_dst = x[:, self.dst]

        i_edge = self.cell_lib(
            x_src=x_src,
            x_dst=x_dst,
            x_max=self.x_max,
        )

        # Edge gate: multiply each edge's current by its gate.
        edge_mask = torch.sigmoid(self.z_logits)  # [E]
        # Degree budget / top-k competition (degree-budget-topk plan).
        # Budget gate is layered on top of the sigmoid gate: independent
        # per-edge gate * competitive per-destination (or per-source) mask.
        # When budget is disabled (budget_frac=0) the budget gate is all-ones
        # and this multiplication is a no-op.
        if self.budget_enabled:
            budget_gate = self._compute_budget_gate()  # [E]
            edge_mask = edge_mask * budget_gate
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

        leak = self._effective_leak(leak_floor=leak_floor).unsqueeze(0)  # [1, N]
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
        t_span: float | None = None,
        num_steps: int | None = None,
        store_trajectory: bool = True,
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
        solver: str = "heun",
        deq_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Integrate stage with fixed-step Heun or solve to fixed point.

        Parameters
        ----------
        solver : str
            ``"heun"`` (default) uses the 2nd-order Heun predictor-corrector
            over ``num_steps``. ``"deq"`` solves ``rhs(x*) = 0`` via
            :func:`deq_solver.solve_equilibrium` and returns implicit gradients.
        deq_cfg : dict or None
            Optional overrides for the DEQ solver. ``None`` uses defaults from
            ``config.DEQ``. Recognized keys: ``backend``, ``f_solver``,
            ``b_solver``, ``f_max_iter``, ``f_tol``, ``b_max_iter``,
            ``anderson_m``, ``deq_step``, ``leak_floor``.

        Returns
        -------
        x_final : torch.Tensor
            Stage output state.
        traj : torch.Tensor or None
            ``[batch, num_nodes, num_steps+1]`` for the Heun path; ``None`` for
            the DEQ path (no trajectory at equilibrium).
        """
        if solver == "heun":
            self.last_deq_info = None
            return self._forward_heun(
                x0=x0, t_span=t_span, num_steps=num_steps,
                store_trajectory=store_trajectory,
                x_drive=x_drive, drive_scale=drive_scale,
            )
        if solver == "deq":
            x_star, _info = self.forward_equilibrium(
                x0=x0,
                x_drive=x_drive, drive_scale=drive_scale,
                deq_cfg=deq_cfg,
            )
            self.last_deq_info = dict(_info)
            traj = x_star.unsqueeze(-1) if store_trajectory else None
            return x_star, traj
        raise ValueError(f"DifferentialStage.forward: unknown solver={solver!r}")

    def _forward_heun(
        self,
        x0: torch.Tensor,
        t_span: float | None,
        num_steps: int | None,
        store_trajectory: bool,
        x_drive: torch.Tensor | None,
        drive_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_span = float(t_span if t_span is not None else SOLVER["t_span"])
        num_steps = int(num_steps if num_steps is not None else SOLVER["num_steps"])
        dt = t_span / float(num_steps)

        x = x0
        traj_chunks = [x] if store_trajectory else None

        for _ in range(num_steps):
            k1 = self.rhs(x, x_drive=x_drive, drive_scale=drive_scale)
            x_pred = x + dt * k1
            k2 = self.rhs(x_pred, x_drive=x_drive, drive_scale=drive_scale)
            x = x + 0.5 * dt * (k1 + k2)
            if store_trajectory:
                traj_chunks.append(x)

        traj = torch.stack(traj_chunks, dim=2) if store_trajectory else None
        return x, traj

    def forward_equilibrium(
        self,
        x0: torch.Tensor,
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
        deq_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Solve ``rhs(x*) = 0`` starting from ``x0`` (Deep Equilibrium).

        The damped fixed-point map is ``Phi(x) = x + dt * rhs(x)`` where
        ``dt = deq_cfg['deq_step']`` (defaulting to ``config.DEQ['deq_step']``).
        Returns ``(x_star, info)``. The DEQ path applies a positive
        ``leak_floor`` so the fixed-point map has positive diagonal damping
        (contractivity). The ``leak_floor`` value is captured by the ``phi``
        closure so the backward pass (re-evaluated by torchdeq's IFT) uses
        the same leak_floor as the forward solve. Solver runs in fp32;
        autocast is disabled by the solver adapter.
        """
        from deq_solver import solve_equilibrium

        cfg = dict(DEQ)
        if deq_cfg:
            cfg.update(deq_cfg)
        lf = float(cfg.get("leak_floor", 0.0))
        dt = float(cfg.get("deq_step", 0.1))

        self.set_leak_floor(lf)
        try:
            def phi(x):
                return x + dt * self.rhs(x,
                                        x_drive=x_drive, drive_scale=drive_scale,
                                        leak_floor=lf)

            x_star, info = solve_equilibrium(phi, x0, cfg)
            self.last_deq_info = {
                "nstep": info.get("nstep"),
                "rel_residual": info.get("rel_residual"),
                "deq_step": dt,
                "leak_floor": lf,
            }
            # Cast to the stage's parameter dtype so AMP/GradScaler and downstream
            # regularizers behave like the Heun path.
            param_dtype = next(self.parameters()).dtype
            if x_star.dtype != param_dtype:
                x_star = x_star.to(dtype=param_dtype)
            return x_star, info
        finally:
            # Restore leak floor to 0.0 even if the solver fails so later
            # Heun/validation calls do not inherit DEQ damping by accident.
            self.set_leak_floor(0.0)

    def edge_gates(self) -> torch.Tensor:
        """Return edge gate values z_e = σ(z_logits), shape [E]."""
        return torch.sigmoid(self.z_logits)

    def node_gates(self) -> torch.Tensor:
        """Return node gate values u_j = σ(u_logits), shape [N].

        DEPRECATED (deprecate-node-gates): node gates are no longer used
        in the forward pass or in pruning. The values are vestigial and
        will be constant (sigmoid of the un-trained ``u_logits`` parameter)
        in practice. Returns an all-ones tensor (or, if you want the raw
        sigmoid value, call :func:`torch.sigmoid` on ``self.u_logits``
        directly) so that any caller that accidentally uses this method
        will not corrupt the dynamics.
        """
        import warnings as _warnings
        _warnings.warn(
            "DifferentialStage.node_gates() is deprecated (deprecate-node-gates); "
            "node gates are no longer used in the forward pass or pruning. "
            "Returns an all-ones tensor.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.ones(self.num_nodes, device=self.u_logits.device,
                           dtype=self.u_logits.dtype)

    def active_edge_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask of edges that survive pruning at the given threshold."""
        return self.edge_gates() > threshold

    def active_node_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask of nodes that survive pruning at the given threshold.

        DEPRECATED (deprecate-node-gates): always returns an all-True
        tensor. Node pruning is now connectivity-only — see
        ``topology.prune_stage`` dead-island purge and the I/O
        connectivity backstop for the only mechanisms that remove nodes.
        """
        import warnings as _warnings
        _warnings.warn(
            "DifferentialStage.active_node_mask() is deprecated "
            "(deprecate-node-gates); node pruning is connectivity-only. "
            "Returns an all-True tensor.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.ones(self.num_nodes, dtype=torch.bool,
                           device=self.u_logits.device)

    def parameter_breakdown(self) -> dict:
        """Return parameter counts including gate parameters (for diagnostics)."""
        if isinstance(self.cell_lib, SimpleEdgeLibrary):
            device_n = int(self.cell_lib.param.numel())
        elif isinstance(self.cell_lib, RealisticTanhLibrary):
            device_n = int(self.cell_lib.alpha_raw.numel())
            if hasattr(self.cell_lib, "bias_raw"):
                device_n += int(self.cell_lib.bias_raw.numel())
        elif isinstance(self.cell_lib, RealisticTanhUpgradeLibrary):
            device_n = (
                int(self.cell_lib.alpha_raw.numel())
                + int(self.cell_lib.gm_raw.numel())
                + int(self.cell_lib.isat_raw.numel())
            )
            if hasattr(self.cell_lib, "bias_raw"):
                device_n += int(self.cell_lib.bias_raw.numel())
        elif isinstance(self.cell_lib, FreeTanhLibrary):
            device_n = (
                int(self.cell_lib.a_raw.numel())
                + int(self.cell_lib.b_raw.numel())
                + int(self.cell_lib.s_raw.numel())
                + int(self.cell_lib.gm_raw.numel())
                + int(self.cell_lib.isat_raw.numel())
            )
            if hasattr(self.cell_lib, "theta_raw"):
                device_n += int(self.cell_lib.theta_raw.numel())
        else:
            device_n = 0
        return {
            "raw_leak": int(self.raw_leak.numel()),
            "z_logits": int(self.z_logits.numel()),
            "u_logits": int(self.u_logits.numel()),
            "device_param": device_n,
            "total": (
                int(self.raw_leak.numel())
                + int(self.z_logits.numel())
                + int(self.u_logits.numel())
                + device_n
            ),
        }
