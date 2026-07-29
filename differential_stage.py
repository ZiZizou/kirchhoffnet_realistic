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
    REF,
    SOLVER,
    VCA,
)
from cell_library import (
    AntiParallelFreeTanhLibrary,
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
        leak_mode: ``"programmable"`` (default) creates a learnable per-node
            ``raw_leak`` parameter. ``"non-programmable"`` uses a fixed scalar
            ``leak_constant`` (see ``leak_constant``) for all nodes, saving
            parameters and eliminating leak gradients.
        leak_constant: Fixed leak value used when ``leak_mode="non-programmable"``.
            When ``None``, defaults to ``config.INIT["leak_constant"]``
            (0.0486, matching ``softplus(raw_leak_init)``). Ignored when
            ``leak_mode="programmable"``.
        freeze_read: When ``True``, edge currents (cell_lib output, edge gate,
            budget gate, and KCL scatter-add) are computed **once** at the
            start of the stage from the initial state ``x0`` and held constant
            throughout all Heun / DEQ sub-iterations. Leak, clip, and drive
            current still read the evolving ``x`` at each step. Default
            ``False`` (standard behavior: read and write at the same time).
        boundary_src: List of input-terminal indices for sparse OTA edges
            from fixed-voltage boundary terminals into the dynamic fabric.
            Length must equal ``len(boundary_dst)``. ``None`` (default)
            disables boundary edges for this stage.
        boundary_dst: List of target dynamic-node indices for the boundary
            OTA edges (same length as ``boundary_src``). Indices are in
            the compact 0..num_nodes-1 coordinate space, matching
            ``write_idx``.
        boundary_cell_lib: Cell library instance used to compute boundary
            edge currents ``I_OTA(u_i, x_j)``. Must match the cell type
            of the core ``cell_lib`` and be sized for ``len(boundary_src)``
            edges. Must be provided when boundary edges are configured.
        enable_ref_edges: When ``True``, every node gets one OTA edge to a
            global per-stage learnable reference voltage ``Vref`` (scalar,
            constrained to ``[0, x_max]`` via ``sigmoid(raw_vref) * x_max``).
            Vref is held constant during the ODE integration of a single
            stage (no current sourced/sinked into Vref; it's an ideal voltage
            source). The reference edge injects ``I_OTA(Vref, x_j)`` into
            node ``j`` only. Implemented via a separate cell library
            (``ref_cell_lib``) sized to ``num_nodes`` so each node has its
            own programmable OTA parameters. Default ``False``.
        ref_cell_lib: Cell library instance used to compute reference edge
            currents ``I_OTA(Vref, x_j)``. Must match the cell type of
            the core ``cell_lib`` and be sized for ``num_nodes`` edges. Required
            when ``enable_ref_edges=True``.
        output_ode_src: List of source node indices (compact 0..num_nodes-1
            coordinate space) for the temporal-readout OTA edges. These edges
            inject current from a hidden (or projection) node into an output
            ODE accumulator node. The source is read-only (its voltage drives
            the OTA current but no current is drained from the source); the
            output ODE node is the writable destination. Length must equal
            ``len(output_ode_dst)`` when provided. ``None`` (default)
            disables temporal-readout edges for this stage.
        output_ode_dst: List of destination node indices (compact 0..num_nodes-1
            coordinate space, same length as ``output_ode_src``) for the
            temporal-readout OTA edges. Indices typically lie in the
            output-ode accumulator region (e.g., ``[core_count, num_nodes)``),
            but any valid node index is permitted so a hidden→hidden
            temporal-readout edge is also expressible.
        output_ode_cell_lib: Cell library instance used to compute temporal
            readout edge currents ``I_OTA(x_src, x_dst)``. Must match the
            cell type of the core ``cell_lib`` and be sized for
            ``len(output_ode_src)`` edges. Required when
            ``output_ode_src``/``output_ode_dst`` are provided.
    """

    def __init__(
        self,
        num_nodes: int,
        src: list[int],
        dst: list[int],
        cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary,
        c_eff: float | None = None,
        x_max: float | None = None,
        clip_current: float | None = None,
        clip_softness: float | None = None,
        write_idx: list[int] | None = None,
        drive_isat: float | None = None,
        leak_mode: str = "programmable",
        leak_constant: float | None = None,
        read_only_source: bool = False,
        freeze_read: bool = False,
        boundary_src: list[int] | None = None,
        boundary_dst: list[int] | None = None,
        boundary_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
        enable_ref_edges: bool = False,
        ref_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
output_ode_src: list[int] | None = None,
        output_ode_dst: list[int] | None = None,
        output_ode_cell_lib: SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary | AntiParallelFreeTanhLibrary | None = None,
        vca_enabled: bool = False,
        vca_rank: int = 2,
        vca_in_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cell_lib = cell_lib
        self.read_only_source = read_only_source
        self.freeze_read = bool(freeze_read)
        if leak_mode not in ("programmable", "non-programmable"):
            raise ValueError(f"leak_mode must be 'programmable' or 'non-programmable', got {leak_mode!r}")
        self.leak_mode = leak_mode
        self.vca_enabled = bool(vca_enabled)
        self.vca_rank = int(vca_rank) if vca_enabled else int(vca_rank)
        self._vca_in_dim = int(vca_in_dim)
        if self.vca_enabled and self.vca_rank < VCA["min_rank"]:
            raise ValueError(
                f"vca_rank must be >= {VCA['min_rank']}, got {self.vca_rank}"
            )

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

        if self.leak_mode == "programmable":
            self.raw_leak = nn.Parameter(torch.full((num_nodes,), float(INIT["raw_leak_init"])))
        else:
            self.leak_constant = float(leak_constant if leak_constant is not None else INIT["leak_constant"])

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

        # Parallel resistive shunt (FreeTanhLibrary): if the cell library
        # exposes a ``resistive_current(x_src, x_dst)`` method, route an
        # additional current ``G * (Vsrc - Vdest)`` per edge from evolving
        # voltages, bypassing ``freeze_read``. Cached here so the dispatch
        # in ``rhs`` is a fast attribute check instead of a hasattr scan.
        self._has_resistive = hasattr(cell_lib, "resistive_current")

        # Boundary-terminal OTA edges (boundary-fan-out plan).
        # Optional sparse programmable edges from fixed-voltage input
        # terminals (carried in ``u``) into dynamic-node targets. The
        # boundary terminals are ideal voltage sources: current flows
        # only into the destination node, the terminal voltage is never
        # drained. Gated by a separate ``boundary_z_logits`` parameter
        # so they can be pruned/trained independently of the core edges.
        if boundary_src is not None or boundary_dst is not None:
            if boundary_src is None or boundary_dst is None:
                raise ValueError(
                    "DifferentialStage: boundary_src and boundary_dst must "
                    "be provided together (got one without the other)"
                )
            if len(boundary_src) != len(boundary_dst):
                raise ValueError(
                    f"DifferentialStage: boundary_src/dst length mismatch: "
                    f"{len(boundary_src)} vs {len(boundary_dst)}"
                )
            if any(s < 0 or s >= self.num_nodes for s in boundary_dst):
                raise ValueError(
                    f"DifferentialStage: boundary_dst entries must be in "
                    f"[0, {self.num_nodes}), got {boundary_dst}"
                )
            if any(s < 0 for s in boundary_src):
                raise ValueError(
                    f"DifferentialStage: boundary_src entries must be "
                    f"non-negative, got {boundary_src}"
                )
            if boundary_cell_lib is None:
                raise ValueError(
                    "DifferentialStage: boundary_cell_lib is required when "
                    "boundary_src/boundary_dst are provided"
                )
            self.register_buffer(
                "boundary_src", torch.tensor(boundary_src, dtype=torch.long),
            )
            self.register_buffer(
                "boundary_dst", torch.tensor(boundary_dst, dtype=torch.long),
            )
            self.boundary_cell_lib = boundary_cell_lib
            self.boundary_z_logits = nn.Parameter(
                torch.full((len(boundary_src),), z_init),
            )
            self._has_boundary = True
        else:
            self.register_buffer(
                "boundary_src", torch.empty(0, dtype=torch.long),
            )
            self.register_buffer(
                "boundary_dst", torch.empty(0, dtype=torch.long),
            )
            self.boundary_cell_lib = None
            self.boundary_z_logits = None
            self._has_boundary = False

        # Reference edges (unary nonlinearities via OTA-to-Vref plan).
        # Every node gets one OTA edge to a global per-stage learnable Vref
        # voltage constrained to [0, x_max]. Vref is held constant during a
        # single stage's ODE integration (ideal voltage source: no current
        # drawn from the Vref rail). Each reference edge has its own OTA
        # cell in ``ref_cell_lib`` (sized to num_nodes) with independent
        # per-node gm/Isat/theta/etc., and its own gate ``ref_z_logits``.
        if enable_ref_edges:
            if ref_cell_lib is None:
                raise ValueError(
                    "DifferentialStage: ref_cell_lib is required when "
                    "enable_ref_edges=True"
                )
            self.ref_cell_lib = ref_cell_lib
            self.raw_vref = nn.Parameter(
                torch.tensor([float(REF["raw_vref_init"])], dtype=torch.float32)
            )
            self.ref_z_logits = nn.Parameter(
                torch.full((num_nodes,), z_init),
            )
            self.register_buffer(
                "ref_dst", torch.arange(num_nodes, dtype=torch.long),
            )
            self._has_ref = True
        else:
            self.register_buffer(
                "ref_dst", torch.empty(0, dtype=torch.long),
            )
            self.ref_cell_lib = None
            self.raw_vref = None
            self.ref_z_logits = None
            self._has_ref = False

        # Temporal-readout OTA edges (temporal-readout plan).
        # Sparse programmable edges from hidden/projection nodes (read-only
        # source) into the output ODE accumulator nodes (writable destination).
        # The output ODE nodes are the last ``output_ode_count`` entries of the
        # state vector and are part of the ODE dynamics (they receive leak,
        # clip, and the OTA current injected here). The source node is never
        # drained — only the destination receives current — matching the
        # boundary-fan-out pattern.
        if output_ode_src is not None or output_ode_dst is not None:
            if output_ode_src is None or output_ode_dst is None:
                raise ValueError(
                    "DifferentialStage: output_ode_src and output_ode_dst "
                    "must be provided together (got one without the other)"
                )
            if len(output_ode_src) != len(output_ode_dst):
                raise ValueError(
                    f"DifferentialStage: output_ode_src/dst length mismatch: "
                    f"{len(output_ode_src)} vs {len(output_ode_dst)}"
                )
            if any(
                s < 0 or s >= self.num_nodes or d < 0 or d >= self.num_nodes
                for s, d in zip(output_ode_src, output_ode_dst)
            ):
                raise ValueError(
                    f"DifferentialStage: output_ode_src/dst entries must be "
                    f"in [0, {self.num_nodes}), got src={output_ode_src} "
                    f"dst={output_ode_dst}"
                )
            if any(s == d for s, d in zip(output_ode_src, output_ode_dst)):
                raise ValueError(
                    "DifferentialStage: self-loops are not allowed in "
                    "output_ode edges."
                )
            if output_ode_cell_lib is None:
                raise ValueError(
                    "DifferentialStage: output_ode_cell_lib is required when "
                    "output_ode_src/output_ode_dst are provided"
                )
            self.register_buffer(
                "output_ode_src", torch.tensor(output_ode_src, dtype=torch.long),
            )
            self.register_buffer(
                "output_ode_dst", torch.tensor(output_ode_dst, dtype=torch.long),
            )
            self.output_ode_cell_lib = output_ode_cell_lib
            self.output_ode_z_logits = nn.Parameter(
                torch.full((len(output_ode_src),), z_init),
            )
            self._has_output_ode = True
        else:
            self.register_buffer(
                "output_ode_src", torch.empty(0, dtype=torch.long),
            )
            self.register_buffer(
                "output_ode_dst", torch.empty(0, dtype=torch.long),
            )
            self.output_ode_cell_lib = None
            self.output_ode_z_logits = None
            self._has_output_ode = False

        # Low-rank input-driven VCA (Voltage-Controlled Amplifier) gating.
        # When enabled, builds per-edge embeddings for boundary and
        # temporal-readout edges plus a shared input projection. The VCA
        # gate per gated edge is ``sigma(u^T W v_e)`` where ``W``
        # (in_dim x rank) is the shared projection and ``v_e`` (rank) is
        # the per-edge embedding. ``W[:, 0]`` is initialized to zero
        # (input-independent bus) so the VCA gate starts as a no-op
        # (``sigma(0 * v_e) = 0.5``) and the optimizer can gradually
        # activate the input-dependent modulation. Incompatible with
        # --vca-rank < VCA['min_rank']; requires at least one of
        # boundary or temporal-readout edges.
        if self.vca_enabled:
            n_b = int(self.boundary_src.numel())
            n_r = int(self.output_ode_src.numel())
            if n_b == 0 and n_r == 0:
                raise ValueError(
                    "DifferentialStage: --vca requires boundary_src or "
                    "output_ode_src to be non-empty (VCA only modulates "
                    "unfrozen edges that read the input)."
                )
            if self._vca_in_dim <= 0:
                raise ValueError(
                    "DifferentialStage: vca_in_dim must be > 0 when "
                    "vca_enabled=True"
                )
            init_scale = float(VCA["scale_init"])
            self.vca_W = nn.Parameter(
                torch.zeros(self._vca_in_dim, self.vca_rank)
            )
            with torch.no_grad():
                if self.vca_rank > 1:
                    nn.init.normal_(
                        self.vca_W[:, 1:],
                        std=init_scale,
                    )
                self.vca_W[:, 0].zero_()
            if n_b > 0:
                self.vca_v_boundary = nn.Parameter(
                    torch.empty(n_b, self.vca_rank).normal_(std=init_scale)
                )
            else:
                self.vca_v_boundary = None
            if n_r > 0:
                self.vca_v_readout = nn.Parameter(
                    torch.empty(n_r, self.vca_rank).normal_(std=init_scale)
                )
            else:
                self.vca_v_readout = None
        else:
            self.vca_W = None
            self.vca_v_boundary = None
            self.vca_v_readout = None

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
        """Return the per-node effective leak.
        
        Programmable: ``leak_floor + softplus(raw_leak)`` (per-node).
        Non-programmable: ``leak_floor + leak_constant`` (scalar, same for all nodes).
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        lf = self.leak_floor if leak_floor is None else float(leak_floor)
        if self.leak_mode == "programmable":
            base = F.softplus(self.raw_leak)
            return base if lf == 0.0 else lf + base
        else:
            l = lf + self.leak_constant
            return torch.full((num_nodes,), l, dtype=torch.float32)

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

    def _compute_vca_gate(
        self,
        u: torch.Tensor,
        v_e: torch.Tensor,
    ) -> torch.Tensor:
        """Low-rank VCA gate for one edge set.

        Computes ``sigma( (u @ W) @ v_e.T )`` of shape ``[batch, E]`` where
        ``W`` is the shared input projection ``(in_dim, rank)``, ``v_e``
        is the per-edge embedding ``(E, rank)``, and ``u`` is the input
        feature batch ``(batch, in_dim)``.

        Caller is responsible for ensuring ``vca_enabled=True``,
        ``u`` is not None and has the expected in_dim, and that the
        shared projection ``self.vca_W`` and per-edge ``v_e`` have been
        built (consistent shapes).
        """
        u_proj = u @ self.vca_W              # [batch, rank]
        vca_logits = u_proj @ v_e.T          # [batch, E]
        return torch.sigmoid(vca_logits)

    def rhs(self, x: torch.Tensor,
            u: torch.Tensor | None = None,
            x_drive: torch.Tensor | None = None, drive_scale: float = 0.0,
            leak_floor: float | None = None,
            i_edge_const: torch.Tensor | None = None) -> torch.Tensor:
        """Compute dx/dt at state x. x: [batch, num_nodes].

        Gate application:
        - Edge gate: i_edge *= sigmoid(z_logits) — multiplies the edge current
          after cell-library evaluation. When z_e -> 0 the edge contributes
          zero current.

        When ``i_edge_const`` is provided (the ``freeze_read=True`` path), the
        cell_lib evaluation, edge gate, budget gate, and KCL scatter-add are
        skipped — the provided ``[batch, num_nodes]`` tensor is used directly
        as the KCL contribution. Leak, clip, and drive current are still
        computed from the current ``x``.

        Boundary-terminal OTA edges (boundary-fan-out plan):
        - When ``u`` is provided and ``self._has_boundary``, the boundary
          edges inject ``I_OTA(u[:, boundary_src], x[:, boundary_dst])``
          into the destination nodes only (no source drain — terminals
          are fixed voltages). They are NOT frozen by ``freeze_read``:
          the destination voltage evolves, so the OTA current is
          recomputed every step.

        Reference edges (unary nonlinearities via OTA-to-Vref plan):
        - When ``self._has_ref``, every node gets one OTA edge to a
          global per-stage learnable ``Vref = sigmoid(raw_vref) * x_max``.
          The reference current ``I_OTA(Vref, x_j)`` is injected into
          node ``j`` only (no source drain — Vref is an ideal voltage
          source). Like boundary edges, reference currents are NOT frozen
          by ``freeze_read`` (the destination voltage ``x_j`` evolves).

        Temporal-readout OTA edges (temporal-readout plan):
        - When ``self._has_output_ode``, ``I_OTA(x[output_ode_src[e]],
          x[output_ode_dst[e]])`` is injected into the destination (output
          ODE accumulator) only. The source (hidden/projection) is a
          read-only voltage — no current is drained from it. Like boundary
          and reference edges, the temporal-readout current is NOT frozen
          by ``freeze_read`` (the destination voltage evolves).
        """
        x_src = x[:, self.src]
        x_dst = x[:, self.dst]

        # Edge gate: multiply each edge's current by its gate. Computed once
        # here and reused for both the tanh current and the resistive shunt
        # (when applicable), avoiding redundant sigmoid + budget_gate calls.
        edge_mask = torch.sigmoid(self.z_logits)  # [E]
        # Degree budget / top-k competition (degree-budget-topk plan).
        # Budget gate is layered on top of the sigmoid gate: independent
        # per-edge gate * competitive per-destination (or per-source) mask.
        # When budget is disabled (budget_frac=0) the budget gate is all-ones
        # and this multiplication is a no-op.
        if self.budget_enabled:
            budget_gate = self._compute_budget_gate()  # [E]
            edge_mask = edge_mask * budget_gate

        if i_edge_const is None:
            i_edge = self.cell_lib(
                x_src=x_src,
                x_dst=x_dst,
                x_max=self.x_max,
            )
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
            if not self.read_only_source:
                acc.index_add_(1, self.src, -i_edge_f32)
            acc = acc.to(dtype=x.dtype)
        else:
            # Frozen path: i_edge_const is the precomputed KCL contribution
            # [batch, num_nodes] in x's dtype. The tanh contribution was
            # computed from x0 and held constant. The resistive shunt (if
            # any) is added below from evolving voltages.
            acc = i_edge_const

        # Parallel resistive shunt (FreeTanhLibrary): always uses evolving
        # voltages, bypassing ``freeze_read``. Gated by the same edge_mask /
        # budget_gate as the tanh current so it can be pruned away.
        if self._has_resistive:
            i_res = self.cell_lib.resistive_current(x_src, x_dst)
            i_res = i_res * edge_mask.unsqueeze(0)              # [B, E]
            i_res_f32 = i_res.float()
            # Clone when freeze_read is active so we don't mutate the
            # shared ``i_edge_const`` tensor across rhs calls.
            if i_edge_const is not None:
                acc = acc.clone()
            acc_res = torch.zeros_like(x, dtype=torch.float32)
            acc_res.index_add_(1, self.dst, i_res_f32)
            if not self.read_only_source:
                acc_res.index_add_(1, self.src, -i_res_f32)
            acc = (acc.float() + acc_res).to(dtype=x.dtype)

        # Boundary-terminal OTA edges: I_OTA(u_i, x_j) injected into dst only.
        # Boundary terminals are ideal voltage sources, never drained.
        if self._has_boundary and u is not None and self.boundary_src.numel() > 0:
            u_src = u[:, self.boundary_src]
            x_dst_b = x[:, self.boundary_dst]
            i_boundary = self.boundary_cell_lib(
                x_src=u_src, x_dst=x_dst_b, x_max=self.x_max,
            )
            boundary_mask = torch.sigmoid(self.boundary_z_logits)  # [Eb]
            i_boundary = i_boundary * boundary_mask.unsqueeze(0)   # [B, Eb]
            if self.vca_enabled and self.vca_v_boundary is not None:
                i_boundary = i_boundary * self._compute_vca_gate(
                    u, self.vca_v_boundary,
                )  # [B, Eb]
            i_boundary_f32 = i_boundary.float()
            # Clone when freeze_read is active so we don't mutate the shared
            # ``acc`` (= i_edge_const) tensor.
            if i_edge_const is not None:
                acc = acc.clone()
            acc_b = torch.zeros_like(x, dtype=torch.float32)
            acc_b.index_add_(1, self.boundary_dst, i_boundary_f32)
            # NOTE: no `acc_b.index_add_(1, boundary_src, -i_boundary_f32)` —
            # boundary terminals are fixed voltages, never drained.
            acc = (acc.float() + acc_b).to(dtype=x.dtype)

        # Reference edges (unary nonlinearities via OTA-to-Vref plan).
        # For each node j: I_ref = I_OTA(Vref, x_j), injected into dst only.
        # Vref = sigmoid(raw_vref) * x_max is a per-stage learnable scalar
        # held constant during the ODE integration (no current sourced
        # from or sinked into the Vref rail — it's an ideal voltage source).
        if self._has_ref:
            vref = torch.sigmoid(self.raw_vref) * self.x_max  # [1], in [0, x_max]
            vref_expanded = vref.view(1, 1).expand(x.size(0), self.num_nodes)  # [B, N]
            i_ref = self.ref_cell_lib(
                x_src=vref_expanded, x_dst=x, x_max=self.x_max,
            )  # [B, N]
            ref_mask = torch.sigmoid(self.ref_z_logits)  # [N]
            i_ref = i_ref * ref_mask.unsqueeze(0)  # [B, N]
            i_ref_f32 = i_ref.float()
            if i_edge_const is not None:
                acc = acc.clone()
            acc_ref = torch.zeros_like(x, dtype=torch.float32)
            acc_ref.index_add_(1, self.ref_dst, i_ref_f32)
            # NOTE: no source drain — Vref is an ideal voltage source.
            acc = (acc.float() + acc_ref).to(dtype=x.dtype)

        # Temporal-readout OTA edges (temporal-readout plan).
        # For each edge e: I_out = I_OTA(x[output_ode_src[e]], x[output_ode_dst[e]]).
        # Current is injected into the destination (output ODE accumulator)
        # only; the source (hidden/projection) is read-only and is not
        # drained. The destination voltage evolves through the ODE so the
        # OTA current is recomputed every step (NOT frozen by freeze_read).
        if self._has_output_ode and self.output_ode_src.numel() > 0:
            x_src_o = x[:, self.output_ode_src]  # hidden (read-only)
            x_dst_o = x[:, self.output_ode_dst]  # output ODE (writable)
            i_out = self.output_ode_cell_lib(
                x_src=x_src_o, x_dst=x_dst_o, x_max=self.x_max,
            )
            out_mask = torch.sigmoid(self.output_ode_z_logits)  # [Eo]
            i_out = i_out * out_mask.unsqueeze(0)  # [B, Eo]
            if self.vca_enabled and self.vca_v_readout is not None and u is not None:
                i_out = i_out * self._compute_vca_gate(
                    u, self.vca_v_readout,
                )  # [B, Eo]
            i_out_f32 = i_out.float()
            if i_edge_const is not None:
                acc = acc.clone()
            acc_out = torch.zeros_like(x, dtype=torch.float32)
            acc_out.index_add_(1, self.output_ode_dst, i_out_f32)
            # NOTE: no source drain on output_ode_src — the hidden/projection
            # grid is untouched, only the output accumulator receives current.
            acc = (acc.float() + acc_out).to(dtype=x.dtype)

        leak = self._effective_leak(leak_floor=leak_floor).unsqueeze(0).to(x.device)  # [1, N]
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
        u: torch.Tensor | None = None,
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
                u=u,
            )
        if solver == "deq":
            x_star, _info = self.forward_equilibrium(
                x0=x0,
                x_drive=x_drive, drive_scale=drive_scale,
                deq_cfg=deq_cfg,
                u=u,
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
        u: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_span = float(t_span if t_span is not None else SOLVER["t_span"])
        num_steps = int(num_steps if num_steps is not None else SOLVER["num_steps"])
        dt = t_span / float(num_steps)

        # freeze_read: precompute edge currents (cell_lib + edge gate + budget
        # gate + KCL scatter-add) once from x0 and hold them constant across
        # all sub-steps. Leak, clip, drive, and boundary-edge currents still
        # read the current x (boundary terminals are fixed but the dynamic
        # target node voltage evolves, so the OTA current is per-step).
        # For cell libraries with a parallel resistive shunt, use
        # ``forward_tanh`` here so the resistive term stays dynamic in ``rhs``
        # (the resistive current is added per-step from evolving voltages).
        i_edge_const = None
        if self.freeze_read:
            x_src0 = x0[:, self.src]
            x_dst0 = x0[:, self.dst]
            if self._has_resistive and hasattr(self.cell_lib, "forward_tanh"):
                i_edge = self.cell_lib.forward_tanh(x_src=x_src0, x_dst=x_dst0, x_max=self.x_max)
            else:
                i_edge = self.cell_lib(x_src=x_src0, x_dst=x_dst0, x_max=self.x_max)
            edge_mask = torch.sigmoid(self.z_logits)
            if self.budget_enabled:
                edge_mask = edge_mask * self._compute_budget_gate()
            i_edge = i_edge * edge_mask.unsqueeze(0)
            i_edge_f32 = i_edge.float()
            acc_const = torch.zeros_like(x0, dtype=torch.float32)
            acc_const.index_add_(1, self.dst, i_edge_f32)
            if not self.read_only_source:
                acc_const.index_add_(1, self.src, -i_edge_f32)
            i_edge_const = acc_const.to(dtype=x0.dtype)

        x = x0
        traj_chunks = [x] if store_trajectory else None

        for _ in range(num_steps):
            k1 = self.rhs(x, u=u, x_drive=x_drive, drive_scale=drive_scale,
                          i_edge_const=i_edge_const)
            x_pred = x + dt * k1
            k2 = self.rhs(x_pred, u=u, x_drive=x_drive, drive_scale=drive_scale,
                          i_edge_const=i_edge_const)
            x = x + 0.5 * dt * (k1 + k2)
            if store_trajectory:
                traj_chunks.append(x)

        traj = torch.stack(traj_chunks, dim=2) if store_trajectory else None
        return x, traj

    def _forward_heun_sequence(
        self,
        x0: torch.Tensor,
        t_span: float,
        num_steps: int,
        u_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate over a sequence of per-sample inputs with state carryover.

        Processes ``u_seq`` as ``T`` consecutive sample windows. For each
        sample, runs ``num_steps`` Heun steps with that sample's ``u``
        value, threading the final state into the next sample.

        This moves the per-sample Python loop into a single C++/CUDA call
        boundary, reducing interpreter overhead for evaluation and training.

        Args:
            x0: Initial state, shape ``(B, N)``.
            t_span: Integration window duration per sample.
            num_steps: Heun steps per sample.
            u_seq: Input sequence. Accepted shapes:
                - ``(T, 1)`` — eval mode, single stream (B=1)
                - ``(B, T)`` — batched, no explicit input-dim axis
                - ``(B, T, 1)`` — batched with explicit input-dim axis

        Returns:
            Final states at each sample boundary, shape ``(T, B, N)``
            (or ``(T, N)`` if B=1).
        """
        dt = t_span / float(num_steps)
        B = x0.shape[0]

        # Determine (batched, T) from u_seq shape.
        # ``batched`` is True when the first axis is the batch dim.
        if u_seq.dim() == 3:
            batched = (u_seq.shape[0] == B)
            T = u_seq.shape[1] if batched else u_seq.shape[0]
        elif u_seq.dim() == 2:
            batched = (u_seq.shape[0] == B)
            T = u_seq.shape[1] if batched else u_seq.shape[0]
        else:
            # 1D: assume (T,) — single-stream eval, B=1
            batched = False
            T = u_seq.shape[0]

        x = x0
        states = torch.empty(T, B, x0.shape[1], dtype=x0.dtype, device=x0.device)

        for t in range(T):
            if not batched:
                if u_seq.dim() == 1:
                    u_t = u_seq[t].view(1, 1)
                else:
                    u_t = u_seq[t].view(1, 1)
            else:
                # u_seq is (B, T) or (B, T, 1)
                if u_seq.dim() == 2:
                    u_t = u_seq[:, t].unsqueeze(-1)  # (B, 1)
                else:
                    u_t = u_seq[:, t, :]  # (B, 1)

            # freeze_read: precompute edge currents from the current state
            # (each sample window freezes from its own starting state, matching
            # the per-sample _forward_heun semantics).
            i_edge_const = None
            if self.freeze_read:
                x_src0 = x[:, self.src]
                x_dst0 = x[:, self.dst]
                if self._has_resistive and hasattr(self.cell_lib, "forward_tanh"):
                    i_edge = self.cell_lib.forward_tanh(x_src=x_src0, x_dst=x_dst0, x_max=self.x_max)
                else:
                    i_edge = self.cell_lib(x_src=x_src0, x_dst=x_dst0, x_max=self.x_max)
                edge_mask = torch.sigmoid(self.z_logits)
                if self.budget_enabled:
                    edge_mask = edge_mask * self._compute_budget_gate()
                i_edge = i_edge * edge_mask.unsqueeze(0)
                i_edge_f32 = i_edge.float()
                acc_const = torch.zeros_like(x, dtype=torch.float32)
                acc_const.index_add_(1, self.dst, i_edge_f32)
                if not self.read_only_source:
                    acc_const.index_add_(1, self.src, -i_edge_f32)
                i_edge_const = acc_const.to(dtype=x.dtype)

            x = self._call_heun_steps(x, u_t, dt, num_steps, i_edge_const)
            states[t] = x

        return states

    def _call_heun_steps(
        self,
        x: torch.Tensor,
        u_t: torch.Tensor,
        dt: float,
        num_steps: int,
        i_edge_const: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch to compiled or uncompiled Heun steps with fallback.

        ``torch.compile`` is lazy: the compilation error (e.g. missing C++
        compiler on CPU) surfaces on the *first call*, not at
        ``enable_sequence_compile()`` time. This wrapper catches such errors
        and falls back to the original Python implementation so training
        can proceed.
        """
        if getattr(self, "_heun_steps_compiled", False):
            try:
                return self._heun_steps_compiled_fn(x, u_t, dt, num_steps, i_edge_const)
            except Exception as e:
                print(
                    f"  [torch.compile] runtime compilation failed: {e}\n"
                    f"  [torch.compile] falling back to uncompiled Heun steps."
                )
                self._heun_steps_compiled = False
                return self._heun_steps(x, u_t, dt, num_steps, i_edge_const)
        return self._heun_steps(x, u_t, dt, num_steps, i_edge_const)

    def _heun_steps(
        self,
        x: torch.Tensor,
        u_t: torch.Tensor,
        dt: float,
        num_steps: int,
        i_edge_const: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run ``num_steps`` Heun integration steps (predictor-corrector).

        Extracted from :meth:`_forward_heun_sequence` so :func:`torch.compile`
        can capture the hot inner loop into fused CUDA kernels. The outer
        per-sample loop in ``_forward_heun_sequence`` stays Python so the
        torch.compile recompilation cost is paid only once per
        (B, N) shape combination, not per (T, num_steps) slice.
        """
        for _ in range(num_steps):
            k1 = self.rhs(x, u=u_t, x_drive=None, drive_scale=0.0,
                          i_edge_const=i_edge_const)
            x_pred = x + dt * k1
            k2 = self.rhs(x_pred, u=u_t, x_drive=None, drive_scale=0.0,
                          i_edge_const=i_edge_const)
            x = x + 0.5 * dt * (k1 + k2)
        return x

    def enable_sequence_compile(self) -> None:
        """Wrap :meth:`_heun_steps` with :func:`torch.compile` for speed.

        Call this once after the stage is built (and before training). On
        CUDA, the inner Heun loop compiles into fused kernels via Inductor,
        typically yielding 2-3x speedup on the per-sample hot path. On CPU
        the speedup is smaller (if any); the flag is still safe to set.

        If compilation fails at runtime (e.g. no C++ compiler on CPU-only
        environments), :meth:`_call_heun_steps` catches the error and falls
        back to the uncompiled version so training can proceed.

        Cached: subsequent calls are no-ops.
        """
        if getattr(self, "_heun_steps_compiled", False):
            return
        try:
            self._heun_steps_compiled_fn = torch.compile(
                self._heun_steps, dynamic=False
            )
            self._heun_steps_compiled = True
        except Exception as e:
            print(
                f"  [torch.compile] skipping sequence compile: {e}\n"
                f"  [torch.compile] training will proceed without compilation."
            )
            self._heun_steps_compiled = False

    def forward_equilibrium(
        self,
        x0: torch.Tensor,
        x_drive: torch.Tensor | None = None,
        drive_scale: float = 0.0,
        deq_cfg: dict | None = None,
        u: torch.Tensor | None = None,
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

        # freeze_read: precompute edge currents (cell_lib + edge gate + budget
        # gate + KCL scatter-add) once from x0 and hold them constant across
        # all fixed-point iterations.
        i_edge_const = None
        if self.freeze_read:
            x_src0 = x0[:, self.src]
            x_dst0 = x0[:, self.dst]
            i_edge = self.cell_lib(x_src=x_src0, x_dst=x_dst0, x_max=self.x_max)
            edge_mask = torch.sigmoid(self.z_logits)
            if self.budget_enabled:
                edge_mask = edge_mask * self._compute_budget_gate()
            i_edge = i_edge * edge_mask.unsqueeze(0)
            i_edge_f32 = i_edge.float()
            acc_const = torch.zeros_like(x0, dtype=torch.float32)
            acc_const.index_add_(1, self.dst, i_edge_f32)
            if not self.read_only_source:
                acc_const.index_add_(1, self.src, -i_edge_f32)
            i_edge_const = acc_const.to(dtype=x0.dtype)

        self.set_leak_floor(lf)
        try:
            def phi(x):
                return x + dt * self.rhs(x, u=u,
                                        x_drive=x_drive, drive_scale=drive_scale,
                                        leak_floor=lf,
                                        i_edge_const=i_edge_const)

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
            if getattr(self.cell_lib, "_parallel_tanh_mult_enabled", False):
                device_n += (
                    int(self.cell_lib.gm_x_raw.numel())
                    + int(self.cell_lib.gm_y_raw.numel())
                    + int(self.cell_lib.isat_parallel_raw.numel())
                )
        elif isinstance(self.cell_lib, AntiParallelFreeTanhLibrary):
            device_n = (
                int(self.cell_lib.kappa_raw.numel())
                + int(self.cell_lib.gm_raw.numel())
                + int(self.cell_lib.isat_raw.numel())
            )
            if hasattr(self.cell_lib, "theta_raw"):
                device_n += int(self.cell_lib.theta_raw.numel())
        else:
            device_n = 0
        raw_leak_n = int(self.raw_leak.numel()) if hasattr(self, "raw_leak") else 0
        bz = int(self.boundary_z_logits.numel()) if self.boundary_z_logits is not None else 0
        bdev = 0
        if self.boundary_cell_lib is not None:
            if isinstance(self.boundary_cell_lib, SimpleEdgeLibrary):
                bdev = int(self.boundary_cell_lib.param.numel())
            elif isinstance(self.boundary_cell_lib, RealisticTanhLibrary):
                bdev = int(self.boundary_cell_lib.alpha_raw.numel())
                if hasattr(self.boundary_cell_lib, "bias_raw"):
                    bdev += int(self.boundary_cell_lib.bias_raw.numel())
            elif isinstance(self.boundary_cell_lib, RealisticTanhUpgradeLibrary):
                bdev = (
                    int(self.boundary_cell_lib.alpha_raw.numel())
                    + int(self.boundary_cell_lib.gm_raw.numel())
                    + int(self.boundary_cell_lib.isat_raw.numel())
                )
                if hasattr(self.boundary_cell_lib, "bias_raw"):
                    bdev += int(self.boundary_cell_lib.bias_raw.numel())
            elif isinstance(self.boundary_cell_lib, FreeTanhLibrary):
                bdev = (
                    int(self.boundary_cell_lib.a_raw.numel())
                    + int(self.boundary_cell_lib.b_raw.numel())
                    + int(self.boundary_cell_lib.s_raw.numel())
                    + int(self.boundary_cell_lib.gm_raw.numel())
                    + int(self.boundary_cell_lib.isat_raw.numel())
                )
                if hasattr(self.boundary_cell_lib, "theta_raw"):
                    bdev += int(self.boundary_cell_lib.theta_raw.numel())
                if getattr(self.boundary_cell_lib, "_parallel_tanh_mult_enabled", False):
                    bdev += (
                        int(self.boundary_cell_lib.gm_x_raw.numel())
                        + int(self.boundary_cell_lib.gm_y_raw.numel())
                        + int(self.boundary_cell_lib.isat_parallel_raw.numel())
                    )
            elif isinstance(self.boundary_cell_lib, AntiParallelFreeTanhLibrary):
                bdev = (
                    int(self.boundary_cell_lib.kappa_raw.numel())
                    + int(self.boundary_cell_lib.gm_raw.numel())
                    + int(self.boundary_cell_lib.isat_raw.numel())
                )
                if hasattr(self.boundary_cell_lib, "theta_raw"):
                    bdev += int(self.boundary_cell_lib.theta_raw.numel())
        # Reference (unary nonlinearity) stats
        ref_n = 0
        ref_device_n = 0
        if self._has_ref:
            if hasattr(self, "raw_vref"):
                ref_n += int(self.raw_vref.numel())
            if hasattr(self, "ref_z_logits"):
                ref_n += int(self.ref_z_logits.numel())
            # Device param: count all parameters in ref_cell_lib
            if self.ref_cell_lib is not None:
                if hasattr(self.ref_cell_lib, "param"):
                    ref_device_n += int(self.ref_cell_lib.param.numel())
                elif hasattr(self.ref_cell_lib, "alpha_raw"):
                    r = self.ref_cell_lib.alpha_raw.numel()
                    ref_device_n += r
                    if hasattr(self.ref_cell_lib, "bias_raw"):
                        ref_device_n += int(self.ref_cell_lib.bias_raw.numel())
                elif hasattr(self.ref_cell_lib, "gm_raw"):
                    r = int(self.ref_cell_lib.gm_raw.numel())
                    ref_device_n += r
                    if hasattr(self.ref_cell_lib, "isat_raw"):
                        ref_device_n += int(self.ref_cell_lib.isat_raw.numel())
                    if hasattr(self.ref_cell_lib, "a_raw"):
                        ref_device_n += int(self.ref_cell_lib.a_raw.numel())
                    if hasattr(self.ref_cell_lib, "b_raw"):
                        ref_device_n += int(self.ref_cell_lib.b_raw.numel())
                    if hasattr(self.ref_cell_lib, "s_raw"):
                        ref_device_n += int(self.ref_cell_lib.s_raw.numel())
                    if hasattr(self.ref_cell_lib, "theta_raw"):
                        ref_device_n += int(self.ref_cell_lib.theta_raw.numel())
                    if hasattr(self.ref_cell_lib, "kappa_raw"):
                        ref_device_n += int(self.ref_cell_lib.kappa_raw.numel())
        # Temporal-readout OTA edge stats
        out_z = int(self.output_ode_z_logits.numel()) if self.output_ode_z_logits is not None else 0
        out_dev = 0
        if self.output_ode_cell_lib is not None:
            if hasattr(self.output_ode_cell_lib, "param"):
                out_dev = int(self.output_ode_cell_lib.param.numel())
            elif hasattr(self.output_ode_cell_lib, "alpha_raw"):
                out_dev = int(self.output_ode_cell_lib.alpha_raw.numel())
                if hasattr(self.output_ode_cell_lib, "bias_raw"):
                    out_dev += int(self.output_ode_cell_lib.bias_raw.numel())
            elif hasattr(self.output_ode_cell_lib, "gm_raw"):
                out_dev = int(self.output_ode_cell_lib.gm_raw.numel())
                if hasattr(self.output_ode_cell_lib, "isat_raw"):
                    out_dev += int(self.output_ode_cell_lib.isat_raw.numel())
                if hasattr(self.output_ode_cell_lib, "a_raw"):
                    out_dev += int(self.output_ode_cell_lib.a_raw.numel())
                if hasattr(self.output_ode_cell_lib, "b_raw"):
                    out_dev += int(self.output_ode_cell_lib.b_raw.numel())
                if hasattr(self.output_ode_cell_lib, "s_raw"):
                    out_dev += int(self.output_ode_cell_lib.s_raw.numel())
                if hasattr(self.output_ode_cell_lib, "theta_raw"):
                    out_dev += int(self.output_ode_cell_lib.theta_raw.numel())
                if hasattr(self.output_ode_cell_lib, "kappa_raw"):
                    out_dev += int(self.output_ode_cell_lib.kappa_raw.numel())
        vca_proj_n = int(self.vca_W.numel()) if self.vca_W is not None else 0
        vca_embed_n = 0
        if self.vca_v_boundary is not None:
            vca_embed_n += int(self.vca_v_boundary.numel())
        if self.vca_v_readout is not None:
            vca_embed_n += int(self.vca_v_readout.numel())
        return {
            "raw_leak": raw_leak_n,
            "z_logits": int(self.z_logits.numel()),
            "u_logits": int(self.u_logits.numel()),
            "device_param": device_n,
            "boundary_z_logits": bz,
            "boundary_device_param": bdev,
            "raw_vref": ref_n,
            "ref_z_logits": ref_n,
            "ref_device_param": ref_device_n,
            "output_ode_z_logits": out_z,
            "output_ode_device_param": out_dev,
            "vca_proj": vca_proj_n,
            "vca_embed": vca_embed_n,
            "total": (
                raw_leak_n
                + int(self.z_logits.numel())
                + int(self.u_logits.numel())
                + device_n
                + bz
                + bdev
                + ref_n
                + ref_device_n
                + out_z
                + out_dev
                + vca_proj_n
                + vca_embed_n
            ),
        }
