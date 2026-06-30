"""Loss function, regularizers, and training loop for the differential KirchhoffNet.

The loss combines:
  - task loss (MSE / BCE / MAE / residual+solution)
  - sparsity regularizer: pushes edges toward Z cell (cell-selection)
  - edge_gate regularizer (CP): Σ_e σ(z_logits) — active edge count proxy
  - node_gate regularizer (CP): Σ_j σ(u_logits) — active hidden node count proxy
  - power regularizer (CP): Σ_e z_e·m_e·Σ_q w_q·gm_q — static power proxy
  - capacitance regularizer (CP): C_eff·Σ_j u_j — capacitance area proxy
   - rail regularizer: ReLU² quadratic barrier for trajectory excursions beyond x_max
  - entropy bonus: -Σ w·log(w) of softmax distribution over cell types

Training loop injects random SimContext per iteration, anneals tau, and
clips gradients to norm 5.0. Regularizers are scheduled with a staged
warm-up: ``[0, W)`` → off, ``[W, W+A)`` → linear anneal from 0 to full
value, ``[W+A, ∞)`` → full value (RR-A + CP).
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from config import (
    LAMBDAS,
    OPTIM,
    SCHEDULE_FOUR_PHASE,
    SCHEDULE_THREE_PHASE,
    SOLVER,
    TAU,
    VARIATION,
)
from sim_context import SimContext, sample_random_context
from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO
from cell_library import (
    RealisticTanhLibrary,
    RealisticTanhUpgradeLibrary,
    SimpleEdgeLibrary,
)


__all__ = [
    "compute_loss",
    "compute_solver_loss",
    "residual_loss",
    "solution_loss",
    "tau_for_epoch",
    "reg_schedule",
    "apply_reg_schedule",
    "phase_for_epoch",
    "phase_boundaries",
    "three_phase_tau",
    "three_phase_lambdas",
    "four_phase_boundaries",
    "phase_for_epoch_four",
    "four_phase_tau",
    "four_phase_lambdas",
    "four_phase_kd_active",
    "prune_readiness_check",
    "compute_solidification_metrics",
    "validate_argmax",
    "make_optimizer",
    "apply_ablation",
    "train_epoch",
    "budget_frac_for_epoch",
    "budget_temperature_for_epoch",
]


# ---------- regularizers ----------

def _stage_soft_weights(stage) -> torch.Tensor:
    if stage.logits is None:
        E = stage.num_edges()
        return torch.ones(E, 1, device=stage.z_logits.device, dtype=stage.z_logits.dtype)
    return F.softmax(stage.logits, dim=-1)


def _stage_multiplicities(stage) -> torch.Tensor:
    if stage.raw_mult is None:
        E = stage.num_edges()
        return torch.ones(E, device=stage.z_logits.device, dtype=stage.z_logits.dtype)
    return F.softplus(stage.raw_mult)


def _stage_edge_gates(stage) -> torch.Tensor:
    """Edge gate values z_e = σ(z_logits), shape [E]."""
    return torch.sigmoid(stage.z_logits)


def _stage_node_gates(stage) -> torch.Tensor:
    """Node gate values u_j = σ(u_logits), shape [N].

    DEPRECATED (deprecate-node-gates): node gates are no longer used in
    the forward pass, in any regularizer, or in pruning. Returns an
    all-ones tensor so any stray caller gets a no-op result instead of
    corrupting the dynamics.
    """
    import warnings as _warnings
    _warnings.warn(
        "train._stage_node_gates() is deprecated (deprecate-node-gates); "
        "node gates are no longer used. Returns an all-ones tensor.",
        DeprecationWarning,
        stacklevel=2,
    )
    return torch.ones(stage.num_nodes, device=stage.u_logits.device,
                       dtype=stage.u_logits.dtype)


def _stage_rail_loss(stage, traj: torch.Tensor) -> torch.Tensor:
    """Mean over (batch, node, time) of ReLU^2(|x| - x_max).

    ReLU² barrier (three-phase-schedule/rail-loss-fix): exactly zero loss
    and zero gradient when all voltages are within [-x_max, x_max].
    Quadratic growth beyond bounds gives a smooth, differentiable penalty
    for voltage excursions. Replaces the previous softplus formulation
    whose non-zero floor (softplus(-x_max) ≈ 0.049 at x_max=3.0) dragged
    the task loss by ~16% even when no voltage was out of bounds.
    """
    excess = F.relu(traj.abs() - stage.x_max)
    return excess.pow(2).mean()


def _compute_regularizers(
    net: KirchhoffNetWithIO | KirchhoffNet,
    trajs: list[torch.Tensor],
    tau: float,
    lambdas: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Shared regularizer computation: sparsity, edge_gate, power, rail, entropy.

    DEPRECATED (deprecate-node-gates): the ``node_gate`` and ``capacitance``
    regularizers are no longer accumulated. Both return zero tensors
    (preserved in the return tuple for backward compatibility with callers
    that read ``parts["node_gate"]`` / ``parts["capacitance"]``). The
    corresponding ``lambdas["node_gate"]`` and ``lambdas["capacitance"]``
    entries are kept at 0.0 across all configs.

    The remaining complexity terms decompose the previous single ``complexity``
    proxy into per-component hardware terms:
      - edge_gate   : Σ_e σ(z_logits)                       (active edge count)
      - power       : Σ_e z_e · m_e · Σ_q w_q · gm_q       (static power proxy)

    The cell-selection sparsity ``Σ w[:, :z_idx]`` is preserved (pushes
    individual edges toward the zero-current Z cell as a fine-grained
    selection, distinct from the edge gate which is a coarse on/off).

    Returns ``(loss_sparsity, loss_edge_gate, loss_node_gate, loss_power,
    loss_capacitance, loss_rail, entropy_bonus, lambda_entropy)``. The
    ``loss_node_gate`` and ``loss_capacitance`` entries are always zero.
    """
    stages = net.core.stages if isinstance(net, KirchhoffNetWithIO) else net.stages
    loss_sparsity = loss_edge_gate = loss_node_gate = loss_power = loss_capacitance = loss_rail = entropy_bonus = trajs[0].new_zeros(())
    for stage, traj in zip(stages, trajs):
        w = _stage_soft_weights(stage)
        mult = _stage_multiplicities(stage)
        z = _stage_edge_gates(stage)
        z_idx = stage.cell_lib.z_index

        # Cell-selection sparsity: push toward Z cell.
        loss_sparsity = loss_sparsity + w[:, :z_idx].sum()

        # Edge gate penalty: soft count of active edges.
        loss_edge_gate = loss_edge_gate + z.sum()

        # Static power proxy: z_e · m_e · weighted gm sum per edge.
        # gm values per cell come from the shared cell library.
        gm_per_cell = stage.cell_lib.gm_values()  # [Q]
        effective_gm = (w * gm_per_cell.unsqueeze(0)).sum(dim=-1)  # [E]
        loss_power = loss_power + (z * mult * effective_gm).sum()

        # DEPRECATED (deprecate-node-gates): node_gate and capacitance
        # regularizers are no longer accumulated — both ``loss_node_gate``
        # and ``loss_capacitance`` stay at the initial zero tensor. Node
        # pruning is connectivity-only; the C_eff·Σu proxy would be a
        # constant per stage and thus useless as a regularizer.

        loss_rail = loss_rail + _stage_rail_loss(stage, traj)

        if stage.logits is not None:
            weights_tau = F.softmax(stage.logits / tau, dim=-1)
            entropy = -(weights_tau * torch.log(weights_tau + 1e-10)).sum(dim=-1).mean()
            entropy_bonus = entropy_bonus + entropy

    lambda_entropy = float(lambdas.get("entropy", 0.0)) * tau
    return (
        loss_sparsity,
        loss_edge_gate,
        loss_node_gate,
        loss_power,
        loss_capacitance,
        loss_rail,
        entropy_bonus,
        lambda_entropy,
    )


# ---------- regularizer warm-up schedule (RR-A + CP) ----------

_REG_KEYS = (
    "sparsity",
    "edge_gate",
    "node_gate",
    "power",
    "capacitance",
)


def reg_schedule(epoch: int, *, warmup: int | None = None, anneal: int | None = None) -> float:
    """Return a scalar in [0, 1] scaling the staged regularizers (RR-A).

    - epoch < warmup: 0.0 (network learns freely).
    - warmup <= epoch < warmup + anneal: linear ramp from 0 → 1.
    - epoch >= warmup + anneal: 1.0 (full penalty).

    Defaults: ``warmup = OPTIM["reg_warmup_epochs"] = 50``,
    ``anneal = OPTIM["reg_anneal_epochs"] = 50``.

    Note (fix-z-death): ``rail`` is NOT in ``_REG_KEYS``. The rail regularizer
    is a safety voltage clamp on differential node states (clamps x_j to
    ±x_max via soft sigmoid), not a structural complexity regularizer. Gating
    it behind a warm-up schedule caused catastrophic loss explosion in the
    retrain phase (0.29→1.37) when the ramp kicked in at retrain epoch 75.
    Rail is now applied at full strength at every epoch.
    """
    if warmup is None:
        warmup = int(OPTIM.get("reg_warmup_epochs", 50))
    if anneal is None:
        anneal = int(OPTIM.get("reg_anneal_epochs", 50))
    if epoch < warmup:
        return 0.0
    if epoch >= warmup + anneal:
        return 1.0
    return float(epoch - warmup + 1) / float(max(1, anneal))


def apply_reg_schedule(lambdas: dict, epoch: int, **kw) -> dict:
    """Return a copy of ``lambdas`` with sparsity/edge_gate/node_gate/
    power/capacitance scaled by ``reg_schedule(epoch)``. ``rail`` and
    ``entropy`` are left untouched (rail is always active; entropy has
    its own τ-scaling)."""
    scale = reg_schedule(epoch, **kw)
    out = dict(lambdas)
    for k in _REG_KEYS:
        if k in out:
            out[k] = float(out[k]) * scale
    return out


# ---------- three-phase schedule (three-phase-schedule plan) ----------

def phase_boundaries(total_epochs: int, schedule_cfg: dict | None = None) -> tuple[int, int, int]:
    """Compute the (a_end, b_end, c_end) epoch boundaries for the three-phase schedule.

    The three phases have fixed epoch counts derived from the schedule
    config's phase fractions (frac_a/frac_b/frac_c). Returns the boundary
    epochs in the closed interval [0, total_epochs]. Phase C ends at
    total_epochs (i.e. c_end == total_epochs).

    Args:
        total_epochs: Total epoch budget.
        schedule_cfg: A ``SCHEDULE_THREE_PHASE`` dict. Defaults to the
            module-level ``SCHEDULE_THREE_PHASE``.

    Returns:
        ``(a_end, b_end, c_end)`` epoch boundaries with
        ``0 <= a_end <= b_end <= c_end == total_epochs``.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    frac_a = float(schedule_cfg.get("frac_a", 0.30))
    frac_b = float(schedule_cfg.get("frac_b", 0.40))
    frac_c = float(schedule_cfg.get("frac_c", 0.30))
    total_frac = frac_a + frac_b + frac_c
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"three_phase schedule fractions must sum to 1.0, got {total_frac} "
            f"(frac_a={frac_a}, frac_b={frac_b}, frac_c={frac_c})"
        )
    a_end = int(round(total_epochs * frac_a))
    b_end = int(round(total_epochs * (frac_a + frac_b)))
    c_end = total_epochs
    if a_end < 1:
        a_end = 1
    if b_end < a_end + 1:
        b_end = a_end + 1
    return (a_end, b_end, c_end)


def phase_for_epoch(epoch: int, total_epochs: int, schedule_cfg: dict | None = None) -> str:
    """Return the active phase name (``'A'``, ``'B'``, or ``'C'``) for a given epoch.

    Boundary semantics:
      - Phase A: ``0 <= epoch < a_end``
      - Phase B: ``a_end <= epoch < b_end``
      - Phase C: ``b_end <= epoch < c_end`` (c_end == total_epochs)

    Pruning happens at the epoch ``b_end`` (the first epoch of Phase C is
    the retrain of the freshly-pruned network).
    """
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)
    if epoch < a_end:
        return "A"
    if epoch < b_end:
        return "B"
    return "C"


def three_phase_tau(epoch: int, total_epochs: int, schedule_cfg: dict | None = None) -> float:
    """Tau value for the current epoch under the three-phase schedule.

    Phase A: fixed at ``tau_a`` (default 1.0).
    Phase B: anneals from ``tau_b_init`` to ``tau_b_final`` (default 1.0→0.6).
    Phase C: anneals from ``tau_c_init`` to ``tau_c_final`` (default 0.6→0.1).

    Reuses the existing ``tau_for_epoch`` exponential-decay + linear-hardening
    behavior, scoped to each phase's local epoch window.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)

    if epoch < a_end:
        return float(schedule_cfg.get("tau_a", 1.0))

    if epoch < b_end:
        tau_init = float(schedule_cfg.get("tau_b_init", 1.0))
        tau_final = float(schedule_cfg.get("tau_b_final", 0.6))
        local_epoch = epoch - a_end
        local_total = max(1, b_end - a_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    tau_init = float(schedule_cfg.get("tau_c_init", 0.6))
    tau_final = float(schedule_cfg.get("tau_c_final", 0.1))
    local_epoch = epoch - b_end
    local_total = max(1, total_epochs - b_end)
    return tau_for_epoch(local_epoch, total_epochs=local_total,
                         tau_init=tau_init, tau_final=tau_final)


def three_phase_lambdas(
    epoch: int,
    total_epochs: int,
    base_lambdas: dict,
    schedule_cfg: dict | None = None,
) -> dict:
    """Effective lambda dict for the current epoch under the three-phase schedule.

    Phase A: all structural lambdas (sparsity, edge_gate, node_gate, power,
      capacitance) are zeroed. ``rail`` is left at its base value as a
      safety net. No warmup.

    Phase B: the Phase B target lambdas (``lambdas_b`` in the schedule
      config) are linearly ramped from 0 to full over the first
      ``warmup_frac_b`` of Phase B's epoch window, then held at full.

    Phase C: the Phase C retrain lambdas (``lambdas_c``) are applied at
      full strength from the start of the phase (no warmup — pruning has
      already happened at the B→C boundary, the surviving network should
      re-fit quickly).

    All returned dicts include ``rail`` (preserved from ``base_lambdas``).
    Returns a fresh dict; ``base_lambdas`` is not mutated.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_THREE_PHASE
    a_end, b_end, _ = phase_boundaries(total_epochs, schedule_cfg)
    rail_val = float(base_lambdas.get("rail", 0.0))
    out = {"rail": rail_val}

    if epoch < a_end:
        # Phase A: zero all structural lambdas. Rail is the only active term.
        for k in _REG_KEYS:
            out[k] = 0.0
        return out

    if epoch < b_end:
        # Phase B: ramp lambdas_b from 0 to full over warmup window.
        lambdas_b = schedule_cfg.get("lambdas_b", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b", 1.0 / 6.0))
        local_epoch = epoch - a_end
        local_total = max(1, b_end - a_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b.get(k, 0.0)) * scale
        return out

    # Phase C: apply lambdas_c at full strength, no warmup.
    lambdas_c = schedule_cfg.get("lambdas_c", {})
    for k in _REG_KEYS:
        out[k] = float(lambdas_c.get(k, 0.0))
    return out


# ---------- four-phase schedule (four-phase-redesign plan) ----------

def four_phase_boundaries(
    total_epochs: int, schedule_cfg: dict | None = None
) -> tuple[int, int, int, int]:
    """Compute (a_end, b1_end, b2_end, c_end) epoch boundaries for the
    four-phase schedule (four-phase-redesign/Phase 3b).

    Phase fractions must sum to 1.0 (frac_a + frac_b1 + frac_b2 + frac_c = 1).
    Each end is computed by rounding total_epochs * cumulative_frac, with
    the constraint that adjacent phases differ by at least 1 epoch.

    Returns (a_end, b1_end, b2_end, c_end) with
    ``0 < a_end < b1_end < b2_end < c_end == total_epochs``.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    frac_a = float(schedule_cfg.get("frac_a", 0.25))
    frac_b1 = float(schedule_cfg.get("frac_b1", 0.20))
    frac_b2 = float(schedule_cfg.get("frac_b2", 0.25))
    frac_c = float(schedule_cfg.get("frac_c", 0.30))
    total_frac = frac_a + frac_b1 + frac_b2 + frac_c
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"four_phase schedule fractions must sum to 1.0, got {total_frac} "
            f"(frac_a={frac_a}, frac_b1={frac_b1}, frac_b2={frac_b2}, frac_c={frac_c})"
        )
    a_end = int(round(total_epochs * frac_a))
    b1_end = int(round(total_epochs * (frac_a + frac_b1)))
    b2_end = int(round(total_epochs * (frac_a + frac_b1 + frac_b2)))
    c_end = total_epochs
    if a_end < 1:
        a_end = 1
    if b1_end < a_end + 1:
        b1_end = a_end + 1
    if b2_end < b1_end + 1:
        b2_end = b1_end + 1
    return (a_end, b1_end, b2_end, c_end)


def phase_for_epoch_four(
    epoch: int, total_epochs: int, schedule_cfg: dict | None = None
) -> str:
    """Return the active phase name for the four-phase schedule.

    Returns one of: 'A', 'B1', 'B2', 'C'.
    Boundary semantics:
      - Phase A:  ``0 <= epoch < a_end``
      - Phase B1: ``a_end <= epoch < b1_end``
      - Phase B2: ``b1_end <= epoch < b2_end``
      - Phase C:  ``b2_end <= epoch < c_end`` (c_end == total_epochs)
    """
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)
    if epoch < a_end:
        return "A"
    if epoch < b1_end:
        return "B1"
    if epoch < b2_end:
        return "B2"
    return "C"


def four_phase_tau(
    epoch: int, total_epochs: int, schedule_cfg: dict | None = None
) -> float:
    """Tau value for the current epoch under the four-phase schedule.

    Phase A:  fixed at ``tau_a`` (default 1.0).
    Phase B1: anneals ``tau_b1_init`` -> ``tau_b1_final`` (1.0 -> 0.6).
    Phase B2: anneals ``tau_b2_init`` -> ``tau_b2_final`` (0.6 -> 0.4).
    Phase C:  anneals ``tau_c_init`` -> ``tau_c_final`` (0.4 -> 0.1).

    Tau is continuous at all phase boundaries (end of B1 == start of B2
    at 0.6; end of B2 == start of C at 0.4 by construction).
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)

    if epoch < a_end:
        return float(schedule_cfg.get("tau_a", 1.0))

    if epoch < b1_end:
        tau_init = float(schedule_cfg.get("tau_b1_init", 1.0))
        tau_final = float(schedule_cfg.get("tau_b1_final", 0.6))
        local_epoch = epoch - a_end
        local_total = max(1, b1_end - a_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    if epoch < b2_end:
        tau_init = float(schedule_cfg.get("tau_b2_init", 0.6))
        tau_final = float(schedule_cfg.get("tau_b2_final", 0.4))
        local_epoch = epoch - b1_end
        local_total = max(1, b2_end - b1_end)
        return tau_for_epoch(local_epoch, total_epochs=local_total,
                             tau_init=tau_init, tau_final=tau_final)

    tau_init = float(schedule_cfg.get("tau_c_init", 0.4))
    tau_final = float(schedule_cfg.get("tau_c_final", 0.1))
    local_epoch = epoch - b2_end
    local_total = max(1, total_epochs - b2_end)
    return tau_for_epoch(local_epoch, total_epochs=local_total,
                         tau_init=tau_init, tau_final=tau_final)


def four_phase_lambdas(
    epoch: int,
    total_epochs: int,
    base_lambdas: dict,
    schedule_cfg: dict | None = None,
) -> dict:
    """Effective lambda dict for the current epoch under the four-phase
    schedule (four-phase-redesign/Phase 3b).

    Phase A:  all structural lambdas = 0. ``rail`` preserved.
    Phase B1: lambdas_b1 ramped from 0 to full over warmup_frac_b1.
    Phase B2: lambdas_b2 ramped from 0 to full over warmup_frac_b2.
    Phase C:  lambdas_c at full strength (no warmup).

    The output dict always includes ``rail`` from ``base_lambdas``.
    Returns a fresh dict; ``base_lambdas`` is not mutated.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE
    a_end, b1_end, b2_end, _ = four_phase_boundaries(total_epochs, schedule_cfg)
    rail_val = float(base_lambdas.get("rail", 0.0))
    out = {"rail": rail_val}

    if epoch < a_end:
        for k in _REG_KEYS:
            out[k] = 0.0
        return out

    if epoch < b1_end:
        lambdas_b1 = schedule_cfg.get("lambdas_b1", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b1", 0.25))
        local_epoch = epoch - a_end
        local_total = max(1, b1_end - a_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b1.get(k, 0.0)) * scale
        return out

    if epoch < b2_end:
        lambdas_b2 = schedule_cfg.get("lambdas_b2", {})
        warmup_frac = float(schedule_cfg.get("warmup_frac_b2", 0.25))
        local_epoch = epoch - b1_end
        local_total = max(1, b2_end - b1_end)
        warmup_epochs = max(1, int(round(warmup_frac * local_total)))
        if local_epoch < warmup_epochs:
            scale = float(local_epoch + 1) / float(warmup_epochs)
        else:
            scale = 1.0
        for k in _REG_KEYS:
            out[k] = float(lambdas_b2.get(k, 0.0)) * scale
        return out

    lambdas_c = schedule_cfg.get("lambdas_c", {})
    for k in _REG_KEYS:
        out[k] = float(lambdas_c.get(k, 0.0))
    return out


def four_phase_kd_active(
    epoch: int,
    total_epochs: int,
    schedule_cfg: dict | None = None,
) -> bool:
    """Whether teacher distillation should be active for the current epoch
    (four-phase-redesign/Phase 3c).

    KD is active in Phase B1 and B2 only (not A, not C). During Phase A
    we are fitting the soft teacher; during Phase C we are retraining the
    pruned compact model and want to fit the task directly.
    """
    phase = phase_for_epoch_four(epoch, total_epochs, schedule_cfg)
    return phase in ("B1", "B2")


# ---------- readiness-based prune trigger (four-phase-redesign) ----------

def prune_readiness_check(
    val_soft_history: list[float],
    val_argmax_history: list[float],
    solidification_metrics: list[dict],
    schedule_cfg: dict | None = None,
    min_readings: int = 10,
) -> tuple[bool, dict]:
    """Evaluate whether the network is ready for pruning
    (four-phase-redesign/Phase 3d).

    ALL four conditions must be satisfied (AND-logic):

    1. **Ratio**: ``val_argmax / val_soft < readiness_ratio_max``
       (hard model close to soft model).
    2. **Probability**: ``mean_max_cell_prob > readiness_prob_min``
       (cells are mostly committed to a single type).
    3. **Stability**: std(frac_sigma_z_below_0.1) over the most recent
       ``readiness_window`` readings < readiness_stability_max
       (gate fractions have stopped moving).
    4. **Improvement**: val_argmax improvement rate over the last
       ``readiness_window`` readings < readiness_improvement_min
       (no longer improving rapidly).

    If there are fewer than ``min_readings`` in the histories, returns
    (False, {...}) with an explanation.

    Args:
        val_soft_history: Recent soft-validation losses (most recent last).
        val_argmax_history: Recent argmax-validation losses (most recent
            last). May be shorter than val_soft_history if argmax validation
            was not computed every epoch.
        solidification_metrics: Recent dicts from
            ``compute_solidification_metrics`` (most recent last). Must
            contain keys ``mean_max_cell_prob`` and
            ``frac_sigma_z_below_0.1``.
        schedule_cfg: A ``SCHEDULE_FOUR_PHASE`` dict. Default reads the
            module-level ``SCHEDULE_FOUR_PHASE``.
        min_readings: Minimum number of readings required before computing
            readiness. Default 10.

    Returns:
        ``(ready, details)`` where ``ready`` is True iff all conditions
        are met, and ``details`` is a dict with keys ``ratio``,
        ``max_cell_prob``, ``stability``, ``improvement_rate``,
        ``all_ready``, and the individual boolean checks.
    """
    if schedule_cfg is None:
        schedule_cfg = SCHEDULE_FOUR_PHASE

    ratio_max = float(schedule_cfg.get("readiness_ratio_max", 1.2))
    prob_min = float(schedule_cfg.get("readiness_prob_min", 0.85))
    stability_max = float(schedule_cfg.get("readiness_stability_max", 0.02))
    improvement_min = float(schedule_cfg.get("readiness_improvement_min", 1e-4))
    window = int(schedule_cfg.get("readiness_window", 10))

    details: dict = {
        "ratio": None,
        "max_cell_prob": None,
        "stability": None,
        "improvement_rate": None,
        "ready_ratio": False,
        "ready_prob": False,
        "ready_stability": False,
        "ready_improvement": False,
        "all_ready": False,
        "note": "",
    }

    # Need enough readings for any condition.
    n_val = len(val_argmax_history)
    n_soft = len(val_soft_history)
    n_solid = len(solidification_metrics)
    if n_val < min_readings or n_soft < min_readings or n_solid < min_readings:
        details["note"] = (
            f"too few readings: val={n_val}, soft={n_soft}, solid={n_solid} "
            f"(need ≥{min_readings})"
        )
        return False, details

    # Condition 1: ratio check
    recent_val = val_argmax_history[-window:]
    recent_soft = val_soft_history[-window:]
    ratios = [v / max(s, 1e-10) for v, s in zip(recent_val, recent_soft)]
    mean_ratio = sum(ratios) / len(ratios)
    details["ratio"] = mean_ratio
    details["ready_ratio"] = mean_ratio < ratio_max

    # Condition 2: max cell probability
    recent_probs = [
        m.get("mean_max_cell_prob", 0.0) for m in solidification_metrics[-window:]
    ]
    mean_prob = sum(recent_probs) / len(recent_probs)
    details["max_cell_prob"] = mean_prob
    details["ready_prob"] = mean_prob > prob_min

    # Condition 3: gate stability
    recent_gate_fracs = [
        m.get("frac_sigma_z_below_0.1", 0.0) for m in solidification_metrics[-window:]
    ]
    if len(recent_gate_fracs) >= 2:
        import statistics
        gate_stability = statistics.stdev(recent_gate_fracs)
    else:
        gate_stability = 1.0  # not stable with < 2 readings
    details["stability"] = gate_stability
    details["ready_stability"] = gate_stability < stability_max

    # Condition 4: improvement rate (val_argmax)
    recent_val_for_rate = val_argmax_history[-window:]
    if len(recent_val_for_rate) >= 2:
        n_points = len(recent_val_for_rate)
        x_vals = list(range(n_points))
        # Simple linear fit slope: cov(x,y) / var(x)
        x_mean = sum(x_vals) / n_points
        y_mean = sum(recent_val_for_rate) / n_points
        cov = sum(
            (x_vals[i] - x_mean) * (recent_val_for_rate[i] - y_mean)
            for i in range(n_points)
        )
        var_x = sum((x - x_mean) ** 2 for x in x_vals)
        if var_x > 0:
            improvement_rate = cov / var_x
        else:
            improvement_rate = 0.0
    else:
        improvement_rate = float("inf")
    details["improvement_rate"] = improvement_rate
    # Negative slope = loss is decreasing (improving). We check that the
    # abs improvement rate is below the threshold (i.e. no longer improving
    # rapidly). A positive slope means loss is increasing.
    details["ready_improvement"] = abs(improvement_rate) < improvement_min

    details["all_ready"] = (
        details["ready_ratio"]
        and details["ready_prob"]
        and details["ready_stability"]
        and details["ready_improvement"]
    )

    return details["all_ready"], details


# ---------- solidification monitoring (three-phase-schedule plan) ----------

def compute_solidification_metrics(
    net: KirchhoffNetWithIO | KirchhoffNet,
    tau: float = 1.0,
) -> dict:
    """Compute edge/node solidification metrics across all stages.

    Returns a dict of Python floats (safe to log to a text file):
      - ``mean_max_cell_prob``: mean over all edges of max(softmax(logits/τ)).
        1.0 = fully discrete cell selection; ~0.25 (1/4 cells) = uniform.
      - ``mean_pZ``: mean over all edges of softmax(logits/τ)[:, z_index].
        Probability mass on the zero-current Z cell.
      - ``mean_sigma_z``: mean over all edges of σ(z_logits). Average edge
        gate openness (0..1).
      - ``frac_sigma_z_below_01/005/001``: fraction of edge gates with
        σ(z_logits) below the listed threshold. Measures how many edges
        are effectively off (eligible for pruning).
      - ``mean_sigma_u``: **VESTIGIAL** (deprecate-node-gates): mean over
        all nodes of σ(u_logits). Node gates are no longer used in the
        forward pass, regularizers, or pruning — this metric reflects the
        stale ``u_logits`` parameter values (which receive no gradient
        pressure) and should not be interpreted as meaningful.
      - ``num_edges``, ``num_nodes``: total counts across all stages.
      - ``tau``: the tau value used (echoed for the log file).
    """
    stages = net.core.stages if isinstance(net, KirchhoffNetWithIO) else net.stages

    max_probs_list = []
    p_z_list = []
    sigma_z_list = []
    sigma_u_list = []
    total_edges = 0
    total_nodes = 0

    for stage in stages:
        n_edges = int(stage.logits.shape[0]) if stage.logits is not None else stage.num_edges()
        n_nodes = int(stage.u_logits.shape[0]) if hasattr(stage, "u_logits") and stage.u_logits.numel() > 0 else 0
        if n_edges == 0:
            total_nodes += n_nodes
            continue

        sigma_z = torch.sigmoid(stage.z_logits)
        sigma_z_list.append(sigma_z.detach())

        if stage.logits is not None:
            weights = F.softmax(stage.logits / float(tau), dim=-1)
            max_probs, _ = weights.max(dim=-1)
            p_z = weights[:, stage.cell_lib.z_index]
            max_probs_list.append(max_probs.detach())
            p_z_list.append(p_z.detach())
        sigma_z_list.append(sigma_z.detach())

        if n_nodes > 0:
            sigma_u_list.append(torch.sigmoid(stage.u_logits).detach())
            total_nodes += n_nodes
        total_edges += n_edges

    out = {"tau": float(tau), "num_edges": total_edges, "num_nodes": total_nodes}
    if max_probs_list:
        all_max = torch.cat(max_probs_list)
        all_pz = torch.cat(p_z_list)
        all_sz = torch.cat(sigma_z_list)
        out["mean_max_cell_prob"] = float(all_max.mean().item())
        out["mean_pZ"] = float(all_pz.mean().item())
        out["mean_sigma_z"] = float(all_sz.mean().item())
        out["frac_sigma_z_below_0.1"] = float((all_sz < 0.1).float().mean().item())
        out["frac_sigma_z_below_0.05"] = float((all_sz < 0.05).float().mean().item())
        out["frac_sigma_z_below_0.01"] = float((all_sz < 0.01).float().mean().item())
    else:
        out["mean_max_cell_prob"] = 0.0
        out["mean_pZ"] = 0.0
        out["mean_sigma_z"] = 0.0
        out["frac_sigma_z_below_0.1"] = 0.0
        out["frac_sigma_z_below_0.05"] = 0.0
        out["frac_sigma_z_below_0.01"] = 0.0

    if sigma_u_list:
        all_su = torch.cat(sigma_u_list)
        out["mean_sigma_u"] = float(all_su.mean().item())
    else:
        out["mean_sigma_u"] = 0.0

    return out


# ---------- argmax validation (three-phase-schedule plan) ----------

def validate_argmax(
    net,
    val_loader,
    task_fn,
    ctx_factory,
    device,
    *,
    argmax_tau: float = 0.001,
    solver: str = "heun",
    deq_cfg: dict | None = None,
) -> float:
    """Validation loss with effectively-argmax cell selection (τ→0).

    Runs the validation loop a second time using ``argmax_tau`` (very small,
    so softmax(logits/τ) becomes a one-hot vector at the argmax index) and
    returns the corresponding task loss. Compare against the normal soft-τ
    validation loss:

      gap = val_argmax - val_soft

    A small gap means the network's cell selection is effectively
    solidified; a large gap means it still relies on blurry cell mixtures
    regardless of what τ reports.

    The original tau is preserved by re-running the forward pass with a
    different tau value; no in-place mutation of network parameters.
    """
    net.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out, _ = net(
                u,
                ctx=ctx,
                tau=argmax_tau,
                store_trajectory=False,
                solver=solver,
                deq_cfg=deq_cfg,
            )
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
    net.train()
    return total / max(1, n)


def compute_loss(
    net: KirchhoffNetWithIO | KirchhoffNet,
    x0: torch.Tensor,
    target: torch.Tensor,
    ctx: SimContext,
    task_fn,
    lambdas: dict | None = None,
    tau: float = 1.0,
    return_parts: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    reg_scale: float = 1.0,
    cell_mode: str = "soft",
    teacher: KirchhoffNetWithIO | KirchhoffNet | None = None,
    lambda_kd: float = 0.0,
    teacher_tau: float = 1.0,
    teacher_cell_mode: str = "soft",
    solver: str = "heun",
    deq_cfg: dict | None = None,
):
    """Compute total loss = task + regularizers + entropy bonus.

    If net is a KirchhoffNetWithIO, x0 is the raw input u. If it is a plain
    KirchhoffNet, x0 is the already-bounded initial differential state.

    ``reg_scale`` (RR-A) is a multiplicative factor on sparsity/complexity/
    rail: pass ``reg_schedule(epoch)`` from the training loop to implement
    staged warm-up. Defaults to 1.0 (no warm-up).

    ``cell_mode`` (four-phase-redesign/Phase 2a): forwarded to the
    network forward pass. ``'soft'`` is the standard mixture; ``'ste'``
    uses one cell per edge in the forward pass with straight-through
    soft gradients.

    ``teacher`` / ``lambda_kd`` (four-phase-redesign/Phase 3c): when a
    teacher network is provided, an additional KD loss
    ``lambda_kd * MSE(y_student, y_teacher)`` is added to the data-side
    loss. Teacher is run with ``torch.no_grad()`` and uses
    ``teacher_cell_mode`` / ``teacher_tau`` (defaults: soft / 1.0). The
    KD loss is a data-side term so it lives in ``total_task`` (gets
    DataParallel averaging like the task loss) rather than in
    ``structural`` (which is not averaged).

    If `amp` is True, wraps forward+loss in torch.cuda.amp.autocast for
    mixed-precision training. Caller is responsible for GradScaler.
    """
    if lambdas is None:
        lambdas = LAMBDAS

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if amp else _NullContext()
    )
    with autocast_ctx:
        if isinstance(net, KirchhoffNetWithIO):
            out, trajs = net(x0, ctx=ctx, tau=tau, store_trajectory=True,
                             cell_mode=cell_mode,
                             solver=solver, deq_cfg=deq_cfg)
        else:
            out, trajs = net(x0, ctx=ctx, tau=tau, store_trajectory=True,
                             cell_mode=cell_mode,
                             solver=solver, deq_cfg=deq_cfg)

        loss_task = task_fn(out, target)

        # four-phase-redesign/Phase 3c: teacher distillation. Compute
        # y_teacher in no_grad mode with its own tau/cell_mode and add
        # MSE to the data-side loss.
        loss_kd = loss_task.new_zeros(())
        if teacher is not None and lambda_kd > 0.0:
            with torch.no_grad():
                if isinstance(teacher, KirchhoffNetWithIO):
                    y_teacher, _ = teacher(x0, ctx=ctx, tau=teacher_tau,
                                           store_trajectory=False,
                                           cell_mode=teacher_cell_mode)
                else:
                    y_teacher, _ = teacher(x0, ctx=ctx, tau=teacher_tau,
                                           store_trajectory=False,
                                           cell_mode=teacher_cell_mode)
            loss_kd = F.mse_loss(out, y_teacher.detach())

        if trajs is None:
            zero = loss_task.new_zeros((), requires_grad=True)
            if return_parts:
                return loss_task, zero, {"task": float(loss_task.item())}
            return loss_task, zero

        (
            loss_sparsity,
            loss_edge_gate,
            loss_node_gate,
            loss_power,
            loss_capacitance,
            loss_rail,
            entropy_bonus,
            lambda_entropy,
        ) = _compute_regularizers(net, trajs, tau, lambdas)

        # Split the loss into two pieces so that DataParallel averages
        # ONLY the data-dependent pieces (task + rail + KD) and does NOT
        # halve the structural regularizers (sparsity/edge_gate/
        # node_gate/power/capacitance/entropy) which depend only on
        # `net.module`'s parameters and have no business being
        # averaged across replicas.  The caller is expected to call
        # `backward()` on `total_task + structural` (a single combined
        # backward call to remain compatible with torch.compile, which
        # forbids retain_graph=True on the first of two backward calls).
        total_task = (
            loss_task
            + float(lambdas.get("rail", 0.0)) * loss_rail
            + float(lambda_kd) * loss_kd
        )
        structural = (
            + reg_scale * float(lambdas.get("sparsity", 0.0)) * loss_sparsity
            + reg_scale * float(lambdas.get("edge_gate", 0.0)) * loss_edge_gate
            + reg_scale * float(lambdas.get("node_gate", 0.0)) * loss_node_gate
            + reg_scale * float(lambdas.get("power", 0.0)) * loss_power
            + reg_scale * float(lambdas.get("capacitance", 0.0)) * loss_capacitance
            - lambda_entropy * entropy_bonus
        )

    if return_parts:
        parts = {
            "task": float(loss_task.item()) if torch.is_tensor(loss_task) else float(loss_task),
            "sparsity": float(loss_sparsity.item()),
            "edge_gate": float(loss_edge_gate.item()),
            "node_gate": float(loss_node_gate.item()),
            "power": float(loss_power.item()),
            "capacitance": float(loss_capacitance.item()),
            "rail": float(loss_rail.item()),
            "entropy": float(entropy_bonus.item()),
            "reg_scale": float(reg_scale),
            "total": float((total_task + structural).item()),
        }
        if teacher is not None and lambda_kd > 0.0:
            parts["kd"] = float(loss_kd.item())
        return total_task, structural, parts
    return total_task, structural


class _NullContext:
    """A no-op context manager for use as a placeholder for autocast."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------- sparse linear solver losses ----------

def residual_loss(x_pred: torch.Tensor, b: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Mean squared residual ||A x_pred - b||^2 over batch.

    Args:
        x_pred: [batch, n] network output (predicted solution).
        b: [batch, n] right-hand side.
        A: [batch, n, n] (or broadcastable) system matrix.

    Returns:
        Scalar mean squared residual.
    """
    if A.dim() == 2:
        Ax = x_pred @ A.T
    else:
        Ax = torch.bmm(A, x_pred.unsqueeze(-1)).squeeze(-1)
    residual = Ax - b
    return residual.pow(2).mean()


def solution_loss(x_pred: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
    """Mean squared error ||x_pred - x_star||^2."""
    return (x_pred - x_star).pow(2).mean()


def compute_solver_loss(
    net: KirchhoffNetWithIO | KirchhoffNet,
    b: torch.Tensor,
    x_star: torch.Tensor,
    A: torch.Tensor,
    ctx: SimContext,
    lambdas: dict | None = None,
    tau: float = 1.0,
    return_parts: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    reg_scale: float = 1.0,
):
    """Solver loss = residual + 0.1 * solution + all regularizers (incl. entropy).

    Same regularizer set as compute_loss (sparsity, edge_gate, node_gate,
    power, capacitance, rail, entropy). Task loss is the residual plus a
    small direct solution error to stabilize early training. ``reg_scale``
    (RR-A) applies the staged warm-up factor to all regularizers except
    entropy.

    If `amp` is True, wraps forward+loss in torch.cuda.amp.autocast for
    mixed-precision training. Caller is responsible for GradScaler.
    """
    if lambdas is None:
        lambdas = LAMBDAS

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if amp else _NullContext()
    )
    with autocast_ctx:
        out, trajs = net(b, ctx=ctx, tau=tau, store_trajectory=True)

        loss_res = residual_loss(out, b, A)
        loss_sol = solution_loss(out, x_star)
        loss_task = loss_res + 0.1 * loss_sol

        if trajs is None:
            if return_parts:
                return loss_task, loss_task.new_zeros(()), {
                    "task": float(loss_task.item()),
                    "residual": float(loss_res.item()),
                    "solution": float(loss_sol.item()),
                }
            return loss_task, loss_task.new_zeros(())

        (
            loss_sparsity,
            loss_edge_gate,
            loss_node_gate,
            loss_power,
            loss_capacitance,
            loss_rail,
            entropy_bonus,
            lambda_entropy,
        ) = _compute_regularizers(net, trajs, tau, lambdas)

        # Same split as compute_loss: rail flows through DP, structural
        # regularizers accumulate directly on `net.module` to avoid
        # being halved by DataParallel averaging.
        total_task = (
            loss_task
            + float(lambdas.get("rail", 0.0)) * loss_rail
        )
        structural = (
            + reg_scale * float(lambdas.get("sparsity", 0.0)) * loss_sparsity
            + reg_scale * float(lambdas.get("edge_gate", 0.0)) * loss_edge_gate
            + reg_scale * float(lambdas.get("node_gate", 0.0)) * loss_node_gate
            + reg_scale * float(lambdas.get("power", 0.0)) * loss_power
            + reg_scale * float(lambdas.get("capacitance", 0.0)) * loss_capacitance
            - lambda_entropy * entropy_bonus
        )

    if return_parts:
        parts = {
            "task": float(loss_task.item()),
            "residual": float(loss_res.item()),
            "solution": float(loss_sol.item()),
            "sparsity": float(loss_sparsity.item()),
            "edge_gate": float(loss_edge_gate.item()),
            "node_gate": float(loss_node_gate.item()),
            "power": float(loss_power.item()),
            "capacitance": float(loss_capacitance.item()),
            "rail": float(loss_rail.item()),
            "entropy": float(entropy_bonus.item()),
            "reg_scale": float(reg_scale),
            "total": float((total_task + structural).item()),
        }
        return total_task, structural, parts
    return total_task, structural


# ---------- tau annealing ----------

def tau_for_epoch(
    epoch: int,
    total_epochs: int | None = None,
    tau_init: float | None = None,
    tau_final: float | None = None,
) -> float:
    """Monotonic exponential decay (R6.1); smooth linear hardening in the
    last fraction of training (R5-prune-retrain-fixes).

    tau(epoch) = max(tau_min, tau_init * exp(-epoch / decay_half_life))
    for the first (1 - 2*hardening_epoch_frac) of training, then linearly
    interpolates from tau_base down to tau_final across the next
    2*hardening_epoch_frac window. The default ``hardening_epoch_frac=0.1``
    keeps the exponential floor until 80% of training, then linearly
    ramps to ``tau_final`` by 100% of training (no step jump).

    When ``tau_init`` or ``tau_final`` is provided, it overrides the
    corresponding ``TAU`` config values. This enables two-phase tau for
    pruning (R2-phase-tau): pre-prune uses ``tau_final=final_pretrain``
    to cap the floor at a gentle value, then retrain uses
    ``tau_init=final_pretrain, tau_final=final`` for a continuous
    schedule. The ``tau_min`` floor is automatically set to
    ``max(config_min, tau_final)`` to prevent non-monotonic artifacts
    when the phase target is higher than the global minimum.
    """
    if total_epochs is None:
        total_epochs = OPTIM["epochs"]
    if tau_init is None:
        tau_init = float(TAU["init"])
    if tau_final is None:
        tau_final = float(TAU["final"])
    config_min = float(TAU.get("min", 0.15))
    hardening_frac = float(TAU.get("hardening_epoch_frac", 0.1))
    decay_half_life = total_epochs * 0.5
    tau_base = max(config_min, tau_init * math.exp(-epoch / decay_half_life))
    tau_base = max(tau_base, tau_final)

    if hardening_frac <= 0.0:
        return tau_base

    hardening_start = int(total_epochs * (1.0 - 2.0 * hardening_frac))
    hardening_end = total_epochs
    if epoch < hardening_start:
        return tau_base
    if epoch >= hardening_end:
        return tau_final
    span = max(1, hardening_end - hardening_start)
    progress = (epoch - hardening_start) / span
    return tau_base + (tau_final - tau_base) * progress


# ---------- degree-budget annealing (degree-budget-topk plan) ----------

def budget_frac_for_epoch(
    epoch: int,
    total_epochs: int,
    frac_start: float = 1.0,
    frac_end: float = 0.75,
    anneal_frac: float = 0.8,
) -> float:
    """Linearly anneal the degree-budget fraction from ``frac_start`` to ``frac_end``.

    The fraction is the proportion of each node's incoming edges that the
    budget keeps open (via per-group softmax renormalization). ``frac=1.0``
    means no restriction (every group keeps all incident edges);
    ``frac=0.0`` disables the budget entirely. Annealing is spread over
    the first ``anneal_frac`` of ``total_epochs``; beyond that the value
    is clamped to ``frac_end``.

    This is a continuous float, unlike the previous integer-``k`` annealing
    that over-pruned high-degree nodes. Each group independently computes
    ``k_eff = max(1, round(count * frac))`` at the call site
    (``DifferentialStage._budget_group_mask``), so all node types get a
    uniform proportion of their incoming connections.

    Args:
        epoch: Current epoch index (0-based).
        total_epochs: Total number of training epochs (the annealing
            denominator).
        frac_start: Initial budget fraction (permissive). Default 1.0
            (no restriction at the start of training).
        frac_end: Final budget fraction (restrictive). Default 0.75
            (75% retention at the prune point).
        anneal_frac: Fraction of ``total_epochs`` over which to anneal.
            Default 0.8 (i.e. the last 20% of training freezes at frac_end).

    Returns:
        Float budget fraction for this epoch, in ``[frac_end, frac_start]``.
        Negative outputs are clamped to 0.0.
    """
    if frac_start < frac_end:
        warnings.warn(
            f"budget_frac_for_epoch: frac_start={frac_start} < frac_end={frac_end}, "
            f"annealing will become more permissive over time (likely a bug)"
        )
    if total_epochs <= 0:
        return max(0.0, float(frac_end))
    max_anneal_epochs = max(1, int(total_epochs * anneal_frac))
    if epoch >= max_anneal_epochs:
        return max(0.0, float(frac_end))
    denom = max(1, max_anneal_epochs - 1)
    alpha = epoch / denom
    frac = (1.0 - alpha) * float(frac_start) + alpha * float(frac_end)
    return max(0.0, frac)


def budget_temperature_for_epoch(
    epoch: int,
    total_epochs: int,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    anneal_frac: float = 0.8,
) -> float:
    """Linearly anneal the degree-budget softmax temperature.

    Annealing is spread over the first ``anneal_frac`` of ``total_epochs``.
    Beyond that, the value is clamped to ``temp_end`` (sharper competition).

    Smaller temperature → sharper softmax → closer to hard top-k. Larger
    temperature → softer → more uniform sharing of the budget.

    Args:
        epoch: Current epoch index (0-based).
        total_epochs: Total number of training epochs.
        temp_start: Initial temperature (typically 1.0, soft).
        temp_end: Final temperature (typically 0.1, sharp).
        anneal_frac: Fraction of ``total_epochs`` over which to anneal.

    Returns:
        Float temperature for this epoch, clamped to be > 0.
    """
    if total_epochs <= 0:
        return max(1e-6, temp_end)
    max_anneal_epochs = max(1, int(total_epochs * anneal_frac))
    if epoch >= max_anneal_epochs:
        return max(1e-6, temp_end)
    denom = max(1, max_anneal_epochs - 1)
    alpha = epoch / denom
    T = (1.0 - alpha) * temp_start + alpha * temp_end
    return max(1e-6, T)


# ---------- optimizer ----------

def make_optimizer(
    net: torch.nn.Module,
    lr: float | None = None,
    weight_decay: float | None = None,
    stage_lr_scale: float = 1.0,
    mapper_lr_scale: float = 1.0,
    struct_lr_scale: float = 1.0,
    dyn_lr_scale: float = 1.0,
):
    """Build the AdamW optimizer.

    Args:
        net: The KirchhoffNetWithIO (or KirchhoffNet) to optimize. May be
            wrapped in DataParallel; the function unwraps internally for
            name inspection but uses the same parameter tensors.
        lr: Base learning rate (default: ``OPTIM['lr']``).
        weight_decay: L2 penalty (default: ``OPTIM['weight_decay']``).
        stage_lr_scale: Per-stage geometric LR multiplier (stage-lr-scaling).
            When ``1.0`` (default), falls back to a single param group for
            backward compatibility. When ``>1.0``, the i-th stage (0-indexed
            from first) gets ``lr * stage_lr_scale ** (S - 1 - i)`` where
            S is the number of stages. Earlier stages (smaller gradient
            norm) receive proportionally larger updates to compensate for
            vanishing gradients through deep ODE stacks.

            Example with ``stage_lr_scale=10`` and 3 stages:
              stage 0 → lr × 100, stage 1 → lr × 10, stage 2 → lr × 1

            Ignored when ``struct_lr_scale != 1.0`` or ``dyn_lr_scale != 1.0``
            (flat grouping active).
        mapper_lr_scale: Multiplier on base LR for the I/O mapper parameter
            group (input_mapper + output_mapper). When ``1.0`` (default) and
            all other scales == 1.0, falls back to a single param group.
            When ``<1.0``, mappers learn more slowly — useful when mapper
            gradient norms dominate core by ~300×.
        struct_lr_scale: Multiplier on base LR for structural core params
            (``z_logits``, ``logits``, ``raw_mult``). Default 1.0 (no
            change). Use >1.0 (e.g. 2.0) to boost learning of gates and
            cell assignments. When != 1.0, uses flat global groups and
            ignores ``stage_lr_scale``.
        dyn_lr_scale: Multiplier on base LR for sensitive dynamical params
            (``raw_leak``, ``raw_drive_g``). Default 1.0 (no change). Use
            <1.0 to protect solver stability. When != 1.0, uses flat global
            groups and ignores ``stage_lr_scale``.
    """
    if lr is None:
        lr = OPTIM["lr"]
    if weight_decay is None:
        weight_decay = OPTIM["weight_decay"]
    for name, sc in [
        ("stage_lr_scale", stage_lr_scale),
        ("mapper_lr_scale", mapper_lr_scale),
        ("struct_lr_scale", struct_lr_scale),
        ("dyn_lr_scale", dyn_lr_scale),
    ]:
        if sc <= 0.0:
            raise ValueError(f"{name} must be positive, got {sc}")

    if (
        stage_lr_scale == 1.0
        and mapper_lr_scale == 1.0
        and struct_lr_scale == 1.0
        and dyn_lr_scale == 1.0
    ):
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    groups: list[dict] = []

    if struct_lr_scale != 1.0 or dyn_lr_scale != 1.0:
        if stage_lr_scale != 1.0:
            warnings.warn(
                f"stage_lr_scale={stage_lr_scale} ignored because "
                f"struct/dyn LR scales are active (flat grouping)."
            )
        struct_params: list[torch.nn.Parameter] = []
        dyn_params: list[torch.nn.Parameter] = []
        mapper_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in raw.named_parameters():
            if "input_mapper" in name or "output_mapper" in name:
                mapper_params.append(p)
            # NOTE: .z_logits MUST be checked before .logits (z_logits also ends with .logits)
            elif name.endswith(".z_logits") or name.endswith(".logits") or name.endswith(".raw_mult"):
                struct_params.append(p)
            elif name.endswith(".raw_leak") or name.endswith(".raw_drive_g"):
                dyn_params.append(p)
            else:
                other_params.append(p)
        if other_params:
            groups.append({"params": other_params, "lr": lr})
        if mapper_params:
            groups.append({"params": mapper_params, "lr": lr * mapper_lr_scale})
        if struct_params:
            groups.append({"params": struct_params, "lr": lr * struct_lr_scale})
        if dyn_params:
            groups.append({"params": dyn_params, "lr": lr * dyn_lr_scale})
    else:
        core = getattr(raw, "core", raw)
        stages = getattr(core, "stages", None)
        num_stages = len(stages) if stages is not None else 0
        if stage_lr_scale > 1.0 and num_stages <= 1:
            warnings.warn(
                f"stage_lr_scale={stage_lr_scale} has no effect with "
                f"{num_stages} stage(s); all parameters will use base LR."
            )

        stage_params: dict[int, list[torch.nn.Parameter]] = {}
        mapper_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in raw.named_parameters():
            if ".stages." in name:
                try:
                    idx = int(name.split(".stages.")[1].split(".")[0])
                except (ValueError, IndexError):
                    other_params.append(p)
                    continue
                stage_params.setdefault(idx, []).append(p)
            elif "input_mapper" in name or "output_mapper" in name:
                mapper_params.append(p)
            else:
                other_params.append(p)

        if stage_lr_scale == 1.0:
            core_params = other_params
            for stage_list in stage_params.values():
                core_params.extend(stage_list)
            if core_params:
                groups.append({"params": core_params, "lr": lr})
        else:
            for i in sorted(stage_params.keys()):
                stage_lr = lr * (stage_lr_scale ** (num_stages - 1 - i))
                groups.append({"params": stage_params[i], "lr": stage_lr})
            if other_params:
                groups.append({"params": other_params, "lr": lr})
        if mapper_params:
            groups.append({"params": mapper_params, "lr": lr * mapper_lr_scale})

    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


# ---------- one-epoch loop ----------

def apply_ablation(net, ablation: str) -> None:
    """Apply a structural ablation in-place to a KirchhoffNetWithIO (R2.1-R2.4).

    - 'none': no change.
    - 'mapper-only': set every stage's t_span to 0.0 so the ODE core is
      identity and only the I/O mappers are exercised.
    - 'empty-graph': zero out the COO edge lists of every stage so no
      inter-node coupling exists, while still exercising the I/O mappers
      and stage evolution (which trivially returns x0).
    """
    if ablation == "none":
        return
    if ablation == "mapper-only":
        for i, _ in enumerate(net.core.stages):
            net.core.stage_times[i] = 0.0
        return
    if ablation == "empty-graph":
        import torch.nn as nn
        for stage in net.core.stages:
            stage.src = stage.src.new_zeros(0)
            stage.dst = stage.dst.new_zeros(0)
            if stage.logits is not None:
                stage.logits = nn.Parameter(stage.logits.new_zeros(0, stage.logits.shape[-1]))
            if stage.raw_mult is not None:
                stage.raw_mult = nn.Parameter(stage.raw_mult.new_zeros(0))
            cell_lib = getattr(stage, 'cell_lib', None)
            if isinstance(cell_lib, SimpleEdgeLibrary):
                stage.cell_lib.param = nn.Parameter(stage.cell_lib.param.new_zeros(3, 0))
            elif isinstance(cell_lib, RealisticTanhLibrary):
                stage.cell_lib.alpha_raw = nn.Parameter(stage.cell_lib.alpha_raw.new_zeros(0))
                stage.cell_lib.bias_raw = nn.Parameter(stage.cell_lib.bias_raw.new_zeros(0))
            elif isinstance(cell_lib, RealisticTanhUpgradeLibrary):
                for name in ("alpha_raw", "gm_raw", "isat_raw", "bias_raw"):
                    setattr(
                        stage.cell_lib,
                        name,
                        nn.Parameter(getattr(stage.cell_lib, name).new_zeros(0)),
                    )
            stage.raw_leak = nn.Parameter(stage.raw_leak.new_zeros(stage.num_nodes))
            stage.z_logits = nn.Parameter(stage.z_logits.new_zeros(0))
        return
    raise ValueError(f"Unknown ablation: {ablation!r}")


def train_epoch(
    net: KirchhoffNetWithIO,
    loader,
    optimizer,
    task_fn,
    ctx_factory,
    epoch: int,
    total_epochs: int | None = None,
    lambdas: dict | None = None,
    grad_clip_norm: float | None = None,
    log_every: int = 0,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    cell_mode: str = "soft",
):
    """Run one training epoch. ctx_factory(batch_size, num_edges_total, device) -> SimContext.

    `loader` is an iterable of (u, target) batches. For our DifferentialStage
    the relevant edge count is the sum across all stages.

    Regularizers (sparsity/rail/complexity) are scheduled with
    :func:`reg_schedule` so the network learns freely in the first
    ``reg_warmup_epochs`` and the penalties anneal in linearly over the
    next ``reg_anneal_epochs`` (RR-A).

    ``cell_mode`` (four-phase-redesign/Phase 2a): forwarded to
    ``compute_loss``. ``'soft'`` uses softmax-weighted cell mixtures;
    ``'ste'`` uses one cell per edge in the forward pass with
    straight-through soft gradients.

    If `scaler` is provided, uses AMP with `scaler.scale(loss).backward()`,
    `scaler.step(optimizer)`, `scaler.update()`. Otherwise standard
    `.backward()` and `optimizer.step()`.
    """
    if grad_clip_norm is None:
        grad_clip_norm = OPTIM["grad_clip_norm"]

    tau = tau_for_epoch(epoch, total_epochs=total_epochs)
    reg_scale = reg_schedule(epoch)
    net.train()
    total_loss = 0.0
    n_batches = 0
    for i, (u, target) in enumerate(loader):
        ctx = ctx_factory(u.size(0), device=u.device)
        optimizer.zero_grad()
        loss_task, loss_structural, parts = compute_loss(
            net, u, target, ctx, task_fn, lambdas=lambdas, tau=tau,
            return_parts=True, amp=amp, reg_scale=reg_scale,
            cell_mode=cell_mode,
        )
        if scaler is not None:
            ( scaler.scale(loss_task) + scaler.scale(loss_structural) ).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            (loss_task + loss_structural).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        total_loss += float((loss_task + loss_structural).item())
        n_batches += 1
        if log_every and (i % log_every == 0):
            print(f"  epoch {epoch} batch {i} loss={parts['total']:.4f} task={parts['task']:.4f} reg_scale={reg_scale:.2f}")
    return {"avg_loss": total_loss / max(1, n_batches), "tau": tau, "reg_scale": reg_scale}


# ---------- convenience ctx factory ----------

def default_ctx_factory(net: KirchhoffNetWithIO):
    """Build a ctx_factory closure tied to a specific net and device.

    Usage:
        factory = default_ctx_factory(net)
        ctx = factory(batch_size=128, device=u.device)

    Note (RR-C): ``temp_choices`` is no longer passed — ``temp_c`` is
    deprecated and ``sample_random_context`` always returns
    ``VARIATION["temp_c_default"]``.
    """
    def _factory(batch_size: int, device: torch.device | str = "cpu", **kwargs):
        total_edges = sum(s.num_edges() for s in net.core.stages)
        return sample_random_context(
            num_edges=total_edges,
            num_cells=net.core.stages[0].cell_lib.num_cells,
            device=device,
            **{
                "gain_shift_std": VARIATION["global_gain_shift_std"],
                "mismatch_std": VARIATION["edge_mismatch_std"],
                "global_isat_shift_std": VARIATION.get("global_isat_shift_std", 0.0),
                "isat_mismatch_std": VARIATION.get("edge_isat_mismatch_std", 0.0),
            },
        )
    return _factory
