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

import torch
import torch.nn.functional as F

from config import (
    LAMBDAS,
    OPTIM,
    SCHEDULE_THREE_PHASE,
    SOLVER,
    TAU,
    Z_INDEX,
    VARIATION,
)
from sim_context import SimContext, sample_random_context
from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO


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
    "compute_solidification_metrics",
    "validate_argmax",
    "make_optimizer",
    "apply_ablation",
    "train_epoch",
]


# ---------- regularizers ----------

def _stage_soft_weights(stage) -> torch.Tensor:
    return F.softmax(stage.logits, dim=-1)


def _stage_multiplicities(stage) -> torch.Tensor:
    return F.softplus(stage.raw_mult)


def _stage_edge_gates(stage) -> torch.Tensor:
    """Edge gate values z_e = σ(z_logits), shape [E]."""
    return torch.sigmoid(stage.z_logits)


def _stage_node_gates(stage) -> torch.Tensor:
    """Node gate values u_j = σ(u_logits), shape [N]."""
    return torch.sigmoid(stage.u_logits)


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
    """Shared regularizer computation: sparsity, edge_gate, node_gate, power,
    capacitance, rail, entropy.

    The four complexity terms (CP-4) decompose the previous single ``complexity``
    proxy into per-component hardware terms:
      - edge_gate   : Σ_e σ(z_logits)                       (active edge count)
      - node_gate   : Σ_j σ(u_logits)                       (active node count)
      - power       : Σ_e z_e · m_e · Σ_q w_q · gm_q       (static power proxy)
      - capacitance : C_eff · Σ_j u_j                      (capacitance area proxy)

    The cell-selection sparsity ``Σ w[:, :Z_INDEX]`` is preserved (pushes
    individual edges toward the zero-current Z cell as a fine-grained
    selection, distinct from the edge gate which is a coarse on/off).

    Returns ``(loss_sparsity, loss_edge_gate, loss_node_gate, loss_power,
    loss_capacitance, loss_rail, entropy_bonus, lambda_entropy)``.
    """
    stages = net.core.stages if isinstance(net, KirchhoffNetWithIO) else net.stages
    loss_sparsity = loss_edge_gate = loss_node_gate = loss_power = loss_capacitance = loss_rail = entropy_bonus = trajs[0].new_zeros(())
    for stage, traj in zip(stages, trajs):
        w = _stage_soft_weights(stage)
        mult = _stage_multiplicities(stage)
        z = _stage_edge_gates(stage)
        u = _stage_node_gates(stage)

        # Cell-selection sparsity: push toward Z cell.
        loss_sparsity = loss_sparsity + w[:, :Z_INDEX].sum()

        # Edge/node gate penalties: soft count of active edges/nodes.
        loss_edge_gate = loss_edge_gate + z.sum()
        loss_node_gate = loss_node_gate + u.sum()

        # Static power proxy: z_e · m_e · weighted gm sum per edge.
        # gm values per cell come from the shared cell library.
        gm_per_cell = stage.cell_lib.gm_values()  # [Q]
        effective_gm = (w * gm_per_cell.unsqueeze(0)).sum(dim=-1)  # [E]
        loss_power = loss_power + (z * mult * effective_gm).sum()

        # Capacitance area proxy: C_eff · Σ_j u_j.
        loss_capacitance = loss_capacitance + stage.c_eff * u.sum()

        loss_rail = loss_rail + _stage_rail_loss(stage, traj)

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


# ---------- solidification monitoring (three-phase-schedule plan) ----------

def compute_solidification_metrics(
    net: KirchhoffNetWithIO | KirchhoffNet,
    tau: float = 1.0,
) -> dict:
    """Compute edge/node solidification metrics across all stages.

    Returns a dict of Python floats (safe to log to a text file):
      - ``mean_max_cell_prob``: mean over all edges of max(softmax(logits/τ)).
        1.0 = fully discrete cell selection; ~0.25 (1/4 cells) = uniform.
      - ``mean_pZ``: mean over all edges of softmax(logits/τ)[:, Z_INDEX].
        Probability mass on the zero-current Z cell.
      - ``mean_sigma_z``: mean over all edges of σ(z_logits). Average edge
        gate openness (0..1).
      - ``frac_sigma_z_below_01/005/001``: fraction of edge gates with
        σ(z_logits) below the listed threshold. Measures how many edges
        are effectively off (eligible for pruning).
      - ``mean_sigma_u``: mean over all nodes of σ(u_logits). Average
        node gate openness.
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
        n_edges = int(stage.logits.shape[0])
        n_nodes = int(stage.u_logits.shape[0]) if hasattr(stage, "u_logits") and stage.u_logits.numel() > 0 else 0
        if n_edges == 0:
            total_nodes += n_nodes
            continue

        weights = F.softmax(stage.logits / float(tau), dim=-1)
        max_probs, _ = weights.max(dim=-1)
        p_z = weights[:, Z_INDEX]
        sigma_z = torch.sigmoid(stage.z_logits)
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
            out, _ = net(u, ctx=ctx, tau=argmax_tau, store_trajectory=False)
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
):
    """Compute total loss = task + regularizers + entropy bonus.

    If net is a KirchhoffNetWithIO, x0 is the raw input u. If it is a plain
    KirchhoffNet, x0 is the already-bounded initial differential state.

    ``reg_scale`` (RR-A) is a multiplicative factor on sparsity/complexity/
    rail: pass ``reg_schedule(epoch)`` from the training loop to implement
    staged warm-up. Defaults to 1.0 (no warm-up).

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
            out, trajs = net(x0, ctx=ctx, tau=tau, store_trajectory=True)
        else:
            out, trajs = net(x0, ctx=ctx, tau=tau, store_trajectory=True)

        loss_task = task_fn(out, target)

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
        # ONLY the data-dependent pieces (task + rail) and does NOT
        # halve the structural regularizers (sparsity/edge_gate/
        # node_gate/power/capacitance/entropy) which depend only on
        # `net.module`'s parameters and have no business being
        # averaged across replicas.  The caller is expected to call
        # `backward(retain_graph=True)` on `total_task` and then
        # `backward()` on `structural` so the structural gradients
        # accumulate directly onto `net.module`'s parameters.
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


# ---------- optimizer ----------

def make_optimizer(
    net: torch.nn.Module,
    lr: float | None = None,
    weight_decay: float | None = None,
    stage_lr_scale: float = 1.0,
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
    """
    if lr is None:
        lr = OPTIM["lr"]
    if weight_decay is None:
        weight_decay = OPTIM["weight_decay"]
    if stage_lr_scale <= 0.0:
        raise ValueError(
            f"stage_lr_scale must be positive, got {stage_lr_scale}"
        )

    if stage_lr_scale == 1.0:
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    stage_params: dict[int, list[torch.nn.Parameter]] = {}
    other_params: list[torch.nn.Parameter] = []

    core = getattr(raw, "core", raw)
    stages = getattr(core, "stages", None)
    num_stages = len(stages) if stages is not None else 0
    if stage_lr_scale > 1.0 and num_stages <= 1:
        import warnings
        warnings.warn(
            f"stage_lr_scale={stage_lr_scale} has no effect with "
            f"{num_stages} stage(s); all parameters will use base LR."
        )

    for name, p in raw.named_parameters():
        if ".stages." in name:
            try:
                idx = int(name.split(".stages.")[1].split(".")[0])
            except (ValueError, IndexError):
                other_params.append(p)
                continue
            stage_params.setdefault(idx, []).append(p)
        else:
            other_params.append(p)

    groups: list[dict] = []
    for i in sorted(stage_params.keys()):
        stage_lr = lr * (stage_lr_scale ** (num_stages - 1 - i))
        groups.append({"params": stage_params[i], "lr": stage_lr})
    if other_params:
        groups.append({"params": other_params, "lr": lr})

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
            stage.logits = nn.Parameter(stage.logits.new_zeros(0, stage.logits.shape[-1]))
            stage.raw_mult = nn.Parameter(stage.raw_mult.new_zeros(0))
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
):
    """Run one training epoch. ctx_factory(batch_size, num_edges_total, device) -> SimContext.

    `loader` is an iterable of (u, target) batches. For our DifferentialStage
    the relevant edge count is the sum across all stages.

    Regularizers (sparsity/rail/complexity) are scheduled with
    :func:`reg_schedule` so the network learns freely in the first
    ``reg_warmup_epochs`` and the penalties anneal in linearly over the
    next ``reg_anneal_epochs`` (RR-A).

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
        )
        if scaler is not None:
            scaler.scale(loss_task).backward(retain_graph=True)
            scaler.scale(loss_structural).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_task.backward(retain_graph=True)
            loss_structural.backward()
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
            },
        )
    return _factory
