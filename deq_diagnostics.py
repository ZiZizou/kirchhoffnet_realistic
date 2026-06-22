"""DEQ diagnostics (deq-core-prototype plan).

Lightweight helpers for:
  1. Gradient-norm comparison: DEQ-implicit vs Heun-BPTT on a single batch.
     Returns a dict with absolute L1 norms of z_logits and logits gradients
     under each solver, plus the DEQ/BPTT ratio. The Kimi note predicts
     DEQ should lift z_logits gradient norms by 2-4 orders of magnitude.
  2. Jacobian conditioning: cond(d(rhs)/dx) at the equilibrium state. Uses
     torch.func.jacrev for batched-vectorised estimation.
  3. Multistart uniqueness: solve DEQ from several random initial guesses
     and report max pairwise distance between converged states.

All helpers accept a fully-built DifferentialStage and run on whatever
device/dtype the stage is on; Jacobians are estimated on a single batch
element to keep cost bounded.
"""

from __future__ import annotations

import math
import torch


__all__ = [
    "gradient_norm_compare",
    "estimate_jacobian_cond",
    "multistart_uniqueness",
]


def _detach_clone(t):
    return t.detach().clone()


def gradient_norm_compare(
    stage,
    x0: torch.Tensor,
    ctx,
    tau: float = 1.0,
    cell_mode: str = "soft",
    x_drive: torch.Tensor | None = None,
    drive_scale: float = 0.0,
    leak_floor: float = 0.05,
    deq_cfg: dict | None = None,
    bptt_t_span: float = 0.3,
    bptt_num_steps: int = 10,
) -> dict:
    """Compare L1 gradient norms of z_logits and logits under BPTT vs DEQ.

    A single zero-mean squared-loss L = sum(x**2) is used so the comparison
    isolates solver behaviour from any task-specific loss.
    """
    deq_cfg_use = dict(deq_cfg or {})
    deq_cfg_use.setdefault("leak_floor", leak_floor)

    # --- Heun BPTT path ---
    if any(p.grad is not None for p in stage.parameters()):
        for p in stage.parameters():
            if p.grad is not None:
                p.grad = None
    stage.set_leak_floor(0.0)
    x_heun, _ = stage.forward(
        x0=x0, ctx=ctx, t_span=bptt_t_span, num_steps=bptt_num_steps,
        tau=tau, store_trajectory=False, cell_mode=cell_mode,
        x_drive=x_drive, drive_scale=drive_scale, solver="heun",
    )
    loss_h = (x_heun ** 2).sum()
    loss_h.backward()
    z_heun = float(stage.z_logits.grad.detach().abs().sum().item()) if stage.z_logits.grad is not None else 0.0
    logits_heun = float(stage.logits.grad.detach().abs().sum().item()) if (stage.logits is not None and stage.logits.grad is not None) else 0.0
    for p in stage.parameters():
        if p.grad is not None:
            p.grad = None

    # --- DEQ path ---
    x_deq, _ = stage.forward_equilibrium(
        x0=x0, ctx=ctx, tau=tau, cell_mode="soft",
        x_drive=x_drive, drive_scale=drive_scale, deq_cfg=deq_cfg_use,
    )
    loss_d = (x_deq ** 2).sum()
    loss_d.backward()
    z_deq = float(stage.z_logits.grad.detach().abs().sum().item()) if stage.z_logits.grad is not None else 0.0
    logits_deq = float(stage.logits.grad.detach().abs().sum().item()) if (stage.logits is not None and stage.logits.grad is not None) else 0.0
    for p in stage.parameters():
        if p.grad is not None:
            p.grad = None

    return {
        "z_logits_heun": z_heun,
        "z_logits_deq": z_deq,
        "logits_heun": logits_heun,
        "logits_deq": logits_deq,
        "z_logits_ratio": z_deq / max(z_heun, 1e-30),
        "logits_ratio": logits_deq / max(logits_heun, 1e-30),
    }


def estimate_jacobian_cond(
    stage,
    x_star: torch.Tensor,
    ctx,
    tau: float = 1.0,
    cell_mode: str = "soft",
    x_drive: torch.Tensor | None = None,
    drive_scale: float = 0.0,
    leak_floor: float = 0.05,
) -> float:
    """Estimate cond(J), J = d(rhs)/dx at x_star, on batch element 0.

    Uses torch.func.jacrev on a single batch slice to keep cost bounded.
    Returns ``float('inf')`` if the Jacobian is singular.
    """
    stage.set_leak_floor(leak_floor)

    if x_star.dim() == 2:
        x = x_star[0].clone().requires_grad_(True)
        batch = x.unsqueeze(0)
    else:
        x = x_star.clone().requires_grad_(True)
        batch = x

    def f(inp):
        return stage.rhs(inp, ctx=ctx, tau=tau, cell_mode=cell_mode,
                         x_drive=x_drive, drive_scale=drive_scale)

    try:
        J = torch.func.jacrev(f)(batch).squeeze(0).squeeze(0)
        # J: [N, N]
        s = torch.linalg.svdvals(J.to(dtype=torch.float32))
        s_max = float(s.max().item())
        s_min = float(s.min().item())
        if s_min <= 0.0 or not math.isfinite(s_min):
            return float("inf")
        cond = s_max / s_min
        stage.set_leak_floor(0.0)
        return cond
    finally:
        stage.set_leak_floor(0.0)


def multistart_uniqueness(
    stage,
    ctx,
    tau: float = 1.0,
    cell_mode: str = "soft",
    x_drive: torch.Tensor | None = None,
    drive_scale: float = 0.0,
    leak_floor: float = 0.05,
    deq_cfg: dict | None = None,
    starts: list[float] | None = None,
    batch_shape: tuple[int, int] = (2, 4),
) -> dict:
    """Solve DEQ from several initial guesses and report pairwise spread.

    Returns dict with ``max_pairwise_diff`` (max L_inf distance across all
    pairs of converged states) and ``converged_states`` (list of tensors).
    Large values indicate multistability; small values (under ~1e-3) suggest
    a unique contractive equilibrium.
    """
    starts = starts if starts is not None else [-1.0, 0.0, 1.0, 5.0]
    cfg = dict(deq_cfg or {})
    cfg.setdefault("leak_floor", leak_floor)
    finals = []
    stage_param = next(stage.parameters())
    for v in starts:
        x0 = torch.full(
            batch_shape,
            float(v),
            device=stage_param.device,
            dtype=stage_param.dtype,
        )
        x_star, _ = stage.forward_equilibrium(
            x0=x0, ctx=ctx, tau=tau, cell_mode="soft",
            x_drive=x_drive, drive_scale=drive_scale, deq_cfg=cfg,
        )
        finals.append(x_star.detach().clone())
    pairwise = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            pairwise.append(float((finals[i] - finals[j]).abs().max().item()))
    return {
        "max_pairwise_diff": max(pairwise) if pairwise else 0.0,
        "converged_states": finals,
        "starts": list(starts),
    }
