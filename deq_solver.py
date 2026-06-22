"""DEQ (Deep Equilibrium) solver adapter.

Stagewise fixed-point solver for the reduced differential KirchhoffNet
(deq-core-prototype plan). Wraps torchdeq with a small, swappable API so the
rest of the codebase can target a single ``solve_equilibrium(phi, x0, cfg) ->
(x_star, info)`` interface.

The default backend is ``torchdeq.get_deq`` (Anderson / Broyden solver with
implicit / phantom-gradient backward). If torchdeq is unavailable on the
installed torch version, a built-in fixed-point-iteration fallback is used so
training never crashes on import. The fallback is intentionally simple (no
implicit backward) and only adequate for sanity checks; production runs should
rely on the torchdeq backend.

Usage:

    from deq_solver import solve_equilibrium

    def phi(x):
        return x + dt * rhs(x, ...)

    cfg = {"f_max_iter": 30, "f_tol": 1e-5, "b_max_iter": 20, "deq_step": 0.1}
    x_star, info = solve_equilibrium(phi, x0, cfg)
"""

from __future__ import annotations

from typing import Callable

import torch


__all__ = ["solve_equilibrium", "available_backends"]


_TORCHDEQ_AVAILABLE: bool | None = None


def _has_torchdeq() -> bool:
    global _TORCHDEQ_AVAILABLE
    if _TORCHDEQ_AVAILABLE is None:
        try:
            import torchdeq  # noqa: F401
            from torchdeq import get_deq  # noqa: F401
            _TORCHDEQ_AVAILABLE = True
        except Exception:
            _TORCHDEQ_AVAILABLE = False
    return _TORCHDEQ_AVAILABLE


def available_backends() -> list[str]:
    """List of available solver backends in priority order."""
    return ["torchdeq", "fixed_point_iter"] if _has_torchdeq() else ["fixed_point_iter"]


def _solve_torchdeq(phi, x0, cfg):
    from torchdeq import get_deq

    f_solver = cfg.get("f_solver", "anderson")
    b_solver = cfg.get("b_solver", "anderson")
    f_max_iter = int(cfg.get("f_max_iter", 30))
    f_tol = float(cfg.get("f_tol", 1e-4))
    b_max_iter = int(cfg.get("b_max_iter", 20))
    anderson_m = int(cfg.get("anderson_m", 5))

    deq = get_deq(
        f_solver=f_solver,
        b_solver=b_solver,
        f_max_iter=f_max_iter,
        f_tol=f_tol,
        b_max_iter=b_max_iter,
        anderson_m=anderson_m,
    )

    # Promote the initial guess to fp32 for stable fixed-point iteration.
    # torchdeq's DEQ.forward runs the solver iteration under torch.no_grad()
    # internally and re-enters autograd for the IFT backward pass. We must NOT
    # wrap the deq() call in torch.no_grad() ourselves or the implicit-grad
    # graph breaks (the returned tensors lose their grad_fn).
    #
    # However, we DO disable autocast here so the solver (Anderson iteration
    # + IFT backward) runs in fp32 regardless of the caller's autocast state.
    # fp16 in the Anderson solver destabilises the fixed-point iteration and
    # corrupts the IFT linear solve.
    x0_f32 = x0.to(dtype=torch.float32)
    with torch.autocast(device_type='cuda', enabled=False):
        z_out_list, info = deq(phi, x0_f32)

    if isinstance(z_out_list, list):
        x_star = z_out_list[-1]
    else:
        x_star = z_out_list
    # Cast to fp32 explicitly so AMP-safe behaviour is preserved regardless
    # of x0's input dtype. Callers should cast to their preferred dtype.
    x_star = x_star.to(dtype=torch.float32)

    def _get(key, default):
        if isinstance(info, dict):
            return info.get(key, default)
        return getattr(info, key, default)

    nstep = _get("nstep", None)
    if nstep is None:
        nstep_val = int(f_max_iter)
    elif torch.is_tensor(nstep):
        nstep_val = int(nstep.flatten()[0].item())
    else:
        nstep_val = int(nstep)

    rel_residual_t = _get("rel_lowest", _get("abs_lowest", None))
    if rel_residual_t is None:
        rel_res = float("nan")
    elif torch.is_tensor(rel_residual_t):
        rel_res = float(rel_residual_t.max().item())
    else:
        rel_res = float(rel_residual_t)

    return x_star, {"nstep": nstep_val, "rel_residual": rel_res}


def _solve_fixed_point_iter(phi, x0, cfg):
    """Naive fixed-point iteration fallback (no implicit backward)."""
    f_max_iter = int(cfg.get("f_max_iter", 30))
    f_tol = float(cfg.get("f_tol", 1e-4))
    x = x0.to(dtype=torch.float32).clone()
    nstep = 0
    rel_res = float("inf")
    for nstep in range(1, f_max_iter + 1):
        x_next = phi(x)
        diff = (x_next - x).flatten()
        denom = x.flatten().abs().clamp_min(1e-8)
        rel_res = float((diff.abs() / denom).max().item()) if denom.numel() > 0 else float("nan")
        x = x_next
        if rel_res < f_tol:
            break
    return x.to(dtype=x0.dtype), {"nstep": nstep, "rel_residual": rel_res}


def solve_equilibrium(
    phi: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    """Solve x* = phi(x*) starting from x0.

    Parameters
    ----------
    phi : callable
        Fixed-point map Phi(x) -> x. Must be differentiable w.r.t. inputs and
        parameters when the torchdeq backend is used so implicit gradients are
        valid.
    x0 : torch.Tensor
        Initial guess. Any dtype; the solver runs in fp32.
    cfg : dict
        Solver configuration. Recognized keys:
          - f_solver: "anderson" or "broyden" (torchdeq)
          - b_solver: same choices for backward
          - f_max_iter, b_max_iter, f_tol
          - anderson_m
          - backend: "torchdeq" | "fixed_point_iter" (default: auto)
          - deq_step: reserved (currently informational; stage-level caller
            applies the damping outside the solver)

    Returns
    -------
    x_star : torch.Tensor
        The converged fixed point, same shape and dtype as ``x0``.
    info : dict
        ``{"nstep": int, "rel_residual": float}``
    """
    backend = cfg.get("backend", "auto")
    if backend == "auto":
        backend = "torchdeq" if _has_torchdeq() else "fixed_point_iter"

    if backend == "torchdeq":
        if not _has_torchdeq():
            raise RuntimeError(
                "deq_solver: backend='torchdeq' requested but torchdeq is not installed. "
                "Install torchdeq or pass backend='fixed_point_iter'."
            )
        return _solve_torchdeq(phi, x0, cfg)
    if backend == "fixed_point_iter":
        return _solve_fixed_point_iter(phi, x0, cfg)
    raise ValueError(f"deq_solver: unknown backend {backend!r}")