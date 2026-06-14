"""Digital baseline solvers for comparison against the KirchhoffNet solver.

Implements batched Jacobi and conjugate gradient (CG) iteration. Both accept
A: [batch, n, n], b: [batch, n] and run for a fixed number of iterations,
matching the analog network's ODE step budget for a fair comparison.
"""

from __future__ import annotations

import numpy as np
import torch


__all__ = ["DigitalSolverBaseline", "compare_against_baselines"]


class DigitalSolverBaseline:
    """Batched digital iterative solvers for Ax = b.

    Args:
        n: matrix dimension (assumed square and same for all matrices).
        max_iters: default iteration cap.
    """

    def __init__(self, n: int, max_iters: int = 50) -> None:
        self.n = int(n)
        self.max_iters = int(max_iters)

    def jacobi(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Standard Jacobi iteration: x_{k+1} = D^{-1}(b - R x_k)."""
        if steps is None:
            steps = self.max_iters
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        d = torch.diagonal(A, dim1=-2, dim2=-1)
        d_safe = d.clamp_min(1e-12)
        residuals: list[float] = []
        for _ in range(int(steps)):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            x = (b - Ax + d * x) / d_safe
            r = (torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b).norm(dim=-1)
            residuals.append(float(r.mean().item()))
        return x, residuals

    def conjugate_gradient(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        steps: int | None = None,
        tol: float = 1e-6,
    ) -> tuple[torch.Tensor, list[float]]:
        """Batched conjugate gradient iteration. Stops early on convergence."""
        if steps is None:
            steps = self.max_iters
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        r = b - torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        p = r.clone()
        rs_old = (r * r).sum(dim=-1)

        residuals: list[float] = []
        for _ in range(int(steps)):
            Ap = torch.bmm(A, p.unsqueeze(-1)).squeeze(-1)
            denom = (p * Ap).sum(dim=-1).clamp_min(1e-12)
            alpha = rs_old / denom
            x = x + alpha.unsqueeze(-1) * p
            r = r - alpha.unsqueeze(-1) * Ap
            rs_new = (r * r).sum(dim=-1)
            r_norm_mean = float(rs_new.sqrt().mean().item())
            residuals.append(r_norm_mean)
            if r_norm_mean < tol:
                break
            beta = rs_new / rs_old.clamp_min(1e-12)
            p = r + beta.unsqueeze(-1) * p
            rs_old = rs_new
        return x, residuals


def compare_against_baselines(
    net,
    val_loader,
    n: int,
    device: str = "cpu",
    max_iters: int = 50,
) -> dict:
    """Run one validation epoch and compare network vs. Jacobi and CG.

    Returns a dict of lists of per-sample residuals and solution errors.
    """
    baseline = DigitalSolverBaseline(n=n, max_iters=max_iters)
    metrics = {
        "net_res": [], "net_sol": [],
        "jacobi_res": [], "jacobi_sol": [],
        "cg_res": [], "cg_sol": [],
    }

    net.eval()
    with torch.no_grad():
        for b_batch, x_star_batch, A_batch in val_loader:
            b_batch = b_batch.to(device)
            x_star_batch = x_star_batch.to(device)
            A_batch = A_batch.to(device)

            x_net, _ = net(b_batch, ctx=None, tau=0.5, store_trajectory=False)
            metrics["net_res"].extend(
                (torch.bmm(A_batch, x_net.unsqueeze(-1)).squeeze(-1) - b_batch)
                .norm(dim=-1).cpu().numpy()
            )
            metrics["net_sol"].extend(
                (x_net - x_star_batch).norm(dim=-1).cpu().numpy()
            )

            x_jac, _ = baseline.jacobi(A_batch, b_batch, steps=max_iters)
            metrics["jacobi_res"].extend(
                (torch.bmm(A_batch, x_jac.unsqueeze(-1)).squeeze(-1) - b_batch)
                .norm(dim=-1).cpu().numpy()
            )
            metrics["jacobi_sol"].extend(
                (x_jac - x_star_batch).norm(dim=-1).cpu().numpy()
            )

            x_cg, _ = baseline.conjugate_gradient(A_batch, b_batch, steps=max_iters)
            metrics["cg_res"].extend(
                (torch.bmm(A_batch, x_cg.unsqueeze(-1)).squeeze(-1) - b_batch)
                .norm(dim=-1).cpu().numpy()
            )
            metrics["cg_sol"].extend(
                (x_cg - x_star_batch).norm(dim=-1).cpu().numpy()
            )

    print(f"{'Method':<12} {'Mean Residual':<18} {'Mean Sol Error':<18}")
    print("-" * 50)
    for method in ("net", "jacobi", "cg"):
        mean_res = float(np.mean(metrics[f"{method}_res"]))
        mean_sol = float(np.mean(metrics[f"{method}_sol"]))
        print(f"{method:<12} {mean_res:<18.4e} {mean_sol:<18.4e}")

    return metrics
