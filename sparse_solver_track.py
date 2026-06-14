"""Convergence diagnostic tracker for the sparse solver ODE trajectory.

Captures (time, state, label) snapshots and plots residual ||Ax - b|| and
solution error ||x - x*|| as the ODE integrates. Useful for visualizing
how the analog network relaxes to the solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


__all__ = ["ConvergenceTracker"]


class ConvergenceTracker:
    """Stores ODE state snapshots and produces diagnostic plots."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[float, torch.Tensor, str]] = []

    def capture(self, t: float, x: torch.Tensor, label: str = "net") -> None:
        """Record a single snapshot of the state at time `t`."""
        self.snapshots.append((float(t), x.detach().cpu().clone(), label))

    def reset(self) -> None:
        self.snapshots.clear()

    def plot_residual_trajectory(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x_star: torch.Tensor,
        save_path: str | None = None,
    ):
        """3-panel plot: residual vs time, solution error vs time, final |x_j|."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.snapshots:
            raise RuntimeError("No snapshots captured; call capture() first")

        times, resids, sol_errs, labels = [], [], [], []
        for t, x, lbl in self.snapshots:
            xv = x.squeeze()
            Ax = A @ xv
            r = (Ax - b).norm().item()
            e = (xv - x_star).norm().item()
            times.append(t)
            resids.append(r)
            sol_errs.append(e)
            labels.append(lbl)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for lbl in sorted(set(labels)):
            mask = [i for i, lb in enumerate(labels) if lb == lbl]
            axes[0].plot(
                [times[i] for i in mask], [resids[i] for i in mask],
                marker="o", label=lbl, markersize=3,
            )
        axes[0].set_xlabel("ODE Time")
        axes[0].set_ylabel("||Ax - b||")
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[0].set_title("Residual Evolution")
        axes[0].grid(True, which="both", ls="--", alpha=0.5)

        for lbl in sorted(set(labels)):
            mask = [i for i, lb in enumerate(labels) if lb == lbl]
            axes[1].plot(
                [times[i] for i in mask], [sol_errs[i] for i in mask],
                marker="o", label=lbl, markersize=3,
            )
        axes[1].set_xlabel("ODE Time")
        axes[1].set_ylabel("||x - x*||")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].set_title("Solution Error")
        axes[1].grid(True, which="both", ls="--", alpha=0.5)

        _, x_final, _ = self.snapshots[-1]
        xv = x_final.squeeze().numpy()
        axes[2].bar(range(len(xv)), np.sort(np.abs(xv))[::-1])
        axes[2].set_xlabel("Sorted Component Index")
        axes[2].set_ylabel("|x_j|")
        axes[2].set_title("Final State Magnitude Distribution")

        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
