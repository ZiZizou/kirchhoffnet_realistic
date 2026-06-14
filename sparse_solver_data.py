"""Sparse linear system dataset for the KirchhoffNet solver benchmark.

Generates random sparse symmetric positive-definite matrices A with controlled
condition number and density, then samples bounded random solutions x* and
computes the right-hand side b = A @ x*.

This guarantees exact ground truth (no numerical inversion) and keeps the
solver's job clean: learn the forward mapping b -> A^{-1}b.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import rand as sparse_rand


__all__ = ["generate_sparse_spd", "SparseLinearSystemDataset"]


def generate_sparse_spd(
    n: int = 48,
    density: float = 0.03,
    cond_target: float = 1e2,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Generate a sparse symmetric positive-definite matrix.

    Strategy:
      1. Sample a random sparse symmetric pattern S via scipy.sparse.rand.
      2. Make it diagonally dominant with a margin of U[1.0, 3.0].
      3. Add a low-rank perturbation U Sigma U^T to spread eigenvalues to
         match cond_target.

    Args:
        n: matrix dimension.
        density: fraction of off-diagonal nonzeros (before symmetrization).
        cond_target: approximate target condition number.
        seed: numpy seed for reproducibility.

    Returns:
        A_dense: [n, n] float32 torch tensor.
        actual_cond: achieved condition number.
    """
    rng = np.random.RandomState(seed)

    S = sparse_rand(n, n, density=density / 2, random_state=rng)
    S = S + S.T

    row_sums = np.abs(S).sum(axis=1)
    if hasattr(row_sums, "A1"):
        row_sums = row_sums.A1
    else:
        row_sums = np.asarray(row_sums).flatten()
    diag_vals = row_sums + rng.uniform(1.0, 3.0, size=n)

    A = S.copy()
    A.setdiag(diag_vals)

    k = max(2, n // 16)
    U = rng.randn(n, k)
    sigma = np.logspace(0, np.log10(cond_target), k)
    A = A + U @ np.diag(sigma) @ U.T

    A = (A + A.T) / 2
    A = A + n * np.eye(n) * 1e-3

    if hasattr(A, "toarray"):
        A_arr = np.asarray(A.toarray())
    elif hasattr(A, "A"):
        A_arr = np.asarray(A.A)
    else:
        A_arr = np.asarray(A)
    A_dense = torch.from_numpy(A_arr).float()

    eigs = torch.linalg.eigvalsh(A_dense)
    assert eigs.min() > 0, "Matrix not SPD"
    actual_cond = float(eigs.max() / eigs.min())
    return A_dense, actual_cond


class SparseLinearSystemDataset(torch.utils.data.Dataset):
    """Random sparse SPD linear system dataset.

    Each sample holds (A, b, x_star) where A is sparse SPD, x_star is bounded
    by `x_max * tanh(randn)`, and b = A @ x_star.

    Args:
        n: matrix dimension.
        num_samples: number of independent systems to generate.
        density: sparsity parameter passed to generate_sparse_spd.
        cond_target: target condition number.
        x_max: bound on the random ground-truth solution.
        seed: base seed (per-sample seed is base+idx).
    """

    def __init__(
        self,
        n: int = 48,
        num_samples: int = 5000,
        density: float = 0.03,
        cond_target: float = 1e2,
        x_max: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.n = int(n)
        self.x_max = float(x_max)
        self.samples: list[dict] = []

        for i in range(int(num_samples)):
            A, cond = generate_sparse_spd(
                n=n, density=density, cond_target=cond_target, seed=seed + i
            )
            x_star = self.x_max * torch.tanh(torch.randn(self.n))
            b = A @ x_star
            self.samples.append({"A": A, "b": b, "x_star": x_star, "cond": cond})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s["b"], s["x_star"], s["A"]
