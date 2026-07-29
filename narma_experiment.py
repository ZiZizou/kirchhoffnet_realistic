"""NARMA reservoir benchmark — the discriminator experiment.

This script is the entry point for the NARMA (Nonlinear AutoRegressive Moving
Average) reservoir benchmark on the KirchhoffNet fabric. It is the existential
discriminator for the evolving nonlinear core:

    - If ``freeze_read=True`` (frozen core: boundary + temporal-readout OTA
      edges only) matches ``freeze_read=False`` (evolving core: all internal
      edges dynamic) on NARMA-20, the evolving nonlinear core is dead.
      Commit fully to the control-path paper.

    - If ``freeze_read=False`` clearly wins (expected NRMSE ~0.1–0.2 vs ~0.3+
      on NARMA-20), the signal-path branch stays alive.

Usage:
    python narma_experiment.py --order 10 [--seeds 0,1,2,3,4] [--output ./out]
    python narma_experiment.py --order 20 [--seeds 0,1,2,3,4]
    python narma_experiment.py --order 10 --baselines-only   # skip fabric training

Three families of conditions are evaluated:

    1. **Baselines** (no fabric, no temporal trickery):
       a. Ridge regression on 30 tapped delays of u(n).
       b. Echo State Network (ESN), 25 nodes, ridge readout.
       c. MLP on 30-tap window (25 hidden, 801 params).
       d. MLP_large on 30-tap window (237 hidden, 7,585 params ≈ KNet).
       e. LSTM on streaming u(t) (grid search hidden_dim=[16,25,32], lr=[1e-3,5e-4]).
       f. LSTM_large on streaming u(t) (42 hidden, 7,603 params ≈ KNet).

    2. **Fabric conditions**:
       g. ``freeze_read=False``: evolving nonlinear core.
       h. ``freeze_read=True``: frozen core (boundary + temporal-readout only).

    3. **Memory capacity** (linear readouts reconstructing u(n-k) for k=1..20):
       Computed for conditions g and h. ``MC = sum_k R^2_k`` directly
       characterises the fading-memory property.

Training protocol (rewritten 2026-07-29 to fix three critical bugs):
    - ``t_span=1.0`` per sample (was 7.0). With leak~0.05, t_span=7.0 gives
      per-sample decay exp(-0.05*7)~0.71 → ~3 sample memory horizon (insufficient
      for NARMA-10). t_span=1.0 → exp(-0.05)~0.95 → ~20 sample memory horizon.
    - Sequential truncated BPTT: ``B`` parallel independent NARMA streams
      (``--n-streams``), state carried across chunks, ``detach()`` only at
      chunk boundaries (``--tbptt-chunk`` samples per chunk). Each chunk is
      one optimizer step. With B=4 streams × 2500 samples / 25 chunk = 100
      optimizer steps per epoch × 200 epochs = 20,000 total Adam steps.
      (Default chunk reduced from 50 → 25 in 2026-07-29 for faster optimizer
      updates; user can override via ``--tbptt-chunk``.)
    - Per-chunk inner loop is BATCHED via
      ``DifferentialStage._forward_heun_sequence`` (2026-07-29), eliminating
      the per-sample ``net()`` wrapper overhead. Was 50 calls/chunk, now 1.
    - ``--compile`` wraps the per-sample Heun step loop with ``torch.compile``
      for fused CUDA kernels (~2-3x additional speedup on GPU; first call
      adds ~10-30s compilation overhead).
    - ``--num-steps 4`` (was 6) cuts per-sample RHS compute by 33%.
    - Targets standardized (zero-mean, unit-variance), denormalized at eval.
    - Bipolar input drive: u ∈ [0, 0.5] → V_u ∈ [-3.0, +3.0] (use
      ``--unipolar`` for the old [0, +3.0] mapping).

Per-epoch validation:
    ``--val-every N`` (default 5) evaluates the running model on the test set
    every N epochs and shows ``nrmse``/``r2`` in the epoch progress bar. This
    gives real-time convergence feedback during long training runs.
    Set to 0 to disable. Validation uses the one-shot ``_evaluate_fabric_direct``
    path (~0.1s overhead per eval).

Diagnostics:
    ``--ridge-diagnostic`` runs ridge-on-frozen-states first: the untrained
    fabric's hidden states are read out by a closed-form ridge regression
    on y. If this hits NRMSE ~0.4–0.6, hardware dynamics are adequate and
    training issues are purely optimizer problems. If NRMSE ~1.0, the
    reservoir itself has no memory and the t_span fix is mandatory before
    training.

Complexity reporting:
    Each condition reports its parameter count alongside NRMSE and R^2 in
    the summary table and CSV output. ESN reports both trained (26) and
    total (676) parameters. LSTM variants report their hidden_dim.

Pre-registered decision rule:
    If both fabric conditions achieve NRMSE < 0.3 on NARMA-20 AND
    ``freeze_read=False`` is within 1 standard deviation of ``freeze_read=True``,
    the evolving core has no incremental value: COMMIT TO CONTROL-PATH.

    If ``freeze_read=False`` clearly beats ``freeze_read=True`` by > 0.05 NRMSE
    on NARMA-20, the evolving core is alive: SIGNAL-PATH STAYS OPEN.

Output:
    <output>/results_table.txt  — summary table (all conditions, all seeds)
    <output>/results_table.csv  — same data as CSV
    <output>/decision.txt       — the pre-registered decision rule's verdict
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from torch.amp import autocast, GradScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from config import OPTIM, PRESET_NARMA10, PRESET_NARMA20, make_narma_preset
from cell_library import make_cell_library
from topology import build_net_from_config, build_net_from_preset


# ---------------------------------------------------------------------------
# Data: NARMA-N generator
# ---------------------------------------------------------------------------

def narma(n_samples: int, order: int = 10, seed: int = 0,
          u_max: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a NARMA-N sequence.

    Args:
        n_samples: Number of output samples to return.
        order: NARMA order (10 or 20).
        seed: RNG seed for reproducibility.
        u_max: Upper bound of the uniform input range ``u(n) in [0, u_max]``.

    Returns:
        ``(u, y)`` each of shape ``(n_samples,)`` — input drive and target.
        The first ``order`` samples are washout and should be discarded.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_samples + order + 1, generator=g) * u_max
    y = torch.zeros(n_samples + order + 1)
    for n in range(order, n_samples + order):
        y[n + 1] = (
            0.3 * y[n]
            + 0.05 * y[n] * torch.sum(y[n - order + 1: n + 1])
            + 1.5 * u[n - order + 1] * u[n]
            + 0.1
        )
    return u[order: order + n_samples], y[order + 1: order + n_samples + 1]


def scale_input_to_rails(u: torch.Tensor, x_max: float = 3.0,
                         u_max: float = 0.5,
                         bipolar: bool = True) -> torch.Tensor:
    """Scale u(n) in [0, u_max] to a voltage drive for the boundary OTAs.

    Args:
        u: Input drive in [0, u_max].
        x_max: Voltage rail magnitude.
        u_max: Upper bound of u (default 0.5, NARMA standard).
        bipolar: If True (default), map to [-x_max, +x_max] so boundary
            OTAs operate around their most sensitive mid-rail point.
            If False, map to [0, x_max] (unipolar).

    Returns:
        Voltage drive tensor of same shape as ``u``.
    """
    if bipolar:
        return 2.0 * x_max * (u / u_max) - x_max
    return (u / u_max) * x_max


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RidgeRegressor:
    """Closed-form ridge regression on a tapped-delay feature matrix."""

    def __init__(self, n_taps: int = 30, l2: float = 1e-2) -> None:
        self.n_taps = n_taps
        self.l2 = float(l2)
        self.W: torch.Tensor | None = None
        self.b: torch.Tensor | None = None

    def _features(self, u: torch.Tensor) -> torch.Tensor:
        """Build the tapped-delay matrix.

        Args:
            u: shape ``(T,)``.
        Returns:
            ``(T - n_taps + 1, n_taps)`` feature matrix; row t contains
            ``[u[t], u[t-1], ..., u[t - n_taps + 1]]``.
        """
        T = u.shape[0]
        rows = []
        for lag in range(self.n_taps):
            rows.append(u[self.n_taps - 1 - lag: T - lag])
        return torch.stack(rows, dim=1)

    def fit(self, u: torch.Tensor, y: torch.Tensor) -> "RidgeRegressor":
        X = self._features(u)
        T = y.shape[0]
        y_aligned = y[self.n_taps - 1:]
        # Closed-form: W = (X^T X + l2 I)^-1 X^T y
        XtX = X.T @ X + self.l2 * torch.eye(self.n_taps)
        Xty = X.T @ y_aligned
        self.W = torch.linalg.solve(XtX, Xty)
        # Single bias learned by augmenting with constant column
        X_aug = torch.cat([X, torch.ones(T - self.n_taps + 1, 1)], dim=1)
        Wt = torch.linalg.solve(
            X_aug.T @ X_aug + self.l2 * torch.eye(self.n_taps + 1),
            X_aug.T @ y_aligned,
        )
        self.W = Wt[: self.n_taps]
        self.b = Wt[self.n_taps]
        return self

    def predict(self, u: torch.Tensor) -> torch.Tensor:
        assert self.W is not None and self.b is not None
        X = self._features(u)
        return X @ self.W + self.b


class ESN:
    """Echo State Network with tanh reservoir and ridge readout.

    Same node count as the fabric (25 nodes by default). Spectral radius
    ~0.9. No backprop through the reservoir — only the linear readout is
    trained via ridge regression.
    """

    def __init__(self, n_reservoir: int = 25, spectral_radius: float = 0.9,
                 input_scaling: float = 1.0, leak: float = 1.0,
                 ridge_l2: float = 1e-2, seed: int = 0) -> None:
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak = leak
        self.ridge_l2 = ridge_l2
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        # Random reservoir weights; rescale to hit the target spectral radius.
        W = torch.randn(n_reservoir, n_reservoir, generator=g)
        eig = torch.linalg.eigvals(W)
        sr = float(eig.abs().max().real)
        self.W = W * (spectral_radius / max(sr, 1e-9))
        self.W_in = torch.randn(n_reservoir, 1, generator=g) * input_scaling
        self.readout_W: torch.Tensor | None = None
        self.readout_b: torch.Tensor | None = None

    def _run(self, u: torch.Tensor) -> torch.Tensor:
        """Run the ESN over the input sequence.

        Returns the reservoir states for every timestep. With ``leak=1``
        the update is purely ``x = tanh(W x + W_in u)``.
        """
        T = u.shape[0]
        x = torch.zeros(self.n_reservoir)
        states = []
        for t in range(T):
            pre = self.W @ x + self.W_in.squeeze(-1) * u[t]
            x_new = torch.tanh(pre)
            x = (1 - self.leak) * x + self.leak * x_new
            states.append(x)
        return torch.stack(states, dim=0)

    def fit(self, u: torch.Tensor, y: torch.Tensor) -> "ESN":
        S = self._run(u)
        # Augment states with constant column for the readout bias
        S_aug = torch.cat([S, torch.ones(S.shape[0], 1)], dim=1)
        # Ridge: W_out = (S^T S + l2 I)^-1 S^T y
        StS = S_aug.T @ S_aug + self.ridge_l2 * torch.eye(S_aug.shape[1])
        W_out = torch.linalg.solve(StS, S_aug.T @ y)
        self.readout_W = W_out[: self.n_reservoir]
        self.readout_b = W_out[self.n_reservoir]
        return self

    def predict(self, u: torch.Tensor) -> torch.Tensor:
        assert self.readout_W is not None and self.readout_b is not None
        S = self._run(u)
        return S @ self.readout_W + self.readout_b


class MLPRegressor(nn.Module):
    """Simple MLP on a 30-tap window. Used as a digital feedforward reference."""

    def __init__(self, n_taps: int = 30, hidden_dim: int = 25,
                 out_dim: int = 1) -> None:
        super().__init__()
        self.n_taps = n_taps
        self.fc1 = nn.Linear(n_taps, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, u_window: torch.Tensor) -> torch.Tensor:
        # u_window: (B, n_taps)
        h = torch.tanh(self.fc1(u_window))
        return self.fc2(h)


class LSTMRegressor(nn.Module):
    """LSTM baseline taking streaming scalar u(t) with state carryover."""

    def __init__(self, hidden_dim: int = 25) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        u_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if u_seq.dim() == 2:
            u_seq = u_seq.unsqueeze(-1)
        out, final_state = self.lstm(u_seq, state)
        preds = self.fc(out).squeeze(-1)
        return preds, final_state


def make_mlp_features(u: torch.Tensor, n_taps: int = 30) -> tuple[torch.Tensor, int]:
    """Build the MLP feature matrix; returns (X, washout)."""
    T = u.shape[0]
    rows = []
    for lag in range(n_taps):
        rows.append(u[n_taps - 1 - lag: T - lag])
    return torch.stack(rows, dim=1), n_taps - 1


def train_and_eval_lstm(
    order: int,
    seed: int,
    hidden_dim: int = 25,
    bptt_window: int = 50,
    epochs: int = 200,
    washout: int = 200,
    lr: float = 1e-3,
    u_train_override: torch.Tensor | None = None,
    y_train_override: torch.Tensor | None = None,
    u_val_override: torch.Tensor | None = None,
    y_val_override: torch.Tensor | None = None,
    device: str = "cpu",
) -> dict[str, float | int | None]:
    """Train an LSTM on NARMA using truncated BPTT.

    When ``u_train_override`` is provided, train on that data instead of
    generating fresh NARMA data. When ``u_val_override`` is provided,
    compute validation NRMSE after training.

    Args:
        order: NARMA order (10 or 20).
        seed: RNG seed.
        hidden_dim: LSTM hidden units.
        bptt_window: Truncated BPTT window in samples.
        epochs: Training epochs.
        washout: Number of initial test samples to discard.
        lr: Learning rate.
        u_train_override: Optional training input (overrides narma gen).
        y_train_override: Optional training target (overrides narma gen).
        u_val_override: Optional validation input.
        y_val_override: Optional validation target.

    Returns:
        Dict with nrmse, r2, val_nrmse, n_params, hidden_dim.
    """
    if u_train_override is not None:
        u_train = u_train_override
        y_train = y_train_override
        u_test, y_test = narma(1000, order=order, seed=seed + 10000)
    else:
        u_train, y_train = narma(3000, order=order, seed=seed)
        u_test, y_test = narma(1000, order=order, seed=seed + 10000)

    u_train = u_train.to(device)
    y_train = y_train.to(device)
    u_test = u_test.to(device)
    y_test = y_test.to(device)

    torch.manual_seed(seed)
    model = LSTMRegressor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_samples = u_train.shape[0]
    n_windows = n_samples // bptt_window

    model.train()
    for epoch in range(epochs):
        window_starts = torch.randperm(n_windows, device=device) * bptt_window
        epoch_loss = 0.0

        for w_start in window_starts:
            u_win = u_train[w_start : w_start + bptt_window].unsqueeze(0)
            y_win = y_train[w_start : w_start + bptt_window].unsqueeze(0)

            optimizer.zero_grad()
            preds, _ = model(u_win)
            loss = F.mse_loss(preds, y_win)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

    # Validation NRMSE (if val data provided)
    val_nrmse: float | None = None
    if u_val_override is not None and y_val_override is not None:
        model.eval()
        with torch.no_grad():
            u_val_seq = u_val_override.to(device).unsqueeze(0)
            y_val_pred, _ = model(u_val_seq)
            y_val_pred = y_val_pred.squeeze(0)
        val_nrmse = nrmse(y_val_pred, y_val_override.to(device))
        model.train()

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        u_te_seq = u_test.unsqueeze(0)
        y_pred, _ = model(u_te_seq)
        y_pred = y_pred.squeeze(0)

    eval_nrmse = nrmse(y_pred[washout:], y_test[washout:])
    eval_r2 = r2(y_pred[washout:], y_test[washout:])

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "nrmse": eval_nrmse,
        "r2": eval_r2,
        "val_nrmse": val_nrmse,
        "n_params": n_params,
        "hidden_dim": hidden_dim,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nrmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Normalized RMSE: RMSE / std(y_true). Standard reservoir metric."""
    rmse = float(torch.sqrt(F.mse_loss(y_pred, y_true)).item())
    std = float(y_true.std().clamp(min=1e-9).item())
    return rmse / std


def r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Coefficient of determination."""
    ss_res = float(((y_pred - y_true) ** 2).sum().item())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum().item())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def memory_capacity(states: torch.Tensor, targets: torch.Tensor,
                    max_delay: int = 20,
                    ridge_l2: float = 1e-2) -> tuple[list[float], float]:
    """Compute memory capacity by training linear readouts on each delay.

    For each k in [1, max_delay], fit a ridge linear regression from the
    reservoir states ``states`` (shape (T, N)) to ``targets`` shifted by k:
    ``MC_k = R^2_k`` of the k-step-back reconstruction.

    The total memory capacity is ``MC = sum_k R^2_k`` (sum over k = 1..20).
    This is the standard reservoir MC measure from Jaeger (2001).
    """
    T = states.shape[0]
    r2_list = []
    for k in range(1, max_delay + 1):
        if T - k <= 0:
            r2_list.append(0.0)
            continue
        X = states[: T - k]
        y = targets[k:]
        X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(X_aug.shape[1], device=X.device)
        W = torch.linalg.solve(XtX, X_aug.T @ y)
        y_pred = X_aug @ W
        r2_list.append(r2(y_pred, y))
    mc_total = sum(max(r, 0.0) for r in r2_list)
    return r2_list, mc_total


def ridge_readout_diagnostic(
    net: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    washout: int = 200,
    ridge_l2: float = 1e-2,
    device: str = "cpu",
) -> dict[str, float]:
    """Decisive 10-minute diagnostic: ridge readout on untrained fabric states.

    Runs the (randomly-initialized, untrained) fabric over the training
    stream, collects the per-sample hidden node states, and fits a ridge
    linear regression ``y = W @ state + b`` offline. The NRMSE/R² of the
    ridge readout cleanly separates two failure modes:
        - NRMSE ~ 0.4–0.6 (ESN-level): hardware dynamics are adequate;
          any training difficulty is an optimizer issue.
        - NRMSE ~ 1.0+: the reservoir itself has no usable memory, and
          timescale tweaks (issue #2: t_span) must precede training.

    Args:
        net: An *untrained* KirchhoffNetWithIO module.
        u_train: Training input drive ``(T,)`` (after scaling).
        y_train: Training targets ``(T,)`` (raw, not standardized).
        washout: Number of initial samples to discard (reservoir convention).
        ridge_l2: Ridge regularization.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with ``ridge_nrmse``, ``ridge_r2``.
    """
    net.eval()
    net.to(device)
    u_train = u_train.to(device)
    y_train = y_train.to(device)

    T = u_train.shape[0]
    # Use the direct stage-call path to capture states
    stage = net.core.stages[0]
    t_span = net.core.stage_times[0]
    num_steps = net.core.stage_steps[0]
    stage_width = net.hid_count + net.proj_count + net.output_ode_count

    x0 = u_train.new_zeros(1, stage_width)
    all_states = stage._forward_heun_sequence(
        x0=x0, t_span=t_span, num_steps=num_steps, u_seq=u_train,
    )
    # all_states: (T, 1, stage_width); take hidden nodes only
    states = all_states[:, 0, :net.hid_count]  # (T, hid_count)

    # Ridge fit on (states -> y), skipping washout
    X = states[washout:]
    y = y_train[washout:]
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(X_aug.shape[1], device=X.device)
    W = torch.linalg.solve(XtX, X_aug.T @ y)
    y_pred = X_aug @ W

    ridge_nrmse_val = nrmse(y_pred, y)
    ridge_r2_val = r2(y_pred, y)
    return {"ridge_nrmse": ridge_nrmse_val, "ridge_r2": ridge_r2_val}


# ---------------------------------------------------------------------------
# Fabric training
# ---------------------------------------------------------------------------

def _progress_iter(iterable, desc, total=None, disable=False):
    """Wrap an iterable with tqdm if available, else pass through."""
    if _HAS_TQDM and not disable:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def _standardize_targets(
    y_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-mean/unit-variance normalization.

    Returns:
        ``(y_norm, y_mean, y_std)`` — normalized targets and stats for
        denormalization at evaluation.
    """
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-9)
    return (y_train - y_mean) / y_std, y_mean, y_std


def _denormalize(y_norm: torch.Tensor, y_mean: torch.Tensor,
                 y_std: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_standardize_targets`."""
    return y_norm * y_std + y_mean


def train_fabric(
    net: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int = 4,
    epochs: int = 200,
    tbptt_chunk: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
    use_amp: bool = True,
    standardize: bool = True,
    grad_clip: float = 1.0,
    val_every: int = 0,
    val_u_test: torch.Tensor | None = None,
    val_y_test: torch.Tensor | None = None,
    val_washout: int = 200,
) -> dict[str, Any]:
    """Train the fabric with sequential truncated BPTT.

    The training protocol is:
        - ``B = batch_size`` parallel independent streams, each a
          contiguous segment of length ``n_samples``.
        - State is carried across chunks (not reset per window).
        - Within each chunk of ``K = tbptt_chunk`` consecutive samples,
          full BPTT: no ``detach()`` between samples. Gradients flow
          through ``K * num_steps`` Heun steps.
        - ``state = state.detach()`` is the ONLY cut, applied every K
          samples (chunk boundary).
        - One optimizer step per chunk.

    ``u_train`` may be 1D ``(n_samples,)`` (a single stream, B=1) or 2D
    ``(B, n_samples)`` (multiple parallel streams). When 1D, it's
    reshaped to 2D with B=1.

    Targets are standardized to zero-mean/unit-variance before training
    (when ``standardize=True``); statistics are returned so the caller
    can denormalize predictions at evaluation.

    Args:
        net: KirchhoffNetWithIO module.
        u_train: Input drive ``(n_samples,)`` or ``(B, n_samples)``.
        y_train: Targets ``(n_samples,)`` or ``(B, n_samples)``.
        batch_size: Number of parallel independent streams.
        epochs: Training epochs.
        tbptt_chunk: TBPTT chunk size K (samples per optimizer step).
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        device: 'cpu' or 'cuda'.
        verbose: Print epoch progress.
        use_amp: Mixed-precision training (no-op on CPU).
        standardize: If True, normalize y to zero-mean/unit-variance.
        grad_clip: Max gradient norm (AdamW clip).
        val_every: If >0, evaluate test-set NRMSE/R^2 every N epochs
            (using ``val_u_test``/``val_y_test``). Default 0 (disabled).
        val_u_test: Validation input sequence (shape ``(T,)``).
        val_y_test: Validation target sequence (shape ``(T,)``).
        val_washout: Initial samples to discard for validation (default 200).

    Returns:
        Dict with ``train_loss_history``, ``final_loss``,
        ``y_mean``, ``y_std``, and (when ``val_every>0``)
        ``val_nrmse_history``, ``val_r2_history``, ``val_epochs``.
    """
    net.to(device)
    u_train = u_train.to(device)
    y_train = y_train.to(device)

    # Standardize targets (always — caller can pass raw; we normalize internally)
    if standardize:
        y_train_used, y_mean, y_std = _standardize_targets(y_train)
    else:
        y_train_used = y_train
        y_mean = torch.zeros(1, device=device)
        y_std = torch.ones(1, device=device)

    # Reshape to (B, n_samples) if needed
    if u_train.dim() == 1:
        u_train = u_train.unsqueeze(0)
        y_train_used = y_train_used.unsqueeze(0)
        y_mean = y_mean.unsqueeze(0)
        y_std = y_std.unsqueeze(0)
        single_stream = True
    else:
        single_stream = False

    B = u_train.shape[0]
    n_samples = u_train.shape[1]

    if B != batch_size:
        if single_stream and batch_size > 1:
            # Tile the single stream into B copies (each starts from different offset)
            offsets = torch.linspace(0, n_samples // 2, batch_size, device=device).long()
            u_streams = []
            y_streams = []
            for b in range(batch_size):
                start = int(offsets[b].item())
                end = start + n_samples
                if end <= u_train.shape[1]:
                    u_streams.append(u_train[0, start:end])
                    y_streams.append(y_train_used[0, start:end])
                else:
                    # Wrap-around for streams that don't fit
                    wrap = end - u_train.shape[1]
                    u_streams.append(
                        torch.cat([u_train[0, start:], u_train[0, :wrap]])
                    )
                    y_streams.append(
                        torch.cat([y_train_used[0, start:], y_train_used[0, :wrap]])
                    )
            u_train = torch.stack(u_streams, dim=0)
            y_train_used = torch.stack(y_streams, dim=0)
            B = batch_size

    if y_train.shape[0] < B and not single_stream:
        # user passed 2D with fewer streams than batch_size; pad/truncate
        B = y_train.shape[0]

    # Use full division so we cover all n_samples; chunks past the stream
    # end are dropped by the bounds check in the inner loop.
    n_chunks = n_samples // tbptt_chunk
    if n_chunks < 1:
        raise ValueError(
            f"u_train has {n_samples} samples, need at least tbptt_chunk="
            f"{tbptt_chunk} samples per stream."
        )

    # Optimizer
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP
    amp_enabled = use_amp and device == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    history: list[float] = []
    raw_leak_grad_history: list[float] = []
    hidden_rms_history: list[float] = []
    val_nrmse_history: list[float] = []
    val_r2_history: list[float] = []
    val_epochs: list[int] = []
    t_start = time.time()

    log_interval = max(1, epochs // 20)
    epoch_times: list[float] = []

    state_width = net.hid_count + net.proj_count + net.output_ode_count

    # Single-stage NARMA: hoist stage references and integrator config
    # out of the chunk loop. The batched inner path uses
    # stage._forward_heun_sequence to eliminate per-sample wrapper
    # overhead (mirrors _evaluate_fabric_direct, but in train mode).
    if not hasattr(net, "core") or not hasattr(net.core, "stages"):
        raise ValueError(
            "train_fabric batched inner loop requires a KirchhoffNetWithIO "
            "wrapper exposing net.core.stages[0]. This net does not have it."
        )
    if len(net.core.stages) != 1:
        raise ValueError(
            f"train_fabric batched inner loop currently supports single-stage "
            f"topologies only, got {len(net.core.stages)} stages."
        )
    stage = net.core.stages[0]
    t_span = float(net.core.stage_times[0])
    num_steps = int(net.core.stage_steps[0])

    for epoch in range(epochs):
        net.train()
        # Carry state across chunks; detach only at chunk boundaries.
        state = torch.zeros(B, state_width, device=device)

        # Permute chunk-start offsets to decorrelate the chunk sequence across streams.
        # Each stream processes a chunk of K samples in lockstep.
        chunk_starts = torch.arange(n_chunks, device=device) * tbptt_chunk
        # Shuffle chunk order once per epoch for variety
        chunk_order = torch.randperm(n_chunks, device=device)

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_raw_leak_grad = 0.0
        epoch_hidden_rms = 0.0

        optim.zero_grad()

        epoch_iter = _progress_iter(
            enumerate(chunk_order), desc=f"chunks",
            total=n_chunks, disable=not verbose,
        )
        for _, ci in epoch_iter:
            cs = int(chunk_starts[int(ci)].item())
            ce = cs + tbptt_chunk
            if ce > n_samples:
                continue

            u_chunk = u_train[:, cs:ce]   # (B, K)
            y_chunk = y_train_used[:, cs:ce]  # (B, K)

            # Batched inner loop: process all K samples in one stage call,
            # eliminating per-sample wrapper overhead. Mirrors the fast
            # _evaluate_fabric_direct path but in train mode (gradients
            # thread through the entire chunk; only the chunk boundary
            # is detached).
            with autocast("cuda", enabled=amp_enabled):
                # _forward_heun_sequence expects (B, K, 1) for multi-stream.
                u_seq_batched = u_chunk.unsqueeze(-1)  # (B, K, 1)
                all_states = stage._forward_heun_sequence(
                    x0=state,
                    t_span=t_span,
                    num_steps=num_steps,
                    u_seq=u_seq_batched,
                )  # (K, B, N)
                # Read slice (output_ode_count tail for temporal-readout).
                x_read = all_states[:, :, net.read_slice]  # (K, B, R)
                y_pred = net.output_mapper(x_read)  # (K, B, out_dim)
                if y_pred.dim() == 3:
                    y_pred = y_pred.squeeze(-1)  # (K, B)
                # y_chunk is (B, K) -> transpose to (K, B) for direct comparison.
                loss = F.mse_loss(y_pred, y_chunk.t())

                # Hidden voltage RMS over the chunk's last sample (matches
                # the original "RMS at chunk boundary" semantic).
                hidden_acc_sq = 0.0
                hidden_count = 0
                if net.hid_count > 0:
                    final_state = all_states[-1]
                    h = final_state[:, :net.hid_count]
                    hidden_acc_sq = float(h.detach().pow(2).sum().item())
                    hidden_count = h.numel()

                # The new state for next chunk: the last sample's state.
                state = all_states[-1]

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)

            # Capture per-param-group gradient norms (every epoch).
            # raw_leak has shape (num_nodes,) = (hid_count + proj_count +
            #output_ode_count), not hid_count alone — match by name
            # rather than dim to be robust across stage topologies.
            raw_leak_grad = 0.0
            for name, p in net.named_parameters():
                if name.endswith("raw_leak") and p.grad is not None:
                    raw_leak_grad = float(p.grad.detach().norm().item())
                    break
            epoch_raw_leak_grad += raw_leak_grad
            chunk_rms = (hidden_acc_sq / max(hidden_count, 1)) ** 0.5
            epoch_hidden_rms += chunk_rms

            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            # The ONLY detach: chunk boundary
            state = state.detach()

            epoch_loss += float(loss.item())
            epoch_steps += 1

            if _HAS_TQDM and verbose:
                epoch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / max(epoch_steps, 1)
        history.append(avg)
        leak_grad_avg = epoch_raw_leak_grad / max(epoch_steps, 1)
        h_rms_avg = epoch_hidden_rms / max(epoch_steps, 1)
        raw_leak_grad_history.append(leak_grad_avg)
        hidden_rms_history.append(h_rms_avg)

        # Periodic validation: evaluate on a held-out test stream.
        # Uses _evaluate_fabric_direct (single-shot Heun over the whole
        # sequence) so the overhead is ~0.1s — negligible vs the chunk loop.
        val_nrmse = float("nan")
        val_r2v = float("nan")
        if (
            val_every > 0
            and val_u_test is not None
            and val_y_test is not None
            and ((epoch + 1) % val_every == 0 or epoch == epochs - 1)
        ):
            y_pred_v, y_te_v = _evaluate_fabric_direct(
                net, val_u_test, val_y_test, washout=val_washout,
                device=device, y_mean=y_mean, y_std=y_std,
            )
            val_nrmse = nrmse(y_pred_v, y_te_v)
            val_r2v = r2(y_pred_v, y_te_v)
            val_nrmse_history.append(val_nrmse)
            val_r2_history.append(val_r2v)
            val_epochs.append(epoch)

        t_epoch = time.time() - t_start
        epoch_times.append(t_epoch)
        if verbose:
            if len(epoch_times) >= 2:
                dt = epoch_times[-1] - epoch_times[-2]
                eta = dt * (epochs - 1 - epoch)
                eta_str = f"ETA {eta:.0f}s"
            else:
                eta_str = ""
            val_str = ""
            if not math.isnan(val_nrmse):
                val_str = (
                    f"  NRMSE={val_nrmse:.4f}  "
                    f"R^2={val_r2v:.4f}"
                )
            print(
                f"  fabric epoch {epoch:3d}/{epochs}  loss={avg:.6f}  "
                f"h_rms={h_rms_avg:.4f}  "
                f"leak_grad={leak_grad_avg:.2e}  "
                f"{t_epoch:.1f}s  {eta_str}{val_str}"
            )

    total_elapsed = time.time() - t_start
    print(f"  fabric training complete: {epochs} epochs in {total_elapsed:.1f}s")
    if standardize:
        print(
            f"  target stats: mean={float(y_train.mean()):.4f} "
            f"std={float(y_train.std()):.4f}"
        )
    return {
        "train_loss_history": history,
        "final_loss": history[-1] if history else float("nan"),
        "raw_leak_grad_history": raw_leak_grad_history,
        "hidden_rms_history": hidden_rms_history,
        "val_nrmse_history": val_nrmse_history,
        "val_r2_history": val_r2_history,
        "val_epochs": val_epochs,
        "y_mean": y_mean,
        "y_std": y_std,
    }


@torch.inference_mode()
def evaluate_fabric(
    net: nn.Module,
    u_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    washout: int = 200,
    device: str = "cpu",
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fabric over the test sequence with state carryover.

    Args:
        y_mean: If provided, denormalize predictions using this and y_std.
        y_std: Standard deviation of training targets for denormalization.

    Returns:
        ``(y_pred, y_test_aligned)``: predictions and ground truth after
        ``washout`` initial samples have been discarded (reservoir
        convention). Predictions are denormalized if y_mean/y_std provided.
    """
    net.eval()
    net.to(device)
    u_test = u_test.to(device)
    y_test = y_test.to(device)
    T = u_test.shape[0]
    preds = torch.empty(T, device=device)
    state = None
    for t in range(T):
        u_b = u_test[t].view(1, 1)
        y_pred, _, final_state = net(
            u_b, initial_state=state, return_final_state=True,
        )
        preds[t] = y_pred.view(-1)
        state = final_state
    if y_mean is not None and y_std is not None:
        preds = _denormalize(preds, y_mean.to(device), y_std.to(device))
    return preds[washout:], y_test[washout:]


@torch.inference_mode()
def _evaluate_fabric_direct(
    net: nn.Module,
    u_seq: torch.Tensor,
    y_test: torch.Tensor,
    *,
    washout: int = 200,
    device: str = "cpu",
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct evaluation bypassing KirchhoffNetWithIO / KirchhoffNet wrappers.

    Calls ``DifferentialStage._forward_heun_sequence`` to process the entire
    test sequence in a single call, eliminating per-sample Python overhead.
    Only valid for the NARMA single-stage topology.

    If ``y_mean``/``y_std`` are provided, predictions are denormalized so
    they live on the same scale as ``y_test`` (needed for NRMSE/R²).
    """
    net.eval()
    net.to(device)
    u_seq = u_seq.to(device)
    y_test = y_test.to(device)
    T = u_seq.shape[0]

    stage = net.core.stages[0]
    output_mapper = net.output_mapper
    read_slice = net.read_slice
    t_span = net.core.stage_times[0]
    num_steps = net.core.stage_steps[0]
    stage_width = net.hid_count + net.proj_count + net.output_ode_count

    x0 = u_seq.new_zeros(1, stage_width)
    all_states = stage._forward_heun_sequence(
        x0=x0,
        t_span=t_span,
        num_steps=num_steps,
        u_seq=u_seq,
    )
    x_read = all_states[:, :, read_slice]
    y_preds = output_mapper(x_read)
    preds = y_preds.view(T)
    if y_mean is not None and y_std is not None:
        preds = _denormalize(preds, y_mean.to(device), y_std.to(device))
    return preds[washout:], y_test[washout:]


@torch.inference_mode()
def collect_fabric_states(
    net: nn.Module,
    u: torch.Tensor,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Collect the per-sample fabric state (last layer only) for MC eval.

    Uses the direct-stage-call path to bypass wrapper overhead.

    Returns:
        Tensor of shape ``(T, stage_width)`` — the final ODE state at
        every timestep.
    """
    net.eval()
    net.to(device)
    u = u.to(device)
    T = u.shape[0]

    stage = net.core.stages[0]
    t_span = net.core.stage_times[0]
    num_steps = net.core.stage_steps[0]
    stage_width = net.hid_count + net.proj_count + net.output_ode_count

    x0 = u.new_zeros(1, stage_width)
    all_states = stage._forward_heun_sequence(
        x0=x0,
        t_span=t_span,
        num_steps=num_steps,
        u_seq=u,
    )
    return all_states[:, 0, :]  # (T, stage_width)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_baselines(order: int, seed: int, *, device: str = "cpu") -> dict[str, dict[str, Any]]:
    """Train and evaluate the non-fabric baselines.

    Returns dict mapping condition name -> {nrmse, r2, n_params, ...}.
    Conditions: ridge, esn, mlp, mlp_large, lstm, lstm_large.
    """
    u_train, y_train = narma(3000, order=order, seed=seed)
    u_test, y_test = narma(1000, order=order, seed=seed + 10000)
    n_taps = 30
    washout = 200
    # After tapping, predictions cover u[n_taps - 1 : T] (length T - n_taps + 1).
    # We align the corresponding y_true slice, then apply the standard
    # washout discard.
    align_offset = n_taps - 1

    # 1. Ridge regression on 30 taps
    ridge = RidgeRegressor(n_taps=n_taps, l2=1e-2).fit(u_train, y_train)
    y_pred = ridge.predict(u_test)
    y_test_aligned = y_test[align_offset:]
    # Apply washout: discard the first washout predictions
    a = nrmse(y_pred[washout:], y_test_aligned[washout:])
    b = r2(y_pred[washout:], y_test_aligned[washout:])
    ridge_res = {"nrmse": a, "r2": b, "n_params": n_taps + 1, "total_params": n_taps + 1}

    # 2. ESN, 25 nodes — grid search input_scaling and leak rate
    # Use last 500 train samples as validation set.
    val_size = 500
    u_esn_train = u_train[:-val_size]
    y_esn_train = y_train[:-val_size]
    u_esn_val = u_train[-val_size:]
    y_esn_val = y_train[-val_size:]

    best_esn_nrmse = float("inf")
    best_esn: ESN | None = None
    for input_scaling in [0.2, 0.5, 1.0]:
        for leak in [0.3, 0.5, 1.0]:
            esn = ESN(n_reservoir=25, spectral_radius=0.9,
                      input_scaling=input_scaling, leak=leak,
                      ridge_l2=1e-2, seed=seed)
            esn.fit(u_esn_train, y_esn_train)
            y_val_pred = esn.predict(u_esn_val)
            val_nrmse = nrmse(y_val_pred, y_esn_val)
            if val_nrmse < best_esn_nrmse:
                best_esn_nrmse = val_nrmse
                best_esn = esn

    # Refit on the full train set with the best hyperparameters
    assert best_esn is not None
    esn = ESN(n_reservoir=25, spectral_radius=0.9,
              input_scaling=best_esn.input_scaling, leak=best_esn.leak,
              ridge_l2=1e-2, seed=seed)
    esn.fit(u_train, y_train)
    y_pred = esn.predict(u_test)
    a = nrmse(y_pred[washout:], y_test[washout:])
    b = r2(y_pred[washout:], y_test[washout:])
    # ESN: 26 trained (readout) / 676 total (readout + reservoir W + W_in)
    esn_trained = esn.n_reservoir + 1
    esn_total = esn.n_reservoir * esn.n_reservoir + esn.n_reservoir + esn.n_reservoir + 1
    esn_res = {"nrmse": a, "r2": b, "n_params": esn_trained, "total_params": esn_total}

    # 3. MLP on 30-tap window (trained via gradient descent)
    X_tr, _ = make_mlp_features(u_train, n_taps=n_taps)
    y_tr_aligned = y_train[align_offset:]
    X_te, _ = make_mlp_features(u_test, n_taps=n_taps)
    y_te_aligned = y_test[align_offset:]
    X_tr = X_tr.to(device)
    y_tr_aligned = y_tr_aligned.to(device)
    X_te = X_te.to(device)
    y_te_aligned = y_te_aligned.to(device)
    mlp = MLPRegressor(n_taps=n_taps, hidden_dim=25).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(200):
        opt.zero_grad()
        y_pred = mlp(X_tr).squeeze(-1)
        loss = F.mse_loss(y_pred, y_tr_aligned)
        loss.backward()
        opt.step()
    with torch.no_grad():
        y_pred = mlp(X_te).squeeze(-1)
    a = nrmse(y_pred[washout:], y_te_aligned[washout:])
    b = r2(y_pred[washout:], y_te_aligned[washout:])
    mlp_n_params = sum(p.numel() for p in mlp.parameters())
    mlp_res = {"nrmse": a, "r2": b, "n_params": mlp_n_params, "total_params": mlp_n_params}

    # 3b. MLP_large on 30-tap window (hidden_dim=237, ~7,585 params ≈ KNet)
    mlp_large = MLPRegressor(n_taps=n_taps, hidden_dim=237).to(device)
    opt = torch.optim.AdamW(mlp_large.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(200):
        opt.zero_grad()
        y_pred = mlp_large(X_tr).squeeze(-1)
        loss = F.mse_loss(y_pred, y_tr_aligned)
        loss.backward()
        opt.step()
    with torch.no_grad():
        y_pred = mlp_large(X_te).squeeze(-1)
    a = nrmse(y_pred[washout:], y_te_aligned[washout:])
    b = r2(y_pred[washout:], y_te_aligned[washout:])
    mlp_large_n_params = sum(p.numel() for p in mlp_large.parameters())
    mlp_large_res = {"nrmse": a, "r2": b, "n_params": mlp_large_n_params, "total_params": mlp_large_n_params}

    # 4. LSTM with grid search (hidden_dim=[16,25,32], lr=[1e-3,5e-4])
    # Use last 500 train samples as validation set (same pattern as ESN).
    val_size_lstm = 500
    u_tr_lstm = u_train[:-val_size_lstm]
    y_tr_lstm = y_train[:-val_size_lstm]
    u_val_lstm = u_train[-val_size_lstm:]
    y_val_lstm = y_train[-val_size_lstm:]

    best_lstm_val_nrmse = float("inf")
    best_lstm_hd = 25
    best_lstm_lr = 1e-3
    for hidden_dim in [16, 25, 32]:
        for lr in [1e-3, 5e-4]:
            res = train_and_eval_lstm(
                order=order, seed=seed, hidden_dim=hidden_dim,
                bptt_window=50, epochs=200, washout=washout, lr=lr,
                u_train_override=u_tr_lstm, y_train_override=y_tr_lstm,
                u_val_override=u_val_lstm, y_val_override=y_val_lstm,
                device=device,
            )
            if res["val_nrmse"] is not None and res["val_nrmse"] < best_lstm_val_nrmse:
                best_lstm_val_nrmse = res["val_nrmse"]
                best_lstm_hd = hidden_dim
                best_lstm_lr = lr

    # Refit on the full train set with the best hyperparameters
    lstm_res_dict = train_and_eval_lstm(
        order=order, seed=seed, hidden_dim=best_lstm_hd,
        bptt_window=50, epochs=200, washout=washout, lr=best_lstm_lr,
        device=device,
    )
    lstm_res = {
        "nrmse": lstm_res_dict["nrmse"],
        "r2": lstm_res_dict["r2"],
        "n_params": lstm_res_dict["n_params"],
        "total_params": lstm_res_dict["n_params"],
        "hidden_dim": best_lstm_hd,
    }

    # 4b. LSTM_large (hidden_dim=42, lr grid search)
    # h=42 gives 4*42^2 + 13*42 + 1 = 7,603 params ≈ KNet's ~7,600
    best_lstm_large_val_nrmse = float("inf")
    best_lstm_large_lr = 1e-3
    for lr in [1e-3, 5e-4]:
        res = train_and_eval_lstm(
            order=order, seed=seed, hidden_dim=42,
            bptt_window=50, epochs=200, washout=washout, lr=lr,
            u_train_override=u_tr_lstm, y_train_override=y_tr_lstm,
            u_val_override=u_val_lstm, y_val_override=y_val_lstm,
            device=device,
        )
        if res["val_nrmse"] is not None and res["val_nrmse"] < best_lstm_large_val_nrmse:
            best_lstm_large_val_nrmse = res["val_nrmse"]
            best_lstm_large_lr = lr

    # Refit on the full train set with the best hyperparameters
    lstm_large_res_dict = train_and_eval_lstm(
        order=order, seed=seed, hidden_dim=42,
        bptt_window=50, epochs=200, washout=washout, lr=best_lstm_large_lr,
        device=device,
    )
    lstm_large_res = {
        "nrmse": lstm_large_res_dict["nrmse"],
        "r2": lstm_large_res_dict["r2"],
        "n_params": lstm_large_res_dict["n_params"],
        "total_params": lstm_large_res_dict["n_params"],
        "hidden_dim": 42,
    }

    return {
        "ridge": ridge_res,
        "esn": esn_res,
        "mlp": mlp_res,
        "mlp_large": mlp_large_res,
        "lstm": lstm_res,
        "lstm_large": lstm_large_res,
    }


def run_fabric_condition(
    order: int,
    seed: int,
    freeze_read: bool,
    *,
    epochs: int = 200,
    tbptt_chunk: int = 50,
    n_streams: int = 4,
    train_samples_per_stream: int = 2500,
    device: str = "cpu",
    use_amp: bool = True,
    bipolar: bool = True,
    do_ridge_diagnostic: bool = False,
    t_span: float | None = None,
    num_steps: int | None = None,
    compile_sequence: bool = False,
    val_every: int = 5,
    val_u_test: torch.Tensor | None = None,
    val_y_test: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train one fabric condition and return its results.

    Args:
        order: NARMA order (10 or 20).
        seed: RNG seed.
        freeze_read: If True, use the frozen-core configuration.
        epochs: Training epochs.
        tbptt_chunk: TBPTT chunk size K (samples per optimizer step).
        n_streams: Number of parallel independent NARMA streams (batch B).
        train_samples_per_stream: Length of each training stream.
        use_amp: Enable mixed-precision training (default True).
        bipolar: If True, scale input to [-x_max, +x_max] (recommended).
        do_ridge_diagnostic: If True, run ridge-on-frozen-states diagnostic
            on an untrained net first to separate hardware vs optimizer issues.
        t_span: ODE integration window per sample. If None, falls back to
            the preset default (1.0; was 7.0 historically).
        num_steps: Heun steps per sample. If None, falls back to preset
            default (6).
        compile_sequence: If True, wrap the per-sample Heun step loop with
            ``torch.compile`` for fused-kernel speedup (~2-3x on GPU).
        val_every: Evaluate test-set NRMSE/R^2 every N epochs (default 5).
            Set to 0 to disable.
        val_u_test: Optional pre-generated validation input (re-used across
            epochs). If None, generated from the standard test seed.
        val_y_test: Optional pre-generated validation target.

    Returns dict with:
        - "nrmse": test NRMSE on the washed test set
        - "r2": test R^2
        - "mc_total": memory capacity (sum of R^2 over k=1..20)
        - "mc_per_delay": list of R^2 per delay
        - "train_loss_history": list of training losses
        - "raw_leak_grad_history": per-epoch gradient norm of raw_leak
        - "hidden_rms_history": per-epoch RMS of hidden node voltages
        - "val_nrmse_history": per-validation NRMSE (only epochs where eval ran)
        - "val_r2_history": per-validation R^2 (only epochs where eval ran)
        - "val_epochs": epoch indices at which val was recorded
        - "n_params": parameter count
        - "ridge_nrmse": (optional) ridge-on-frozen-states NRMSE
        - "ridge_r2": (optional) ridge-on-frozen-states R^2
        - "t_span": t_span actually used (after fallback)
        - "num_steps": num_steps actually used (after fallback)
    """
    # Build the preset inline so CLI overrides (t_span, num_steps) take
    # effect regardless of module-level PRESET_NARMA10/20 imports. This
    # avoids the "patch _cfg.PRESETS doesn't reach run_fabric_condition"
    # trap of binding a separate dict reference at import time.
    base = PRESET_NARMA20 if order == 20 else PRESET_NARMA10
    if t_span is None:
        t_span = base["stages"][0]["t_span"]
    if num_steps is None:
        num_steps = base["stages"][0]["num_steps"]
    preset = make_narma_preset(
        order=order,
        hidden_dim=base["stages"][0]["num_hidden"],
        t_span=t_span,
        num_steps_per_sample=num_steps,
    )

    cell_lib = make_cell_library("tanh")
    torch.manual_seed(seed)
    net = build_net_from_config(
        cfg=preset,
        cell_lib=cell_lib,
        boundary_fan_out=preset["boundary_fan_out"],
        enable_temporal_readout=True,
        freeze_read=freeze_read,
    )

    # Optionally wrap the per-sample Heun loop with torch.compile for
    # ~2-3x speedup on GPU (fused CUDA kernels via Inductor). The first
    # call adds ~10-30s compilation overhead, then subsequent calls run
    # at compiled speed.
    if compile_sequence:
        for stage in net.core.stages:
            stage.enable_sequence_compile()

    # ---- (Optional) Ridge-on-frozen-states diagnostic BEFORE training ----
    ridge_result: dict[str, float] = {}
    if do_ridge_diagnostic:
        # Generate the train stream identically to training (so the
        # diagnostic reflects what the actual training data looks like).
        u_diag, y_diag = narma(
            n_streams * train_samples_per_stream, order=order, seed=seed,
        )
        u_diag = scale_input_to_rails(u_diag, bipolar=bipolar)
        print("  [ridge diagnostic] running untrained fabric on train stream...")
        ridge_result = ridge_readout_diagnostic(
            net, u_diag, y_diag, washout=200, ridge_l2=1e-2, device=device,
        )
        print(
            f"  [ridge diagnostic] NRMSE={ridge_result['ridge_nrmse']:.4f}  "
            f"R^2={ridge_result['ridge_r2']:.4f}"
        )

    # ---- Generate training data: B independent NARMA streams ----
    u_streams = []
    y_streams = []
    for b in range(n_streams):
        u_b, y_b = narma(
            train_samples_per_stream, order=order, seed=seed + b * 7919,
        )
        u_streams.append(u_b)
        y_streams.append(y_b)
    u_train = torch.stack(u_streams, dim=0)        # (B, n_samples)
    y_train = torch.stack(y_streams, dim=0)        # (B, n_samples)

    # Test data (single stream, fresh seed)
    u_test, y_test = narma(1000, order=order, seed=seed + 10000)

    # Scale input to fabric rails (bipolar recommended).
    u_train = scale_input_to_rails(u_train, bipolar=bipolar)
    u_test = scale_input_to_rails(u_test, bipolar=bipolar)

    res = train_fabric(
        net, u_train, y_train,
        batch_size=n_streams, epochs=epochs, tbptt_chunk=tbptt_chunk,
        lr=1e-3, weight_decay=1e-4, device=device,
        verbose=True,
        use_amp=use_amp,
        standardize=True,
        grad_clip=1.0,
        val_every=val_every,
        val_u_test=val_u_test if val_u_test is not None else u_test,
        val_y_test=val_y_test if val_y_test is not None else y_test,
        val_washout=200,
    )

    # Evaluate: denormalize predictions using training-set stats.
    print("  evaluating on test set...")
    y_pred, y_te = _evaluate_fabric_direct(
        net, u_test, y_test, washout=200, device=device,
        y_mean=res["y_mean"], y_std=res["y_std"],
    )
    nr = nrmse(y_pred, y_te)
    r2v = r2(y_pred, y_te)

    # Memory capacity: collect fabric states on the train set
    # MC reconstructs u(n-k) for k=1..20 (Jaeger 2001), not y.
    # Use the first stream only to keep MC computation cheap.
    states_tr = collect_fabric_states(net, u_train[0], device=device)
    u_train_first = u_train[0].to(device)
    mc_per_delay, mc_total = memory_capacity(
        states_tr, u_train_first, max_delay=20, ridge_l2=1e-2,
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    out = {
        "nrmse": nr,
        "r2": r2v,
        "mc_total": mc_total,
        "mc_per_delay": mc_per_delay,
        "train_loss_history": res["train_loss_history"],
        "raw_leak_grad_history": res.get("raw_leak_grad_history", []),
        "hidden_rms_history": res.get("hidden_rms_history", []),
        "val_nrmse_history": res.get("val_nrmse_history", []),
        "val_r2_history": res.get("val_r2_history", []),
        "val_epochs": res.get("val_epochs", []),
        "n_params": n_params,
        "freeze_read": freeze_read,
        "t_span": t_span,
        "num_steps": num_steps,
    }
    if ridge_result:
        out["ridge_nrmse"] = ridge_result["ridge_nrmse"]
        out["ridge_r2"] = ridge_result["ridge_r2"]
    return out


def decide(fr_off_res: dict[str, Any], fr_on_res: dict[str, Any],
           threshold: float = 0.05, abs_nrmse_threshold: float = 0.5) -> str:
    """Pre-registered decision rule.

    Returns one of:
        "CONTROL_PATH_COMMIT" — evolving core adds no value
        "SIGNAL_PATH_OPEN"    — evolving core is alive
        "NEUTRAL"             — within tolerance or both conditions too poor
    """
    # If both conditions have poor absolute NRMSE, neither is working
    # well enough to draw conclusions.
    if fr_off_res["nrmse"] > abs_nrmse_threshold:
        return "NEUTRAL"
    diff = fr_on_res["nrmse"] - fr_off_res["nrmse"]
    if abs(diff) < threshold * 0.5:
        return "NEUTRAL"
    if diff > 0:
        # freeze_read on is worse than off -> evolving core wins
        return "SIGNAL_PATH_OPEN"
    return "CONTROL_PATH_COMMIT"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--order", type=int, choices=[10, 20], default=10,
                        help="NARMA order (10 or 20, default 10)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated list of RNG seeds (default '0,1,2')")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Fabric training epochs per condition (default 200)")
    parser.add_argument("--tbptt-chunk", type=int, default=25,
                        help="TBPTT chunk size K (samples per optimizer step; default 25; "
                             "was 50 historically — 25 gives 2x more optimizer steps "
                             "per epoch at half the sequential compute per step)")
    parser.add_argument("--n-streams", type=int, default=4,
                        help="Number of parallel independent NARMA streams (default 4)")
    parser.add_argument("--train-samples", type=int, default=2500,
                        help="Samples per training stream (default 2500)")
    parser.add_argument("--output", type=Path,
                        default=Path("./output/narma_exp"),
                        help="Output directory (default ./output/narma_exp)")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run the baselines (ridge, esn, mlp, mlp_large, lstm, lstm_large); skip fabric training")
    parser.add_argument("--fabric-only", action="store_true",
                        help="Only run the two fabric conditions; skip baselines")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device (default cpu)")
    parser.add_argument("--unipolar", action="store_true",
                        help="Use unipolar input drive [0, +x_max] instead of bipolar [-x_max, +x_max]")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed-precision (AMP) training")
    parser.add_argument("--ridge-diagnostic", action="store_true",
                        help="Run ridge-on-frozen-states diagnostic before training")
    parser.add_argument("--t-span", type=float, default=1.0,
                        help="Override t_span per sample for NARMA (default 1.0; "
                             "old value 7.0 gives ~3-sample memory horizon, too short)")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Heun steps per sample (default 4; with t_span=1.0, dt=0.25; "
                             "was 6 historically — 4 cuts per-sample compute by 33%)")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the per-sample Heun steps with torch.compile "
                             "for fused CUDA kernels. ~2-3x speedup on GPU. "
                             "Adds ~10-30s compilation overhead on first call.")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Evaluate test-set NRMSE/R^2 every N epochs (default 5). "
                             "Set to 0 to disable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[narma] order={args.order}, seeds={seeds}, output={out_dir}")
    print(
        f"[narma] n_streams={args.n_streams}, train_samples={args.train_samples}, "
        f"tbptt_chunk={args.tbptt_chunk}, amp={not args.no_amp}, device={args.device}"
    )
    print(
        f"[narma] t_span={args.t_span} (per sample), num_steps={args.num_steps}, "
        f"bipolar={not args.unipolar}"
    )

    # Note: --t-span and --num-steps are passed to run_fabric_condition
    # explicitly, where the preset is built inline via make_narma_preset().
    # We do NOT patch config.PRESETS / config.PRESET_NARMA10 here — those
    # bindings are imported at module load and would silently ignore the
    # patch. The inline build guarantees the override actually takes effect.

    all_results: list[dict[str, Any]] = []

    if not args.fabric_only:
        print("\n[narma] === Baselines ===")
        for i, seed in enumerate(seeds, 1):
            print(f"  [{i}/{len(seeds)}] Baselines seed={seed}")
            base = run_baselines(args.order, seed, device=args.device)
            for cond, vals in base.items():
                all_results.append({
                    "seed": seed,
                    "condition": cond,
                    "nrmse": vals["nrmse"],
                    "r2": vals["r2"],
                    "n_params": vals.get("n_params", ""),
                    "total_params": vals.get("total_params", ""),
                    "hidden_dim": vals.get("hidden_dim", ""),
                })
                print(
                    f"    {cond:>12}  NRMSE={vals['nrmse']:.4f}  "
                    f"R^2={vals['r2']:.4f}  "
                    f"params={vals.get('n_params', '')}"
                )

    if not args.baselines_only:
        print("\n[narma] === Fabric conditions ===")
        # Build a flat list of (freeze_read, seed) for progress tracking
        fabric_jobs = [(fr, s) for fr in (False, True) for s in seeds]
        for idx, (freeze_read, seed) in enumerate(fabric_jobs, 1):
            cond_name = "fabric_fr_off" if not freeze_read else "fabric_fr_on"
            fr_str = "OFF (evolving core)" if not freeze_read else "ON (frozen core)"
            print(f"  [{idx}/{len(fabric_jobs)}] {cond_name}  seed={seed}  "
                  f"freeze_read={fr_str}")
            res = run_fabric_condition(
                args.order, seed, freeze_read=freeze_read,
                epochs=args.epochs,
                tbptt_chunk=args.tbptt_chunk,
                n_streams=args.n_streams,
                train_samples_per_stream=args.train_samples,
                device=args.device,
                use_amp=not args.no_amp,
                bipolar=not args.unipolar,
                do_ridge_diagnostic=args.ridge_diagnostic,
                t_span=args.t_span,
                num_steps=args.num_steps,
                compile_sequence=args.compile,
                val_every=args.val_every,
            )
            result_row = {
                "seed": seed,
                "condition": cond_name,
                "nrmse": res["nrmse"],
                "r2": res["r2"],
                "mc_total": res["mc_total"],
                "n_params": res["n_params"],
                "total_params": res["n_params"],
                "hidden_dim": "",
            }
            if "ridge_nrmse" in res:
                result_row["ridge_nrmse"] = res["ridge_nrmse"]
                result_row["ridge_r2"] = res["ridge_r2"]
            all_results.append(result_row)
            extra = ""
            if "ridge_nrmse" in res:
                extra = f"  ridge_NRMSE={res['ridge_nrmse']:.4f}"
            print(
                f"  RESULT: {cond_name} seed={seed}  NRMSE={res['nrmse']:.4f}  "
                f"R^2={res['r2']:.4f}  MC={res['mc_total']:.2f}  "
                f"params={res['n_params']}{extra}"
            )

    # ---- Aggregate per condition ----
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        by_cond.setdefault(r["condition"], []).append(r)

    summary_lines = [
        f"NARMA-{args.order} -- {len(seeds)} seeds -- final results",
        "=" * 60,
    ]
    csv_lines = ["condition,seed,nrmse,r2,mc_total,trained_params,total_params,hidden_dim"]
    for cond in sorted(by_cond):
        runs = by_cond[cond]
        nrmse_vals = [r["nrmse"] for r in runs]
        r2_vals = [r["r2"] for r in runs]
        mc_vals = [r.get("mc_total") for r in runs if "mc_total" in r]
        mc_str = (
            f"  MC={sum(mc_vals) / len(mc_vals):.2f} +/- "
            f"{(sum((m - sum(mc_vals) / len(mc_vals)) ** 2 for m in mc_vals) / len(mc_vals)) ** 0.5:.2f}"
            if mc_vals else ""
        )
        # Complexity string: trained/total params + hidden_dim
        tp_vals = [r.get("total_params") for r in runs if r.get("total_params") != ""]
        tp_str = ""
        if tp_vals:
            tp = tp_vals[0]
            if isinstance(tp, (int, float)) and tp > 0:
                # Check if trained differs from total (ESN case)
                tr_vals = [r.get("n_params") for r in runs if r.get("n_params") != ""]
                tr = tr_vals[0] if tr_vals else tp
                if tr != tp:
                    tp_str = f"  params={tr}t/{tp}T"
                else:
                    hd_vals = [r.get("hidden_dim") for r in runs if r.get("hidden_dim") != ""]
                    if hd_vals and hd_vals[0] != "":
                        tp_str = f"  params={tp} (hid={hd_vals[0]})"
                    else:
                        tp_str = f"  params={tp}"
        summary_lines.append(
            f"{cond:>14}  NRMSE={sum(nrmse_vals) / len(nrmse_vals):.4f} +/- "
            f"{(sum((v - sum(nrmse_vals) / len(nrmse_vals)) ** 2 for v in nrmse_vals) / len(nrmse_vals)) ** 0.5:.4f}  "
            f"R^2={sum(r2_vals) / len(r2_vals):.4f} +/- "
            f"{(sum((v - sum(r2_vals) / len(r2_vals)) ** 2 for v in r2_vals) / len(r2_vals)) ** 0.5:.4f}"
            f"{mc_str}{tp_str}"
        )
        for r in runs:
            mc = r.get("mc_total", "")
            tp = r.get("n_params", "")
            tt = r.get("total_params", "")
            hd = r.get("hidden_dim", "")
            csv_lines.append(
                f"{cond},{r['seed']},{r['nrmse']:.6f},{r['r2']:.6f},"
                f"{mc if mc != '' else ''},{tp if tp != '' else ''},"
                f"{tt if tt != '' else ''},{hd if hd != '' else ''}"
            )

    summary_text = "\n".join(summary_lines) + "\n"
    print("\n" + summary_text)

    (out_dir / "results_table.txt").write_text(summary_text)
    (out_dir / "results_table.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- Decision (pre-registered, NARMA-20 only per spec) ----
    if (
        args.order == 20
        and not args.baselines_only
        and "fabric_fr_off" in by_cond
        and "fabric_fr_on" in by_cond
    ):
        fr_off = by_cond["fabric_fr_off"]
        fr_on = by_cond["fabric_fr_on"]
        # Use mean NRMSE across seeds
        fr_off_mean = sum(r["nrmse"] for r in fr_off) / len(fr_off)
        fr_on_mean = sum(r["nrmse"] for r in fr_on) / len(fr_on)
        verdict = decide(
            {"nrmse": fr_off_mean}, {"nrmse": fr_on_mean},
        )
        verdict_text = (
            f"Pre-registered decision rule on NARMA-{args.order}:\n"
            f"  fabric freeze_read OFF: NRMSE = {fr_off_mean:.4f}\n"
            f"  fabric freeze_read ON:  NRMSE = {fr_on_mean:.4f}\n"
            f"  difference (ON - OFF):  {fr_on_mean - fr_off_mean:+.4f}\n"
            f"\n"
            f"Verdict: {verdict}\n"
            f"\n"
            f"Interpretation:\n"
            f"  - CONTROL_PATH_COMMIT: evolving core is dead. Commit fully to\n"
            f"    the control-path equalization paper (Experiment 2).\n"
            f"  - SIGNAL_PATH_OPEN: evolving core is alive. Continue with\n"
            f"    Experiment 3 (signal-path equalization).\n"
            f"  - NEUTRAL: within tolerance. Decide based on wall-clock\n"
            f"    cost and architecture simplicity — frozen core is cheaper.\n"
        )
        (out_dir / "decision.txt").write_text(verdict_text)
        print("\n" + verdict_text)

    print(f"\n[narma] Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())