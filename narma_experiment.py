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

from config import OPTIM, PRESET_NARMA10, PRESET_NARMA20
from cell_library import make_cell_library
from topology import build_net_from_preset


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
                         u_max: float = 0.5) -> torch.Tensor:
    """Scale u(n) in [0, u_max] to V_u(n) in [0, x_max] (fills rails)."""
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

    torch.manual_seed(seed)
    model = LSTMRegressor(hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_samples = u_train.shape[0]
    n_windows = n_samples // bptt_window

    model.train()
    for epoch in range(epochs):
        window_starts = torch.randperm(n_windows) * bptt_window
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
            u_val_seq = u_val_override.unsqueeze(0)
            y_val_pred, _ = model(u_val_seq)
            y_val_pred = y_val_pred.squeeze(0)
        val_nrmse = nrmse(y_val_pred, y_val_override)
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
        X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
        XtX = X_aug.T @ X_aug + ridge_l2 * torch.eye(X_aug.shape[1])
        W = torch.linalg.solve(XtX, X_aug.T @ y)
        y_pred = X_aug @ W
        r2_list.append(r2(y_pred, y))
    mc_total = sum(max(r, 0.0) for r in r2_list)
    return r2_list, mc_total


# ---------------------------------------------------------------------------
# Fabric training
# ---------------------------------------------------------------------------

def _progress_iter(iterable, desc, total=None, disable=False):
    """Wrap an iterable with tqdm if available, else pass through."""
    if _HAS_TQDM and not disable:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def train_fabric(
    net: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int = 128,
    epochs: int = 200,
    bptt_window: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
    use_amp: bool = True,
    accum_steps: int = 2,
) -> dict[str, Any]:
    """Train the fabric with state carryover and truncated BPTT.

    For efficiency we pack ``bptt_window`` consecutive samples as a single
    batch and call the network ONCE per window (not once per sample). The
    state is initialized per window and propagates through the integration
    via Heun — each of the ``num_steps_per_sample`` Heun steps samples a
    bit of the boundary OTA current and the state evolves. This is the
    same physical setup as calling the network per sample (since the
    sampled-and-held input is constant during integration) but orders of
    magnitude faster in Python.

    Args:
        batch_size: Number of windows per micro-batch. Gradients are
            accumulated over ``accum_steps`` micro-batches before each
            optimizer step (effective batch = batch_size * accum_steps).
        use_amp: Enable mixed-precision training via ``torch.amp``.
        accum_steps: Gradient accumulation steps before optimizer.step().

    Returns:
        Dict with ``train_loss_history`` and ``final_loss``.
    """
    net.to(device)
    u_train = u_train.to(device)
    y_train = y_train.to(device)
    n_samples = u_train.shape[0]
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    t_start = time.time()

    n_windows = n_samples // bptt_window
    if n_windows == 0:
        raise ValueError(
            f"u_train has only {n_samples} samples, must be > bptt_window="
            f"{bptt_window}; reduce --bptt-window"
        )

    # Micro-batch size for splitting window starts
    B = min(batch_size, n_windows)

    # Mixed-precision scaler (no-op on CPU)
    amp_enabled = use_amp and device == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # Logging interval: every ~5% of epochs, plus first and last
    log_interval = max(1, epochs // 20)
    epoch_times: list[float] = []

    for epoch in range(epochs):
        net.train()
        # Randomise the window order each epoch
        window_starts = torch.randperm(n_windows) * bptt_window
        # Split into micro-batches of size B
        chunks = window_starts.split(B)

        epoch_loss = 0.0
        n_batches = 0
        optim.zero_grad()

        epoch_iter = _progress_iter(
            enumerate(chunks), desc=f"epoch {epoch:3d}/{epochs}",
            total=len(chunks), disable=not verbose,
        )
        for step_idx, chunk in epoch_iter:
            # Build a single batch of ``bptt_window`` consecutive samples
            # for each starting position. Shape: (B, bptt_window).
            idx = (
                chunk.unsqueeze(1)
                + torch.arange(bptt_window, dtype=torch.long).unsqueeze(0)
            )
            u_chunk = u_train[idx]                    # (B, bptt_window)
            y_chunk = y_train[idx]                    # (B, bptt_window)
            total_loss = 0.0
            state = None  # state is (B, hid+proj+out_ode)
            with autocast("cuda", enabled=amp_enabled):
                for s in range(bptt_window):
                    u_b = u_chunk[:, s].unsqueeze(-1)     # (B, 1)
                    y_b = y_chunk[:, s]                    # (B,)
                    y_pred, _, final_state = net(
                        u_b, initial_state=state, return_final_state=True,
                    )
                    y_pred = y_pred.squeeze(-1)             # (B,)
                    loss_step = F.mse_loss(y_pred, y_b)
                    total_loss = total_loss + loss_step
                    state = final_state.detach()  # truncated BPTT
                total_loss = total_loss / bptt_window

            # Scaled backward for AMP
            scaler.scale(total_loss).backward()
            epoch_loss += float(total_loss.item())
            n_batches += 1

            # Gradient accumulation: step every accum_steps micro-batches
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(chunks):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            if _HAS_TQDM and verbose:
                epoch_iter.set_postfix(loss=f"{total_loss.item():.4f}")

        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)

        # Logging
        t_epoch = time.time() - t_start
        epoch_times.append(t_epoch)
        if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
            if len(epoch_times) >= 2:
                dt = epoch_times[-1] - epoch_times[-2]
                eta = dt * (epochs - 1 - epoch)
                eta_str = f"ETA {eta:.0f}s"
            else:
                eta_str = ""
            print(
                f"  fabric epoch {epoch:3d}/{epochs}  loss={avg:.6f}  "
                f"{t_epoch:.1f}s  {eta_str}"
            )

    total_elapsed = time.time() - t_start
    print(f"  fabric training complete: {epochs} epochs in {total_elapsed:.1f}s")
    return {"train_loss_history": history, "final_loss": history[-1] if history else float("nan")}


@torch.no_grad()
def evaluate_fabric(
    net: nn.Module,
    u_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    washout: int = 200,
    device: str = "cpu",
    chunk: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fabric over the test sequence with state carryover.

    Returns:
        ``(y_pred, y_test_aligned)``: predictions and ground truth after
        ``washout`` initial samples have been discarded (reservoir
        convention).
    """
    net.eval()
    net.to(device)
    u_test = u_test.to(device)
    y_test = y_test.to(device)
    T = u_test.shape[0]
    # Start from zero state
    state = None
    preds: list[torch.Tensor] = []
    # Run sample by sample, threading the state
    # We do this in chunks for efficiency
    for t in range(0, T):
        u_b = u_test[t].view(1, 1)
        y_pred, _, final_state = net(
            u_b, initial_state=state, return_final_state=True,
        )
        preds.append(y_pred.view(-1))
        state = final_state.detach()
    y_pred = torch.stack(preds).squeeze(-1)
    return y_pred[washout:], y_test[washout:]


def collect_fabric_states(
    net: nn.Module,
    u: torch.Tensor,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Collect the per-sample fabric state (last layer only) for MC eval.

    Returns:
        Tensor of shape ``(T, stage_width)`` — the final ODE state at
        every timestep.
    """
    net.eval()
    net.to(device)
    u = u.to(device)
    T = u.shape[0]
    states: list[torch.Tensor] = []
    state = None
    with torch.no_grad():
        for t in range(T):
            u_b = u[t].view(1, 1)
            _, _, final_state = net(
                u_b, initial_state=state, return_final_state=True,
            )
            states.append(final_state.view(-1))
            state = final_state.detach()
    return torch.stack(states)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_baselines(order: int, seed: int) -> dict[str, dict[str, Any]]:
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
    mlp = MLPRegressor(n_taps=n_taps, hidden_dim=25)
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
    mlp_large = MLPRegressor(n_taps=n_taps, hidden_dim=237)
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
            )
            if res["val_nrmse"] is not None and res["val_nrmse"] < best_lstm_val_nrmse:
                best_lstm_val_nrmse = res["val_nrmse"]
                best_lstm_hd = hidden_dim
                best_lstm_lr = lr

    # Refit on the full train set with the best hyperparameters
    lstm_res_dict = train_and_eval_lstm(
        order=order, seed=seed, hidden_dim=best_lstm_hd,
        bptt_window=50, epochs=200, washout=washout, lr=best_lstm_lr,
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
        )
        if res["val_nrmse"] is not None and res["val_nrmse"] < best_lstm_large_val_nrmse:
            best_lstm_large_val_nrmse = res["val_nrmse"]
            best_lstm_large_lr = lr

    # Refit on the full train set with the best hyperparameters
    lstm_large_res_dict = train_and_eval_lstm(
        order=order, seed=seed, hidden_dim=42,
        bptt_window=50, epochs=200, washout=washout, lr=best_lstm_large_lr,
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
    bptt_window: int = 50,
    device: str = "cpu",
    batch_size: int = 128,
    use_amp: bool = True,
    accum_steps: int = 2,
) -> dict[str, Any]:
    """Train one fabric condition and return its results.

    Args:
        order: NARMA order (10 or 20).
        seed: RNG seed.
        freeze_read: If True, use the frozen-core configuration.
        epochs: Training epochs.
        bptt_window: Truncated BPTT window in samples.
        batch_size: Micro-batch size for training (default 128).
        use_amp: Enable mixed-precision training (default True).
        accum_steps: Gradient accumulation steps (default 2).

    Returns dict with:
        - "nrmse": test NRMSE on the washed test set
        - "r2": test R^2
        - "mc_total": memory capacity (sum of R^2 over k=1..20)
        - "mc_per_delay": list of R^2 per delay
        - "train_loss_history": list of training losses
        - "n_params": parameter count
    """
    preset = PRESET_NARMA20 if order == 20 else PRESET_NARMA10
    cell_lib = make_cell_library("tanh")
    torch.manual_seed(seed)
    net = build_net_from_preset(
        f"narma{order}",
        cell_lib=cell_lib,
        boundary_fan_out=preset["boundary_fan_out"],
        enable_temporal_readout=True,
        freeze_read=freeze_read,
    )

    u_train, y_train = narma(3000, order=order, seed=seed)
    u_test, y_test = narma(1000, order=order, seed=seed + 10000)

    # Scale input to fabric rails: V_u(n) = 3.0 * u(n) / 0.5 V
    u_train = scale_input_to_rails(u_train)
    u_test = scale_input_to_rails(u_test)

    res = train_fabric(
        net, u_train, y_train,
        batch_size=batch_size, epochs=epochs, bptt_window=bptt_window,
        lr=1e-3, weight_decay=1e-4, device=device,
        verbose=True,
        use_amp=use_amp,
        accum_steps=accum_steps,
    )

    # Evaluate
    print("  evaluating on test set...")
    y_pred, y_te = evaluate_fabric(net, u_test, y_test, washout=200, device=device)
    nr = nrmse(y_pred, y_te)
    r2v = r2(y_pred, y_te)

    # Memory capacity: collect fabric states on the train set
    # MC reconstructs u(n-k) for k=1..20 (Jaeger 2001), not y.
    states_tr = collect_fabric_states(net, u_train, device=device)
    u_train = u_train.to(device)
    mc_per_delay, mc_total = memory_capacity(
        states_tr, u_train, max_delay=20, ridge_l2=1e-2,
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {
        "nrmse": nr,
        "r2": r2v,
        "mc_total": mc_total,
        "mc_per_delay": mc_per_delay,
        "train_loss_history": res["train_loss_history"],
        "n_params": n_params,
        "freeze_read": freeze_read,
    }


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
    parser.add_argument("--bptt-window", type=int, default=50,
                        help="Truncated BPTT window in samples (default 50)")
    parser.add_argument("--output", type=Path,
                        default=Path("./output/narma_exp"),
                        help="Output directory (default ./output/narma_exp)")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run the baselines (ridge, esn, mlp, mlp_large, lstm, lstm_large); skip fabric training")
    parser.add_argument("--fabric-only", action="store_true",
                        help="Only run the two fabric conditions; skip baselines")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device (default cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Micro-batch size for fabric training (default 128)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed-precision (AMP) training")
    parser.add_argument("--accum-steps", type=int, default=2,
                        help="Gradient accumulation steps before optimizer step (default 2)")
    parser.add_argument("--no-tanh-cell-lib", action="store_true",
                        help="Use simple cell library instead of tanh (testing)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[narma] order={args.order}, seeds={seeds}, output={out_dir}")
    print(f"[narma] batch_size={args.batch_size}, amp={not args.no_amp}, "
          f"accum_steps={args.accum_steps}, device={args.device}")

    all_results: list[dict[str, Any]] = []

    if not args.fabric_only:
        print("\n[narma] === Baselines ===")
        for i, seed in enumerate(seeds, 1):
            print(f"  [{i}/{len(seeds)}] Baselines seed={seed}")
            base = run_baselines(args.order, seed)
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
                bptt_window=args.bptt_window,
                device=args.device,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                accum_steps=args.accum_steps,
            )
            all_results.append({
                "seed": seed,
                "condition": cond_name,
                "nrmse": res["nrmse"],
                "r2": res["r2"],
                "mc_total": res["mc_total"],
                "n_params": res["n_params"],
                "total_params": res["n_params"],
                "hidden_dim": "",
            })
            print(
                f"  RESULT: {cond_name} seed={seed}  NRMSE={res['nrmse']:.4f}  "
                f"R^2={res['r2']:.4f}  MC={res['mc_total']:.2f}  "
                f"params={res['n_params']}"
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