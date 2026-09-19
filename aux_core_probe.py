"""Aux-core input stage probe (plan aux-core-drive, feature spec aux-core-probe).

Twin-level eval-only probe: a learnable feedforward aux core (echo-state,
shift-register, slow-feature bank, residual-distilled echo) replaces the
static 8-tap delay bank as drive for the main twin. Same streams/seeds
everywhere; aux legs (aux-only, aux+scalar, aux-concat readout) plus
references (taps8, scalar, scalar-ESN parity, compressed SVD tapsVD4)
at the Gate-1 best corner with pre-registered SUCCESS/KILL verdicts.
No fabric changes, no feedback, no training; fabric integration earned
only by SUCCESS.

Architecture (strict feedforward; feedback explicitly out of scope per spec):

    input u(t) -> aux core -> aux states (T, n_aux)
                          \\-> main core (Gate-1 best) -> (T, n_main)
    input u(t) ------------------------------------------------> +scalar col (aux+scalar leg only)

    Output states -> ridge readout -> NRMSE/R^2 + per-delay MC + PR + Jacobian
    spectra. Verdict function decides SUCCESS / KILL per flavor.

The module is *side-effect free at import*; only directly-invoked functions
build nets (none here) or read checkpoints (none here).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import narma_advisor_probes as npr  # noqa: E402
import narma_experiment as ne  # noqa: E402
import post_twin_gates as ptg  # noqa: E402


# ---------------------------------------------------------------------------
# Locked control thresholds (aux-core-probe spec).
# ---------------------------------------------------------------------------

PROBE_WASHOUT: int = npr.PROBE_WASHOUT
CANONICAL_HIDDEN: int = 25  # twin node count; matches discrete-twin default
CANONICAL_T_SPAN: float = 1.0  # per-sample integration window (one NARMA step)
CANONICAL_DT_MAX: float = 0.125  # fixed-dt rule (same as post-twin gates)

# Main core operating point (Gate-1 best corner, locked per spec):
# Heun flow-twin with radius=1.0, lambda*T=1.0, input_scale=0.2.
# ``input_scale`` is the drive-power parity point: every leg's drive into
# the main core is scaled to RMS == input_scale (mirroring the taps8
# branch of ``flow_twin_drive``), so MC comparisons across legs compare
# architectures at a matched operating point, not drive powers.
MAIN_RADIUS: float = 1.0
MAIN_LAMT: float = 1.0
MAIN_INPUT_SCALE: float = 0.2
# Pre-linear-regime guard threshold (same as post-twin gates).
PRE_LINEAR_REGIME_MEDIAN: float = ptg.G1_PRE_LINEAR_REGIME_MEDIAN

# Aux flavor set (echo-state first per spec).
AUX_FLAVORS: tuple[str, ...] = ("echo_state", "shift_register", "slow_feature")
# D4 echo flavor tag (built via the standard echo builder; selection is
# by residual-teacher R^2, not NARMA MC).  Defined up here (not with the
# D4 section below) because ALL_FLAVORS is built at module scope.
D4_FLAVOR: str = "residual_distilled"
AUX_LEGS: tuple[str, ...] = ("aux_only", "aux_scalar", "aux_concat")
# D4 flavor joins the standard flavor axis (aux-only leg is its primary).
ALL_FLAVORS: tuple[str, ...] = AUX_FLAVORS + (D4_FLAVOR,)
REF_LEGS: tuple[str, ...] = (
    "taps8", "scalar", "scalar_esn_parity", "tapsvd",
)
# Sparse main fan-out (D2): per aux column the main core sees k
# random dense nodes (radius-restored at build time).
SPARSE_FANOUT_K: tuple[int, ...] = (3, 6)

# Echo-state aux (8 nodes matches tap count for fair comparison).
ECHO_N_AUX: int = 8
ECHO_RADIUS: float = 0.95
ECHO_LEAK_GRID: tuple[float, ...] = (0.3, 1.0)

# Shift-register aux (chain topology, head injection).
SR_N_AUX: int = 8
SR_GAIN_GRID: tuple[float, ...] = (0.7, 0.9)

# Slow-feature bank (decoupled leaky integrators at log-spread tau).
SFB_N_AUX: int = 8
SFB_LOG_TAU_RANGE: tuple[float, float] = (math.log(1.0), math.log(40.0))

# Aux-seeds (small grid per spec).
AUX_SEEDS: tuple[int, ...] = (0, 1, 2)
DATA_SEEDS: tuple[int, ...] = (0,)

# D4 — residual-distilled aux: top-r PCs of the tap-minus-main residual
# form the teacher; aux hyperparam selection maximizes mean R^2 against
# the teacher.  ``D4_RESIDUAL_TEACHER_K`` caps the teacher dimension.
D4_RESIDUAL_TEACHER_K: int = 4

# D4 orthogonality gate: max |corr| between any teacher column and any
# main state column, asserted at audit time (aux must learn features
# orthogonal to what main+scalar already covers).
D4_TEACHER_ORTHOG_MAX_CORR: float = 0.10

# Sparse W_in builder for the main core (D2) reuses the Gate-3
# ``ptg.sparse_random_W_in`` machinery (one column per drive col, k
# nonzero entries each, radius-power-preserving rescale).

# D3 locked ref name (spec: tapsVD4 = SVD top-4 PCs of taps8).
TAPSVD_K: int = 4

# Canonical reference edge budget (taps8 main fan-in at default dims:
# CANONICAL_HIDDEN x 8 = 200).  Kept as documentation of the pre-registered
# budget; live rows/verdicts derive ``ref_total_edges`` from the actual
# hidden_dim x n_taps (see _aux_probe_row).
REF_FAN_IN_EDGES: int = CANONICAL_HIDDEN * 8  # 200

# Verdict thresholds (pre-registered per spec).
# SUCCESS path 1: aux matches taps8 MC within tolerance at <= edge cost.
SUCCESS_MC_MATCH_RATIO: float = 0.85  # aux MC >= 0.85 * taps8 MC
SUCCESS_EDGE_TOL_FACTOR: float = 1.50  # allow up to 1.5x ref edge budget
# SUCCESS path 2: aux beats scalar by a margin much greater than its cost.
SUCCESS_BEAT_SCALAR_MARGIN: float = 1.50  # aux MC >= 1.5 * scalar MC
# KILL: canonical-correlation (aux states vs taps8 delay bank).
TAP_SUBSPACE_KILL_CCORR: float = 0.90
# KILL: aux void/dead (low PR or near-zero MC).
VOID_PR_THRESHOLD: float = 1.05  # below this, states are rank-1 / dead
VOID_MC_THRESHOLD: float = 0.20  # below this, aux is dead
# KILL: no gain over main+scalar (aux+scalar must beat scalar alone).
NO_GAIN_OVER_SCALAR_THRESHOLD: float = 0.05  # aux+scalar MC - scalar MC below this
# D1 concat-readout note (informational only — never a SUCCESS/KILL
# path): appended when an engaged aux_concat row's [main; aux] readout
# beats its main-only readout by more than this NRMSE margin.
CONCAT_GAIN_NOTE_DELTA: float = 0.02

# NOTE: no twin-MC "holds" bar is needed here; the verdict rules carry
# their own pre-registered thresholds (SUCCESS_MC_MATCH_RATIO,
# VOID_MC_THRESHOLD, ...).


# ---------------------------------------------------------------------------
# Aux builders.
# ---------------------------------------------------------------------------


def echo_state_aux_build(
    *, n_aux: int, leak: float, radius: float,
    w_seed: int, win_seed: int,
    dtype: torch.dtype, device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build an echo-state aux core.

    Returns a dict with ``W_aux`` (n_aux, n_aux, spectral-radius normalized),
    ``W_in_aux`` (n_aux, 1) scalar input driver, ``leak`` (scalar), and
    ``n_aux``.
    """
    if not (0.0 < float(leak) <= 1.0):
        raise ValueError(f"echo_state aux leak must satisfy 0 < l <= 1, got {leak}")
    if n_aux < 2:
        raise ValueError(f"echo_state aux n_aux must be >= 2, got {n_aux}")
    g_w = torch.Generator(device="cpu").manual_seed(int(w_seed))
    W_raw = torch.randn(n_aux, n_aux, generator=g_w, dtype=torch.float32)
    eigs = torch.linalg.eigvals(W_raw.to(torch.float64))
    rho = float(eigs.abs().max().item())
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError(f"echo_state aux W spectral radius invalid: {rho}")
    W_aux = (W_raw * (float(radius) / rho)).to(dtype=dtype, device=device)
    g_in = torch.Generator(device="cpu").manual_seed(int(win_seed))
    W_in_aux = torch.randn(
        n_aux, 1, generator=g_in, dtype=torch.float32,
    ).to(dtype=dtype, device=device)
    return {
        "W_aux": W_aux, "W_in_aux": W_in_aux,
        "leak": torch.tensor(float(leak), dtype=dtype, device=device),
        "n_aux": int(n_aux),
    }


def shift_register_aux_build(
    *, n_aux: int, gain: float, head_inject: float,
    w_seed: int,
    dtype: torch.dtype, device: torch.device,
) -> dict[str, torch.Tensor | int]:
    """Build a shift-register aux core (exact chain; no randomness).

    Topology (per spec, "interpretable learned delay line; exact chain
    asserted"):

        x[0](t) = head_inject * u(t)
        x[i](t) = gain * x[i-1](t-1)  for i = 1..n_aux-1

    All edges are scalar *g*; there are n_aux-1 recurrent edges.  No
    randomness is needed for the chain itself; ``w_seed`` is reserved for
    future variants but unused here.
    """
    if n_aux < 2:
        raise ValueError(f"shift_register aux n_aux must be >= 2, got {n_aux}")
    if not (0.0 < float(gain) <= 1.5):
        raise ValueError(
            f"shift_register aux gain must satisfy 0 < g <= 1.5, got {gain}"
        )
    if not (0.0 <= float(head_inject) <= 10.0):
        raise ValueError(
            f"shift_register aux head_inject must satisfy "
            f"0 <= hi <= 10, got {head_inject}"
        )
    return {
        "gain": float(gain),
        "head_inject": float(head_inject),
        "n_aux": int(n_aux),
        # Reserved for future variants; unused by the current chain.
        "w_seed": int(w_seed),
        "dtype": dtype,
        "device": device,
    }


def slow_feature_aux_build(
    *, n_aux: int, log_tau_lo: float, log_tau_hi: float,
    w_seed: int,
    dtype: torch.dtype, device: torch.device,
) -> dict[str, torch.Tensor | int]:
    """Build a slow-feature aux core (decoupled leaky integrators).

    Per integrator i in 0..n_aux-1:

        x[i](t+1) = (1 - 1/tau[i]) * x[i](t) + (1/tau[i]) * u(t)

    tau[i] = exp(log_tau_lo + (log_tau_hi - log_tau_lo) * i / (n_aux-1)),
    log-spread across the bank.  No recurrent edges (decoupled), so
    aux_recurrent = 0; aux_fan_in = n_aux (each integrator has its own
    scalar drive path).
    """
    if n_aux < 2:
        raise ValueError(f"slow_feature aux n_aux must be >= 2, got {n_aux}")
    if not (float(log_tau_lo) < float(log_tau_hi)):
        raise ValueError(
            f"slow_feature aux requires log_tau_lo < log_tau_hi, "
            f"got {log_tau_lo} >= {log_tau_hi}"
        )
    # tau values: linspace from lo to hi on log scale, then exp.
    grid = torch.linspace(
        float(log_tau_lo), float(log_tau_hi), n_aux,
        dtype=torch.float32,
    )
    tau = torch.exp(grid).to(dtype=dtype, device=device)
    return {
        "tau": tau,
        "n_aux": int(n_aux),
        "w_seed": int(w_seed),
        "dtype": dtype,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Aux drive functions (deterministic, eval-only).
# ---------------------------------------------------------------------------


def echo_state_aux_run(
    *, spec: dict[str, torch.Tensor], u_stream: torch.Tensor,
) -> torch.Tensor:
    """Run the echo-state aux core and return ``(T, n_aux)`` states.

    Update: ``x[t+1] = (1-leak)*x[t] + leak*tanh(W_aux@x[t] + W_in_aux@u[t])``.
    """
    W_aux = spec["W_aux"]
    W_in_aux = spec["W_in_aux"]
    leak = float(spec["leak"].item())
    n_aux = int(spec["n_aux"])
    dev = W_aux.device
    dtype = W_aux.dtype
    u_dev = u_stream.to(dtype=dtype, device=dev)
    T = u_dev.shape[0]
    states = torch.zeros(T, n_aux, dtype=dtype, device=dev)
    x = torch.zeros(n_aux, dtype=dtype, device=dev)
    for t in range(T):
        pre = W_aux @ x + W_in_aux.squeeze(-1) * u_dev[t]
        x_new = torch.tanh(pre)
        x = (1.0 - leak) * x + leak * x_new
        states[t] = x.detach()
    return states


def shift_register_aux_run(
    *, spec: dict[str, Any], u_stream: torch.Tensor,
) -> torch.Tensor:
    """Run the shift-register aux core (exact chain).

    For each t in 0..T-1:
        x[t][0] = head_inject * u[t]
        x[t][i] = gain * x[t-1][i-1]  for i in 1..n_aux-1

    For t=0, the lag term is zero (the chain starts empty).
    """
    gain = float(spec["gain"])
    head_inject = float(spec["head_inject"])
    n_aux = int(spec["n_aux"])
    dtype = spec["dtype"]
    dev = spec["device"]
    u_dev = u_stream.to(dtype=dtype, device=dev)
    T = u_dev.shape[0]
    states = torch.zeros(T, n_aux, dtype=dtype, device=dev)
    if T == 0:
        return states
    states[0, 0] = head_inject * u_dev[0]
    for t in range(1, T):
        states[t, 0] = head_inject * u_dev[t]
        # Chain shifts previous column right.
        states[t, 1:] = gain * states[t - 1, :-1]
    return states


def slow_feature_aux_run(
    *, spec: dict[str, Any], u_stream: torch.Tensor,
) -> torch.Tensor:
    """Run the slow-feature bank (decoupled leaky integrators).

    For each i in 0..n_aux-1:
        x[t+1][i] = (1 - 1/tau[i]) * x[t][i] + (1/tau[i]) * u[t]

    All tau[i] are log-spread; each integrator carries its own scalar
    drive path (fan-in = n_aux, recurrent = 0).
    """
    tau = spec["tau"]
    n_aux = int(spec["n_aux"])
    dtype = spec["dtype"]
    dev = spec["device"]
    T = u_stream.shape[0]
    states = torch.zeros(T, n_aux, dtype=dtype, device=dev)
    if T == 0:
        return states
    inv_tau = 1.0 / tau  # (n_aux,)
    one_m = 1.0 - inv_tau  # (n_aux,)
    u_dev = u_stream.to(dtype=dtype, device=dev)
    # x[0] is the result of integrating u[0] from a zero initial state for
    # one unit of time: x[0] = (1/tau) * u[0].
    states[0] = inv_tau * u_dev[0]
    for t in range(1, T):
        states[t] = one_m * states[t - 1] + inv_tau * u_dev[t]
    return states


# ---------------------------------------------------------------------------
# D3 — compressed-bank reference (SVD top-k PCs of the taps bank) and
# D4 — residual-distilled teacher (offline, ridge/SVD only).
# ---------------------------------------------------------------------------


def tapsvd_pc_bank(
    u_stream: torch.Tensor, *, k: int, n_taps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic SVD top-``k`` PCs of the taps delay bank.

    Computed offline on the caller's stream (per spec: deterministic,
    sign-pinned; no RNG is involved — the bank is a pure function of
    the stream, so no seed parameter exists).  Returns ``(pc_bank
    (T, k), mean (n_taps,), components (k, n_taps))``.  Sign pinning:
    each component's sign is fixed so that its element of largest
    magnitude is positive, removing the SVD sign ambiguity; ordering is
    by descending singular value, which ``torch.linalg.svd`` guarantees.
    """
    if k < 1 or k > n_taps:
        raise ValueError(
            f"tapsvd_pc_bank: k must satisfy 1 <= k <= n_taps, got {k}"
        )
    taps = ptg.nlc_twin_delay_bank(u_stream, n_taps=n_taps)
    taps64 = taps.detach().to(dtype=torch.float64)
    mean = taps64.mean(dim=0)
    centered = taps64 - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    comps = Vh[:k]  # (k, n_taps), descending singular values
    # Sign pinning: flip so the largest-|.| element of each component is
    # positive (deterministic under BLAS sign ambiguity).
    for i in range(comps.shape[0]):
        j = int(comps[i].abs().argmax().item())
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    pc_bank = ((centered @ comps.T) + 0.0).to(torch.float32)
    return pc_bank, mean.to(torch.float32), comps.to(torch.float32)


def residual_teacher_build(
    *, u_stream: torch.Tensor,
    hidden_dim: int, n_taps: int, input_scale: float,
    teacher_k: int,
    radius: float = MAIN_RADIUS, lamT: float = MAIN_LAMT,
    dt: float = CANONICAL_DT_MAX,
) -> dict[str, torch.Tensor | float]:
    """Offline Phase-A residual teacher (D4, eval-only ridge/SVD).

    Steps (per spec):
      (a) run the flow-twin on the scalar harness (u * input_scale) at
          the pinned corner — the paired control readout;
      (b) SVD top PCs of the taps bank;
      (c) project out the main+scalar state span from those PCs, keep
          the top residual directions as the teacher.

    NOTE: the teacher never sees the NARMA targets (no ``y`` parameter
    exists); it is an input-memory teacher, so the whole construction
    stays inside the eval-only ridge/SVD instrument family.  The main
    core here mirrors the scalar reference (same radius/lamT/dt, same
    w_seed = 0) so "what main already covers" is measured at the same
    operating point the verdicts use.

    Returns a dict with the teacher matrix ``(T, r)`` plus provenance:
    ``r`` (kept directions), ``resid_r2_mean`` (mean per-direction
    fraction of teacher variance EXPLAINED by main+scalar states —
    near 0 by construction since the teacher is the projected-out
    residual; large values would flag a broken projection), and
    ``max_abs_corr`` (max |corr| between teacher columns and main
    state columns, asserted < D4_TEACHER_ORTHOG_MAX_CORR at audit
    time).  Washout is applied downstream by :func:`teacher_r2`, not
    here, so the teacher covers the full stream.
    """
    # NOTE: device follows the input stream (a hardcoded CPU device here
    # used to crash flow_twin_run under --device cuda via a CPU/GPU
    # weight/drive mismatch).
    dev = u_stream.device
    dtype = torch.float32
    # (a) main+scalar states: flow-twin driven by the scalar harness
    # (u * input_scale) — the paired control readout.
    scalar_drive = u_stream.to(dtype=dtype, device=dev) * float(input_scale)
    W, W_in = ptg.flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=1, w_seed=0, win_seed=0,
        target_radius=radius, dtype=dtype, device=dev,
    )
    lam = float(lamT) / float(CANONICAL_T_SPAN)
    states = ptg.flow_twin_run(
        W=W, W_in=W_in, drive=scalar_drive.unsqueeze(-1).to(dev),
        lam=lam, dt=dt,
    )
    main_aug = torch.cat(
        [states.to(torch.float64),
         scalar_drive.to(torch.float64).unsqueeze(-1)], dim=1,
    )
    # (b) SVD PCs of the taps bank.
    pc_bank, _mean, comps = tapsvd_pc_bank(
        u_stream, k=min(8, n_taps), n_taps=n_taps,
    )
    pc64 = pc_bank.to(dtype=torch.float64, device=main_aug.device)
    # (c) Project out the main-state span from the PC columns.
    M_aug = torch.cat(
        [main_aug,
         torch.ones(main_aug.shape[0], 1, dtype=torch.float64,
                    device=main_aug.device)], dim=1,
    )
    Q, _R = torch.linalg.qr(M_aug)
    proj = Q @ Q.T
    resid = pc64 - proj @ pc64
    # Re-orthogonalize residual directions via SVD; keep top-teacher_k.
    Ur, Sr, Vrh = torch.linalg.svd(resid, full_matrices=False)
    kept = min(int(teacher_k), int((Sr > 1e-8).sum().item()))
    if kept < 1:
        # Degenerate: taps PCs lie entirely in the main span (no
        # residual information).  Return an empty teacher with NaN
        # diagnostics instead of crashing on an empty max().
        T = pc64.shape[0]
        return {
            "teacher": torch.zeros(T, 0, dtype=torch.float32),
            "r": 0,
            "resid_r2_mean": float("nan"),
            "max_abs_corr": float("nan"),
        }
    teacher = Ur[:, :kept] * Sr[:kept]  # (T, r) residual coordinates
    # Orthogonality diagnostic: max |corr| teacher col vs main col.
    def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_c = a - a.mean(dim=0, keepdim=True)
        b_c = b - b.mean(dim=0, keepdim=True)
        na = a_c.norm(dim=0).clamp_min(1e-12)
        nb = b_c.norm(dim=0).clamp_min(1e-12)
        return (a_c.T @ b_c) / na.unsqueeze(1) / nb.unsqueeze(0)
    corr_mat = _corr(teacher, main_aug)
    max_abs_corr = float(corr_mat.abs().max().item())
    # Fraction of teacher variance explained by main states (want small;
    # ~0 by construction — the teacher is the projected-out residual).
    fit = proj @ teacher
    r2_per = 1.0 - (teacher - fit).pow(2).sum(dim=0) / \
        teacher.pow(2).sum(dim=0).clamp_min(1e-12)
    return {
        "teacher": teacher.to(torch.float32),
        "r": int(kept),
        "resid_r2_mean": float(r2_per.mean().item()),
        "max_abs_corr": max_abs_corr,
    }


def teacher_r2(aux_states: torch.Tensor, teacher: torch.Tensor,
               *, washout: int) -> tuple[float, float]:
    """Mean/max ridge R^2 of the residual teacher predicted FROM aux.

    Direction: each teacher column is regressed on the aux-state block
    (plus bias), then averaged — i.e. "how much of the teacher can the
    aux span".  Centering matches :func:`tap_subspace_canonical_corr`
    (variance around the column mean); the bias column and eye matrix
    live on the aux device (a CPU default here used to crash the solve
    under --device cuda).
    """
    a = aux_states[washout:].detach().to(dtype=torch.float64)
    t = teacher[washout:].detach().to(dtype=torch.float64)
    T = min(a.shape[0], t.shape[0])
    a, t = a[:T], t[:T]
    if T < 4 or a.shape[1] < 1 or t.shape[1] < 1:
        return float("nan"), float("nan")
    dev = a.device
    A_aug = torch.cat(
        [a, torch.ones(T, 1, dtype=torch.float64, device=dev)], dim=1,
    )
    XtX = A_aug.T @ A_aug + 1e-2 * torch.eye(
        A_aug.shape[1], dtype=torch.float64, device=dev,
    )
    try:
        W = torch.linalg.solve(XtX, A_aug.T @ t)
    except Exception:
        return float("nan"), float("nan")
    pred = A_aug @ W
    t_c = t - t.mean(dim=0, keepdim=True)
    ss_tot = t_c.pow(2).sum(dim=0)
    ss_res = (t - pred).pow(2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    r2 = torch.clamp(r2, 0.0, 1.0)
    return float(r2.mean().item()), float(r2.max().item())


# Convenience dispatch by flavor name (builders take per-flavor kwargs,
# so the sweep constructs each flavor explicitly; only the runners share
# one call signature).  D4 reuses the echo-state runner (its aux IS an
# echo-state core; only the SELECTION objective differs).
AUX_RUNNERS = {
    "echo_state": echo_state_aux_run,
    "shift_register": shift_register_aux_run,
    "slow_feature": slow_feature_aux_run,
    D4_FLAVOR: echo_state_aux_run,
}


# ---------------------------------------------------------------------------
# Aux drive assembly (aux-only and aux+scalar legs).
# ---------------------------------------------------------------------------


def aux_drive_concat(
    aux_states: torch.Tensor, u_stream: torch.Tensor, *, leg: str,
) -> torch.Tensor:
    """Concatenate aux states with optional scalar column for the main core.

    ``aux_only``: drive = aux_states (T, n_aux).
    ``aux_scalar``: drive = [aux_states | u_stream.unsqueeze(-1)] (T, n_aux+1).
    ``aux_concat``: drive = aux_states (T, n_aux) -- same drive as
        aux_only, but the readout (handled downstream) is over
        [main_states; aux_states] instead of main-only.  Declared per
        spec; concatenation at the readout happens in
        :func:`_aux_probe_instrument` (``ridge_concat_*`` outputs).
    """
    if leg == "aux_only":
        return aux_states
    if leg == "aux_scalar":
        col = u_stream.to(dtype=aux_states.dtype, device=aux_states.device)
        return torch.cat([aux_states, col.unsqueeze(-1)], dim=1)
    if leg == "aux_concat":
        return aux_states
    raise ValueError(
        f"aux_drive_concat: leg must be one of {AUX_LEGS}, got {leg!r}"
    )


def parity_scale_drive(
    drive: torch.Tensor, u_stream: torch.Tensor, *, leg: str,
    input_scale: float,
) -> tuple[torch.Tensor, float]:
    """Scale an aux drive to the pinned Gate-1 drive power.

    The taps8 reference branch of ``flow_twin_drive`` normalizes its bank
    to drive RMS == ``input_scale``; aux legs must meet the same operating
    point or MC comparisons confound architecture with drive power
    (measured pre-fix: echo l1.0 RMS 0.76, SR g0.9 RMS 1.24, slow-feature
    RMS 0.82 vs taps8 RMS 0.20).

    ``aux_only`` and ``aux_concat``: scale the whole ``(T, n_aux)`` block
    to RMS == ``input_scale``.  ``aux_scalar``: scale the aux block the
    same way and set the scalar column to ``u * input_scale`` (mirroring
    the scalar reference branch exactly).  Returns ``(scaled_drive, scale)``
    where ``scale`` is the scalar applied to the aux block (1.0 means the
    block already sat at parity).  A zero-RMS (dead) block is returned
    unscaled with ``scale`` 0.0 so the downstream void KILL still fires
    instead of a divide-by-zero.
    """
    if leg not in AUX_LEGS:
        raise ValueError(
            f"parity_scale_drive: leg must be one of {AUX_LEGS}, "
            f"got {leg!r}"
        )
    if not (math.isfinite(float(input_scale)) and float(input_scale) > 0):
        raise ValueError(
            f"parity_scale_drive: input_scale must be positive finite, "
            f"got {input_scale}"
        )
    dev = drive.device
    dtp = drive.dtype
    if leg in ("aux_only", "aux_concat"):
        block = drive
    else:
        block = drive[:, :-1]
    rms = float(block.pow(2).mean().sqrt().item())
    if not math.isfinite(rms) or rms <= 0:
        return drive.to(device=dev, dtype=dtp), 0.0
    scale = float(input_scale) / rms
    if leg in ("aux_only", "aux_concat"):
        return (block * scale).to(device=dev, dtype=dtp), float(scale)
    u_col = u_stream.to(dtype=dtp, device=dev) * float(input_scale)
    return (
        torch.cat([block * scale, u_col.unsqueeze(-1)], dim=1),
        float(scale),
    )


# ---------------------------------------------------------------------------
# Itemized edge counts (device budget reading).
# ---------------------------------------------------------------------------


def echo_state_aux_edge_counts(n_aux: int) -> dict[str, int]:
    """Itemized edge counts for an echo-state aux core."""
    return {
        "aux_recurrent": n_aux * n_aux,
        "aux_fan_in": n_aux * 1,
    }


def shift_register_aux_edge_counts(n_aux: int) -> dict[str, int]:
    """Itemized edge counts for a shift-register aux core."""
    return {
        "aux_recurrent": n_aux - 1,  # chain
        "aux_fan_in": 1,  # head injection
    }


def slow_feature_aux_edge_counts(n_aux: int) -> dict[str, int]:
    """Itemized edge counts for a slow-feature aux core (decoupled)."""
    return {
        "aux_recurrent": 0,
        "aux_fan_in": n_aux * 1,
    }


AUX_EDGE_COUNTERS = {
    "echo_state": echo_state_aux_edge_counts,
    "shift_register": shift_register_aux_edge_counts,
    "slow_feature": slow_feature_aux_edge_counts,
    # D4 is an echo-state core under the hood; same itemized budget.
    D4_FLAVOR: echo_state_aux_edge_counts,
}


def main_fan_in_edges(n_main: int, n_drive: int) -> int:
    """Main core fan-in (W_main_in) edge count for a given drive width."""
    return int(n_main) * int(n_drive)


# ---------------------------------------------------------------------------
# Main core probe (eval-only, Heun flow-twin at Gate-1 best corner).
# ---------------------------------------------------------------------------


@dataclass
class AuxProbeRow:
    """Single aux-drive probe row (one flavor x leg x hyperparam x seed)."""

    config_tag: str
    flavor: str
    leg: str
    seed: int
    device: str
    hidden_dim: int
    n_aux: int
    n_drive: int
    aux_param: str
    aux_value: float
    aux_seed: int
    aux_recurrent: int
    aux_fan_in: int
    main_fan_in: int
    main_total_edges: int
    ref_total_edges: int
    drive_scale: float
    radius: float
    lamT: float
    input_scale: float
    dt: float
    num_steps: int
    washout: int
    max_delay: int
    mc_total: float
    mc_per_delay: list[float]
    ridge_nrmse: float
    ridge_r2: float
    ridge_concat_nrmse: float
    ridge_concat_r2: float
    tap_r2: float  # D1: copy-flag (taps -> aux block ridge R^2) per row
    state_pr: float
    activity: float
    activity_in_band: bool
    tanh_engaged: bool
    states_finite: bool
    pre_median: float
    aux_void_kill: bool
    # D2 sparse-fanout bookkeeping; defaults keep the D1/D3/D4 legs
    # working without these fields.
    fanout_kind: str = "dense"
    fanout_k: int = 0
    fanout_edges: int = 0
    # D4 residual-distilled teacher info (empty for non-D4 legs).
    d4_teacher_r2_mean: float = float("nan")
    d4_teacher_r2_max: float = float("nan")
    note: str = ""


def _aux_probe_instrument(
    *, drive: torch.Tensor, hidden_dim: int, n_drive: int,
    w_seed: int, win_seed: int, radius: float, lamT: float,
    dt: float, device: str,
    washout: int, max_delay: int, jacobian_samples: int,
    y_seq: torch.Tensor, u_scaled: torch.Tensor,
    aux_states: torch.Tensor | None = None,
    fanout_kind: str = "dense", fanout_k: int = 0,
) -> dict[str, Any]:
    """Run the Heun flow-twin at the Gate-1 best corner on an arbitrary
    ``(T, n_drive)`` drive signal and return the same instrument set as
    :func:`post_twin_gates.flow_twin_node_tanh`.

    When ``aux_states`` is provided, additionally compute a concat-readout
    ridge over ``[main_states; aux_states]`` and store the result in
    ``ridge_concat_nrmse`` / ``ridge_concat_r2``.  This is the D1
    concat readout path; the main-only ridge stays the primary metric.
    """
    if fanout_kind not in ("dense", "sparse_random"):
        raise ValueError(
            f"unknown fanout_kind {fanout_kind!r} "
            f"(must be 'dense' or 'sparse_random')"
        )
    dev = torch.device(device)
    dtype = torch.float32
    W, W_in = ptg.flow_twin_build_weights(
        n_nodes=hidden_dim, n_in=int(n_drive), w_seed=w_seed,
        win_seed=win_seed, target_radius=radius,
        dtype=dtype, device=dev,
    )
    if fanout_kind == "sparse_random":
        # D2 sparse main fan-out: Gate-3 sparse_random_W_in reuse, one
        # column per drive col with k nonzero entries each (exact nnz
        # asserted in the audit); radius restoration is carried by W
        # (unchanged); the rescale preserves total drive power.
        W_in = ptg.sparse_random_W_in(
            n_nodes=hidden_dim, n_in=int(n_drive),
            k_per_tap=int(fanout_k), seed=int(win_seed),
            dtype=dtype, device=dev,
        )
    lam = float(lamT) / float(CANONICAL_T_SPAN)
    states = ptg.flow_twin_run(
        W=W, W_in=W_in, drive=drive.to(dev),
        lam=lam, dt=dt,
    )
    pre_stats = ptg.flow_twin_pre_stats(
        W=W, W_in=W_in, states=states, drive=drive.to(dev),
        lam=lam, dt=dt,
    )
    tanh_engaged = bool(pre_stats["median_abs_pre"] >= PRE_LINEAR_REGIME_MEDIAN)
    mc_per, mc_total = ptg.nlc_safe_per_delay_mc(
        states, u_scaled.to(dev),
        washout=washout, max_delay=max_delay,
    )
    ridge_nrmse, ridge_r2 = ptg._ridge_nrmse_r2(
        states, y_seq.to(dev), washout=washout,
    )
    jac_rows, _eig_per = ptg.flow_twin_eigs_per_transition(
        W=W, W_in=W_in, states=states, drive=drive.to(dev),
        washout=washout, n_samples=jacobian_samples, lam=lam, dt=dt,
    )
    state_pr = ptg._safe_participation_ratio_local(states[washout:])
    activity = float(states[washout:].abs().mean().item())
    activity_in_band = bool(0.20 <= activity <= 0.80)
    states_finite = bool(torch.isfinite(states).all().item())
    result: dict[str, Any] = {
        "states": states.detach(),
        "mc_total": float(mc_total),
        "mc_per_delay": [float(v) for v in mc_per],
        "ridge_nrmse": float(ridge_nrmse),
        "ridge_r2": float(ridge_r2),
        "state_pr": float(state_pr),
        "activity": activity,
        "activity_in_band": activity_in_band,
        "tanh_engaged": tanh_engaged,
        "states_finite": states_finite,
        "pre_stats": pre_stats,
        "jac_rows": jac_rows,
        "ridge_concat_nrmse": float("nan"),
        "ridge_concat_r2": float("nan"),
        "tap_r2": float("nan"),
    }
    if aux_states is not None and aux_states.shape[0] == states.shape[0]:
        aux_dev = aux_states.to(device=dev, dtype=dtype)
        concat_states = torch.cat([states, aux_dev], dim=1)
        c_nrmse, c_r2 = ptg._ridge_nrmse_r2(
            concat_states, y_seq.to(dev), washout=washout,
        )
        result["ridge_concat_nrmse"] = float(c_nrmse)
        result["ridge_concat_r2"] = float(c_r2)
    return result


def _aux_probe_row(
    *, flavor: str, leg: str, seed: int, device: str,
    hidden_dim: int, n_taps: int,
    aux_param: str, aux_value: float, aux_seed: int,
    aux_spec: dict[str, Any],
    drive: torch.Tensor, drive_scale: float, u_scaled: torch.Tensor,
    radius: float, lamT: float, input_scale: float, dt: float,
    washout: int, max_delay: int, jacobian_samples: int,
    y_seq: torch.Tensor,
    fanout_kind: str = "dense", fanout_k: int = 0,
) -> AuxProbeRow:
    """Build a single AuxProbeRow from a finished aux drive and main probe.

    ``drive`` must already sit at drive-power parity (see
    :func:`parity_scale_drive`); ``drive_scale`` records the applied gain
    for provenance.  ``ref_total_edges`` is derived from the live
    ``hidden_dim``/``n_taps`` (``hidden_dim * n_taps``), not the
    canonical constant, so custom dims stay consistent.
    """
    n_aux = int(aux_spec["n_aux"])
    n_drive = int(drive.shape[1])
    aux_edges = AUX_EDGE_COUNTERS[flavor](n_aux)
    aux_rec = int(aux_edges["aux_recurrent"])
    aux_fan = int(aux_edges["aux_fan_in"])
    # D2: sparse fan-out replaces the dense main fan-in with k nnz per
    # drive column (radius-restored at build time, per spec).  Unknown
    # kinds raise here rather than silently running dense under a
    # sparse label.
    if fanout_kind == "dense":
        main_fan = main_fan_in_edges(hidden_dim, n_drive)
        fanout_edges = main_fan
    elif fanout_kind == "sparse_random":
        if not (1 <= int(fanout_k) <= hidden_dim):
            raise ValueError(
                f"fanout_k={fanout_k} must satisfy 1 <= k <= "
                f"hidden_dim={hidden_dim}"
            )
        main_fan = int(fanout_k) * n_drive
        fanout_edges = main_fan
    else:
        raise ValueError(
            f"unknown fanout_kind {fanout_kind!r} "
            f"(must be 'dense' or 'sparse_random')"
        )
    main_total = aux_rec + aux_fan + main_fan
    n_in_drive = int(drive.shape[1])
    inst = _aux_probe_instrument(
        drive=drive, hidden_dim=hidden_dim, n_drive=n_in_drive,
        w_seed=aux_seed, win_seed=aux_seed,
        radius=radius, lamT=lamT,
        dt=dt, device=device, washout=washout,
        max_delay=max_delay, jacobian_samples=jacobian_samples,
        y_seq=y_seq, u_scaled=u_scaled,
        aux_states=aux_spec.get("_aux_states_for_concat"),
        fanout_kind=fanout_kind, fanout_k=fanout_k,
    )
    # Per-row copy-flag (D1 amendment): taps -> aux-block ridge R^2 on
    # every leg, including aux_concat.  Uses the unscaled aux states
    # stashed on the spec by the sweep driver.
    aux_states_raw = aux_spec.get("_aux_states_for_concat")
    if aux_states_raw is not None:
        taps8_drive = ptg.flow_twin_drive(
            u_scaled, harness="taps8", n_taps=n_taps,
            input_scale=input_scale,
        )
        tap_r2 = tap_subspace_canonical_corr(
            aux_states_raw, taps8_drive, washout=washout,
        )
    else:
        tap_r2 = float("nan")
    # Verdict amendment: flavor ref budget = min(full taps8 bank,
    # compressed tapsVD4 bank) is decided at verdict time (the compressed
    # budget only applies if tapsVD4 matches taps8); the row reports the
    # live full-bank budget hidden_dim * n_taps.
    ref_total = main_fan_in_edges(hidden_dim, n_taps)
    pr = float(inst["state_pr"])
    mc_total = float(inst["mc_total"])
    aux_void = bool(
        math.isfinite(pr) and pr < VOID_PR_THRESHOLD
    ) or bool(
        math.isfinite(mc_total) and mc_total < VOID_MC_THRESHOLD
    )
    config_tag = (
        f"aux_{flavor}_{leg}_seed{seed}_{device}_h{hidden_dim}"
        f"_naux{n_aux}_nd{n_in_drive}_{aux_param}{aux_value:g}"
        f"_auxseed{aux_seed}_r{radius:g}_lamT{lamT:g}"
        f"_is{input_scale:g}_washout{washout}"
    )
    return AuxProbeRow(
        config_tag=config_tag,
        flavor=flavor, leg=leg,
        seed=seed, device=device,
        hidden_dim=hidden_dim, n_aux=n_aux, n_drive=n_in_drive,
        aux_param=aux_param, aux_value=float(aux_value),
        aux_seed=int(aux_seed),
        aux_recurrent=aux_rec, aux_fan_in=aux_fan,
        main_fan_in=main_fan,
        main_total_edges=main_total,
        ref_total_edges=ref_total,
        drive_scale=float(drive_scale),
        radius=radius, lamT=lamT, input_scale=input_scale,
        dt=float(dt),
        num_steps=int(round(CANONICAL_T_SPAN / dt)),
        washout=washout, max_delay=max_delay,
        mc_total=mc_total,
        mc_per_delay=inst["mc_per_delay"],
        ridge_nrmse=inst["ridge_nrmse"],
        ridge_r2=inst["ridge_r2"],
        ridge_concat_nrmse=inst["ridge_concat_nrmse"],
        ridge_concat_r2=inst["ridge_concat_r2"],
        tap_r2=tap_r2,
        fanout_kind=fanout_kind,
        fanout_k=int(fanout_k),
        fanout_edges=int(fanout_edges),
        state_pr=pr,
        activity=inst["activity"],
        activity_in_band=inst["activity_in_band"],
        tanh_engaged=inst["tanh_engaged"],
        states_finite=inst["states_finite"],
        pre_median=inst["pre_stats"]["median_abs_pre"],
        aux_void_kill=aux_void,
        note=(
            f"aux-drive flavor={flavor}, leg={leg}, "
            f"{aux_param}={aux_value:g}; aux_seed={aux_seed}; "
            f"edges (rec/fan/main/total)={aux_rec}/{aux_fan}/{main_fan}"
            f"/{main_total} vs ref={ref_total}; "
            f"pre_median={inst['pre_stats']['median_abs_pre']:.4f}, "
            f"engaged={inst['tanh_engaged']}"
        ),
    )


# ---------------------------------------------------------------------------
# Reference rows (taps8 + scalar + scalar ESN parity).
# ---------------------------------------------------------------------------


def _reference_row(
    *, ref_leg: str, seed: int, device: str, hidden_dim: int,
    n_taps: int,
    radius: float, lamT: float, input_scale: float, dt: float,
    washout: int, max_delay: int, jacobian_samples: int,
    u_scaled: torch.Tensor, y_seq: torch.Tensor,
) -> dict[str, Any]:
    """Run the Heun flow-twin reference leg (taps8 or scalar) at the Gate-1
    best corner and return a flat dict matching the aux-row schema.

    The harness is derived from ``ref_leg`` internally (a caller-supplied
    mismatch between leg label and drive construction would silently
    mislabel rows, so no separate harness parameter exists).

    For ``scalar_esn_parity``, build an in-harness ESN (no training of the
    main core; only the ridge readout is fit, mirroring the post-twin
    gates ESN parity diagnostic) and return its MC / ridge / state PR.
    """
    if ref_leg not in REF_LEGS:
        raise ValueError(
            f"reference leg must be one of {REF_LEGS}, got {ref_leg!r}"
        )
    if ref_leg == "taps8":
        drive = ptg.flow_twin_drive(
            u_scaled, harness="taps8", n_taps=n_taps,
            input_scale=input_scale,
        )
        inst = _aux_probe_instrument(
            drive=drive, hidden_dim=hidden_dim, n_drive=drive.shape[1],
            w_seed=0, win_seed=0, radius=radius, lamT=lamT,
            dt=dt, device=device,
            washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            y_seq=y_seq, u_scaled=u_scaled,
        )
        return {
            "mc_total": float(inst["mc_total"]),
            "ridge_nrmse": float(inst["ridge_nrmse"]),
            "ridge_r2": float(inst["ridge_r2"]),
            "state_pr": float(inst["state_pr"]),
            "activity": float(inst["activity"]),
            "tanh_engaged": bool(inst["tanh_engaged"]),
            "states_finite": bool(inst["states_finite"]),
            "states": inst["states"],
            "drive": drive,
        }
    if ref_leg == "scalar":
        drive = ptg.flow_twin_drive(
            u_scaled, harness="scalar", n_taps=n_taps,
            input_scale=input_scale,
        )
        inst = _aux_probe_instrument(
            drive=drive, hidden_dim=hidden_dim, n_drive=drive.shape[1],
            w_seed=0, win_seed=0, radius=radius, lamT=lamT,
            dt=dt, device=device,
            washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            y_seq=y_seq, u_scaled=u_scaled,
        )
        return {
            "mc_total": float(inst["mc_total"]),
            "ridge_nrmse": float(inst["ridge_nrmse"]),
            "ridge_r2": float(inst["ridge_r2"]),
            "state_pr": float(inst["state_pr"]),
            "activity": float(inst["activity"]),
            "tanh_engaged": bool(inst["tanh_engaged"]),
            "states_finite": bool(inst["states_finite"]),
            "states": inst["states"],
            "drive": drive,
        }
    if ref_leg == "tapsvd":
        # D3 compressed-bank reference: SVD top-``TAPSVD_K`` PCs of the
        # taps8 bank (deterministic, offline) drive the main core at
        # parity.  Reported as REFERENCE only (never in _aux_verdict).
        pc_bank, _mean, _comps = tapsvd_pc_bank(
            u_scaled, k=TAPSVD_K, n_taps=n_taps,
        )
        drive = pc_bank.to(device=u_scaled.device, dtype=torch.float32)
        rms = float(drive.pow(2).mean().sqrt().item())
        if rms > 0 and input_scale > 0:
            drive = drive * (float(input_scale) / rms)
        inst = _aux_probe_instrument(
            drive=drive, hidden_dim=hidden_dim, n_drive=drive.shape[1],
            w_seed=0, win_seed=0, radius=radius, lamT=lamT,
            dt=dt, device=device,
            washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            y_seq=y_seq, u_scaled=u_scaled,
        )
        return {
            "mc_total": float(inst["mc_total"]),
            "ridge_nrmse": float(inst["ridge_nrmse"]),
            "ridge_r2": float(inst["ridge_r2"]),
            "state_pr": float(inst["state_pr"]),
            "activity": float(inst["activity"]),
            "tanh_engaged": bool(inst["tanh_engaged"]),
            "states_finite": bool(inst["states_finite"]),
            "states": inst["states"],
            "drive": drive,
        }
    if ref_leg == "scalar_esn_parity":
        # ESN (CPU) on the same input, with no main-core interaction.
        esn = ne.ESN(
            n_reservoir=hidden_dim, spectral_radius=0.9,
            input_scaling=1.0, leak=1.0,
            ridge_l2=1e-2, seed=int(seed),
        )
        esn_in = (u_scaled * float(input_scale)).to("cpu")
        esn.fit(esn_in, y_seq.to("cpu"))
        esn_states = esn._run(esn_in)
        esn_states_dev = esn_states.to(torch.device(device))
        _, esn_mc = ptg.nlc_safe_per_delay_mc(
            esn_states_dev, esn_in.to(device),
            washout=washout, max_delay=max_delay,
        )
        esn_pred = (
            esn_states_dev @ esn.readout_W.to(device)
            + esn.readout_b.to(device)
        )
        esn_nrmse = float(ne.nrmse(
            esn_pred[washout:], y_seq.to(device)[washout:],
        ))
        esn_pr = ptg._safe_participation_ratio_local(esn_states_dev[washout:])
        return {
            "mc_total": float(esn_mc),
            "ridge_nrmse": esn_nrmse,
            "ridge_r2": float("nan"),
            "state_pr": float(esn_pr),
            "activity": float(esn_states_dev[washout:].abs().mean().item()),
            "tanh_engaged": True,  # ESN is by construction a tanh reservoir
            "states_finite": bool(torch.isfinite(esn_states).all().item()),
            "states": esn_states_dev,
            "drive": esn_in.to(device).unsqueeze(-1),
        }
    raise AssertionError(f"unreachable: ref_leg={ref_leg!r} passed validation")


# ---------------------------------------------------------------------------
# Tap-subspace KILL diagnostic (canonical correlation).
# ---------------------------------------------------------------------------


def tap_subspace_canonical_corr(
    aux_states: torch.Tensor, taps: torch.Tensor, *, washout: int,
    ridge_l2: float = 1e-2,
) -> float:
    """Tap-subspace projection diagnostic.

    The spec's "canonical-correlation ≈ delay-line copy = expensive taps"
    KILL signal is operationalized here as: the per-column ridge
    regression of the aux states on the taps delay bank.  If the taps
    can predict each aux column with R^2 close to 1, the aux states live
    in the tap subspace (the aux core is an expensive re-implementation
    of the static bank); if the taps can barely predict the aux states
    (R^2 near 0), the aux has unique information that the static bank
    cannot provide.

    Returns the MEAN across aux columns of the ridge R^2 (each column
    gets its own ridge readout, then we average).  A value >=
    :data:`TAP_SUBSPACE_KILL_CCORR` is a KILL signal.
    """
    a = aux_states[washout:].detach().to(dtype=torch.float64)
    t = taps[washout:].detach().to(dtype=torch.float64)
    if a.shape[0] != t.shape[0] or a.shape[0] < 4:
        return float("nan")
    if a.shape[1] < 1 or t.shape[1] < 1:
        return float("nan")
    n = a.shape[0]
    # NOTE: the bias column must live on the taps' device (a CUDA run
    # keeps ``t`` on GPU; a CPU-default ones() here used to crash the
    # concatenation under --device cuda).
    T_aug = torch.cat(
        [t, torch.ones(n, 1, dtype=torch.float64, device=t.device)], dim=1,
    )
    r2_per_col: list[float] = []
    for j in range(a.shape[1]):
        y = a[:, j]
        y_mean = float(y.mean().item())
        ss_tot = float(((y - y_mean) ** 2).sum().item())
        if not math.isfinite(ss_tot) or ss_tot <= 1e-12:
            continue
        XtX = T_aug.T @ T_aug + ridge_l2 * torch.eye(
            T_aug.shape[1], dtype=torch.float64, device=T_aug.device,
        )
        try:
            w = torch.linalg.solve(XtX, T_aug.T @ y)
        except Exception:
            return float("nan")
        pred = T_aug @ w
        ss_res = float(((y - pred) ** 2).sum().item())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        if math.isfinite(r2):
            r2_per_col.append(float(min(max(r2, 0.0), 1.0)))
    if not r2_per_col:
        return float("nan")
    return float(sum(r2_per_col) / len(r2_per_col))


# ---------------------------------------------------------------------------
# Sweep driver.
# ---------------------------------------------------------------------------


@dataclass
class AuxVerdict:
    """Pre-registered SUCCESS / KILL verdict for a single aux flavor."""

    flavor: str
    verdict: str  # "SUCCESS" | "KILL" | "INSUFFICIENT_DATA"
    aux_mc: float
    taps8_mc: float
    scalar_mc: float
    aux_void: bool
    tap_subspace_kill: bool
    no_gain_over_scalar: bool
    aux_matches_taps8: bool
    aux_beats_scalar: bool
    aux_edge_cost_factor: float
    reasons: list[str]
    note: str = ""


def _best_aux_row(
    rows: list[AuxProbeRow], *, flavor: str, leg: str,
    select_by: str = "mc_total",
) -> AuxProbeRow | None:
    """Pick the best engaged-and-finite aux row for the given leg.

    ``select_by="teacher"`` (D4 only) maximizes the mean ridge R^2 to
    the residual-distilled teacher instead of the NARMA MC — the D4
    aux's objective is distinct from main's by construction.
    """
    pool = [
        r for r in rows
        if r.flavor == flavor and r.leg == leg
        and r.tanh_engaged and r.states_finite
    ]
    if not pool:
        return None
    if select_by == "teacher":
        return max(pool, key=lambda r: (
            r.d4_teacher_r2_mean
            if math.isfinite(r.d4_teacher_r2_mean) else float("-inf")
        ))
    return max(pool, key=lambda r: r.mc_total)


def _aux_verdict(
    *, flavor: str, aux_rows: list[AuxProbeRow],
    taps8_mc: float, scalar_mc: float, scalar_esn_mc: float,
    aux_states_by_leg: dict[str, torch.Tensor],
    taps8_drive: torch.Tensor,
    washout: int = PROBE_WASHOUT,
    tapsvd_mc: float | None = None,
    tapsvd_edges: int | None = None,
) -> AuxVerdict:
    """Apply the pre-registered SUCCESS / KILL verdict for one flavor.

    Selection: the *best* engaged-and-finite row across aux-seeds and
    aux-hyperparam settings, per leg — over ALL legs present in
    ``aux_rows`` (aux_only, aux_scalar, aux_concat).  Ties on MC break
    in AUX_LEGS order so the winner is deterministic (aux_only and
    aux_concat share a drive, hence identical MC).  Verdict rules (per
    spec):

    - SUCCESS path 1: aux MC >= SUCCESS_MC_MATCH_RATIO * taps8 MC AND
      aux edge cost <= SUCCESS_EDGE_TOL_FACTOR * ref edge cost.
    - SUCCESS path 2: aux MC >= SUCCESS_BEAT_SCALAR_MARGIN * scalar MC
      (regardless of taps8).
    - KILL path 1 (tap subspace): max canonical correlation between aux
      states and taps8 delay bank >= TAP_SUBSPACE_KILL_CCORR.
    - KILL path 2 (void/dead): aux state PR below VOID_PR_THRESHOLD or
      aux MC below VOID_MC_THRESHOLD.
    - KILL path 3 (no gain over scalar): aux+scalar MC - scalar MC <
      NO_GAIN_OVER_SCALAR_THRESHOLD.

    D1 informational note (never a verdict path): when an engaged
    aux_concat row's [main; aux] readout beats its main-only readout
    by more than CONCAT_GAIN_NOTE_DELTA NRMSE, the gain is reported.
    """
    reasons: list[str] = []
    best_by_leg: dict[str, AuxProbeRow] = {}
    for leg in AUX_LEGS:
        best = _best_aux_row(aux_rows, flavor=flavor, leg=leg)
        if best is not None:
            best_by_leg[leg] = best
    best_aux = best_by_leg.get("aux_only")
    best_aux_s = best_by_leg.get("aux_scalar")
    if not best_by_leg:
        return AuxVerdict(
            flavor=flavor, verdict="INSUFFICIENT_DATA",
            aux_mc=float("nan"), taps8_mc=taps8_mc,
            scalar_mc=scalar_mc,
            aux_void=False, tap_subspace_kill=False,
            no_gain_over_scalar=False,
            aux_matches_taps8=False, aux_beats_scalar=False,
            aux_edge_cost_factor=float("nan"),
            reasons=["no aux row had engaged+finite states"],
            note=f"no aux row survived for flavor={flavor}",
        )
    # Winner: best MC across legs; ties break in AUX_LEGS order
    # (aux_only and aux_concat share a drive, hence identical MC —
    # the order makes the pick deterministic regardless of leg subset).
    _leg_rank = {leg: i for i, leg in enumerate(AUX_LEGS)}
    candidates = sorted(
        best_by_leg.values(),
        key=lambda r: (r.mc_total, -_leg_rank.get(r.leg, len(AUX_LEGS))),
        reverse=True,
    )
    winner = candidates[0]
    aux_mc = float(winner.mc_total)
    # D1 informational note: best concat-readout gain across engaged
    # concat rows (main-only NRMSE minus [main; aux] NRMSE).
    _concat_gains = [
        (r.ridge_nrmse - r.ridge_concat_nrmse)
        for r in aux_rows
        if r.flavor == flavor and r.leg == "aux_concat"
        and r.tanh_engaged and r.states_finite
        and math.isfinite(r.ridge_nrmse)
        and math.isfinite(r.ridge_concat_nrmse)
    ]
    if _concat_gains and max(_concat_gains) > CONCAT_GAIN_NOTE_DELTA:
        reasons.append(
            f"concat readout gain (info only): best [main; aux] readout "
            f"beats main-only by {max(_concat_gains):.4f} NRMSE "
            f"(>{CONCAT_GAIN_NOTE_DELTA:g})"
        )
    # D5 amendment: flavor ref budget = min over applicable refs.  The
    # compressed tapsVD4 bank (hidden_dim * TAPSVD_K edges) only counts
    # if it matched taps8 MC within the pre-registered ratio; otherwise
    # the full taps8 budget stands.
    _ref_budget = float(winner.ref_total_edges)
    if tapsvd_edges is not None and tapsvd_mc is not None \
            and math.isfinite(tapsvd_mc) and taps8_mc > 0 \
            and tapsvd_mc >= SUCCESS_MC_MATCH_RATIO * taps8_mc:
        _ref_budget = min(_ref_budget, float(tapsvd_edges))
    aux_edges_factor = (
        float(winner.main_total_edges) / _ref_budget
        if _ref_budget > 0 else float("nan")
    )
    # KILL: void / dead.
    aux_void = bool(winner.aux_void_kill)
    if aux_void:
        reasons.append(
            f"aux void/dead (PR={winner.state_pr:.3f}, MC={aux_mc:.3f})"
        )
    # KILL: tap subspace (canonical correlation).
    tap_kill = False
    cc_aux_only = float("nan")
    if "aux_only" in aux_states_by_leg and taps8_drive is not None:
        cc_aux_only = tap_subspace_canonical_corr(
            aux_states_by_leg["aux_only"], taps8_drive,
            washout=washout,
        )
        if math.isfinite(cc_aux_only) and cc_aux_only >= TAP_SUBSPACE_KILL_CCORR:
            tap_kill = True
            reasons.append(
                f"aux states project onto tap subspace "
                f"(canonical corr ~{cc_aux_only:.3f} >= "
                f"{TAP_SUBSPACE_KILL_CCORR:g})"
            )
    # KILL: no gain over scalar (aux+scalar vs scalar).
    no_gain_scalar = False
    if best_aux_s is not None and math.isfinite(scalar_mc):
        delta = best_aux_s.mc_total - scalar_mc
        if math.isfinite(delta) and delta < NO_GAIN_OVER_SCALAR_THRESHOLD:
            no_gain_scalar = True
            reasons.append(
                f"aux+scalar MC ({best_aux_s.mc_total:.3f}) - scalar "
                f"MC ({scalar_mc:.3f}) = {delta:.3f} below "
                f"{NO_GAIN_OVER_SCALAR_THRESHOLD:g}"
            )
    # SUCCESS path 1: matches taps8 within tolerance at <= edge cost.
    aux_matches = bool(
        math.isfinite(aux_mc)
        and math.isfinite(taps8_mc)
        and taps8_mc > 0
        and aux_mc >= SUCCESS_MC_MATCH_RATIO * taps8_mc
    )
    aux_at_cost = bool(aux_edges_factor <= SUCCESS_EDGE_TOL_FACTOR)
    success_path1 = bool(aux_matches and aux_at_cost)
    # SUCCESS path 2: beats scalar by margin >> cost.
    aux_beats_scalar = bool(
        math.isfinite(aux_mc)
        and math.isfinite(scalar_mc)
        and scalar_mc > 0
        and aux_mc >= SUCCESS_BEAT_SCALAR_MARGIN * scalar_mc
    )
    success_path2 = bool(aux_beats_scalar)
    if success_path1:
        reasons.append(
            f"SUCCESS path 1: aux MC {aux_mc:.3f} >= "
            f"{SUCCESS_MC_MATCH_RATIO:g} * taps8 MC {taps8_mc:.3f}; "
            f"edge cost factor {aux_edges_factor:.2f} <= "
            f"{SUCCESS_EDGE_TOL_FACTOR:g}"
        )
    elif success_path2:
        reasons.append(
            f"SUCCESS path 2: aux MC {aux_mc:.3f} >= "
            f"{SUCCESS_BEAT_SCALAR_MARGIN:g} * scalar MC {scalar_mc:.3f}"
        )
    if aux_void or tap_kill or no_gain_scalar:
        verdict = "KILL"
    elif success_path1 or success_path2:
        verdict = "SUCCESS"
    else:
        verdict = "KILL"
        reasons.append(
            "no SUCCESS path satisfied: did not match taps8 within "
            "tolerance at <= edge cost and did not beat scalar by "
            "the pre-registered margin"
        )
    return AuxVerdict(
        flavor=flavor, verdict=verdict,
        aux_mc=aux_mc, taps8_mc=taps8_mc,
        scalar_mc=scalar_mc,
        aux_void=aux_void, tap_subspace_kill=tap_kill,
        no_gain_over_scalar=no_gain_scalar,
        aux_matches_taps8=aux_matches,
        aux_beats_scalar=aux_beats_scalar,
        aux_edge_cost_factor=aux_edges_factor,
        reasons=reasons,
        note=(
            f"flavor={flavor}, best aux MC={aux_mc:.3f}, "
            f"taps8 MC={taps8_mc:.3f}, scalar MC={scalar_mc:.3f}, "
            f"esn parity MC={scalar_esn_mc:.3f}, "
            f"edge factor={aux_edges_factor:.2f}, "
            f"canonical corr={cc_aux_only:.3f}, "
            f"verdict={verdict}"
        ),
    )


def _jsonable(value: Any) -> Any:
    """Convert a value (possibly a tensor / NaN / inf) into JSON-safe form."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# CSV / JSON / TXT writers.
# ---------------------------------------------------------------------------


def _rows_to_csv(path: Path, rows: list[AuxProbeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonable(v) for k, v in asdict(r).items()})


def _verdict_to_json(path: Path, verdicts: list[AuxVerdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(
        [asdict(v) for v in verdicts]
    ), indent=2))


# ---------------------------------------------------------------------------
# Sweep driver (eval-only).
# ---------------------------------------------------------------------------


def _aux_hparam_grid(flavor: str) -> list[tuple[str, float]]:
    """Return the (param_name, value) grid for a flavor (per spec, small)."""
    if flavor == "echo_state":
        return [(f"leak{l:g}", l) for l in ECHO_LEAK_GRID]
    if flavor == "shift_register":
        return [(f"gain{g:g}", g) for g in SR_GAIN_GRID]
    if flavor == "slow_feature":
        # Single corner: full log-spread.  The spec says "log-spread tau"
        # so we sweep only the boundaries as aux-hparams.
        return [("logspread_full", 1.0)]
    if flavor == D4_FLAVOR:
        # D4 residual-distilled aux: echo-state runner; selection axis
        # is the same leak grid (per spec: grid over existing
        # leak/radius/seed axes only — no new optimizer).
        return [(f"leak{l:g}", l) for l in ECHO_LEAK_GRID]
    raise ValueError(f"unknown aux flavor: {flavor!r}")


def run_aux_sweep(
    *, flavors: tuple[str, ...] = AUX_FLAVORS,
    legs: tuple[str, ...] = AUX_LEGS,
    aux_seeds: tuple[int, ...] = AUX_SEEDS,
    data_seeds: tuple[int, ...] = DATA_SEEDS,
    hidden_dim: int = CANONICAL_HIDDEN,
    n_taps: int = 8,
    n_streams: int = 1,
    train_samples_per_stream: int = 300,
    washout: int = PROBE_WASHOUT,
    max_delay: int = 20,
    jacobian_samples: int = 3,
    radius: float = MAIN_RADIUS,
    lamT: float = MAIN_LAMT,
    input_scale: float = MAIN_INPUT_SCALE,
    device: str = "cpu",
    dt: float = CANONICAL_DT_MAX,
    max_corners: int | None = None,
    fanout_kinds: tuple[str, ...] = ("dense",),
) -> tuple[
    list[AuxProbeRow], list[dict[str, Any]], list[AuxVerdict], float,
]:
    """Run the aux-core drive sweep over flavors x legs x aux-hparams x
    aux-seeds x data-seeds.  Returns the aux rows, the reference rows
    (taps8 / scalar / scalar-ESN parity), the per-flavor verdicts, and
    elapsed seconds.

    ``max_corners`` caps the Cartesian grid BEFORE running (traversal
    prefix in flavor-major order); smoke caps save compute.
    """
    for f in flavors:
        if f not in ALL_FLAVORS:
            raise ValueError(
                f"run_aux_sweep: unknown flavor {f!r} "
                f"(must be a subset of {ALL_FLAVORS})"
            )
    for l in legs:
        if l not in AUX_LEGS:
            raise ValueError(
                f"run_aux_sweep: unknown leg {l!r} "
                f"(must be a subset of {AUX_LEGS})"
            )
    for fk in fanout_kinds:
        if fk not in ("dense", "sparse_random"):
            raise ValueError(
                f"run_aux_sweep: unknown fanout kind {fk!r} "
                f"(must be a subset of ('dense', 'sparse_random'))"
            )
    if not data_seeds:
        raise ValueError("run_aux_sweep: data_seeds must be non-empty")
    t0 = time.time()
    aux_rows: list[AuxProbeRow] = []
    refs: list[dict[str, Any]] = []
    verdicts: list[AuxVerdict] = []
    # Per-data-seed stream; cache the (u, y) per seed to avoid re-rolling.
    per_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for ds in data_seeds:
        u_raw, y_raw = ne._gen_narma_train_streams(
            order=10, seed=ds, n_streams=n_streams,
            n=train_samples_per_stream,
        )
        u_stream = u_raw[0].to(torch.device(device))
        y_stream = y_raw[0].to(torch.device(device))
        u_scaled = ne._scale_drive(
            u_stream, bipolar=True, order=10, input_scale=1.0,
        )
        per_seed[ds] = (u_scaled, y_stream)
    # Build the canonical reference rows once per (data_seed, ref_leg).
    ref_by_seed: dict[tuple[int, str], dict[str, Any]] = {}
    for ds in data_seeds:
        u_scaled, y_stream = per_seed[ds]
        for ref_leg in ("taps8", "scalar"):
            r = _reference_row(
                ref_leg=ref_leg, seed=ds, device=device,
                hidden_dim=hidden_dim, n_taps=n_taps,
                radius=radius, lamT=lamT,
                input_scale=input_scale, dt=dt,
                washout=washout, max_delay=max_delay,
                jacobian_samples=jacobian_samples,
                u_scaled=u_scaled, y_seq=y_stream,
            )
            r["config_tag"] = (
                f"ref_{ref_leg}_seed{ds}_{device}_h{hidden_dim}"
                f"_r{radius:g}_lamT{lamT:g}_is{input_scale:g}"
                f"_washout{washout}"
            )
            r["data_seed"] = ds
            ref_by_seed[(ds, ref_leg)] = r
            refs.append(r)
        # ESN parity (scalar input, ESN reservoir).
        esn_r = _reference_row(
            ref_leg="scalar_esn_parity", seed=ds, device=device,
            hidden_dim=hidden_dim, n_taps=n_taps,
            radius=radius, lamT=lamT, input_scale=input_scale,
            dt=dt, washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            u_scaled=u_scaled, y_seq=y_stream,
        )
        esn_r["config_tag"] = (
            f"ref_scalar_esn_parity_seed{ds}_{device}_h{hidden_dim}"
            f"_washout{washout}"
        )
        esn_r["data_seed"] = ds
        ref_by_seed[(ds, "scalar_esn_parity")] = esn_r
        refs.append(esn_r)
        # D3 compressed-bank reference (tapsVD4).
        tvd_r = _reference_row(
            ref_leg="tapsvd", seed=ds, device=device,
            hidden_dim=hidden_dim, n_taps=n_taps,
            radius=radius, lamT=lamT, input_scale=input_scale,
            dt=dt, washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples,
            u_scaled=u_scaled, y_seq=y_stream,
        )
        tvd_r["config_tag"] = (
            f"ref_tapsvd_k{TAPSVD_K}_seed{ds}_{device}_h{hidden_dim}"
            f"_washout{washout}"
        )
        tvd_r["data_seed"] = ds
        ref_by_seed[(ds, "tapsvd")] = tvd_r
        refs.append(tvd_r)
    # D4: build the residual teacher once per data-seed (offline Phase A,
    # ridge/SVD only).
    d4_teacher_by_seed: dict[int, dict[str, Any]] = {}
    if D4_FLAVOR in flavors:
        for ds in data_seeds:
            u_s, _y_s = per_seed[ds]
            d4_teacher_by_seed[ds] = residual_teacher_build(
                u_stream=u_s,
                hidden_dim=hidden_dim, n_taps=n_taps,
                input_scale=input_scale,
                teacher_k=D4_RESIDUAL_TEACHER_K,
                radius=radius, lamT=lamT, dt=dt,
            )
    # Aux sweep.
    corners: list[tuple[str, str, int, int, str, float, str, int]] = []
    for flavor in flavors:
        for leg in legs:
            for ds in data_seeds:
                for aux_seed in aux_seeds:
                    for param_name, param_value in _aux_hparam_grid(flavor):
                        # D2 sparse fan-out applies to all flavors (a
                        # drive-shape change, not a flavor); dense rows
                        # stay the primary comparison set.
                        if fanout_kinds == ("dense",):
                            fanout_opts = (("dense", 0),)
                        else:
                            fanout_opts = tuple(
                                (fk, k) for fk in fanout_kinds
                                for k in ((SPARSE_FANOUT_K if fk ==
                                           "sparse_random" else (0,)))
                            )
                        for fk, k in fanout_opts:
                            corners.append(
                                (flavor, leg, ds, aux_seed,
                                 param_name, param_value, fk, k),
                            )
    if max_corners is not None and max_corners > 0:
        corners = corners[: max_corners]
    for flavor, leg, ds, aux_seed, param_name, param_value, fk, k in corners:
        u_scaled, y_stream = per_seed[ds]
        # Build aux core.
        if flavor == D4_FLAVOR:
            # D4 uses the echo-state runner; hparam axis = leak.
            aux_spec = echo_state_aux_build(
                n_aux=ECHO_N_AUX, leak=param_value,
                radius=ECHO_RADIUS,
                w_seed=aux_seed, win_seed=aux_seed,
                dtype=torch.float32, device=torch.device(device),
            )
        elif flavor == "echo_state":
            aux_spec = echo_state_aux_build(
                n_aux=ECHO_N_AUX, leak=param_value,
                radius=ECHO_RADIUS,
                w_seed=aux_seed, win_seed=aux_seed,
                dtype=torch.float32, device=torch.device(device),
            )
        elif flavor == "shift_register":
            aux_spec = shift_register_aux_build(
                n_aux=SR_N_AUX, gain=param_value,
                head_inject=1.0, w_seed=aux_seed,
                dtype=torch.float32, device=torch.device(device),
            )
        elif flavor == "slow_feature":
            aux_spec = slow_feature_aux_build(
                n_aux=SFB_N_AUX,
                log_tau_lo=SFB_LOG_TAU_RANGE[0],
                log_tau_hi=SFB_LOG_TAU_RANGE[1],
                w_seed=aux_seed,
                dtype=torch.float32, device=torch.device(device),
            )
        else:
            raise ValueError(f"unknown aux flavor: {flavor!r}")
        # Run aux drive.
        aux_states = AUX_RUNNERS[flavor](
            spec=aux_spec, u_stream=u_scaled,
        )
        # Stash raw aux states on the spec for the D1 concat readout and
        # the per-row copy-flag (taps -> aux-block ridge R^2).
        aux_spec["_aux_states_for_concat"] = aux_states.detach()
        # Concatenate with scalar if needed, then scale to drive-power
        # parity (RMS == input_scale) so the main core sits at the
        # pinned Gate-1 operating point on every leg.
        drive_raw = aux_drive_concat(
            aux_states, u_scaled, leg=leg,
        )
        drive, drive_scale = parity_scale_drive(
            drive_raw, u_scaled, leg=leg, input_scale=input_scale,
        )
        # Run main core probe.
        row = _aux_probe_row(
            flavor=flavor, leg=leg, seed=ds, device=device,
            hidden_dim=hidden_dim, n_taps=n_taps,
            aux_param=param_name,
            aux_value=param_value, aux_seed=aux_seed,
            aux_spec=aux_spec,
            drive=drive, drive_scale=drive_scale, u_scaled=u_scaled,
            radius=radius, lamT=lamT, input_scale=input_scale,
            dt=dt, washout=washout, max_delay=max_delay,
            jacobian_samples=jacobian_samples, y_seq=y_stream,
            fanout_kind=fk, fanout_k=k,
        )
        if flavor == D4_FLAVOR:
            t_r2m, t_r2x = teacher_r2(
                aux_states, d4_teacher_by_seed[ds]["teacher"],
                washout=washout,
            )
            row.d4_teacher_r2_mean = t_r2m
            row.d4_teacher_r2_max = t_r2x
        aux_rows.append(row)
    # Per-flavor verdicts (use the *mean* across data-seeds of the taps8
    # / scalar / scalar-ESN MC for the comparison; the spec frames the
    # comparison as single-seed eval-only screening, but with multiple
    # data-seeds we use the per-seed best aux row and average the refs).
    aux_rows_by_seed: dict[int, list[AuxProbeRow]] = {}
    for r in aux_rows:
        aux_rows_by_seed.setdefault(r.seed, []).append(r)
    for flavor in flavors:
        # Per-seed verdicts averaged into one final flavor verdict.
        per_seed_verdicts: list[AuxVerdict] = []
        for ds in data_seeds:
            seed_rows = aux_rows_by_seed.get(ds, [])
            taps8_mc = float(ref_by_seed[(ds, "taps8")]["mc_total"])
            scalar_mc = float(ref_by_seed[(ds, "scalar")]["mc_total"])
            scalar_esn_mc = float(
                ref_by_seed[(ds, "scalar_esn_parity")]["mc_total"]
            )
            tapsvd_mc = float(ref_by_seed[(ds, "tapsvd")]["mc_total"])
            tapsvd_edges = main_fan_in_edges(hidden_dim, TAPSVD_K)
            taps8_drive = ref_by_seed[(ds, "taps8")]["drive"]
            # Track best aux states per (flavor, leg, seed) for KILL check.
            states_for_verdict: dict[str, torch.Tensor] = {}
            for leg in legs:
                best = _best_aux_row(
                    seed_rows, flavor=flavor, leg=leg,
                    select_by=("teacher" if flavor == D4_FLAVOR
                               else "mc_total"),
                )
                if best is not None:
                    # Re-run the aux drive for the best row's hparams to
                    # get its states (cheaper than caching all of them).
                    # D4 shares the echo-state construction (its runner
                    # entry in AUX_RUNNERS is the echo runner).
                    if flavor in ("echo_state", D4_FLAVOR):
                        sp = echo_state_aux_build(
                            n_aux=ECHO_N_AUX, leak=best.aux_value,
                            radius=ECHO_RADIUS,
                            w_seed=best.aux_seed, win_seed=best.aux_seed,
                            dtype=torch.float32,
                            device=torch.device(device),
                        )
                    elif flavor == "shift_register":
                        sp = shift_register_aux_build(
                            n_aux=SR_N_AUX, gain=best.aux_value,
                            head_inject=1.0, w_seed=best.aux_seed,
                            dtype=torch.float32,
                            device=torch.device(device),
                        )
                    else:
                        sp = slow_feature_aux_build(
                            n_aux=SFB_N_AUX,
                            log_tau_lo=SFB_LOG_TAU_RANGE[0],
                            log_tau_hi=SFB_LOG_TAU_RANGE[1],
                            w_seed=best.aux_seed,
                            dtype=torch.float32,
                            device=torch.device(device),
                        )
                    states_for_verdict[leg] = AUX_RUNNERS[flavor](
                        spec=sp, u_stream=per_seed[ds][0],
                    ).detach()
            v = _aux_verdict(
                flavor=flavor, aux_rows=seed_rows,
                taps8_mc=taps8_mc, scalar_mc=scalar_mc,
                scalar_esn_mc=scalar_esn_mc,
                aux_states_by_leg=states_for_verdict,
                taps8_drive=taps8_drive,
                washout=washout,
                tapsvd_mc=tapsvd_mc, tapsvd_edges=tapsvd_edges,
            )
            per_seed_verdicts.append(v)
        # Aggregate: SUCCESS wins over KILL if any data-seed says SUCCESS.
        # INSUFFICIENT_DATA only stands alone.
        if any(v.verdict == "SUCCESS" for v in per_seed_verdicts):
            agg = "SUCCESS"
            winner = next(
                v for v in per_seed_verdicts if v.verdict == "SUCCESS"
            )
        elif all(
            v.verdict == "INSUFFICIENT_DATA" for v in per_seed_verdicts
        ):
            agg = "INSUFFICIENT_DATA"
            winner = per_seed_verdicts[0]
        else:
            agg = "KILL"
            winner = per_seed_verdicts[0]
        agg_reasons: list[str] = [
            f"seed={ds}: {r}"
            for ds, v in zip(data_seeds, per_seed_verdicts)
            for r in v.reasons
        ]
        # Mean edge factor over seeds that produced one; NaN when no
        # seed did (INSUFFICIENT_DATA must not report a 0.0 factor).
        _finite_factors = [
            v.aux_edge_cost_factor for v in per_seed_verdicts
            if math.isfinite(v.aux_edge_cost_factor)
        ]
        _agg_factor = (
            float(sum(_finite_factors) / len(_finite_factors))
            if _finite_factors else float("nan")
        )
        agg_verdict = AuxVerdict(
            flavor=flavor, verdict=agg,
            aux_mc=winner.aux_mc,
            taps8_mc=float(
                sum(v.taps8_mc for v in per_seed_verdicts)
                / max(len(per_seed_verdicts), 1)
            ),
            scalar_mc=float(
                sum(v.scalar_mc for v in per_seed_verdicts)
                / max(len(per_seed_verdicts), 1)
            ),
            aux_void=any(v.aux_void for v in per_seed_verdicts),
            tap_subspace_kill=any(
                v.tap_subspace_kill for v in per_seed_verdicts
            ),
            no_gain_over_scalar=any(
                v.no_gain_over_scalar for v in per_seed_verdicts
            ),
            aux_matches_taps8=any(
                v.aux_matches_taps8 for v in per_seed_verdicts
            ),
            aux_beats_scalar=any(
                v.aux_beats_scalar for v in per_seed_verdicts
            ),
            aux_edge_cost_factor=_agg_factor,
            reasons=agg_reasons,
            note=(
                f"flavor={flavor}, aggregated verdict across "
                f"{len(per_seed_verdicts)} data-seed(s): {agg}"
            ),
        )
        verdicts.append(agg_verdict)
    elapsed = float(time.time() - t0)
    return aux_rows, refs, verdicts, elapsed


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--order", type=int, choices=[10, 20], default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--output", type=Path,
        default=Path("./output/aux_core_probe"),
    )
    p.add_argument("--hidden-dim", type=int, default=CANONICAL_HIDDEN)
    p.add_argument("--n-streams", type=int, default=1)
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--washout", type=int, default=PROBE_WASHOUT)
    p.add_argument("--max-delay", type=int, default=20)
    p.add_argument("--jacobian-samples", type=int, default=3)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--aux-seeds", type=int, nargs="+", default=list(AUX_SEEDS))
    p.add_argument(
        "--flavors", nargs="+", default=list(AUX_FLAVORS),
        help="Subset of aux flavors (echo_state, shift_register, "
             "slow_feature, residual_distilled).",
    )
    p.add_argument(
        "--legs", nargs="+", default=list(AUX_LEGS),
        help="Subset of aux legs (aux_only, aux_scalar, aux_concat).",
    )
    p.add_argument(
        "--fanout-kinds", nargs="+", default=["dense"],
        choices=["dense", "sparse_random"],
        help="D2 sparse main fan-out: add sparse_random to run the "
             "k-in-"
             f"{list(SPARSE_FANOUT_K)} fan-out legs alongside dense.",
    )
    p.add_argument(
        "--max-corners", type=int, default=None,
        help="Cap the aux Cartesian grid BEFORE running (traversal "
             "prefix; smoke runs stay cheap).",
    )
    p.add_argument(
        "--dt", type=float, default=CANONICAL_DT_MAX,
        help=f"Heun substep in samples (must satisfy 0 < dt <= "
             f"{CANONICAL_DT_MAX}; fixed-dt rule).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aux-core drive probe (plan aux-core-drive, "
                    "feature spec aux-core-probe).",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_sweep = sub.add_parser(
        "sweep", help="Run the aux-core sweep (eval-only).",
    )
    _add_common(p_sweep)

    p_smoke = sub.add_parser(
        "smoke", help="Lightweight smoke: single seed, single flavor, "
                       "tiny corners (no full verdict).",
    )
    _add_common(p_smoke)
    p_smoke.add_argument(
        "--flavor", default="echo_state", choices=list(ALL_FLAVORS),
    )

    args = parser.parse_args(argv)
    if args.order != 10:
        parser.error(
            "aux-core-probe decisions are pre-registered for --order 10 only"
        )
    if not (0.0 < float(args.dt) <= CANONICAL_DT_MAX):
        parser.error(
            f"--dt must satisfy 0 < dt <= {CANONICAL_DT_MAX} "
            f"(fixed-dt rule), got {args.dt}"
        )
    flavors = tuple(args.flavors)
    for f in flavors:
        if f not in ALL_FLAVORS:
            parser.error(f"--flavors must be a subset of {ALL_FLAVORS}")
    legs = tuple(args.legs)
    for l in legs:
        if l not in AUX_LEGS:
            parser.error(f"--legs must be a subset of {AUX_LEGS}")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        flavors = (args.flavor,)
        legs = ("aux_only",)
        max_corners = 1
        aux_seeds = (args.aux_seeds[0] if args.aux_seeds else 0,)
    else:
        max_corners = args.max_corners
        aux_seeds = tuple(args.aux_seeds)
    rows, refs, verdicts, elapsed = run_aux_sweep(
        flavors=flavors, legs=legs,
        aux_seeds=aux_seeds, data_seeds=(args.data_seed,),
        hidden_dim=args.hidden_dim, n_taps=8,
        n_streams=args.n_streams,
        train_samples_per_stream=args.train_samples,
        washout=args.washout, max_delay=args.max_delay,
        jacobian_samples=args.jacobian_samples,
        radius=MAIN_RADIUS, lamT=MAIN_LAMT,
        input_scale=MAIN_INPUT_SCALE,
        device=args.device, dt=float(args.dt),
        max_corners=max_corners,
        fanout_kinds=tuple(args.fanout_kinds),
    )
    _rows_to_csv(args.output / "aux_rows.csv", rows)
    refs_path = args.output / "aux_refs.json"
    refs_path.write_text(json.dumps(_jsonable(refs), indent=2))
    _verdict_to_json(args.output / "aux_verdicts.json", verdicts)
    summary_lines = [
        f"Aux-core probe -- flavors={flavors} legs={legs} "
        f"n_corners={len(rows)} n_refs={len(refs)} "
        f"elapsed={elapsed:.1f}s",
    ]
    for v in verdicts:
        summary_lines.append(
            f"  flavor={v.flavor} verdict={v.verdict} "
            f"aux_mc={v.aux_mc:.3f} taps8_mc={v.taps8_mc:.3f} "
            f"scalar_mc={v.scalar_mc:.3f} "
            f"edge_factor={v.aux_edge_cost_factor:.2f}"
        )
        for r in v.reasons:
            summary_lines.append(f"    - {r}")
    summary = "\n".join(summary_lines) + "\n"
    (args.output / "aux_summary.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
