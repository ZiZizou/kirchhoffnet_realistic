"""Single-shot spectral-radius normalization for the NARMA fabric.

Step 2 of the narma-node-activation plan: a one-time, init-only rescaling
that shifts every cell library's ``gm_raw`` so the per-sample Heun-map
Jacobian ``J = dHeun(x=0, u=0)/dx`` lands near ``target_sr`` (default 0.95,
edge of chaos), mirroring ESN spectral-radius practice.

Single pass by default; the caller may allow re-application (``max_passes``)
because the sigmoid parameterization compresses large shifts (``gm_raw=-5``
sits in the exponential tail, so one nominal x180 scale lands ~x70 in ``gm``).
Each pass is one linearize-plus-shift of the same hook; passes stop early
when ``|sr - target_sr| <= tol``. Two honest caveats are documented:

1. The sigmoid parameterization compresses large shifts (``gm_raw=-5`` sits
   in the exponential tail, so a nominal x180 scale lands ~x70 in ``gm``);
   ``sr_after`` is measured and reported, never assumed.
2. ``J`` is not linear in ``gm`` (tanh saturation, leak floor), so one shot
   may undershoot when the operating point is leak-pinned. The caller
   (Step-3 rung-1) reports ``sr_after`` alongside MC/PR and decides.

NARMA-side only (narma-cell-isolation spec): imported solely by
``narma_experiment.py`` and the node-activation audit. Never by
``train_script.py``, ``kn_bayes_opt.py``, or benchmarks. Side-effect-free
at import; mutates the net in place only when explicitly invoked.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


__all__ = ["normalize_spectral_radius"]


_RECURRENT_LIBS = ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib")


def _jacobian_at_zero(
    stage: nn.Module, *, t_span: float, num_steps: int, device: str,
) -> torch.Tensor:
    """Build ``J = dHeun(x=0, u=0)/dx`` for one stage by autograd.

    Mirrors ``narma_advisor_probes._one_sample_transition`` exactly so the
    Jacobian reflects the actually-used one-sample Heun map (frozen /
    periodic-refresh / VCA / node-activation paths included).
    """
    from narma_advisor_probes import _one_sample_transition

    n_nodes = int(stage.num_nodes)
    x0 = torch.zeros(n_nodes, device=device, dtype=torch.float32)
    u_next = torch.zeros(1, 1, device=device, dtype=torch.float32)
    dt = float(t_span) / float(num_steps)

    def transition_map(x_flat: torch.Tensor) -> torch.Tensor:
        return _one_sample_transition(stage, x_flat.view(1, -1), u_next, dt, int(num_steps))

    return torch.autograd.functional.jacobian(
        transition_map, x0.detach().clone().requires_grad_(True),
        create_graph=False,
    )


def _spectral_radius(J: torch.Tensor) -> float:
    """``max|eig(J)|`` via float32 eig, float64-CPU retry, else NaN."""
    try:
        vals = torch.linalg.eigvals(J).abs()
        if bool(torch.isfinite(vals).all()):
            return float(vals.max().item())
    except Exception:
        pass
    try:
        m = J.detach().to(dtype=torch.float64, device="cpu")
        if bool(torch.isfinite(m).all()):
            vals = torch.linalg.eigvals(m).abs()
            if bool(torch.isfinite(vals).all()):
                return float(vals.max().item())
    except Exception:
        pass
    return float("nan")


def _stage_device(stage: nn.Module) -> str:
    """Device holding the stage's parameters (for GPU-safe linearization)."""
    try:
        return str(next(stage.parameters()).device)
    except StopIteration:
        return "cpu"


def normalize_spectral_radius(
    net: nn.Module,
    *,
    target_sr: float = 0.95,
    t_span: float = 1.0,
    num_steps: int = 8,
    device: str = "auto",
    tol: float = 0.02,
    max_passes: int = 15,
) -> dict:
    """Shift ``gm_raw`` so each stage's Heun-map Jacobian hits ``target_sr``.

    Args:
        net: Built fabric net exposing ``net.core.stages`` (post
            ``build_net_from_config`` / ``make_narma_preset``). Each stage
            is scaled independently to the same ``target_sr``.
        target_sr: Desired per-sample ``|J|`` (default 0.95).
        t_span: Per-sample ODE window fallback. When the net exposes
            ``core.stage_times``/``core.stage_steps`` (always, post
            ``build_net_from_config``), each stage is linearized at its
            own split window instead.
        num_steps: Heun steps per sample fallback (same per-stage
            preference as ``t_span``).
        device: Linearization device. ``"auto"`` (default) uses each
            stage's parameter device, so a CUDA-built net linearizes on
            CUDA with no host/device mismatch; ``"cpu"`` forces host.
        tol: Stop re-applying when ``|sr - target_sr| <= tol``.
        max_passes: Maximum linearize-plus-shift passes per stage
            (default 1 = single shot). Passes are cumulative in
            ``gm_raw``; early-stop on ``tol``.

    Returns:
        Dict with ``target_sr`` plus one entry per stage:
        ``{"sr_before", "sr_after", "scale_applied" (cumulative product),
        "gm_shift" (cumulative log-shift actually added), "libs_touched",
        "passes", "note"}``. The fabric is mutated in place
        (readout/structural params untouched).
    """
    if not (0.0 < float(target_sr) < 2.0):
        raise ValueError(f"target_sr must be in (0, 2), got {target_sr}")
    if int(max_passes) < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")
    out: dict = {
        "target_sr": float(target_sr), "tol": float(tol),
        "max_passes": int(max_passes), "stages": [],
    }
    # Per-stage integration windows (multistage-sequence spec: the
    # per-sample window is SPLIT across stages). Prefer the net's own
    # ``stage_times``/``stage_steps`` when present so each stage is
    # linearized at its true resolution; fall back to the caller args
    # (exact for single-stage).
    _times = list(getattr(getattr(net, "core", None), "stage_times", []) or [])
    _steps = list(getattr(getattr(net, "core", None), "stage_steps", []) or [])
    for stage_idx, stage in enumerate(net.core.stages):
        st = float(_times[stage_idx]) if stage_idx < len(_times) else float(t_span)
        sn = int(_steps[stage_idx]) if stage_idx < len(_steps) else int(num_steps)
        dev = _stage_device(stage) if device == "auto" else str(device)
        row: dict = {
            "stage_index": int(stage_idx), "t_span": st, "num_steps": sn,
            "device": dev,
        }
        J = _jacobian_at_zero(stage, t_span=st, num_steps=sn, device=dev)
        sr_before = _spectral_radius(J)
        row["sr_before"] = float(sr_before)
        if not math.isfinite(sr_before) or sr_before <= 0:
            row.update(
                sr_after=float("nan"), scale_applied=1.0, gm_shift=0.0,
                libs_touched=0, passes=0,
                note="non-finite or non-positive sr_before; skipped",
            )
            warnings.warn(
                f"normalize_spectral_radius: stage {stage_idx} sr_before "
                f"non-finite ({sr_before}); leaving gm_raw untouched.",
                stacklevel=2,
            )
            out["stages"].append(row)
            continue
        cumulative_scale = 1.0
        cumulative_shift = 0.0
        touched = 0
        sr_now = float(sr_before)
        passes = 0
        for _ in range(int(max_passes)):
            if abs(sr_now - float(target_sr)) <= float(tol):
                break
            scale = float(target_sr) / float(sr_now)
            shift = float(math.log(scale))
            n = 0
            with torch.no_grad():
                for lib_name in _RECURRENT_LIBS:
                    lib = getattr(stage, lib_name, None)
                    if lib is not None and hasattr(lib, "gm_raw"):
                        lib.gm_raw.data.add_(shift)
                        n += 1
            if n == 0:
                break
            touched = n
            cumulative_scale *= scale
            cumulative_shift += shift
            passes += 1
            sr_now = _spectral_radius(
                _jacobian_at_zero(stage, t_span=st, num_steps=sn, device=dev)
            )
            if not math.isfinite(sr_now) or sr_now <= 0:
                break
        row["scale_applied"] = float(cumulative_scale)
        row["gm_shift"] = float(cumulative_shift)
        row["libs_touched"] = int(touched)
        row["passes"] = int(passes)
        row["sr_after"] = float(sr_now)
        note = ""
        if touched == 0 and passes == 0 and abs(float(sr_before) - float(target_sr)) > float(tol):
            note = "no gm_raw libraries found; nothing scaled"
        elif math.isfinite(sr_now) and abs(sr_now - float(target_sr)) > 0.1:
            note = (
                f"{passes} pass(es) landed at {sr_now:.3f} vs target "
                f"{float(target_sr):.2f} (sigmoid compression and/or "
                "leak-pinned operating point; raise max_passes)"
            )
        row["note"] = note
        out["stages"].append(row)
    return out
