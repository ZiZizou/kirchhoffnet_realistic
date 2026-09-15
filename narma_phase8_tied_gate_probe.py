"""E0 falsification probe for Phase-8 tied causal tap gates.

This is intentionally a frozen-core, ridge-readout experiment.  It compares
the identity gate with small random gate mixes, recalibrating only the single
band-preserving leak scale on an input-only stream so gate conditions are
activity-matched before their MC/PR/lag-product metrics are compared.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import narma_experiment as ne
from narma_phase7_leak_bands import (
    BAND_GROUPS, BAND_LEAKS, HIDDEN_DIM, N_TAPS, build_base, delay_bank,
    evaluate, set_leaks,
)
from tap_rails import TapRails


def _band_values() -> torch.Tensor:
    out = torch.empty(HIDDEN_DIM)
    for name, nodes in BAND_GROUPS.items():
        out[nodes] = BAND_LEAKS[name]
    return out


def _calibrate(net, taps, washout: int, target: float, rail_max: float,
               grid: list[float], device: str) -> dict:
    """Choose a one-dimensional leak scale from an input-only calibration."""
    base = _band_values()
    candidates = []
    for alpha in grid:
        trial = copy.deepcopy(net)
        set_leaks(trial.core.stages[0], alpha * base)
        states = ne.collect_fabric_states(trial, taps, device=device).cpu()[washout:, :HIDDEN_DIM]
        activity = float(states.abs().mean() / trial.core.stages[0].x_max)
        rails = float((states.abs() > .9 * trial.core.stages[0].x_max).float().mean())
        candidates.append({"alpha": alpha, "activity": activity, "rail_frac": rails,
                           "finite": bool(torch.isfinite(states).all())})
    ok = [r for r in candidates if r["finite"] and r["rail_frac"] <= rail_max]
    if not ok:
        raise RuntimeError("no finite, rail-safe calibration candidate")
    chosen = min(ok, key=lambda r: abs(r["activity"] - target))
    set_leaks(net.core.stages[0], chosen["alpha"] * base)
    return {"selected": chosen, "candidates": candidates}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--washout", type=int, default=200)
    p.add_argument("--core-seeds", default="0,1,2")
    p.add_argument("--B", type=int, choices=(2, 3, 4), default=2)
    p.add_argument("--w-scales", default="0.02,0.05,0.10")
    p.add_argument("--alpha-grid", default="0.40,0.50,0.60,0.70,0.80,0.90,1.0")
    p.add_argument("--activity-target", type=float, default=.49)
    p.add_argument("--rail-max", type=float, default=.20)
    p.add_argument("--n-local-jac", type=int, default=8)
    a = p.parse_args()
    if a.samples <= a.washout + 21:
        raise ValueError("samples must exceed washout by at least 21")
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.core_seeds.split(",") if x.strip()]
    scales = [float(x) for x in a.w_scales.split(",") if x.strip()]
    grid = [float(x) for x in a.alpha_grid.split(",") if x.strip()]
    cal_u, _ = ne.narma(a.samples, order=10, seed=77)
    cal_taps = delay_bank(ne._scale_drive(cal_u, bipolar=True, order=10, input_scale=1.0))
    u, y = ne.narma(a.samples, order=10, seed=0)
    taps = delay_bank(ne._scale_drive(u, bipolar=True, order=10, input_scale=1.0))
    rows = []
    for seed in seeds:
        for scale in [0.0, *scales]:
            net = build_base(seed).to(a.device)
            stage = net.core.stages[0]
            stage.tap_rails = TapRails(N_TAPS, HIDDEN_DIM, B=a.B).to(a.device)
            if scale:
                with torch.no_grad():
                    generator = torch.Generator(device="cpu").manual_seed(8000 + 100 * seed + round(scale * 1000))
                    stage.tap_rails.w.copy_(torch.randn(stage.tap_rails.w.shape, generator=generator).to(a.device) * scale)
            calibration = _calibrate(net, cal_taps, a.washout, a.activity_target,
                                     a.rail_max, grid, a.device)
            tag = "identity_gate" if scale == 0 else f"random_gate_w{scale:g}"
            row = evaluate(net, tag, seed, taps, u, y, a.washout, a.n_local_jac)
            with torch.no_grad():
                rail, z = stage.tap_rails.rails(taps.to(a.device)), stage.tap_rails.gate_and_multiplier(taps.to(a.device))[0]
            row.update({"B": a.B, "w_scale": scale, "calibration": calibration,
                        "rail_mean": [float(v) for v in rail.mean(0).cpu()],
                        "rail_var": [float(v) for v in rail.var(0, unbiased=False).cpu()],
                        "z_temporal_mean": [float(v) for v in z.mean(0).cpu()],
                        "z_temporal_std": [float(v) for v in z.std(0, unbiased=False).cpu()],
                        "z_near_endpoint_fraction": float(((z < .05) | (z > .95)).float().mean()),
                        "void_fraction_nodes": float((z.std(0, unbiased=False) < .02).float().mean())})
            rows.append(row)
            print("TIED_GATE_PROBE_RESULT " + json.dumps(row, sort_keys=True), flush=True)
    payload = {"experiment": "phase8_tied_tap_gate_random_existence_probe",
               "frozen_core": True, "ridge_reporting": True,
               "activity_matched_by": "input-only scalar band-leak calibration",
               "void_rule": "std_t(z_j) < 0.02", "rows": rows}
    (a.out / "tied_gate_probe.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
