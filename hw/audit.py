"""Post-training hardware audit of a trained HR-KNet checkpoint (Phase 3b, W4 step 4).

    python -m hw.audit --checkpoint PATH [--hw-config J] [--hw-preset P] [--n-rows 256]
                       [--canonical NPZ] [--sensitivity] [--json OUT] [--strict]

Loads the model through ``checkpoints.load_paper_checkpoint``: the paper
``LocalKirchhoffStudentWrapper`` layout, which is also what the Phase-A retrain
writes.  The HWConfig is rebuilt from the flags, or from the ``hw_config.json``
next to the checkpoint when no flags are given.  The audit runs ``N`` canonical
test rows and prints a pass/fail table:

* **slope**: ``gm_eff * max(A, B)`` per edge vs ``HWUnits.slope_max``.  The GLN
  modulation of the boundary and readout families is included; it is evaluated
  per row.
* **rail**: ``max |x|`` per stage over every Heun state (predictor
  included).  PASS ``<= x_max``, WARN ``<= 1.05 x_hard``, FAIL above that.
* **stiffness**: ``max(dt * rho(J))`` with ``J = d rhs / d x`` taken from the
  actual stage ``rhs`` (with ``freeze_read`` the frozen core currents are
  constants, so they correctly do not enter ``J``).  The amplification factor
  ``|1 + z + z^2/2|`` of Heun is reported as well.
* **isat**: ``Isat`` (cell) and ``Isat_eff = Isat * sigma(z) * g_vca`` vs
  ``[0.6, 16]`` (Phase 3a 2.5).
* **ratio**: ``max(A/B, B/A)`` vs ``R_MAX = 4``.

All edge checks use the *active* edges, ``sigma(z) >= 0.01``, the pruning
threshold; pruned edges carry no current.  ``--sensitivity`` adds the
time-multiplexed S&H probe: the output shift per mV of hold-cap droop versus
per mV of charge injection, in eval mode with the timemux model enabled.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from . import HWConfig, TimeMuxConfig, hw_config_from_args, hw_or_none, parse_hw_args  # noqa: E402
from .constraints import ISAT_HW_MAX, R_MAX, STIFFNESS_LIMIT, ab_mode  # noqa: E402
from .units import HWUnits  # noqa: E402

ISAT_HW_MIN = 0.6          # Phase 3a 2.5: internal-pole floor of the OTA bias
ACTIVE_GATE = 0.01         # pruning threshold of sigma(z)
DEFAULT_CANONICAL = os.path.join(
    os.path.expanduser("~"),
    "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz")
GOLDEN_PAPER = os.path.join(REPO, "tests", "golden", "data", "paper_trial_0011.pt")

FAMILIES = (  # name, cell-lib attr, gate-logit attr, VCA family, GLN family
    ("core", "cell_lib", "z_logits", "core", None),
    ("boundary", "boundary_cell_lib", "boundary_z_logits", "boundary", "boundary"),
    ("readout_sense", "readout_sense_cell_lib", "readout_sense_z_logits", "readout", "readout"),
    ("output_ode", "output_ode_cell_lib", "output_ode_z_logits", "readout", "readout"),
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_model(checkpoint: Optional[str], hw: Optional[HWConfig], dtype=torch.float32,
               arch: Optional[dict] = None):
    """The paper wrapper rebuilt with ``hw`` (None/off = legacy) and ``checkpoint`` loaded.

    ``checkpoint=None`` returns the freshly initialised (raw) model.  ``arch``
    overrides keys of ``checkpoints.PAPER_CONFIG`` (e.g. ``isat_max=16`` for a
    hardware retrain: the cell bounds are not stored in the state_dict).
    """
    import checkpoints as ck
    cfg = copy.deepcopy(ck.PAPER_CONFIG)
    cfg.update(arch or {})
    cfg["build"] = dict(cfg["build"], hw=hw_or_none(hw))
    if checkpoint is None:
        return ck.PaperStudent(cfg).to(dtype=dtype).eval()
    return ck.load_paper_checkpoint(checkpoint, dtype=dtype, config=cfg)


def canonical_rows(n: int, path: str = DEFAULT_CANONICAL) -> torch.Tensor:
    """The first ``n`` canonical Phase-A TEST rows (golden fallback: its 64 rows)."""
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        idx = np.asarray(z["test_idx"], dtype=np.int64)[:n]
        return torch.from_numpy(np.asarray(z["specs"][idx], dtype=np.float32)).clone()
    if os.path.isfile(GOLDEN_PAPER):
        g = torch.load(GOLDEN_PAPER, weights_only=False)
        return g["inputs"][:n].float().clone()
    raise FileNotFoundError(f"no canonical dataset at {path} and no golden fallback")


def resolve_hw(args, checkpoint: Optional[str]) -> tuple:
    """(HWConfig, source).  Flags win; else ``hw_config.json`` beside the checkpoint."""
    hw = hw_config_from_args(args)
    explicit = bool(getattr(args, "hw_config", None) or getattr(args, "hw_preset", None)) \
        or not hw.is_off()
    if explicit:
        return hw, "flags"
    if checkpoint:
        side = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "hw_config.json")
        if os.path.isfile(side):
            return HWConfig.from_json(side), side
    return hw, "default (off)"


# ---------------------------------------------------------------------------
# forward with capture of every rhs call
# ---------------------------------------------------------------------------
def _stages(model):
    return model.net.core.stages


def run_capture(model, x: torch.Tensor):
    """Forward ``x``; record ``(x, u, kwargs)`` of every ``rhs`` call per stage."""
    stages = _stages(model)
    recs: List[list] = [[] for _ in stages]
    for i, st in enumerate(stages):
        orig = type(st).rhs.__get__(st)

        def wrapped(xx, u=None, _i=i, _orig=orig, **kw):
            recs[_i].append((xx.detach(), None if u is None else u.detach(), kw))
            return _orig(xx, u=u, **kw)
        st.rhs = wrapped
    try:
        with torch.no_grad():
            y = model(x)
    finally:
        for st in stages:
            if "rhs" in st.__dict__:
                del st.rhs
    finals = [s.detach() for s in model.net.core.last_stage_outputs]
    return y.detach(), recs, finals


def _accumulator_nodes(st) -> List[int]:
    n = int(st.num_nodes)
    if getattr(st, "_has_shared_readout", False) and int(getattr(st, "_readout_dst_start", 0)) > 0:
        return list(range(int(st._readout_dst_start), n))
    if getattr(st, "output_ode_dst", None) is not None and st.output_ode_dst.numel() > 0:
        return sorted(set(int(v) for v in st.output_ode_dst.tolist()))
    return []


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _status(ok: bool, warn: bool = False) -> str:
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def edge_stats(model, recs, units: HWUnits) -> Dict[str, dict]:
    """Slope / Isat / ratio per family, aggregated over stages (active edges only)."""
    agg: Dict[str, dict] = {}
    for i, st in enumerate(_stages(model)):
        u = recs[i][0][1] if recs[i] else None
        for fam, lib_attr, z_attr, vca_fam, gln_fam in FAMILIES:
            lib = getattr(st, lib_attr, None)
            z = getattr(st, z_attr, None)
            if lib is None or z is None or z.numel() == 0 or not hasattr(lib, "ab_values"):
                continue
            with torch.no_grad():
                A, B = lib.ab_values()
                gate = torch.sigmoid(z)
                active = gate >= ACTIVE_GATE
                gm = lib.gm_eff_values().unsqueeze(0)                     # [1, E]
                if gln_fam is not None and u is not None:
                    gz = st._gln_rails_z(u)
                    over = st.edge_gm_scale(gln_fam, lib, gz)
                    if isinstance(over, torch.Tensor):
                        gm = lib._gm_hw(lib.hw_constraints, over) if (
                            lib.hw_constraints is not None and (
                                lib.hw_constraints.hw_slope_bound
                                or lib.hw_constraints.hw_ic_coupling)) else over
                slope = gm * torch.maximum(A, B).unsqueeze(0)              # [B|1, E]
                isat = lib.isat_values()
                g_vca = st.edge_gain(vca_fam, u) if u is not None else None
                isat_eff = (isat * gate).unsqueeze(0)
                if g_vca is not None and g_vca.shape[-1] == isat.shape[-1]:
                    isat_eff = isat_eff * g_vca
                ratio = torch.maximum(A / B, B / A)
            a = agg.setdefault(fam, {"n_edges": 0, "n_active": 0, "slope": [], "isat": [],
                                     "isat_eff": [], "ratio": [], "gain": [], "stages": []})
            a["n_edges"] += int(gate.numel())
            a["n_active"] += int(active.sum())
            a["stages"].append(i)
            if active.any():
                a["slope"].append(slope[:, active].reshape(-1))
                a["isat"].append(isat[active])
                a["isat_eff"].append(isat_eff[:, active].reshape(-1))
                a["ratio"].append(ratio[active])
                a["gain"].append(torch.cat([A[active], B[active]]))
    out = {}
    for fam, a in agg.items():
        r = {"n_edges": a["n_edges"], "n_active": a["n_active"], "stages": a["stages"]}
        for k in ("slope", "isat", "isat_eff", "ratio", "gain"):
            if a[k]:
                v = torch.cat(a[k]).double()
                r[k] = (float(v.min()), float(v.max()))
                if k == "slope":
                    r["slope_frac_over"] = float((v > units.slope_max).double().mean())
                if k in ("isat", "isat_eff"):
                    r[k + "_frac_out"] = float(((v < ISAT_HW_MIN) | (v > ISAT_HW_MAX)).double().mean())
                if k == "ratio":
                    r["ratio_frac_over"] = float((v > R_MAX).double().mean())
        out[fam] = r
    return out


def rail_stats(model, recs, finals, units: HWUnits) -> List[dict]:
    out = []
    for i, st in enumerate(_stages(model)):
        xs = [r[0] for r in recs[i]] + [finals[i]]
        X = torch.stack(xs, 0).abs()                                      # [S, B, N]
        acc = _accumulator_nodes(st)
        hid = [j for j in range(int(st.num_nodes)) if j not in set(acc)]
        m_h = float(X[..., hid].max()) if hid else float("nan")
        m_a = float(X[..., acc].max()) if acc else float("nan")
        m = max(v for v in (m_h, m_a) if v == v)
        # stage-entry state = the interstage transfer output (written onto the node)
        m_entry = float(X[0].max())
        frac = float((X.amax(0) > units.x_max).any(-1).double().mean())
        out.append({"stage": i, "max_abs_hidden": m_h, "max_abs_acc": m_a, "max_abs": m,
                    "max_abs_entry": m_entry, "frac_rows_over_xmax": frac,
                    "status": _status(m <= units.x_max, m <= 1.05 * units.x_hard)})
    return out


def _rows(v, n_rows: int, batch: int):
    """The first ``n_rows`` batch rows of an rhs kwarg (tensor or 3.4 ``FrozenLegs``)."""
    if isinstance(v, torch.Tensor):
        return v[:n_rows] if v.dim() > 0 and v.shape[0] == batch else v
    if hasattr(v, "i_pos") and hasattr(v, "i_neg"):          # differential_stage.FrozenLegs
        return type(v)(_rows(v.i_pos, n_rows, batch), _rows(v.i_neg, n_rows, batch))
    return v


def stiffness_stats(model, recs, n_rows: int = 64) -> List[dict]:
    """``max(dt * rho(J))`` per stage with the exact Jacobian of the stage ``rhs``."""
    core = model.net.core
    out = []
    for i, st in enumerate(_stages(model)):
        dt = float(core.stage_times[i]) / float(core.stage_steps[i])
        rho_max, amp_max, re_min, re_max = 0.0, 0.0, 0.0, 0.0
        for (x, u, kw) in recs[i]:
            xb = x[:n_rows].clone().requires_grad_(True)
            ub = None if u is None else u[:n_rows]
            kwb = {k: _rows(v, n_rows, x.shape[0]) for k, v in kw.items()}
            with torch.enable_grad():
                f = type(st).rhs(st, xb, u=ub, **kwb)
                N = f.shape[1]
                rows = [torch.autograd.grad(f[:, k].sum(), xb, retain_graph=(k < N - 1))[0]
                        for k in range(N)]
            J = torch.stack(rows, 1).double()                             # [B, N, N]
            lam = torch.linalg.eigvals(J)                                  # [B, N]
            z = dt * lam
            rho_max = max(rho_max, float(z.abs().max()))
            amp_max = max(amp_max, float((1 + z + z * z / 2).abs().max()))
            re_min = min(re_min, float(z.real.min()))
            re_max = max(re_max, float(z.real.max()))
        # |R| > 1 with dt*rho << 2 comes from eigenvalues with Re > 0: modes the
        # ODE itself grows (positive feedback), not an integrator instability.
        out.append({"stage": i, "dt": dt, "max_dt_rho": rho_max, "min_dt_re_lambda": re_min,
                    "max_dt_re_lambda": re_max,
                    "max_heun_amp": amp_max, "status": _status(rho_max <= STIFFNESS_LIMIT)})
    return out


# ---------------------------------------------------------------------------
# time-mux S&H sensitivity
# ---------------------------------------------------------------------------
def _tm_hw(base: Optional[HWConfig], **tm) -> HWConfig:
    d = (base or HWConfig()).to_dict()
    d["timemux"] = dict(TimeMuxConfig(enable=True, inject_mv=0.0, droop_mv=0.0,
                                      ktc=False).to_dict(), **tm)
    return HWConfig.from_dict(d)


def core_read_gain(model, recs, n_rows: int = 64) -> List[dict]:
    """Per stage: ``T_stage * ||d i_edge_const / d x_core||`` (the droop leverage).

    A droop ``delta`` on the hold cap shifts only the frozen core current, by
    ``J_core delta``; integrated over the stage that is a state shift of about
    ``T * J_core * delta``, versus about ``delta`` for an injection, which
    moves the node itself.
    """
    core = model.net.core
    out = []
    for i, st in enumerate(_stages(model)):
        if not recs[i] or not getattr(st, "freeze_read", False):
            out.append({"stage": i, "T_Jcore_rowsum": float("nan"), "T_Jcore_spec": float("nan")})
            continue
        x0, u, _ = recs[i][0]
        xb = x0[:n_rows].clone().requires_grad_(True)
        ub = None if u is None else u[:n_rows]
        with torch.enable_grad():
            g = st.edge_gain("core", ub)
            f = st._compute_i_edge_const(xb, g)
            N = f.shape[1]
            rows = [torch.autograd.grad(f[:, k].sum(), xb, retain_graph=(k < N - 1))[0]
                    for k in range(N)]
        J = torch.stack(rows, 1).double()
        T = float(core.stage_times[i])
        out.append({"stage": i,
                    "T_Jcore_rowsum": float(T * J.abs().sum(-1).mean()),
                    "T_Jcore_spec": float(T * torch.linalg.matrix_norm(J, ord=2).mean())})
    return out


def sensitivity(checkpoint: Optional[str], x: torch.Tensor, base: Optional[HWConfig] = None,
                mv: float = 1.0, seed: int = 0, arch: Optional[dict] = None) -> dict:
    """Mean |d logits| per mV of droop and per mV of injection, eval mode."""
    torch.manual_seed(seed)
    legacy = load_model(checkpoint, base, arch=arch)
    with torch.no_grad():
        y_leg = legacy(x)
    res = {"mv": mv}

    def run(**tm):
        torch.manual_seed(seed)          # the raw-init model must be the same draw
        m = load_model(checkpoint, _tm_hw(base, **tm), arch=arch)
        with torch.no_grad():
            y = m(x)
        return y, [s.detach() for s in m.net.core.last_stage_outputs], m
    y0, s0, m0 = run()
    res["identity_max_abs"] = float((y0 - y_leg).abs().max())
    for name, kw in (("inject", {"inject_mv": mv}), ("droop", {"droop_mv": mv}),
                     ("inject_x10", {"inject_mv": 10 * mv}), ("droop_x10", {"droop_mv": 10 * mv})):
        y, s, _ = run(**kw)
        scale = (10 * mv) if name.endswith("x10") else mv
        res[name] = {
            "dy_per_mv": float((y - y0).abs().mean()) / scale,
            "dy_rel_per_mv": float((y - y0).norm() / y0.norm().clamp_min(1e-30)) / scale,
            "stage_dx_per_mv": [float((a - b).abs().mean()) / scale for a, b in zip(s, s0)],
        }
    _, recs, _ = run_capture(m0, x)
    res["core_read_gain"] = core_read_gain(m0, recs)
    res["droop_over_inject"] = res["droop"]["dy_per_mv"] / max(res["inject"]["dy_per_mv"], 1e-300)
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def audit(checkpoint: str, hw: HWConfig, x: torch.Tensor, stiff_rows: int = 64,
          arch: Optional[dict] = None) -> dict:
    model = load_model(checkpoint, hw, arch=arch)
    units = hw.units
    y, recs, finals = run_capture(model, x)
    edges = edge_stats(model, recs, units)
    rails = rail_stats(model, recs, finals, units)
    stiff = stiffness_stats(model, recs, stiff_rows)
    rows = []
    for fam, r in edges.items():
        if "slope" in r:
            rows.append(("slope gm*max(A,B)", fam, r["slope"], f"<= {units.slope_max:.3f}",
                         f"{100 * r['slope_frac_over']:.1f}% over",
                         _status(r["slope"][1] <= units.slope_max)))
    for r in rails:
        rows.append(("rail max|x|", f"stage {r['stage']}",
                     f"hid {r['max_abs_hidden']:.3g} / acc {r['max_abs_acc']:.3g}",
                     f"x_max {units.x_max:g} / 1.05 x_hard {1.05 * units.x_hard:.2f}",
                     f"entry {r['max_abs_entry']:.3g}, {100 * r['frac_rows_over_xmax']:.1f}% rows > x_max",
                     r["status"]))
    for r in stiff:
        rows.append(("Heun max(dt*rho)", f"stage {r['stage']}", f"{r['max_dt_rho']:.3g}",
                     f"<= {STIFFNESS_LIMIT:g}",
                     f"Re(dt*lam) {r['min_dt_re_lambda']:.3g}..{r['max_dt_re_lambda']:.3g}, "
                     f"|R|max {r['max_heun_amp']:.3f}", r["status"]))
    for fam, r in edges.items():
        for k, lab in (("isat", "Isat"), ("isat_eff", "Isat_eff")):
            if k in r:
                ok = ISAT_HW_MIN <= r[k][0] and r[k][1] <= ISAT_HW_MAX
                rows.append((f"{lab} range", fam, r[k], f"[{ISAT_HW_MIN:g}, {ISAT_HW_MAX:g}]",
                             f"{100 * r[k + '_frac_out']:.1f}% out", _status(ok)))
    mode = ab_mode(hw.constraints)
    for fam, r in edges.items():
        if "ratio" in r:
            if mode == "cap_divider":
                lo = 1.0 / hw.constraints.hw_ab_rmax
                rows.append(("A/B cap-divider gains", fam, r["gain"], f"[{lo:g}, 1]",
                             "STE capacitor DAC", _status(r["gain"][0] >= lo and r["gain"][1] <= 1.0)))
            elif mode == "differential":
                rows.append(("A/B differential unity", fam, r["gain"], "A = B = 1",
                             "no per-edge A/B", _status(r["gain"] == (1.0, 1.0))))
            else:
                rows.append(("A/B ratio max(A/B,B/A)", fam, r["ratio"], f"<= {R_MAX:g}",
                             f"{100 * r['ratio_frac_over']:.1f}% over", _status(r["ratio"][1] <= R_MAX)))
    return {"rows": rows, "edges": edges, "rails": rails, "stiffness": stiff,
            "n_rows": int(x.shape[0]), "hw_hash": hw.hash(), "hw_off": hw.is_off(),
            "y_finite": bool(torch.isfinite(y).all())}


def format_table(rows) -> str:
    def rng(v):
        if isinstance(v, str):
            return v
        a, b = v
        f = lambda t: "nan" if t != t else f"{t:.4g}"  # noqa: E731
        return f"{f(a)} .. {f(b)}" if not isinstance(a, str) else f"{a} .. {b}"
    hdr = ("check", "where", "range", "limit", "note", "status")
    body = [(r[0], r[1], rng(r[2]), r[3], r[4], r[5]) for r in rows]
    w = [max(len(str(c)) for c in col) for col in zip(hdr, *body)]
    line = lambda c: "| " + " | ".join(str(v).ljust(n) for v, n in zip(c, w)) + " |"  # noqa: E731
    return "\n".join([line(hdr), "|" + "|".join("-" * (n + 2) for n in w) + "|"] + [line(c) for c in body])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hw.audit", description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-rows", type=int, default=256)
    ap.add_argument("--canonical", default=DEFAULT_CANONICAL)
    ap.add_argument("--stiffness-rows", type=int, default=64)
    ap.add_argument("--sensitivity", action="store_true",
                    help="add the time-mux droop-vs-injection probe (eval mode)")
    ap.add_argument("--sens-mv", type=float, default=1.0)
    ap.add_argument("--sens-raw-init", action="store_true",
                    help="also run the probe on the untrained (raw-init) model for contrast")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 when any check FAILs")
    ap.add_argument("--kn-isat-max", type=float, default=None,
                    help="cell Isat upper bound the model was trained with (paper 28.69; the "
                         "hardware retrain uses 16).  Not stored in the checkpoint.")
    ap.add_argument("--kn-gm-max", type=float, default=None,
                    help="cell gm upper bound the model was trained with (paper 4.24197)")
    parse_hw_args(ap)
    a = ap.parse_args(argv)
    hw, src = resolve_hw(a, a.checkpoint)
    arch = {}
    if a.kn_isat_max is not None:
        arch["isat_max"] = float(a.kn_isat_max)
    if a.kn_gm_max is not None:
        arch["gm_max"] = float(a.kn_gm_max)
    x = canonical_rows(a.n_rows, a.canonical)
    print(f"[audit] checkpoint={a.checkpoint}")
    print(f"[audit] HWConfig from {src}: {'OFF' if hw.is_off() else 'ON'} hash={hw.hash()}")
    res = audit(a.checkpoint, hw, x, a.stiffness_rows, arch=arch)
    res["arch_overrides"] = arch
    print(f"[audit] rows={res['n_rows']} (canonical test split)  outputs finite={res['y_finite']}")
    print(format_table(res["rows"]))
    n_fail = sum(r[5] == "FAIL" for r in res["rows"])
    print(f"[audit] {n_fail} FAIL, {sum(r[5] == 'WARN' for r in res['rows'])} WARN, "
          f"{sum(r[5] == 'PASS' for r in res['rows'])} PASS")
    if a.sensitivity:
        s = sensitivity(a.checkpoint, x, hw, a.sens_mv, arch=arch)
        res["sensitivity"] = s
        print(f"[audit] time-mux S&H sensitivity (eval, kT/C off), {a.sens_mv:g} mV:")
        print(f"  identity (tm on, 0 mV) vs legacy: max|dy| = {s['identity_max_abs']:.3g}")
        for k in ("inject", "droop", "inject_x10", "droop_x10"):
            print(f"  {k:10s}: mean|dy| per mV = {s[k]['dy_per_mv']:.4g}  "
                  f"(rel {s[k]['dy_rel_per_mv']:.3g}/mV)  per-stage mean|dx|/mV = "
                  + " ".join(f"{v:.3g}" for v in s[k]["stage_dx_per_mv"]))
        print(f"  droop / inject = {s['droop_over_inject']:.4g}")
        print("  core-read leverage T*|J_core| (row-sum mean) per stage = "
              + " ".join(f"{r['T_Jcore_rowsum']:.3g}" for r in s["core_read_gain"]))
        if a.sens_raw_init:
            r0 = sensitivity(None, x, hw, a.sens_mv, arch=arch)
            res["sensitivity_raw_init"] = r0
            print(f"  raw-init model: inject {r0['inject']['dy_per_mv']:.3g}/mV, droop "
                  f"{r0['droop']['dy_per_mv']:.3g}/mV, droop/inject {r0['droop_over_inject']:.3g}; "
                  "T*|J_core| = " + " ".join(f"{r['T_Jcore_rowsum']:.3g}" for r in r0["core_read_gain"]))
    if a.json:
        def _j(o):
            if isinstance(o, tuple):
                return list(o)
            raise TypeError(type(o))
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=2, default=_j)
    return 1 if (a.strict and n_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
