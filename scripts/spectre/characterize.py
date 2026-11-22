#!/usr/bin/env python3
"""Cell-level Spectre characterization against the analytic/PyTorch contract.

All Cadence-generated files are placed in .agent_scratch/taskD2/runs.  The
measured cases cover OTA, shunt, transfer, VCA, and GLN DC primitives; modules
not measured are explicitly reported as uncharacterized rather than silently
being treated as passing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD2/runs")
SETUP = "~/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
VU, VCM, IU = 0.07, 0.4, 50e-15 * 0.07 / (350e-9 / 4.660583)
VA = ROOT / "veriloga"


def cases():
    out = []
    for s in (-1.0, 1.0):
        for x in (-6.0, -4.28, -2.0, -0.1, 0.1, 2.0, 4.28, 6.0):
            out.append({"module": "ota_edge", "mode": "legacy", "xs": x,
                        "xd": 0.17, "params": {"mode": 0, "A": 0.8, "B": 1.2,
                        "s": s, "theta": 0.15, "gm": 1.3, "Isat": 0.7,
                        "x_max": 4.28, "gate_width": 0.02}})
    for ab, ca, cb in ((2, 7, 7), (3, 2, 6), (3, 5, 1)):
        for s in (-1.0, 1.0):
            out.append({"module": "ota_edge", "mode": f"hw_ab{ab}_codes{ca}_{cb}",
                        "xs": 1.3, "xd": -0.6,
                        "params": {"mode": 1, "ab_mode": ab, "cap_code_a": ca,
                        "cap_code_b": cb, "cap_bits": 3, "cap_rmax": 2.0,
                        "s": s, "theta": 0.08, "gm": 1.2, "Isat": 0.8,
                        "g_hw": 0.7, "g_min": 0.0625, "f_ic": 0.9,
                        "lambda_v": 0.01}})
    for xs, xd in ((-2.0, 0.3), (0.0, 0.0), (1.7, -0.8)):
        out.append({"module": "shunt", "mode": "dc", "xs": xs, "xd": xd,
                    "params": {"G": 0.75, "z_gate": 0.8, "vca": 1.1,
                               "vca_gate_shunt": 1}})
    for vector in ((0.1, -0.2, 0.3, -0.4), (-1.0, 0.5, 0.25, 0.75)):
        weights = (0.3, -0.2, 0.15, 0.1)
        out.append({"module": "vca_bias", "mode": "input_vector", "u": vector,
                    "weights": weights, "bias": 0.05})
        out.append({"module": "gln_bias", "mode": "tail_trim", "u": vector,
                    "weights": weights, "bias": 0.1, "delta_max": math.log(4.0)})
    for x in (-2.0, 0.2, 1.5, 6.0):
        out.append({"module": "interstage_transfer", "mode": "hw_clamp", "x": x,
                    "params": {"w1": 0.8, "w2": 0.4, "w3": 0.2, "vth": 0.1,
                               "x_hard": 5.714, "smooth_width": 1e-5}})
    # Rail-clip DC points span both soft rails and the interior.
    for x in (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        out.append({"module": "rail_clip", "mode": "dc", "x": x,
                    "params": {"x_max": 4.0, "sharp": 0.02, "Iclip": 0.05}})
    # Transient node capacitor/leak tests: trajectories start at normalized x0.
    for x0 in (-4.0, -1.0, 2.0, 6.0):
        out.append({"module": "node_cap_leak", "mode": "tran", "x0": x0,
                    "params": {"c_eff": 1.0, "g_leak": 0.0486}})
    return out


def ref_current(case, vs, vd):
    p = case["params"]
    xs, xd = (vs - VCM) / VU, (vd - VCM) / VU
    if case["module"] == "shunt":
        gate = p["z_gate"] * p["vca"] if p["vca_gate_shunt"] else p["z_gate"]
        return p["G"] * gate * (xs - xd) * IU
    aa, bb = p.get("A", 1.0), p.get("B", 1.0)
    if p["mode"] == 1 and p["ab_mode"] == 2:
        aa = bb = 1.0
    elif p["mode"] == 1 and p["ab_mode"] == 3:
        aa = 0.5 + 0.5 * p["cap_code_a"] / 7.0
        bb = 0.5 + 0.5 * p["cap_code_b"] / 7.0
    u = p["gm"] * (p["s"] * (aa * xs - bb * xd) + p["theta"])
    if p["mode"]:
        u *= p["g_hw"] * p["f_ic"]
        t = math.tanh(u)
        ip = 0.5 * p["Isat"] * p["gln"] if "gln" in p else 0.5 * p["Isat"]
        ip *= 1.0 + t
        inn = 0.5 * p["Isat"] * (1.0 - t)
        # The measured current is the destination leg, in the va's physical sign convention.
        vtop = 0.8 - vd
        vbot = vd
        vt = 0.03086
        return (ip * (1 - p["lambda_v"] * (vd - VCM)) * (1 - math.exp(-vtop / vt))
                - inn * (1 + p["lambda_v"] * (vd - VCM)) * (1 - math.exp(-vbot / vt))) * IU
    gate = 1.0 / (1.0 + math.exp(-(p["x_max"] - abs(xs)) / p["gate_width"]))
    gate *= 1.0 / (1.0 + math.exp(-(p["x_max"] - abs(xd)) / p["gate_width"]))
    gate *= p.get("z_gate", 1.0) * p.get("sigma_z", 1.0) * p.get("vca", 1.0)
    return p["Isat"] * gate * (math.tanh(u) + 0.01 * u) * IU


def ref_value(case, values):
    module = case["module"]
    if module in ("ota_edge", "shunt"):
        return ref_current(case, values["src"], values["dst"]), IU, "Iu"
    if module in ("vca_bias", "gln_bias"):
        if module == "vca_bias":
            import torch
            from gating.vca import vca_edge_gain
            u = torch.tensor([case["u"]], dtype=torch.float64)
            W = torch.eye(4, dtype=torch.float64)
            edge = torch.tensor([case["weights"]], dtype=torch.float64)
            bias = torch.tensor([case["bias"]], dtype=torch.float64)
            return float(vca_edge_gain(u, W, edge, bias)[0, 0]), 1.0, "gain"
        # GLN's integrated call requires a GLNRails instance and a cell family;
        # this standalone exporter primitive is its clamped log-tail equation.
        z = case["bias"] + sum(w*u for w,u in zip(case["weights"], case["u"]))
        return math.exp(max(-case["delta_max"], min(case["delta_max"], z))), 1.0, "trim"
    if module == "rail_clip":
        import torch
        from cell_library import FreeTanhLibrary
        from differential_stage import DifferentialStage
        p = case["params"]
        stage = DifferentialStage(1, [], [], FreeTanhLibrary(0), x_max=p["x_max"],
                                  clip_current=p["Iclip"], clip_softness=p["sharp"])
        x = torch.tensor([[case["x"]]], dtype=torch.float64)
        return float(stage.soft_clip(x)[0, 0]), 1.0, "Iu"
    # Use the repository's actual transfer module, including its hardware rail clamp.
    import torch
    from types import SimpleNamespace
    from stage_transfer import StageTransfer
    p = case["params"]
    x = torch.tensor([[case["x"]]], dtype=torch.float64)
    hw = SimpleNamespace(
        constraints=SimpleNamespace(hw_clamp_transfer=True),
        units=SimpleNamespace(x_hard=p["x_hard"]))
    transfer = StageTransfer(1, 1, activation="residual-relu-tanh", hw=hw).double()
    with torch.no_grad():
        transfer.residual_w1.fill_(p["w1"])
        transfer.residual_w2.fill_(p["w2"])
        transfer.residual_w3.fill_(p["w3"])
        transfer.residual_vth.fill_(p["vth"])
    y = float(transfer(x)[0, 0].detach())
    return VCM+VU*y, VU, "x"


def psf_waveforms(path):
    traces = {}
    time_value = None
    in_value = False
    for line in path.read_text(errors="replace").splitlines():
        if line.strip() == "VALUE":
            in_value = True
            continue
        if line.strip() == "END":
            in_value = False
            continue
        if not in_value:
            continue
        m = re.match(r'"([^"]+)"\s+([-+0-9.eE]+)', line.strip())
        if not m:
            continue
        name, value = m.group(1), float(m.group(2))
        if name == "time":
            time_value = value
        else:
            traces.setdefault(name, []).append((time_value, value))
    return traces


def psf_nodes(path):
    text = path.read_text(errors="replace")
    vals = {}
    in_value = False
    for line in text.splitlines():
        if line.strip() == "VALUE":
            in_value = True
            continue
        if line.strip() == "END":
            in_value = False
        if in_value:
            m = re.match(r'"([^\"]+)"\s+"(V|I)"\s+([-+0-9.eE]+)', line.strip())
            if m:
                vals[m.group(1)] = float(m.group(3))
    return vals


def run(args):
    if shutil.which("tcsh") is None:
        raise RuntimeError("tcsh unavailable")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUN_ROOT / f"characterize_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir()
    cs = cases()
    modules = sorted({c["module"] for c in cs})
    lines = ["simulator lang=spectre"] + [f'ahdl_include "{VA / (m + ".va")}"' for m in modules]
    lines.append("Vcm (vcm 0) vsource dc=0.4")
    for i, c in enumerate(cs):
        if c["module"] in ("ota_edge", "shunt"):
            vs, vd = VCM + VU*c["xs"], VCM + VU*c["xd"]
            lines += [f"Vsrc{i} (src{i} 0) vsource dc={vs:.17g}",
                      f"Vbias{i} (bias{i} 0) vsource dc={vd:.17g}",
                      f"Rload{i} (dst{i} bias{i}) resistor r=1meg",
                      f"Xdut{i} (src{i} dst{i}) {c['module']} " + " ".join(
                          f"{k}={v}" for k,v in c["params"].items())]
        elif c["module"] in ("vca_bias", "gln_bias"):
            ports = []
            for j, u in enumerate(c["u"]):
                node = f"u{j}_{i}"
                ports.append(node)
                lines.append(f"Vu{j}_{i} ({node} 0) vsource dc={u:.17g}")
            output = f"out{i}"
            ports.append(output)
            params = {f"w{j}": w for j,w in enumerate(c["weights"])}
            params.update({"bias": c["bias"]})
            if c["module"] == "gln_bias":
                params["delta_max"] = c["delta_max"]
            lines.append(f"Xdut{i} ({' '.join(ports)}) {c['module']} " +
                         " ".join(f"{k}={v}" for k,v in params.items()))
        elif c["module"] == "rail_clip":
            node = f"clip{i}"
            lines += [f"Vclip{i} ({node} 0) vsource dc={VCM+VU*c['x']:.17g}",
                      f"Xdut{i} ({node}) rail_clip " + " ".join(
                          f"{k}={v}" for k,v in c["params"].items())]
        elif c["module"] == "node_cap_leak":
            node = f"leak{i}"
            lines += [f"Xdut{i} ({node}) node_cap_leak " + " ".join(
                          f"{k}={v}" for k,v in c["params"].items()),
                      f"ic {node}={VCM+VU*c['x0']:.17g}"]
        else:
            vin, en, output = f"in{i}", f"en{i}", f"out{i}"
            lines += [f"Vin{i} ({vin} 0) vsource dc={VCM+VU*c['x']:.17g}",
                      f"Ven{i} ({en} 0) vsource dc=1.0",
                      f"Rout{i} ({output} vcm) resistor r=1e12",
                      f"Xdut{i} ({vin} {output} {en}) interstage_transfer " +
                      " ".join(f"{k}={v}" for k,v in c["params"].items())]
    lines += ["saveOptions options save=all", "dcOp dc"]
    if any(c["module"] == "node_cap_leak" for c in cs):
        from hw.units import HWUnits
        units = HWUnits()
        tau = units.cu / (0.0486 * units.gu)
        lines += ["simulatorOptions options reltol=1e-8 vabstol=1e-12 iabstol=1e-18",
                  f"tran tran stop={5*tau:.17g} maxstep={tau/200:.17g} errpreset=conservative"]
    deck = run_dir / "characterize.scs"
    deck.write_text("\n".join(lines) + "\n")
    command = (f"cd {run_dir}; source {SETUP} >& /dev/null; "
               f"spectre -format psfascii {deck.name} +log characterize.log")
    launch = subprocess.run(["bash", "-c", f"nohup tcsh -c '{command}' > {run_dir}/run.out 2>&1 < /dev/null & echo $!"],
                            text=True, capture_output=True, check=True)
    pid = int(launch.stdout.strip())
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    log = (run_dir / "characterize.log").read_text(errors="replace")
    if not re.search(r"spectre completes with 0 errors, 0 warnings", log):
        raise RuntimeError(f"Spectre failed; see {run_dir / 'characterize.log'}")
    raw = run_dir / "characterize.raw" / "dcOp.dc"
    nodes = psf_nodes(raw)
    rows = []
    traces = psf_waveforms(run_dir / "characterize.raw" / "tran.tran.tran")
    hw_units = __import__("hw.units", fromlist=["HWUnits"]).HWUnits()
    for i,c in enumerate(cs):
        module = c["module"]
        if module in ("ota_edge", "shunt"):
            vs, vd = nodes[f"src{i}"], nodes[f"dst{i}"]
            measured = nodes[f"Vbias{i}:p"]
            if module == "ota_edge" and c["params"].get("mode") == 1:
                measured = -measured  # HW has an independent sink/source leg at dst.
            values = {"src": vs, "dst": vd}
        elif module in ("vca_bias", "gln_bias"):
            measured = nodes[f"out{i}"]
            values = {}
        elif module == "rail_clip":
            # Vsource branch current is positive into the source; convert A to Iu.
            measured = -nodes[f"Vclip{i}:p"] / IU
            values = {}
        elif module == "node_cap_leak":
            # Transient waveform is parsed from the PSFASCII tran dataset below.
            continue
        else:
            measured = nodes[f"out{i}"]
            values = {}
        expected, scale, units = ref_value(c, values)
        abs_err = abs(measured - expected) / scale
        rel_err = abs(measured - expected) / max(abs(expected), scale * 1e-12)
        rows.append({"module": module, "mode": c["mode"], "error_units": units,
                     "abs_error_model_units": abs_err, "rel_error": rel_err,
                     "measured_model_units": measured/scale,
                     "reference_model_units": expected/scale})
    for i, c in enumerate(cs):
        if c["module"] != "node_cap_leak":
            continue
        samples = traces[f"leak{i}"]
        errors = []
        for t, voltage in samples:
            x_meas = (voltage - VCM) / VU
            x_ref = c["x0"] * math.exp(-c["params"]["g_leak"] * (t / hw_units.tu) / c["params"]["c_eff"])
            errors.append(abs(x_meas - x_ref))
        max_error = max(errors)
        rows.append({"module": "node_cap_leak", "mode": "tran", "x0": c["x0"],
                     "samples": len(samples), "error_units": "x",
                     "abs_error_model_units": max_error, "rel_error": max_error / max(abs(c["x0"]), 1e-12)})
    report = {"run_dir": str(run_dir), "measured_cases": len(rows), "uncharacterized": [
        "sample_hold", "boundary_sense_ota", "crossbar_accumulator", "phase_ctrl"], "results": rows}
    out = run_dir / "characterization.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for module in sorted({r["module"] for r in rows}):
        r = [x for x in rows if x["module"] == module]
        print(f"{module}: {len(r)} cases; max abs={max(x['abs_error_model_units'] for x in r):.6g} "
              f"{r[0]['error_units']}, max rel={max(x['rel_error'] for x in r):.6g}")
    print(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        print(f"{len(cases())} DC cases")
        return 0
    run(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
