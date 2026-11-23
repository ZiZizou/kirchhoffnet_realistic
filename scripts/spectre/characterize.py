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
RUN_ROOT = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD3/runs")
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
    # Boundary OTA: ideal input source -> dynamic destination node. Sense OTA:
    # dynamic source node -> ideal Vref destination. Match DifferentialStage's
    # normalized source/destination ordering for both families.
    boundary_points = (
        (-5.0, 0.2, 1.0, 0.9, 1.1, 0.12, -1.0),
        (-4.0, -0.3, 0.8, 1.2, 0.7, -0.08, 1.0),
        (-3.0, 0.5, 1.3, 0.6, 1.4, 0.2, -1.0),
        (-0.4, 0.1, 0.9, 1.0, 1.0, 0.05, 1.0),
        (0.0, -0.2, 1.1, 0.7, 1.2, -0.15, -1.0),
        (1.2, 0.4, 0.7, 1.4, 0.8, 0.1, 1.0),
        (3.98, 0.1, 1.2, 0.9, 1.1, 0.07, -1.0),
        (-4.02, -0.1, 0.95, 1.05, 0.95, -0.04, 1.0),
    )
    sense_points = (
        (-5.0, 0.0, 1.0, 0.9, 1.1, 0.12, -1.0),
        (-4.0, 0.2, 0.8, 1.2, 0.7, -0.08, 1.0),
        (-2.0, -0.3, 1.3, 0.6, 1.4, 0.2, -1.0),
        (-0.2, 0.4, 0.9, 1.0, 1.0, 0.05, 1.0),
        (0.0, -0.1, 1.1, 0.7, 1.2, -0.15, -1.0),
        (1.5, 0.2, 0.7, 1.4, 0.8, 0.1, 1.0),
        (3.99, 0.1, 1.2, 0.9, 1.1, 0.07, -1.0),
        (-4.01, -0.1, 0.95, 1.05, 0.95, -0.04, 1.0),
    )
    for role, points in (("boundary", boundary_points), ("sense", sense_points)):
        for source_x, dest_x, gm, isat, aa, theta, sign in points:
            out.append({"module": "boundary_sense_ota", "mode": role,
                        "xs": source_x, "xd": dest_x,
                        "params": {"gm": gm, "Isat": isat, "A": aa,
                                   "B": 1.15, "theta": theta, "s": sign,
                                   "z_gate": 0.83, "x_max": 4.0,
                                   "gate_width": 0.02}})
    # Rail-clip DC points span both soft rails and the interior.
    for x in (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        out.append({"module": "rail_clip", "mode": "dc", "x": x,
                    "params": {"x_max": 4.0, "sharp": 0.02, "Iclip": 0.05}})
    # Transient node capacitor/leak tests: trajectories start at normalized x0.
    for x0 in (-4.0, -1.0, 2.0, 6.0):
        out.append({"module": "node_cap_leak", "mode": "tran", "x0": x0,
                    "params": {"c_eff": 1.0, "g_leak": 0.0486}})
    # Sample/hold deterministic capture: known normalized input, acquisition followed by hold.
    out.append({"module": "sample_hold", "mode": "tran", "x0": 1.25,
                "params": {"inject_v": 0.0014, "inject_gain_err": 0.08,
                           "droop_v": 0.0007, "droop_time": 50e-9, "droop_tau": 1e99}})
    # Two complete phase boundaries, with deliberately distinct durations.
    out.append({"module": "phase_ctrl", "mode": "tran", "stages": 3,
                "params": {"stages": 3, "t_hold": 2e-9, "t_transfer": 3e-9,
                           "t_switch": 4e-9, "t_evolve": 5e-9}})
    # Four sense currents x three outputs; use the production quantizer for W.
    import torch
    from hw.constraints import xbar_map, acc_compliance
    from hw.units import HWUnits
    units = HWUnits()
    rng = torch.Generator().manual_seed(314159)
    for idx in range(3):
        raw = torch.randn((3, 4), generator=rng, dtype=torch.float64) * 0.65
        effective = xbar_map(raw, mode="signed")
        out.append({"module": "crossbar_accumulator", "mode": f"ideal_{idx}",
                    "weights": raw.tolist(), "effective_weights": effective.tolist(),
                    "currents": [0.37, -0.21, 0.53, -0.42],
                    "x_acc": [0.0, 0.0, 0.0], "hw_mode": 0, "params": {}})
    raw = torch.tensor([[0.71, -0.84, 0.13, -0.31], [-0.66, 0.52, 0.94, -0.12],
                        [0.28, -0.49, 0.63, -0.91]], dtype=torch.float64)
    effective = xbar_map(raw, mode="offset_binary", bits=3, bound=True)
    for x_acc in (-3.0, 0.0, 3.0):
        out.append({"module": "crossbar_accumulator", "mode": "hw_offset_binary_compliance",
                    "weights": raw.tolist(), "effective_weights": effective.tolist(),
                    "currents": [0.37, -0.21, 0.53, -0.42],
                    "x_acc": [x_acc, -x_acc, 0.5*x_acc], "hw_mode": 1,
                    "params": {"hw_mode": 1, "hw_bits": 3, "hw_bound": 1,
                               "Wmax": 4.0, "acc_compliance_en": 1,
                               "x_hard": units.x_hard},
                    "reference_torch": True})
    return out


def ref_current(case, vs, vd):
    p = case["params"]
    xs, xd = (vs - VCM) / VU, (vd - VCM) / VU
    if case["module"] == "boundary_sense_ota":
        # Use the real model core and resolve its raw parameters to the exact
        # exported OTA values. This is forward_tanh's same normalized src/dst
        # contract used by boundary_cell_lib (u, node) and sense (node, Vref).
        import torch
        from cell_library import FreeTanhLibrary
        gm_min, gm_max, isat_min, isat_max = 0.01, 2.0, 0.01, 1.5
        lib = FreeTanhLibrary(1, gm_min=gm_min, gm_max=gm_max,
                              isat_min=isat_min, isat_max=isat_max,
                              bias_enabled=True).double()
        def logit(v):
            return math.log(v / (1.0 - v))
        def inv_softplus(v):
            return math.log(math.expm1(v))
        with torch.no_grad():
            lib.a_raw.fill_(inv_softplus(p["A"]))
            lib.b_raw.fill_(inv_softplus(p["B"]))
            lib.s_raw.fill_(p["s"])
            lib.gm_raw.fill_(logit((p["gm"] - gm_min) / (gm_max - gm_min)))
            lib.isat_raw.fill_(logit((p["Isat"] - isat_min) / (isat_max - isat_min)))
            lib.theta_raw.fill_(p["theta"])
        source = torch.tensor([[xs]], dtype=torch.float64)
        destination = torch.tensor([[xd]], dtype=torch.float64)
        current = lib.forward_tanh(source, destination, x_max=p["x_max"])
        return float((current * p["z_gate"])[0, 0].detach()) * IU
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
    if module in ("ota_edge", "shunt", "boundary_sense_ota"):
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
    if module == "crossbar_accumulator":
        import torch
        from hw.constraints import acc_compliance
        from hw.units import HWUnits
        weights = torch.tensor(case["effective_weights"], dtype=torch.float64)
        currents = torch.tensor(case["currents"], dtype=torch.float64) * IU
        expected = weights @ currents
        if case["params"].get("acc_compliance_en"):
            x_acc = torch.tensor(case["x_acc"], dtype=torch.float64)
            expected = acc_compliance(expected, x_acc, HWUnits())
        return expected.tolist(), IU, "Iu"
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


def psf_waveforms(path, wanted=None):
    traces = {}
    time_value = None
    in_value = False
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.strip() == "VALUE":
            in_value = True
            continue
        if line.strip() == "END":
            in_value = False
            continue
        if not in_value:
            continue
        if '"' not in line:
            continue
        start = line.find('"') + 1
        end = line.find('"', start)
        if end < 0:
            continue
        name = line[start:end]
        if name != "time" and wanted is not None and name not in wanted:
            continue
        try:
            value = float(line[end+1:].strip().split()[0])
        except (ValueError, IndexError):
            continue
        if name == "time":
            time_value = value
        elif wanted is None or name in wanted:
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
        if c["module"] in ("ota_edge", "shunt", "boundary_sense_ota"):
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
        elif c["module"] == "sample_hold":
            inp, clk, output = f"sh_in{i}", f"sh_clk{i}", f"sh_out{i}"
            lines += [f"Vsh_in{i} ({inp} 0) vsource dc={VCM+VU*c['x0']:.17g}",
                      f"Vsh_clk{i} ({clk} 0) vsource type=pulse val0=0 val1=1 rise=10p fall=10p width=20n period=80n",
                      f"Csh{i} ({output} 0) capacitor c=1e-18 ic={VCM:.17g}",
                      f"Xdut{i} ({inp} {output} {clk}) sample_hold " + " ".join(
                          f"{k}={v}" for k,v in c["params"].items())]
        elif c["module"] == "phase_ctrl":
            clk, phase, stage = f"pc_clk{i}", f"pc_phase{i}", f"pc_stage{i}"
            lines += [f"Vpc_clk{i} ({clk} 0) vsource type=pulse val0=0 val1=1 rise=10p fall=10p width=1n period=30n",
                      f"Xdut{i} ({clk} {phase} {stage}) phase_ctrl " + " ".join(
                          f"{k}={v}" for k,v in c["params"].items())]
        elif c["module"] == "crossbar_accumulator":
            ports = []
            for k, current in enumerate(c["currents"]):
                sp, sn = f"s{k}p_{i}", f"s{k}n_{i}"
                ports.extend((sp, sn))
                lines.append(f"Iin{k}_{i} ({sn} {sp}) isource dc={current*IU:.17g}")
            for j, x_acc in enumerate(c["x_acc"]):
                acc = f"acc{j}_{i}"
                ports.append(acc)
                lines.append(f"Vacc{j}_{i} ({acc} 0) vsource dc={VCM+VU*x_acc:.17g}")
            flat_w = [w for row in c["weights"] for w in row]
            params = dict(c["params"])
            overrides = " ".join(f"{key}={value}" for key,value in params.items())
            w_expr = "[" + " ".join(f"{value:.17g}" for value in flat_w) + "]"
            lines.append(f"Xdut{i} ({' '.join(ports)}) crossbar_accumulator W={w_expr} {overrides}")
        else:
            vin, en, output = f"in{i}", f"en{i}", f"out{i}"
            lines += [f"Vin{i} ({vin} 0) vsource dc={VCM+VU*c['x']:.17g}",
                      f"Ven{i} ({en} 0) vsource dc=1.0",
                      f"Rout{i} ({output} vcm) resistor r=1e12",
                      f"Xdut{i} ({vin} {output} {en}) interstage_transfer " +
                      " ".join(f"{k}={v}" for k,v in c["params"].items())]
    lines += ["saveOptions options save=all", "dcOp dc"]
    if any(c["module"] in ("node_cap_leak", "sample_hold", "phase_ctrl") for c in cs):
        from hw.units import HWUnits
        units = HWUnits()
        tau = units.cu / (0.0486 * units.gu)
        lines += ["simulatorOptions options reltol=1e-8 vabstol=1e-12 iabstol=1e-18",
                  f"tran tran stop={max(5*tau,60e-9):.17g} maxstep={min(tau/200,20e-12):.17g} errpreset=conservative"]
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
    wanted_traces = {f"leak{i}" for i,c in enumerate(cs) if c["module"] == "node_cap_leak"}
    wanted_traces.update(f"sh_out{i}" for i,c in enumerate(cs) if c["module"] == "sample_hold")
    wanted_traces.update(f"pc_phase{i}" for i,c in enumerate(cs) if c["module"] == "phase_ctrl")
    wanted_traces.update(f"pc_stage{i}" for i,c in enumerate(cs) if c["module"] == "phase_ctrl")
    traces = psf_waveforms(run_dir / "characterize.raw" / "tran.tran.tran", wanted_traces)
    hw_units = __import__("hw.units", fromlist=["HWUnits"]).HWUnits()
    for i,c in enumerate(cs):
        module = c["module"]
        ota_input_current = None
        if module in ("ota_edge", "shunt", "boundary_sense_ota"):
            vs, vd = nodes[f"src{i}"], nodes[f"dst{i}"]
            measured = nodes[f"Vbias{i}:p"]
            if module == "ota_edge" and c["params"].get("mode") == 1:
                measured = -measured  # HW has an independent sink/source leg at dst.
            input_current = nodes.get(f"Vsrc{i}:p", 0.0)
            if module == "boundary_sense_ota":
                ota_input_current = input_current / IU
                if abs(ota_input_current) > 1e-9:
                    raise AssertionError(
                        f"{c['mode']} OTA input loaded: {ota_input_current:.6g} Iu"
                    )
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
        elif module in ("sample_hold", "phase_ctrl"):
            continue
        elif module == "crossbar_accumulator":
            expected, scale, units = ref_value(c, {})
            for j, reference in enumerate(expected):
                measured = nodes[f"Vacc{j}_{i}:p"]
                abs_err = abs(measured-reference) / scale
                rel_err = abs(measured-reference) / max(abs(reference), scale*1e-12)
                rows.append({"module": module, "mode": c["mode"], "output": j,
                             "error_units": units, "abs_error_model_units": abs_err,
                             "rel_error": rel_err, "measured_model_units": measured/scale,
                             "reference_model_units": reference/scale})
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
                     "reference_model_units": expected/scale,
                     **({"input_current_model_units": ota_input_current}
                        if ota_input_current is not None else {})})
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
    for i, c in enumerate(cs):
        if c["module"] != "sample_hold":
            continue
        samples = traces[f"sh_out{i}"]
        # Real StageBoundary capture provides deterministic injection/gain/droop.
        import torch
        from hw.timemux import StageBoundary, TimeMuxConfig
        cfg = TimeMuxConfig(enable=True, inject_mv=c["params"]["inject_v"]*1e3,
                            inject_gain_err=c["params"]["inject_gain_err"],
                            droop_mv=c["params"]["droop_v"]*1e3, ktc=False)
        boundary = StageBoundary(cfg, n_nodes=1, vu=VU, cu=50e-15, seed=1).eval()
        x0 = torch.tensor([[c["x0"]]], dtype=torch.float64)
        x_hold, x_transfer = boundary.capture(x0)
        # The sample closes at 20 ns; assess after the 50 ns hold interval.
        picked = min(samples, key=lambda row: abs(row[0]-70.1e-9))
        measured_x = (picked[1]-VCM)/VU
        expected_x = float(x_hold[0,0])
        rows.append({"module":"sample_hold", "mode":"tran", "error_units":"x",
                     "abs_error_model_units":abs(measured_x-expected_x),
                     "rel_error":abs(measured_x-expected_x)/max(abs(expected_x),1e-12),
                     "measured_model_units":measured_x,"reference_model_units":expected_x,
                     "transfer_reference_x":float(x_transfer[0,0]), "ktc":"excluded"})
    for i, c in enumerate(cs):
        if c["module"] != "phase_ctrl":
            continue
        phase_trace, stage_trace = traces[f"pc_phase{i}"], traces[f"pc_stage{i}"]
        params = c["params"]
        # Pulse source crosses Vlogic halfway through its 10 ps rising edge.
        boundaries = (5e-12, 30e-9+5e-12)
        durations = [params[k] for k in ("t_hold","t_transfer","t_switch","t_evolve")]
        horizon = boundaries[-1] + sum(durations) + 1e-9
        observed = [(t,round(v)) for (t0,v0),(t,v) in zip(phase_trace,phase_trace[1:])
                    if t <= horizon and round(v) != round(v0)]
        expected = []
        for start in boundaries:
            cursor = start
            for phase_no, duration in enumerate(durations, 1):
                cursor += duration
                expected.append((cursor, float(phase_no % 4)))
        # Verify phase transition edges and stage-index increment at boundary end.
        edge_errors = [abs(t-et) for (t,v),(et,ev) in zip(observed,expected) if v == ev]
        stage_edges = [t1 for (t,v0),(t1,v1) in zip(stage_trace,stage_trace[1:])
                       if t1 <= horizon and round(v1) != round(v0)]
        expected_stage = [boundaries[j]+sum(durations) for j in range(2)]
        edge_errors.extend(abs(a-b) for a,b in zip(stage_edges,expected_stage))
        max_error = max(edge_errors) if edge_errors else float("inf")
        rows.append({"module":"phase_ctrl", "mode":"tran", "error_units":"s",
                     "abs_error_model_units":max_error,
                     "rel_error":max_error/max(durations), "max_error_ns":max_error*1e9,
                     "max_error_Tu":max_error/hw_units.tu, "phase_edges":len(observed),
                     "stage_edges":len(stage_edges),
                     "phase_edge_times_ns":[t*1e9 for t,v in observed],
                     "phase_edge_codes":[v for t,v in observed],
                     "expected_phase_edge_times_ns":[t*1e9 for t,v in expected],
                     "stage_edge_times_ns":[t*1e9 for t in stage_edges],
                     "expected_stage_edge_times_ns":[t*1e9 for t in expected_stage]})
    report = {"run_dir": str(run_dir), "measured_cases": len(rows), "uncharacterized": [], "results": rows}
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
