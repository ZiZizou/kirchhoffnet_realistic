#!/usr/bin/env python3
"""Cell-level Spectre characterization against the analytic/PyTorch contract.

All Cadence-generated files are placed in .agent_scratch/taskD2/runs.  The
current measured cases focus on the OTA and shunt current primitives; modules
not listed in CASES are explicitly reported as uncharacterized rather than
silently treated as passing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

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
    lines = ["simulator lang=spectre", f'ahdl_include "{VA / "ota_edge.va"}"',
             f'ahdl_include "{VA / "shunt.va"}"',
             "Vcm (vcm 0) vsource dc=0.4"]
    for i, c in enumerate(cs):
        vs, vd = VCM + VU*c["xs"], VCM + VU*c["xd"]
        vb = VCM + VU*c["xd"]
        lines += [f"Vsrc{i} (src{i} 0) vsource dc={vs:.17g}",
                  f"Vbias{i} (bias{i} 0) vsource dc={vb:.17g}",
                  f"Rload{i} (dst{i} bias{i}) resistor r=1meg",
                  f"Xdut{i} (src{i} dst{i}) {c['module']} " + " ".join(
                      f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k,v in c["params"].items())]
    lines += ["saveOptions options save=all", "dcOp dc"]
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
    for i,c in enumerate(cs):
        vs, vd, vb = nodes[f"src{i}"], nodes[f"dst{i}"], nodes[f"bias{i}"]
        measured = nodes[f"Vbias{i}:p"]
        if c["module"] == "ota_edge" and c["params"].get("mode") == 1:
            measured = -measured  # HW has an independent sink/source leg at dst.
        expected = ref_current(c, vs, vd)
        abs_err = abs(measured - expected) / IU
        rel_err = abs(measured - expected) / max(abs(expected), IU * 1e-12)
        rows.append({"module": c["module"], "mode": c["mode"], "abs_error_model_units": abs_err,
                     "rel_error": rel_err, "measured_model_units": measured/IU,
                     "reference_model_units": expected/IU})
    report = {"run_dir": str(run_dir), "measured_cases": len(rows), "uncharacterized": [
        "node_cap_leak", "rail_clip", "sample_hold", "boundary_sense_ota",
        "crossbar_accumulator", "interstage_transfer", "vca_bias", "gln_bias", "phase_ctrl"],
        "results": rows}
    out = run_dir / "characterization.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for module in ("ota_edge", "shunt"):
        r = [x for x in rows if x["module"] == module]
        print(f"{module}: {len(r)} cases; max abs={max(x['abs_error_model_units'] for x in r):.6g} Iu, "
              f"max rel={max(x['rel_error'] for x in r):.6g}")
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
