#!/usr/bin/env python3
"""Compile each Verilog-A module independently with the site's Spectre setup.

The caller must run this on ssr0 with access to the GF22FDX setup and a
Spectre license. All generated inputs, Cadence setup drops, and results live
under .agent_scratch/taskD2/runs, never in the source worktree.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VA = ROOT / "veriloga"
RUN_ROOT = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD3/runs/compile_modules")
SETUP = "~/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
PORTS = {
    "ota_edge": ("src", "dst"),
    "shunt": ("src", "dst"),
    "node_cap_leak": ("node",),
    "rail_clip": ("node",),
    "sample_hold": ("in", "out", "clk"),
    "boundary_sense_ota": ("src", "dst"),
    "crossbar_accumulator": ("s0p", "s0n", "s1p", "s1n", "s2p", "s2n", "s3p", "s3n", "acc0", "acc1", "acc2"),
    "interstage_transfer": ("in", "out", "enable"),
    "vca_bias": ("u0", "u1", "u2", "u3", "gain"),
    "gln_bias": ("u0", "u1", "u2", "u3", "trim"),
    "phase_ctrl": ("clk", "phase", "stage"),
    "stage_gate": ("node", "source"),
    "ota_edge_sampled": ("src_hold", "dst_hold", "src_live", "dst_live"),
    "ota_edge_banked": ("src", "dst", "stage", "en"),
    "ota_edge_sampled_banked": ("src_hold", "dst_hold", "src_live", "dst_live", "stage", "en"),
    "shunt_banked": ("src", "dst", "stage", "en"),
    "boundary_sense_ota_banked": ("src", "dst", "stage", "en"),
    "node_cap_leak_banked": ("node", "stage", "en"),
    "rail_clip_banked": ("node", "stage", "en"),
}


def write_deck(module: str, run_dir: Path) -> Path:
    terminals = []
    for i, port in enumerate(PORTS[module]):
        node = f"n{i}"
        terminals.append(node)
        # Give every terminal a DC path without fighting a module's own voltage contribution.
        (run_dir / f"src_{i}.scs").write_text(
            f"R{i} ({node} 0) resistor r=1G\n"
        )
    deck = run_dir / f"{module}.scs"
    sources = "".join((run_dir / f"src_{i}.scs").read_text() for i in range(len(terminals)))
    deck.write_text(
        "simulator lang=spectre\n"
        f'ahdl_include "{VA / (module + ".va")}"\n'
        + sources
        + f"Xdut ({' '.join(terminals)}) {module}\n"
        + "simulatorOptions options reltol=1e-4\n"
        + "dcOp dc\n"
    )
    return deck


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=(*PORTS, "all"), default="all")
    parser.add_argument("--keep", action="store_true", help="retain run directories")
    args = parser.parse_args()
    if shutil.which("tcsh") is None:
        parser.error("tcsh is required to source the Cadence setup")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    modules = PORTS if args.module == "all" else (args.module,)
    failed = []
    for module in modules:
        run_dir = RUN_ROOT / module
        run_dir.mkdir(parents=True, exist_ok=True)
        deck = write_deck(module, run_dir)
        command = (
            f"cd {run_dir}; source {SETUP} >& /dev/null; "
            f"spectre -format psfascii {deck.name} +log {module}.log"
        )
        launch = subprocess.run(
            ["bash", "-c", f"nohup tcsh -c '{command}' > {run_dir}/run.out 2>&1 < /dev/null & echo $!"],
            cwd=run_dir, text=True, stdout=subprocess.PIPE, check=True,
        )
        pid = int(launch.stdout.strip())
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.5)
        output = (run_dir / "run.out").read_text(errors="replace")
        (run_dir / "spectre.log").write_text(output)
        summary = re.search(r"spectre completes with (\d+) errors, (\d+) warnings", output)
        ok = summary is not None and tuple(map(int, summary.groups())) == (0, 0)
        print(f"{'PASS' if ok else 'FAIL'} {module}: {run_dir / 'spectre.log'}")
        if not ok:
            failed.append(module)
    print(f"Compiled {len(modules)-len(failed)}/{len(modules)} modules")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
