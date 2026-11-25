#!/usr/bin/env python3
"""Export a KirchhoffNet deck, optionally simulate it, and decode CTLE knobs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from checkpoints import DEFAULT_CKPT, load_paper_checkpoint
from export.spectre_export import _load_checkpoint_for_hw, export_model
from export.spectre_netlist import render, render_timemux
from export.netlist_interp import _t, _transfer
from hw import HWConfig
from scripts.spectre.characterize import psf_waveforms

from scripts.ctle_closure.mapping import PARAM_COLS, bounded_params, write_params

DEFAULT_DATA = Path.home() / "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz"
BATCH = 16


def _ordered_specs(data, split, n, seed=100):
    indices = np.asarray(data[f"{split}_idx"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(indices))
    rng.shuffle(shuffled)
    selected = np.sort(shuffled[:n])
    return np.asarray(data["specs"], dtype=np.float32)[indices[selected]], indices[selected]


def _decode_spectre(deck_dir, params, specs, indices, fabric):
    """Read the final physical node state and reproduce netlist_interp readout."""
    raw = deck_dir / "hrknet.raw" / "tran.tran.tran"
    traces = psf_waveforms(raw)
    resolved = params["resolved"]
    units = params["units"]
    vu, vcm, tu = (float(units[k]) for k in ("Vu_V", "Vcm_V", "Tu_s"))
    spans = [float(s["solver"]["t_span"]) * tu for s in resolved["stages"]]
    if fabric == "copies":
        end_times = [sum(spans)] * len(specs)
        node_name = lambda i, j: f"n{i}_6_{j}"
    else:
        overhead = min(2e-9, min(spans)/100)*2 + min(0.2e-9, min(spans)/1000)
        starts = [sum(spans[:k]) + k*overhead for k in range(7)]
        end_times = [starts[6]+spans[6]] * len(specs)
        node_name = lambda i, j: f"n{i}_{j}"
    out = []
    stage = resolved["stages"][-1]
    start = int(stage["topology"]["readout_dst_start"])
    count = int(resolved["topology"]["readout_count"])
    rd = resolved["readout"]
    for i, spec in enumerate(specs):
        state = []
        for j in range(int(stage["num_nodes"])):
            trace = np.asarray(traces[node_name(i, j)], dtype=np.float64)
            voltage = np.interp(end_times[i], trace[:, 0], trace[:, 1])
            state.append((voltage-vcm)/vu)
        y = torch.tensor([state[start:start+count]], dtype=torch.float64)
        y = _t(rd["gain"], y)*y + _t(rd["bias"], y) + _t(rd["tanh_gain"], y)*torch.tanh(y)
        y = y + _t(rd["relu_gain"], y)*torch.relu(y-_t(rd["relu_threshold"], y))
        y = _transfer(y, {"in_nodes": y.shape[1], "out_nodes": y.shape[1], **rd["post_transfer"]})
        out.append(y[0].cpu().numpy())
    return np.asarray(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--fabric", choices=("copies", "timemux"), default="copies")
    p.add_argument("--mode", choices=("legacy", "hw"), default="legacy")
    p.add_argument("--hw-preset", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--specs", type=int, default=64)
    g.add_argument("--split", choices=("test",), default=None)
    p.add_argument("--limit", type=int, default=None, help="Number of split specs; defaults to 64.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-spectre", action="store_true")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cadence-setup", type=Path, default=Path.home()/"workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh")
    a = p.parse_args(argv)
    n = a.limit if a.split else a.specs
    n = 64 if n is None else n
    if n < 1: p.error("spec count must be positive")
    if a.mode == "hw" and a.fabric == "timemux": p.error("the exporter currently supports hw mode only with copies")
    a.out.mkdir(parents=True, exist_ok=True)
    data = np.load(a.dataset)
    split = a.split or "test"
    specs, spec_ids = _ordered_specs(data, split, n)
    device = torch.device("cpu")
    if a.mode == "hw":
        if not a.hw_preset: p.error("--hw-preset is required with --mode hw")
        hw = HWConfig.from_json(a.hw_preset)
        model = _load_checkpoint_for_hw(a.checkpoint, hw)
        hw_json = a.hw_preset
    else:
        hw = None
        model = load_paper_checkpoint(a.checkpoint, dtype=torch.float64)
        hw_json = None
    model.eval()
    torch.set_num_threads(a.threads)
    with torch.no_grad():
        spec_tensor = torch.as_tensor(specs, dtype=next(model.parameters()).dtype, device=device)
        torch_logits = model(spec_tensor)
        torch_log = bounded_params(torch_logits, model).cpu().numpy()
    write_params(a.out/"pred_torch.csv", torch_log)
    with (a.out/"specs.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(("power","stage_2_jitter","stage_2_eye_max_height","stage_2_eye_max_width")); w.writerows(specs)
    np.save(a.out/"spec_ids.npy", spec_ids)
    all_logits=[]
    deck_root=a.out/"decks"; deck_root.mkdir(exist_ok=True)
    setup=a.cadence_setup.expanduser()
    for start in range(0,n,BATCH):
        stop=min(start+BATCH,n); chunk=specs[start:stop]
        d=deck_root/f"deck_{start//BATCH:03d}"; d.mkdir(parents=True,exist_ok=True)
        if a.mode == "hw":
            table=export_model(model,d,"hw",a.fabric,len(chunk),hw_json,hw)
        else:
            table=export_model(model,d,"legacy",a.fabric,len(chunk))
        deck=d/"hrknet.scs"
        (render if a.fabric=="copies" else render_timemux)(table,deck,inputs=chunk)
        if not a.run_spectre:
            print(f"Planned: cd {d} && source {setup} && spectre -format psfascii hrknet.scs +log hrknet.log")
            continue
        cmd=f"cd '{d}'; source '{setup}' >& /dev/null; which spectre; spectre -format psfascii hrknet.scs +log hrknet.log"
        proc=subprocess.run(["tcsh","-c",cmd],cwd=d,check=False)
        if proc.returncode: raise RuntimeError(f"Spectre failed for {d} (exit {proc.returncode})")
        spectre_logits=_decode_spectre(d,table,chunk,spec_ids[start:stop],a.fabric)
        all_logits.extend(spectre_logits)
    if a.run_spectre:
        spectre_log=bounded_params(torch.as_tensor(all_logits,dtype=model.log_lo.dtype),model).cpu().numpy()
        write_params(a.out/"pred_spectre.csv",spectre_log)
    else:
        # Deck-only mode cannot make predicted Spectre knobs; clearly mark the absent file.
        print(f"Deck-only mode: predicted Spectre CSV will be created after --run-spectre; Torch CSV: {a.out/'pred_torch.csv'}")
    meta={"checkpoint":str(a.checkpoint),"mode":a.mode,"fabric":a.fabric,"split":split,"n_specs":n,"batch_specs":BATCH,"spec_ids":spec_ids.tolist(),"csv_columns":PARAM_COLS,"csv_units":"log10 physical units, per spice_failure_check.py"}
    (a.out/"prediction_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
