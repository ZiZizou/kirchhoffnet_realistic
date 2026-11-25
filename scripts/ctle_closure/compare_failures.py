#!/usr/bin/env python3
"""Compare spice_failure_check results for Spectre and PyTorch knob sets."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import math
import numpy as np

PARAM_COLS = ("fW", "current", "ind", "Rd", "Cs", "Rs", "VDD")


def read_results(path):
    with Path(path).open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or not {"idx","failure"}.issubset(rows[0]):
        raise ValueError(f"{path} is not a spice_failure_check results.csv")
    result={}
    for row in rows:
        i=int(row["idx"])
        if i in result: raise ValueError(f"duplicate result idx {i} in {path}")
        result[i]={"failure":bool(int(float(row["failure"]))),"error":bool(int(float(row.get("error",0))))}
    return result


def read_knobs(path):
    with Path(path).open(newline="") as f:
        reader=csv.DictReader(f)
        missing=set(PARAM_COLS)-set(reader.fieldnames or ())
        if missing: raise ValueError(f"{path} missing expected knob columns {sorted(missing)}")
        rows=np.asarray([[float(r[k]) for k in PARAM_COLS] for r in reader],dtype=np.float64)
    if rows.ndim!=2 or rows.shape[1]!=7 or not np.isfinite(rows).all():
        raise ValueError(f"invalid knob array in {path}")
    return rows


def compare(spectre_path, torch_path, spec_csv, spectre_knobs, torch_knobs, out):
    sr, tr=read_results(spectre_path), read_results(torch_path)
    sk, tk=read_knobs(spectre_knobs), read_knobs(torch_knobs)
    if len(sk)!=len(tk): raise ValueError("prediction CSV row counts differ")
    n=len(sk)
    if set(sr)!=set(range(n)) or set(tr)!=set(range(n)):
        raise ValueError(f"result row ids must exactly cover 0..{n-1}; spectre={len(sr)}, torch={len(tr)}")
    deltas=np.abs(np.power(10.0,sk)-np.power(10.0,tk))
    maxima=deltas.max(axis=0)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    spec_ids=list(range(n))
    if spec_csv:
        p = Path(spec_csv)
        if p.suffix == ".npy":
            spec_ids = np.load(p).astype(np.int64).tolist()
        else:
            with p.open(newline="") as f:
                spec_ids = [int(float(r[0])) for r in csv.reader(f) if r]
                if spec_ids and not str(spec_ids[0]).lstrip("+-").isdigit(): spec_ids=spec_ids[1:]
    if len(spec_ids)!=n: spec_ids=list(range(n))
    rows=[]
    for i in range(n):
        rows.append({"result_idx":i,"canonical_spec_id":spec_ids[i],"spectre_pass":not sr[i]["failure"],"torch_pass":not tr[i]["failure"],"spectre_error":sr[i]["error"],"torch_error":tr[i]["error"],"disagree":sr[i]["failure"]!=tr[i]["failure"],"max_abs_knob_delta":float(deltas[i].max()),**{f"delta_{k}":float(deltas[i,j]) for j,k in enumerate(PARAM_COLS)}})
    fields=list(rows[0]) if rows else ["result_idx","canonical_spec_id"]
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    sf=sum(r["spectre_pass"] is False for r in rows); tf=sum(r["torch_pass"] is False for r in rows)
    disagree=sum(r["disagree"] for r in rows)
    md=["# CTLE closure summary","",f"Specs: **{n}**","","| Metric | Spectre knobs | PyTorch knobs |","|---|---:|---:|",f"| 10%-rule failures | {sf}/{n} ({100*sf/max(1,n):.2f}%) | {tf}/{n} ({100*tf/max(1,n):.2f}%) |",f"| Pass rate | {(n-sf)}/{n} ({100*(n-sf)/max(1,n):.2f}%) | {(n-tf)}/{n} ({100*(n-tf)/max(1,n):.2f}%) |","",f"Failure-label disagreements: **{disagree}/{n}**", "", "## Maximum absolute knob delta (physical units)", "", "| Knob | Max |", "|---|---:|"]
    md.extend(f"| {name} | {maxima[j]:.8g} |" for j,name in enumerate(PARAM_COLS))
    md += ["", "Per-spec pass/fail and knob deltas are in `summary.csv`.", ""]
    (out/"summary.md").write_text("\n".join(md))
    print(f"Spectre failure rate: {sf}/{n} ({100*sf/max(1,n):.2f}%)")
    print(f"Torch failure rate:   {tf}/{n} ({100*tf/max(1,n):.2f}%)")
    print(f"Disagreements: {disagree}/{n}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectre-results",required=True,type=Path)
    p.add_argument("--torch-results",required=True,type=Path)
    p.add_argument("--spectre-knobs",required=True,type=Path)
    p.add_argument("--torch-knobs",required=True,type=Path)
    p.add_argument("--spec-ids",type=Path,default=None,help="Optional one-column NPY not supported; pass specs.csv or omit.")
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(argv)
    compare(a.spectre_results,a.torch_results,None,a.spectre_knobs,a.torch_knobs,a.out)
    return 0
if __name__=="__main__": raise SystemExit(main())
