"""Torch reference interpreter for exported parameter tables.

The JSON is the sole parameter source; no checkpoint is opened here.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from checkpoints import PaperStudent

def simulate(params_json, specs):
    table=json.loads(Path(params_json).read_text())
    state={k:torch.as_tensor(v["value"]) for k,v in table["state_dict"].items()
           if v["kind"] != "topology" or k in PaperStudent().state_dict()}
    model=PaperStudent().to(dtype=torch.float64)
    # Topology is also in the exported table and restored from it above.
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unsupported exported entries: {result.unexpected_keys}")
    model.eval()
    with torch.no_grad():
        logits=model(torch.as_tensor(specs,dtype=torch.float64))
    return {"logits":logits,"readout":logits,"offchip_affine":None}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("params_json"); p.add_argument("specs_json")
    a=p.parse_args(); print(simulate(a.params_json,json.loads(Path(a.specs_json).read_text())))
