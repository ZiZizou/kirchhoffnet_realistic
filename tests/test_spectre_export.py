import json
from pathlib import Path
import pytest
import torch

from checkpoints import load_paper_checkpoint
from export import spectre_export
from export.netlist_interp import simulate

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "assets/baselines/paper_trial_0011/phase_a_best.pt"

@pytest.mark.slow
def test_paper_legacy_interpreter_matches_forward(tmp_path):
    model=load_paper_checkpoint(CKPT,dtype=torch.float64)
    spec=torch.tensor([[0.6,0.8,1.0,1.2],[1.2,0.9,0.7,1.1]],dtype=torch.float64)
    spectre_export.export(CKPT,tmp_path,mode="legacy",specs=2)
    got=simulate(tmp_path/"params.json",spec)["logits"]
    with torch.no_grad(): want=model(spec)
    err=(got-want).abs().max().item()
    assert err < 1e-12, f"max absolute error {err:.3g}"

def test_export_fabric_parameter_values_identical(tmp_path):
    tables=[]
    for fabric in ("timemux","copies"):
        out=tmp_path/fabric
        spectre_export.export(CKPT,out,mode="legacy",fabric=fabric,specs=1)
        tables.append(json.loads((out/"params.json").read_text()))
    assert tables[0]["state_dict"] == tables[1]["state_dict"]

def test_unmapped_state_key_fails(monkeypatch,tmp_path):
    class Fake:
        net=type("Net",(),{"config":{}})()
        def state_dict(self): return {"mystery_feature":torch.tensor(1.)}
    monkeypatch.setattr(spectre_export,"load_paper_checkpoint",lambda *a,**k:Fake())
    with pytest.raises(ValueError,match="unmapped state_dict key"):
        spectre_export.export("ignored",tmp_path)
