import json
from pathlib import Path
import pytest
import torch

from checkpoints import load_paper_checkpoint
from export import spectre_export
from export.netlist_interp import simulate, stage_rhs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "assets/baselines/paper_trial_0011/phase_a_best.pt"

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


def test_stage_rhs_matches_model_call_path():
    model=load_paper_checkpoint(CKPT,dtype=torch.float64)
    table=json.loads(json.dumps(spectre_export.build_param_table(model)))
    torch.manual_seed(814)
    u=model.scale_input(torch.rand(5,4,dtype=torch.float64))
    for stage_idx in (0,3):
        stage=model.net.core.stages[stage_idx]
        x_live=torch.randn(5,stage.num_nodes,dtype=torch.float64)
        x_held=torch.randn(5,stage.num_nodes,dtype=torch.float64)
        gate=stage.edge_gain("core",u)
        frozen=stage._compute_i_edge_const(x_held,gate)
        want=stage.rhs(x_live,u=u,i_edge_const=frozen,vca_gate_core=gate)
        got=stage_rhs(table,stage_idx,x_live,x_held,u)
        err=(got-want).abs().max().item()
        assert err <= 1e-9, f"stage {stage_idx} max absolute error {err:.3g}"


def test_standalone_interp_matches_paper_forward():
    model=load_paper_checkpoint(CKPT,dtype=torch.float64)
    table=json.loads(json.dumps(spectre_export.build_param_table(model)))
    # Fixed canonical physical-domain rows (positive component values).
    x=torch.tensor([
        [0.001,0.010,0.10,1.00], [0.002,0.020,0.20,1.20],
        [0.004,0.030,0.30,1.50], [0.010,0.060,0.60,1.80],
        [0.030,0.100,0.90,2.00], [0.060,0.200,1.10,2.20],
        [0.100,0.400,1.30,2.40], [0.200,0.800,1.50,2.60],
    ],dtype=torch.float64)
    captured=[]
    hooks=[]
    for stage in model.net.core.stages:
        hooks.append(stage.register_forward_hook(lambda _m,_i,out: captured.append(out[0].detach().clone())))
    with torch.no_grad():
        want=model(x)
    for hook in hooks: hook.remove()
    got=simulate(table,x)
    err=(got["logits"]-want).abs().max().item()
    assert err <= 1e-6, f"logit max absolute error {err:.3g}"
    state_err=max((a-b).abs().max().item() for a,b in zip(got["stage_states"],captured))
    assert state_err <= 1e-6, f"per-stage final-state max absolute error {state_err:.3g}"
