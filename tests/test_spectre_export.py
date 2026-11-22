import ast
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


def test_unconsumed_known_name_fails(monkeypatch,tmp_path):
    class Fake:
        net=type("Net",(),{"config":{}})()
        def state_dict(self): return {"net.core.stages.0.unused.z_logits":torch.tensor(1.)}
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


def test_hw_physical_view_round_trips_to_dimensionless_values():
    from hw.units import HWUnits
    model=load_paper_checkpoint(CKPT,dtype=torch.float64)
    table=spectre_export.build_param_table(model,mode="hw")
    units=HWUnits()
    for stage in table["stages"]:
        dim=stage["cells"]["core"]
        phys=stage["physical"]["cells"]["core"]
        assert torch.allclose(torch.tensor(phys["Isat_eff_A"],dtype=torch.float64)/units.iu,
                              torch.tensor(dim["isat"],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        assert torch.allclose(torch.tensor(phys["G_eff_S"],dtype=torch.float64)/units.gu,
                              torch.tensor(dim["conductance"],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        assert torch.allclose(torch.tensor(phys["gm_eff_S"],dtype=torch.float64)/units.gu,
                              torch.tensor(dim["gm"],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        assert torch.allclose(torch.tensor(stage["physical"]["C_node_F"])/units.cu,
                              torch.full((stage["num_nodes"],),float(model.net.core.stages[stage["index"]].c_eff)),
                              atol=1e-12,rtol=1e-12)

def test_interpreter_has_no_model_imports():
    source=(ROOT/"export/netlist_interp.py").read_text()
    tree=ast.parse(source)
    forbidden={"checkpoints","differential_stage","cell_library","gating","config","kirchhoff_net"}
    imported=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (forbidden & imported), f"forbidden model imports: {sorted(forbidden & imported)}"


@pytest.mark.parametrize("preset,overrides", [
    ("phase3a_core", {}),
    ("phase3a_core_diff", {}),
    ("phase3a_core", {"hw_vref_symmetric": True, "hw_vref_bits": 6}),
    ("phase3a_core", {"hw_xbar_bound": True, "hw_xbar_bits": 5, "hw_xbar_mode": "signed"}),
    ("phase3a_core", {"hw_clamp_transfer": True}),
    ("phase3a_core", {"hw_c_node": "physical"}),
    ("phase3a_core", {"hw_gln_mode": "tail", "hw_gln_delta_max": 1.3862943611198906}),
    ("phase3a_full", {"hw_compliance": False}),
])
def test_toy_net_hw_interpreter_roundtrip(tmp_path, preset, overrides):
    """The resolved table must reproduce the actual constrained TOY_NET forward."""
    import copy
    import sys
    from hw import HWConfig, load_preset
    from tests.golden.make_golden import TOY_NET, make_model, CtleStudent, _build_net, seed_all

    config = copy.deepcopy(load_preset(preset))
    config["constraints"].update(overrides)
    cfg = HWConfig.from_dict(config)
    spec = copy.deepcopy(TOY_NET)
    seed_all(20261122)
    net = _build_net(spec, hw=cfg).double()
    model = CtleStudent(net, 4, 3, 4.0).double().eval()
    # Small fixed perturbations ensure gates, signs and capacitor codes are not
    # left at their symmetric initialization points.
    gen = torch.Generator().manual_seed(7319)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            p.add_(0.015 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    table = json.loads(json.dumps(spectre_export.build_param_table(model, mode="hw", hw=cfg)))
    inputs = torch.tensor([[0.08, 0.2, 0.7, 1.4], [0.12, 0.4, 0.9, 1.8],
                           [0.16, 0.5, 1.1, 2.2]], dtype=torch.float64)
    captured, hooks = [], []
    for stage in model.net.core.stages:
        hooks.append(stage.register_forward_hook(lambda _m, _i, out: captured.append(out[0].detach().clone())))
    with torch.no_grad():
        want = model(inputs)
    for hook in hooks:
        hook.remove()
    got = simulate(table, inputs)
    out_err = (got["logits"] - want).abs().max().item()
    state_err = max((a - b).abs().max().item() for a, b in zip(got["stage_states"], captured))
    print(f"{preset}: output max error={out_err:.3g}, final-state max error={state_err:.3g}")
    assert out_err <= 1e-6, f"{preset} output max error {out_err:.3g}"
    assert state_err <= 1e-6, f"{preset} state max error {state_err:.3g}"
