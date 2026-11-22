"""Strict state-dict exporter for the behavioural Spectre fabric."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from checkpoints import load_paper_checkpoint
from hw.units import HWUnits

# These tensors intentionally remain digital/off-chip: input log scaling and
# affine/post-readout transforms are performed by the external testbench.
DIGITAL_OFFCHIP_KEYS = [
    "input_log_min", "input_log_max", "log_lo", "log_hi", "net.output_mapper.*", "net.post_readout_transfer.*",
]

def _tensor(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    return v

def export(checkpoint, out, mode="legacy", fabric="timemux", specs=16, hw_config=None):
    if mode not in {"legacy", "hw"} or fabric not in {"timemux", "copies"}:
        raise ValueError("mode must be legacy|hw and fabric timemux|copies")
    model = load_paper_checkpoint(checkpoint, dtype=torch.float64)
    sd = model.state_dict()
    # Every key is accounted for. Topology tensors are netlist connectivity;
    # all remaining tensors are explicit behavioral parameters in the table.
    topology_leaves = {"src", "dst", "boundary_src", "boundary_dst", "output_ode_src", "output_ode_dst", "readout_sense_src"}
    param_leaves = {"clip_sharpness_raw","z_logits","boundary_z_logits","readout_sense_z_logits","readout_crossbar_W","raw_vref_sense","vca_W","vca_W_core","vca_v_boundary","vca_v_readout","vca_v_core","a_raw","b_raw","s_raw","gm_raw","isat_raw","theta_raw","g_resistive_raw","residual_w1","residual_w2","residual_w3","residual_vth","a","c","alpha_raw","P","Q"}
    entries, mapped, topology = {}, [], []
    for key, value in sd.items():
        leaf=key.rsplit(".",1)[-1]
        if key in DIGITAL_OFFCHIP_KEYS or key.startswith(("net.output_mapper.","net.post_readout_transfer.")):
            entries[key] = {"kind":"digital_offchip", "value":_tensor(value)}
        elif leaf in topology_leaves:
            topology.append(key); entries[key] = {"kind":"topology", "value":_tensor(value)}
        elif leaf in param_leaves and (key.startswith("net.core.stages.") or key.startswith("net.core.transfers.") or key.startswith("net.gln_rails.")):
            entries[key] = {"kind":"parameter", "value":_tensor(value)}; mapped.append(key)
        else:
            raise ValueError(f"unmapped state_dict key: {key}")
    missing = set(sd) - set(entries)
    if missing: raise ValueError(f"unmapped state_dict keys: {sorted(missing)}")
    cfg = getattr(model.net, "config", {})
    units = HWUnits()
    params = {"format":"hrknet-spectre-params-v1", "mode":mode, "fabric":fabric,
      "specs":int(specs), "units":{"Vu_V":units.vu,"Cu_F":units.cu,"Tu_s":units.tu,
      "Iu_A":units.iu,"Gu_S":units.gu,"Vcm_V":units.vcm,"x_hard":units.x_hard},
      "solver":{"name":"heun", "t_span":4.660583, "num_steps":256},
      "interstage_transfer":"instantaneous stage-boundary rewrite", "hw_config":hw_config,
      "state_dict":entries, "mapping":{"mapped_parameter_keys":mapped,"topology_keys":topology,
      "digital_offchip_keys":[k for k,v in entries.items() if v["kind"]=="digital_offchip"]}}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"params.json").write_text(json.dumps(params,indent=2)+"\n")
    va=Path(__file__).resolve().parents[1]/"veriloga"
    lines=[f'// generated mode={mode} fabric={fabric}', f'ahdl_include "{va}/hrknet_constants.vams"',
      f'ahdl_include "{va}/ota_edge.va"', 'simulator lang=spectre', 'parameters Vcm=0.4']
    for i in range(int(specs)):
      lines += [f'Vspec{i} (in{i} 0) vsource dc=0.4']
    for k,v in entries.items():
      if k.endswith("src") and v["kind"]=="topology":
        vals=v["value"]
        if isinstance(vals,list):
          for j,_ in enumerate(vals): lines.append(f'// edge {j}: topology indices are recorded in params.json')
    lines += ['tran tran stop=350n']
    (out/"hrknet.scs").write_text("\n".join(lines)+"\n")
    return params

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--hw-config")
    p.add_argument("--mode",choices=["legacy","hw"],default="legacy")
    p.add_argument("--fabric",choices=["timemux","copies"],default="timemux")
    p.add_argument("--specs",type=int,default=16); p.add_argument("--out",required=True)
    a=p.parse_args(); export(a.checkpoint,a.out,a.mode,a.fabric,a.specs,a.hw_config)
if __name__=="__main__": main()
