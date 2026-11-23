"""Spectre writer for the legacy copied-fabric network.

The crossbar is tiled from the checked-in fixed 4x3 Verilog-A primitive.
Input-dependent VCA and GLN gains are resolved in Python once per inference
specification and passed as scalar Verilog-A instance parameters.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .netlist_interp import _gain, _gln_gm, _t


def _flat(v):
    if isinstance(v, (list, tuple)):
        return v
    return [v]


def _at(v, i):
    a = _flat(v)
    return float(a[i if len(a) > 1 else 0])


def _fmt(v):
    return f"{float(v):.16g}"


def _inst(name, nodes, module, **params):
    p = " ".join(f"{k}={_fmt(v) if isinstance(v, (float, int)) else v}" for k, v in params.items())
    return f"{name} ({' '.join(nodes)}) {module}" + (f" {p}" if p else "")


def _spec_inputs(resolved, count, inputs):
    if inputs is None:
        return [[0.0] * len(resolved["input"]["log_min"]) for _ in range(count)]
    rows = torch.as_tensor(inputs, dtype=torch.float64)
    if rows.ndim == 1:
        rows = rows.unsqueeze(0)
    if rows.shape[0] != count:
        raise ValueError(f"expected {count} specs, got {rows.shape[0]}")
    log_min = _t(resolved["input"]["log_min"], rows)
    log_max = _t(resolved["input"]["log_max"], rows)
    u = (2 * (torch.log10(rows.clamp(min=1e-12)) - log_min) / (log_max-log_min).clamp(min=1e-8)-1).clamp(-float(resolved["input"]["input_rail"]), float(resolved["input"]["input_rail"]))
    return u.tolist()


def render(params, out, inputs=None, stages=None, transient=True):
    """Render a legacy copies deck. ``inputs`` are the raw physical spec rows."""
    mode = params.get("mode", "legacy")
    if mode not in {"legacy", "hw"} or params.get("fabric") != "copies":
        raise ValueError("Spectre network writer supports legacy|hw copies")
    resolved = params.get("resolved", params)
    if mode == "hw":
        # First validated hardware slice: phase3a_core only. Keep this an
        # explicit allowlist; unsupported forward-affecting flags must never
        # silently fall through to legacy equations.
        flags = resolved.get("hw_flags", {})
        ab_mode = flags.get("hw_ab_mode")
        if ab_mode == "cap_divider":
            expected = {"hw_slope_bound": True, "hw_ic_coupling": True,
                        "hw_no_alpha": True, "hw_ab_mode": "cap_divider",
                        "hw_ab_bits": 3, "hw_ab_rmax": 2.0}
            preset_name = "phase3a_core"
        elif ab_mode == "differential":
            expected = {"hw_slope_bound": True, "hw_ic_coupling": True,
                        "hw_no_alpha": True, "hw_ab_mode": "differential"}
            preset_name = "phase3a_core_diff"
        else:
            raise ValueError("hw netlist currently supports phase3a_core and phase3a_core_diff only")
        for key, value in expected.items():
            if flags.get(key) != value:
                raise ValueError(f"hw netlist currently requires {preset_name} {key}={value!r}")
        allowed = set(expected)
        defaults = {"hw_ab_mode": None, "hw_ab_bits": 3, "hw_ab_rmax": 2.0,
                    "hw_slope_g_min": 0.0625, "hw_xbar_mode": "signed",
                    "hw_gln_delta_max": 1.3862943611198906,
                    "hw_c_node": None, "hw_gln_mode": None,
                    "hw_mismatch_sigma": None, "hw_vca_bits": None,
                    "hw_vref_bits": None, "hw_xbar_bits": None}
        active = [k for k, v in flags.items() if k not in allowed and v != defaults.get(k, False)]
        if active:
            raise ValueError("hw netlist does not support flags: " + ", ".join(sorted(active)))
    n_specs = int(params.get("specs", 1))
    u_rows = _spec_inputs(resolved, n_specs, inputs)
    units = params.get("units", {})
    vu = float(units.get("Vu_V", 0.07)); vcm = float(units.get("Vcm_V", 0.4))
    tu = float(units.get("Tu_s", 350e-9/4.660583)); dtmax = tu * min(float(s["solver"]["t_span"])/int(s["solver"]["num_steps"]) for s in resolved["stages"]) / 100
    selected = range(len(resolved["stages"])) if stages is None else list(stages)
    va = Path(__file__).resolve().parents[1] / "veriloga"
    lines = ["// Legacy copies deck: sequential stage releases use cumulative t_span; earlier copies continue evolving but are not read again.", "simulator lang=spectre", f'ahdl_include "{va}/hrknet_constants.vams"']
    for module in ("ota_edge", "shunt", "node_cap_leak", "rail_clip", "sample_hold", "boundary_sense_ota", "interstage_transfer", "phase_ctrl", "stage_gate", "ota_edge_sampled"):
        lines.append(f'ahdl_include "{va}/{module}.va"')
    lines += ["parameters Vcm=0.4", "simulator lang=spectre", f"Vcm_ref (vcm_ref 0) vsource dc={_fmt(vcm)}"]
    readout_stages=[resolved["stages"][i] for i in selected]
    sense_count=max(len(s["topology"]["readout_sense_src"]) for s in readout_stages)
    output_count=max(len(s["readout"]["crossbar_W"]) for s in readout_stages)
    sense_ports=[f"s{k}p,s{k}n" for k in range(sense_count)]
    acc_ports=[f"acc{j}" for j in range(output_count)]
    ports=",".join(sense_ports+acc_ports)
    decl="\n".join([f"  input s{k}p,s{k}n; electrical s{k}p,s{k}n;" for k in range(sense_count)]+[f"  inout acc{j}; electrical acc{j};" for j in range(output_count)])
    wlen=sense_count*output_count
    sums="\n".join([f"    V(s{k}p,s{k}n) <+ 0.0; isense[{k}]=I(s{k}p,s{k}n);" for k in range(sense_count)])
    calc="\n".join([f"    i_acc[{j}]=0.0;"+"".join([f" i_acc[{j}]=i_acc[{j}]+W[{j*sense_count+k}]*isense[{k}];" for k in range(sense_count)])+f" I(acc{j}) <+ -i_acc[{j}];" for j in range(output_count)])
    wdefault="'{"+",".join("0.0" for _ in range(wlen))+"}"
    crossbar_va=chr(10).join(['`include "constants.vams"', '`include "disciplines.vams"', f'`include "{va}/hrknet_constants.vams"', f"module crossbar_accumulator_{sense_count}x{output_count}({ports});", decl, f"  parameter real W[0:{wlen-1}]={wdefault};", f"  real isense[0:{sense_count-1}],i_acc[0:{output_count-1}];", "  analog begin", sums, calc, "  end", "endmodule", ""])
    crossbar_path=Path(out).parent/f"crossbar_accumulator_{sense_count}x{output_count}.va"
    crossbar_path.write_text(crossbar_va)
    lines.append(f'ahdl_include "{crossbar_path}"')
    for spec_i, u in enumerate(u_rows):
        for b, uv in enumerate(u):
            lines.append(f"V{spec_i}_input{b} (in{spec_i}_{b} 0) vsource dc={_fmt(vcm+vu*uv)}")
        for stage_idx in selected:
            st = resolved["stages"][stage_idx]; top = st["topology"]; setting=st["settings"]
            prefix=f"q{spec_i}_s{stage_idx}_"; n=int(st["num_nodes"])
            for node in range(n):
                node_name=f"n{spec_i}_{stage_idx}_{node}"
                lines.append(_inst(prefix+f"cap{node}",[node_name],"node_cap_leak",c_eff=float(setting["c_eff"]),g_leak=_at(setting["leak"],node)))
                lines.append(_inst(prefix+f"clip{node}",[node_name],"rail_clip",x_max=float(setting["x_max"]),sharp=float(setting["clip_sharpness"]),Iclip=float(setting["clip_current"])))
                if stage_idx == 0:
                    lines.append(f"ic {node_name}={_fmt(vcm)}")
                else:
                    prev=resolved["stages"][stage_idx-1]
                    tr=resolved["transfers"][stage_idx-1]
                    src_node=f"n{spec_i}_{stage_idx-1}_{node}" if node<int(prev["num_nodes"]) else "vcm_ref"
                    release=sum(float(resolved["stages"][j]["solver"]["t_span"]) for j in range(stage_idx))*tu
                    activation={"residual-relu-tanh":0,"relu":1,"residual":2,"residual_mixing":2}.get(tr["activation"],0)
                    lines.append(_inst(prefix+f"gate{node}",[node_name,src_node],"stage_gate",w1=_at(tr["residual_w1"],node),w2=_at(tr["residual_w2"],node),w3=_at(tr["residual_w3"],node),vth=_at(tr["residual_vth"],node),activation=activation,x_hard=float(resolved["units"]["x_hard"]),release_s=release,smooth_s=max(tu*1e-8,1e-13),g_clamp=1.0))
            core=st["cells"]["core"]
            gains=_gain(torch.tensor([u],dtype=torch.float64),st["vca"],"core")
            vca_core=None if gains is None else gains[0].tolist()
            held_nodes=[]
            if setting["freeze_read"]:
                release_time=sum(float(resolved["stages"][j]["solver"]["t_span"]) for j in range(stage_idx))*tu
                if stage_idx:
                    # Acquire before the stage is released so the sample tracks
                    # the final transferred state and settles for 20 RC constants.
                    capture_delay=max(1e-12,release_time-21e-12); capture_width="20p"
                else:
                    capture_delay=1e-12; capture_width="2p"
                for node in range(n):
                    hold=f"hold{spec_i}_{stage_idx}_{node}"; held_nodes.append(hold)
                    lines.append(f"Vclk{spec_i}_{stage_idx}_{node} (clk{spec_i}_{stage_idx}_{node} 0) vsource type=pulse val0=0 val1=1 delay={_fmt(capture_delay)} rise=1p fall=1p width={capture_width} period=1e30")
                    lines.append(_inst(prefix+f"fh{node}",[f"n{spec_i}_{stage_idx}_{node}",hold,f"clk{spec_i}_{stage_idx}_{node}"],"sample_hold",C_hold=1e-15,R_on=1e3,R_off=1e12))
            else:
                held_nodes=[f"n{spec_i}_{stage_idx}_{i}" for i in range(n)]
            for edge,(src,dst) in enumerate(zip(top["src"],top["dst"])):
                gate=_at(st["gates"]["core"],edge); vca=1.0 if vca_core is None else float(vca_core[edge])
                if setting["freeze_read"]:
                    ota_nodes=[held_nodes[src],held_nodes[dst],f"n{spec_i}_{stage_idx}_{src}",f"n{spec_i}_{stage_idx}_{dst}"]
                    ota_module="ota_edge_sampled"
                    ota_params=dict(A=_at(core["A"],edge),B=_at(core["B"],edge),s=_at(core["sign"],edge),theta=_at(core["theta"],edge),gm=_at(core["gm"],edge),Isat=_at(core["isat"],edge),z_gate=gate,sigma_z=1.0,vca=vca,x_max=float(setting["x_max"]),gate_width=float(core["beta_softness"]),floor_term=float(core["alpha"]))
                else:
                    ota_nodes=[held_nodes[src],held_nodes[dst]]; ota_module="ota_edge"
                    ota_params=dict(mode=0,A=_at(core["A"],edge),B=_at(core["B"],edge),s=_at(core["sign"],edge),theta=_at(core["theta"],edge),gm=_at(core["gm"],edge),Isat=_at(core["isat"],edge),z_gate=gate,sigma_z=1.0,vca=vca,x_max=float(setting["x_max"]),gate_width=float(core["beta_softness"]),floor_term=float(core["alpha"]))
                if mode == "hw":
                    if flags.get("hw_ab_mode") == "cap_divider":
                        codes=core.get("ab_codes")
                        if not codes:
                            raise ValueError("phase3a_core hardware export requires resolved cap-divider codes")
                        ab_params=dict(ab_mode=3,
                                       cap_code_a=int(codes["k_A"][edge]),
                                       cap_code_b=int(codes["k_B"][edge]),
                                       cap_bits=int(codes["levels"]).bit_length(),
                                       cap_rmax=float(codes["rmax"]))
                    else:
                        ab_params=dict(ab_mode=2)
                    hw_params=dict(mode=1, gm=1.0,
                                   g_hw=_at(core["gm"],edge), f_ic=1.0,
                                   floor_term=0.0, **ab_params)
                    ota_params.update(hw_params)
                lines.append(_inst(prefix+f"e{edge}",ota_nodes,ota_module,**ota_params))
                g=_at(core["conductance"],edge)
                if g:
                    lines.append(_inst(prefix+f"sh{edge}",[f"n{spec_i}_{stage_idx}_{src}",f"n{spec_i}_{stage_idx}_{dst}"],"shunt",G=g,z_gate=gate,vca=vca,vca_gate_shunt=int(st["vca"]["gate_shunt"])))
            # Boundary OTAs use the physical input sources and live destination.
            bc=st["cells"]["boundary"]
            vg=_gain(torch.tensor([u],dtype=torch.float64),st["vca"],"boundary")
            bvg=None if vg is None else vg[0].tolist()
            for edge,(src,dst) in enumerate(zip(top["boundary_src"],top["boundary_dst"])):
                cell_gm=float(_gln_gm(st,"boundary",bc,torch.tensor([u],dtype=torch.float64))[0,edge])
                gate=_at(st["gates"]["boundary"],edge); vca=1.0 if bvg is None else float(bvg[edge])
                p=dict(gm=cell_gm,Isat=_at(bc["isat"],edge),A=_at(bc["A"],edge),B=_at(bc["B"],edge),theta=_at(bc["theta"],edge),s=_at(bc["sign"],edge),z_gate=gate*vca,alpha=float(bc["alpha"]),x_max=float(setting["x_max"]),gate_width=float(bc["beta_softness"]))
                lines.append(_inst(prefix+f"b{edge}",[f"in{spec_i}_{src}",f"n{spec_i}_{stage_idx}_{dst}"],"boundary_sense_ota",**p))
                g=_at(bc["conductance"],edge)
                if g: lines.append(_inst(prefix+f"bsh{edge}",[f"in{spec_i}_{src}",f"n{spec_i}_{stage_idx}_{dst}"],"shunt",G=g,z_gate=gate*vca,vca=1.0,vca_gate_shunt=1))
            # Shared-sense OTAs feed one sized crossbar generated from the
            # checked-in 4x3 primitive contract (same current summation law).
            rc=st["cells"]["readout"]; sense_src=top["readout_sense_src"]
            rv=_gain(torch.tensor([u],dtype=torch.float64),st["vca"],"readout")
            rvg=None if rv is None else rv[0].tolist()
            sense_nodes=[]
            for sense,src in enumerate(sense_src):
                sn=f"sense{spec_i}_{stage_idx}_{sense}"; sense_nodes.append(sn)
                vref=_at(st["readout"]["vref"],sense)
                ref=f"sense{spec_i}_{stage_idx}_{sense}n"
                lines.append(f"{ref} ({ref} 0) vsource dc={_fmt(vcm+vu*vref)}")
                gm=float(_gln_gm(st,"readout",rc,torch.tensor([u],dtype=torch.float64))[0,sense]); gate=_at(st["gates"]["readout"],sense); vca=1 if rvg is None else float(rvg[sense])
                lines.append(_inst(prefix+f"r{sense}",[f"n{spec_i}_{stage_idx}_{src}",sn],"boundary_sense_ota",gm=gm,Isat=_at(rc["isat"],sense),G=_at(rc["conductance"],sense),A=_at(rc["A"],sense),B=_at(rc["B"],sense),theta=_at(rc["theta"],sense),s=_at(rc["sign"],sense),z_gate=gate*vca,alpha=float(rc["alpha"]),x_max=float(setting["x_max"]),gate_width=float(rc["beta_softness"])))
            weights=st["readout"]["crossbar_W"]; acc_start=int(top["readout_dst_start"])
            xb_ports=[]
            for k in range(sense_count):
                if k<len(sense_nodes): xb_ports += [sense_nodes[k],f"sense{spec_i}_{stage_idx}_{k}n"]
                else: xb_ports += ["0","0"]
            for j in range(output_count):
                xb_ports.append(f"n{spec_i}_{stage_idx}_{acc_start+j}" if j<len(weights) else "0")
            w=[float(weights[j][k]) if j<len(weights) and k<len(weights[j]) else 0.0 for j in range(output_count) for k in range(sense_count)]
            warr="["+",".join(_fmt(x) for x in w)+"]"
            lines.append(f"{prefix}xb ({' '.join(xb_ports)}) crossbar_accumulator_{sense_count}x{output_count} W={warr}")
    stop=sum(float(resolved["stages"][i]["solver"]["t_span"]) for i in selected)*tu if transient else tu
    lines.append("simulatorOptions options reltol=1e-8 vabstol=1e-12 iabstol=1e-18")
    lines.append(f"tran tran stop={_fmt(stop)} maxstep={_fmt(dtmax)} errpreset=conservative")
    lines.append("saveOptions options save=allpub")
    lines.append("save " + " ".join(f"n{s}_{i}_{node}" for s in range(n_specs) for i in selected for node in range(int(resolved["stages"][i]["num_nodes"]))))
    text="\n".join(lines)+"\n"
    Path(out).write_text(text)
    return text
