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
    if params.get("mode") != "legacy" or params.get("fabric") != "copies":
        raise ValueError("Spectre network writer supports only legacy/copies")
    resolved = params.get("resolved", params)
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


def _write_banked_crossbar(out, sense_count=22, output_count=7):
    """Generate a seven-bank crossbar VA sized for the resolved readout."""
    va = Path(__file__).resolve().parents[1] / "veriloga"
    ports = [f"s{k}p,s{k}n" for k in range(sense_count)]
    ports += [f"acc{j}" for j in range(output_count)]
    ports += ["stage", "en"]
    declarations = [f"  input s{k}p,s{k}n; electrical s{k}p,s{k}n;" for k in range(sense_count)]
    declarations += [f"  inout acc{j}; electrical acc{j};" for j in range(output_count)]
    declarations.append("  input stage,en; electrical stage,en;")
    wlen = sense_count * output_count
    terms = []
    for j in range(output_count):
        terms.append(f"    i_acc[{j}]=0.0;" + "".join(
            f" i_acc[{j}]=i_acc[{j}]+W[k*{wlen}+{j*sense_count+kidx}]*I(s{kidx}p,s{kidx}n);"
            for kidx in range(sense_count)) + f" I(acc{j}) <+ -eg*i_acc[{j}];")
    body = ["`include \"constants.vams\"", "`include \"disciplines.vams\"",
            f'`include "{va}/hrknet_constants.vams"',
            f"module crossbar_accumulator_banked_{sense_count}x{output_count}({','.join(ports)});",
            *declarations, f"  parameter real W[0:{7*wlen-1}]='{{" + ",".join("0.0" for _ in range(7*wlen)) + "};", 
            "  integer k; real eg;", f"  real i_acc[0:{output_count-1}];", "  analog begin",
            "    k=$rtoi(V(stage)+0.5); if(k<0) k=0; if(k>6) k=6;",
            "    eg=1.0/(1.0+limexp(-(V(en)-0.5)/0.01));"]
    body.extend(f"    V(s{k}p,s{k}n)<+0.0;" for k in range(sense_count))
    body.extend(terms)
    body += ["  end", "endmodule", ""]
    path = Path(out).parent / f"crossbar_accumulator_banked_{sense_count}x{output_count}.va"
    path.write_text("\n".join(body))
    return path


def render_timemux(params, out, inputs=None, stages=None, transient=True):
    """Render one shared 18-node, seven-bank legacy fabric.

    Boundary transfers capture the live fabric into S&H first, compute the
    coefficient-bank rewrite into isolated target nodes, and clamp only during
    a recurring win pulse. Each stage then evolves for its configured span.
    """
    if params.get("mode") != "legacy":
        raise ValueError("time-multiplexed writer currently supports legacy mode")
    resolved = params.get("resolved", params)
    n_specs = int(params.get("specs", 1))
    u_rows = _spec_inputs(resolved, n_specs, inputs)
    units = params.get("units", {})
    vu = float(units.get("Vu_V", 0.07)); vcm = float(units.get("Vcm_V", 0.4))
    tu = float(units.get("Tu_s", 350e-9/4.660583))
    all_stages = resolved["stages"]
    selected = list(range(len(all_stages))) if stages is None else list(stages)
    if selected != list(range(len(all_stages))):
        raise ValueError("timemux requires the full ordered stage bank")
    if len(all_stages) != 7 or any(int(s["num_nodes"]) != 18 for s in all_stages):
        raise ValueError("timemux expects the canonical seven-stage, 18-node network")
    va = Path(__file__).resolve().parents[1] / "veriloga"
    includes = ("sample_hold", "ota_edge_banked", "ota_edge_sampled_banked", "shunt_banked",
                "boundary_sense_ota_banked", "node_cap_leak_banked", "rail_clip_banked",
                "interstage_transfer_banked", "window_clamp")
    lines = ["// Legacy time-multiplexed fabric: shared 18-node state; seven parameter banks.",
             "// Boundary timing: en falls at each stage-span boundary; boundary S&H captures",
             "// immediately, transfer clamp opens after capture and closes before stage switch.",
             "// en rises after switch; each evolve interval is exactly t_span[k]*Tu.",
             "simulator lang=spectre", f'ahdl_include "{va}/hrknet_constants.vams"']
    lines += [f'ahdl_include "{va}/{m}.va"' for m in includes]
    max_sense = max(len(s["topology"]["readout_sense_src"]) for s in all_stages)
    max_outputs = max(len(s["readout"]["crossbar_W"]) for s in all_stages)
    crossbar = _write_banked_crossbar(Path(out), max_sense, max_outputs)
    lines.append(f'ahdl_include "{crossbar}"')
    lines += ["simulator lang=spectre", f"Vcm_ref (vcm_ref 0) vsource dc={_fmt(vcm)}"]
    spans = [float(s["solver"]["t_span"]) * tu for s in all_stages]
    # Per-stage controls have explicit sub-microsecond transfer windows. The
    # ODE-active time itself remains exactly spans[k]; boundary overhead is not
    # counted as evolution.
    capture = min(2e-9, min(spans) / 100.0)
    xfer = min(2e-9, min(spans) / 100.0)
    settle = min(0.2e-9, capture / 10.0)
    boundary_overhead = capture + xfer + settle
    boundary_times = [sum(spans[:k+1]) + k * boundary_overhead for k in range(6)]
    evolve_starts = [0.0]
    for k, tb in enumerate(boundary_times):
        evolve_starts.append(tb + capture + xfer + settle)
    stop = evolve_starts[-1] + spans[-1]
    def pwl(points):
        return "type=pwl wave=[" + " ".join(f"{_fmt(t)} {_fmt(v)}" for t,v in points) + "]"
    def step_points(points):
        """Expand control transitions into near-vertical PWL steps."""
        stepped = [points[0]]
        for edge, level in points[1:]:
            previous_time, previous_level = stepped[-1]
            delta = min(1e-12, (edge - previous_time) / 2.0)
            stepped.extend(((edge - delta, previous_level), (edge, level)))
        return stepped
    for si, u in enumerate(u_rows):
        for b, uv in enumerate(u):
            lines.append(f"V{si}_input{b} (in{si}_{b} 0) vsource dc={_fmt(vcm+vu*uv)}")
        st_points = [(0.0, 0.0)]
        en_points = [(0.0, 1.0)]
        win_points = [(0.0, 0.0)]
        xfer_points = [(0.0, 0.0)]
        for k, tb in enumerate(boundary_times):
            begin = tb
            cap_end = begin + capture
            xfer_end = cap_end + xfer
            active = xfer_end + settle
            en_points += [(begin, 0.0), (active, 1.0)]
            st_points += [(begin, float(k)), (xfer_end, float(k+1))]
            win_points += [(cap_end, 1.0), (xfer_end, 0.0)]
            xfer_points += [(cap_end, 1.0), (xfer_end, 0.0)]
        st_points.append((stop, 6.0)); en_points.append((stop, 1.0)); win_points.append((stop, 0.0))
        xfer_points.append((stop, 0.0))
        for name, pts in (("stage", st_points), ("en", en_points), ("win", win_points), ("xfer_en", xfer_points)):
            lines.append(f"V{si}_{name} ({name}{si} 0) vsource {pwl(step_points(pts))}")
        # Stage-state devices are banked on the shared fabric nodes.
        for node in range(18):
            node_name=f"n{si}_{node}"
            caps=[]; leaks=[]; xmax=[]; sharp=[]; iclip=[]
            for st in all_stages:
                setting=st["settings"]
                caps.append(float(setting["c_eff"]))
                leaks.append(_at(setting["leak"], node))
                xmax.append(float(setting["x_max"])); sharp.append(float(setting["clip_sharpness"]))
                iclip.append(float(setting["clip_current"]))
            lines.append(_inst(f"q{si}_cap{node}",[node_name,f"stage{si}",f"en{si}"],"node_cap_leak_banked",c_eff="["+" ".join(map(_fmt,caps))+"]",g_leak="["+" ".join(map(_fmt,leaks))+"]"))
            lines.append(_inst(f"q{si}_clip{node}",[node_name,f"stage{si}",f"en{si}"],"rail_clip_banked",x_max="["+" ".join(map(_fmt,xmax))+"]",sharp="["+" ".join(map(_fmt,sharp))+"]",Iclip="["+" ".join(map(_fmt,iclip))+"]"))
            lines.append(f"ic {node_name}={_fmt(vcm)}")
            # Boundary-start sample for transfer; independent evolve-start
            # sample for freeze_read so neither overwrites the other.
            bc_clock=[(0.0,0.0)]
            for tb in boundary_times: bc_clock += [(tb+settle,1.0),(tb+settle+capture,0.0)]
            read_clock=[(0.0,0.0)]
            for t in evolve_starts: read_clock += [(t+settle,1.0),(t+settle+capture,0.0)]
            lines.append(f"Vbc{si}_{node} (bcclk{si}_{node} 0) vsource {pwl(bc_clock)}")
            lines.append(f"Vrc{si}_{node} (rclk{si}_{node} 0) vsource {pwl(read_clock)}")
            lines.append(_inst(f"q{si}_bhold{node}",[node_name,f"bhold{si}_{node}",f"bcclk{si}_{node}"],"sample_hold",C_hold=1e-15,R_on=1e3,R_off=1e12))
            lines.append(_inst(f"q{si}_rhold{node}",[node_name,f"rhold{si}_{node}",f"rclk{si}_{node}"],"sample_hold",C_hold=1e-15,R_on=1e3,R_off=1e12))
            trw1=[_at(t["residual_w1"],node) for t in resolved["transfers"]]
            trw2=[_at(t["residual_w2"],node) for t in resolved["transfers"]]
            trw3=[_at(t["residual_w3"],node) for t in resolved["transfers"]]
            trth=[_at(t["residual_vth"],node) for t in resolved["transfers"]]
            # Transfer coefficient arrays are indexed by source stages 0..5.
            lines.append(f"Ctarget{si}_{node} (target{si}_{node} 0) capacitor c=1f")
            lines.append(_inst(f"q{si}_xfer{node}",[f"bhold{si}_{node}",f"target{si}_{node}",f"stage{si}",f"xfer_en{si}"],"interstage_transfer_banked",w1="["+" ".join(map(_fmt,trw1))+"]",w2="["+" ".join(map(_fmt,trw2))+"]",w3="["+" ".join(map(_fmt,trw3))+"]",vth="["+" ".join(map(_fmt,trth))+"]",x_hard=float(resolved["units"]["x_hard"])))
            lines.append(_inst(f"q{si}_clamp{node}",[node_name,f"target{si}_{node}",f"win{si}"],"window_clamp",g_clamp=1.0))
        # Build the banked edge cells; topology is common across seven paper stages.
        first=all_stages[0]["topology"]
        for edge,(src,dst) in enumerate(zip(first["src"],first["dst"])):
            cells=[s["cells"]["core"] for s in all_stages]
            gates=[_at(s["gates"]["core"],edge) for s in all_stages]
            vcas=[]
            for s in all_stages:
                gain=_gain(torch.tensor([u],dtype=torch.float64),s["vca"],"core")
                vcas.append(1.0 if gain is None else float(gain[0,edge]))
            def bank(key): return "["+" ".join(_fmt(_at(c[key],edge)) for c in cells)+"]"
            lines.append(_inst(f"q{si}_e{edge}",[f"rhold{si}_{src}",f"rhold{si}_{dst}",f"n{si}_{src}",f"n{si}_{dst}",f"stage{si}",f"en{si}"],"ota_edge_sampled_banked",A=bank("A"),B=bank("B"),s=bank("sign"),theta=bank("theta"),gm=bank("gm"),Isat=bank("isat"),z_gate="["+" ".join(map(_fmt,gates))+ "]",vca="["+" ".join(map(_fmt,vcas))+ "]",x_max=float(all_stages[0]["settings"]["x_max"]),gate_width=float(cells[0]["beta_softness"]),floor_term=float(cells[0]["alpha"])))
            sh=[_at(c["conductance"],edge) for c in cells]
            if any(sh):
                shgate=[_at(s["gates"]["core"],edge) for s in all_stages]
                shvca=[]
                for s in all_stages:
                    gain=_gain(torch.tensor([u],dtype=torch.float64),s["vca"],"core")
                    shvca.append(1.0 if gain is None else float(gain[0,edge]))
                lines.append(_inst(f"q{si}_sh{edge}",[f"n{si}_{src}",f"n{si}_{dst}",f"stage{si}",f"en{si}"],"shunt_banked",G="["+" ".join(map(_fmt,sh))+ "]",z_gate="["+" ".join(map(_fmt,shgate))+ "]",vca="["+" ".join(map(_fmt,shvca))+ "]",vca_gate_shunt=int(all_stages[0]["vca"]["gate_shunt"])))
        for edge,(src,dst) in enumerate(zip(first["boundary_src"],first["boundary_dst"])):
            bcs=[s["cells"]["boundary"] for s in all_stages]
            vals={k:[] for k in ("gm","Isat","G","A","B","theta","s")}; gates=[]; vcas=[]
            for s,bc in zip(all_stages,bcs):
                inp=torch.tensor([u],dtype=torch.float64)
                vals["gm"].append(float(_gln_gm(s,"boundary",bc,inp)[0,edge]))
                for k,source in (("Isat","isat"),("G","conductance"),("A","A"),("B","B"),("theta","theta"),("s","sign")): vals[k].append(0.0 if k == "G" else _at(bc[source],edge))
                gain=_gain(inp,s["vca"],"boundary")
                vcas.append(1.0 if gain is None else float(gain[0,edge]))
                gates.append(_at(s["gates"]["boundary"],edge)*(1.0 if gain is None else float(gain[0,edge])))
            srcnode=f"in{si}_{src}"
            lines.append(_inst(f"q{si}_b{edge}",[srcnode,f"n{si}_{dst}",f"stage{si}",f"en{si}"],"boundary_sense_ota_banked",**{k:"["+" ".join(map(_fmt,v))+"]" for k,v in vals.items()},z_gate="["+" ".join(map(_fmt,gates))+"]",alpha=float(bcs[0]["alpha"])))
            bg=[_at(bc["conductance"],edge) for bc in bcs]
            if any(bg):
                bgvca=[_gain(torch.tensor([u],dtype=torch.float64),s["vca"],"boundary") for s in all_stages]
                bgmul=[1.0 if g is None else float(g[0,edge]) for g in bgvca]
                bggate=[_at(s["gates"]["boundary"],edge) for s in all_stages]
                lines.append(_inst(f"q{si}_bsh{edge}",[srcnode,f"n{si}_{dst}",f"stage{si}",f"en{si}"],"shunt_banked",G="["+" ".join(map(_fmt,bg))+"]",z_gate="["+" ".join(map(_fmt,bggate))+"]",vca="["+" ".join(map(_fmt,bgmul))+"]",vca_gate_shunt=1))
        # Banked readout sense and crossbar; Vref sources follow the stage select.
        for sense,src in enumerate(first["readout_sense_src"]):
            rcs=[s["cells"]["readout"] for s in all_stages]; inp=torch.tensor([u],dtype=torch.float64)
            rvals={k:[] for k in ("gm","Isat","G","A","B","theta","s")}; rg=[]
            refs=[]
            for s,rc in zip(all_stages,rcs):
                rvals["gm"].append(float(_gln_gm(s,"readout",rc,inp)[0,sense]))
                for k,source in (("Isat","isat"),("G","conductance"),("A","A"),("B","B"),("theta","theta"),("s","sign")): rvals[k].append(_at(rc[source],sense))
                gain=_gain(inp,s["vca"],"readout")
                rg.append(_at(s["gates"]["readout"],sense)*(1.0 if gain is None else float(gain[0,sense])))
                refs.append(vcm+vu*_at(s["readout"]["vref"],sense))
            refpts=[(0.0,refs[0])]
            for k,t in enumerate(boundary_times): refpts.append((evolve_starts[k+1],refs[k+1]))
            lines.append(f"Vref{si}_{sense} (ref{si}_{sense} 0) vsource {pwl(refpts)}")
            lines.append(_inst(f"q{si}_r{sense}",[f"n{si}_{src}",f"sense{si}_{sense}",f"stage{si}",f"en{si}"],"boundary_sense_ota_banked",**{k:"["+" ".join(map(_fmt,v))+"]" for k,v in rvals.items()},z_gate="["+" ".join(map(_fmt,rg))+"]",alpha=float(rcs[0]["alpha"])))
        weights=[]
        for st in all_stages:
            for j in range(max_outputs):
                row=st["readout"]["crossbar_W"][j] if j<len(st["readout"]["crossbar_W"]) else []
                weights.extend(float(row[k]) if k<len(row) else 0.0 for k in range(max_sense))
        xbports=[]
        for k in range(max_sense): xbports += [f"sense{si}_{k}",f"ref{si}_{k}"]
        acc=int(first["readout_dst_start"])
        xbports += [f"n{si}_{acc+j}" if j<max_outputs else "0" for j in range(max_outputs)]
        xbports += [f"stage{si}",f"en{si}"]
        lines.append(f"q{si}_xb ({' '.join(xbports)}) crossbar_accumulator_banked_{max_sense}x{max_outputs} W=["+" ".join(_fmt(v) for v in weights)+"]")
    dtmax = min(spans[k]/int(all_stages[k]["solver"]["num_steps"]) for k in range(7))/100
    lines += ["simulatorOptions options reltol=1e-8 vabstol=1e-12 iabstol=1e-18",
              f"tran tran stop={_fmt(stop if transient else tu)} maxstep={_fmt(dtmax)} errpreset=conservative",
              "saveOptions options save=allpub",
              "save "+" ".join(f"n{s}_{j}" for s in range(n_specs) for j in range(18))]
    text="\n".join(lines)+"\n"
    Path(out).write_text(text)
    return text
