"""Standalone interpreter for resolved KirchhoffNet parameter tables.

Only JSON numerical data is consumed; importing this module never rebuilds a model.
"""
from __future__ import annotations
import json
import math
import numpy as np
import torch


def _table(table):
    if isinstance(table, (str, bytes)) or hasattr(table, "read_text"):
        text=table.read_text() if hasattr(table, "read_text") else open(table, encoding="utf-8").read()
        table=json.loads(text)
    return table.get("resolved", table)

def _t(value, ref):
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

def _gain(u, vca, family):
    if not vca["enabled"] or u is None:
        return None
    if family == "core":
        if not vca["core_enabled"] or vca["v_core"] is None:
            return None
        W = vca["W_core"] if vca["separate_core_bus"] else vca["W"]
        v = vca["v_core"]
        bias = vca["b_core"]
    else:
        v = vca["v_" + family]
        bias = vca["b_" + family]
        W = vca["W"]
    if v is None:
        return None
    z = u @ _t(W, u)
    logits = z @ _t(v, u).T
    if bias is not None:
        logits = logits + _t(bias, u)
    return 2.0 * torch.sigmoid(logits)

def _gln_gm(table, family, cell, u):
    if table.get("gln") is None or u is None or family not in table["gln"]["families"]:
        return _t(cell["gm"], u) if u is not None else None
    gln = table["gln"]
    if gln.get("mode") == "tail":
        return _t(cell["gm"],u)
    a, c, alpha = _t(gln["a"],u), _t(gln["c"],u), _t(gln["alpha"],u)
    z = torch.tanh(alpha * (u @ a.T + c))
    W = _t(gln["families"][family]["W"],u)
    gm = _t(cell["gm"],u) * torch.exp(z @ W.T)
    return gm.clamp(float(gln["gm_min"]),float(gln["gm_max"]))

def _cell(cell, src, dst, u, family, stage):
    ref = src
    A, B = _t(cell["A"],ref), _t(cell["B"],ref)
    sign, isat = _t(cell["sign"],ref), _t(cell["isat"],ref)
    theta = _t(cell["theta"],ref)
    gm = _gln_gm(stage, family, cell, u) if family in ("boundary","readout") else _t(cell["gm"],ref)
    if gm is None: gm = _t(cell["gm"],ref)
    isat = _t(cell["isat"],ref)
    gln = stage.get("gln")
    if (gln is not None and gln.get("mode") == "tail" and u is not None
            and family in gln.get("families", {})):
        a, c, alpha = _t(gln["a"],u), _t(gln["c"],u), _t(gln["alpha"],u)
        rails = torch.tanh(alpha * (u @ a.T + c))
        W = _t(gln["families"][family]["W"],u)
        delta = rails @ W.T
        scale = torch.exp(torch.clamp(delta, -float(gln["delta_max"]), float(gln["delta_max"])))
        isat = isat.unsqueeze(0) * scale
    z = (sign * (A * src - B * dst) + theta) * gm
    cur = isat * (torch.tanh(z) + float(cell["alpha"]) * z)
    gate_src = torch.sigmoid((float(stage["settings"]["x_max"]) - src.abs()) / float(cell["beta_softness"]))
    gate_dst = torch.sigmoid((float(stage["settings"]["x_max"]) - dst.abs()) / float(cell["beta_softness"]))
    cur = cur * gate_src * gate_dst
    G = _t(cell["conductance"],ref)
    return cur + G * (src - dst)

def _leg_factors(xn, units):
    vu, ut = float(units["vu"]), float(units["u_t"])
    hard, lam, k = float(units["x_hard"]), float(units["lambda_v"]), float(units["compliance_k"])
    scale = vu / ut
    y0 = hard * scale
    c0 = 1.0 - math.exp(-float(torch.nn.functional.softplus(torch.tensor(y0, dtype=xn.dtype), beta=k)))
    top = (1.0-lam*vu*xn).clamp_min(0.0) * (1.0-torch.exp(-torch.nn.functional.softplus((hard-xn)*scale, beta=k))) / c0
    bot = (1.0+lam*vu*xn).clamp_min(0.0) * (1.0-torch.exp(-torch.nn.functional.softplus((hard+xn)*scale, beta=k))) / c0
    return top, bot

def _legs(cell, src, dst, u, family, stage):
    gm = _gln_gm(stage, family, cell, u) if family in ("boundary", "readout") else _t(cell["gm"],src)
    if gm is None: gm = _t(cell["gm"],src)
    isat = _t(cell["isat"],src)
    gln = stage.get("gln")
    if gln is not None and gln.get("mode") == "tail" and u is not None and family in gln.get("families", {}):
        rails = torch.tanh(_t(gln["alpha"],u) * (u @ _t(gln["a"],u).T + _t(gln["c"],u)))
        delta = rails @ _t(gln["families"][family]["W"],u).T
        isat = isat.unsqueeze(0) * torch.exp(torch.clamp(delta, -float(gln["delta_max"]), float(gln["delta_max"])))
    z = (_t(cell["sign"],src) * (_t(cell["A"],src)*src-_t(cell["B"],src)*dst) + _t(cell["theta"],src))*gm
    t=torch.tanh(z)
    return 0.5*isat*(1+t), 0.5*isat*(1-t)

def _compliant_destination(ip, ineg, x, units):
    ft, fb = _leg_factors(x, units)
    return ip*ft-ineg*fb

def _compliant_source(ip, ineg, x, units):
    ft, fb = _leg_factors(x, units)
    return ineg*ft-ip*fb

def _scatter(acc, indices, values):
    tmp = torch.zeros_like(acc, dtype=torch.float32)
    tmp.index_add_(1, torch.as_tensor(indices, dtype=torch.long, device=acc.device), values.float())
    return tmp

def stage_rhs(table, stage_idx, x_live, x_held, u_inputs):
    """Evaluate one stage's ``dx/dt`` using live, held, and boundary inputs."""
    resolved = _table(table)
    st = resolved["stages"][int(stage_idx)]
    top, settings = st["topology"], st["settings"]
    x, u = x_live, u_inputs
    acc = torch.zeros_like(x, dtype=torch.float32)
    core_cell = st["cells"]["core"]
    core_gate = _t(st["gates"]["core"],x)
    core_vca = _gain(u,st["vca"],"core")
    held = x_held if x_held is not None and settings["freeze_read"] else x
    cs, cd = held[:,top["src"]], held[:,top["dst"]]
    units = resolved.get("units", {})
    compliance = bool(resolved.get("hw_flags", {}).get("hw_compliance", False))
    edge_acc = torch.zeros_like(x,dtype=torch.float32)
    if compliance:
        ip, inn = _legs(core_cell,cs,cd,u,"core",st)
        ip, inn = ip*core_gate.unsqueeze(0), inn*core_gate.unsqueeze(0)
        if core_vca is not None: ip, inn = ip*core_vca, inn*core_vca
        into_d = _compliant_destination(ip,inn,x[:,top["dst"]],units)
        edge_acc.index_add_(1,torch.as_tensor(top["dst"],dtype=torch.long,device=x.device),into_d.float())
        if not settings["read_only_source"]:
            into_s = _compliant_source(ip,inn,x[:,top["src"]],units)
            edge_acc.index_add_(1,torch.as_tensor(top["src"],dtype=torch.long,device=x.device),into_s.float())
    else:
        core_i = _cell(core_cell,cs,cd,u,"core",st) - _t(core_cell["conductance"],x) * (cs-cd)
        core_i = core_i * core_gate.unsqueeze(0)
        if core_vca is not None: core_i = core_i * core_vca
        edge_acc.index_add_(1,torch.as_tensor(top["dst"],dtype=torch.long,device=x.device),core_i.float())
        if not settings["read_only_source"]:
            edge_acc.index_add_(1,torch.as_tensor(top["src"],dtype=torch.long,device=x.device),-core_i.float())
    acc = edge_acc
    # Core shunt is dynamic even when the nonlinear cell currents are held.
    cs, cd = x[:,top["src"]], x[:,top["dst"]]
    ires = _t(core_cell["conductance"],x) * (cs-cd) * core_gate.unsqueeze(0)
    if core_vca is not None and st["vca"]["gate_shunt"]: ires = ires * core_vca
    res_acc = torch.zeros_like(x,dtype=torch.float32)
    res_acc.index_add_(1,torch.as_tensor(top["dst"],dtype=torch.long,device=x.device),ires.float())
    if not settings["read_only_source"]:
        res_acc.index_add_(1,torch.as_tensor(top["src"],dtype=torch.long,device=x.device),-ires.float())
    acc = (acc.float()+res_acc).to(dtype=x.dtype)

    if u is not None and top["boundary_src"]:
        src = u[:,top["boundary_src"]]
        dst = x[:,top["boundary_dst"]]
        cell=st["cells"]["boundary"]
        gate=_t(st["gates"]["boundary"],x).unsqueeze(0)
        vg=_gain(u,st["vca"],"boundary")
        if compliance:
            ip,inn=_legs(cell,src,dst,u,"boundary",st)
            ip,inn=ip*gate,inn*gate
            if vg is not None: ip,inn=ip*vg,inn*vg
            shunt=_t(cell["conductance"],x)*(src-dst)*gate
            if vg is not None: shunt=shunt*vg
            cur=_compliant_destination(ip,inn,dst,units)+shunt
        else:
            cur=_cell(cell,src,dst,u,"boundary",st)*gate
            if vg is not None: cur=cur*vg
        acc=(acc.float()+_scatter(x,top["boundary_dst"],cur)).to(dtype=x.dtype)

    # The paper checkpoint uses shared-sense crossbar readout.
    if st["settings"]["readout_mode"] == "shared_sense" and top["readout_sense_src"]:
        src=x[:,top["readout_sense_src"]]
        dst=_t(st["readout"]["vref"],x).reshape(1,1).expand_as(src)
        cell=st["cells"]["readout"]
        gate=_t(st["gates"]["readout"],x).unsqueeze(0)
        vg=_gain(u,st["vca"],"readout")
        if compliance:
            ip,inn=_legs(cell,src,dst,u,"readout",st)
            ip,inn=ip*gate,inn*gate
            if vg is not None: ip,inn=ip*vg,inn*vg
            shunt=_t(cell["conductance"],x)*(src-dst)*gate
            if vg is not None: shunt=shunt*vg
            cur=_compliant_destination(ip,inn,dst,units)+shunt
        else:
            cur=_cell(cell,src,dst,u,"readout",st)*gate
            if vg is not None: cur=cur*vg
        W=_t(st["readout"]["crossbar_W"],x)
        i_acc=cur.float() @ W.float().T
        if compliance:
            acc_start=int(top["readout_dst_start"])
            acc_x=x[:,acc_start:acc_start+i_acc.shape[1]]
            ft,fb=_leg_factors(acc_x,units)
            i_acc=torch.where(i_acc>=0,i_acc*ft,i_acc*fb)
        dst_start=int(top["readout_dst_start"])
        acc_f32=acc.float().clone()
        acc_f32[:,dst_start:dst_start+i_acc.shape[1]] += i_acc
        acc=acc_f32.to(dtype=x.dtype)

    leak=settings["leak"]
    leak=_t(leak,x)
    if leak.ndim==0: leak=leak.expand(x.shape[1])
    x_max=float(settings["x_max"])
    sharp=float(settings["clip_sharpness"])
    clip=(torch.zeros_like(x) if compliance else float(settings["clip_current"])*(torch.sigmoid((x-x_max)/sharp)-torch.sigmoid((-x-x_max)/sharp)))
    divisor = _t(settings["c_node"],x) if settings.get("c_node") is not None else float(settings["c_eff"])
    return (acc.to(dtype=x.dtype)-leak.unsqueeze(0)*x-clip)/divisor

def _transfer(x, t, clamp_x_hard=None):
    n=int(t["in_nodes"])
    if t["out_nodes"] < n: out=x[:,:int(t["out_nodes"])]
    elif t["out_nodes"] > n: out=torch.cat((x,x.new_zeros(x.shape[0],int(t["out_nodes"])-n)),dim=1)
    else: out=x
    act=t["activation"]
    if act == "residual-relu-tanh":
        out = _t(t["residual_w1"],out)*out+_t(t["residual_w2"],out)*torch.tanh(out)+_t(t["residual_w3"],out)*torch.relu(out-_t(t["residual_vth"],out))
    elif act == "relu":
        out = torch.relu(out)
    elif act in ("residual","residual_mixing"):
        out = _t(t["residual_w1"],out)*out+_t(t["residual_w2"],out)*torch.tanh(out)
    if clamp_x_hard is not None:
        out = torch.clamp(out.detach(), -float(clamp_x_hard), float(clamp_x_hard)) + (out - out.detach())
    return out

def _uniform_quant(x, bits, lo, hi):
    if bits is None or int(bits) <= 0 or hi <= lo:
        return x
    step = (float(hi) - float(lo)) / (2 ** int(bits) - 1)
    return torch.round((x.clamp(float(lo), float(hi)) - float(lo)) / step) * step + float(lo)

def _timemux_transfer(x, t, cfg, clamp_x_hard=None):
    mode = cfg.get("xfer_mode", "full")
    n = int(t["in_nodes"])
    if t["out_nodes"] < n: x = x[:, :int(t["out_nodes"])]
    elif t["out_nodes"] > n: x = torch.cat((x, x.new_zeros(x.shape[0], int(t["out_nodes"])-n)), dim=1)
    if any(t.get(k) is None for k in ("residual_w1", "residual_w2", "residual_w3", "residual_vth")):
        return _transfer(x, t, clamp_x_hard=clamp_x_hard)
    w1, w2, w3, vth = (_t(t[k], x) for k in ("residual_w1", "residual_w2", "residual_w3", "residual_vth"))
    def finish(y):
        return y if clamp_x_hard is None else y.clamp(-float(clamp_x_hard), float(clamp_x_hard))
    if mode == "shared_vth" and vth.numel() > 1:
        vth = vth.mean()
    elif mode == "quant":
        bits = cfg.get("param_bits")
        if bits:
            w1, w2, w3, vth = (_uniform_quant(v, bits, -2.0, 2.0) for v in (w1, w2, w3, vth))
    if mode == "no_tanh": return finish(w1*x + w3*torch.relu(x-vth))
    if mode == "no_relu": return finish(w1*x + w2*torch.tanh(x))
    if mode == "kick": return finish(x + (w1-1.0)*x + w2*torch.tanh(x) + w3*torch.relu(x-vth))
    if mode not in ("full", "shared_vth", "quant"):
        raise ValueError(f"unknown timemux xfer_mode {mode!r}")
    return finish(w1*x + w2*torch.tanh(x) + w3*torch.relu(x-vth))

def _timemux_boundary(x, cfg, units, boundary_idx, noise=None):
    if cfg.get("ktc", True) and noise is not None:
        x = x + _t(noise, x) * float(units.get("ktc_rms_units", 0.0))
    vu = float(units["vu"])
    if cfg.get("inject_mv", 0.0):
        gen = torch.Generator(device="cpu").manual_seed(12345 + int(boundary_idx))
        signs = torch.randint(0, 2, (x.shape[-1],), generator=gen, dtype=torch.float32).to(x.device)*2.0-1.0
        x = x + signs.to(dtype=x.dtype) * (float(cfg["inject_mv"])*1e-3/vu)
    if cfg.get("inject_gain_err", 0.0):
        x = x * (1.0 + float(cfg["inject_gain_err"]))
    x_transfer = x
    x_hold = x
    if cfg.get("droop_mv", 0.0):
        gen = torch.Generator(device="cpu").manual_seed(12346 + int(boundary_idx))
        signs = torch.randint(0, 2, (x.shape[-1],), generator=gen, dtype=torch.float32).to(x.device)*2.0-1.0
        x_hold = x + signs.to(dtype=x.dtype) * (float(cfg["droop_mv"])*1e-3/vu)
    bits = cfg.get("node_quant_bits")
    if bits:
        rail = float(units["x_hard"])
        x_transfer = _uniform_quant(x_transfer, bits, -rail, rail)
        x_hold = _uniform_quant(x_hold, bits, -rail, rail)
    return x_hold, x_transfer

def simulate(table, x, noise=None):
    resolved=_table(table)
    x=torch.as_tensor(x)
    u=(2.0*(torch.log10(x.clamp(min=1e-12))-_t(resolved["input"]["log_min"],x))/(_t(resolved["input"]["log_max"],x)-_t(resolved["input"]["log_min"],x)).clamp(min=1e-8)-1.0).clamp(min=-float(resolved["input"]["input_rail"]),max=float(resolved["input"]["input_rail"]))
    batch=x.shape[0]
    state=x.new_zeros((batch,int(resolved["stages"][0]["num_nodes"])))
    finals=[]
    tm=resolved.get("timemux")
    tm_enabled=bool(tm and tm.get("enable", False))
    held_input=None
    for i,st in enumerate(resolved["stages"]):
        t=float(st["solver"]["t_span"]); steps=int(st["solver"]["num_steps"]); dt=t/steps
        held=held_input if tm_enabled and i > 0 else state
        for _ in range(steps):
            k1=stage_rhs(resolved,i,state,held,u)
            pred=state+dt*k1
            k2=stage_rhs(resolved,i,pred,held,u)
            state=state+0.5*dt*(k1+k2)
        finals.append(state)
        if i<len(resolved["transfers"]):
            clamp = resolved.get("units", {}).get("x_hard") if resolved.get("hw_flags", {}).get("hw_clamp_transfer") else None
            if tm_enabled:
                noise_i = None if noise is None else noise[i]
                held_sample, transfer_sample = _timemux_boundary(state, tm, resolved.get("units", {}), i, noise_i)
                tr = resolved["transfers"][i]
                state = _timemux_transfer(transfer_sample, tr, tm, clamp_x_hard=clamp)
                held_input = _timemux_transfer(held_sample, tr, tm, clamp_x_hard=clamp)
            else:
                state=_transfer(state,resolved["transfers"][i],clamp_x_hard=clamp)
    top=resolved["topology"]
    start=int(resolved["stages"][-1]["topology"]["readout_dst_start"])
    count=int(resolved["topology"]["readout_count"])
    y=state[:,start:start+count]
    rd=resolved["readout"]
    y=_t(rd["gain"],y)*y+_t(rd["bias"],y)+_t(rd["tanh_gain"],y)*torch.tanh(y)+_t(rd["relu_gain"],y)*torch.relu(y-_t(rd["relu_threshold"],y))
    y=_transfer(y,{"in_nodes":y.shape[1],"out_nodes":y.shape[1],**rd["post_transfer"]})
    return {"logits":y,"stage_states":finals}
