"""Strict state-dict exporter for the behavioural Spectre fabric."""
from __future__ import annotations
import argparse, json
from torch.utils._python_dispatch import TorchDispatchMode
from pathlib import Path
import torch
from checkpoints import load_paper_checkpoint
from hw.units import HWUnits

# Explicit digital/off-chip exceptions. Each reason describes work performed
# outside the behavioral network represented by the resolved table.
DIGITAL_OFFCHIP_KEYS = [
    "input_log_min",  # Input-domain log scaling is applied by the testbench.
    "input_log_max",  # Input-domain log scaling is applied by the testbench.
    "log_lo",         # Legacy input log bounds are testbench preprocessing.
    "log_hi",         # Legacy input log bounds are testbench preprocessing.
    "net.output_mapper.*",  # Output affine/nonlinear mapping is off-chip.
    "net.post_readout_transfer.*",  # Post-readout residual transfer is off-chip.
    "net.core.stages.*.cell_lib.a_raw",  # Legacy A is unused when cap-divider A/B mode is active.
    "net.core.stages.*.cell_lib.b_raw",  # Legacy B is unused when cap-divider A/B mode is active.
    "net.core.stages.*.boundary_cell_lib.a_raw",  # Same retained legacy parameter in boundary cells.
    "net.core.stages.*.boundary_cell_lib.b_raw",  # Same retained legacy parameter in boundary cells.
    "net.core.stages.*.readout_sense_cell_lib.a_raw",  # Same retained legacy parameter in readout cells.
    "net.core.stages.*.readout_sense_cell_lib.b_raw",  # Same retained legacy parameter in readout cells.
]

def _tensor(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    return v

def _sigmoid(x):
    return torch.sigmoid(x.detach()).cpu().tolist()

def _softplus(x):
    return torch.nn.functional.softplus(x.detach()).cpu().tolist()

def _cell_table(lib):
    A, B = lib.ab_values()
    theta = lib.theta_values()
    constraints = getattr(lib, "hw_constraints", None)
    alpha = 0.0 if constraints is not None and constraints.hw_no_alpha else float(lib.ALPHA)
    result = {
        "A": _tensor(A), "B": _tensor(B),
        "sign": _tensor(torch.sign(lib.s_raw)),
        "gm": _tensor(lib.gm_eff_values()), "isat": _tensor(lib.isat_values()),
        "theta": _tensor(torch.zeros_like(lib.s_raw) if theta is None else theta),
        "conductance": _tensor(torch.nn.functional.softplus(lib.g_resistive_raw)),
        "gm_min": float(lib.gm_min), "gm_max": float(lib.gm_max),
        "isat_min": float(lib.isat_min), "isat_max": float(lib.isat_max),
        "alpha": alpha, "beta_softness": float(lib._beta_softness),
        "has_bias": theta is not None, "has_resistive": hasattr(lib, "resistive_current"),
    }
    if constraints is not None and constraints.hw_ab_mode == "cap_divider":
        levels = 2 ** int(constraints.hw_ab_bits) - 1
        result["ab_codes"] = {
            "k_A": torch.round(torch.sigmoid(lib.cap_a_raw.detach()) * levels).to(torch.int64).cpu().tolist(),
            "k_B": torch.round(torch.sigmoid(lib.cap_b_raw.detach()) * levels).to(torch.int64).cpu().tolist(),
            "levels": levels, "rmax": float(constraints.hw_ab_rmax)}
    return result

def _physical_cell(cell, units):
    """Physical-unit companion; normalized cell values remain interpreter inputs."""
    return {"A_hat": cell["A"], "B_hat": cell["B"],
            "Isat_eff_A": (torch.as_tensor(cell["isat"], dtype=torch.float64) * units.iu).tolist(),
            "G_eff_S": (torch.as_tensor(cell["conductance"], dtype=torch.float64) * units.gu).tolist(),
            "gm_eff_S": (torch.as_tensor(cell["gm"], dtype=torch.float64) * units.gu).tolist(),
            "theta_V": (torch.as_tensor(cell["theta"]) * units.vu).tolist()}

def _build_param_table(model, mode="legacy", hw=None) -> dict:
    """Return a JSON-ready, resolved behavioral description of a network.

    Accepts any model exposing the repository's KirchhoffNetWithIO interface;
    ``hw`` may be an HWConfig or a constraints-like object.  The model's own
    resolved cell values remain authoritative because those are what forward uses.
    """
    if mode not in {"legacy", "hw"}:
        raise ValueError("mode must be legacy|hw")
    net = model.net
    from hw.constraints import net_hw
    built_hw = net_hw(net)
    if hw is not None and not getattr(hw, "is_off", lambda: False)():
        if built_hw is None:
            raise ValueError("hw config was supplied, but the model was not built with that HWConfig; export cannot retrofit flags")
        if hasattr(hw, "to_dict") and hasattr(built_hw, "to_dict") and hw.to_dict() != built_hw.to_dict():
            raise ValueError("supplied HWConfig differs from the config used to build the model")
    if hw is None:
        hw = built_hw
    if mode == "hw" and hw is not None:
        hw.validate() if hasattr(hw, "validate") else hw.constraints.validate()
    constraints = getattr(hw, "constraints", hw)
    hw_flags = constraints.to_dict() if constraints is not None and hasattr(constraints, "to_dict") else {}
    units = getattr(hw, "units", HWUnits())
    if mode == "hw" and constraints is not None:
        # Stochastic mismatch draws and training-only objectives cannot be
        # represented by one deterministic inference table.
        if constraints.mismatch_active():
            raise ValueError("export does not support hw_mismatch_sigma: stochastic mismatch draws")
        if constraints.hw_power_lambda:
            raise ValueError("export does not support hw_power_lambda: training-only power objective")
        if constraints.hw_vca_bits is not None:
            raise ValueError("export does not support hw_vca_bits: quantized VCA logits are not resolved")
        refused = {
            "hw_isat_eff_clamp": constraints.hw_isat_eff_clamp,
        }
        active = [name for name, enabled in refused.items() if enabled]
        if active:
            raise ValueError("export does not support forward-affecting flags: " + ", ".join(active))
    stages = []
    for i, s in enumerate(net.core.stages):
        n = int(s.num_nodes)
        sharp = s.clip_sharpness()
        sharp = float(sharp.detach().cpu()) if isinstance(sharp, torch.Tensor) else float(sharp)
        gln = net.gln_rails
        gln_table = None
        if gln is not None:
            gln_table = {"a": _tensor(gln.a), "c": _tensor(gln.c),
                         "alpha": _tensor(torch.nn.functional.softplus(gln.alpha_raw) + 1e-6),
                         "mode": None if constraints is None else constraints.hw_gln_mode,
                         "delta_max": None if constraints is None else float(constraints.hw_gln_delta_max),
                         "gm_min": gln.gm_min, "gm_max": gln.gm_max,
                         "families": {name: {"W": _tensor(m.weight())} for name,m in gln.families.items()}}
        vca = {"enabled": bool(s.vca_enabled), "core_enabled": bool(s._vca_core_enabled),
               "separate_core_bus": bool(s.vca_separate_core_bus), "gate_shunt": bool(s.vca_gate_shunt),
               "W": _tensor(s.vca_W) if s.vca_W is not None else None,
               "W_core": _tensor(s.vca_W_core) if s.vca_W_core is not None else None,
               "v_core": _tensor(s.vca_v_core) if s.vca_v_core is not None else None,
               "v_boundary": _tensor(s.vca_v_boundary) if s.vca_v_boundary is not None else None,
               "v_readout": _tensor(s.vca_v_readout) if s.vca_v_readout is not None else None,
               "b_core": _tensor(s.vca_b_core) if s.vca_b_core is not None else None,
               "b_boundary": _tensor(s.vca_b_boundary) if s.vca_b_boundary is not None else None,
               "b_readout": _tensor(s.vca_b_readout) if s.vca_b_readout is not None else None}
        core_cell = _cell_table(s.cell_lib)
        boundary_cell = _cell_table(s.boundary_cell_lib) if s.boundary_cell_lib else None
        readout_cell = _cell_table(s.readout_sense_cell_lib) if s.readout_sense_cell_lib else None
        output_cell = _cell_table(s.output_ode_cell_lib) if s.output_ode_cell_lib else None
        stage_table = {
            "index": i, "num_nodes": n,
            "topology": {"src": _tensor(s.src), "dst": _tensor(s.dst),
                "boundary_src": _tensor(s.boundary_src), "boundary_dst": _tensor(s.boundary_dst),
                "output_ode_src": _tensor(s.output_ode_src), "output_ode_dst": _tensor(s.output_ode_dst),
                "readout_sense_src": _tensor(s.readout_sense_src),
                "readout_dst_start": int(s._readout_dst_start)},
            "solver": {"method":"heun", "t_span": float(net.core.stage_times[i]), "num_steps":int(net.core.stage_steps[i])},
            "settings": {"c_eff":float(s.c_eff), "c_node":_tensor(s.node_capacitance(torch.zeros((1,n), dtype=s.z_logits.dtype, device=s.z_logits.device))) if constraints is not None and constraints.hw_c_node is not None else None, "x_max":float(s.x_max), "clip_current":float(s.clip_current),
                "clip_sharpness":sharp, "leak":_tensor(s._effective_leak()),
                "freeze_read":bool(s.freeze_read), "freeze_boundary":bool(s.freeze_boundary),
                "freeze_temporal_read":bool(s.freeze_temporal_read), "read_only_source":bool(s.read_only_source),
                "readout_mode":s.readout_mode, "leak_mode":s.leak_mode},
            "cells":{"core":core_cell, "boundary":boundary_cell, "readout":readout_cell,
                "output_ode":output_cell},
            "gates":{"core":_sigmoid(s.z_logits), "boundary":_sigmoid(s.boundary_z_logits) if s.boundary_z_logits is not None else [],
                "readout":_sigmoid(s.readout_sense_z_logits) if s.readout_sense_z_logits is not None else [],
                "output_ode":_sigmoid(s.output_ode_z_logits) if s.output_ode_z_logits is not None else []},
            "readout":{"crossbar_W":_tensor(s.crossbar_weights()), "vref":_tensor(s.sense_vref())},
            "vca":vca, "gln":gln_table}
        if mode == "hw":
            node_cap = s.node_capacitance(torch.zeros((1,n), dtype=s.z_logits.dtype, device=s.z_logits.device))
            node_cap = torch.as_tensor(_tensor(node_cap)).reshape(-1)
            if node_cap.numel() == 1:
                node_cap = node_cap.expand(n)
            stage_table["physical"] = {
                "cells": {name: (_physical_cell(cell, units) if cell is not None else None)
                          for name, cell in stage_table["cells"].items()},
                "C_node_F": (node_cap * units.cu).tolist(),
                "leak_S": (torch.as_tensor(_tensor(s._effective_leak())).reshape(-1) * units.gu).tolist(),
                "Vref_V": (units.vcm + torch.as_tensor(stage_table["readout"]["vref"]) * units.vu).tolist(),
                "clip_current_A": float(s.clip_current) * units.iu,
                "t_span_s": float(net.core.stage_times[i]) * units.tu,
                "dt_s": float(net.core.stage_times[i]) / int(net.core.stage_steps[i]) * units.tu,
                "A_hat": core_cell["A"], "B_hat": core_cell["B"],
            }
        stages.append(stage_table)
    if mode == "hw" and constraints is not None and constraints.hw_compliance:
        if any(getattr(s, "_has_output_ode", False) for s in net.core.stages):
            raise ValueError("export does not support hw_compliance with temporal output mesh; "
                             "the standalone interpreter only resolves shared-sense readout")
    transfers=[]
    for t in net.core.transfers:
        transfers.append({"in_nodes":t.in_nodes,"out_nodes":t.out_nodes,"activation":t.activation,
            "residual_w1":_tensor(t.residual_w1) if t.residual_w1 is not None else None,
            "residual_w2":_tensor(t.residual_w2) if t.residual_w2 is not None else None,
            "residual_w3":_tensor(t.residual_w3) if t.residual_w3 is not None else None,
            "residual_vth":_tensor(t.residual_vth) if t.residual_vth is not None else None})
    post=net.post_readout_transfer
    affine=net.output_mapper
    return {"schema":"kirchhoffnet-resolved-params-v1", "mode":mode, "hw_flags":hw_flags,
        "units":{"x_hard":float(units.x_hard), "vu":float(units.vu), "u_t":float(units.u_t), "temp_c":float(units.temp_c), "cu":float(units.cu), "ktc_rms_units":float(units.ktc_rms_units), "n_slope":float(units.n_slope), "lambda_v":0.15, "compliance_k":2000.0},
        "timemux": (None if getattr(hw, "timemux", None) is None else hw.timemux.to_dict()),
        "input":{"kind":"log10_minmax", "log_min":_tensor(model.input_log_min), "log_max":_tensor(model.input_log_max),
            "input_rail":float(model.input_rail), "initial_state":"zeros", "boundary_enabled":bool(net.enable_boundary)},
        "topology":{"stages":len(stages), "hidden_nodes":int(net.hid_count), "readout_start":int(net.read_start),
            "readout_count":int(net.read_dim), "write_idx":net.write_idx, "read_idx":net.read_idx},
        "stages":stages, "transfers":transfers,
        "readout":{"kind":"output_affine", "gain":_tensor(affine.gain), "bias":_tensor(affine.bias),
            "tanh_gain":_tensor(affine.tanh_gain), "relu_gain":_tensor(affine.relu_gain),
            "relu_threshold":_tensor(affine.relu_threshold),
            "post_transfer":{"activation":post.activation,"residual_w1":_tensor(post.residual_w1),"residual_w2":_tensor(post.residual_w2),
                "residual_w3":_tensor(post.residual_w3),"residual_vth":_tensor(post.residual_vth)}}}

class _StateReadTracker(TorchDispatchMode):
    """Record state-dict tensors participating in builder tensor operations."""
    def __init__(self, state_dict):
        super().__init__()
        self.keys_by_ptr = {}
        self.consumed = set()
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                self.keys_by_ptr.setdefault(value.data_ptr(), set()).add(key)

    def _observe(self, value):
        if isinstance(value, torch.Tensor):
            self.consumed.update(self.keys_by_ptr.get(value.data_ptr(), ()))
        elif isinstance(value, (tuple, list)):
            for item in value: self._observe(item)
        elif isinstance(value, dict):
            for item in value.values(): self._observe(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self._observe(args)
        self._observe(kwargs or {})
        return func(*args, **(kwargs or {}))


def _build_param_table_tracked(model, mode="legacy", hw=None):
    state = model.state_dict(keep_vars=True)
    tracker = _StateReadTracker(state)
    with tracker:
        table = _build_param_table(model, mode, hw)
    return table, tracker.consumed


def build_param_table(model, mode="legacy", hw=None) -> dict:
    """Build a resolved table while preserving the public table-only API."""
    return _build_param_table_tracked(model, mode, hw)[0]


def export_model(model, out, mode="legacy", fabric="timemux", specs=16, hw_config=None, hw=None):
    """Export an already-built KirchhoffNetWithIO (paper or custom architecture)."""
    if mode not in {"legacy", "hw"} or fabric not in {"timemux", "copies"}:
        raise ValueError("mode must be legacy|hw and fabric timemux|copies")
    from fnmatch import fnmatchcase
    if not hasattr(model, "net") or not hasattr(model.net, "core"):
        for key in model.state_dict():
            if not any(fnmatchcase(key, pattern) for pattern in DIGITAL_OFFCHIP_KEYS):
                raise ValueError(f"unmapped state_dict key: {key}")
        raise TypeError("model must expose the KirchhoffNetWithIO .net.core interface")
    sd = model.state_dict(keep_vars=True)
    resolved, consumed = _build_param_table_tracked(model, mode, hw)
    # Every serialized tensor is consumed by the builder, used as topology, or
    # explicitly declared digital/off-chip below.
    topology_leaves = {"src", "dst", "boundary_src", "boundary_dst", "output_ode_src", "output_ode_dst", "readout_sense_src"}
    entries, mapped, topology = {}, [], []
    for key, value in sd.items():
        leaf=key.rsplit(".",1)[-1]
        always_digital = any(fnmatchcase(key, pattern) for pattern in DIGITAL_OFFCHIP_KEYS[:6])
        if always_digital or (key not in consumed and any(fnmatchcase(key, pattern) for pattern in DIGITAL_OFFCHIP_KEYS[6:])):
            entries[key] = {"kind":"digital_offchip", "value":_tensor(value)}
        elif key in consumed and leaf in topology_leaves:
            topology.append(key); entries[key] = {"kind":"topology", "value":_tensor(value)}
        elif key in consumed:
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
      "solver":{"name":"heun", "t_span":float(sum(model.net.core.stage_times)), "num_steps":int(sum(model.net.core.stage_steps))},
      "interstage_transfer":"instantaneous stage-boundary rewrite", "hw_config":hw_config,
      "resolved":resolved,
      "state_dict":entries, "mapping":{"mapped_parameter_keys":mapped,"topology_keys":topology,
      "digital_offchip_keys":[k for k,v in entries.items() if v["kind"]=="digital_offchip"]}}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"params.json").write_text(json.dumps(params,indent=2)+"\n")
    if mode == "legacy" and fabric == "copies":
      from .spectre_netlist import render
      render(params, out/"hrknet.scs")
    else:
      va=Path(__file__).resolve().parents[1]/"veriloga"
      lines=[f'// generated mode={mode} fabric={fabric}', f'ahdl_include "{va}/hrknet_constants.vams"',
        f'ahdl_include "{va}/ota_edge.va"', 'simulator lang=spectre', 'parameters Vcm=0.4']
      for i in range(int(specs)):
        lines += [f'Vspec{i} (in{i} 0) vsource dc=0.4']
      lines += ['tran tran stop=350n']
      (out/"hrknet.scs").write_text("\n".join(lines)+"\n")
    return params

def export(checkpoint, out, mode="legacy", fabric="timemux", specs=16, hw_config=None):
    """Paper-checkpoint compatibility wrapper around :func:`export_model`."""
    model = load_paper_checkpoint(checkpoint, dtype=torch.float64)
    hw = None
    if hw_config:
        from hw import HWConfig
        hw = HWConfig.from_json(hw_config)
    return export_model(model, out, mode, fabric, specs, hw_config, hw)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--mode",choices=["legacy","hw"],default="legacy")
    p.add_argument("--fabric",choices=["timemux","copies"],default="timemux")
    p.add_argument("--specs",type=int,default=16); p.add_argument("--out",required=True)
    from hw import parse_hw_args, hw_config_from_args
    parse_hw_args(p)
    a=p.parse_args(); cfg=hw_config_from_args(a)
    model=load_paper_checkpoint(a.checkpoint,dtype=torch.float64)
    export_model(model,a.out,a.mode,a.fabric,a.specs,cfg.to_dict(),cfg)
if __name__=="__main__": main()
