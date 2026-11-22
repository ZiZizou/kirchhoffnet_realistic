import sys, importlib.util, argparse, torch, numpy as np
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"]); args.kn_solver_scale=1.0; args.student_kind="knet"
m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"]))
m.load_state_dict(b["model"]); m.eval()
c=m.net.core; s=c.stages[0]
print("stage_times",c.stage_times,"stage_steps",c.stage_steps)
for a in ["freeze_read","freeze_boundary","freeze_temporal_read","vca_enabled","_vca_core_enabled","vca_separate_core_bus","vca_gate_shunt","c_eff","x_max","node_activation","_has_boundary","_has_ref","_has_output_ode","_has_shared_readout","_has_resistive","read_only_source","core_refresh_interval","clip_softness","clip_current","_learnable_clip_sharpness","budget_enabled","dynamic_leak_enabled","leak_floor","vca_use_hidden","vca_b_boundary","vca_b_readout","vca_b_core"]:
    print(a,getattr(s,a,"<none>"))
print("gm range",s.cell_lib.gm_min,s.cell_lib.gm_max,"isat",s.cell_lib.isat_min,s.cell_lib.isat_max, "bias_enabled",s.cell_lib._bias_enabled)
print("leak",s._effective_leak(num_nodes=21) if True else None)
print("src",s.src.tolist()); print("dst",s.dst.tolist()); print("bsrc",s.boundary_src.tolist(),"bdst",s.boundary_dst.tolist())
print("out src",s.output_ode_src[:20].tolist(),"dst",s.output_ode_dst[:20].tolist())
print("num_nodes",s.num_nodes, "gln",getattr(s,"gln_rails",None), "tap", s.tap_rails)
print("core hid_count",m.net.hid_count,m.net.proj_count,m.net.output_ode_count,"read_slice",m.net.read_slice)
print("params",sum(p.numel() for p in m.parameters()))
print(m.net.parameter_breakdown())
