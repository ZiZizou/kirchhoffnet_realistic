import sys, importlib.util, argparse, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"]); args.kn_solver_scale=1.0; args.student_kind="knet"
m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"])); m.load_state_dict(b["model"]); m.eval()
S=m.net.core.stages
print("boundary_cell_lib shared object across stages:", all(S[0].boundary_cell_lib is s.boundary_cell_lib for s in S))
print("boundary a_raw identical across stage state_dict:", all(torch.equal(b["model"][f"net.core.stages.{i}.boundary_cell_lib.a_raw"],b["model"]["net.core.stages.0.boundary_cell_lib.a_raw"]) for i in range(4)))
for i,s in enumerate(S):
    print(f"stage{i} core a_raw==0 all: {bool((s.cell_lib.a_raw==0).all())}, b_raw==0 all: {bool((s.cell_lib.b_raw==0).all())}; |a_raw|max {s.cell_lib.a_raw.abs().max():.3f}; readout frac a_raw==0 {(s.output_ode_cell_lib.a_raw==0).float().mean():.2f}")
# sign +1 fraction detail
print("sum edges/stage",56+8+98,"x4 =",4*162)
# shunt-only and u-offset: share of current that is constant (theta-driven) in stage0 core
s=S[0]; 
with torch.no_grad():
    gm=s.cell_lib.gm_min+(s.cell_lib.gm_max-s.cell_lib.gm_min)*torch.sigmoid(s.cell_lib.gm_raw)
    isat=s.cell_lib.isat_min+(s.cell_lib.isat_max-s.cell_lib.isat_min)*torch.sigmoid(s.cell_lib.isat_raw)
    i0=isat*(torch.tanh(gm*s.cell_lib.theta_raw)+0.01*gm*s.cell_lib.theta_raw)*torch.sigmoid(s.z_logits)
print("stage0 core frozen const edge current (theta-driven) |I|: max %.4f med %.4f"%(i0.abs().max(),i0.abs().median()))
print("dt per step", 1.75/18)
# does a 4-spec input also get used: u only via boundary OTAs + VCA
