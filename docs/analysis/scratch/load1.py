import sys, importlib.util, json, argparse, torch, numpy as np
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"])
for k,v in dict(kn_solver_scale=1.0, student_kind="knet").items():
    if not hasattr(args,k): setattr(args,k,v); print("added",k,v)
m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"]))
r=m.load_state_dict(b["model"],strict=False)
print("missing",r.missing_keys); print("unexpected",r.unexpected_keys)
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
from config import SOLVER; print(SOLVER)
