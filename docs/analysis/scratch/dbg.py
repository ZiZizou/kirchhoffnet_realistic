import sys, argparse, torch, numpy as np, joblib
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
import importlib.util
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
from ctle_dagger_common import *
import ctle_dagger_common as C
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
z=np.load(d+"zig_failure_comparison.npz")
specs=z["specs"]
for nm in ["student","teacher"]:
    ps=z[nm+"_pred_specs"]
    mt=C.compute_forward_errors(ps,specs,default_threshold=0.10)
    print(nm,"recomputed fail from saved pred_specs:",mt["failure_mask"].mean(),"saved mask",z[nm+"_failure_mask"].mean())
    print("  pred_specs med",np.median(ps,0),"target med",np.median(specs,0))
# now my pred specs
zd=r"Y:\Documents\ASPDAC_2026\kirchhoff_redesign\mlp_distillation_target\\"
sx=joblib.load(zd+"hybrid_hurdle_scaler_X.pkl"); sp_=joblib.load(zd+"hybrid_hurdle_scaler_y_power.pkl"); cfg=joblib.load(zd+"hybrid_hurdle_config.pkl")
zig=HybridHurdleModel(dropout=0.0,per_target=False); zig.load_state_dict(torch.load(zd+"hybrid_hurdle_ctle_model.pt",map_location="cpu")); zig.eval()
C._set_active_context({"DEVICE":torch.device("cpu"),"zig_model":zig,"scaler_X":sx,"scaler_y_p":sp_,"VALIDITY_THRESHOLD":0.5,"DEGRADE_REL_THRESHOLD":0.2,"MIN_DEGRADED_DIMS":2,"ERROR_THRESHOLD":0.10})
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"]); args.kn_solver_scale=1.0; args.student_kind="knet"
m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"])); m.load_state_dict(b["model"]); m.eval()
ev=StudentEvaluator(m,scaler_X=sx,zig_model=zig,eye_scale_h=float(cfg["eye_scale_h"]),eye_scale_w=float(cfg["eye_scale_w"]),eye_scale_j=float(cfg["eye_scale_j"]),scaler_y_p=sp_,device=torch.device("cpu"))
f,mt=ev.identify_failures(specs)
ps=mt["pred_specs"]; sps=z["student_pred_specs"]
print("my pred_specs vs saved: max rel diff per col", (np.abs(ps-sps)/np.abs(sps)).max(0), "median rel", np.median(np.abs(ps-sps)/np.abs(sps),0))
print("p_valid invalid rate mine",mt["invalid_mask"].mean())
print("my fail",f.mean())
print("scaler_X mean",sx.mean_, "scale",sx.scale_); print("scaler_p",sp_.mean_,sp_.scale_)
print("saved mask rel err from saved pred:")
import inspect; print(inspect.getsource(C.compute_forward_errors)[:2500])
