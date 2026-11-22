import sys, numpy as np
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
import ctle_dagger_common as C
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
z=np.load(d+"zig_failure_comparison.npz"); specs=z["specs"]
C._set_active_context({"MIN_DEGRADED_DIMS":2,"DEGRADE_REL_THRESHOLD":0.2})
for nm in ["student","teacher"]:
    ps=z[nm+"_pred_specs"]; saved=z[nm+"_failure_mask"]
    for thr in [0.10,0.20]:
        f=C.compute_forward_errors(ps,specs,default_threshold=thr)["failure_mask"]
        print(nm,"thr",thr,"deg-only rate",f.mean().round(4),"saved",saved.mean().round(4))
