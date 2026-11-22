import torch, joblib
d=r"Y:\Documents\ASPDAC_2026\kirchhoff_redesign\mlp_distillation_target\\"
for f in ["dagger_student_moe.pt","hybrid_hurdle_ctle_model.pt","ctle_conditional_flow.pt"]:
    o=torch.load(d+f,map_location="cpu",weights_only=False)
    print("==",f,type(o))
    sd=o if isinstance(o,dict) and all(hasattr(v,"shape") for v in o.values()) else None
    if sd is None and isinstance(o,dict):
        for k,v in o.items():
            print(" key",k,type(v).__name__, (list(v.keys())[:5] if isinstance(v,dict) else (v if not hasattr(v,'shape') else v.shape)))
            if isinstance(v,dict) and all(hasattr(x,'shape') for x in v.values()): sd=v
    if sd:
        n=sum(v.numel() for v in sd.values() if v.dtype.is_floating_point); print(" n float params",n, "n keys",len(sd))
        for k,v in list(sd.items())[:40]: print("  ",k,tuple(v.shape))
for f in ["hybrid_hurdle_config.pkl"]:
    print(joblib.load(d+f))
