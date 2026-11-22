import torch, json
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
for f in ["best_student.pt","student_final.pt","student_bundle.pt","last_checkpoint.pt"]:
    o=torch.load(d+f,map_location="cpu",weights_only=False)
    print("=====",f,type(o))
    def show(o,ind=0,name=""):
        p=" "*ind
        if isinstance(o,dict):
            for k,v in o.items():
                if isinstance(v,dict):
                    print(p+str(k),"dict",len(v)); 
                    if ind<4: show(v,ind+2)
                elif hasattr(v,"shape"): print(p+str(k),tuple(v.shape),v.dtype)
                else: print(p+str(k),repr(v)[:300])
    show(o)
