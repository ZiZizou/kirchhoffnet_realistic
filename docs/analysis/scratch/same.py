import torch
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
L=lambda f: torch.load(d+f,map_location="cpu",weights_only=False)
best=L("best_student.pt"); fin=L("student_final.pt"); bun=L("student_bundle.pt")["model"]; last=L("last_checkpoint.pt")
def same(a,b): return all(torch.equal(a[k],b[k]) for k in a) and a.keys()==b.keys()
print("best==final",same(best,fin),"final==bundle",same(fin,bun),"best==last_ckpt.model",same(best,last["model"]),"last epoch idx",last["epoch"],"best_val",last["best_val"],"hist len",len(last["history"]))
import numpy as np
mx=max((best[k].float()-last["model"][k].float()).abs().max().item() for k in best if best[k].dtype.is_floating_point); print("max param diff best vs last_ckpt",mx)
