import sys, numpy as np, torch
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
import torch.nn as nn
from ctle_dagger_common import load_plain_mlp_teacher, PlainMLP, plain_param_count
f=np.load(r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\fixed_teacher_dataset.npz")
t=load_plain_mlp_teacher(r"C:\Users\Atharva\Downloads\improved-zig-nf-spline-pytorch-default-v4\dagger_student_plain.pt",torch.device("cpu"),trunk_width=48,trunk_layers=3,activation=nn.SiLU,use_layernorm=False,input_preprocessing="knet")
print("input_log_min attached:",t.input_log_min)
if t.input_log_min is None:
    t.attach_scaler(1.0,0.0,1.0,1.0,1.0,input_preprocessing="knet",input_log_min=f["input_log_min"],input_log_max=f["input_log_max"])
t.eval()
with torch.no_grad(): lg=t(torch.from_numpy(f["specs"])).numpy()
print("max|logit diff| vs fixed_teacher_dataset:",np.abs(lg-f["teacher_logits"]).max(), "mean",np.abs(lg-f["teacher_logits"]).mean())
print("n teacher params (trainable)",sum(p.numel() for p in t.parameters() if p.requires_grad))
t2=PlainMLP(trunk_width=96,trunk_layers=3,activation=nn.SiLU,use_layernorm=True,use_log_features=False)
print("PlainMLP 96x3 LN trainable params",sum(p.numel() for p in t2.parameters() if p.requires_grad))
print("fixed dataset rows",len(f["specs"]),"spec min",f["specs"].min(0),"max",f["specs"].max(0))
