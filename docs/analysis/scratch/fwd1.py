import sys, importlib.util, argparse, torch, numpy as np, time
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"]); args.kn_solver_scale=1.0; args.student_kind="knet"
m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"]))
m.load_state_dict(b["model"]); m.eval()
p=np.load(d+"test_teacher_student_predictions.npz")
sp=torch.from_numpy(p["specs"][:256])
t=time.time()
with torch.no_grad(): out=m(sp)
print("time",time.time()-t)
sv=p["student_logits"][:256]; tl=p["teacher_logits"][:256]
o=out.numpy()
print("max abs diff vs saved student",np.abs(o-sv).max(),"mean",np.abs(o-sv).mean())
print("mse recomputed vs teacher",((o-tl)**2).mean(),"saved-vs-teacher",((sv-tl)**2).mean())
for st in m.net.core.stages: print(st.num_nodes, st.t_span if hasattr(st,'t_span') else None, getattr(st,'num_steps',None), st.x_max)
print(m.net)
