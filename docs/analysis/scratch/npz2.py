import numpy as np
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
z=np.load(d+"zig_failure_comparison.npz")
t=z["teacher_failure_mask"]; s=z["student_failure_mask"]
print("n",len(t),"teacher",t.mean(),"student",s.mean(),"both",(t&s).mean(),"disagree",(t!=s).mean(),"student-only",(s&~t).mean(),"teacher-only",(t&~s).mean())
p=np.load(d+"test_teacher_student_predictions.npz")
print("specs same as zig specs:",np.allclose(p["specs"],z["specs"]))
print("logit mse test",((p["teacher_logits"]-p["student_logits"])**2).mean())
f=np.load(d+"fixed_teacher_dataset.npz")
print("test specs == dataset[test_idx]:",np.allclose(f["specs"][f["test_idx"]],p["specs"]))
print("teacher_logits match dataset:",np.allclose(f["teacher_logits"][f["test_idx"]],p["teacher_logits"]))
