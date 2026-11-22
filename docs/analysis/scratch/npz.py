import numpy as np
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
for f in ["fixed_teacher_dataset.npz","test_teacher_student_predictions.npz","zig_failure_comparison.npz"]:
    z=np.load(d+f,allow_pickle=True); print("==",f)
    for k in z.files: print(" ",k,z[k].shape,z[k].dtype)
z=np.load(d+"zig_failure_comparison.npz")
for s in ["validation","test"]:
    print(s,"teacher fail",z[s+"_teacher_failure_mask"].mean(),"student fail",z[s+"_student_failure_mask"].mean(), "n",len(z[s+"_specs"]),
      "both",(z[s+"_teacher_failure_mask"]&z[s+"_student_failure_mask"]).mean(),"disagree",(z[s+"_teacher_failure_mask"]!=z[s+"_student_failure_mask"]).mean())
