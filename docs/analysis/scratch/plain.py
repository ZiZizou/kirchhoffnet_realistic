import torch
v=r"C:\Users\Atharva\Downloads\improved-zig-nf-spline-pytorch-default-v4\dagger_student_plain.pt"
o=torch.load(v,map_location="cpu",weights_only=False)
print(type(o))
sd=o if all(hasattr(x,"shape") for x in o.values()) else o
for k,x in sd.items(): print(k,tuple(x.shape) if hasattr(x,"shape") else x)
print(sum(x.numel() for x in sd.values() if hasattr(x,"numel") and x.dtype.is_floating_point))
