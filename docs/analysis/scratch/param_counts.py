"""Audit-only: precise parameter counts + init-vs-trained comparison for the artifacts."""
import torch

RUNS = {
    "smooth2d_w_tanh": r"C:\Users\Atharva\Downloads\trained_knet\test_smooth2d_w_tanh\model.pt",
    "friedman2_normal": r"C:\Users\Atharva\Downloads\trained_knet\friedman2_normal\model.pt",
    "deq": r"C:\Users\Atharva\Downloads\trained_knet\deq\model.pt",
}

for name, path in RUNS.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)
    print("=" * 90)
    print(name)
    floats = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
    ints = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor) and not v.is_floating_point()}
    n_float = sum(v.numel() for v in floats.values())
    print(f"  float tensors: {len(floats)}  numel={n_float}")
    print(f"  int/bool buffers: {len(ints)}  numel={sum(v.numel() for v in ints.values())}")
    # per-prefix breakdown
    pre = {}
    for k, v in floats.items():
        key = ".".join(k.split(".")[:-1]) if "." in k else k
        pre[key] = pre.get(key, 0) + v.numel()
    for k in sorted(pre):
        print(f"    {k:<55s} {pre[k]}")
    # duplicate (aliased) tensors
    seen = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            seen.setdefault(id(v), []).append(k)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        print("  ALIASED (same tensor object under several keys):")
        for _, keys in dups.items():
            print(f"    {keys}")
    print()

# DEQ: compare stage 0 / 1 / 2 parameter tensors for exact equality
print("=" * 90)
print("DEQ: pairwise stage parameter equality (are stages 0 and 1 still at init?)")
sd = torch.load(RUNS["deq"], map_location="cpu", weights_only=False)
keys = [k for k in sd if k.startswith("core.stages.")]
for suffix in sorted({k.split("core.stages.", 1)[1].split(".", 1)[1] for k in keys
                      if "." in k.split("core.stages.", 1)[1]}):
    a = sd.get(f"core.stages.0.{suffix}")
    b = sd.get(f"core.stages.1.{suffix}")
    c = sd.get(f"core.stages.2.{suffix}")
    if not (isinstance(a, torch.Tensor) and a.is_floating_point()):
        continue
    eq01 = bool(torch.equal(a, b))
    eq02 = bool(torch.equal(a, c))
    print(f"  {suffix:<28s} stage0==stage1: {str(eq01):<6} stage0==stage2: {eq02}")
