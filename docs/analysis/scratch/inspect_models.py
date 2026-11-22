"""Audit-only: inspect model.pt artifacts from trained_knet runs.

Prints: top-level type, keys, state_dict tensor names/shapes/param counts,
min/max/mean for float tensors, and any non-tensor entries.
"""
import sys
import torch

RUNS = {
    "smooth2d_w_tanh": r"C:\Users\Atharva\Downloads\trained_knet\test_smooth2d_w_tanh\model.pt",
    "friedman2_normal": r"C:\Users\Atharva\Downloads\trained_knet\friedman2_normal\model.pt",
    "deq": r"C:\Users\Atharva\Downloads\trained_knet\deq\model.pt",
}

def describe(obj, indent=0, maxdepth=3, prefix=""):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{prefix}dict with {len(obj)} keys")
        if indent < maxdepth:
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    print(f"{pad}  [{k}] Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                    if v.is_floating_point() and v.numel() > 0:
                        print(f"{pad}      min={v.min().item():.6g} max={v.max().item():.6g} "
                              f"mean={v.float().mean().item():.6g}")
                elif isinstance(v, (dict, list, tuple)):
                    print(f"{pad}  [{k}] {type(v).__name__} len={len(v)}")
                    describe(v, indent + 2, maxdepth, prefix=f"[{k}] ")
                else:
                    s = repr(v)
                    print(f"{pad}  [{k}] {type(v).__name__} = {s[:200]}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{prefix}{type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:12]):
            if isinstance(v, torch.Tensor):
                print(f"{pad}  [{i}] Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                if v.is_floating_point() and v.numel() > 0:
                    print(f"{pad}      min={v.min().item():.6g} max={v.max().item():.6g}")
            else:
                print(f"{pad}  [{i}] {type(v).__name__} = {repr(v)[:200]}")
    else:
        print(f"{pad}{prefix}{type(obj).__name__} = {repr(obj)[:300]}")

for name, path in RUNS.items():
    print("=" * 100)
    print(f"RUN: {name}  ({path})")
    print("=" * 100)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        continue
    print(f"top-level type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print("top-level keys:", list(obj.keys()))
        sd = obj.get("state_dict", obj.get("model_state_dict", None))
        if sd is None:
            # maybe it IS a state dict
            vals = list(obj.values())
            if vals and all(isinstance(v, torch.Tensor) for v in vals):
                sd = obj
        if isinstance(sd, dict):
            print(f"\nstate_dict: {len(sd)} entries, "
                  f"total params = {sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))}")
            print("-" * 90)
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    extra = ""
                    if v.is_floating_point() and v.numel() > 0:
                        extra = (f"  min={v.min().item():+.6g} max={v.max().item():+.6g} "
                                 f"mean={v.float().mean().item():+.6g}")
                    print(f"  {k:<70s} {str(tuple(v.shape)):<22s} {str(v.dtype):<14s}{extra}")
                else:
                    print(f"  {k:<70s} {type(v).__name__}")
        # non state-dict metadata
        meta = {k: v for k, v in obj.items() if k not in ("state_dict", "model_state_dict")}
        if meta:
            print("\n--- other top-level entries ---")
            describe(meta, indent=0, maxdepth=2)
    else:
        describe(obj, indent=0, maxdepth=3)
    print()
