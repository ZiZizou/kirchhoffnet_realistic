"""Audit-only: dump connectivity buffers of the two KNet artifacts."""
import torch

S = r"C:\Users\Atharva\Downloads\trained_knet\test_smooth2d_w_tanh\model.pt"
D = r"C:\Users\Atharva\Downloads\trained_knet\deq\model.pt"

sd = torch.load(S, map_location="cpu", weights_only=False)
print("=== smooth2d_w_tanh: stage 0 connectivity ===")
for k in ("core.stages.0.src", "core.stages.0.dst", "core.stages.0.boundary_src",
          "core.stages.0.boundary_dst", "core.stages.0.output_ode_src",
          "core.stages.0.output_ode_dst", "core.stages.0.ref_dst",
          "core.stages.0._drive_idx"):
    v = sd.get(k)
    if v is None:
        continue
    if v.numel() <= 40:
        print(f"  {k:<40s} {v.tolist()}")
    else:
        print(f"  {k:<40s} n={v.numel()} first20={v.tolist()[:20]}")
print("  n_nodes implied by u_logits:", sd["core.stages.0.u_logits"].numel())
print("  transfers keys:", [k for k in sd if k.startswith("core.transfers")][:6])

sd2 = torch.load(D, map_location="cpu", weights_only=False)
print("\n=== deq: stage 0 connectivity ===")
for k in ("core.stages.0.src", "core.stages.0.dst", "core.stages.0._drive_idx"):
    v = sd2.get(k)
    if v is None:
        continue
    print(f"  {k:<40s} n={v.numel()} first24={v.tolist()[:24]}")
print("  u_logits (nodes):", sd2["core.stages.0.u_logits"].numel())
print("  read_idx implied by output_mapper.proj:", sd2["output_mapper.proj.weight"].shape)
print("  cell_type_code:", sd2["core.stages.0.cell_lib.cell_type_code"].tolist())
print("  cell_lib.gm:", sd2["core.stages.0.cell_lib.gm"].tolist())
print("  cell_lib.isat:", sd2["core.stages.0.cell_lib.isat"].tolist())
print("  cell_lib.gleak:", sd2["core.stages.0.cell_lib.gleak"].tolist())
print("  cell_lib.rho:", sd2["core.stages.0.cell_lib.rho"].tolist())
print("  cell_lib.beta:", sd2["core.stages.0.cell_lib.beta"].tolist())
print("  cell_lib.theta:", sd2["core.stages.0.cell_lib.theta"].tolist())
print("  cell_lib._is_rect:", sd2["core.stages.0.cell_lib._is_rect"].tolist())
print("  stage0 logits row0 (edge 0 type mix):", sd2["core.stages.0.logits"][0].tolist())
print("  stage0 logits row100:", sd2["core.stages.0.logits"][100].tolist())
print("  stage0 z_logits: min/max", float(sd2["core.stages.0.z_logits"].min()),
      float(sd2["core.stages.0.z_logits"].max()))
print("  stage2 z_logits: min/max", float(sd2["core.stages.2.z_logits"].min()),
      float(sd2["core.stages.2.z_logits"].max()))
print("  stage2 raw_mult: min/max", float(sd2["core.stages.2.raw_mult"].min()),
      float(sd2["core.stages.2.raw_mult"].max()))
print("  stage2 raw_leak: min/max", float(sd2["core.stages.2.raw_leak"].min()),
      float(sd2["core.stages.2.raw_leak"].max()))
