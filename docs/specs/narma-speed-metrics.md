# Spec: narma-speed-metrics

Scope: feature

# NARMA Training Speed + Validation Metrics

## Problem

1. **Training is extremely slow**: ~2.6s per chunk, 50 chunks/epoch, 200 epochs = ~7.2 hours. The inner loop calls `net()` once per sample (K=50) through the full `KirchhoffNetWithIO.forward()` wrapper, dominating runtime for a tiny 657-parameter network.

2. **No intermediate validation metrics**: Only `loss`, `h_rms`, `leak_grad` are tracked per epoch. NRMSE and R² are computed only after all 200 epochs finish, making convergence monitoring impossible.

## Requirements

### Speed (inner loop batching)
- Replace `for s in range(tbptt_chunk): ... net(u_b, ...)` with a single `stage._forward_heun_sequence()` call per chunk (mirrors `_evaluate_fabric_direct`)
- Access `net.core.stages[0]`, `net.output_mapper`, `net.read_slice`, `net.core.stage_times[0]`, `net.core.stage_steps[0]`
- Pass `u_chunk.unsqueeze(-1)` for shape `(B, K, 1)`
- Batch-compute `output_mapper` over all K states → single MSE loss
- Maintain: AMP autocast, gradient clipping, state.detach() at chunk boundaries
- `torch.compile` the sequence method for fused GPU kernel execution (PyTorch 2.x)

### Validation metrics
- Run `_evaluate_fabric_direct` on test set every N epochs (default 5)
- Compute NRMSE and R² (denormalized using training stats)
- Display in tqdm epoch-level postfix as `nrmse=0.42 r2=0.83`
- Store in `val_nrmse_history` / `val_r2_history` and return in result dict

### Hyperparameter tuning
- Change default `num_steps` from 6 to 4
- Change default `tbptt_chunk` from 50 to 25
- Keep both as CLI overrides (`--num-steps`, `--tbptt-chunk`)

## Out of scope
- Multi-stage network batching (NARMA is single-stage only)
- CUDA graph capture (covered by torch.compile)
- Non-NARMA experiments