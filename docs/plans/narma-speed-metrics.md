---
plan name: narma-speed-metrics
plan description: Speed up NARMA training + add validation metrics
plan status: done
---

## Idea
NARMA fabric training is extremely slow (~2.6s/chunk, 7+ hours for 200 epochs) due to a per-sample Python loop calling net() for each of 50 samples per chunk. Additionally, there are no intermediate validation metrics (NRMSE, R²) during training — only loss/h_rms/leak_grad — making it impossible to tell if training is working until the full 200 epochs finish. We need to: (1) batch the inner loop using DifferentialStage._forward_heun_sequence to eliminate wrapper overhead, (2) apply torch.compile to the sequence method for fused GPU kernels, (3) add periodic validation-set NRMSE/R² every N epochs, and (4) tune hyperparameters (num_steps, tbptt_chunk) for further speed.

## Implementation
- Replace per-sample inner loop (lines 741-756 in train_fabric) with a single `stage._forward_heun_sequence()` call per chunk — accessing stage via `net.core.stages[0]`, passing `u_chunk.unsqueeze(-1)` for (B, K, 1) shape, batch-computing `output_mapper` over all K states, and computing a single MSE loss over all K×B predictions. Maintain AMP autocast, grad clipping, state detach at chunk boundary.
- Apply `@torch.compile` to `DifferentialStage._forward_heun_sequence` so the inner Python loop (per-sample Heun integration) compiles to fused CUDA kernels. Should provide 2-5x speedup on GPU.
- Add periodic validation every N epochs (default every 5): run `_evaluate_fabric_direct` on the test set, compute NRMSE and R², display in tqdm postfix, store in `val_nrmse_history` / `val_r2_history` lists, and return in the result dict. Overhead is ~0.1s per eval.
- Reduce `num_steps` from 6 to 4 (fewer Heun steps per sample, speeds up each chunk by ~33%). Keep as CLI override so user can increase if needed.
- Reduce `tbptt_chunk` from 50 to 25 (shorter BPTT windows = less sequential compute per optimizer step, more frequent updates). Keep as CLI override.

## Required Specs
<!-- SPECS_START -->
- narma-speed-metrics
<!-- SPECS_END -->