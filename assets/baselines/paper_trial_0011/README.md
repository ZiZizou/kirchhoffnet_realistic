# paper_trial_0011: trained PAPER HR-KNet (Phase-A distillation)

The checkpoint behind the paper's reported CTLE failure rate.

## Provenance

- Run: `manual_knet_distillation_mix_trials/trial_0011`, SLURM job 59014723, seed 100, CUDA.
- Command: [run_command.txt](run_command.txt). It matches the final command of
  [docs/logs/run_train_knet_single_shot_distillation_2.sh](../../../docs/logs/run_train_knet_single_shot_distillation_2.sh).
  Log: `docs/logs/run_train_knet_single_shot_distillation_2.sh-59014723.out`.
- Logged result (10% degrade rule, `ctle_metrics_config.DEGRADE_REL_THRESHOLD`):
  `[phaseA] final failure: train= 7.67% val= 8.05% test= 8.15%`
- Training code: git commit `5c6a4f4` of `kirchhoffnet_realistic` ("Added featuers to make training with losses more comparable to kn_bayes_bo with distillation and other terms"). The model had 5649 trainable parameters then; at HEAD it has 5523 (126 = 7 stages x 18 `u_logits`).
- Data: `canonical_ctle_phase_a_v2.npz` (fingerprint `1d866ecd...a9610f7`, 16000/2000/2000 train/val/test, schema 2) and the ZIG forward surrogate `improved-zig-nf-spline-pytorch-default-v2`.

## Files

| file | content |
|---|---|
| `phase_a_best.pt` | best-validation-failure state (the one the final line evaluates). sha256 `29e18c96...f9d2` |
| `phase_a_history.json` | `{"history": per-epoch, "final": per-split metrics, "args": run args}` |
| `run_command.txt` | training command line |

`phase_a_last.pt` (last epoch, 400 KB, includes optimizer state) was not copied.

## Checkpoint format (`phase_a_best.pt`)

A plain flat `state_dict` (`dict[str, Tensor]`, 345 entries, float32 and int64). There is no nested `state_dict`, scaler or meta entry. It is the state of `LocalKirchhoffStudentWrapper`:

- `log_lo`, `log_hi` (7,): frozen output bounds (`nn.Parameter(requires_grad=False)`).
- `input_log_min`, `input_log_max` (4,): input scaling buffers. These are the values the training script computed from its own spec data (`[-3.7205, -1.9412, -2.7712, -1.3602]`, `[-0.5175, 2.1876, 2.1286, 2.1600]`). They differ slightly from the `input_log_min/max` stored in the canonical npz. The loader uses the checkpoint's buffers.
- `net.core.stages.{0..6}.*`: per-stage cell libs, `z_logits`, `u_logits`, GLN/VCA/readout parameters and int64 topology buffers (`src`, `dst`, `boundary_*`, `readout_sense_src`, `output_ode_*`, empty `_drive_idx`, `ref_dst`).
- `net.core.transfers.{0..5}.*`, `net.post_readout_transfer.*`: residual-relu-tanh interstage transfers (+ empty `_drive_mask`).
- `net.output_mapper.{gain,bias,tanh_gain,relu_gain,relu_threshold}`: the OutputAffine readout.
- `net.gln_rails.*`: GLN rails.


## Loading

```python
from checkpoints import load_paper_checkpoint
model = load_paper_checkpoint()          # float32, CPU, eval mode
```

At HEAD, the surgery removed 28 keys that the file still contains: `net.core.stages.*.u_logits` (7), `*._drive_idx` (7), `*.ref_dst` (7), `net.core.transfers.*._drive_mask` and `net.post_readout_transfer._drive_mask` (7). The loader drops exactly those, asserts nothing else is missing or unexpected, and loads the rest with `strict=True`.

## Reproduce

```
python scripts/eval_paper_checkpoint.py        # about 10 s on 48 CPU cores
```

prints train 7.67% (1227/16000), val 8.05% (161/2000), test 8.15% (163/2000), identical to the log. Tests: `tests/test_paper_checkpoint.py`; golden: `tests/golden/data/paper_trial_0011.pt`.
