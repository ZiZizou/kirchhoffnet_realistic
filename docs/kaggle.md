# Kaggle retraining guide

## Kaggle setup and inputs

The notebook clones the GitHub repository directly; `scripts/kaggle/make_kaggle_bundle.sh` is no longer needed (it remains in the tree for reference). **Before running the clone cell, push branch `hw-refactor` to GitHub `origin` (`ZiZizou/kirchhoffnet_realistic`).** The cell fetches that branch and cannot see local, unpushed commits. Create a Kaggle secret named `GITHUB_TOKEN` with permission to read the repository; the clone cell reads it with `kaggle_secrets.UserSecretsClient` and never puts a plaintext token in notebook source.

Attach these Kaggle inputs to the notebook:

- Model `awekill/improved-zig-nf-spline`, version 4. Its model directory supplies `dagger_student_plain.pt` and the flow/ZIG/scaler files listed below.
- Dataset `awekill/augmented-cvae-ctle`, mounted at `/kaggle/input/datasets/awekill/augmented-cvae-ctle`.

The notebook defaults are overridable with `KNB_TEACHER_CKPT`, `KNB_TEACHER_DIR`, `KNB_CSV_DIR`, and `KNB_CANONICAL`. It checks all expected files at startup, prints SHA-256 values for the teacher and flow/scaler files, and warns (rather than failing) if they differ from the ssr0 reference hashes. To regenerate the notebook after changing its generator, run `python scripts/kaggle/build_notebook.py`.

The paper's `canonical_ctle_phase_a_v2.npz` is included at `assets/canonical/canonical_ctle_phase_a_v2.npz` (SHA-256 `8429ed258c8d05a330c45fe6aee48eea096c98149c6b8d9a908127d8c0a1808`). The notebook uses this exact file by default because its sanity check asserts the paper checkpoint results (validation 8.05%, test 8.15%). The dagger script has a `--prepare-canonical-dataset` path that builds a canonical NPZ from the teacher flow/ZIG using a seed for spec sampling and split generation. Flow candidate sampling and the supplied model/runtime can vary, so the option is not guaranteed to reproduce this exact paper file; the notebook does not rely on it.

The early input check requires these files under `TEACHER_DIR` (used for both `--teacher-dir` and `--zig-dir`): `ctle_conditional_flow.pt`, `flow_scaler_X.pkl`, `flow_scaler_C.pkl`, `hybrid_hurdle_ctle_model.pt`, `hybrid_hurdle_scaler_X.pkl`, `hybrid_hurdle_scaler_y_power.pkl`, and `hybrid_hurdle_config.pkl`. It also requires `dagger_student_plain.pt` at `TEACHER_CKPT`. Under `CSV_DIR`, the DAgger script currently names 15 files: `dataset_log_part_{1..6}_may3.csv`, `dataset_log_part_{1..6}_may4.csv`, `dataset_log_may_7.csv`, `dataset_log_march12.csv`, and `dataset_log_may_18.csv`.

## Notebook settings and run order

Use **GPU (strongly prefer T4 over P100)**, turn **Internet on** (environment setup installs any missing Python packages), and enable notebook persistence. T4 (`sm_75`) supports Triton `torch.compile` for the fused Heun ODE kernels, whereas P100 (`sm_60`) automatically falls back to the eager hoisted fast path. Run one rung per session in this sequence:

1. R0 — paper configuration, unconstrained;
2. R1 — `phase3a_core`;
3. R1d — `phase3a_core_diff`;
4. R2 — `phase3a_full`;
5. R3 — `phase3b_tm`.

The notebook does not pass removed `--kn-moe-*` flags. It uses `--teacher-dir`, `--data-dir`, and `--canonical-dataset` to override host-specific paths. Outputs are separated under `OUT_DIR/<rung>_seed<seed>/`.

Notebook parameters can be changed in the first cell or headlessly via `KNB_RUNG`, `KNB_EPOCHS`, `KNB_SEED`, `KNB_LR`, `KNB_STEPS_PER_STAGE`, `KNB_BATCH_SIZE` (default `512`), `KNB_COMPILE` (`auto`/`on`/`off`, default `auto`), `KNB_MAX_HOURS` (default `10.5`; `<=0` disables), `KNB_DATA_ROOT`, `KNB_OUT_DIR`, `KNB_RESUME` (`""`, `"auto"` to resume `phase_a_last.pt` in `run_dir`, or an explicit path), `KNB_SMOKE`, and `KNB_RUN_TOY_ABLATION`. `SMOKE=1` runs one epoch and sets `--phase-a-sample-limit 32`; it is for plumbing only, not an accuracy result. For the hardware rungs, begin at LR 0.03 and compare 0.1/0.01 as validation-driven alternatives. The paper solver uses 70 Heun steps over 7 stages, or 10 per stage (`round(70 / 7)`). Hardware rungs default to 40 steps per stage (4× the paper setting); R0 omits the override and retains the paper solver default. Use the documented stiffness/trajectory diagnostics to validate this integration grid.

## Runtime optimizations and Kaggle limits

The CTLE network has 5,649 parameters (18 nodes, ~52 OTAs/stage, 7 stages), so Kaggle GPU runtime is dominated by Python dispatch and CUDA kernel launch overhead across the 560 `rhs()` evaluations per batch (40 steps/stage x 2 Heun passes x 7 stages). Before optimization, R0 took ~3 h on Kaggle and R1/R2 timed out at the 12 h session cap because R1 dispatched ~5.4x and R2 ~7.9x as many ATen ops per batch as R0.

The pipeline now applies exact mathematical and training-loop optimizations (no change to `STEPS_PER_STAGE=40`, `BATCH_SIZE=512`, `EPOCHS=300`, or float32 precision):

1. **Stage-invariant hoisting & fused linear operator (`fast_heun.py`)**: All per-stage parameter transforms (cap-divider STE, `ic_factor`, VCA `edge_gain`, GLN `edge_gm_scale`, `vref_map`, `xbar_map`, masks, mismatch), core frozen-leg scatters, core resistive Laplacian + leak matrix (`M_lin`), and `Vref` compliance factors are computed once per stage instead of 80x per stage.
2. **Vectorized compliance & scalar math (`hw/constraints.py`, `fast_heun.py`)**: Eliminates per-substep CPU scalar tensor allocations in `leg_factors` and evaluates `leg_factors` once per substep across all 18 nodes rather than 5x across 81 edge columns.
3. **Zero-graph-break `torch.compile` (`--compile-mode auto`, T4 `sm_75+`)**: `_heun_step_standard` and `_heun_step_compliance` compile with `torch.compile(backend="inductor", fullgraph=True, dynamic=True)` on `sm_70+` GPUs (T4), fusing the Heun step kernels and eliminating CUDA launch overhead. On P100 (`sm_60`) or CPU, `auto` seamlessly uses the eager hoisted fast path (~3.5-4x fewer ATen ops on R1, ~4-5x fewer on R2/R3).
4. **Training-loop deduplication (`ctle_dagger_common.py`)**: Preloads training/validation tensors directly on GPU (eliminating per-epoch DataLoader `__getitem__` and H2D transfers), fuses the ZIG forward + validity NLL into a single ZIG pass per batch, and reuses validation logits across Huber loss, failure evaluation, and power diagnostics (1 student pass instead of 3 on eval epochs).
5. **Graceful session-budget stop (`KNB_MAX_HOURS=10.5`)**: Checks elapsed wall time + `1.5 x last_epoch_time` after saving `phase_a_last.pt`. If a session approaches `KNB_MAX_HOURS`, it stops cleanly between epochs so `phase_a_best.pt`, full split evaluation, `hw.audit`, plots, `results.csv`, and the output ZIP are always saved rather than killed at 12 h.

**GPU recommendation:** Choose **T4** (`sm_75`) rather than P100 (`sm_60`) on Kaggle so `torch.compile` (`inductor`/Triton) activates automatically. Check epoch 1-2 `dt=` timings printed in `train.log` to verify throughput.

To continue after a `--max-hours` stop, persist/download the output directory and set `KNB_RESUME` to `"auto"` (or the explicit path to `phase_a_last.pt`). Resume restores model and optimizer, scheduler, epoch, history, best-so-far model/loss, and early-stop patience. Keep the same rung, seed, dataset, and configuration when resuming; use the same total `KNB_EPOCHS` target. The best evaluated model is `phase_a_best.pt`.

## Results and artifact retrieval

The evaluation cell checks train/val/test against the shared 10% failure rule, runs `hw.audit` for hardware presets, saves training plots and audit JSON, and appends (never replaces) a row in `OUT_DIR/results.csv`. The final cell creates a ZIP of `OUT_DIR`. Download the notebook output ZIP or use Kaggle's persisted `/kaggle/working/out`; bring the ZIP back to ssr0 and extract it outside the source checkout for comparison. Keep `train.log`, `phase_a_history.json`, checkpoint files, `results.csv`, audit JSON, and plots together. Do not compare a smoke run as a trained result.

The optional toy ablation setting is off by default. The existing CPU-learning report warns that its three-seed ablation is too noisy; any follow-up should use at least ten seeds and report the mean and 95% confidence interval.
