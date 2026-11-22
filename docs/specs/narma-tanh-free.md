# Spec: narma-tanh-free

Scope: feature

# NARMA tanh_free Cell Library Feature Spec

## Goal
Enable running NARMA fabric experiments with `tanh_free` cell library (and other CELL_LIBRARIES options) via CLI, while preserving existing freeze_read comparison behavior.

## Changes

### CLI Arguments (parse_args)
- `--cell-library`: Cell library to use for fabric conditions. Choices: `tanh`, `tanh_free`, `tanh_realistic`, `tanh_realistic_upgrade`, `tanh_anti`, `relu`. Default: `tanh` (backward compatible).
- No new flag for PARALLEL_TANH_MULT_ENABLED (uses config default False).

### Function Changes
- `run_fabric_condition()`: Add `cell_library: str` parameter, pass to `make_cell_library()` instead of hardcoded `"tanh"`.
- `main()`: Pass `args.cell_library` to both `run_fabric_condition()` calls (freeze_read=False and freeze_read=True).

### Behavior
- Fabric conditions continue to run both `freeze_read=False` (fabric_fr_off, evolving core) and `freeze_read=True` (fabric_fr_on, frozen core) for each seed.
- Cell library applies to both conditions identically.

## Usage (Kaggle GPU)
```bash
python narma_experiment.py \
  --order 20 \
  --device cuda \
  --compile \
  --cell-library tanh_free \
  --epochs 200 \
  --n-streams 4 \
  --train-samples 2500 \
  --seeds 0,1,2 \
  --output ./output/narma20_tanh_free
```

## Expected Output
- Baselines: ridge, esn, mlp, mlp_large, lstm, lstm_large (unchanged)
- Fabric: fabric_fr_off (evolving core) + fabric_fr_on (frozen core) × 3 seeds = 6 fabric runs
- Each fabric run: test NRMSE, R^2, MC, param count, ridge diagnostic (if --ridge-diagnostic)
- Summary CSV at output dir