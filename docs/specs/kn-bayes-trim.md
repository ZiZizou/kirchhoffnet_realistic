# Spec: kn-bayes-trim

Scope: feature

# kn-bayes-trim search space + per-dataset start points

Feature spec for plan `kn-bayes-trim`. Defines the final BO search space (18 dims) and the
per-dataset validated start points seeded as trial 0.

## Goal

- Trim redundant/coupled dims from the previous 21-dim space.
- Add `device_l2_lambda`, `freeze_boundary`, `freeze_temporal_read` (all already supported
  as train_script CLI flags — no train_script.py changes needed).
- Seed trial 0 with the user's validated 800-epoch configs for friedman1, friedman2, smooth2d
  via `study.enqueue_trial`, then explore the trimmed space around it.

## Final search space (18 dims)

| Group | Dim | Type / range | Notes |
|---|---|---|---|
| Topology | `num_hidden` | int, per-dataset range | **friedman2 max must be raised to >= 25** (start point = 25, currently capped at 24). |
| Topology | `small_world_k` | categorical even {2..14}, pruned if >= num_hidden | static choices for TPE (no dynamic spaces). |
| Topology | `small_world_p` | float [0,1] | |
| Topology | `num_stages` | int [1, max] | **max must reach 10** (smooth2d start = 10); currently `NUM_STAGES_CHOICES=[1,2,3]` and `--num-stages-max 3`. |
| Topology | `fanout_count` | categorical {1,2} | boundary fanout pattern. |
| Solver | `t_span` | float [0.5,10] log-uniform | kept; `c_eff` removed (degenerate with t_span via dx/dt=ΣI/C_eff). |
| Solver | `num_steps` | int [10,150] | |
| Optimizer | `lr` | log [1e-4,1e-2] | |
| Optimizer | `weight_decay` | log [1e-6,1e-2] | |
| Optimizer | `batch_size` | categorical {512,1024,2048,4096} | |
| Physics | `x_max` | float [0.5,8] | ODE rail + input mapper rail. |
| Cell bounds | `gm_max` | log [1,50] | `gm_min` removed (fixed at config default 0.01). |
| Cell bounds | `isat_max` | log [1,50] | `isat_min` removed (fixed at config default 0.01). |
| Regularizer | `sparsity_lambda` | log [1e-7,1e-3] | |
| Regularizer | `entropy_lambda` | log [1e-6,1e-2] | |
| Regularizer | `device_l2_lambda` | log [1e-7,1e-3] | NEW. Scale-invariant L2 on tanh_free gain/device params (train_script `--device-l2-lambda`). |
| Freeze | `freeze_boundary` | categorical {0,1} | NEW. Emits `--freeze-boundary` when 1. |
| Freeze | `freeze_temporal_read` | categorical {0,1} | NEW. Emits `--freeze-temporal-read` when 1. |

Removed vs previous space: `c_eff`, `gm_min`, `isat_min`. (Total was 21 with the 3 new dims;
trimming 3 redundant brings it to 18.)

## Seed-trial mechanism

- Define `START_POINTS: dict[str, dict]` keyed by problem with the validated values below.
- In `main()`, after creating the study, call `study.enqueue_trial(start_params)` for
  datasets that have a start point. Trial 0 then runs the exact validated config.
- `_build_command` gains an `is_seed`/`start_point` param: when True it (a) emits the exact
  `--boundary-fan-out` map from the start point instead of `build_boundary_fan_out()`, and
  (b) passes fixed flags for the dataset. Search dims outside their ranges (e.g.
  `num_stages=5`/`10`, `num_hidden=25`) must be honored by the seed (bypass suggest bounds for
  trial 0).
- Exploration trials (number >= 1) keep current behavior: `build_boundary_fan_out()` regenerates
  the map for the sampled num_hidden; num_stages stays within search bounds.

## Fixed flags for friedman1 / friedman2 / smooth2d (all trials)

These come from the user's validated commands and are NOT searched:

- `--solver heun` (config default anyway, emit explicitly)
- `--interstage-activation residual-relu-tanh`
- `--mapper-lr-scale 1.0`
- `--struct-lr-scale 4`
- `--dyn-lr-scale 1.0`
- `--grad-log --grad-log-every 10`
- `--validate-every 10`
- `--seed 100`
- `--epochs` from CLI (default in user commands was 800)

Existing permanent flags already in `_build_command` stay: `--cell-library tanh_free`,
`--leak non-programmable`, `--temporal-readout`, `--freeze-read`, `--vca --vca-core
--vca-separate-core-bus`, `--no-early-stop`, `--hidden-family small_world`,
`--target-noise-std 1.0` (friedman only).

## Start point values

### friedman1 (in_dim=10, out_dim=1)
- `num_hidden: 25`, `small_world_k: 4`, `small_world_p: 0.2`, `num_stages: 5`
- `fanout_count: 2` (map below overrides generation)
- `boundary_fan_out`:
  `{"0": [2,12], "1": [7,17], "2": [22,5], "3": [10,15], "4": [4,14], "5": [8,19], "6": [6,16], "7": [9,23], "8": [1,13], "9": [11,24]}`
- Other 13 dims: fix at config defaults so trial 0 exactly reproduces the validated run:
  t_span=7.0, num_steps=70, lr=OPTIM lr (auto-scaled), wd=1e-4, bs=OPTIM batch_size,
  x_max=3.0, gm_max=10, isat_max=10, sparsity=1e-6, entropy=1e-4, device_l2=0,
  freeze_boundary=0, freeze_temporal_read=0.

### friedman2 (in_dim=4, out_dim=1)
- `num_hidden: 25` (requires raising friedman2 `num_hidden_range` max to >= 25),
  `small_world_k: 4`, `small_world_p: 0.2`, `num_stages: 5`, `fanout_count: 2`
- `boundary_fan_out`: `{"0": [2,12], "1": [7,17], "2": [22,5], "3": [10,15]}`
- Other dims: same as friedman1 (config defaults).

### smooth2d (in_dim=2, out_dim=1)
- `num_hidden: 14`, `small_world_k: 4`, `small_world_p: 0.2`, `num_stages: 10`
  (requires raising `--num-stages-max` to >= 10), `fanout_count: 2`
- `boundary_fan_out`: `{"0": [0,1], "1": [3,4]}`
- Other dims: same as above (config defaults).

## Validation protocol

1. `py_compile` kn_bayes_opt.py.
2. train_script smoke (1 epoch, cpu) with `--device-l2-lambda --freeze-boundary
   --freeze-temporal-read` + the friedman2 fixed flags to confirm forwarding.
3. kn_bayes_opt 1-trial CPU smoke on friedman2 with the start point active: verify trial 0 uses
   the exact boundary map + fixed flags, `config_snapshot.txt` shows `device_l2_lambda`, freeze
   flags appear in the trial cmd log, metrics parse.
4. Verify `best_hyperparams.txt` lists all 18 dims + dataset/start-point metadata.