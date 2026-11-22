# Spec: kn-bayes-opt

Scope: feature

# KNet Bayesian Optimization (kn_bayes_opt)

Optuna-driven Bayesian hyperparameter search for the reduced differential
KirchhoffNet (`train_script.py`), mirroring `mlp_bayes_opt.py`.

## Runner contract
- `kn_bayes_opt.py` runs each trial as a `train_script.py` subprocess.
- Artifact contract: `final_metrics.txt` -> `param_count`, `best_val`,
  `best_epoch`, `epochs_run`, `elapsed_seconds` (already written by train_script).
- Objective: `best_val` (huber loss on standardized targets), default `--epochs 800`.
- Optuna sqlite study, TPESampler, `--resume`, CUDA round-robin via
  `CUDA_VISIBLE_DEVICES`, artifacts `best_hyperparams.txt`, `results.csv`,
  `objective_history.png`, `trial_<NNNN>/log.txt`.

## Permanently-on flags (every trial)
`--freeze-read --temporal-readout --no-early-stop --leak non-programmable
--vca --vca-core --vca-separate-core-bus --cell-library tanh_free
--hidden-family small_world --boundary-fan-out <json>`

## Search dimensions
- `num_hidden`: per-dataset int range, min = `in_dim * fanout_count`.
- `small_world_k`: even categorical in `[2, min(14, num_hidden-1)]`, sampled
  AFTER `num_hidden` so always valid.
- `small_world_p`: continuous `[0, 1]`.
- `num_stages`: int `[1, 3]`; per-stage `t_span`/`num_steps` = total / n.
- `t_span`: total ODE horizon, `[0.5, 10.0]` (default 7.0).
- `num_steps`: total Heun steps, `[10, 150]` (default 70).
- `fanout_count`: categorical `{1, 2}`. Fixed-spread map: 2 conn -> input i
  -> `{i, i+in_dim}`; 1 conn -> input i -> `{i}`. Targets unique, < num_hidden.
- `lr`: log `[1e-4, 1e-2]`; `weight_decay`: log `[1e-6, 1e-2]`;
  `batch_size`: categorical `[512, 1024, 2048, 4096]`.

## Datasets
housing (in 8), smooth2d (in 2), friedman1 (in 10), friedman2/3 (in 4);
out_dim 1, huber. friedman runs add `--target-noise-std 1.0`.

## train_script.py additions (required)
New CLI flags: `--t-span`, `--num-steps`, `--weight-decay`, `--batch-size`
(total-budget semantics, divided across stages; defaults from SOLVER/OPTIM).

## Validity
- `num_hidden >= in_dim * fanout_count` (enough distinct fanout targets).
- `small_world_k` even, `2 <= k < num_hidden`, capped at 14.
- `TrialPruned` on any invalid combo.

## Deferred (follow-up plan)
Learned long-range small-world edges via continuous STE indices
(`src_raw`/`dst_raw` rounded with straight-through estimator), which would
also make the exact boundary-fanout target nodes learnable by gradient
descent rather than fixed-spread.