---
plan name: kn-bayes-opt
plan description: Bayesian optimization for KirchhoffNet training
plan status: done
---

## Idea
Port the MLP bayesian hyperparameter optimization pattern (mlp_bayes_opt.py: optuna TPESampler study, subprocess-per-trial, final_metrics.txt contract, CUDA round-robin pinning via CUDA_VISIBLE_DEVICES, sqlite resume, results.csv + best_hyperparams.txt + objective_history.png artifacts) to the reduced differential KirchhoffNet training script (train_script.py).

A new runner `kn_bayes_opt.py` spawns `train_script.py` subprocesses per trial. The following flags are permanently ON for every trial:
  --freeze-read --temporal-readout --no-early-stop --leak non-programmable --vca --vca-core --vca-separate-core-bus --cell-library tanh_free --hidden-family small_world --boundary-fan-out <json>

Search dimensions (each trial):
  - num_hidden (small_world size; min bounded by in_dim*fanout_count so the fanout has enough distinct targets)
  - small_world_k (even categorical in [2, min(14, num_hidden-1)]; sampled AFTER num_hidden so it is always valid)
  - small_world_p (continuous [0,1])
  - num_stages (int [1,3]; per-stage t_span/num_steps = total/n)
  - t_span (total ODE horizon, default 7.0)
  - num_steps (total Heun steps, default 70)
  - fanout_count (categorical {1,2}) with a deterministic fixed-spread target map: input i -> {i, i+in_dim} for 2 connections, {i} for 1
  - lr (log [1e-4,1e-2]), weight_decay (log [1e-6,1e-2]), batch_size (categorical [512,1024,2048,4096])

Objective: best_val from final_metrics.txt; default epoch budget 800 (matches MLP runner). Datasets: housing (8-in), smooth2d (2-in), friedman1 (10-in), friedman2/3 (4-in), all out_dim=1, huber loss. friedman runs get --target-noise-std 1.0.

train_script.py currently exposes no CLI flags for t_span, num_steps, weight_decay, or batch_size (they come from config.py SOLVER/OPTIM). These four flags must be added to train_script.py and wired into batch_size resolution (line ~3129), make_optimizer weight_decay (line ~3723), and per-stage t_span/num_steps overrides (both the dynamic-preset path in _make_dynamic_preset and the preset-propagation path).

Validity is the tricky part: small_world k must be even, 2<=k<num_hidden; fanout targets must be unique across all inputs and < num_hidden (num_hidden >= in_dim*fanout_count). The objective function samples num_hidden first, derives valid k choices, then samples k.

Learned long-range small-world edges via continuous STE indices (src_raw/dst_raw rounded with straight-through estimator) are explicitly DEFERRED to a follow-up plan as the user requested.

Target problems confirmed by user: housing, smooth2d, friedman1, friedman2, friedman3.

## Implementation
- Add --t-span, --num-steps, --weight-decay, --batch-size CLI flags to train_script.py. Wire batch_size into line ~3129 (`batch_size = int(OPTIM['batch_size'])` -> args override), weight_decay into make_optimizer call (line ~3723), and t_span/num_steps into per-stage preset config: add t_span/num_steps params to _make_dynamic_preset (defaulting to SOLVER values, divided by n_stages) AND propagate overrides into active_preset stages after resolution (mirroring the --edge-repeats propagation at line ~3316). Interpret --t-span/--num-steps as TOTAL budgets across stages (per-stage = total/n), consistent with SOLVER semantics.
- Create kn_bayes_opt.py skeleton: DATASETS table (housing in_dim=8, smooth2d in_dim=2, friedman1 in_dim=10, friedman2/3 in_dim=4; out_dim=1; loss huber), permanently-on flag list, run_dir naming (`<dataset>_knet_e<E>`), optuna TPESampler study with sqlite storage + --resume, subprocess-per-trial runner with CUDA_VISIBLE_DEVICES round-robin pinning and CPU fallback (mirror mlp_bayes_opt.py structure).
- Implement the search space with validity constraints. Sample num_hidden from a per-dataset range (min = in_dim*fanout_count, max from a new --num-hidden-max flag, default per-dataset). Sample fanout_count categorical {1,2}; derive fanout JSON with fixed spread (2 conn: input i -> [i, i+in_dim]; 1 conn: input i -> [i]). After num_hidden, sample small_world_k as categorical from even values in [2, min(14, num_hidden-1)]. Sample small_world_p in [0,1], num_stages int [1,3], t_span float [0.5,10.0], num_steps int [10,150], lr log [1e-4,1e-2], weight_decay log [1e-6,1e-2], batch_size categorical [512,1024,2048,4096]. Use optuna.TrialPruned on invalid combos.
- Implement _build_command: assemble the train_script.py subprocess command with all permanently-on flags (--freeze-read --temporal-readout --no-early-stop --leak non-programmable --vca --vca-core --vca-separate-core-bus --cell-library tanh_free --hidden-family small_world --num-hidden N --small-world-k K --small-world-p P --num-stages S --t-span T --num-steps NS --boundary-fan-out <json> --lr --weight-decay --batch-size --epochs E --seed SEED --output <trial_dir> --device <dev>) plus per-problem extras (--target-noise-std 1.0 for friedman problems). Run in repo dir with PYTHONIOENCODING/PYTHONUTF8 and CREATE_NO_WINDOW on win32.
- Implement objective() and artifacts: parse trial_dir/final_metrics.txt (param_count, best_val, best_epoch, epochs_run, elapsed_seconds), set user_attrs, return best_val. On subprocess failure or missing metrics return inf and mark trial failed. Write best_hyperparams.txt, results.csv (all sampled params + fanout_count/fanout JSON + metrics), and objective_history.png (reuse mlp_bayes_opt plotting), all under run_dir.
- Smoke test: run 1-2 CPU trials per dataset (housing, smooth2d, friedman1, friedman2, friedman3) with --epochs 2 --n-trials 1 --n-workers 1 --device cpu to verify train_script accepts every generated flag, the boundary-fan-out JSON passes validation (unique targets, in range), small_world k/num_hidden validity holds, and final_metrics.txt parses. Fix any flag/validation issues found.
- Validate full single-dataset run: one short multi-trial run on friedman2 (e.g. --epochs 50 --n-trials 5 --n-workers 2) to confirm study aggregation, results.csv completeness, best_hyperparams.txt output, and that permanently-on flags (VCA, temporal readout, boundary fanout) produce valid training. Document defaults/ranges in the script docstring; note STE long-range edge learning is deferred to a follow-up plan.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- kn-bayes-opt
<!-- SPECS_END -->