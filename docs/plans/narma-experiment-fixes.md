---
plan name: narma-experiment-fixes
plan description: Remediate NARMA experiment audit issues
plan status: done
---

## Idea
Fix the 9 issues found in the NARMA experiment audit. Order by severity: P0 (input scaling, bptt_window, MC targets), P1 (dead params, decision rule gating/thresholding), P2 (docstring cleanup, ESN tuning).

## Implementation
- Add scale_input_to_rails() calls in run_baselines() and run_fabric_condition() data pipelines — both train and test splits
- Increase default bptt_window from 15 to 50; add --bptt-window CLI arg to narma_experiment.py
- Fix memory_capacity() call in run_fabric_condition() to pass u_train instead of y_train
- Remove unused hidden_dim and num_steps_per_sample params from run_fabric_condition() signature
- Gate decision rule on args.order == 20 to match spec (NARMA-20 only)
- Add absolute-NRMSE guard to decide(): if fr_off_mean > 0.5 return NEUTRAL
- Remove narma_curves.png from docstring (no plotting code exists)
- Add ESN grid search: loop over input_scaling=[0.2,0.5,1.0], leak=[0.3,0.5,1.0]; pick best val NRMSE

## Required Specs
<!-- SPECS_START -->
- narma-experiment-fixes
<!-- SPECS_END -->