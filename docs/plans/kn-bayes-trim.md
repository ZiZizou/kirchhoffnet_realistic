---
plan name: kn-bayes-trim
plan description: Trim and extend BO dims
plan status: done
---

## Idea
Refine kn_bayes_opt.py search space: (1) reduce dimensionality by removing redundant/coupled dims — drop c_eff (degenerate with t_span under the ODE time constant), fix gm_min and isat_min at config defaults, searching only gm_max/isat_max; (2) add 3 new optimization dims that already have train_script CLI support: device_l2_lambda (scale-invariant L2 on tanh_free device/gain params: a_raw/b_raw/gm_raw/isat_raw/theta_raw/g_resistive_raw + VCA + OutputAffine, log 1e-7..1e-3), freeze_boundary {0,1}, and freeze_temporal_read {0,1}. train_script.py already has --device-l2-lambda (train_script.py:2353, default 0.0), --freeze-boundary (train_script.py:2770), --freeze-temporal-read (train_script.py:2780) all wired through to build_net_from_preset and compute_loss, so no train_script.py changes are needed. End result: 18 total dims (fanout_count, num_hidden, small_world_k, small_world_p, num_stages, t_span, num_steps, lr, weight_decay, batch_size, x_max, gm_max, isat_max, sparsity_lambda, entropy_lambda, device_l2_lambda, freeze_boundary, freeze_temporal_read). Also remove the now-obsolete gm_max<=gm_min / isat_max<=isat_min TrialPruned checks (mins are fixed at config). Deliver a final list of everything optimizable to the user.

## Implementation
- Edit kn_bayes_opt.py argparse: remove --c-eff-min/--c-eff-max, --gm-min-min/--gm-min-max, --isat-min-min/--isat-min-max bound args; add --device-l2-lambda-min (default 1e-7) and --device-l2-lambda-max (default 1e-3).
- Edit kn_bayes_opt.py objective(): delete c_eff, gm_min, isat_min sampling; remove the gm_max<=gm_min and isat_max<=isat_min TrialPruned blocks; add device_l2_lambda (log), freeze_boundary (categorical {0,1}), freeze_temporal_read (categorical {0,1}).
- Edit kn_bayes_opt.py _build_command(): remove --c-eff/--gm-min/--isat-min emission; add --device-l2-lambda always; emit --freeze-boundary only when freeze_boundary==1; emit --freeze-temporal-read only when freeze_temporal_read==1.
- Update _build_command signature + objective() call site for the removed c_eff/gm_min/isat_min args and new device_l2_lambda/freeze_boundary/freeze_temporal_read args.
- Update kn_bayes_opt.py module docstring and best_hyperparams.txt header/artifact comments to reflect the new 18-dim space.
- Run py_compile on kn_bayes_opt.py; run a train_script.py smoke with --device-l2-lambda + --freeze-boundary + --freeze-temporal-read to confirm flag forwarding still works end-to-end.
- Run kn_bayes_opt.py 1-trial CPU smoke on friedman2 (--num-hidden-min 14 --num-hidden-max 16); verify all new dims sampled, freeze flags appear in the trial cmd log, config_snapshot shows device_l2_lambda, and metrics parse.
- Update project memory block kn-bayes-opt with the trimmed 18-dim space + new flags; report final optimizable list to the user.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- kn-bayes-trim
<!-- SPECS_END -->