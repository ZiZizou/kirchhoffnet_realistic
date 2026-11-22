---
plan name: knet-bo-round1
plan description: Two-rung seeded distillation search
plan status: active
---

## Idea
Run a fresh 20-trial fixed-distillation KNet BO study on Alliance that closes the 5x gap (BO-best val_logit_mse 0.00397 vs manual 0.00073) found in the outputs_alliance_can/study.db autopsy. Replace the fixed 0.035 probe bar (which killed 7/13 strict-phase trials inside the noise band) with two-rung rank-based halving (15ep -> 100ep, promote top third vs same-rung peers, finite returns only so the optuna#5260 multi-objective TPE crash guards hold). Expand the space to 6 TPE-native dims (arch categorical over a ~6000-budget feasible set admitting hidden-14, plus lr, int batch, x_max, weight_decay, new solver_scale harness flag) with multivariate+grouped TPE, seed-enqueue the manual winner and DB trial 3 by arch tuple, extend the sampling fingerprint to cover rung config (closing the mid-study threshold-change hole), and validate via count-invariance check plus CPU smoke before the ~5-6h Alliance launch. TuRBO deliberately rejected (no categoricals, needs network fetch, unvendored); gm/isat and small_world_p deferred to round 2.

## Implementation
- Add --kn-solver-scale to fixed-mlp-distillation-kirchhoffnet.py KirchhoffStudent (scale per-stage t_span and num_steps together, default 1.0) and verify --count-params-only invariance across solver_scale/x_max/wd/batch corners for two fixed arches
- Thread sampled values through fixed-distillation-bayes-opt.py base_command: --weight-decay (today forced 0), --kn-x-max (today default 4.0), --batch-size int [128,512]; extend sample_config with the 5 continuous/int dims and rebased ~6000-budget feasible arch list
- Implement --seed-trial-from (repeatable, tuple-based arch re-resolution, two seeds: manual (14,4,4,2) lr1e-3 b256 and DB trial-3 (10,3,4,4) lr0.0033 b128, seeds always promote) and switch sampler to TPESampler(multivariate=True, group=True, n_startup_trials=4)
- Replace fixed probe gate with two-rung rank halving (15ep rung 1, 100ep rung 2 via existing --resume chain, promote iff lexicographic (failure,mse) rank <= max(1,N/3), first ~4 always promote, finite rung returns only) and extend fingerprint to sampler+rung+dims+seeds
- Extend live artifacts (trial_table.txt / results_live.csv / trial_log.jsonl) with new dims, rung, rank and promoted fields; keep results.csv/bo_summary.json schemas backward compatible
- Validate locally: py_compile all touched files, count-invariance check, 2-trial CPU smoke of full sample->rung->promote path, probe_study_db.py clean on smoke DB
- Launch fresh Alliance study (new --output, 20 trials, teacher trial_0019, shared fixed dataset path, --param-budget ~6000) and document the --resume procedure for allocation-boundary kills
- Post-study analysis: compare best in-window val_logit_mse against 0.001 bar (manual 0.00073, MLP 0.00035), run 7-dim per-target breakdown, record go/no-go for round 2 (readout/weighting vs wider space)

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- plain-mlp-config
- fixed-bo-round1
<!-- SPECS_END -->