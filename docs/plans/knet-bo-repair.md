---
plan name: knet-bo-repair
plan description: Repair KNet BO sample efficiency
plan status: active
---

## Idea
Fixed-KNet BO at budget 5000 trails MLP 20x (best KNet val MSE ~0.007-0.01 vs MLP 0.0003; manual 6775-param KNet reaches 0.00073). Root causes: (1) BO trains every trial with diagonal boundary fanout {i:[i,i+4]} while the proven manual run uses the validated harness-default map, so BO never searches the good connectivity; (2) search space too wide for the evaluation cost — 99 KNet arches + x_max[2,6] + lr[2e-4,5e-3] + batch{128..1024} vs 32 MLP arches, and KNet trials cost 30-60min vs MLP 25s, so 11 trials in 6h cannot cover the space and low-LR/large-batch corners (all observed catastrophes: lr<7e-4 or batch=1024 give 60-70% failure) waste the budget; (3) no seed trial and no early stopping, so stragglers burn full 100 epochs. Fix within hard budget [4250,5000]: pin validated fanout, narrow LR/batch/x_max, bias the feasible arch set toward user priors (stages 4-5, k=4, rank>=2-3, small hidden), add a seeded trial, and add TPE-safe multi-fidelity (25-epoch gate, no TrialPruned because multi-obj TPE crashes on None values per optuna#5260 — stragglers return finite 25-epoch values, survivors resume to 100ep via harness --resume). Calibrate the gate threshold from rank-correlation of 25ep vs 100ep on a few arch/LR combos before full rollout.

## Implementation
- Audit sampler vs harness defaults: confirm diagonal fanout override, x_max/lr/batch ranges, and analytic-vs-actual param mismatch in fixed-distillation-bayes-opt.py sample_config/base_command
- Mine outputs_alliance_can priors: tabulate winning friedman1/ctle KNet configs (stages, hidden, k, rank, lr, batch, x_max, fanout) and derive constrained feasible arch list for [4250,5000] (stages 4-5, k=4, rank 2-4, hidden 8-14) with exact build-based counts
- Implement sampler hotfix: validated fanout everywhere (sampler + analytic + preflight), lr low bound 1e-3, batch {128,256}, x_max fixed 4.0 (or narrow [3.5,4.5]), enqueue proven manual-style seed trial
- Add TPE-safe multi-fidelity gate: 25-epoch probe then resume-to-100 for survivors; stragglers return finite probe values (never TrialPruned/inf); gate threshold from 25-vs-100 rank correlation pilot on 4-6 arch/LR combos
- Validate: py_compile + test_bo_guard update (seed, ranges, gate, fingerprint), fresh --output smoke for knet (2-3 trials incl. seed + one gated kill) and mlp parity check, then full 30-trial KNet run on Alliance with results.csv/bo_summary comparison to MLP trial 8 (0.00035)

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- plain-mlp-config
<!-- SPECS_END -->