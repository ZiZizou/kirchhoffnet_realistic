---
plan name: narma-advisor-probes
plan description: Advisor-ordered memory-failure probes
plan status: active
---

## Idea
Execute the external advisor's prescribed sequence against the NARMA-10 plateau (test NRMSE 0.66-0.68 vs 0.15-0.29 band; ridge-untrained 0.7194; MC 0.2; carryover load-bearing but informationally empty): (1) in-harness baselines calibration with the existing --baselines-only path, (2) frozen-state ridge plus small-MLP probe with participation ratio and gate-zero add-on on the existing k2/k8 checkpoints, (3) zero-training 36-point gain/leak/drive sweep with washout-corrected MC, ridge-from-full-states, Jacobian eigenvalue instrument, and rail guard. Pre-registered decisions: ESN far above band halts everything to task-setup; probe ridge below 0.45 promotes the dense-readout fix; probe ridge at/above 0.6 leaves init-regime sole primary; a sweep corner with MC above 3 and ridge below 0.5 triggers a one-time init search plus one confirmation train; a flat sweep opens structural talk with the systolic delay-line as the only candidate. Depth-per-sample stays paused, E5 stays dropped, all submitted as one sbatch batch.

## Implementation
- Run baselines calibration via the existing --baselines-only path (ridge, ESN, MLP, LSTM through identical pipeline/metric/seeds) and apply the halt-or-continue decision (ESN ~= 0.6 stops everything)
- Implement the frozen-state probe harness (eval-only: ridge and small-MLP fits on full hidden states plus 5-lag stack, participation ratio, gate-zero add-on) and run it on the existing k2 and k8 checkpoints
- Implement the build-time gain override hook (deterministic gm_raw/isat_raw overwrite, per-node randomized raw_leak init, existing input-scale for drive) plus the Jacobian eigenvalue instrument and washout-corrected MC probe path
- Run the locked 36-point E0 sweep (gain x leak x drive, eval-only) with per-point MC/ridge/eigenvalue/rail rows and PASS/FAIL tags
- Apply the pre-registered sweep decision (init search plus confirmation train on a passing corner, or structural-talk escalation on a flat sweep) and record all outcomes in the crossroads problem-statement plus project memory

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
<!-- SPECS_END -->