# Spec: advisor-probe-suite

Scope: repo

# Spec: advisor-probe-suite (repo scope)

Zero-training probe discipline for reservoir-memory experiments, prescribed from the external advisor review of the NARMA-10 plateau (ridge-untrained 0.7194 ~= trained 0.676; MC ~= 0.2; carry-mask ablation proving carryover load-bearing but informationally empty). Reusable across NARMA and any future echo-state-regime work.

## Baselines calibration (always first, no code)
- Run the in-harness baselines (ridge taps, ESN, MLP, LSTM) through the identical data pipeline, metric, seeds, and washout as fabric legs before any fabric probe is interpreted.
- Decision: ESN at or below the success band (NARMA-10: 0.3) -> band calibrated, fabric guilty; ESN far above band (~0.6) -> halt everything, task-setup bug hunt (standardization, washout, target construction).

## Frozen-state probe (eval-only, existing checkpoints)
- Feature sets: (a) all hidden states, (b) 5-lag state stack. Models: ridge + small torch MLP (single hidden layer ~32 units, few-hundred-epoch fit, <=5 min each; tests information presence, not trainability).
- Instruments: participation ratio PR = (sum λ)^2 / sum λ^2 of the washed state covariance (PR ~= 1-3 confirms rank collapse); gate-zero add-on (core-edge gates forced to zero at eval; no NRMSE move -> fabric decorative, boundary-to-accumulator feedforward does all work).
- Decision: ridge-from-states below 0.45 -> readout fix (dense read over hidden nodes) leads; at or above 0.6 -> init-regime (H1) sole primary.

## E0 gain sweep (eval-only, build-time gain override)
- Post-build, pre-training, deterministic-under-seed override of gm_raw/isat_raw (core + boundary + readout libs symmetrically unless a leg isolates one family); per-node randomized raw_leak init; drive via existing input-scale flag. Locked grid: gain {-5,-2,0,1.5} x leak {slow-fixed, randomized, 0.15} x drive {0.25,0.5,1.0} (~36 points, minutes each).
- Per point: washout-corrected MC (discard first 200 states before ridge fits; the legacy no-washout collector is grandfathered for backward comparison only and must be labeled), ridge-from-full-states, Jacobian eigenvalue instrument, rail fraction.
- Jacobian instrument: discrete single-sample map F: x -> x' (one full Heun window at fixed u, mid-stream operating points, >=3 samples per config) via torch.autograd.functional.jacobian (state dim ~26, cheap). Healthy corner: max |lambda| ~= 0.95-1.05 with spread magnitudes; rank-1 symptom: one eigenvalue ~= retention, rest ~= 0. Rail guard: rail fraction above 5% disqualifies a corner regardless of MC.
- Gain-override contract: logged as gm_init/isat_init/leak_mode in every probe row; cell_library.py init defaults are never edited (init discipline stays in one place).
- Decision: some corner with MC above 3 and ridge below 0.5 -> H1 confirmed; fix is one-time init search plus leak/gain diversification (no architecture change), then a single confirmation train. No corner breaks ridge 0.6 -> compliance/boundary-drive structurally suppresses the echo-state regime; only then talk architecture (systolic delay-line candidate; depth-per-sample stays paused).

## Threshold discipline
- All gates are relative deltas on single-seed eval-only numbers (screening, not certification); every probe row carries config tag plus PASS/FAIL. Probe outputs land in the leg output dir next to (not inside) training tables.