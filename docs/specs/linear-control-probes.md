# Spec: linear-control-probes

Scope: feature

# Linear Reservoir Controls

Scope: feature

## Purpose

Localize the NARMA-10 memory suppressor with calibrated controls before any further sweeping, training, or architectural build. This spec governs the R0 reconciliation matrix, the C0 ESN calibration control, the C1 linear-reservoir-in-fabric control at hidden-49, and the C2/C3/C4 single-element bisection. All controls are zero-training and eval-only.

## Locked control scope

- Task: NARMA-10, seed 0, CUDA on Alliance.
- Width: hidden-49 (7x7 torus) for C1 through C4 immediately.
- Canonical integration: `t_span = 1.0`, `num_steps = 8`.
- Training-stream construction: 4 canonical streams, 2,500 samples per stream, probe uses stream zero, washout 200.
- Compute is not a constraint: collect maximal instrumentation on every control.

## R0 reconciliation matrix

On one fixed seed and stream, vary one factor at a time and publish Ridge deltas for:

1. Legacy hidden-only ridge diagnostic versus E0 full-state path, identical net and data.
2. k8-frozen core versus fully dynamic core.
3. Drive scale 1.0 versus 0.5.
4. Stream length 10,000 versus 300 at fixed washout 200.
5. Raw-delay Ridge recomputed on the exact E0 stream, washout, and target convention.

After R0, freeze a matched parity rule (proposed: E0 Ridge at most matched raw-delay Ridge plus 0.03). No gate is re-anchored until R0 lands.

## C0 ESN calibration control

Feed the P0 ESN hidden states through the identical instrument set: Ridge, washout-corrected MC with per-delay curves, participation ratio, and Jacobian spectra at observed transitions, with the same stream, washout, delays, and regularization. Rail fraction is reported as not applicable with an explicit note, since ESN tanh states live in [-1, 1].

- ESN MC around 3-8 with PR around 15-25: gates calibrated, fabric verdicts stand.
- ESN MC around 0.5: pipeline or task-scaling problem; revise all MC-based gates before drawing conclusions.

## C1 linear-reservoir-in-fabric control

Convert a hidden-49 fabric into a linear reservoir:

- Core: resistive-only currents with random per-edge conductances; tanh path bypassed; no compliance gating on core edges.
- Input: boundary OTAs fully gated off; dense linear injection with random weights rescaled so RMS over nodes and time equals the RMS of the canonical drive-1.0 volt sequence.
- Leak: randomized per-node leaks at fixed seed 0, held constant across tuning iterations.
- Rails: clip current zeroed and effective rail pushed far outside the operating range.
- Spectral targeting: iterate conductance rescaling until measured per-sample Jacobian spectral radius lands in [0.93, 0.97] at 3 or more observed transitions.
- Reporting: Ridge, MC, and PR on both hidden states and full states.

PASS is washout-corrected MC of at least 3, with PR of at least 6 and Ridge within matched parity as secondary confirmations.

- Pass: integrator, carryover, washout, collection, and linear memory physics are sane; the analog cell stack is the suppressor.
- Fail: harness or integrator bug; all prior conclusions become suspect until root-caused.

Implementation constraint: build a `linear_core` mode or bypass flag plus a linear injection path. Gate-zero overrides alone cannot build C1 because the shared edge gate also kills the resistive shunt. Never edit `cell_library.py` init defaults.

## C2/C3/C4 bisection

Restore exactly one analog element at a time onto the passing C1, holding seed, stream, washout, and instruments fixed:

- C2: boundary-OTA input path with the standard full-rail mapping.
- C3: tanh_free core at the best-sweep gain.
- C4: compliance and clip rail handling.

The first restoration collapsing MC and PR is the suppressor. If C2 alone collapses it, the input-times-gain coupling hypothesis is confirmed directly.

## Reporting requirements

Per-control JSON and CSV artifacts with config tags, thresholds, seeds, spectral measurements, per-delay MC curves, Jacobian spectra, saturation histograms, and gate-zero comparisons. Paired C1-versus-restoration deltas for Ridge, MC, PR, and rail behavior.

## Acceptance

- R0 delta table published and matched parity rule frozen.
- C0 calibrates or forces revision of the gates.
- C1 passes MC of at least 3, or its failure is root-caused as a harness bug.
- C2/C3/C4 localize the suppressor, or their joint behavior is documented as the next escalation.
- No training run starts until a control passes the E0-equivalent gate.
- Results, thresholds, code provenance, and follow-up actions recorded in project memory and the problem statement.