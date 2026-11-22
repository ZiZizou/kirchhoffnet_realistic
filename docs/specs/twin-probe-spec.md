# Spec: twin-probe-spec

Scope: feature

# Spec: twin-probe-spec (feature spec for plan twin-node-tanh)

Scope: feature

## Purpose

Pin the digital node-tanh twin probe requirements so the build, audit, and Alliance run share one contract. The twin tests the untested bottom-right cell: dense linear mixing -> sum -> per-node tanh, `x = (1-a)*x + a*tanh(W_mix@x + W_in@drive)`, with retention carried by the mixing matrix spectral radius (not leak gymnastics). Neither rung-1 (leak-dominated, per-edge tanh in linear regime) nor C1b (pure linear, no nonlinearity) tested this object.

## Construction (locked)

- Standalone discrete map in `narma_linear_controls.py` (new `twin_node_tanh()`); NO stage/cell-library mutation, NO rhs override. Heun-integrated rhs-override variant is explicitly deferred future work.
- `W_mix`: dense Gaussian, normalized to target spectral radius (verify normalization in audit; QR-orthogonal optional variant, not required).
- Two drive harnesses, BOTH run unconditionally: (a) 25x8 eight-tap delay-bank first; (b) 25x1 scalar C1-style (`u_scaled`, drive_scale=1.0, G_in RMS-normalized) second.
- Tanh-engagement guard: assert operating args are NOT in the ~1e-2 linear regime (audit + runtime check); a twin running as accidental-linear is void.
- Shared instruments only: `_per_delay_mc` (SVD fallback), `_ridge_fit_predict`, `_raw_delay_ridge`, `_safe_participation_ratio`, `_standardized_state_pr`; in-harness ESN parity ref (`ne.ESN`) on identical streams; Jacobian eig spectrum via autograd on the discrete map.

## Grid (locked)

- radius {0.7, 0.9, 1.0, 1.1} x leak a {0.3, 1.0} x input scale {0.2, 0.5, 1.0} x seeds {0, 1, 2} = 72 corners x 2 harnesses, eval-only, NARMA-10, 2500 samples, washout 200, max_delay 20.

## Activity discipline

- Twin tanh states live in [-1,1] (rails N/A per C0 convention). Activity screen defined in tanh units against the ESN parity ref's activity (not rail fraction); input-only calibration stream reused where applicable.

## Gates and verdict (pre-registered)

- PASS (architecture inversion confirmed): twin lands near ESN numbers (NRMSE ~0.3, MC 8-12) on either harness -> promote analog realization (linear-OTA mixer + diff-pair node tanh) and the multistage/anneal-prune program.
- FAIL (fabric exonerated): C1b-like numbers (MC ~1-3, ridge >= ~0.6) on both harnesses -> pivot to harness/task-mapping investigation; do NOT proceed to analog cell design.
- VOID (not negative): tanh-engagement guard tripped, non-finite states, or determinism audit failure -> fix harness, no verdict.

## Reporting

- Per-corner JSON+CSV+TXT matching C1 conventions (config tags, seeds, per-delay MC, Jacobian spectrum, activity, ESN parity row); batch summary table; verdict entry appended to docs/narma-progress-summary.md.