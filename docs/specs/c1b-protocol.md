# Spec: c1b-protocol

Scope: feature

# C1b Orthogonal-Mixing Revision (feature spec for plan c1b-revision)

## Purpose
Falsify-or-confirm the weak-mixing hypothesis for C1's failure (MC 2.53 at PR 1.1, J in band) with a linear reservoir whose modes decay uniformly slowly, then translate the result into real fabric parameterization.

## Gate table (revised; linear controls only)
- Memory gate: washout-corrected MC >= 4 (≈2× inside the C0 in-harness reference MC_ref = 8.07).
- Operating-regime gate: rail fraction ≤ 5% (unchanged).
- Spectral gate: per-sample Jacobian max|λ| in the run's J-band (C1b default [0.97, 0.99]).
- Parity (E0 Ridge ≤ raw-delay + 0.03): reported as diagnostic ONLY for linear legs — a tapped delay line is the optimal finite-window linear basis, so ridge-on-taps ≥ ridge-on-any-linear-reservoir by construction. Retained as a pass gate for nonlinear fabrics only.
- PR: diagnostic only (a healthy MC-8 ESN scores raw PR 3.02, falsifying PR ≥ 6). Report both raw PR and standardized-state PR (zero-mean/unit-variance per node before covariance).

## C1b-override protocol
- W: QR-of-seeded-Ginibre orthogonal matrix (50×50 incl. output node), auto-scaled by the existing spectral tuner to the J-band. Pre-registered expectation: MC 5–15 if integrator/carryover path is sound.
- Leak: uniform 0.05, non-programmable (log the param-count shift in the report).
- Injection: unchanged dense linear path rescaled to drive RMS.
- PASS = MC ≥ 5 AND rails ≤ 5% AND J in band. Decision fork: pass → C2/C4 degradation tests on the C1b base; fail → root-cause Heun/carryover/state-handling (instruments proven by C0).
- Tuned base persistence (required): C1 writes tuned W + leak vector + G_in to a .pt sidecar (JSON pointer); bisection legs load it. Reinstalling an untuned base in the legs is a protocol violation.

## C1b-real protocol (resistive core, no rhs override)
- Core gm/isat → min (shunt dominates ~1000:1); per-edge G from softplus-inverse targets, J-tuner reused against the real rhs; boundary OTAs in small-signal linear regime; uniform leak. Same C1b gates. Rationale: symmetric Laplacian is normal with orthogonal eigenvectors — the maximal-linear-memory construction expressed natively.

## C3 redefined (tanh crossover)
- On the passing C1b base, sweep core gm_raw fills (default grid mirroring E0 gains) and track the gain where nonlinearity arrives before MC collapses. That crossover is the operating point the real fabric init needs.

## Instrument hardening
- Per-delay MC: standardize features per delay, keep λ=1e-2, truncated-SVD fallback on ill-conditioned solves (the delay-19 −159 class must be impossible).
- Protocol stream constant: ≥2500 samples, washout 200; log effective sample counts; never compare Ridge/MC across stream lengths.
- R0 canonical reference: long-stream leg (0.719, cross-validated vs independent 0.7194).

## Re-measurements
- Local CPU: targeted high-gm E0 corners under fixed MC; k2 frozen-state MC recollect on the full stream (apples-to-apples vs C1 ridge 0.6179).
- Alliance: k8 frozen MC recollect; full P2 E0 grid under fixed MC.