---
plan name: twin-node-tanh
plan description: Digital ESN-architecture probe
plan status: active
---

## Idea
Test the untested bottom-right cell (dense linear mixing -> sum -> per-node tanh, x=(1-a)x+a*tanh(Wx+W_in*u)) as a standalone discrete-map digital twin in narma_linear_controls.py, reusing the C1/C1b instrument set (ridge/MC/PR/Jacobian) and ESN parity ref. Two drive harnesses: 25x8 eight-tap bank first, scalar C1-style second; both run unconditionally. Grid radius {0.7,0.9,1.0,1.1} x leak a {0.3,1.0} x input scale {0.2,0.5,1.0} x seeds {0,1,2}, eval-only, one GPU-day. Heun rhs-override variant explicitly deferred as future work. Pre-registered verdict: near-ESN numbers (NRMSE ~0.3, MC 8-12) confirm the architecture inversion and promote analog realization + multi-stage/anneal-prune program; C1b-like failure exonerates the fabric and indicts harness/task-mapping.

## Implementation
- Implement twin_node_tanh() in narma_linear_controls.py as standalone discrete map x=(1-a)x+a*tanh(W_mix@x+W_in@drive) with dense Gaussian W normalized to target spectral radius, two drive paths (25x8 tap-bank first, 25x1 scalar C1-style second), reusing _per_delay_mc/_ridge_fit_predict/_raw_delay_ridge/_safe_participation_ratio and in-harness ESN parity ref on identical streams
- Add twin CLI subcommand with --radius/--leak-a/--input-scale/--harness/--w-seed/--win-seed flags plus JSON+CSV+TXT artifacts matching C1 report conventions (config tags, per-delay MC, Jacobian eig spectrum via autograd on discrete map, activity/rails in tanh units, ESN parity row)
- Extend probe_audit2.py with twin checks: spectral-radius normalization correctness, determinism at fixed seeds, tanh-engagement assertion (reject args ~1e-2 linear regime), standalone purity (no stage/cell-library mutation), CLI help smoke
- Local CPU verification: py_compile all touched files, audit suite green, thin 300-sample smoke of both harnesses reproducing ESN-parity plumbing without claiming inference
- Write Alliance batch narma_twin_batch.sh (gitignored, ship via git add -f): preflight tiny twin + full 24-corner x 2-harness x 3-seed grid with per-leg timeout, resume-skip, and summary table; one GPU-day budget
- Analyze outputs against pre-registered gates and append verdict entry to docs/narma-progress-summary.md (twin vs ESN 0.296/MC 12 vs C1b MC 1.62); on pass, promote mixing-bank realization + multistage + anneal-prune program, on fail pivot to harness/task-mapping investigation
- Record Heun-integrated rhs-override twin variant as deferred future work with rationale (discrete map decides architecture first; integrator fidelity second)

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- twin-probe-spec
<!-- SPECS_END -->