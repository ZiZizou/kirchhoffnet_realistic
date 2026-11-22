---
plan name: post-twin-gates
plan description: Heun, sparsity, mismatch gates
plan status: active
---

## Idea
Post-twin fatal-risk gates before any multistage work. Gate 1: Heun flow-twin sweep over relaxations-per-sample λT in {0.3,1,3,10} with fixed-dt resolution (t_span=1.0, num_steps scaled to hold dt<=0.125), standalone integrator reusing twin instruments, taps8 main + scalar confirm; verdicts: holds at λT<=3 proceed, only λT>=10 records speed requirement, collapse everywhere means digital-only redirect. Gate 2 (parallel): sparsity/device-budget sweep with magnitude-prune-from-dense (radius restored) plus random-sparse at matched counts {100,200,400,625} and one low-rank+sparse row; mismatch sweep with multiplicative Gaussian sigma {0,0.02,0.05,0.10} on W_mix and W_in; heterogeneous operating-point row. Gate 3: sparse structured W_in (localized tap groups) vs dense vs sparse-random with dense mixer. Each gate has pre-registered fatal verdicts that stop downstream lines; no training anywhere; multistage stays stubbed until all three pass with a buildable spec (λT point, edge target, mismatch tolerance, W_in geometry).

## Implementation
- Implement standalone Heun flow-twin (_twin_flow_run, flowtwin CLI, fixed-dt audit) and run λT {0.3,1,3,10} x radius x input-scale sweep taps8 main + scalar confirm
- Implement W_mix sparsity masks (random-sparse + magnitude-prune-from-dense with radius restore) and run matched-count {100,200,400,625} sweep plus low-rank+sparse row
- Implement build-time multiplicative mismatch perturbation and run sigma {0,0.02,0.05,0.10} sweep plus heterogeneous operating-point row
- Implement sparse structured W_in (localized tap groups) vs dense vs sparse-random legs with dense mixer
- Record per-gate verdicts against pre-registered fatal criteria and emit buildable spec or redirect

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- flow-sparse-gates
<!-- SPECS_END -->