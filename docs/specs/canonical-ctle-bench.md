# Spec: canonical-ctle-bench

Scope: repo

# Canonical CTLE Benchmark (REPO spec)

Neutral, teacher-comparable CTLE inverse-design protocol shared by `kn_bayes_opt.py`
and `mlp_bayes_opt.py` CTLE branches and all dagger harnesses.

## Dataset
- `create_canonical_flow_dataset` in `ctle_dagger_common.py`: X specs via
  `sample_validation_specs` (boundary_ratio=0.5, fixed seed) → flow
  `label_batch(top_k=8, valid_threshold=0.9)` → **min-predicted-power** pick among
  ZIG-valid candidates (0.5 gate, 0.3 relaxed fallback) → frozen 80/10/10 split.
- `.npz` schema identical to DAgger `DistillationDataset` entries; sha256
  fingerprint covers seed + teacher identity + top_k + tie-break + boundary_ratio.
- The min-power rule is frozen before either BO runs and never tuned per student.

## Training (Phase A)
- Static data, Huber-on-log-params only (delta 0.5, the `L_imit` term of
  `RegimeAwareLoss`); no relabeling, no scheduler restarts, no surrogate-through terms.
- KNet harnesses use the Friedman recipe: differential LR groups
  (mapper 1.0 / structural 4.0 / dynamic 1.0), `OutputAffine` bias init to
  train-logit means (gains 1, nonlinear branches 0), `gm_max`/`isat_max` exposed
  (default 10.0), batches to 2048. All param-neutral (`--count-params-only`
  bit-identical).
- Shared eval: `StudentEvaluator` on frozen COMMON_EVAL seed; best-failure checkpointing.

## BO protocol
- Budget 7000/7000, 30 trials each, 300 epochs, seed 100, no halving/pruning.
- Objective = **validation** failure rate for both drivers (test reserved for
  frozen finalists). Sampler fingerprints bumped on any search-space change.
- Selection: lexicographic validation failure, then validation logit MSE.

## Reporting
- `results.csv` + E0 diagnostics: per-head teacher variance vs per-head student MSE,
  per eye-pair co-failure matrix, boundary/interior split, ZIG-gap verdict (3pp).
- Decision gate: KNet parity ⇒ optimization/convergence story; ≥2× loss with full
  recipe ⇒ H3 dead, readout-rank probes become part 2.