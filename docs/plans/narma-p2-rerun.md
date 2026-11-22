---
plan name: narma-p2-rerun
plan description: Expanded E0 probe rerun
plan status: active
---

## Idea
Rerun only the failed P2 E0 probe leg after the Alliance argument-parsing failure. Keep P0 and P1 unchanged. Verify the local and server batch files, normalize submission formatting, expand the E0 schema with hidden dimension, parameter count, washed-state participation ratio, and full transient gate-zero statistics, and run the complete locked 36-corner gain/leak/drive grid at both hidden-25 and hidden-49. Apply the agreed Ridge-parity-first gates, paired cross-width comparisons, and the post-sweep decision tree, while preserving P2 as zero-training and eval-only.

## Implementation
- Verify the local and server Slurm batch copies by checksum and line endings, normalize the authoritative submission copy, and use robust grid-argument binding.
- Add hidden-dimension plumbing through fabric construction and the E0 CLI while retaining hidden-25 as the canonical default.
- Expand the E0 row, CSV, JSON, and text schema with hidden dimension, parameter count, state participation ratio, complete gate-zero statistics, configuration tags, and PASS/FAIL fields.
- Implement the transient boundary-only trajectory with guaranteed gate restoration and identical Ridge, memory-capacity, participation-ratio, rail, and Jacobian instruments.
- Update the audit suite for the expanded schema, hidden-49 construction, transient gate restoration, observed-transition Jacobian replication, exact corner traversal, and revised thresholds.
- Run local compile checks, the audit suite, and small one-corner CPU smoke tests for hidden-25 and hidden-49 without running the full grids locally.
- Submit a dedicated P2-only Alliance batch with a one-corner GPU preflight, followed by full hidden-25 and hidden-49 sweeps and paired cross-width comparison.
- Apply the Ridge-parity-first decision tree and record thresholds, code/batch provenance, results, and follow-up actions in project memory and the problem statement.

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- p2-rerun-probes
<!-- SPECS_END -->