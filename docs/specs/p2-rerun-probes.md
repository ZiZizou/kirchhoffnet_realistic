# Spec: p2-rerun-probes

Scope: feature

# P2 Rerun Probes

Scope: feature

## Purpose

Rerun only the failed P2 E0 probe leg with an expanded, Ridge-parity-first instrument set. P0 and P1 are already complete and are not recomputed. The rerun remains zero-training and eval-only; it screens untrained reservoir configurations before any confirmation training is authorized.

## Locked experimental scope

- Task: NARMA-10.
- Seed: 0.
- Device: CUDA on Alliance.
- Canonical integration settings:
  - `t_span = 1.0`
  - `num_steps = 8`
- Training-stream construction:
  - 4 canonical streams;
  - 2,500 samples per stream;
  - probe uses stream zero;
  - washout: first 200 samples.
- Full locked traversal:
  - gain: `-5, -2, 0, 1.5`
  - leak: `slow-fixed, randomized, 0.15`
  - drive: `0.25, 0.5, 1.0`
- Full grid at both:
  - hidden-25: 36 corners;
  - hidden-49: 36 corners.
- Total: 72 normal-trajectory corners.

## Required per-corner measurements

For every corner, collect:

1. One normal full-state trajectory.
2. One transient boundary-only trajectory with core-edge gates suppressed.
3. Identical instruments on both trajectories:
   - Ridge-from-states NRMSE and \(R^2\);
   - Washout-corrected memory capacity;
   - Washed-state participation ratio;
   - Rail fraction and maximum \(|x|/x_\mathrm{max}\);
   - Jacobian eigenvalue statistics at observed transitions.
4. Exact trainable parameter count.
5. Hidden dimension.
6. Row-level configuration tag.
7. PASS/FAIL status.

The boundary-only pass must save and restore all suppressed gate parameters, including on error. It must not permanently mutate the corner’s network.

## Revised Ridge-parity gates

A normal-trajectory corner passes when all of the following hold:

```text
ridge_nrmse <= 0.40
memory_capacity > 3
state_participation_ratio >= 6
rail_fraction <= 0.05
```

A boundary-only corner uses the analogous gate-zero fields.

The `0.40` Ridge rule is parity-with-tolerance against the `0.3777` raw-delay Ridge baseline. It is a screening gate, not certification.

## Reporting requirements

Separate canonical outputs are required for:

```text
outputs/advisor_probes/e0_full_h25/
outputs/advisor_probes/e0_full_h49/
```

Each width must contain:

- `e0_sweep.csv`
- `e0_sweep.json`
- `e0_sweep.txt`

Reports must preserve the hidden-25 canonical PASS statistics separately from hidden-49 results. Cross-width comparisons must pair identical gain, leak, and drive corners and report changes in Ridge, MC, participation ratio, and rail behavior.

## Acceptance

The P2 rerun is accepted only when:

- Both 36-corner grids complete.
- Normal and boundary-only trajectories are present for every corner.
- Gate parameters are restored after every boundary-only pass.
- Every row has a configuration tag and PASS/FAIL status.
- Audit coverage includes the expanded schema, hidden-49 construction, transient gate restoration, exact corner traversal, and revised thresholds.
- Results are recorded with thresholds, code/batch provenance, and the applicable post-sweep decision.