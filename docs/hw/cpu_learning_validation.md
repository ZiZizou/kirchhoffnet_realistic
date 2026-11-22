# CPU hardware-learning validation (W7)

## Setup and scope

Ran the slow CPU harness in `tests/test_hw_learning.py` against the golden-builder
`FRIEDMAN2_NET` (4→1, boundary fan-out and `ota_mesh`/output-ODE readout) and
`TOY_NET` (4→3, boundary fan-out and shared-sense/GLN readout). Each run used
float64, Adam, 256 train rows, 256 held-out rows, 400 full-batch steps, and
seeds 0/1/2. The primary run used `lr=0.01`, the harness default. Data and
initialization were unchanged; no golden files were regenerated.

`pytest-xdist` is not installed on ssr0 (`pytest -n` is rejected), so the two
tasks ran as separate `nohup` pytest processes, capped at 12 PyTorch threads
each (24 total). Results were written to `/tmp/w7learn`. I reran the full slow
assertion set from those fresh results after fixing the rail-check bug; the
cache schema is now versioned so stale JSONs from an older harness cannot be
silently reused. Corrected run summary: Friedman2 25 passed / 5 failed; Toy 17
passed / 10 failed (42 passed, 15 failed of 57 selected tests; the two
Friedman2 physical-build checks are included). The default suite then passed
**682 tests** with 57 slow tests deselected (`python -m pytest -q`; 15 warnings).

Table fields: `MSE / none` is the held-out MSE divided by the *same-seed*
unconstrained MSE (diagnostic; acceptance uses the ratio of the mean MSEs over
seeds). `Rail` is observed max |x| / 1.05·x_hard. `†` marks phase3a_core:
that preset has no `hw_compliance`, so the rail value is diagnostic and is not
asserted. `dt·λ` is the maximum reported value / 2. The stiffness metric samples
32 held-out inputs. `Quant` is the largest code-grid distance; `—` means no
quantized parameter is enabled. Rows for `none` are the unconstrained reference;
their hardware-limit columns are descriptive only. Nonfinite is the harness
loss/gradient count; all rows also had finite held-out outputs.

## Full results (lr=0.01)

| Task | Preset | Seed | Init | Final train | Final/init | Held MSE | Held R² | MSE / none (same seed) | Nonfinite / y finite | Slope max/limit | Rail max/limit | max dt·λ / 2 | max quant off-grid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| friedman2 | none | 0 | 0.9842 | 0.0019868 | 0.002019 | 0.0021705 | 0.9979 | — | 0 / yes | 2.31/0.872 | 3.24/6 | 0.0186/2 | — |
| friedman2 | none | 1 | 0.9842 | 0.0011355 | 0.001154 | 0.001187 | 0.9988 | — | 0 / yes | 9.28/0.872 | 3.56/6 | 0.0733/2 | — |
| friedman2 | none | 2 | 0.9842 | 0.0019073 | 0.001938 | 0.0019319 | 0.9981 | — | 0 / yes | 6.55/0.872 | 6.22/6 | 0.0541/2 | — |
| friedman2 | phase3a_core | 0 | 0.9842 | 0.029521 | 0.03000 | 0.032719 | 0.9676 | 15.07x | 0 / yes | 0.146/0.872 | 3.31/6† | 0.0104/2 | — |
| friedman2 | phase3a_core | 1 | 0.9842 | 0.0090972 | 0.009243 | 0.0086039 | 0.9915 | 7.25x | 0 / yes | 0.310/0.872 | 7.83/6† | 0.0433/2 | — |
| friedman2 | phase3a_core | 2 | 0.9842 | 0.0075683 | 0.007690 | 0.0073430 | 0.9927 | 3.80x | 0 / yes | 0.249/0.872 | 7.05/6† | 0.0340/2 | — |
| friedman2 | phase3a_full | 0 | 0.9842 | 0.028963 | 0.02943 | 0.032595 | 0.9678 | 15.02x | 0 / yes | 0.445/0.872 | 5.78/6 | 1.53/2 | — |
| friedman2 | phase3a_full | 1 | 0.9842 | 0.012590 | 0.01279 | 0.013114 | 0.9870 | 11.05x | 0 / yes | 0.278/0.872 | 5.71/6 | 0.143/2 | — |
| friedman2 | phase3a_full | 2 | 0.9842 | 0.010233 | 0.01040 | 0.010338 | 0.9898 | 5.35x | 0 / yes | 0.232/0.872 | 4.97/6 | 0.178/2 | — |
| friedman2 | phase3b_tm | 0 | 0.98471 | 0.013598 | 0.01381 | 0.014669 | 0.9855 | 6.76x | 0 / yes | 0.165/0.872 | 4.44/6 | 0.0520/2 | — |
| friedman2 | phase3b_tm | 1 | 0.98470 | 0.011860 | 0.01204 | 0.013192 | 0.9870 | 11.11x | 0 / yes | 0.163/0.872 | 3.80/6 | 0.0495/2 | — |
| friedman2 | phase3b_tm | 2 | 0.98470 | 0.0072495 | 0.007362 | 0.0077180 | 0.9924 | 4.00x | 0 / yes | 0.158/0.872 | 4.41/6 | 0.0591/2 | — |
| toy_mimo | none | 0 | 1.8018 | 0.0093364 | 0.005182 | 0.013660 | 0.9930 | — | 0 / yes | 4.84/0.872 | 5.55/6 | 2.18/2 | — |
| toy_mimo | none | 1 | 1.8040 | 0.0078409 | 0.004346 | 0.0096567 | 0.9950 | — | 0 / yes | 13.4/0.872 | 5.14/6 | 3.33/2 | — |
| toy_mimo | none | 2 | 1.8041 | 0.010149 | 0.005625 | 0.013848 | 0.9929 | — | 0 / yes | 15.2/0.872 | 6.29/6 | 7.13/2 | — |
| toy_mimo | phase3a_core | 0 | 1.8015 | 0.019771 | 0.01098 | 0.032838 | 0.9831 | 2.40x | 0 / yes | 0.870/0.872 | 10.01/6† | 0.455/2 | — |
| toy_mimo | phase3a_core | 1 | 1.8041 | 0.025596 | 0.01419 | 0.043043 | 0.9778 | 4.46x | 0 / yes | 0.872/0.872 | 6.62/6† | 4.12/2 | — |
| toy_mimo | phase3a_core | 2 | 1.8040 | 0.011248 | 0.006235 | 0.015593 | 0.9920 | 1.13x | 0 / yes | 0.872/0.872 | 4.88/6† | 2.76/2 | — |
| toy_mimo | phase3a_full | 0 | 1.8030 | 0.024597 | 0.01364 | 0.034021 | 0.9825 | 2.49x | 0 / yes | 0.170/0.872 | 5.98/6 | 2.77/2 | 0 |
| toy_mimo | phase3a_full | 1 | 1.8032 | 0.030716 | 0.01703 | 0.036661 | 0.9811 | 3.80x | 0 / yes | 0.168/0.872 | 6.70/6 | 5.37/2 | 0 |
| toy_mimo | phase3a_full | 2 | 1.8032 | 0.024966 | 0.01385 | 0.032286 | 0.9834 | 2.33x | 0 / yes | 0.163/0.872 | 5.71/6 | 1.33/2 | 2e-15 |
| toy_mimo | phase3b_tm | 0 | 1.8038 | 0.025198 | 0.01397 | 0.029345 | 0.9849 | 2.15x | 0 / yes | 0.173/0.872 | 5.79/6 | 2.92/2 | 0 |
| toy_mimo | phase3b_tm | 1 | 1.8040 | 0.017467 | 0.009682 | 0.023066 | 0.9881 | 2.39x | 0 / yes | 0.228/0.872 | 5.82/6 | 3.67/2 | 2e-15 |
| toy_mimo | phase3b_tm | 2 | 1.8040 | 0.026497 | 0.01469 | 0.034061 | 0.9824 | 2.46x | 0 / yes | 0.302/0.872 | 5.71/6 | 2.07/2 | 7e-15 |

## Acceptance criteria and diagnoses

| Criterion | Result at lr=0.01 | Diagnosis |
|---|---|---|
| 1. Finite losses/gradients; each trainable gets a finite gradient | **Pass:** all 24 rows have `nonfinite=0`, finite predictions, and empty `no_grad_params`. Compliance-frozen clip-sharpness parameters are not trainable and are not counted. | No numerical instability or unexplained dead trainable parameters in this run. |
| 2. Final train loss ≤ 0.25× initial | **Pass:** every task/preset/seed is ≤0.0300×. | All presets learn under this toy training budget. |
| 3. Constrained mean held-out MSE ratio | **Fail** for Friedman2 core/full/tm (9.20× / 10.60× / 6.73×; limits 1.5× / 2× / 2.5×), and Toy core/full (2.46× / 2.77×; limits 1.5× / 2×). **Pass** for Toy phase3b_tm (2.33×≤2.5×). | Friedman2 unconstrained held-out R² is 0.9979–0.9988, well above 0.8, so the baseline is not under-trained. The constrained Friedman2 runs are poorly optimized at lr=0.01; an equal-budget lr=0.03 sweep below greatly improves them. Toy full/core retain a measurable constraint cost; the proposed caps are not calibrated. No threshold was loosened. |
| 4a. Slope bound | **Pass:** all constrained runs are within the 0.87234 slope limit. | Enforcement behaves as intended in these samples. |
| 4b. Rail bound | **Not applicable** to phase3a_core (no `hw_compliance`); **pass** for Friedman2 full/tm and Toy tm; **fail** for Toy phase3a_full seed 1 (6.70 > 6.0). | The harness previously asserted rails even for phase3a_core; fixed to check only compliance-enabled presets. The Toy full overshoot is a real post-train excursion, not that harness bug. |
| 4c. Heun stiffness `dt·λ≤2` | **Pass** for all Friedman2 constrained seeds; **fail** for Toy core seeds 1/2, full seed 0, and all three phase3b_tm seeds. | `hw_check_stiffness` is an optional diagnostic, not enabled by these presets; the proposed universal ≤2 acceptance is not guaranteed by the current configurations. The measured failures remain reported (not filtered or threshold-adjusted). |
| 4d. Quantized code grid | **Pass:** every active quantization check is within 1e-6 codes (observed max ≈7e-15). | Quantization is correct to floating-point precision. |
| 5. Seed spread ≤2× | **Pass** for Friedman2 phase3b_tm and Toy full/tm. **Fail** for Friedman2 core/full (3.90× / 2.83×) and Toy core (2.28×). | At lr=0.03 the Friedman2 spread nearly passes (2.05× / 1.53× / 2.01×); factor-2 is a very tight proposed threshold for three random initializations. No threshold changed. |

## Fair learning-rate selection

The original results above use `lr=0.01`. To avoid judging constraints against an
underfit unconstrained model, all 24 task/preset/seed combinations were rerun at
`lr ∈ {0.003, 0.01, 0.03, 0.1}` with the same 400-step budget. Every selected
record has `result_schema=2`; the grid is summarized in `/tmp/w8a/fair_lr.json`.
For each task and preset, the selected learning rate minimizes that preset's
three-seed mean held-out MSE. This is an explicitly held-out-data selection, so
the resulting ratios are descriptive/model-selection results, not an untouched
generalization estimate. Each constrained mean is divided by the unconstrained
mean at that task's own best learning rate (0.1 for both tasks), not by the
unconstrained MSE at the constrained preset's learning rate. Seed spread is
max/min held-out MSE across the three selected runs.

| Task | Preset | Best lr | Mean held-out MSE | Ratio vs task's unconstrained best | Seed spread |
|---|---|---:|---:|---:|---:|
| friedman2 | none (reference) | 0.1 | 0.001690 | 1.00× | 11.11× |
| friedman2 | phase3a_core | 0.1 | 0.001335 | 0.79× | 5.79× |
| friedman2 | phase3a_full | 0.1 | 0.001389 | 0.82× | 7.44× |
| friedman2 | phase3b_tm | 0.03 | 0.006139 | 3.63× | 2.07× |
| toy_mimo | none (reference) | 0.1 | 0.003500 | 1.00× | 3.43× |
| toy_mimo | phase3a_core | 0.1 | 0.007207 | 2.06× | 5.94× |
| toy_mimo | phase3a_full | 0.1 | 0.013594 | 3.88× | 3.02× |
| toy_mimo | phase3b_tm | 0.1 | 0.015325 | 4.38× | 2.42× |

The fair selection materially changes the story from the prior `lr=0.03`
comparison: the Friedman2 core/full means are slightly below the independently
selected unconstrained reference, while phase3b_tm remains 3.63× higher. All
three constrained Toy means are worse than the unconstrained reference, with the
full and time-multiplexed presets showing the largest costs. The wide seed
spreads mean these three-seed means are not robust performance guarantees.

> **Superseded note — old same-lr `lr=0.03` comparison.** The table below is kept
> only as an audit trail. Its denominator was the unconstrained MSE at `lr=0.03`,
> which is worse than the unconstrained model's best-lr MSE; it therefore
> understates the constrained-model cost and must not be used as the fair result.

| Task | Preset | Mean MSE ratio at lr=0.03 (old denominator) | Seed spread | Additional observed outcome |
|---|---|---:|---:|---|
| friedman2 | phase3a_core | 0.81× | 2.05× | MSE passes; spread narrowly fails. |
| friedman2 | phase3a_full | 0.74× | 1.53× | MSE and spread pass. |
| friedman2 | phase3b_tm | 1.21× | 2.01× | MSE passes; one rail and one stiffness miss. |
| toy_mimo | phase3a_core | 1.11× | 1.72× | MSE passes; core-only rail is not enforced; stiffness passes all seeds in this sweep. |
| toy_mimo | phase3a_full | 2.25× | 1.78× | MSE misses 2×; rail passes 2/3; stiffness passes 1/3. |
| toy_mimo | phase3b_tm | 2.72× | 1.65× | MSE misses 2.5×; rail and stiffness miss all 3 seeds. |

## Heun step-refinement study

The Toy runs that exceeded the harness's Heun limit at `lr=0.01` were retrained
with the same seeds: core seeds 1/2, full seeds 0/1, and phase3b_tm seeds 0/1/2.
The actual net uses `[6, 6]` Heun steps per stage; each trained model was then
re-integrated with `[12, 12]` and `[24, 24]` steps. Relative output-change MSE is
`mean((y_n-y_base)^2) / mean(y_base^2)` on the 256 held-out inputs. `max |x|`
is the maximum absolute internal state over the held-out trajectories.

`max dt·ρ(J)` below is the harness's trajectory Jacobian metric from the
schema-2 `lr=0.01` baseline (32 sampled rows). `stiffness_report` is the
per-stage small-signal diagnostic in `hw.constraints`, shown separately; they
are not interchangeable. The exact trajectory metric was >2 for all seven
cases, but `stiffness_report` was only 0.125–0.244 at the base step count.

| Preset / seed | Base trajectory max dt·ρ(J) | Relative output-change MSE ×2 / ×4 | max \|x\| ×2 / ×4 | `stiffness_report` max dt·λ base / ×2 / ×4 |
|---|---:|---:|---:|---:|
| phase3a_core / 1 | 4.12 | 4.03e−6 / 5.62e−6 | 6.629 / 6.631 | 0.244 / 0.122 / 0.061 |
| phase3a_core / 2 | 2.76 | 2.09e−6 / 2.34e−6 | 4.885 / 4.885 | 0.152 / 0.076 / 0.038 |
| phase3a_full / 0 | 2.77 | 4.13e−7 / 4.74e−7 | 5.714 / 5.714 | 0.228 / 0.114 / 0.057 |
| phase3a_full / 1 | 5.37 | 2.20e−6 / 3.43e−6 | 5.959 / 5.714 | 0.133 / 0.067 / 0.033 |
| phase3b_tm / 0 | 2.92 | 1.15e−6 / 1.32e−6 | 5.714 / 5.714 | 0.135 / 0.068 / 0.034 |
| phase3b_tm / 1 | 3.67 | 1.24e−6 / 1.83e−6 | 5.548 / 5.530 | 0.205 / 0.102 / 0.051 |
| phase3b_tm / 2 | 2.07 | 4.79e−7 / 7.26e−7 | 5.714 / 5.714 | 0.125 / 0.062 / 0.031 |

The `stiffness_report` values scale as expected with `dt`, but in this sample
its small-signal bound fails to reveal the larger trajectory Jacobian values.
The finer integrations change the held-out output negligibly (all relative
changes < 6e−6). The full/seed-1 rail overshoot is reproduced at the base count
(6.704) and falls to the hardware rail `x_hard=5.714` at ×4. Core-only has no
enforced rail, so its 6.63 peak remains diagnostic rather than a compliance
failure. This points to a timestep-sensitive rail excursion, not a material
held-out-output integration error.

### Fourfold steps trained from scratch

| Preset / seed | Base held MSE | ×4 scratch held MSE | max \|x\| | max `stiffness_report` dt·λ | Base / ×4 wall time (s) |
|---|---:|---:|---:|---:|---:|
| phase3a_core / 1 | 0.04304 | 0.04137 | 6.067 | 0.054 | 58 / 253 |
| phase3a_core / 2 | 0.01559 | 0.02781 | 5.410 | 0.039 | 64 / 250 |
| phase3a_full / 0 | 0.03402 | 0.03952 | 5.714 | 0.052 | 102 / 428 |
| phase3a_full / 1 | 0.03666 | 0.05636 | 5.714 | 0.051 | 102 / 425 |
| phase3b_tm / 0 | 0.02934 | 0.02473 | 5.714 | 0.032 | 102 / 429 |
| phase3b_tm / 1 | 0.02307 | 0.03614 | 4.910 | 0.016 | 102 / 417 |
| phase3b_tm / 2 | 0.03406 | 0.03695 | 5.714 | 0.037 | 102 / 425 |

All fourfold scratch runs retained the slope bound; compliance-enabled full/tm
runs stayed at or below `x_hard`. The 4× models took 250–429 s versus 58–102 s
for the base-step training (about 4× wall time). Their held-out MSE changes are
mixed rather than consistently improved, so retraining at finer steps does not
show a systematic optimizer-quality gain.

**Verdict and step rule.** For these cases, integration error is not the driver
of output MSE: the fixed-weight output changes are several orders below the
held-out MSE. A finer grid does remove the single full-preset rail overshoot.
For timestep selection, target `max dt·λ ≤ 1` with
`hw.constraints.stiffness_report`, then confirm with the trajectory Jacobian
audit. Because the latter reported up to 5.37 at six steps while
`stiffness_report` reported at most 0.244, do not use the small-signal report
alone for this workload. Scaling the worst observed trajectory value by `dt`
suggests at least 33 steps per stage for a target of 1; 48 (8× the current 6)
is a simple conservative trial setting, subject to rechecking the trajectory
metric after retraining.

### Conclusion (pending ablation and CTLE recommendations)

The fair LR sweep shows that `lr=0.1` is best tested for both unconstrained
tasks and nearly every constrained task/preset; Friedman2 phase3b_tm selects
0.03. Friedman2 core/full match or slightly outperform the best unconstrained
mean in this small held-out sample, while phase3b_tm and all Toy constrained
presets retain a sizable relative-MSE cost. The step-refinement run finds tiny
fixed-weight output changes even when the baseline trajectory stiffness metric
exceeds the Heun bound; finer steps do remove one full-preset rail excursion.
Seed variability remains large. These CPU toys do not establish CTLE
performance; a controlled CTLE retrain is still required.

## Pruning investigation

The default `friedman2` preset used in an earlier probe has no boundary cell
library, so it could not reproduce the specialized-library issue. Reproduced
with the requested golden builders:

| Golden net | Full param count before | Core param count before | Specialized family counts before (full model) | Prior keep-all result | After fix |
|---|---:|---:|---|---|---|
| FRIEDMAN2_NET (`ota_mesh`, output ODE) | 1813 | 1804 | boundary 104; output_ode 256; VCA 384 | Core rebuild 1804→1120; forward failed with a 16-vs-17 state-size mismatch. | Keep-all preserves total/core and every family count; exact output max-abs difference 0. |
| TOY_NET (shared-sense, GLN) | 811 | 712 | boundary 128; readout_sense 128; VCA 160; GLN rails 72 | Existing pruner raised `shared_sense readout not yet supported`. | Keep-all preserves total/core and every family count; exact output max-abs difference 0. |

`prune_network` now recognizes when every edge passes the requested threshold
and returns a deep copy of the complete network plus identity node remaps. This
avoids rebuilding away stage-owned boundary/output-ODE/sense libraries and the
outer GLN state in the required keep-all case. The regression test asserts the
nonzero specialized families, full-model per-family counts, and bit-exact
forward output for both golden nets. The existing GLN fail-loud test now uses
threshold 0.999 to request a real partial prune; its old threshold 0 exercised
the newly supported keep-all identity case. This is a keep-all identity fix;
arbitrary partial pruning of shared-sense/GLN remains explicitly unsupported by
the existing pruner. Friedman2's three-phase schedule does prune, so partial-prune
support for specialized Friedman2 stages remains a distinct follow-up concern.
