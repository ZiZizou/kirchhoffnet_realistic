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
Friedman2 physical-build checks are included).

Table fields: `MSE / none` is the held-out MSE divided by the *same-seed*
unconstrained MSE (diagnostic; acceptance uses the ratio of the mean MSEs over
seeds). `Rail` is observed max |x| / 1.05·x_hard. `†` marks phase3a_core:
that preset has no `hw_compliance`, so the rail value is diagnostic and is not
asserted. `dt·λ` is the maximum reported value / 2. The stiffness metric samples
32 held-out inputs. `Quant` is the largest code-grid distance; `—` means no
quantized parameter is enabled. Nonfinite is the harness loss/gradient count;
all rows also had finite held-out outputs.

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

The lr=0.01 runs show final/initial loss reduction on every preset, but that alone
does not mean the optimizer reached a comparable validation solution. The
Friedman2 MSE failures are primarily an under-trained **constrained** optimizer
at the default learning rate, not a weak unconstrained reference. This was
checked with a fair same-400-step, same-seed LR sweep at `lr=0.03` (all models,
both tasks; prior run records in `/tmp/w5learn/lr3e-2`). The unconstrained
Friedman2 R² remains ≥0.988 in that sweep, and all three Friedman2 constrained
MSE ratios improve to within their proposed limits. Toy phase3a_full and
phase3b_tm still exceed their MSE caps at lr=0.03, despite small training-loss
ratios and a good unconstrained reference: that is a remaining constrained-model
quality cost and/or an uncalibrated proposed cap, not a reason to loosen it
without broader data.

| Task | Preset | Mean MSE ratio at lr=0.03 | Seed spread | Additional observed outcome |
|---|---|---:|---:|---|
| friedman2 | phase3a_core | 0.81× | 2.05× | MSE passes; spread narrowly fails. |
| friedman2 | phase3a_full | 0.74× | 1.53× | MSE and spread pass. |
| friedman2 | phase3b_tm | 1.21× | 2.01× | MSE passes; one rail and one stiffness miss. |
| toy_mimo | phase3a_core | 1.11× | 1.72× | MSE passes; core-only rail is not enforced; stiffness passes all seeds in this sweep. |
| toy_mimo | phase3a_full | 2.25× | 1.78× | MSE misses 2×; rail passes 2/3; stiffness passes 1/3. |
| toy_mimo | phase3b_tm | 2.72× | 1.65× | MSE misses 2.5×; rail and stiffness miss all 3 seeds. |

### Conclusion

All four presets can reduce training loss substantially on both small synthetic
tasks. At the default lr, constrained Friedman2 has unacceptable relative
validation MSE, but a fair lr retune recovers it; the proposed seed-spread cap
still narrowly misses for two presets. On Toy, the core-only and full presets
miss MSE at lr=0.01, while phase3b_tm passes that single MSE cap; at lr=0.03,
core-only passes but full/tm still miss. The Toy stability/rail misses show that
learning success does not establish physical invariants. These are small CPU
toys only; none substitutes for the CTLE SLURM retrain required to validate the
actual workload.

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
forward output for both golden nets. This is a keep-all identity fix; arbitrary
partial pruning of shared-sense/GLN remains explicitly unsupported by the
existing pruner. Friedman2's three-phase schedule does prune, so partial-prune
support for specialized Friedman2 stages remains a distinct follow-up concern.
