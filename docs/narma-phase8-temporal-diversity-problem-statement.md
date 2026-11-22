# NARMA-10 Delay-Bank KNet: ESN Gap and Temporal-State-Diversity Problem Statement

**Status:** 2026-09-15. Read this companion brief with `README.md` and
`docs/narma-progress-summary.md`.

## Bottom line

Phase 8 is a **mechanism win, not an ESN-parity win**. The activity-restored
banded KNet scores NRMSE 0.556 and MC 8.48, beating both the Phase-6 baseline
(0.598, 7.90) and an activity-matched uniform-leak control (0.603, 7.82) in
every paired seed. It establishes that several physical retention times are
useful. It does not close the gap to the tuned in-harness ESN (about 0.296
NRMSE); even the quieter banked banded KNet is 0.537.

The evidence does not support either extreme claim. Earlier KNet training
usually plateaued near 0.67--0.79 or gained fit by using rails, so generic
unconstrained training is not an easy answer. But the successful Phase-8
banded core has not yet received a paired, rail-managed training or multi-stage
test. Training and depth are therefore hypotheses, not explanations.

## Fixed physical setting

- 25 dynamic voltage nodes in the directed 5x5 torus, `linear_ota` core cells,
  bounded tanh nodes, KCL summation, and eight Heun steps per sample.
- `[u[t], ..., u[t-7]]` enters only through eight causal delay terminals and
  the fixed 200-OTA heterogeneous boundary. This is a real delay-bank input,
  not an engineered recurrent shift register.
- Phase-6 boundary `isat_raw` mean is `-4.4`; desired activity is 0.45--0.55
  mean `abs(x) / x_max`; rail fraction is at most 0.20.
- Report held-out NRMSE/R2, MC by delay, lag-9 product R2, activity, rails,
  raw covariance PR, correlation PR, and actual spectra. Raw PR is diagnostic,
  not an objective.

Activity is voltage swing relative to the rail. Raw PR measures concentration
of total state variance; correlation PR asks whether nodes have distinct
temporal signals after their voltage scales are normalized.

## What we know

Phase 6 gave the core many healthy-looking linear ingredients: all 25
zero-point Jacobian magnitudes exceed 0.8, eigenvalue-magnitude PR is about
25, input-Jacobian PR is about 5.4, and finite-horizon controllability PR is
about 5.9--6.2. Driven local Jacobians are also near-unit. Yet actual driven
state PR is only 1.7--1.8 and NRMSE is about 0.598. Bidirectional and
degree-matched small-world graphs did not change that.

This rejects “too few long-lived modes,” “only one input direction,” and
“spectral radius is too small” as complete explanations. The missing property
is separable, useful **driven** temporal states.

| Condition | NRMSE | MC | raw PR | correlation PR | activity | rails |
|---|---:|---:|---:|---:|---:|---:|
| Phase-6 baseline | 0.598 | 7.90 | 1.73 | 2.32 | 0.496 | 0.161 |
| Banded alpha=1 banked reference | **0.537** | **8.74** | 1.94 | **3.46** | 0.401 | 0.103 |
| Banded, activity-restored | **0.556** | **8.48** | 1.88 | **3.24** | 0.453 | 0.131 |
| Uniform, activity-matched | 0.603 | 7.82 | 1.77 | 2.35 | 0.460 | 0.137 |

The restored banded condition improves baseline NRMSE by 0.039--0.048 and MC
by 0.48--0.62 in every pair. Its slow/medium/fast realised pole means remain
about 0.96--0.97, 0.83--0.85, and 0.67--0.69. It also beats uniform in ordinary
and training-only-standardized ridge. This establishes that the fixed delay-bank
KNet benefits from time-scale diversity, not merely quieter activity.

It does not establish a true KNet shift-register, a path to ESN performance,
or raw PR as the correct optimization target. The old uniform control had
higher raw PR than alpha=1 banded leaks while scoring worse.

## Why the ESN remains ahead

An ESN is not just a higher-PR reservoir. It combines a selected stable
recurrent map, fixed input projections, nonlinear expansion, and a trained
linear readout. KNet has analog counterparts for sparse recurrence, signed
input OTAs, nonlinearity, capacitive state, and ridge readout. Its gap is in
the quality and accessibility of its driven temporal features.

The data point to three specific deficiencies:

1. Phase 6 gave many modes similar retention, so shared delayed drive created
   similar histories. Phase 8's separated time constants improve correlation
   PR, MC, nonlinear lag-product access, and NRMSE.
2. Boundary/controllability PR is about six while raw state PR is below two.
   The core only partly transforms injected directions into observable useful
   features.
3. Present memory is a leaky voltage continuously mixed and overwritten by the
   same fabric. There is no protected pathway that retains selected state while
   other components write and nonlinearly mix new inputs.

## Is this merely an untrained-initialization result?

No definitive answer exists yet.

- Frozen-core ridge is deliberately an information-content probe. It cannot
  certify the best trainable KNet.
- Earlier unconstrained training did not close the gap and frequently raised
  rail occupancy; this is evidence against easy, generic BPTT rescue.
- Those runs were not paired rail-managed training from the Phase-8 banded
  initialization. They cannot falsify that specific hypothesis.
- Existing depth results are incomplete and have not shown a dividend. More
  stages can compose features, but can also repeatedly compress an already
  redundant state. Depth is not a replacement for temporal diversity.

The correct conclusion is that training and multi-stage KNet are **untested
hypotheses for this mechanism**, not current explanations for the ESN gap.

## Analog-compatible mechanisms to test

**ESN and multirate lesson.** Keep sparse signed OTA recurrence and deliberate
leak/capacitance bands. A structured local transport/permutation between bands
or fixed multi-rate sample/hold enables could mix differently aged traces
without a dense digital recurrent crossbar. Compare with the same hardware but
permuted band/rate assignment. Promote only if NRMSE, delayed MC, lag-product
access, and correlation PR improve at fixed rails.

**GRU lesson.** A leak-only modulation is not a GRU: it changes decay but does
not stop candidate current from overwriting voltage. A minimal analog
equivalent uses a bounded causal gate voltage to control both a candidate
current OTA and a complementary retention shunt, approximating
`x[t+1] = z[t] * x[t] + (1-z[t]) * candidate[t]`. A few shared control rails
with per-node mixing are more credible than a dense controller per edge.

**LSTM lesson.** A protected analog memory needs a storage capacitor,
controlled write OTA, controlled retention shunt, and separately gated read
OTA into an ordinary mixing node. This write/retain/expose two-compartment cell
is more structural and device-costly. It must be compared with an equal-device
count two-compartment control whose gates are fixed open.

## Primary next experiment: low-rank causal gated leak bank

Start from Phase-8 activity-restored banded leaks. Keep the fixed torus,
boundary, eight causal taps, and paired seeds `0,1,2`. Add two or three bounded
causal gate rails from present/past-only taps. Their fixed or learned per-node
mixes modulate OTA tail current and retention shunt. Initialize exactly at
static banded behavior. Train only gate mixes and, if necessary, the three
band scalars; do not open all core or boundary OTAs to unconstrained training.

| Condition | Purpose |
|---|---|
| Static Phase-8 banded core | Known temporal-diversity reference |
| Same added hardware, identity gate | Separates gating from extra devices |
| Gate rails permuted or independent-stream driven | Tests causal routing rather than added noise |
| Activity-matched trained uniform core | Tests whether diversity remains necessary |

Use a differentiable activity/rail penalty and report it separately. Preserve
train/validation/test separation, choose epochs on validation only, and retain
the input-only calibration stream. Log gate occupancy, per-band variance,
leading-PC loadings, poles, trajectory input Jacobians, and time-varying
controllability alongside the normal task metrics.

Promote this line only if every seed improves held-out NRMSE by at least 0.02
over both static activity-restored banded and trained uniform controls, retains
MC gain beyond the supplied taps, improves correlation PR or trajectory
controllability, and stays at rails <= 0.20. A rail-driven fit is not a pass.

## Guardrails

- Do not declare the ESN gap solved because correlation PR rose.
- Do not optimize raw PR directly.
- Do not repeat generic bidirectional or small-world sweeps.
- Do not call leak-only modulation a GRU/LSTM.
- Do not add a dense digital recurrent crossbar, future-input shortcut, or
  target-based activity calibration.

## Research sources

The companion note
[`narma-analog-temporal-diversity-research.md`](narma-analog-temporal-diversity-research.md)
records the source-backed reasoning. Key primary references are leaky ESNs
([Jaeger et al., DOI](https://doi.org/10.1016/j.neunet.2007.04.016)), driven
echo-state conditions ([Yildiz et al., DOI](https://doi.org/10.1016/j.neunet.2012.07.005)),
GRUs ([Cho et al.](https://arxiv.org/abs/1406.1078)),
LSTM ([Hochreiter and Schmidhuber](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf)),
Clockwork RNN ([Koutnik et al.](https://arxiv.org/abs/1402.3511)), and a
CMOS continuous-time gated recurrent circuit
([Odame et al.](https://arxiv.org/abs/2208.01201)). These motivate mechanisms;
they do not establish that any proposed KNet circuit will work.
