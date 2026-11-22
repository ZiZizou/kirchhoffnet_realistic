# NARMA-10 Delay-Bank KNet: Core-Dynamics Problem Statement

**Status:** 2026-09-14. This is the authoritative handoff brief for the
current NARMA work. It is designed to be read with `README.md` (machine and
cell equations) and `docs/narma-progress-summary.md` (earlier experiments).
The Phase 5/6 evidence below supersedes older statements that the main issue
is simply a weak or low-rank linearized core.

## The question

Design the next small, falsifiable experiment that improves the *realistic
KirchhoffNet (KNet)* reservoir's useful state diversity and NARMA-10
performance without cheating through a dense digital recurrent crossbar,
future-input/target leakage, or uncontrolled boundary-input changes.

The immediate scientific question is:

> Why does a physical delay-bank KNet have a rich boundary injection and a
> full, near-unit linearized spectrum, yet its actual driven state covariance
> collapses to roughly two effective dimensions? What core change is most
> likely to break that collapse while retaining moderate rail activity?

The preferred deliverable is a ranked proposal for one primary experiment and
one or two discriminating controls, not a broad hyperparameter sweep.

## Non-negotiable experimental setup

The experiment is a single-stage NARMA-10 reservoir. The detailed KCL, cell,
Heun, compliance, and topology definitions are in the README; these details
are the configuration that matters for this problem:

- **Reservoir:** 25 dynamic hidden voltage nodes; effective capacitance 1;
  per-sample integration window `t_span=1.0` with eight Heun steps. The
  carried hidden state is the reservoir feature vector.
- **Core cell:** `linear_ota` in the latest experiment. Its edge current is
  `I = Isat * gm * sign * (A*Vsrc - B*Vdst)`, with compliance gates at both
  terminals; the node activation is bounded `tanh`. Thus the core is not a
  conventional dense ESN matrix, even though it has a well-defined
  one-sample Jacobian.
- **Input:** eight causal taps of one scalar NARMA input,
  `[u[t], u[t-1], ..., u[t-7]]`. Each tap is an ideal delay-terminal voltage.
  Every tap drives every hidden node through an independently parameterized
  boundary OTA: 8 x 25 = **200 boundary OTAs**. Their currents sum by KCL.
  The delay bank is permitted because it is causal and can be implemented by
  sampled analog/digital delay elements. No future sample is used.
- **Boundary policy:** hold its heterogeneous spatial pattern fixed. The
  boundary destination term is strongly suppressed, so a boundary-drive
  adjustment should not redesign the recurrent Jacobian. Do **not** propose
  another input-symmetry-breaking or VCA-boundary campaign as the first next
  step.
- **Operating point:** common boundary `isat_raw` mean `-4.4` is the chosen
  calibrated setting. It gives about 0.49 mean `|state| / rail` and about
  0.16 rail occupancy. The acceptable mean-amplitude band is 0.45--0.55;
  rail occupancy should stay at or below 0.20 if possible.
- **Readout and score:** a ridge readout is fit on the 25 hidden states after
  washout. Report washed NRMSE (lower), R2 (higher), memory capacity (MC,
  higher), and state covariance participation ratio (state PR, higher). Use
  the same streams, washout, and ridge convention for all comparisons.

The initial core realization uses heterogeneous edge parameters and
log-uniform programmable leaks in roughly `[0.06, 0.16]`, then adjusts core
gain so the measured zero-input one-sample spectral radius is near 0.95. The
normalizer actually lands around 0.97, which is close enough for comparison
but must always be reported rather than assumed.

## Established controls and observations

### Phase 5: the delay-bank architecture is useful

A deliberately nonphysical dense recurrent crossbar with the same eight
delays achieved MC 8.77, state PR 5.66, and ridge NRMSE 0.555 at moderate
activity. It is a ceiling/control, not an allowed implementation.

The physical KNet delay-bank experiment achieved MC 8.09 and NRMSE 0.586,
beating a ridge readout on the raw eight delay taps (NRMSE 0.812). Therefore
the KNet core contributes useful computation; the task and causal delay bank
are sound. However, it ran too hot: mean amplitude 0.689 and rail occupancy
0.284. Its boundary-injection PR was 6.04 but state PR only 1.89.

### Phase 6: topology did not resolve the collapse

Phase 6 lowered drive to `isat_raw=-4.4` and compared five core graphs across
three reproducible core seeds. Small-world graphs use `k=8`, giving 100
undirected branches, matched to the directed 5x5 kernel-3 torus branch count.
Values below are mean across three seeds; the standard deviations are small
enough for descriptive comparison but **three seeds are not a significance
test**.

| Core topology | Core edges | NRMSE | MC | State PR | Mean amplitude | Rail fraction |
|---|---:|---:|---:|---:|---:|---:|
| Directed 5x5 torus | 100 | 0.598 | 7.90 | 1.73 | 0.496 | 0.161 |
| Bidirectional 5x5 torus | 200 | 0.602 | 7.83 | 1.77 | 0.489 | 0.156 |
| Small-world, `k=8, p=.25` | 100 | 0.600 | 7.88 | 1.77 | 0.486 | 0.156 |
| Small-world, `k=8, p=.50` | 100 | 0.604 | 7.80 | 1.74 | 0.499 | 0.162 |
| Small-world, `k=8, p=.25`, bidirectional | 200 | 0.598 | 7.88 | 1.79 | 0.493 | 0.158 |

Interpretation: at this scale, rewiring and adding reverse branches do not
produce a meaningful improvement. The bidirectional small-world result is
numerically lowest NRMSE but is statistically indistinguishable from the
simple directed torus. Do not spend the next experiment budget on another
generic topology sweep.

### The central diagnostic paradox

For every Phase-6 topology, evaluated at zero input/state:

- all 25 eigenvalue magnitudes of the one-sample Jacobian exceed 0.8;
- eigenvalue-magnitude PR is about 24.96 (nearly all magnitudes are alike);
- the input-Jacobian spatial PR is about 5.44;
- the finite-horizon (20 sample) controllability-Gramian PR is about 5.9--6.2;
- driven local Jacobians at 16 washed trajectory locations are similar: about
  24.8 of 25 magnitudes exceed 0.8 and mean spectral radius is about 0.96;
- the measured nonnormality proxy is only about `1e-4`.

Yet the **actual washed state covariance PR is only 1.7--1.8**. The core has
many slowly decaying linear modes and the zero-point input map can reach about
six directions, but the driven state distribution uses only about two.

This means the phrase "increase eigenstructure" must be precise. Merely
raising spectral radius, increasing edge count, or adding a small-world graph
is not supported by the evidence. The current spectrum is broad in *count*
but extremely flat in *magnitude* (most modes retain at roughly the same
rate), nearly normal, and measured controllability is only a zero-operating-
point quantity. The missing mechanism may be a lack of timescale hierarchy,
weak transient/non-normal mixing, state-dependent collapse of the driven
input map, or some combination.

## What has and has not been ruled out

Supported facts:

- The causal delay-bank injection is rich enough to be worth preserving.
- A physical KNet core beats the raw-delay ridge control, so removing the core
  is not a solution.
- The calibrated drive lands in the requested activity range.
- Generic small-world rewiring and bidirectionality are not strong levers at
  25 nodes / this gain normalization.
- A flat near-unit spectrum is not sufficient for high state PR.

Not yet measured, and therefore not ruled out:

- the trajectory-local input Jacobian `B_t = d x[t+1] / d u[t]`;
- a finite-time, time-varying controllability Gramian built from actual
  `J_t, B_t` along the driven sequence;
- the PR of the net boundary-current vector along the driven sequence;
- whether deliberately spaced poles/timescales outperform the current flat
  near-unit spectrum at the same amplitude and rail fraction;
- whether a modest, explicitly designed non-normal/rotational component
  improves state PR without rail growth.

## Scope for proposed solutions

The preferred direction is **core eigenstructure design**, evaluated under
the fixed delay-bank and fixed boundary policy. Plausible proposal families
include, but are not limited to:

1. A core initialization/normalization objective that shapes a *distribution*
   of one-sample eigenvalue magnitudes (for example, several stable bands
   between roughly 0.55 and 0.95) instead of only setting the maximum.
2. A physically realizable reciprocal plus antisymmetric coupling recipe that
   creates controlled complex or non-normal transient modes while retaining a
   stable KCL realization.
3. Node/edge time-constant assignments designed to create distinct memory
   bands, followed by stability normalization that does not erase those bands.
4. A diagnostic-first run that measures `B_t`, time-varying controllability,
   and boundary-current PR before changing architecture.

Do not assume these candidates work. A good proposal explains why its chosen
mechanism addresses the gap between zero-point controllability PR (~6) and
state PR (~1.8), and how it remains compatible with KCL, bounded nodes, and
causal delay terminals.

## Required form of an outside proposal

For each recommended experiment, provide:

1. A precise core-only intervention, including what remains fixed.
2. The mechanism: what it should change in `J`, `B_t`, the time-varying
   controllability Gramian, or state covariance.
3. One primary condition and one ablation/control that falsifies the claimed
   mechanism.
4. Diagnostics to log: NRMSE, R2, MC, state PR, amplitude, rail fraction,
   actual spectral radius, eigenvalue distribution, trajectory-local `B_t`,
   and time-varying controllability PR.
5. Predeclared success/failure criteria. A useful initial bar is state PR
   materially above 2, no increase above 0.20 rail fraction, and a repeatable
   NRMSE/MC improvement over the directed-torus baseline. State what effect
   size would justify additional seeds.
6. Hardware/physical assumptions and any way the suggestion might accidentally
   become a prohibited dense digital recurrent crossbar or target-leaking
   shortcut.

The objective is not an attractive new graph. It is a reproducible mechanism
that turns the existing rich delay-bank injection into a higher-dimensional,
useful driven reservoir state.
