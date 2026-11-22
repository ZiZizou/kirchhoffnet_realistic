# Analog Temporal-Diversity Research Note for NARMA

**Purpose.** This note separates published facts about ESNs, gated RNNs, and
multi-rate recurrence from the design inferences relevant to the current
25-node delay-bank KNet. It is not evidence that any proposed circuit will
improve KNet; it identifies small, falsifiable mechanisms worth testing after
Phase 8.

## What the comparison with an ESN does and does not establish

An ESN is a fixed recurrent nonlinear system driven by the input, with a
trained readout. Its central mechanism is expansion into many nonlinear state
responses, not merely long memory. Leaky-integrator ESNs add an individual
state time constant; their published control parameters include input scaling,
leaking rate, and reservoir spectral radius [Jaeger *et al.*,
2007](https://doi.org/10.1016/j.neunet.2007.04.016). Thus, an ESN's higher
state PR is not, by itself, proof that KNet needs a larger spectral radius or
less leak. It can reflect a different recurrent map, input map, operating
range, node count, and/or trained parameters.

Nor is a small-signal radius below one a complete certificate for useful
driven dynamics or even for the echo-state property: the necessary/sufficient
conditions depend on the nonlinear, driven system [Yildiz, Jaeger & Kiebel,
2012](https://doi.org/10.1016/j.neunet.2012.07.005). This directly supports
keeping the measured finite-drive gates and trajectory diagnostics in KNet.

**Inference for KNet.** Phase 6 already rejects the simple explanation “too
few long-lived linear modes”: its 25 local Jacobian magnitudes are near unit
yet driven PR is about two. Phase 8 rejects “only quieter activity”: leak
bands help relative to an activity-matched uniform control. The missing
property is more specifically *separable, driven state trajectories*. The
appropriate next objective is therefore to increase trajectory state/Gramian
PR while preserving the demonstrated activity and rail limits—not to maximise
spectral radius or raw PR in isolation.

## Published mechanisms that are relevant

### 1. Static, deliberately spaced time constants (ESN leak bands)

**Direct support.** Leaky-integrator ESNs were introduced precisely so that
individual state dynamics can accommodate a task's temporal characteristics;
the same work gives stability conditions and optimizes leak, spectral radius,
and input/output-feedback scaling [Jaeger *et al.*,
2007](https://doi.org/10.1016/j.neunet.2007.04.016).

**KNet interpretation.** The Phase-8 slow/medium/fast bands are the closest
analog of this mechanism: a node capacitor plus a programmable shunt/leak
sets a physical low-pass time constant. They should be retained as a baseline,
but static bands alone cannot explain away the large ESN gap because they
raised correlation PR and task scores only modestly.

### 2. State-dependent retention (GRU/LSTM principle)

**Direct support.** The original GRU-style unit has reset and update gates
that adaptively control how much a hidden unit remembers/forgets
[Cho *et al.*, 2014, Sec. 2.3](https://arxiv.org/abs/1406.1078). LSTM uses
multiplicative gates to control access to a protected memory path
[Hochreiter & Schmidhuber, 1997](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf).
This is qualitatively different from choosing a fixed leak once at
initialization.

**Direct support for analog plausibility.** Odame *et al.* implemented a
continuous-time gated recurrent network as a 0.18-um CMOS IC. Their update
gate controls an adaptive low-pass filter: a small gate current lengthens the
time constant/holds state, whereas a large current speeds the update
[Odame *et al.*, 2022, Secs. II-B and III-C](https://arxiv.org/abs/2208.01201).
Their paper explicitly reports a current-mode implementation without DACs,
ADCs, op-amps, or four-quadrant multipliers.

**KNet interpretation.** A minimal candidate is *not* a full trainable GRU
crossbar. Add a bounded, low-rank/broadcast control voltage that modulates
each node's OTA tail current or shunt conductance:

\[
C_j \dot x_j = g_j(u_t,x_j) I_{\mathrm{candidate},j}
                - \ell_j(g_j) x_j - I_{\mathrm{clip},j}, \qquad 0<g_j<1.
\]

In a sampled view this implements the GRU idea
\(x_{t+1}=z_j x_t+(1-z_j)\tilde x_{t+1}\): some nodes retain, others update,
on the same input. OTA bias-current modulation, the existing VCA/GLN-style
control path, and the existing capacitive state are plausible primitives. A
single gate shared by every node is unlikely to increase PR; use a small
number of shared control rails with fixed or learned per-node mixing, and
count the added devices.

### 3. Explicit multi-rate modules (Clockwork principle)

**Direct support.** Clockwork RNN partitions its hidden layer into modules
that operate at prescribed temporal granularities rather than updating every
module identically on every sample [Koutnik *et al.*,
2014](https://arxiv.org/abs/1402.3511).

**KNet interpretation.** A hardware-conservative version is three node
groups with fixed banded RC constants *and* phase/update-enable signals: fast
nodes update each sample, medium nodes every 2 samples, slow nodes every 4
samples. The enable can be realized as a periodic switched-current or
sample/hold control, with causally sampled input; it is not a dense digital
recurrent matrix. This proposal is an inference, not a claim established by
the Clockwork paper for NARMA or KNet. Its risk is aliasing or a clocked
delay-bank shortcut, so the matched control must retain the same eight input
taps and number of update phases but randomize which nodes receive each rate.

### 4. Train the physical dynamics, not only the readout or initial draw

**Direct support.** The cited ESN work studies tuning of global dynamical
parameters; GRU/LSTM learn their gates and recurrent weights rather than
merely selecting an initialization. This does **not** imply that unconstrained
KNet BPTT will work or remain hardware-realizable.

**KNet inference.** The current results test fixed, normalized cores plus a
linear ridge readout. It is therefore premature to conclude that KNet's
physical family cannot approach a better ESN after constrained training. The
credible question is narrower: can training a *small analog control set*
(per-node static leaks, a few gate-rail mixing weights, or a sparse directed
transport path) raise useful state diversity without becoming a prohibited
dense recurrent crossbar? Training every OTA freely before identifying the
mechanism would make a negative or positive result difficult to interpret.

## Recommended next experiment: gated leak-bank, not a broad retraining run

Keep the Phase-8 delay bank, torus, boundary pattern, normalized core,
readout protocol, activity target, and rail cap fixed. Start from the
activity-restored three-band core. Introduce two or three bounded causal gate
rails from the *present and past-only* input taps; mix them into node update
currents with low-rank, initially zero-effect couplings. Train only those
couplings and optionally the three band leak scalars under a rail penalty. Do
not train a dense 25-by-25 digital recurrence.

Predeclare the decisive comparisons:

| Condition | Tests |
|---|---|
| Static Phase-8 banded core | Does any gain exceed the demonstrated static-timescale mechanism? |
| Same gate hardware, gate fixed at its identity value | Are gains due to gating rather than extra circuit elements? |
| Gate rails permuted across nodes or driven by an independent stream | Does causal state-dependent routing, rather than added noise, cause the effect? |

Log NRMSE, MC, nonlinear lag probe, raw and correlation state PR, actual
activity/rails, per-node gate occupancy, trajectory-local input Jacobian, and
time-varying controllability PR. A promotion criterion should require
replicated NRMSE and MC improvement over static banded Phase 8, an increase in
correlation PR or controllability PR, no rail fraction above 0.20, and no
evidence that the control accesses target or future input.

## What to avoid concluding

- Higher ESN PR is a symptom of a more useful representation, not a target
  that guarantees lower NRMSE. Phase 7 already showed that raw PR can rank a
  worse uniform-leak condition above a better banded one.
- A near-unit spectral radius can make memory longer but need not make delayed
  inputs separable; the leaky-ESN literature treats it as one dynamical
  control among several, not a sufficient recipe.
- “Analog” does not excuse an arbitrary digital recurrent controller. The
  proposed gate must be a bounded causal physical control (bias current,
  transconductance, shunt, or sample/hold) with an explicit component and
  parameter budget.

## Primary sources

1. H. Jaeger, M. Lukoševičius, D. Popovici, and U. Siewert, “Optimization and
   applications of echo state networks with leaky-integrator neurons,”
   *Neural Networks* 20(3), 335–352 (2007),
   [DOI](https://doi.org/10.1016/j.neunet.2007.04.016).
2. K. Yildiz, H. Jaeger, and S. Kiebel, “Re-visiting the echo state
   property,” *Neural Networks* 35, 1–9 (2012),
   [DOI](https://doi.org/10.1016/j.neunet.2012.07.005).
3. K. Cho *et al.*, “Learning Phrase Representations using RNN
   Encoder–Decoder for Statistical Machine Translation” (2014),
   [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
4. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural
   Computation* 9(8), 1735–1780 (1997),
   [author-hosted paper](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf).
5. J. Koutnik *et al.*, “A Clockwork RNN” (2014),
   [arXiv:1402.3511](https://arxiv.org/abs/1402.3511).
6. K. Odame *et al.*, “Analog Gated Recurrent Neural Network for Detecting
   Chewing Events” (2022), [arXiv:2208.01201](https://arxiv.org/abs/2208.01201).
