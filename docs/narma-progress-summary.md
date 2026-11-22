# NARMA Fabric Memory — Progress Summary (all legs in one place)

**Date:** 2026-09-15
**Combines:** `docs/narma-fabric-memory-problem-statement.md` (v1–v7),
`docs/narma-revised-plan.md` (§1–§15),
and Alliance logs in `outputs_alliance_can/`:
`narma_split-59625759.out`, `narma_transplant-59664806.out`,
`narma_transplant-59667050.out` (resubmit — full 54 corners),
`narma_gating_e0-59691417.out` (Phase-1 48-corner op-point grid),
`narma_gating_e0x-59712084.out` (Phase-1 hotter extension, 18 corners),
`narma_gating_spike-59691420.out` (A/B/C/D 20-epoch gating trains),
`narma_rung1-59724426.out` (Phase-2 native-linear screen),
`narma_depth-59719147.out` (Phase-4 depth, s1 timed out, s2 running),
`narma_round2_gated-59587991.out`,
`narma_revised_plan-59367932.out`, `narma_revised_plan-59444668.out`,
`narma_revised_plan-59492665.out`
(+ earlier `narma_diag_latest_log.out`, linear-controls `58460596.out` via the v7 record).
**Canary anchor:** `output/canary/canary.json`.
**Language:** plain on purpose.

> **Current-status note (Phase 5--8, 2026-09-15):** This document preserves
> the earlier NARMA campaign. Its broad historical framing predates the causal
> eight-tap delay-bank experiment and should not be treated as the current
> problem definition. The authoritative current handoff is
> [`narma-phase8-temporal-diversity-problem-statement.md`](narma-phase8-temporal-diversity-problem-statement.md).
> Phase 6 calibrated the real KNet delay-bank architecture to mean amplitude
> about 0.49 rail, found MC about 7.8--7.9 but state PR only 1.7--1.8, and
> found no meaningful improvement from bidirectional or small-world topology.
> It also showed a near-unit 25-mode zero-point spectrum and controllability
> PR near 6. The active question is therefore driven-state collapse under a
> rich input, not generic graph connectivity or merely increasing spectral
> radius. Phase 7 then found a reproducible, core-only improvement from
> deliberately separated leak bands: NRMSE 0.598 -> 0.537 and MC 7.90 -> 8.74
> across three paired seeds. This is promising but not yet a clean predeclared
> win because the banded condition ran below the required activity band and
> raw covariance PR remained below its 2.5 screen. Phase 8 then restored
> banded activity with one scalar leak multiplier and matched a uniform-leak
> control to the same activity. It preserved the NRMSE/MC advantage in all
> three pairs. Thus timescale diversity is now a clean mechanism result, but
> KNet remains well behind the tuned ESN reference. See the Phase-8 update
> immediately below.

---

## Phase 8: activity-restored three-band leaks (completed 2026-09-15)

**Sources:** `outputs_alliance_can/narma_phase8_activity_leaks/`
(`leak_bands_results.json`, `leak_bands_rows.jsonl`).

### Question and controlled construction

Phase 7 improved NRMSE and MC with three interleaved leak bands, but its
mean activity was only 0.401 rail versus 0.496 for the original baseline.
Phase 8 asked whether the result was truly about temporal diversity or merely
about operating the reservoir more quietly.

The core graph, 200 fixed delay-bank boundary OTAs, boundary pattern, eight
Heun steps, ridge readout, task stream, and paired seeds `0,1,2` were held
fixed. The only primary intervention was the one-dimensional scaling
`leak_j(alpha) = alpha * leak_j(banded)`: it preserves the slow/medium/fast
ratios without independently tuning three bands. Alpha was calibrated on a
separate input-only stream to 0.490 +/- 0.010 mean `|x| / rail`; NARMA targets
and task scores were not used in calibration. A scalar uniform leak was
calibrated by the same procedure. Candidate selections required finite states,
rail fraction <= 0.20, zero-point rho <= 1.05, and (for banded leaks) retained
separation of the realised slow/medium/fast pole groups. No post-leak global
gain normalization was applied.

### Paired results (mean across three seeds)

| Condition | NRMSE (lower) | MC | raw state PR | correlation PR | lag-9 product R2 | evaluation activity | rail fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-6 baseline | 0.598 | 7.90 | 1.73 | 2.32 | 0.467 | 0.496 | 0.161 |
| Banded alpha=1 banked Phase-7 reference | **0.537** | **8.74** | 1.94 | **3.46** | **0.520** | 0.401 | 0.103 |
| Banded, activity-restored | **0.556** | **8.48** | 1.88 | **3.24** | **0.496** | 0.453 | 0.131 |
| Uniform, activity-matched | 0.603 | 7.82 | 1.77 | 2.35 | 0.466 | 0.460 | 0.137 |

The calibration stream hit 0.490 for both new conditions in each seed. Their
evaluation-stream activities differ by only 0.007 on average and both lie in
the prospective 0.45--0.55 screening range; their rail fractions are also
similar and below 0.20. This is a valid amplitude/damping control despite the
expected stream-to-stream operating-point shift.

The activity-restored banded condition beats baseline NRMSE by 0.0388,
0.0404, and 0.0485 and gains 0.484, 0.625, and 0.608 MC in seeds 0, 1, and 2.
It beats its activity-matched uniform counterpart in NRMSE and in the
training-only-standardized ridge diagnostic in every pair. All three paired
prospective screening decisions are therefore wins.

### Interpretation and boundary of the claim

**Mechanism verdict: passed.** At matched activity and rail use, a core with
separated retention times produces better task error, memory, and lag-9
nonlinear accessibility than one with a uniform retention time. The restored
cores retain distinct realised pole groups: slow means about 0.96--0.97,
medium about 0.83--0.85, and fast about 0.67--0.69; their largest pole is
0.99--1.01. The unchanged boundary-current PR remains about 6, so richer
boundary injection is not the explanation.

**System-level verdict: not yet ESN-competitive.** The quieter alpha=1 banded
reference remains the best KNet condition (NRMSE 0.537), while the restored
condition is 0.556 and the tuned in-harness ESN reference is about 0.296.
Raw state PR remains below 2, although correlation PR is materially higher
than the baseline. PR is therefore retained as a diagnosis of unequal
state-variance usage, not a hard optimization target: the uniform Phase-7
control had higher raw PR but worse task performance.

Phase 8 supports keeping the real eight-tap KNet delay-bank path and adding
structured temporal diversity. It does **not** demonstrate a true engineered
shift-register delay line in the recurrent fabric, prove that core training or
multiple stages cannot close the ESN gap, or justify another generic topology
sweep. The next companion statement frames those unresolved questions.

---

## Phase 8B: tied causal tap-gate random existence probe (B=2 complete; B=3 partial, 2026-09-16)

**Source:** `outputs_alliance_can/narma_p8_tied_gate-60106240.out` (job still
running when this entry was written). The completed B=2 rows are the only
complete result; B=3 is partial and B=4 has not yet started.

### What was tested

This is the minimal *tied* gate, not the earlier failed dynamic-leak-only
mechanism. Eight present/past causal delay-bank voltages drive 2--4 shared
rails. Each node receives a mix of those rails and forms
`z_j = sigmoid(beta_j + sum_b w_jb*r_b)` and
`m_j = 2*(1-z_j)`. Once per sample, `m_j` multiplies the node's complete
post-KCL candidate current (core, boundary, and shunt contributions) and its
leak; clip stays outside. Thus `m -> 0` is a sample/hold and `m -> 2` is an
aggressive write/erase. This is **per-node accumulator gating**, not separate
per-core-edge routing.

The exact identity audit passed: `w=0, beta=0` gives `m=1` and the Phase-8
banded forward is bit-identical. The frozen-core probe then uses ridge only,
draws random `w` at scales 0.02, 0.05, 0.10, 0.30, and 1.0, and recalibrates
only the single band-preserving leak scale on a separate input-only stream.
It is an existence/falsification test, not gate training.

### B=2 findings

| random mix scale | NRMSE | MC | correlation PR | mean void-node fraction | result |
|---:|---:|---:|---:|---:|---|
| identity | 0.5560 | 8.478 | 3.235 | 1.00 | Phase-8 reference |
| 0.02 | 0.5562 | 8.473 | 3.235 | 1.00 | effectively off |
| 0.05 | 0.5560 | 8.468 | 3.236 | 1.00 | effectively off |
| 0.10 | 0.5566 | 8.436 | 3.236 | 1.00 | effectively off |
| 0.30 | 0.5597 | 8.340 | 3.237 | 1.00 | worse, still void |
| 1.00 | 0.5638 | 8.278 | 3.242 | 0.77 | gates begin to move, but worse |

The void rule is `std_t(z_j) < 0.02`. At scales through 0.30 every B=2 node
is void in every completed seed; the largest observed node gate standard
deviation is only 0.0155. At scale 1.0 some nodes finally exceed the threshold
(largest standard deviation 0.0404), but mean NRMSE is worse by 0.0078 and MC
falls by 0.200. Correlation PR is unchanged to the displayed precision.

### Interpretation and training decision

This **does not prove that learned tied gating is impossible**. A trained mix
could, in principle, align a few rails with useful delayed traces rather than
apply a random multiplicative disturbance. The identity initialization is also
not a dead gradient saddle: the nonzero rail projection gives `w` a gradient
at step zero. But it is a poor prior for an expensive full factorial training
campaign: weak random gates do not engage, while random gates strong enough to
engage erase useful memory rather than create separable temporal features.

The underlying PR bottleneck is not a shortage of linear modes. The banded
core already has 25 separated local modes and boundary/input-Jacobian PR about
5--6, yet the driven state is dominated by a few correlated histories. A
scalar gate per node scales all of that node's incoming write currents and
leak together. It does not introduce a new delayed signal, a protected storage
compartment, or a new mixing direction. Near identity it therefore only
rescales the same correlated trajectories; at large random scale it suppresses
or amplifies those trajectories indiscriminately, reducing delayed MC. Raw PR
is diagnostic rather than an objective, but its failure to move together with
correlation PR, lag-product access, and MC means there is no evidence here of
more useful state diversity, hence no reason for NRMSE to improve.

**Decision rule:** the activity-restored banded Phase-8 core is already the
accepted baseline; we do not need a better frozen initialization before any
training. However, do not launch the planned 3 seeds x 3 rail counts x all
controls gate-training matrix on this probe alone. First run one bounded B=2,
three-seed trainability pilot (train rails/mixes/bias and a trainable linear
readout; refit ridge for reporting). Promote it to the full matrix only if,
on validation-selected epochs, all three seeds satisfy all of:

- fewer than 90% void nodes (`std_t(z_j) >= 0.02` for at least 10% of nodes);
- activity 0.45--0.55 and rails <= 0.20;
- no degradation versus the static banded baseline and improvement in at least
  one independent mechanism metric: MC beyond the eight supplied taps,
  correlation PR, trajectory controllability, or lag-9 product R2.

The final pre-registered task promotion remains stricter: every seed must
improve held-out NRMSE by at least 0.02 over both static banded and trained
uniform controls, retain the beyond-tap MC gain, improve correlation PR or
trajectory controllability, and stay within activity/rail limits. A pilot that
does not clear the engagement and mechanism checks is a void/negative result,
not a reason to keep searching random initializations. B=2's completed random
probe provides no reason to expect B=3 or B=4 to rescue the design; their
unconditional execution in this batch is diagnostic only.

---

## Phase 7: interleaved three-band leaks (completed 2026-09-14)

**Sources:** `outputs_alliance_can/narma_p7_leak_bands-59846147.out` and
`outputs_alliance_can/narma_phase7_leak_bands/narma_phase7_leak_bands/`
(`leak_bands_results.json`, `leak_bands_rows.jsonl`).

### Question and controlled construction

Phase 6 showed that the physical delay-bank reservoir already had useful
memory, rich boundary injection (PR about 6), and 25 near-unit linearized
modes, while its *driven* state covariance still used only about 1.7--1.8
effective dimensions. The Phase-7 question was therefore not "add more
edges?" but "does a deliberately spaced set of node memory timescales turn
the same injection into more useful temporal features?"

All conditions use the same 25-node directed 5x5 torus, eight causal delay
terminals, 200 fixed LinearOTA boundary edges, fixed heterogeneous boundary
pattern with mean `isat_raw=-4.4`, tanh node broadcast, eight Heun steps per
sample, ridge readout, NARMA-10 stream, washout, and paired core seeds
`0,1,2`. The pre-existing core is normalized once (zero-point rho about 0.97)
and **no global gain normalization is run after changing leaks**.

- **Baseline:** existing log-uniform programmable core leaks, roughly
  `0.06--0.16`.
- **Banded primary:** spatially interleaved nodes `0,3,...,24` get slow leak
  `0.051` (9 nodes); nodes `1,4,...,22` get medium `0.223` (8); nodes
  `2,5,...,23` get fast `0.511` (8). These nominally correspond to per-sample
  retentions about 0.95, 0.80, and 0.60 before coupling/activation effects.
- **Uniform control:** all nodes use leak `0.18`, selected only on an
  independent calibration input stream as the nearest supplied scalar to the
  banded condition's mean amplitude. It is an *approximate*, not exact,
  amplitude match: it landed 0.019--0.039 rail below banded in calibration.

First principles: each leak is a voltage drain in `dx/dt = ... - leak*x`.
Slow nodes retain old inputs, fast nodes react mainly to new inputs, and
coupling can combine their differently aged traces. NARMA-10 requires both
current/past inputs and nonlinear products of them, especially near lag 9.

### Paired results (mean +/- population std, 3 seeds)

| Condition | NRMSE (lower) | R2 (higher) | MC | state PR | correlation PR | lag-9 product R2 | mean |x| / rail | rail fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5981 +/- 0.0069 | 0.6421 +/- 0.0082 | 7.904 +/- 0.125 | 1.728 +/- 0.065 | 2.322 +/- 0.075 | 0.4667 +/- 0.0019 | 0.496 +/- 0.013 | 0.161 +/- 0.009 |
| Banded leaks | **0.5372 +/- 0.0057** | **0.7113 +/- 0.0061** | **8.737 +/- 0.097** | 1.945 +/- 0.030 | **3.465 +/- 0.037** | **0.5198 +/- 0.0089** | 0.401 +/- 0.011 | 0.103 +/- 0.006 |
| Uniform leak 0.18 | 0.5785 +/- 0.0123 | 0.6650 +/- 0.0141 | 8.112 +/- 0.125 | **2.131 +/- 0.028** | 2.774 +/- 0.055 | 0.4761 +/- 0.0075 | 0.378 +/- 0.004 | 0.082 +/- 0.002 |

The task improvement is paired and uniform: banded beats baseline in NRMSE
by 0.0598, 0.0589, and 0.0640 in seeds 0, 1, and 2, respectively; it gains
0.779, 0.889, and 0.833 MC; and it beats the uniform control in NRMSE in all
three pairs. The lower rail fraction is an improvement, not a saturation
trade. Thus changing *leak diversity* helped beyond simply reducing voltage
and clipping.

### What the diagnostics say

- **The intended pole hierarchy was realized.** Baseline zero-input poles
  were broadly near-unit (roughly 0.80--0.98). Banded leaks created slow modes
  near 0.93--1.00, medium modes around 0.75--0.83, and fast modes around
  0.57--0.62. The uniform control instead made the spectrum narrow and faster
  (about 0.80--0.88). Its higher raw state PR but worse NRMSE is direct evidence
  that PR alone is not the objective.
- **Useful delayed/nonlinear information improved.** The banded condition
  increases MC beyond the eight supplied taps (e.g. lag-8 memory roughly
  0.25--0.29 versus 0.15 baseline) and improves the lag-9 product R2 from
  0.467 to 0.520. This is consistent with timescale diversity helping the
  NARMA product path, not merely passive variance inflation.
- **States are less redundant after variance normalization.** Raw covariance
  PR rises only 1.73 -> 1.94 because high-variance slow nodes still dominate
  amplitude. Correlation-matrix PR, which first normalizes per-node variance,
  rises 2.32 -> 3.46. The reservoir therefore has more distinct spatial
  signals even though its total voltage variance is still uneven.
- **The boundary did not change.** Net boundary-current PR remains about 6.01
  for every condition and trajectory-local all-tap input-Jacobian PR remains
  about 5.3. The gain is a core-timescale effect, not an accidental richer
  injection.
- **A loud all-ones common mode is not the main explanation.** Leading-PC
  alignment with the all-ones vector is already small in baseline (~0.08) and
  lower in banded (~0.04); variance exactly along that direction is about 1%
  or less. The causal linear surrogate's PR (~1.64 baseline) is close to the
  observed raw state PR, supporting the view that shared causal taps and
  similar filters already predict much of the concentration.

### Verdict and next experiment

**Verdict: promising mechanism demonstrated; predeclared screening win not
yet earned.** Banded leaks pass the NRMSE (>0.02 absolute improvement in every
pair), MC (>0.3 gain), rail (<=0.20), and uniform-control tests. They fail
the raw state-PR >=2.5 screen and run outside the requested mean-activity
band 0.45--0.55 (mean 0.401). The uniform control is also below the target and
was only approximately amplitude matched because the scalar calibration grid
was coarse.

Do not interpret this as a failure of timescale diversity: the pole bands,
paired NRMSE/MC gain, longer-delay memory, and product probe all move in the
predicted direction. Before expanding the seed count, repeat the paired run
with (1) a finer uniform-leak calibration around 0.14--0.17 and (2) a
core-only band adjustment that restores the banded activity to 0.45--0.55
while retaining separated poles and the fixed boundary drive. Controlled
rotation and non-normal amplification remain deferred; this small physical
leak intervention is now the strongest evidence-backed core lever.

---

## 1. Historical pre-delay-bank verdict

This section records the earlier, non-delay-bank campaign; it is not the
current quantitative verdict. The Phase-5--8 real delay-bank KNet improved the
frozen-core result to NRMSE **0.537--0.556** and MC **8.48--8.74**, but remains
well behind the tuned in-harness ESN at **0.296**. Before that delay-bank
intervention, the fabric sat at **0.67–0.79 error** with memory **0.4–3.5**
and behaved like ~1 node instead of 25 (`PR ~1.0–1.7`).
Every lever tried — recurrence rate, readout speed, drive strength, leak,
hetero-leak, small-world wiring, sparse drive, dense readout, VCA input gate,
orthogonal fast mixing, dynamic leak, hidden-conditioned VCA — either does
nothing or trades memory for saturation (nodes stuck at max voltage). The split
test proves the amplifier branch does all the work and the shunt branch is
decoration. The transplant resubmit finished all 54 corners: dead. The Phase-1
48-corner operating-point grid + hotter extension: dead. The 20-epoch gating
spike trains move error 0.755 → **0.670** (best fabric number to date) but LOSE
memory (3.03 → ~2.0) and saturate — the floor **0.67 is the memory ceiling in
target units** (see §10). The rung-1 linear screen certifies the pipeline
cannot hold linear memory at small signal (MC 0.003, Jacobian ~0.005). Depth
(Phase-4) is running; s1 timed out, s2 tracks s1's curve with no dividend yet.

Error scale used everywhere: **NRMSE 0 = perfect, 1.0 = just predict the average.**
Lower is better. Memory (MC) = how many past inputs the state can reconstruct
(healthy = 5–20+, gate = ≥4). Rails % = share of node-time stuck near max
voltage (gate ≤5%).

---

## 2. Why we compare untrained fabric to ESN (the init-only question)

We are not claiming training can never fix a bad start. We screen starts
because training is expensive and has never paid off here:

- Untrained fabric + linear readout: **0.7194**. Same net trained 35–80 epochs:
  **~0.676**. Gain ≈ 0.05, while clipping rises 0% → ~10%.
- Gated 20-epoch trains (spike B/C/D) buy a further 0.755 → **0.670** (≈0.085),
  but entirely through saturation (rails 15–18%, MC falls 3.03 → ~2.0).
- So the bucket holds almost no usable memory at birth, and training just turns
  the volume up until nodes clip. It does not turn 1-effective-node into a
  10-step memory. The 0.67 floor = fitting NARMA-10 with MC ≈ 2 (see §10).
- The ESN comparison is a control group with the identical data/washout/metric:
  random ESN bucket + same linear readout gets ~0.2–0.3. That proves the harness
  can measure memory (C0 ESN MC = 8.07) and the fabric bucket is guilty, not the
  task setup.
- Rule: **E0 gate first** (ridge <0.6 and/or MC clearly above baseline, rails
  sane → only then train). A failed start is "do not burn GPU", not "proven
  untrainable". The standing offer: one ~20-epoch train from the canary corner;
  predicted outcome is error stays ~0.7 and rails climb.

We never "moved the rail voltage". Supply `x_max = 3.0` is fixed. `rail%`
measures saturation, and every gain increase buys a little memory hand-in-hand
with a lot of saturation — a cliff, never a sweet spot.

---

## 2.5 System primer (what the fabric actually is — NARMA config)

Read `README.md` §§2–6 for the general machine; this is the NARMA instance
(single stage, 25 hidden nodes, torus 5×5, 100 edges, `tanh_free` cells,
~1261 params with the temporal readout):

- Node ODE per node `j` (effective capacitance 1):
  `dx_j/dt = Σ_in I_edge − Σ_out I_edge + I_boundary(j) + I_readout(j) − leak_j·x_j − clip(x_j)`.
  Integrated with fixed-step Heun, `t_span = 1.0`, `num_steps = 8` per input
  sample (`dt = 0.125`). State (26-D = 25 hidden + 1 output accumulator) is
  carried across samples; each epoch re-zeros it.
- Core edge current per directed edge: `u = gm·(s·(A·Vsrc − B·Vdst) + θ)`,
  `I_tanh = Isat·(tanh(u) + 0.01·u)`, plus an ungated resistive shunt
  `I_res = G·(Vsrc − Vdst)` with `G = softplus(g_raw)`, times a structural gate
  `σ(z_logits)` and per-endpoint compliance gates
  `σ((x_max − |V|)/0.02)` that shut the tanh branch off near the rails.
  Init is deliberately weak/near-linear: `gm_raw = isat_raw = −5.0` → gm ≈ 0.077
  in [0.01, 10]; `g_raw = −5.0` → G ≈ 0.0067; `s_raw ~ N(0, 0.1²)` (signs only,
  never touched by gain overrides — same seed means same sign pattern);
  `z_logits = 0` (gate 0.5); `raw_leak = −3.0` → leak ≈ 0.05 (≈0.97
  retention/sample); `x_max = 3.0`.
- Input enters ONLY through 25 boundary OTA edges (scalar input fanned out to
  every hidden node; the `u_logits` drive path exists but is all-zeros/unused).
  Raw `u ∈ [0, 0.5]` is affine-stretched to the full bipolar swing `V_u ∈
  [−3, +3]` by construction; `--input-scale` is a post-hoc multiplier.
  Targets are standardized (zero-mean/unit-variance; MSE in normalized space).
- Readout (two variants tested): **temporal** = single output-ODE accumulator
  node fed by a hidden→accumulator OTA mesh + quasi-linear `OutputAffine`
  (trained gains stayed ≈1.0/0.06–0.13); **dense** = linear map over all 25
  hidden (+ accumulator), 1080 params, fewer than temporal's 1261.
- `core_refresh_interval = k`: recompute the frozen core current every k Heun
  steps (k=8 ≈ legacy frozen, k=2 = 4 evals, k=1 ≈ fully dynamic). k=2/4/8 all
  plateau identically — closed line of inquiry.

## 2.6 Task, protocol, metrics, gates (read any log with this)

- **NARMA-10**: input `u ~ U[0, 0.5]`; target `y` depends nonlinearly (including
  the hard product term `1.5·u(n−9)·u(n)`) on the last ~10 inputs/outputs.
  Train: 4 streams × 2500 samples, truncated BPTT (chunk 25, state carried
  across chunks); test: fresh 1000-sample stream, washout 200. AdamW lr=1e-3.
- **NRMSE** = RMSE / target-std (1.0 = mean predictor). **R²** = 1 −
  SSE/SST. **MC** (Jaeger) = Σ_{k=1..20} R² of ridge reconstructing `u(n−k)`
  from states (healthy 5–20+; gate ≥4; C0-ESN reference 8.07). **PR**
  (participation ratio) = (Σλ)²/Σλ² of washed-state covariance (≈1 = all nodes
  move together; demoted to diagnostic after a healthy-MC ESN scored PR ≈ 3).
  **rail_frac** = fraction of state entries with |x| > 0.9·x_max (gate ≤5%).
  **Jacobians**: single-sample Heun-map eigenvalues at mid-stream points
  (healthy: max|λ| ≈ 0.95–1.05 with spread; observed: leak-pinned ≈0.94 or
  exploded 235–1325 at high gain).
- **Instruments**: ridge-from-states (linear info present?), MC + per-delay
  curve, y-MC (same machinery, target `y(n−k)` — measures the accumulator
  output-trace channel), IPC-2 (Legendre `P_2(u(n−k))` + `u(n−i)·u(n−j)` pairs
  around lag 9 — the nonlinearity detector), gate-zero (`z_logits → −12` at
  eval, restored after; kills tanh AND shunt — full-kill control), carry masks
  (reset-hidden keeps accumulator only; reset-all zeros everything).
- **E0 (zero-training) gate**: build the exact training-feature net, forward
  the train stream once, score ridge/MC/PR/rails. Train only on
  ridge clearly <0.65–0.6 and/or MC clearly above baseline with rails sane.
  E0 never certifies — it screens (single seed, eval-only).
- **Canary gates** (NARMA-10): anchor ridge in [0.60, 0.68] (else instrument
  blind/off-nominal); canary pass needs MC ≥ 4 AND ridge < 0.6 AND rails ≤ 5%;
  gate-zero flip = gz-ridge − ridge > 0.05 AND gz-MC < MC (core load-bearing).
- **Log-reading guide**: `ALL_*_AUDITS_OK` = code-correctness checks passed;
  `DECISION: ...` = pre-registered science verdict; `FAILED legs:
  split-verdict-void` = a ±0.05 determinism drift gate (see §4), not a crash;
  `CANCELLED DUE TO TIME LIMIT` = wall-clock kill, no science verdict;
  `rc=124` = per-leg timeout; E0 pace ≈ 80 s/corner (Jacobians + gate-zero
  double pass dominate).

---

## 3. Timeline of everything tried

| When / where | Test | Result — plain words |
|---|---|---|
| Baseline k2/k4/k8 | Recurrence rate ladder, 80ep trains | All **0.676–0.678**. Recurrence closed as a lever |
| E3 k8 + ridge diag | Frozen baseline + untrained readout | Untrained **0.7194** ≈ trained. Training buys ~0.05 |
| E4a–d | Readout LR x10, half drive, fixed leak 0.15, combo | **0.657–0.673**, all noise. Best (leak 0.15) only −0.02 |
| E2 ablation | Zero hidden state at eval, keep accumulator / zero all | **0.676 → 0.87 → 1.02**, MC → 0. State is load-bearing but holds blur, not memory |
| Baselines cal | Local ESN 0.5596, LSTM 0.2965 | Pipeline sound, ESN undertuned → BAND-UNKNOWN (later settled by S2a) |
| Frozen probe k2 | Ridge/MLP/PR/gate-zero on ckpt | Ridge 0.6434 ≥ 0.6 → init-regime primary; PR ~1–2; gate-zero barely moves → boundary feedforward dominates |
| Linear R0/C0/C1 | Raw taps / ESN / linear-reservoir-in-fabric | R0 taps 0.38 beat fabric; C0 ESN **MC 8.07** validates meter; C1 MC 2.53, PR 1.12 → FAIL |
| C1b / C1b-real / C3 | Orthogonal fast mixer / resistive-core version / gain sweep | C1b MC 1.62, ridge 2.43 (blowup); C1b-real MC 1.35, PR 1.00 (tuner drove G to floor); C3 peak MC 2.21 at gm=−2 with rails 50% + Jacobian 235→1325. Weak-mixing-as-sole-cause falsified |
| Canary | Anchor 0.6398 + priority corner | **CORE-USEFUL-NO-PASS**: anchor calibrated, canary 0.7611/MC 1.68/rail 15.2%, gate-zero flip 0.8122 confirms core matters but fails gates |
| S1-extended (k2/k8) | u-MC / y-MC / IPC-2 / product pairs on frozen ckpts | u-MC 2.56/2.80, y-MC 7.1/6.8, IPC-2 0.39/0.47 smeared, **all 7 product pairs 0.01–0.03 → lag-9 product path provably absent**. y-MC ≫ u-MC = accumulator channel is real |
| S2a tuned ESN (288) | sr/leak/size/input/ridge grid | Best **0.2964 < 0.4 → BAND-STANDS**. Fabric guilty, not setup |
| S3 dense readout | Train 25-dim read, 80ep fp32 | **0.6773 ≈ temporal 0.676 → H3 demoted**. Readout rank is not the binder. AMP overflow → `--no-amp` |
| S2b torus E0 (48) | gain × leak × drive | **0/48 pass**. gm=−5 rail-clean but MC ≤2; gm=−2 peaks MC 2.75 but rails 31–42%; gm≥0 garbage |
| S2c small-world (48) | Same on small-world graph | **0/48 pass**, best MC 3.19/ridge 0.64. Topology ≈ torus → torus kept |
| Round-2 E0 dense/sparse (18+18) | Full vs sparse input fan-out | Dense best MC 1.98; sparse best 1.24, no gate-zero flip → **sparse drive killed** |
| Round-2 trains | Plain-full control + VCA-1 | Plain **0.6834** (3rd reproduction of plateau); VCA-1 **0.7860 (Δ +0.10, day-0 rails 36%)** → VCA hurt on railed init; fork resolves to Step 6 forget gate |
| **Split 59625759** | Shunt-kill / tanh-kill / both-kill + per-node diag | See §4 |
| **Transplant 59664806** | 54-corner Isat-led grid | Killed by wall-clock, 1/54 corners — resubmitted as 59667050, see §5 |
| **Transplant 59667050** | Full 54 corners | Best rail-free **0.619/MC 2.90**; best MC **3.50 at 34.5% rails** (disqualified). isat=-2 = memory lever, isat=-6 = poison. TRANSPLANT-DEAD, see §5 |
| **Phase-1 E0 59691417** | 48 corners gm×isat×drive, leak 0.1 | Best **0.635/MC 2.26/PR 1.5, 0% rails** (gm-1/isat-5). 0/48 pass, see §10 |
| **Phase-1 ext 59712084** | 18 hotter corners (gm to +1) | MC > 3 exists ONLY with ~30% rails; rail-free cap MC ~2.3, see §10 |
| **Spike trains 59691420** | A/B/C/D × 20ep (dyn-leak, hidden-VCA) | A 0.755/MC 3.03/0% rails; B/C/D **0.670**/MC ~2.0/15–18% rails. B/C/D digit-identical trajectories → gates don't engage as gates. NRMSE win, memory loss, see §10 |
| **Rung-1 59724426** | Native-linear screen + steps16/gm6 variants | MC 0.003, J max\|λ\| 0.005 (ESN ref MC 12.0 same streams). Step-size and 7×-weight hypotheses dead. Certified FAIL, see §10 |
| **Depth 59719147** | s1/s2/s3 × 80ep (partial) | s1 TIMED OUT rc=124 (~ep40, no result); s2 at ep40 tracks s1 (loss ~0.60, NRMSE 0.68, rails 2–4% climbing). No depth dividend yet. LEG_TIMEOUT too short — resubmit with 12000s, see §10 |

---

## 4. Latest run 1: split leg `narma_split-59625759.out` — what each arm was

Setup: the exact canary priority corner (core gm −3.5 / Isat −2.0, fast leak,
weak drive), rebuilt on CUDA, full streams (4 × 2500, washout 200).

- **Audit: PASSED.** `ALL_SPLIT_AUDITS_OK` (S1–S6). The later `FAILED legs:
  split-verdict-void` is not a code failure — it is a ±0.05 drift gate tripping
  on one of five numbers (see below).
- **Base (normal circuit):** ridge 0.7327 / MC 1.58 / PR 1.06 / rails 15.2%,
  5 loud nodes. Reproduces canary (0.7611/1.68/15.2%) except MC.
- **Per-node check:** loudness follows the plus/minus wiring pattern
  (corr −0.556), not drain size (0.000) or boundary push (0.125).
- **Arm A (cut drains, keep amplifiers):** ridge 0.7985, MC 1.46, PR 1.07,
  rails **19.1%**, still 5 loud. Not the homogenizer — clipping gets worse.
- **Arm B (cut amplifiers, keep drains):** ridge 0.8129, MC 0.86→0.39 thin,
  PR 1.00, rails **0%**, 0 loud. Everything dies.
- **Arm C (master off, cut both):** ridge 0.8122, MC 0.86, PR 1.00, rails 0%.
  Matches old 0.8122/0.86 to 4 decimals → rebuild is valid.

Reading: **B == C, A ≈ base.** All memory lives in the tanh amplifiers; the
shunt drains carry nothing. Fix target = jittered gm/Isat, not jittered-G.

Why the "VOID": base MC drifted 1.68 → 1.58 (0.096 > 0.05 tolerance) while
ridge/rail/arm-C all passed. MC is a sum of 20 noisy R² fits — the thin
preflight on the identical net gives MC 0.67 vs 1.58 full-length. The 0.05
absolute gate is too tight for that meter; determinism evidence (rails to
0.0001, arm-C exact, corrs identical) is strong.

---

## 5. Transplant: `narma_transplant-59664806.out` (killed) → `narma_transplant-59667050.out` (full 54)

Plan: 54-corner core-Isat-led grid (2 gm × 3 Isat × 3 leak × 3 drive),
triggered by the canary CORE-USEFUL-NO-PASS verdict. Eval-only.

- **Audit: PASSED.** `ALL_TRANSPLANT_AUDITS_OK` (T1–T10), both jobs.
- **59664806:** `CANCELLED DUE TO TIME LIMIT` before any corner — no verdict.
- **59667050 full grid:** `DECISION: TRANSPLANT-DEAD`, 0/54 pass. Best
  rail-free: gm-3/coup/raw:-0.5/drive-0.3 → ridge **0.6186**, MC 2.90,
  PR 1.30, 0% rails. Highest MC anywhere: gm-3/isat-2/drive-0.3/raw:-1.5 →
  MC **3.50** at **34.5% rails** (disqualified).
- **isat is the memory axis:** every high-MC corner uses isat=-2; every
  isat=-6 corner collapses (MC 0.1–0.8, PR 1.00). Hot saturation current
  buys memory; cold kills it. Coupled-isat sits in between (MC ~2–2.9).
- Same coupling as everywhere: MC ≳ 2.5 arrives with rails ≳ 15%.

---

## 5b. Gating campaign (knet-gated-memory): Phase-1 E0, spike trains, rung-1, depth

**Phase-1 operating-point grid** (`narma_gating_e0-59691417.out`, 48 corners,
gm {-4..-1} × isat {-10..-5} × drive, fixed leak 0.1, 1923 s): 0/48 pass.
Best gm-1/isat-5 → ridge **0.6347**, MC 2.20–2.26, PR 1.31–1.49, 0% rails.
Clean gradient (hotter = better ridge + more MC) but PR never exceeds ~1.5.

**Hotter extension** (`narma_gating_e0x-59712084.out`, 18 corners, gm to +1,
1513 s): first MC > 3 corner (gm-1/isat-3, MC 3.14) — at **30.7% rails**.
Rail-free cap stays MC ~2.3. Memory and rails are coupled: the unsaturated
fabric cannot hold more than ~2–3 delays of input.

**Spike trains** (`narma_gating_spike-59691420.out`, A/B/C/D × 20ep, k2):
A 0.7549/MC 3.03/0% rails → B/C/D all **0.6697**/MC 1.9–2.1/15–18% rails.
Epoch-0 equivalence held on GPU (epochs 0–6 digit-identical across legs).
B/C/D trajectories are mutually digit-identical → dynamic leak and
hidden-VCA do not engage as gates; all three fall into the same saturated
attractor. NRMSE win (−0.085, best fabric number to date), memory loss.

**Rung-1 native-linear** (`narma_rung1-59724426.out`, preflight + canonical +
steps16 + gm6): MC 0.003–0.005, J max|λ| 0.0046 (0.0014 at 16 steps), state
RMS ~3e-4, vs ESN reference MC 12.01 / 0.4588 on identical streams.
Preflight reproduces the CPU smoke bit-for-bit (cross-device determinism).
Finer steps and 7× hotter gm move nothing → step-size and modest-weight
hypotheses dead. The per-sample map forgets ~99.9% (leak 6.9); the ESN
leak=1→L=6.9 retention match reproduces forgetting without mixing.

**Depth** (`narma_depth-59719147.out`, still running): s1 leg TIMED OUT
(rc=124, LEG_TIMEOUT 5400 s < ~9000 s needed) around epoch ~40 with no
result row — salvage via `--init-from` checkpoint or rerun with
LEG_TIMEOUT=12000 (epochs flatten by ~35 anyway, so 40ep also suffices).
s2 at ep40 tracks s1's curve (loss ~0.60, NRMSE 0.68, rails 2→4% climbing);
no depth dividend visible yet. s3 pending.

**The 0.67 floor:** best NRMSE anywhere is 0.6697 and every gain arrives
through saturation while unsaturated MC caps at ~3. The floor IS the memory
ceiling in target units — fitting NARMA-10 (which needs MC ~12) with MC ≈ 2
caps at ~0.67 regardless of lever.

---

## 6. Standing model (5 requirements) + ranked hypotheses H1–H6

What the fabric needs as an RNN, per the revised plan: (1) persistent state —
has it (E2); (2) stability/echo-state property — has it (leak pins max|J| ≈
0.94; the one thing done right); (3) timescale diversity — LACKS it (deficit
#1: uniform leak; hetero ≈ fixed; C1b's 50 identical slow integrators scored
worse than weak random mixing); (4) input separability — LACKS it (deficit #2:
identical full fan-out + weak gm → PR ≈ 1; sparse drive and small-world both
failed); (5) nonlinear expansion / product path — ABSENT (deficit #3:
gm ≈ 0.077 near-linear at init; S1 product pairs ~0.02; C3 gain arrives with
the saturation cliff; VCA-1 negative on a railed init).

Ranked hypotheses for "untrained ≈ trained, masking carryover destroys
everything, tuning everything else changes nothing":

- **H1 (leading, strengthened ×3): reservoir near-linear and weak at init.**
  ~100 edges couple through a tiny-gain, almost-linear map — a slow linear
  integrator, not a reservoir. NARMA needs nonlinear memory; the ridge
  ceiling is low and training grows gains straight into compliance shutdown.
  Would flip: any E0 corner with MC > 3 and ridge < 0.5. Update 09-14: three
  independent screens agree — 48-corner + 18-corner E0 grids (PR ≤ ~2
  rail-free), rung-1 (J ~0.005, MC 0.003), spike trains (gates don't
  engage; NRMSE win is saturation, not memory). The unsaturated fabric
  holds MC ≈ 2–3 max; memory arrives only with rails.
- **H2: compliance/saturation shutdown.** Nodes near ±x_max decouple; training
  pushes voltages outward in every run (h_rms rises monotonically, rails track
  the plateau). Counter-evidence: half drive still saturates and scores the
  same — symptom as much as cause. Would flip: a rail-penalized run where MC
  recovers before NRMSE moves.
- **H3 (demoted by S3): single-accumulator readout bottleneck.** Frozen
  full-hidden ridge 0.6434 already beat trained 0.676, but the trained dense
  readout landed 0.6773 ≈ temporal — readout rank is not the binder.
- **H4 (dead): credit assignment through frozen core.** k=2 (dynamic) scores
  identically to k=8 (frozen); loss descends 0.99→0.59 — gradients work, they
  lead to a poor basin.
- **H5 (settled against the fabric by S2a): task-setup artifact.** Tuned ESN
  0.2964 in-harness reproduces the band — the bar is real. (C0-ESN readout
  0.77 stays off-band on one corner; noted, does not invalidate MC.)
- **H6: raw capacity.** ESN-25 reaches the band and MC ≈ 0.2 signals
  qualitative mixing failure, not gradual shortfall; h49 control (C1) also
  fails. Would flip: hidden-49 E0 corner passing all gates.

Known instrument wart (do not misread old MC numbers): the legacy
`memory_capacity` paired past-states with future-targets (forward prediction);
a noiseless delay line scored MC ≈ 0.02 under the old convention vs 1.00 fixed.
All "MC ≈ 0.2" values before the v5 fix were overfit-level artifacts; verdicts
were re-anchored via the C0 full-stream control (MC 8.07).

Ruled out: recurrence rate, readout speed/rank, drive strength (not sparsity →
then sparsity too), leak value, topology family, resetting as a feature,
under-training.

---

## 7. Open legs (in order)

1. **Depth s2/s3 verdict** (running) — resubmit s1-salvage + raise
   LEG_TIMEOUT to 12000 s; make s1 skippable in future depth batches
   (in-batch control already banked). If s2/s3 ≈ s1 with same rail creep,
   close Phase-4.
2. **Rail-managed long train from a spike init** (NRMSE track) — leg-B-type
   start (0.670 @ 20ep) with rail pressure, 80ep. Tests whether the fit
   survives without full saturation.
3. **Slow-leak rung-1 variant** (`--esn-leak 0.3` → L ≈ 0.36) — one leg,
   closes the leak axis of Phase-2 airtight if MC stays ~0.
4. **Canonical-init VCA rerun** — VCA-1 ran on a 36%-railed init with fallback
   not taken; one clean test still owed before cells. (Downgraded: spike C
   suggests hidden-VCA adds nothing, but that was on a saturating track.)
5. **Drive-amplitude-at-fixed-gain sweep / standardized-u** — sparsity killed,
   amplitude never isolated; inputs stretch to full ±3 V by construction while
   targets are standardized.
6. **Step 6 forget gate** (`narma-forget-gate`, softplus-unbounded
   `leak_eff(t) = softplus(a·V_u + c·x + b)`) — note: the spike dyn-leak is
   its eval-only cousin and did not engage; the trainable version's case
   now rests on depth failing first.
7. Then cells (bistable revival), then systolic-as-ceiling-diagnostic. Two-core,
   feedback, Sugiura, Mackey-Glass stay deferred per revised-plan §9–§10.
   Structural talk is now triggered per the advisor rule (two consecutive
   NO-CORNER-BREAKS-RIDGE + spike/rung confirmations) IF depth closes
   without a dividend.

---

## 8. Number appendix (copy-paste sources)

Raw logs live in `outputs_alliance_can/`: `narma_split-59625759.out` (79 lines:
audit + thin-smoke + full leg + 5-number verdict gate),
`narma_transplant-59664806.out` (40 lines: audit + 1-corner preflight +
TIME-LIMIT kill), `narma_transplant-59667050.out` (160 lines: full 54-corner
table, TRANSPLANT-DEAD), `narma_gating_e0-59691417.out` (99 lines: 48-corner
table, best 0.6347/MC 2.26), `narma_gating_e0x-59712084.out` (69 lines:
18 hotter corners, MC>3 only with ~30% rails),
`narma_gating_spike-59691420.out` (245 lines: A/B/C/D 20ep curves —
A 0.7549/MC 3.03, B/C/D 0.6697/MC ~2.0),
`narma_rung1-59724426.out` (96 lines: preflight + canonical/steps16/gm6,
MC ~0.003, ESN ref 12.01), `narma_depth-59719147.out` (running: s1 TIMEOUT
rc=124, s2 at ep40 tracking s1),
`narma_round2_gated-59587991.out` (345 lines: audits A–R3 +
E0 dense/sparse tables + plain/VCA train curves + gate summary),
`narma_revised_plan-59367932.out` (S1 k2/k8 + 288-corner ESN cal),
`narma_revised_plan-59492665.out` (48+48 E0 tables + dense-train result).
Canary anchor: `output/canary/canary.json` (priority `config_tag` encodes
order/seed/cell/hidden/refresh/tspan/steps/gain/leak/drive/washout/jacobian +
per-lib overrides).

- Canary: anchor 0.6398 [0.60,0.68]; priority 0.7611 / MC 1.68 / PR 1.06 /
  rail 15.2%; gz 0.8122 / MC 0.86 / PR 1.00 / rail 0%.
- S1 k2: ridge 0.641, u-MC 2.56, y-MC 7.10, IPC-2 0.39, pairs 0.01–0.03.
  S1 k8: 0.645 / 2.80 / 6.79 / 0.47 / 0.01–0.03.
- S2a ESN best 0.2964 → BAND-STANDS.
- S3 dense: 0.6773 / R² 0.5408 / MC 3.46, 1080 params; untrained 0.7194 both paths.
- S2b best rail-sane gm=−5/hetero/drive-1.0: 0.6606 / MC 1.98 / PR 1.17 / 0%.
  gm=−2 peaks MC 2.6–2.75 at 31–42% rails.
- Round-2 plain-full: 0.6834 / R² 0.5323 / MC 2.73; VCA-1: 0.7860 / 0.3814 /
  MC 2.47 (Δ +0.1026).
- Split full: base 0.7327/1.58/1.06/15.2%; A 0.7985/1.46/1.07/19.1%;
  B 0.8129/0.86/1.00/0%; C 0.8122/0.86/1.00/0%.
- Transplant preflight corner: −3.5/coupled/raw:−1.5/0.1 → 0.8524/0.63/1.10/0% FAIL.
- Transplant full best rail-free: −3.0/coup/raw:−0.5/0.3 → 0.6186/2.90/1.30/0%.
  Best MC 3.50 at 34.5% rails (−3.0/−2.0/raw:−1.5/0.3).
- Phase-1 E0 best: −1.0/−5.0/leak-0.1 → 0.6347/2.20–2.26/1.31–1.49/0% (all drives).
  Extension: MC 3.14 at 30.7% rails (−1.0/−3.0/0.25).
- Spike trains (20ep k2): A 0.7549/0.4294/MC 3.03/0% rails/1261p;
  B 0.6698/0.5509/MC 2.06/18.3%/1363p; C 0.6697/0.5509/MC 1.93/15.8%/1563p;
  D 0.6697/0.5510/MC 1.93/15.4%/1563p.
- Rung-1 CUDA: canonical MC 0.003/J 0.0046; steps16 MC 0.003/J 0.0014;
  gm6 MC 0.005/J 0.0047; ESN ref 0.4588/MC 12.01 (2500-samp), 0.5831/9.62 (300-samp).
- Depth s2 at ep40: loss ~0.60, NRMSE 0.68, rails 2–4% climbing (s1 lost to rc=124).

---

## 9. What we want from an outside reviewer (answerable from this file + README)

Narrowed position since v7: **no nonlinearity lever has moved MC rail-free,
so nonlinearity is no longer the suspect — memory persistence is.** Five
independent screens agree the unsaturated fabric holds MC ≈ 2–3 max
(E0 grids, transplant, rung-1, spike trains, frozen probes), while the task
needs MC ~12 (ESN: 0.459) and the NRMSE floor sits at 0.67 exactly where an
MC≈2 fit caps out (§5b). The open question is no longer "which curve" but
"what holds echo without railing."

1. **Memory–rail coupling: known failure mode?** Every corner with MC ≳ 2.5
   rails at ~15–35%; every rail-free corner caps at MC ~2.3 (transplant +
   Phase-1 extension grids). Is "unsaturated = amnesic, memory = saturated"
   a recognized analog/neuromorphic-reservoir failure mode, and what is the
   standard breaker (operating-range regularization, input-scaling
   discipline, spectral-radius-style init, something structural)?
2. **Full-rank contraction, not rank collapse — what init fixes that?**
   Rung-1 measures per-sample Jacobian max|λ| ≈ 0.005 with rank proxy ≈ 26:
   all directions exist but shrink ~200× per sample (leak 6.9 + tiny
   weights). 7× hotter gm and 2× finer steps move nothing. What weight/leak
   construction gives per-sample spectral radius ~0.9–1.0 with spread AND
   stays rail-free — is there a known-good answer for OTA-fabric reservoirs,
   or does this already imply the topology (local torus, degree ~8) cannot
   support it?
3. **Does the 0.67 floor confirm the memory-ceiling framing?** Best NRMSE
   anywhere is 0.6697 (spike B/C/D @20ep, railed); best rail-free trained is
   ~0.69–0.75. If MC ≈ 2 implies an NRMSE floor near 0.67 on NARMA-10, what
   MC does the 0.15–0.29 band require — and is there any path from MC≈2 to
   MC≳5 that does not go through saturation?
4. **Do the dead gates kill Step-6 (trainable forget gate)?** Dynamic leak +
   hidden-VCA train to digit-identical trajectories as their ablations
   (B/C/D) — they add fit capacity via saturation, not echo. Is that
   evidence against *trainable* gating generally, or only against
   zero-init gating on a saturating track? What would distinguish the two
   in one ≤1-GPU-day experiment?
5. **If depth closes without a dividend (s2/s3 ≈ s1, decision pending),
   rank the structural options:** systolic delay-line (advisor candidate),
   bistable/cellular revival, two-core/feedback — by expected MC-per-rail
   and by implementation risk on this fabric. Which would you build first?
6. **What 3 diagnostics (each ≤1 GPU-day) would you request first?**
   Current shortlist: (a) slow-leak rung-1 (`--esn-leak 0.3`, closes the
   leak axis); (b) rail-managed 80ep train from a spike init (does the
   0.67 fit survive without saturation?); (c) your pick to replace one of
   these.

Glossary for the reviewer: E0 = zero-training probe (build → forward once →
ridge/MC/PR/rails); gz/gate-zero = force core edge-gates off at eval
(full-kill control); carryover = hidden+accumulator state passed
sample-to-sample (vs re-zeroed); hetero-leak = per-node log-uniform time
constants τ ∈ [1,40]; VCA = low-rank input-driven edge multiplier
`2σ(b + (u·W)·v_e)` (product-path candidate); canary = 1-corner tripwire
before a grid; transplant = 54-corner core-Isat-led grid; split = shunt-kill
(A) vs tanh-kill (B) vs both-kill (C) decomposition; rung-1 =
tanh-core native-linear screen (small-signal fills, shunt killed,
ESN-mapped leak/drive, in-harness ESN parity ref); spike = dynamic-leak
(3a) + hidden-conditioned core VCA (3b) trains, epoch-0 identical to
legacy; depth = 1/2/3-stage per-sample pipeline comparison; twin =
standalone digital node-tanh probe `x=(1−a)x+a·tanh(Wx+W_in·d)` (plan
`twin-node-tanh`, spec `twin-probe-spec`), no stages, no training, ridge
readout only.

---

## Phase 9: digital node-tanh twin — architecture inversion confirmed (completed 2026-09-17)

**Sources:** `outputs_alliance_can/narma_twin-60146708.out` (coarse, 144
corners), `outputs_alliance_can/narma_twin_fine-60149835.out` (fine, 369
corners). Code: `twin_node_tanh()` in `narma_linear_controls.py`; audits
`audit_twin_standalone.py` 7/7 green (+ `probe_audit2.py` §§12–18).
Batches: `narma_twin_batch.sh`, `narma_twin_fine_batch.sh` (both gitignored,
ship with `git add -f`). Full write-up: `docs/twin-node-tanh-report.md`.

### Question and construction

Neither prior "linear" result tested the cell under discussion — dense
linear mixing → sum → **per-node** tanh (the ESN update). Rung-1 ran
per-edge tanh in its linear regime under a leak-dominated map (unwinnable
by construction); C1/C1b ran pure-linear reservoirs with no nonlinearity
anywhere. The twin fills that bottom-right cell digitally, as a bare
once-per-sample discrete map (zero stages — no Heun, no transfers; the
integrator is removed as a confound), before any analog cell design.

Two drive harnesses, both run: **taps8** (8-tap causal delay bank through
25×8 `W_in`, mirrors Phase-6–8 fabric input) and **scalar** (single
rail-mapped value through 25×1 `W_in`, C1-style; ESN parity sees
bit-identical drive). Pre-registered gates: PASS_ARCH = NRMSE ≤ 0.50 and
MC ≥ 5; FAIL_FABRIC = NRMSE ≥ 0.60 and MC ≤ 3; VOID = tanh disengaged or
non-finite.

### Grids

- **Coarse:** radius {0.7,0.9,1.0,1.1} × leak a {0.3,1.0} × input scale
  {0.2,0.5,1.0} × seeds {0,1,2} × 2 harnesses = 144 legs, 144/144
  complete, 0 voids (tanh engaged everywhere).
- **Fine:** taps8 radius {1.0,1.1} × a {0.2,0.3,0.5,1.0} × scale
  {0.10,0.15,0.20,0.25,0.30} × W-seeds {0,1,2} × seeds {0,1,2} (360)
  + scalar 3 best-guess combos × 3 seeds (9) = 369/369 complete, 0 voids.

### Results (full 2500-sample streams, washout 200)

| sweep | harness | NRMSE mean / best | MC mean / max | parity-ESN | verdicts |
|---|---|---|---|---|---|
| coarse | taps8 (72) | 0.647 / 0.505 | 8.10 / 10.29 | 0.763 / 5.22 | 0 pass, 0 fail |
| coarse | scalar (72) | 0.688 / 0.547 | 4.65 / 8.94 | 0.763 / 5.22 | 7 fail* |
| fine | taps8 (360) | 0.536 / **0.393** | 9.35 / 13.11 | 0.660 / 7.88 | **78 pass**, 0 fail |
| fine | scalar (9) | 0.632 / 0.547 | 7.35 / 8.94 | — | 0 pass |

\*All 7 FAIL legs are one dead corner (scalar/a=0.3/is=1.0: hot drive +
slow leak + scalar bottleneck, NRMSE ~0.64/MC ~2.8) — a drive/retention
mismatch, no blowup anywhere (worst twin NRMSE 0.84 vs C1b's 2.43).

Best corner: taps8 r=1.0/a=1.0/is=0.1/W-seed 2/seed 1 → **0.3926 /
MC 12.45 / PR 7.17** vs parity ESN 0.5219 same stream. Top-12 corners span
0.39–0.44, MC 10.7–13.1. Participation ratio: fine taps8 PR mean 4.68
(2.4–8.4), tracking leak (a=0.2 → 2.95 … a=1.0 → 7.55); raw and
standardized PR agree within ~1 (no variance-domination pathology, unlike
the old fabric). Jacobians healthy (max|λ| 0.35–1.0, rank-proxy up to ~25).

Three clean gradients: (1) **quieter drive monotonically better**
(is=0.1 mean 0.491 → is=0.3 mean 0.576); (2) **a=1.0 maximizes memory**
(MC ~12–13, PR ~7) at quiet drive; (3) **W-seed moves NRMSE ~0.03**
(mean, 0.07 max) at fixed settings — one-W-per-corner left performance
on the table. Radius nearly flat (±0.01–0.02). Seed 1 kindest on both
sides (parity ESN agrees — a stream effect, not luck).

### Interpretation and boundary of the claim

**Mechanism verdict: passed with margin.** The coarse grid's 0.5047
near-miss (0.005 over gate) converted to 78 passes on the fine grid:
dense mixing → sum → per-node tanh holds MC 9–13 with PR 4–7.5 and
bounded, engaged tanh across 513 corners. The twin beats its
identical-stream ESN parity ref in virtually every group mean, sits
where the real banded KNet fabric sits (0.537–0.556), and leaves old
fabric MC ~2 and C1b far behind. Remaining gap to the tuned 0.296 ESN
reads as tuning headroom (W search, finer scale/leak — note the twin
got 3 W draws/corner vs parity ESN's one, ~0.03 flattery on best-corner
comparisons; group means are unaffected), not a second missing mechanism.

**Not shown:** any analog realization (linear-OTA mixer + diff-pair node
tanh still unbuilt); the multistage reframing (stage = mixing bank,
transfer = nonlinearity); Heun-integrated rhs-override twin (deferred).
Random-vs-random comparison only — tuned-twin vs 288-corner tuned ESN
remains unrun on both sides.

**Promoted per twin-probe-spec:** (a) analog realization study; (b)
multistage with learned mixing banks; (c) dense-start anneal-prune under
a device budget. Scalar-harness ceiling (MC ~4–5 at a=0.3) vs taps8
(MC ~9–13) further confirms the delay bank earns its keep — fabric as
nonlinear expander, not memory.

### Recontextualization (2026-09-18): the taps carry the memory

**New evidence:** good two-bar performance appears *only* with the
explicit 8-tap delay bank. Same net, same streams, taps removed → every
bar fails in both campaigns (discrete map and Heun flow map):
- Flow twin, config-matched (r=1.1/is=0.2/λT∈{1,3}, n=18+18,
  `outputs_alliance_can/post_twin_gates/post_twin_gates/all/gate1_flowtwin.csv`
  vs `gate1_scalar_confirm.csv`): taps8 MC mean **8.27** [7.41, 9.17] vs
  scalar mean **4.33** [3.25, 5.63] → **Δ +3.94**. Scalar verdict
  DIGITAL_ONLY; taps8 FLOW_VIABLE. (Mean ridge 0.685 vs 0.659 on the
  same grid — the task gap is small; the memory gap is the whole story.)
- Discrete twin fine grid, config-matched (r=1.1/a=0.3/is=0.2,
  `outputs_alliance_can/narma_twin_fine-60149835.out`): taps8 mean
  **8.90** (9 legs) vs scalar mean **4.53** (4.521/4.665/4.404) →
  **Δ +4.37**. Full-grid means agree: coarse 8.10 vs 4.65 (Δ +3.45);
  fine 9.35 vs 7.35 (mixed configs).
- Scalar's best case still fails: discrete scalar best NRMSE 0.547
  (> 0.50 gate, **0 passes** in both campaigns; scalar a=1.0 legs reach
  MC ~8.7 but at NRMSE ~0.65+, still failing PASS_ARCH); flow scalar
  best MC 5.63 (< 8.0 gate).

**Mechanism (code, not speculation):** the taps8 drive hands the net
u[t..t−7] directly (`flow_twin_drive` taps8 branch,
`post_twin_gates.py:148–153`; `twin_delay_bank`,
`narma_linear_controls.py:3904–3918`), while MC sums R² over delays
k=1..20 against the scalar stream (`memory_capacity`,
`narma_experiment.py:~490–503`). Delays 1–7 are therefore
reconstructible from the input alone — up to **+7 MC points are free**,
so MC > 8 proves only ~1+ units of genuine recurrent memory. The
measured taps value-add (+3.9 flow, +4.4 discrete-matched) sits below
the +7 ceiling because tanh mixing distorts the window and the no-tap
baseline already holds short-delay memory recurrently. Corollary: the
MC ≥ 5 half of discrete PASS_ARCH lies *entirely inside the free
window* — the memory half of a pass is achievable on window alone
(the NRMSE ≤ 0.50 half still does real work).

**ESN-parity correction:** `ne.ESN` is scalar-only by construction
(25×1 `W_in`, `narma_experiment.py:262`, scalar multiply in
`_run`), and the discrete-twin code says so itself
(`narma_linear_controls.py:4146–4151`: taps8 parity "approximate by
construction (documented, not gated)"). On identical streams (matched
grid above): taps8 twin **8.27** / scalar ESN **7.61** / scalar twin
**4.33** (ESN range ~7–11 across all streams). The "beats parity in
virtually every group mean" line above compares a twin with 8 taps of
free history against a tap-less ESN — flattered by roughly the free
window. Drive-matched ranking is ESN (all genuine) > scalar twin (all
genuine); the taps8 twin's beyond-window portion is the small remainder.

**Standing:** the Phase-9 headline "fabric as nonlinear expander, not
memory" is strengthened, not weakened — the net contributes genuine but
modest recurrent memory (~3–5 units: scalar MC in both campaigns, both
maps) and the delay bank contributes the rest. The buildable spec (λT
3.0 / 200 edges / 5% / torus groups) still stands as the device recipe,
but any multistage readout design must assume single-digit recurrent MC
and size explicit delay handling accordingly. Taps8-vs-taps8 ESN parity
(matrix-input ESN) is the open measurement that would close this.

---

## Phase 10: aux-core drive probe (eval-only twin, 2026-09-18)

**Plan:** `aux-core-drive`. **Spec:** `aux-core-probe`. **Code:**
`aux_core_probe.py` (+ `audit_aux_core.py`). **Status:** implemented
and locally audited on Windows CPU; GPU deployment deferred.

### Question and construction

Can any learnable feedforward aux core replace the static 8-tap delay
bank as the main-core drive? The static bank costs 8·M routed voltages
plus (nodes × 8·M) dense mixer weights per input and is unprunable /
unoptimizable; the spec demands an aux core that EARN fabric integration
against the bank on MC at comparable edge cost. Three flavors per spec:
**echo-state** (8-node ESN at radius 0.95, leak ∈ {0.3, 1.0}),
**shift-register** (exact chain, gain ∈ {0.7, 0.9}, head injection 1.0;
"interpretable learned delay line; exact chain asserted"),
**slow-feature bank** (8 decoupled leaky integrators, log-spread
τ ∈ [1, 40]).

Twin-level only. No fabric changes. No feedback. No training. Main core
pinned to the Gate-1 best corner (Heun flow-twin r=1.0, λT=1.0,
scale=0.2, dt=0.125, hidden_dim=25). Reuses every flow-twin instrument
already validated in `post_twin_gates.py` (ridge, washout-corrected per-
delay MC, PR, tanh-engagement, ESN parity). Per row carries an itemized
edge count: aux_recurrent + aux_fan_in (spec items (ii)/(iii)) +
main_fan_in (drive width × hidden_dim), compared against the reference
``ref_total_edges = hidden_dim × n_taps`` (200 at default dims).

Drive-power parity (audit fix, 2026-09-18): every leg's drive into the
main core is scaled to RMS == input_scale, mirroring the taps8 branch
of `flow_twin_drive`. Pre-fix the aux legs ran 1.6–6× hotter than the
taps8 reference (echo l1.0 RMS 0.76, SR g0.9 RMS 1.24, slow-feature RMS
0.82 vs taps8 RMS 0.20), confounding architecture with operating
point. The applied gain is recorded per row as `drive_scale`.

### Pre-registered verdicts

- **SUCCESS path 1**: aux MC ≥ 0.85 × taps8 MC AND edge cost ≤ 1.50 × ref.
- **SUCCESS path 2**: aux MC ≥ 1.50 × scalar MC (margin ≫ cost).
- **KILL (i) tap subspace**: per-column ridge R² of aux states predicted
  from the taps8 delay bank averaged ≥ 0.90.
- **KILL (ii) void/dead**: aux state PR < 1.05 OR aux MC < 0.20.
- **KILL (iii) no gain over scalar**: aux+scalar MC − scalar MC <
  0.05.

Tap subspace KILL takes precedence over SUCCESS path 1: a literal delay
line that matches taps8 within tolerance is still an expensive taps8.

### Local CPU verdict (n=300 train samples, 1 data-seed, 2 aux-seeds,
drive-power parity ON)

| Flavor | aux MC | taps8 MC | scalar MC | edge factor | tap-subspace R² | verdict |
|---|---:|---:|---:|---:|---:|---:|
| echo_state (best: aux_scalar leak=1.0 seed1) | 8.13 | 11.47 | 5.53 | 1.49 | 0.902 | **KILL** |
| shift_register (best: aux_scalar gain=0.9 seed1) | 11.40 | 11.47 | 5.53 | 1.17 | 1.000 | **KILL** |
| slow_feature (full log-spread) | 6.40 | 11.47 | 5.53 | 1.04 | <0.9 | **KILL** |

The shift-register aux is the most informative: at gain=0.9 it
ALREADY satisfies SUCCESS path 1 (MC 11.40 ≥ 0.85 × 11.47 = 9.75, edge
factor 1.17 ≤ 1.50), but its R² against the taps bank is 1.000 — the
aux IS the taps bank, just stored more expensively. Echo-state with
leak=1.0 sits right at the R² threshold (0.902); echo-state with leak=0.3
sits at R²=0.76 well below the threshold but its MC ~5.2–5.7 doesn't
satisfy either SUCCESS path. The slow-feature bank at full log-spread
sits at MC 6.40 with an MC curve without a peak; beating scalar is
neither wide-margin nor cheap. (Parity-ON numbers; pre-parity the same
three KILLs fired at echo 7.37 / SR 10.12 / SF 6.85 — parity moved MCs
up slightly and the verdicts did not move.)

### Mechanism summary (per-flavor)

- **echo_state**: marginal MC vs scalar at small leak, large enough to
  be detectable. The diagnostic shows it as borderline tap-subspace.
- **shift_register**: a learnable gain can match taps8 (so the load the
  bank carries is REAL) — but the load is pure delay, so it isn't an
  aux core, it's a parameterised taps8 with extra cost.
- **slow_feature**: log-spread τ covers the canonical TW MC bands but
  contributes little peak MC; the decoupled bank can't synthesise the
  same cross-feature mixing that taps8+main core achieves.

### Fabric-integration decision (per spec)

All three flavors emit **KILL**. Per the spec, fabric integration is
"earned only by a SUCCESS verdict, never started speculatively." No
`DifferentialStage` surgery is in scope. The static taps8 delay bank
remains the main-core drive. The aux-core-drive plan is closed at the
twin level.

The local verdict is single-seed eval-only screening (per spec). The
Alliance GPU upgrade (full 3 aux-seeds × 3 data-seeds × leg × hparam,
~minutes per leg) is the closing measurement if the verdict needs to be
re-anchored, but the local evidence is consistent enough that KILL is
the expected outcome across seeds (the echo-state borderline at R²=0.90
is the only one Alliance could plausibly flip, and only by moving the
threshold).

### Files & artifacts

- `aux_core_probe.py` (~750 LOC, side-effect-free import): aux builders
  + runners, aux drive concat, edge counters, twin-level probe
  reusing `post_twin_gates` instruments, tap-subspace R² diagnostic,
  verdict function, CSV/JSON/TXT writers, `sweep` / `smoke` CLI
  subcommands (`--flavors`, `--legs`, `--aux-seeds`, `--data-seed`,
  `--max-corners`, `--dt`, `--order`).
- `audit_aux_core.py` (17 audits, all green locally): import-side-
  effect-free, locked thresholds, echo-state builder+runner
  determinism + spectral radius, shift-register EXACT chain
  asserted (x[t][i] == gain^i * hi * u[t-i], bit-exact at fixed
  seeds), slow-feature runner matches the per-step leaky-integrator
  formula, concat shapes, edge counters, reference harnesses
  (taps8 / scalar / scalar-ESN parity) + unknown-leg fail-fast,
  tap-subspace R² diagnostic on self / random / shift-register
  (KILL) / echo-state-leak-0.3 (below KILL), end-to-end smoke,
  CSV/JSON/TXT writers, CLI registration with `--dt` / `--order`
  guards, verdict determinism on synthetic row lists (SUCCESS path 1,
  KILL by void, KILL by no gain over scalar, INSUFFICIENT_DATA),
  methodology guards (drive-power parity RMS, aux_scalar width 9 +
  u·input_scale column, ref_total on custom dims, flavor/leg/seed
  fail-fast, washout threading, dead-block 0.0 scale,
  INSUFFICIENT NaN edge factor).
- Audit fixes applied 2026-09-18: drive-power parity normalization
  (aux legs ran 1.6–6× hotter than the taps8 ref pre-fix); CUDA device
  bug in the tap-subspace bias column; INSUFFICIENT_DATA edge factor
  0.0 → NaN; washout threaded into the verdict (was pinned to
  PROBE_WASHOUT); ref_total derived from live hidden_dim × n_taps;
  fail-fast validation of flavors/legs/data_seeds; removed dead code
  (unused nlc import, AUX_BUILDERS, TW_HOLDS_MC_ABOVE, prev_max shim,
  best_aux_meta tracking, return_states flag, unused
  input_scale/aux_states/harness params).
- Local CPU smoke took 0.5s for 1 corner / 1 flavor and ~3s for the
  full 20-row × 3-flavor sweep on the same machine.

---

## Phase 11: aux-core-drive-2 follow-up + concat-readout check (2026-09-18)

**Plan:** `aux-core-drive-2` (follow-up; `aux-core-drive` stays closed).
**Code:** extended `aux_core_probe.py` (~1870 LOC) + `audit_aux_core.py`
(23 audits, all green locally). **Status:** implemented and audited on
Windows CPU; verdicts unchanged (all KILL); fabric integration still
not earned.

### Why a follow-up was opened

The Phase-10 KILL drew a challenge: the shift-register aux scored MC
11.40 vs taps8 11.47 — a near-match — and died *only* on the
tap-subspace rule (i) (R² = 1.000), while satisfying both SUCCESS paths
on the numbers (match-within-tolerance at edge factor 1.17;
1.5×-scalar margin). Four gaps were named: (a) nothing was ever actually
*learned* (all aux params frozen at hand-picked values); (b) the
spec-permitted concat readout `[main; aux]` was never run; (c) main
fan-in was pinned dense (25×8 = 200), baking in the loss against any
ring-like aux; (d) aux never got its own objective distinct from main's.
Votes taken: pursue D1–D4 scope *and* keep rule (i) as an unconditional
KILL. Both hold in what follows.

### What was built (D1–D4)

- **D1 — concat readout leg** (`aux_concat`): same drive as aux_only,
  ridge over `[main_states(25); aux_states(8)]` = 33 features, reported
  as `ridge_concat_*` on every row plus a per-row `tap_r2` copy-flag.
  Informational only by design (never a verdict path).
- **D2 — sparse main fan-out** (`--fanout-kinds dense sparse_random`,
  k ∈ {3, 6} via the Gate-3 `sparse_random_W_in` builder,
  radius-restored): SR at k=3 totals 7 + 1 + 24 = **32 edges vs 200**.
- **D3 — compressed-bank reference** (`tapsvd`, locked k=4 = tapsVD4):
  SVD top-4 PCs of the taps bank (deterministic, sign-pinned) drive the
  main core at parity; 25×4 = 100 edges. Reference only, never
  verdict-ed.
- **D4 — residual-distilled aux** (`residual_distilled` flavor): offline
  Phase A builds a teacher from tap PCs *minus* the main+scalar state
  span (ridge/SVD only, eval-only-legal — the teacher never sees NARMA
  targets), then echo hyperparams are selected by teacher-R² instead of
  MC.
- **Verdict amendment** (additive; rule (i) untouched): refs gain
  tapsVD4; flavor edge factors computed against
  `min(full taps8, compressed)` budget, the compressed budget applying
  only if tapsVD4 matches taps8 within the 0.85 ratio.

### Audit round 2 (2026-09-18 evening)

The merged tree did not import at all (`ALL_FLAVORS` referenced
`D4_FLAVOR` before its definition — the D1–D4 code had never executed).
Fixed plus: D4 entries in the runner/edge-counter dicts (else KeyError
on every D4 row); verdict now considers `aux_concat` legs (concat-only
runs previously returned INSUFFICIENT_DATA); fail-fast on unknown
fanout kinds (previously silent dense-under-sparse-label); CUDA device
bugs in the teacher/R² helpers; `residual_teacher_build` device +
dead-param + corner-threading fixes; removed dead code
(`TAPSVD_K_GRID`, `tapsvd_reconstruct`, `fixed_stream_seed`, `_dev_of`,
`RowStub2`). Two audit expectations corrected against measured reality
(see next section). Final: **23/23 audits green** (AUDIT18–22 cover
D1–D4 + mixed budgets; AUDIT23 covers the fix regressions).

### D1 concat readout: in-sample promise, held-out evaporation

In-sample (fit + score on the train stream — the probe's standard
methodology), reading `[main; aux]` helps everywhere:

| Flavor | Ridge (main → concat) | MC (main → concat) |
|---|---|---|
| echo l1.0 | 0.645 → 0.603 (+0.042) | 7.13 → 8.63 (+1.5) |
| SR g0.9 | 0.528 → 0.453 (+0.074) | 11.38 → **12.39** (+1.0) |
| slowfeat | 0.587 → 0.572 (+0.015) | 6.40 → 7.11 (+0.7) |

SR-concat MC 12.39 is the first aux number anywhere to exceed
taps8-main (11.47). Held-out (fit train stream, score fresh stream),
the gains scatter ±0.05 around zero across seeds (SR mean ≈ +0.01,
SF mean ≈ −0.03; taps8 held-out ridge 0.625 stays best). Conclusion:
the in-sample concat gain is 8 extra readout degrees of freedom
overfitting ~100 post-washout samples, not aux information the main
lacks. No hidden SUCCESS in D1; the informational-only design stands.

### Threshold fragility found (not fixed — pre-registered rule stands)

On the autocorrelated NARMA stream, *every* echo/SF row lands at tap-R²
0.86–0.92 (measured: echo l0.3 s0 0.920, l1.0 s0 0.901, l0.3 s1 0.861,
SF 0.865; white-noise input gives 0.76). The 0.90 KILL line cuts
straight through the echo family, so echo verdicts can flip by seed.
This does not change any verdict (rows near the line still lose on MC),
but an Alliance re-anchor should report per-seed R², not just the
winner's.

### Remaining numbers

- tapsVD4 = MC 9.21 vs taps8 11.47 (0.80 < 0.85 gate) → compressed
  budget does **not** apply; the full 200-edge reference stands. The
  bank is not compressible to 4 PCs at preserved memory.
- D4 teacher-selected echo lands at tap-R² ~0.90 → KILL. Distilling to
  the main-missing residual did not produce a non-copy aux with
  taps8-grade MC — what main misses turns out to be more delays.
- D2-SR at 32 edges dies on rule (i) exactly as pre-registered
  (confirmatory, fan-in-independent copy-ness).

### Standing

All flavors, all legs: **KILL**. The delay bank stays. The productive
output of the whole aux-core episode is negative knowledge with sharp
edges: (1) the bank's memory load is real (an independent structure
recovered 99% of it); (2) it is incompressible below ~8 taps at
preserved MC; (3) miniature learnt replacements either recapitulate it
or underperform it; (4) direct aux readout adds fittable parameters,
not transferable memory. Next memory work, if any, should target
making the bank cheaper (fewer taps, sparser fan-out — both unrun),
not replacing it; and any future probe should score ridge/MC on a
held-out stream, since every absolute MC in this document (taps8's
11.47 included) is fit-and-scored in-sample and therefore optimistic.
