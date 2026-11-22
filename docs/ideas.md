# IDEAS — Analog Memory Operators on a Reconfigurable OTA Fabric

**Created:** 2026-09-18. Living document: update statuses as legs run and citations get read.
**Scope:** every research idea discussed across the NARMA aux-core, niche-audit, analog-native-features,
critique-verification, and math-operator rounds — with prior-art boundaries stated as precisely as the
evidence allows, so that each gap reads as "paper A only does X."
**Citation honesty tags:** `[V]` = existence + claim verified by web search this program;
`[C]` = canonical result, details to confirm before citing (year/venue/numbers);
`[U]` = unverified, from training knowledge or another agent's memory — check before writing.

**Thesis (one paragraph):** Reconfigurable analog fabrics exist (FPAA). Trained physical networks exist (PAT).
Optimal memory operators exist (gamma/LDN/HiPPO). What does not exist is the stack connecting them: a
compiler that takes a trained dynamical operator, realizes it under analog cost constraints (locality,
fan-in, quantized gm settings, state dynamic range), and maps it onto a mismatch-tolerant reconfigurable
OTA fabric — with per-chip closed-loop retuning as the final layer. Demonstration vehicle: fully-analog
neural equalization with no data-converter tax.

---

## BACKGROUND & CONTEXT — where this program comes from and where it is going

**What we are building:** an analog RNN — a continuous-time recurrent neural network realized as a
physical OTA fabric (KirchhoffNet/KNet: 5-stage analog network, Heun-stepped ODE, device-realistic
cells with tanh/compliance-gating/rail-clip/gm-Isat bounds, mismatch-aware twin simulator). The network
must *generate and hold its own memory* as recurrent dynamics, not ride on an external shift register.

**Why NARMA (the testbed, not the destination):** NARMA-10 is the memory benchmark. Task needs: ~12 units
of memory capacity; the sufficient statistics include the lag-9 product u(n−9)·u(n), which sits *beyond*
the 8-tap input delay bank (taps8 covers delays 0–7) — so genuine recurrent memory is measurable, not
hand-supplied. Established facts from the campaign (see `docs/NARMA Fabric Memory Progress Summary.md`):
the static 200-edge taps8 bank works (MC ~9–13 with it, ~3–5 without); the core's genuine recurrent memory
is ~3–5 units (scalar-drive harness, both maps, both campaigns); frozen aux cores (echo/shift-register/
slow-feature/distilled, Phases 10/11) all KILL — including a 32-edge learned shift register killed only
by the old copy rule (since amended to cost-conditional); concat-readout gains evaporate held-out, so
held-out scoring is primary; banded/multi-scale leaks (Phase 7/8) move beyond-tap memory 0.15→0.25–0.29;
tapsVD4 (MC 9.21 at half edges, k locked) warns delay operators resist naive compression. Method inherited:
pre-registered gates, KILL rules, cost-matched baselines (E0), bounded pilots, twin-based evaluation
(flow-twin harness with MC/PR/ridge diagnostics).

**Where it is going (the destination):** the analog RNN, once it holds memory cheaply and robustly, goes
into a **closed-loop system in the control path of an RX equalizer — tuning CTLE and/or FFE and/or DFE**
(P2 control-path pitch, PART 12). The receiver data path stays conventional hardwired analog; the learned
network sits above it, optimizing measured performance (eye/BER proxy) via BO at link-up + slow tracking,
absorbing PVT and drift by closed-loop optimization. Prior control-path-adjacent assets already in repo:
`kn_bayes_opt.py` (18-dim BO with seed-trial mechanism — the link-up optimizer skill), CTLE
generative-distillation line (MoE student, PlainMLP teacher/control, DAgger, BO-driven via
`mlp_bayes_opt.py`, trial_0011/0024 checkpoints), and `spice_failure_check.py` (true-SPICE failure-rate
harness with MoE/KNet-distill loaders — the bridge from learned models to silicon-grounded validation).
NARMA proves the memory; the RX control loop is what the memory is for.

---

## PART 0 — STATUS LEGEND (for every idea entry)

- `IDEA` — proposed, untested.
- `PRIOR ART` — occupied; cite, don't attack.
- `GAP (standing)` — no counterexample surfaced in searches; falsifiable, keep looking.
- `GAP (confirmed-absent)` — targeted searches returned the neighbors but not the thing.
- `TESTABLE NOW` — runs in the existing twin/simulator with no new cells.
- `NEEDS-HW` — needs cells, tapeout, or lab measurement.
- `CLOSED` — killed by your own data or by prior art; kept so nobody reopens it.

---

## PART 1 — FABRIC & COMPILER STACK

### 1.1 Overcomplete quantized-OTA reconfigurable fabric (`IDEA` / `TESTABLE NOW` in twin)

**Summary.** One physical chip, deliberately larger than any single design: N OTA integrator stages
(N ~ hundreds), a switch matrix, capacitor banks, comparator/slicer resources, per-cell digitally
quantized bias currents (e.g., 6-bit DAC codes, not floating-gate continuous programming). Any single
operator compiles to a subset of cells; the rest power-gate. Report "34/200 OTAs active" the way digital
reports gate count.

**Prior art — what exists, and only what it does:**
- Hasler RASP line (Georgia Tech): RASP 2.8 is a 32-CAB floating-gate FPAA with >50,000 programmable
  FG elements `[V]` (Basu CICC 2008; Schlottmann 2012 TVLSI). RASP 3.0 SoC FPAA adds on-chip MSP430,
  DACs/ADCs, digital CLBs `[U]`. These fabrics do hand-designed signal processing (filters, VMM
  classifiers, AM receivers) — they do **not** compile gradient-trained dynamical operators.
- Anadigm/Okika AN231E04 (now FlexAnalog): 2×2 CABs, switched-capacitor, **dynamically reconfigurable**
  with shadow SRAM and single-clock-cycle Shadow→Config transfer plus partial reconfiguration `[V]`
  (datasheet + User Manual). It proves runtime reconfiguration hardware is feasible — but its toolchain
  (AnadigmDesigner2) maps human-picked CAM blocks, not trained operators.
- BrainScaleS-2 (Heidelberg): analog neurons with per-neuron calibration (24 parameters, Pehle 2022)
  `[V]`; fixed neuron model, not a general OTA fabric.
- TinyTapeout analog (sky130A, ihp-sg13g2, gf180 open PDKs; TT05 ran "neural network on chip") `[V]`:
  proves a cheap open tapeout path exists — it is a *vehicle*, not prior art against you.

**The gap `GAP (standing)`.** A fabric whose design point is *discrete-quantized biasing + trained
dynamics*: neither floating-gate-continuous (Hasler) nor fixed-model (BrainScaleS). No studied,
published characterization of "quantized-bias OTA fabric as compilation target for learned memory
operators" surfaced.

### 1.2 The 4-layer compiler (`IDEA` — the thesis engine)

**Summary.** (1) Frontend: operator spec in Python/PyTorch (matrices, cell nonlinearities, constraints)
— no new language. (2) Middle: realization search under an analog cost functional (locality, fan-in,
quantized settings, state dynamic range, mismatch sensitivity) — **this layer is the thesis**.
(3) Backend: place/route onto CABs, emit bias codes + switch settings (plumbing; AnadigmDesigner2 and
the RASP Simulink→SPICE→P&R flow prove buildability `[V]`). (4) Runtime: per-chip probe-fit-nudge
retuning + drift tracking.

**Prior art — only what each does:**
- AnadigmDesigner2 / RASP Simulink tool (Schlottmann 2012 TVLSI) `[V]`: compile *human-designed blocks*.
  Neither searches realizations of a trained operator.
- XLA/HLO, TVM, Vitis-AI (digital ML compilers) `[C]`: the correct structural analogy (cost-model +
  mapping search + codegen), for the wrong substrate. Cite as analogy, not lineage.
- Verilog-A/Verilog-AMS `[C]`: modeling/simulation languages, not synthesis. A new analog HDL would
  give a nicer simulator input and zero mapping capability — **do not build one** (`CLOSED` as a direction).

**The gap `GAP (standing)`.** No toolchain takes a *trained continuous-time memory operator* through
*realization search under analog cost* onto a *reconfigurable fabric* with *per-chip retuning*.
Each clause has prior art; the stack does not.

### 1.3 CABs (background, not an idea)

**CAB = Configurable Analog Block**: the repeating tile of an FPAA, i.e., the analog equivalent of an
FPGA's CLB — a cluster of OTAs/opamps, capacitors, switches, local routing set by configuration bits.
"Place/route onto CABs" = FPGA P&R where timing closure is pole locations and stability, not clock slack.

---

## PART 2 — MEMORY OPERATORS (MATH SIDE)

### 2.1 Modal / biquad realization as compilation target (`IDEA` / `TESTABLE NOW`)

**Summary.** Eigen-decompose trained A; implement each complex pole pair as one OTA-C biquad (the
canonical analog block), each real pole as one integrator. Converts "arbitrary dense A" into a shopping
list of standard blocks; composes with per-mode AGC, companding, mismatch analysis.

**Prior art:** modal decomposition is textbook control `[C]`; OTA-C biquads are textbook circuits `[C]`.
*Using modal form as the compilation target for gradient-trained RNN memory* did not surface
`GAP (standing)`.

### 2.2 Lattice-form learned memory (`IDEA` / `TESTABLE NOW`)

**Summary.** Parameterize memory as a lattice (identical stages, reflection coefficients |k|<1):
best-case matching, quantize-friendly coefficients, stability by construction (answers stability
certification for free).

**Prior art — only what each does:**
- Burg / gradient-adaptive lattice (GAL) filters `[C]`: classical DSP, adaptive-filter training, not
  gradient-trained neural memory, not analog fabric.
- Fejzo/Lev-Ari 1997 gradient-adaptive Laguerre lattice `[U]`: the Laguerre+lattice math combo exists
  in DSP — neural/fabric version does not.
- Harris 1999 (adaptive Laguerre filter in analog VLSI, gradient-based) `[V]`: closest ancestor —
  4-tap scale, adaptive-filter (system-ID) training, fixed structure. Differentiate on programmability +
  learned-operator training.
- Cauwenberghs 1996 "Adaptation, Learning and Storage in Analog VLSI" `[V]` (search-surfaced):
  adaptation compensating analog implementation inaccuracies — read it; it may own part of the
  calibrate-vs-absorb framing.

**The gap `GAP (standing)`.** Lattice-structured neural memory trained by BPTT and mapped to a
programmable analog fabric.

### 2.3 Givens-structured orthogonal memory (`IDEA` / `TESTABLE NOW`)

**Summary.** Memory as products of learned Givens rotations: norm-preserving (no vanishing), each
rotation = one differential pair, quantization = angle quantization, common-mode mismatch cancels
across the pair (mismatch as offense, not defense).

**Prior art — only what each does:**
- Arjovsky uRNN; Jing EUNN (271 cites); Helfrich; Lezcano-Casado (339 cites); Foucault 2024 quantized
  ORNN `[V]` (all surfaced in search): digital orthogonal parameterization + quantization — none touch
  analog implementation.
- Bausch 2020 parametrized quantum circuits as unitary RNNs `[V]`: different substrate entirely.

**The gap `GAP (confirmed-absent)`.** Targeted search returned the full digital lineage and no analog
implementation.

### 2.4 Input-gated LDN / liquid timescales (`IDEA` / half-built — your dyn-leak spike)

**Summary.** LDN coupling matrix with input-gated leaks ("liquid Legendre memory"): structured memory +
data-dependent timescales. Your `--dynamic-leak` mechanism + audit already built the circuit half.

**Prior art — only what each does:**
- Hasani LTCs/NCPs (MIT 2020–22) `[C]`: input-dependent τ, *unstructured* — no LDN basis, no fabric.
- de Vries & Principe gamma memory (1992) `[C]`: trainable μ interpolates delay↔dispersive — single
  global depth parameter, not per-state gating, not LDN-structured.
- Pure LDN/LMU (Voelker 2019, 381 cites) `[V]`: fixed timescales.

**The gap `GAP (standing)`.** Neither liquid-unstructured nor fixed-structured literature owns gated LDN.
Needs GPU verdicts + task metrics (your deferred spike trains).

### 2.5 Selective / learned-measure HiPPO (`IDEA` / `TESTABLE NOW` in twin)

**Summary.** Mamba-style input-dependent Δ (learned time-warping of the HiPPO measure) compiled to
modulated bias currents — with honest wire-cost accounting for one modulation path per timescale.

**Prior art — only what each does:** Mamba/S4D/S5 learn Δ digitally, where per-step updates are free
`[V]` (S5 paper verified). Analog modulation paths cost wires + DACs — that cost analysis is itself
unpublished `GAP (standing)`. Position with Mamba explicitly; same math, different substrate.

### 2.6 Koopman lift (`IDEA` / time-boxed)

**Summary.** Learn observables linearizing the channel; implement *linear* Koopman evolution in fabric
(precise, low-power, mismatch-tolerant), nonlinear lifting only at input/output edges.

**Prior art:** Koopman autoencoders/DMD exist digitally `[C]`; analog Koopman equalizer did not surface
`GAP (standing)`. Gated on finding good finite observables — time-box; your CTLE channel models may
suggest the lift.

### 2.7 Multi-window mixture multi-θ LDN (`IDEA` / fold into E1)

Block-diagonal A, one window θ per block, learned mixing; analog cost is one capacitor size per block
(free). Math exists (S5 block-diagonal HiPPO-N init `[V]`); analog+quantized+mismatch version does not.
Cheap twin leg — do not standalone.

### 2.8 Piecewise / gain-scheduled "gearbox" memory (`IDEA` / `TESTABLE NOW`)

**Summary.** Regime-switched operators: long-window/low-leak when input quiet, short-window/fast when
loud (comparator + switched bias codes *is* a piecewise dynamical system); state-energy-triggered
graceful forgetting as a dynamical vii answer.

**Prior art:** gain scheduling / switched systems + dwell-time stability (Liberzon textbook) `[C]`:
classical control, not learned memory operators. Learned regime boundaries + per-regime operators via
STE: `GAP (standing)`. Verify against "switched neural networks / mixture of SSMs" ML literature
before writing.

### 2.9 Volterra-readout co-design (`IDEA` / methods-note scale)

LDN + polynomial readout *is* a Volterra series; jointly learn basis order + readout order against task
sufficient statistics (NARMA needs order-2 at specific lags). Gives the E2-arm-B overkill question a
principled stop criterion. Small, sharp.

### 2.10 Basis choice: Legendre vs Laguerre vs gamma (`TESTABLE NOW` — F5/E1 leg)

**Summary.** Learned-pole Laguerre (one repeated pole → best matching) at d=6–12 vs Legendre-LDN on the
*same* fabric under mismatch sweep. Prediction: Laguerre degrades gracefully (single error source).

**Prior art — only what each does:**
- Juan/Harris/Principe 1997–99 `[V]`: *fabricated* adaptive gamma AND Laguerre (4-tap, system-ID task);
  quote: all-pass-cascade Laguerre "will potentially outperform the gamma." Not trained RNN memory, not
  programmable fabric, not equalization, not at LDN orders.
- Multi-scale gamma (exponentially spread τ) `[C]`: is your Phase-7 banded leaks, ~30 years prior — cite.
- Focused gamma network (memory plane → nonlinear mapper) `[C]`: is your taps8+main-core architecture — cite.

**The gap `GAP (standing)`.** Same-fabric, mismatch-swept, learned-pole basis comparison at memory orders.

### 2.11 Exotic / deferred math (future-work lines, not scheduled)

- **M9 integrate-and-reset states:** graded memory + comparator resets; thin as *learned* memory.
- **M10 free-Wiener SDE states:** thermal noise *is* the Wiener process digital must simulate; run only if M1–M4 stall.
- **M11 parametric pumping:** refresh states by 2×-resonance pumping instead of gain power; absent from ML memory — one line in future work.
- **Fractance (F4, §3.4):** listed with features but is equally a math-side operator; see §3.4.

---

## PART 3 — ANALOG-NATIVE FEATURES (PHYSICS DOES THE WORK)

### 3.1 F1 — Time-multiplexed delay-loop virtual nodes, trained, fully-analog, at rate
(`IDEA` / needs analog delay model)

One nonlinear node + delay loop = N virtual nodes; no dense A exists. **Prior art — only what each does:**
photonic reservoirs (Borghi 2021 microring `[V]`; Tesfaye 2026; Jeon 2025 Nature single→10 virtual nodes
`[V]`) — photonics, fixed physics; electronic versions are toy/mixed: Appeltant thesis (analog
Mackey-Glass node, *digitally implemented* delay) `[V]`; Soriano/Ortín (analog node + digital delay)
`[V]`; Zhao 2016 (analog delay node for spikes) `[V]`; Arduino Mackey-Glass demos `[V]`. All use *fixed*
Mackey-Glass. **Gap `GAP (confirmed-absent)`:** fully-analog integrated delay + *learned* delay operator
+ real signal-rate task. Pairs with aux-core as the two structural answers to one cost question.
Risk: delay thermal drift moves virtual-node alignment — couples to drift retuning (makes vi a feature).

### 3.2 F2 — Companding / log-domain state elements (`IDEA` / 1–2 months, vii answer)

Log-domain integrators compress large signals as device physics; state-SNR balancing becomes a circuit
property. **Prior art:** Frey 2001 syllabic companding `[V]`; Kolios 2010 maximized-DR log-domain `[V]`;
Serra-Graells 2004 1-V log-companding `[V]`; Kircay 2006 allpass log-domain `[V]` — all *filters*.
**Gap `GAP (confirmed-absent)`:** polynomial (LDN/HiPPO) memory from companding integrators; no
state-SNR-vs-order data for companding reservoirs. Your subthreshold OTAs are nearly there already.

### 3.3 F3 — Per-state AGC loops = analog LayerNorm (`IDEA` / vii answer)

Slow AGC (peak detector + VGA, textbook SerDes blocks) per state/order-group regulating equal RMS —
LayerNorm as continuous-time feedback; AGC control voltages double as free slow statistic features.
**Prior art:** AGC everywhere (Jung 2022 neural-recording AGC AFE `[V]`; SerDes loops) — never as
*normalization inside a recurrent network* `GAP (standing)`. Formalize loop/state timescale separation +
stability; report cost (one detector+VGA per group).

### 3.4 F4 — Fractional-order (fractance) memory (`IDEA` / high-risk parallel thread)

Fractional integrators (order α∈(0,1)) = power-law kernels, long memory in few states; analog cost is a
passive RC ladder, while digital pays O(T) history buffers — the asymmetry *is* the paper. Channel
symmetry argument: skin effect is ~half-order, so the equalizer kernel should be too.
**Prior art — only what each does:** Yao 2024 fractional-integer ESN (26 cites) `[V]`; Dong 2024
fractional fuzzy ESN `[V]`; SP-FIO-ESN 2026 `[V]` — all discrete-time digital algorithms. Analog
fractance circuits exist, never as RNN reservoirs. **Gap `GAP (confirmed-absent)`.**

### 3.5 F5 — Learned-pole Laguerre (see §2.10; listed once, executed once)

### 3.6 F6 — Mismatch-as-projection + cross-die single readout (`IDEA` / Monte-Carlo harness)

Don't program W: measure the mismatch-realized matrix once per die, train biases + readout only
(ESN theory says fixed random projections suffice). **Gap:** (a) MC/PR of mismatch vs digital-drawn
matrices; (b) per-die readout (= calibration-as-training, merges ii+vi); (c) **one readout across N
dies** — kills programming + calibration + drift-tracking in one measurement `GAP (standing)`.
Physical-reservoir concept exists generally (photonics); OTA-fabric + measured-frozen + cross-die is the
slice.

### 3.7 F7 — Analog neural DFE, slicer-in-the-loop, no converters (the application paper)

DFE is already recurrent; replace feedback with a small analog RNN on comparator decisions (1-bit, not
ADCs); train through the slicer (STE) against post-slicer BER, not MSE.
**Prior art — only what each does:** Kim 2023 neural equalizer "inspired by forward DFE" `[V]`; Zhang
2023 nonlinear-DFE DNN `[V]`; Roshan-Zamir 2017 PAM4 FIR+IIR DFE `[V]` — all digital/ADC+DSP; 112G
DSP-SerDes dominates; ML-for-SerDes is modeling/simulation. **Gap `GAP (confirmed-absent)`:** no-converter
fully-analog neural DFE. Own the sentence: everyone else pays ADC/DSP power (the dominant receiver term);
analog-in/analog-out sidesteps it structurally. Behavioral models can start now.

### 3.8 F8 — Cellular / locally-coupled fabric memory (deferred)

Chua–Yang 1988 CNN + chips `[C]`: old paradigm; *learned space-time memory with LDN readout vs dense
LDN Pareto* is not. Wins wire cost, loses compactness — the curve is the contribution. Promote on evidence.

### 3.9 F9 — Passive SC / bucket-brigade delay bank (cheapest experiment; do first)

Same history in switches+caps, no active-OTA power/mismatch in the memory path; OTAs only at mixer/readout.
SC/BBD = textbook `[C]`; *learned sparse mixer on passive bank for trained analog RNN equalizer* is the
integration. Quantifies the Phase-11 "make the bank cheaper" fallback. Days–weeks in twin.

### 3.10 F10 — Bilinear / second-order states via Gilbert cells (deferred)

NARMA-class tasks need products; Gilbert multipliers make them first-class. Math lineage: bilinear
systems / second-order RNNs (Giles) `[C]`; analog trained bilinear reservoirs thin. Gives arm-B a76849 architecture answer.

### 3.11 F11 — Event-driven / level-crossing updates (deferred; weak SerDes fit)

Energy-proportional-to-activity memory; level-crossing ADC + event SNN lineage `[C]`. Note for sensor/
always-on story; not SerDes.

### 3.12 F12 — Learning rule for dissipative dynamics (position, don't compete)

HEB (Lopez-Pastor/Marquardt 2023 PRX) `[V]`; RHEL (Pourcel/Ernoult NeurIPS 2025 oral) `[V]`; LEP/2026
unification `[V]` — exact physical gradients for *non-dissipative Hamiltonian* systems; dissipative case
explicitly open (their App. A.3.1). OTA-C is dissipative. Your calibrated twin = physics-aware training,
the engineering answer — cite the crowd, out-measure them, don't out-theory them.

---

## PART 4 — TRAINING & ROBUSTNESS

### 4.1 PAT differentiation (posture, not a leg)

Wright et al. Nature 2022 (1071+ cites) + instructional repo `[V]`: hybrid in-situ/in-silico backprop
through measured response. **PAT absorbs one instance's quirks; the compiler must generalize across
instances, closing only residuals per chip.** Read the repo, not just the paper. (Also: Cauwenberghs
1996 analog-VLSI adaptation-compensation `[V]` may pre-own part of this framing — check.)

### 4.2 PVT-injected BPTT through the device-physics twin (pipeline leg, 2–3 months)

Sample (Δgm, ΔIsat, ΔVth) per cell from process corners during training. Neighbors: IBM offset-corrected
in-memory training (Rasch/Carta) `[V]`; Wang 2022 variation injection `[V]`; VANT 2025 `[V]`; BrainScaleS-2
in-the-loop (Cramer 2022) `[V]` — all feedforward/inference-flavored. **Gap:** PVT-aware training of
*continuous-time recurrent memory operators* `GAP (standing)`. Never claim "first mismatch-aware" —
claim the dynamical-operator scope.

### 4.3 Aux-local training objectives (E2; bounded pilot)

Window-12 reconstruction (taps8 covers 0–7; NARMA-10 needs u(n−9) — size past the bank, not at it) +
product-term arm B (the overkill experiment). End-to-end-loss worry is backwards (gradients are zero for
duplicated features; D4 says the residual is more delays anyway). Frozen-aux screens were structurally
unwinnable (bank = optimal frozen memory) — training is the only informative test; honor E0-gating via
bounded pilot + pre-registered promotion gates vs the *cost-matched* bank.

### 4.4 Slicer-STE + constellation-aware training (C2; sub-task of F7)

Digital trains MSE; truth is post-slicer BER. STE through slicer with analog dynamics: thin — don't let
F7 forget it.

### 4.5 Stability certification via structure (C3; mostly free)

Lattice |k|<1 (M2), port-Hamiltonian/passive scaffolding, Givens norm-preservation (M3) each preempt the
reviewer's stability question differently. Prefer structural certificates over post-hoc analysis.

---

## PART 5 — CALIBRATION, DRIFT, CROSS-DIE

- **Per-chip probe-fit-nudge runtime** (compiler layer 4): chirp probe → fit realized response → correct
  few LSBs → run → re-probe hourly. Hardware loop proven feasible by AN231E04 shadow-SRAM mechanism `[V]`.
- **Pole-drift closed loop** (niche-vi slice): measure realized memory function (impulse response fit),
  re-tune leaks/gm to restore operator. FG re-programmability `[V]` + in-situ re-absorption `[V]` prove
  pieces; pole-specific loop on a fabric not found — `GAP (standing, unconfirmed — verify)`.
- **Cross-die single readout** (F6c): the money measurement; also the calibration-free deployment story.
- **Hybrid framing (sharpest positioning candidate):** RASP calibrates (measure→compensate per device
  `[V]`); PAT absorbs (train through device `[V]`); you train against a *statistical device model* and let
  quantized per-chip re-tuning close residuals. `GAP (standing)` — nobody owns the hybrid.

---

## PART 6 — BENCHMARKS & METHODOLOGY (service contributions with teeth)

- **E0 cost-matched bank baselines (UNRUN — do first):** tapsVD-k k=2..6, taps4/6, sparse fan-out taps8;
  Pareto MC-vs-edges `R(edges)`; every aux judged against the frontier, never taps8-200. Eval-only,
  ~1 GPU-hour. Note the internal tension: tapsVD4 (MC 9.21 vs 11.47, k locked) suggests delay operators
  resist naive compression — the pruning dividend is *unproven in your own lab*; E0/E3 before sparsity
  novelty claims.
- **Held-out scoring as primary:** fit-on-train / score-on-fresh-stream (own 200-sample washout);
  per-seed tap-R² (echo family 0.86–0.92 flips by seed). Phase 11 proved in-sample gains evaporate.
- **Rule (i) amended (locked):** copy at ≥1× cost-matched-bank edges = KILL; copy below with held-out
  parity = PASS.
- **B1 multi-input bench:** adding/multiplication tasks are inherently 2-input `[C]`; cross-NARMA is one
  generator line; MIMO channel models are the in-domain version; MC generalizes per-channel. "No
  benchmarks" is wrong — unmeasured-on-analog-fabric is the true gap.
- **B2 NARMA-C robustness suite:** ImageNet-C analog for RNNs — pinned seeds, PVT shifts, washout
  protocol. Nobody has it; everyone would cite it.
- **Phase-12 leg structure (E0→E6)** retained from prior plan: E0 baselines → E1 analytical (LDN/gamma/
  Laguerre/ms-gamma — the related-work anchor with in-harness numbers) → E2 trained pilot → E3 width
  knee → E4 prune-and-realize (≤0.5× frontier edges, ≥0.85× MC under mismatch = the paper figure) →
  E5 M=2 → E6 fabric integration decision.

---

## PART 7 — APPLICATION: ANALOG NEURAL EQUALIZATION

F7 (§3.7) + CTLE program + converter-tax positioning (§3.7 last paragraph). Supporting slices: D3
mixed-signal hybrid (digital taps for linear post-cursor where digital is exact, analog reservoir for
nonlinear residual — product architecture, thin novelty, keep as industry slide); M6 Koopman where
channel physics suggests observables; skin-effect/fractance symmetry (F4) as the ambitious pairing.

---

## PART 8 — HARDWARE PATH

- **D1 TinyTapeout analog test chip** (sky130A/ihp-sg13g2/gf180 `[V]`): 1–2 mm² (OTA array + SC bank +
  readout); schedule when Tier-S/A converge — taping the wrong circuit early is the failure mode.
- **F9 passive bank** doubles as the cheapest tapeout-friendly memory.
- **D2 quantized biasing** is a design point to characterize in twin first (STE vs annealing; 4/6/8-bit
  trade) — the DAC resolution answer gates the fabric spec.

---

## PART 9 — CLOSED / DEPRIORITIZED (kept so nobody reopens them)

1. "First reconfigurable fabric" — FPAAs since the 90s (RASP, Anadigm) `[V]`.
2. "First mismatch-aware training" — IBM/BrainScaleS-2/PAT `[V]`.
3. "First LDN pruning" — S4D diagonal HiPPO + Mamba pruning `[V]`; plus your own tapsVD4 warning.
4. "RNN that learns its own memory" — gamma 1992 `[C]`.
5. "LDN on analog hardware" — Braindrop/Loihi LMU, Legendre-SNN Loihi-2 `[C/U]`.
6. New HDL for analog; general analog synthesizer; out-theory-ing HEB/RHEL.
7. Aux-core-drive Phases 10/11 verdicts stand: frozen echo/shift-register/slow-feature/distilled = KILL;
   scalar-drive core memory ~3–5 MC units; concat gains evaporate held-out. Reopen only via E2-trained,
   held-out, amended-rule protocol.

---

## PART 10 — RANKING (tiers; within-tier order matters)

- **S:** compiler middle layer (M1 first target) → F7 analog neural DFE → F5/Laguerre basis decision →
  M4 gated LDN.
- **A:** F2/F3 companding+AGC → M2 lattice → M3 Givens → quantized+PVT+prune pipeline → F1 delay-loop →
  F6 mismatch-projection/cross-die.
- **B:** B1/B2 benchmarks (continuous) → F9 passive bank (first) → M7 (fold in) → M8 note → C2 (F7
  sub-task) → M6 (time-boxed) → D1 (on convergence).
- **C:** F4 (risk thread) → F8/F10/M9 (on evidence) → F12 (position) → F11/M10/M11 (future work) → D3
  (industry slide).

---

## APPENDIX A — ANCHOR BIBLIOGRAPHY (verify details before citing)

1. Basu et al., RASP 2.8 (CICC 2008) + JSSC follow-up `[V core/U numbers]`.
2. Schlottmann et al., Simulink FPAA tool (TVLSI 2012) `[V]`.
3. Kim/Hasler, FG SoC FPAA calibration (TVLSI 2017) `[V paper/U numbers]`.
4. Wright et al., PAT (Nature 2022) + mcmahon-lab repo `[V]`.
5. Voelker thesis + LMU (NeurIPS 2019) `[V paper/U thesis-scaling]`.
6. Gu et al. S4D (2022); Smith et al. S5 (2022); Gupta DSS (2022) `[V]`.
7. Mullis & Roberts (1976, CAS-23 551–562); Moore (1981) `[V/U]`.
8. de Vries & Principe (1992); Juan/Harris/Principe (1997–99) `[V existence]`.
9. Du et al. (Nat. Comm. 2017); Moon (2019) `[V]`.
10. AN231E04 datasheet + User Manual, Okika Devices `[V]`.
11. Hasani LTC/NCP (2020–22); Arjovsky uRNN; Jing EUNN; Foucault QORNN 2024 `[V existence via search/C]`.
12. Lopez-Pastor/Marquardt HEB (PRX 2023); Pourcel/Ernoult RHEL (NeurIPS 2025) `[V]`.
13. Yao 2024 / Dong 2024 fractional ESN; Jeon 2025 Nature single-node RC; Kim 2023 neural EQ `[V]`.
14. Frey 2001; Kolios 2010; Jung 2022 AGC IC `[V]`.
15. Pehle 2022 / Cramer 2022 BrainScaleS-2; Rasch/Carta offset-corrected training `[V]`.

## APPENDIX B — GAP REGISTRY (one line each; "A only does X" form)

- G1 trained-dynamics compilation: AnadigmDesigner2/RASP tools only map human blocks.
- G2 quantized-bias fabric design point: Hasler continuous-FG; BrainScaleS fixed-model.
- G3 realization search under analog cost: XLA/TVM do it for digital; Mullis-Roberts for roundoff.
- G4 Givens-analog: Arjovsky–Foucault lineage all digital.
- G5 lattice-neural on fabric: Burg/GAL + Harris-1999 adaptive only.
- G6 gated LDN: LTC unstructured; LDN fixed-τ; gamma single-μ.
- G7 pole-drift closed loop: FG re-programmable; loop unconfirmed.
- G8 analog neural DFE: Kim/Zhang digital-only; SerDes is DSP-based.
- G9 fractance reservoir: Yao/Dong digital-only.
- G10 companding polynomial states: Frey/Kolios filters-only.
- G11 per-state AGC norm: Jung/SerDes never inside RNNs.
- G12 trained fully-analog delay-loop at rate: photonics fixed-physics; electronics toy/mixed-signal.
- G13 cross-die single readout: per-device calibration is the norm everywhere.
- G14 hybrid statistical-model + quantized retune: RASP calibrates; PAT absorbs; none hybridize.
- G15 NARMA-C / multi-input analog bench: ImageNet-C has no analog-RNN counterpart.
- G16 dissipative physical credit assignment: HEB/RHEL require Hamiltonian; open by their own appendix.
- G17 compiled-static + self-healing fabric: FPAA runs config always (switch fabric on critical path);
  SerDes adapts taps but never the operator; none compile static then background-heal.
- G18 design-for-surrogacy blocks: Afacan/Zhang fit surrogates to arbitrary circuits; none constrain the
  block design so that LUT/MLP surrogates are exact-by-construction.
- G19 feedforward analog reservoir EQ: industry DFE pays 1-UI loop timing (unroll/speculate/loop-break);
  no-loop feedforward analog memory sidesteps it structurally.
- G20 learned online control policy for production RX: offline BO link optimization exists (Kiguradze 2020
  eye-height BO; Wang 2022 COM+ML; Chen joint FFE/CTLE/DFE); production adaptation is LMS (Bazargani);
  DRL param optimization is design-time (2025 DRAM paper). No BO-link-up + neural-tracking closed loop
  demonstrated end-to-end with drift.

---

## PART 11 — POST-FPAA-REALITY-CHECK PIVOT (2026-09-18/19 round)

**Why FPAAs aren't neural equalizers — user's five reasons, verified:**
1. **Bandwidth/interconnect `[V]`:** FG FPAA max ~57 MHz through switch matrix, ~5 MHz LPF (Basu/Brink).
   SerDes needs 14–28 GHz. Three orders of magnitude. Switch-matrix routing is the killer — Nuniz 1994
   thesis documents high-frequency OTA-C FPAA difficulties. Hasler does not touch wireline speeds.
2. **Calibration/PVT `[V — covered statically, not online]`:** Kim/Hasler 2017 TVLSI calibrates VT0 maps
   at program time; FG programming itself is slow iterative tunneling+injection (Kim FG-programming TVLSI,
   100 cites; Rumberg 2021 continuous-time programming). No continuous in-operation healing.
3. **CMOS scaling (not covered by Hasler; industry answered):** inverter-based front-ends scale
   (Zheng 2018 56Gb/s 16nm, 87 cites; Im 2020 112Gb/s 7nm; Palermo lecture notes) — cells must be
   inverter-based; prototype on old nodes via TinyTapeout.
4. **Hardwired wins (inherent):** full-crossbar reconfigurability = switches + parasitics on the signal
   path. Resolution: compile-time flexibility, run-time static (switches set once, off critical path).
5. **Training/modeling (partially covered):** Hasler VMM+WTA classifiers (Schlottmann 2010; Ramakrishnan
   2012, 76 cites; Moreno 2023 on-chip-training cluster) — but no gradient-through-dynamics, no recurrent
   operators. Static-circuit surrogate/BO field mature (Afacan 2021 review, 199 cites; Zhang 2019 BO+NN,
   100 cites; Du 2021; Beaulieu 2023; Wang 2024) — dynamics training is the hole.
6. **Loop latency (the +1 reason; Hasler silent):** industry pays 1-UI DFE timing with unrolling (Kimura
   2014 28Gb/s: half-rate, first-tap unrolled, sign-sign LMS, replica bias, rotatable error latches),
   speculation (Beukema 2005), loop-break parallel chains (Kim 2024, −54% area vs LA-DFE), unclocked DFE
   (Petricli 2020). **A strictly-feedforward analog reservoir has no feedback loop — no 1-UI problem
   structurally.** Keep recurrence feedforward (input→memory→readout); any feedback you keep gets
   unrolled/speculated per industry practice.

**Pivot (locked):** reconfigurable at COMPILE time, STATIC at run time, SELF-HEALING in background.
Reconfigurability retained for (a) one chip serving many operators across designs, (b) spare cells for
repair. Signal path sees set-once switches = hardwired speed.

**Self-healing HOW (layered, slowest-first):**
- L0 replica biasing (always-on, free): Kimura-2014-style half-circuit replica generates PVT-tracking
  references for all cells.
- L1 SerDes-native slow LMS (always-on, kHz): sign-sign LMS on slicer error drives bias DAC codes —
  extends Kimura's tap adaptation to memory-operator biases; needs only 1-bit decisions, zero
  signal-path converters.
- L2 foreground probe-fit-nudge (link-up/periodic): reuse SerDes training patterns; fit realized impulse/
  memory response; adjust codes; swap via AN231E04-style shadow-SRAM single-cycle mechanism (no disruption).
- L3 spare-cell repair (rare): overcomplete fabric routes around dead/drifted cells — overcompleteness
  pays twice (flexibility + repair).
- L4 continuous local: per-state AGC loops (F3) as always-on gain healing.
- Two-timescale programming: FG coarse one-time trim + DAC fast adaptation (solves FG slowness).

**Mixed-signal split without converters-everywhere (rate-separation principle):** signal-rate analog
path; symbol-rate 1-bit decisions (comparators, not ADCs); kHz-rate digital adaptation (slow DACs).
This is exactly how production SerDes already splits (Kimura 2014: analog AFE + DFE, digital LMS/CDR).
AIMC ADC-bottleneck lessons apply at the readout only: partial-sum quantization (Kim 2022), where-to-cut
ADC resolution (Leugering), ADC-less hybrid (HCiM 2024) — keep accumulation charge-domain, convert once.
Precision lives at thresholds/offsets (digital trim); speed lives in analog.

**Design-for-surrogacy (user idea v, upgraded):** don't just fit surrogates to arbitrary blocks — constrain
blocks so LUT/MLP surrogates are exact-by-construction (smooth, low-dimensional bias→behavior maps).
Hierarchy: LUT (static tanh/compliance, DPD-style) → block-MLP (codes + operating point → gm/Isat/noise,
differentiable, lives inside compiler search) → twin ODE (dynamics) → SPICE (final verify). Makes the
search tractable (no SPICE in the loop) and the training accurate. `GAP (standing)` per G18.

**Updated one-line thesis:** a compiler that emits *static, hardwired-speed* analog memory circuits from
trained operators, with *background self-healing* (replica + LMS + probe-nudge + spares) and
*surrogate-friendly* blocks — demonstrated as a no-converter analog equalizer that dodges the 1-UI loop
by being feedforward.

---

## PART 12 — COHERENT PITCH LIST (2026-09-19 round; pitches, not fragments)

Shared facts all pitches assume: hardware static at run time; backend compiler converts intent (Python
operator spec) into static circuitry; physics-aware twin + surrogate hierarchy underneath; small digital
LMS block as the self-healing actuator. Structure: **1 platform + 2 applications + 1 methods pitch.**
Shared infrastructure (twin, surrogates, LMS healing, cost-matched evaluation) is built once, serves all.

- **P0 — The compiler-stack platform pitch.** "A compiler mapping trained continuous-time memory
  operators onto static analog circuits under an analog cost functional, with per-chip background
  retuning." Venue-independent (ML-systems/EDA crossover, TVM-analog framing). Kill criterion: if
  realization search never beats hand mapping on the cost functional, it's plumbing, not research.
- **P1 — Datapath application pitch.** "Compiled static feedforward analog memory replacing/augmenting
  DFE: no converters, no 1-UI loop, nonlinear ISI cancellation." Highest risk/reward. Kill criterion:
  state-referred-noise-vs-NRMSE measurement (twin, today) — if memory drowns in its own noise, demote
  to parallel thread (it is already gated behind P0 convergence + D1 timing).
- **P2 — Control-path application pitch (strongest near-term).** "Learned control plane (surrogate-prior
  BO at link-up + neural/bandit tracking) tuning conventional hardwired CTLE/FFE/DFE; absorbs PVT/drift
  by optimizing measured eye/BER; zero data-path risk." Prior-art boundary G20 (offline BO exists;
  online learned closed loop doesn't; production = LMS). Kill criterion: joint-BO + tracking must beat
  per-block sign-sign LMS by measurable dB or convergence time — else it's a controller with no customer.
- **P3 — Methods/benchmark pitch (cheapest, most citable, lead temporally).** "Cost-matched baselines +
  held-out MC discipline + NARMA-C PVT robustness suite for analog RNNs." Nobody has it; everyone cites
  it; every other pitch's evaluation section depends on it. No kill criterion — service work ships regardless.
- **P4 (optional, EDA venue) — Design-for-surrogacy pitch.** "Analog cells constrained so LUT/MLP
  surrogates are exact-by-construction + the 4-level hierarchy." DAC/DATE/TCAD-shaped. Promote only if
  hierarchy demonstrably accelerates compiler search vs SPICE-in-loop baseline.
- **Killer-result anchor (belongs to P1/P2 evaluations):** F6c cross-die single readout — "one readout,
  N dies" — the measurement that makes calibration-free deployment credible.

Enablers (explicitly NOT pitches): PAT-twin positioning, inverter-based cells, TinyTapeout vehicle,
spare-cell redundancy, shadow-SRAM swap mechanism, slicer-STE training, stability-by-structure. Each
supports ≥2 pitches; none stands alone.
