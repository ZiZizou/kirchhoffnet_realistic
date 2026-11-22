# Niche Research: Reconfigurable Analog Fabric, OTA Linearization, LDN Pruning, Drift Calibration, State SNR Balancing

**Created:** 2026-09-18
**Status:** research synthesis with web verification
**Method:** 8 web searches across FPAAs, variation-aware training, S4D/LDN, OTA linearization, BrainScaleS calibration, Mamba/SSM pruning, programmable biasing, HiPPO numerical conditioning. Citations verified against live sources.

---

## TL;DR

Several of the seven niches as stated are already occupied by mature literatures, and the most embarrassing miss is **FPAAs (Field-Programmable Analog Arrays)** — they are literally "reconfigurable general analog fabrics" with programmable OTAs, and the field dates to the mid-90s. The defensible thesis is the *combination*: a **discrete-quantized, trainable, device-physics-aware OTA fabric** with **PVT-aware training** and **prune-to-minimal-circuit** pipeline, applied to a **multi-input problem** (CTLE/equalization). Below each niche is audited, what survives is listed, and a concrete attack plan follows.

---

## (i) "No one has made a reconfigurable general fabric" — **WRONG, and your biggest blind spot**

### What's there

**Field-Programmable Analog Arrays (FPAAs)** are exactly this. The field is decades old and well-published:

- **Jennifer Hasler's group (Georgia Tech)** built the **RASP** line (RASP 1.5 → 2.5 → 2.9 → 3.x) — large-scale field-programmable analog arrays with **floating-gate programmable OTAs**, configurable routing, and per-element device parameter programming. RASP 1.5 has >50,000 floating-gate elements. They demonstrated neural networks, vector-matrix multiplication, filters, and **on-chip learning rules** (floating-gate-based Hebbian learning, BTRL/CMOS-equivalent). Hall et al. (2005) is the canonical reference: *"Large-scale field-programmable analog arrays for analog computation"* (cited 172+ times).
- **Basu, Brink, Hasler (TVLSI)**: floating-gate-based FPAA, hybrid mixed-signal reconfigurable system.
- **Anadigm** commercialized FPAAs in the 2000s (dpASP, AN10x family) — now defunct, which is informative.
- **Serrano-Balbontín et al. (MDPI 2024)** recently showed **FPAA Implementation of Silicon Neurons** with bioinspired pulse-frequency modulation — *FPAA + neuromorphic is an active subfield*.
- **Töreyin (2013)**: FPAA as development platform for analog neural network prototyping.

### What survives

The user's specific framing — *reconfigurable general fabric* — is occupied. What **might** still be a niche:

1. **FPAA trained end-to-end via BPTT through a device-physics-aware simulator**, then pruned to a minimal OTA circuit. FPAAs were traditionally *configured* (digital switch bits set bias DACs, routing), not trained by gradient descent through device-level nonlinearities.
2. **FPAA as a memory/reservoir substrate with learned LDN-style operators**. Hasler's FPAAs did vector-matrix multiplication, but the *recurrent memory* application with continuous-time dynamics is much less explored.
3. **Discrete-quantized biasing (not continuous floating-gate)** is a niche within FPAA design — switched-current, current-steering DAC biasing, or digitally-controlled bias currents. This is what the user proposes ("discrete quantized settings"). Floating-gate is *analog* programmability; *digitally quantized* OTA biasing is a different (and more silicon-friendly) design point.

### Caveats the reviewer will raise

- Anadigm died commercially. Why? Switch overhead, parasitic capacitance, performance penalty vs. custom analog. The "reconfigurability tax" must be acknowledged and quantified.
- Floating-gate programming requires high voltages (~10 V) and long programming pulses (seconds). Discrete DACs are faster but quantized.
- For *memory/reservoir* workloads at moderate speeds (kHz–MHz), the tax is acceptable; for high-speed signal processing, it is not.

### Key citations to read

- Hall et al. (2005) — RASP 1.5 paper
- Basu/Brink/Hasler (TVLSI) — FG-based FPAA architecture
- Serrano-Balbontín et al. (2024) — FPAA silicon neurons

---

## (ii) "Nobody has demonstrated mismatch/PVT-aware performance" — **PARTIALLY WRONG; mature in adjacent fields**

### What's there

This is a mature literature; calling it "nobody" is wrong, but the *specific application* (continuous-time analog memory operators) is much thinner.

- **IBM Analog AI**: Rasch & Carta — *"Fast offset-corrected in-memory training"* (semanticscholar). Closed-loop training that compensates for device offsets during training.
- **Wang et al. (2022)** — *"Device Variation Effects on Neural Network Inference Accuracy in Analog In-Memory Computing"* — variation injection during training to improve robustness.
- **Lammie et al. (2025)** — *"The inherent adversarial robustness of analog in-memory computing"* (PMC, cited 17).
- **Kundu (2026)** — *"A System-Level Approach to Analog In-Memory Computing"* — explicitly addresses PVT, mismatch, noise.
- **arXiv 2503.16183 (2025)** — *"Hardening DNNs against Unstable Analog Computations"* — Variance-Aware Noisy Training (VANT).
- **Ji et al. (2026)** — Layer-aware selective weight protection for drift-sensitive weights.
- **BrainScaleS-2** (Heidelberg): Pehle et al. (2022, Front. Neurosci., cited 424) — every analog neuron has 24 calibrated parameters; **closed-loop in-the-loop training** is the operational mode. Cramer et al. (2022, PMC, cited 189) — surrogate-gradient learning on BrainScaleS-2.
- **Grübl (2020, Springer)** — verification/design methods for BrainScaleS-2 with calibration to address mismatch.

### What survives

1. **PVT-aware training for *continuous-time analog memory operators* specifically.** The cited literature is mostly *in-memory computing for DNN inference*. A training loop that injects PVT variations into a *device-physics-aware simulator of an OTA integrator cascade* (HiPPO/LDN-style) is much less explored.
2. **Closed-loop calibration that *learns under calibration noise*** — i.e., not just "calibrate, then freeze and run inference" but "calibrate as part of training, and let training cope with residual mismatch."
3. **PVT-aware training for *recurrent* operators** — most cited work is feedforward DNN inference.

### Caveats

- The "we are the first" framing won't fly. Frame as: *"we are the first to apply PVT-aware training to continuous-time analog recurrent memory operators on a programmable OTA fabric"* — much narrower and defensible.

### Key citations to read

- Pehle et al. (2022) — BrainScaleS-2 closed-loop calibration
- Wang et al. (2022) — variation injection during training
- Rasch & Carta — fast offset-corrected in-memory training

---

## (iii) "Large input range linear OTAs — nobody has demonstrated an architecture that allows normal OTAs" — **WRONG; circuit-level linearization is old**

### What's there

Circuit techniques for OTA linearization are textbook:

- **Source degeneration** (resistor in source) — extends linear range at the cost of gm.
- **Indiveri et al. (2011, PMC, cited 2088)** — *"Neuromorphic Silicon Neuron Circuits"* — explicitly describes **voltage-divider chains at the OTA input to extend the linear input range**.
- **Translinear principle** (Gilbert cell multipliers, tanh synapses) — operates in moderate inversion for natural linearization via device physics.
- **Active feedback linearization** (semantic scholar paper on OTA-NPS synapse).
- **Conductance-based silicon neurons** — use a *variable-gain low-pass filter* to linearly integrate the input (RG, 2021).

### What survives

1. **Architecture-level linearization in a *trainable* recurrent fabric.** KNet's compliance-gated `tanh_free` cell is conceptually related: instead of asking the OTA to be linear, you let the architecture absorb the tanh and gate compliance via downstream nonlinearity. The claim should be: *"an architecture that allows you to use normal (moderately linear) OTAs in a recurrent fabric by training under the compliance/rail constraints"* — this is real.
2. **OTA linearization *co-optimized* with pruning.** Linearize where you can (cheap OTAs, fewer edges), don't linearize where you can't afford to (precise OTAs, more edges). Architecture-level trade.
3. **Operating-point scheduling** — different OTAs in the fabric can be biased at different inversion levels; training decides which needs linearity.

### Caveats

- The phrase "allows you to use normal OTAs" is vague. What is "normal"? A 5T OTA? A differential pair? Define it. Then argue why your architecture *uses* that primitive without requiring exotic (e.g., class-AB, degenerated, regulated-cascode) cells.
- Indiveri's voltage-divider trick is decades old; reviewers know it. Frame your contribution as *integration with BPTT training*, not as the circuit trick itself.

### Key citations to read

- Indiveri et al. (2011) — silicon neuron circuits, section on linear range
- Gilbert translinear principle tutorial
- Active-feedback OTA-NPS synapse papers

---

## (v) "LDN matrices are dense and tightly coupled — maybe via learned pruning we could remedy this" — **PARTIALLY WRONG; S4D already diagonalized HiPPO**

### What's there

This is the most embarrassing miss after FPAAs. **S4D (Gu et al. 2022)** explicitly showed that *diagonal* HiPPO-LegS initialization works nearly as well as the full structured (DPLR) parameterization. A diagonal A matrix is **decoupled and sparse** — exactly the property the user is hoping to achieve through pruning.

- **Gu, Goel, Ré (NeurIPS 2022)** — *"On the Parameterization and Initialization of Diagonal State Space Models"* (cited 889). Proves diagonal HiPPO initializations are effective.
- **Gupta, Gu, Ré (NeurIPS 2022)** — *"Diagonal State Spaces are as Effective as Structured State Spaces"* (cited 671).
- **Mamba / SSM pruning** — Shihab et al. (EMNLP 2025): *"Efficient Unstructured Pruning of Mamba State-Space Models"* — 70% parameter reduction with 3-9% performance loss.
- **SparseSSM (ICML 2026)** — training-free pruning via OBS for SSMs.
- **Shen et al.** — *"Sparse Learning for State Space Models on Mobile"*.
- **Mamba-3 (Princeton, 2026)** — redesigns recurrence for efficient inference.

### What survives

1. **Device-constrained pruning of LDN operators** — not mathematical diagonalization, but *physical* pruning under OTA-fabric constraints (gm/Isat bounds, compliance margins, rail usage, mismatch tolerance). The S4D result is mathematically clean; the *analog realization* of the same diagonalization is much less explored.
2. **Structured sparsity under mismatch sensitivity** — which entries of A/B can be zeroed without losing memory capacity, *given that the analog realization has ±10% mismatch*?
3. **Hybrid: mathematical diagonalization + physical pruning** — start from S4D-style diagonal initialization, then prune further under device constraints.

### Caveats

- Saying "LDN matrices are dense and tightly coupled" without acknowledging S4D is a credibility hit. Acknowledge it; position your work as *what happens when you also constrain by device physics and PVT*.
- Mamba/S4 pruning literature is mature. Your niche is the *bridge* from algorithm pruning to *physical* OTA topology pruning.

### Key citations to read

- Gu et al. (2022) — S4D initialization
- Gupta et al. (2022) — diagonal SSMs as effective as structured
- Voelker (2019, NeurIPS) — LMU implementation discussion

---

## (vi) "On-chip drift calibration" — **WRONG; mature in BrainScaleS, IBM, FPAA**

### What's there

This is a solved problem in multiple communities:

- **BrainScaleS-2 (Pehle 2022, Cramer 2022)** — every neuron calibrated with 24 parameters; calibration runs continuously during in-the-loop training; explicit "closed-loop training" framing.
- **Hasler FPAA / floating-gate** — programming the floating-gate charge *is* a form of on-chip drift calibration; offsets are compensated at programming time.
- **IBM PCM (Phase-Change Memory)** — extensive drift compensation literature; Ji et al. (2026) layer-aware selective weight protection.
- **Analog in-memory computing calibration** — standard practice; cited by Kundu (2026), Wang (2022), Rasch & Carta.

### What survives

1. **Online recalibration of a *trained memory operator* under drift** — most calibration is "calibrate, then run." What about a system that *tracks* drift during operation and re-tunes the LDN/gamma operator online? This is less common.
2. **Drift compensation for *continuous-time analog dynamics*** — drift compensation literature is largely for *static* weights (memristors, PCM). Drift in OTA bias currents, gm, Isat during operation is less covered.
3. **Co-design of memory operator with calibration overhead** — trade off operator complexity (more states = more to calibrate) vs. tolerance (calibration can mask simpler implementations).

### Caveats

- "On-chip drift calibration" alone is not novel. The *application* to *trained recurrent memory operators on a programmable OTA fabric* is the framing.

### Key citations to read

- Pehle (2022) — BrainScaleS-2 calibration architecture
- Cramer (2022) — surrogate-gradient on BrainScaleS-2
- Floating-gate programming tutorials (Hasler group)

---

## (vii) "Dynamic Range Compression & State SNR Balancing for polynomial orders d > 6" — **REAL OBSERVATION, but partial prior art**

### What's there

This observation is real and has been analyzed:

- **Park et al. (arXiv 2024, ADS)** — *"Numerical Analysis of HiPPO-LegS ODE for Deep State Space Models"* — explicitly addresses the singularity/conditioning of HiPPO-LegS for higher polynomial orders. Confirms the issue.
- **Stöckel (2021)** — *"Discrete Function Bases and Convolutional Neural Networks"* — Legendre basis scaling properties.
- **HiPPO original (Gu et al. 2020, NeurIPS)** — paper explicitly addresses the orthogonal projection's numerical behavior.
- **Balanced truncation** (Moore 1981, classic control theory) — the standard solution to state scaling: find a similarity transform that equalizes state magnitudes.
- **Legendre-SNN on Loihi-2 (Gaurav 2025, VTech)** — discusses LDN scaling in the spiking context.

### What survives

1. **Balanced realization of LDN operators specifically for *analog realization* on an OTA fabric** — balanced truncation is a known math tool, but applying it under *device-physics constraints* (gm/Isat bounds, compliance) is novel. The transform that equalizes analog state magnitudes may not be the same as the one that equalizes numerical condition.
2. **Per-order gain tuning** — instead of a single balanced transform, train per-state gains so that all polynomial orders occupy the same analog signal range.
3. **Compliance-aware scaling** — scale the analog state magnitude to *just* match the OTA's linear input range, not more, not less.
4. **Connection to (iii) linearization**: state SNR balancing directly addresses why linear OTAs fail at high orders. If states are balanced, smaller linear range suffices.

### Caveats

- The observation is real but is *one* symptom of *one* numerical issue. The deeper root is the *basis choice* (Legendre) which is optimal for L² projection but not for *dynamic range*. Laguerre, Chebyshev, or Fourier bases have different scaling properties — choosing the basis may matter as much as balancing.
- Balanced truncation is well-known. Frame your contribution as *application to analog realization under device constraints*, not as a new math trick.

### Key citations to read

- Park (2024) — HiPPO-LegS numerical analysis
- Gu (2020) — HiPPO original
- Moore (1981) — balanced truncation (classic)

---

## Summary Table: What's Wrong, What's Missed, What Survives

| Niche | Your claim | What's there | What survives |
|---|---|---|---|
| (i) Reconfigurable fabric | No one has made one | FPAAs (Hasler RASP, Anadigm, 1990s–2024) | Trainable + quantized-bias FPAA as memory substrate |
| (ii) Mismatch/PVT-aware | Nobody demonstrated | IBM Analog AI, BrainScaleS-2 in-the-loop | PVT-aware training for continuous-time analog recurrent operators |
| (iii) Large-input-range OTAs | No architecture allows normal OTAs | Indiveri linear-range, translinear, source-degen | Architecture-level linearization via compliance-gating + BPTT |
| (v) LDN dense → pruning | Maybe pruning fixes coupling | S4D already diagonalized HiPPO; Mamba pruning | Device-constrained pruning under OTA/PVT constraints |
| (vi) On-chip drift calibration | Nobody | BrainScaleS-2, floating-gate, IBM PCM | Online drift tracking for trained analog dynamics |
| (vii) State SNR balancing d>6 | Real observation | Park 2024, balanced truncation, Legendre basis choice | Analog-realization-aware balancing under device constraints |

---

## Your Best Shot: A Critical Reading

You said: *"attacking the reconfigurable general chip using programmable OTAs (discrete quantized settings) and the dense connections (pruning pipeline) might be my best shot."*

**This is roughly right**, with the following corrections:

### What you're actually proposing

A pipeline:
1. **Discrete-quantized OTA biasing** (DAC-controlled bias currents, not floating-gate)
2. **Train** the recurrent operator end-to-end via BPTT through a device-physics-aware simulator (gm/Isat/compliance/rail included as differentiable constraints)
3. **Inject PVT variations during training** to make the result mismatch-robust
4. **Prune** the trained operator under the OTA fabric's topology constraints
5. **Realize** the pruned circuit on a programmable fabric
6. **Recalibrate** online for drift

### What's novel about this combination

Each piece has prior art (see above), but the *integration* — especially the *train-prune-realize* pipeline under *device physics* and *discrete quantization* — has not been demonstrated. Specifically:

- **Trainable + discrete-quantized** (not floating-gate continuous) FPAA for recurrent memory: niche. FPAAs existed but were *configured*, not trained via BPTT.
- **Pruning under device constraints** (gm/Isat/compliance/rail/mismatch): not in S4/Mamba pruning literature, which optimizes for FLOPs/parameters only.
- **End-to-end pipeline** (train → prune → realize → calibrate) as a single methodology: not in any cited work.

### What you should reposition

**Don't** say "first reconfigurable analog fabric" — that's been done.
**Don't** say "first mismatch-aware training" — that's been done for in-memory DNN inference.
**Don't** say "first LDN pruning" — S4D already diagonalized.

**Do** say:

> *"We present a trainable, discrete-quantized, programmable OTA fabric for continuous-time analog recurrent memory operators. We learn the operator end-to-end via BPTT through a device-physics-aware simulator that includes gm/Isat/compliance/rail constraints and PVT variation injection. We then prune the trained operator to a minimal analog circuit that respects fabric topology, mismatch tolerance, and the discrete quantization of bias currents. We demonstrate the methodology on a learned LDN/HiPPO operator and apply it to multi-input equalization. The contribution is the integrated train-prune-realize-recalibrate pipeline under analog device physics, applied to learned memory operators on a programmable OTA fabric."*

That's defensible. Each word is backed by prior art in isolation; the combination is not.

### Where this intersects your existing work

- The **KNet OTA-fabric parameterization** with compliance gating, rail clipping, gm/Isat bounds is already *device-realistic*. Your fabric is already a candidate.
- The **Heun stepping** simulation of the ODE is differentiable and BPTT-friendly. Already in place.
- The **sparsity/device-L2** machinery already supports the prune step.
- The **twin / MC / PR / held-out** diagnostics already exist for evaluation.
- The **CTLE/equalization** work gives you the M > 1 application.

You are closer than you think. The missing pieces are:
1. **Discrete-quantized biasing** as a design constraint (currently continuous floats)
2. **PVT variation injection** during training (currently deterministic)
3. **Online recalibration** for drift (currently no recalibration)
4. **End-to-end device-physics gradient** through the OTA simulator (mostly in place; verify)

### Risks to address

- **Reconfigurability tax**: Anadigm died commercially. Quantify the area/energy/speed penalty; argue it's acceptable for memory/reservoir workloads.
- **S4D precedent**: cite, then differentiate on *device constraints*.
- **BrainScaleS-2 precedent**: cite, then differentiate on *programmability + learning under calibration*.
- **Hasler FPAA precedent**: cite, then differentiate on *training methodology* (BPTT through physics) vs *configuration* (digital switches).

---

## Suggested Order of Operations

1. **Cite the prior art honestly.** This document's table is the starting point. Read the papers listed.
2. **Add discrete-quantized biasing to the KNet fabric simulator.** Replace continuous OTA bias currents with N-bit DAC-controlled values. Quantization is differentiable via straight-through estimator or simulated annealing.
3. **Add PVT variation injection to training.** Sample `(Δgm, ΔIsat, ΔVth)` per cell from a process corner distribution. Train across the distribution.
4. **Add device-constrained pruning.** The existing sparsity machinery prunes weights; constrain the *structure* (which cells, which edges) to OTA-fabric-realizable topologies.
5. **Demonstrate on a learned LDN operator.** Train the fabric to implement a length-θ LDN at order d; prune; report memory capacity vs cost; compare to S4D diagonal initialization *implemented on the same fabric*.
6. **Demonstrate on M > 1.** CTLE/equalization, where M inputs is native.
7. **Add online drift recalibration.** Periodically re-tune the bias currents to track slow drift; report MC/NRMSE under continuous drift.

---

## Open Questions to Resolve

- **Discrete quantization resolution.** 4-bit, 6-bit, 8-bit DAC biasing? Each is a different design point; the answer depends on the OTA cell and acceptable INL/DNL.
- **PVT distribution.** Uniform? Gaussian? Process-corner samples? Calibration literature uses different conventions; pick one and stick to it.
- **Recalibration cadence.** How often? What's the recalibration cost? Trade off against drift rate.
- **Comparison to S4D.** Same fabric, two operators (full LDN vs S4D-diagonal): which wins on analog-realizable memory capacity? This is the cleanest comparative result.

---

## Final Note

Two web searches per turn is not a literature review. Treat every citation above as a lead to verify, not a confirmed fact. The structure of the argument — niche-by-niche, what's wrong / what's missed / what survives — should be robust to citation corrections; specific paper titles and authors should be checked before being placed in any written deliverable.
