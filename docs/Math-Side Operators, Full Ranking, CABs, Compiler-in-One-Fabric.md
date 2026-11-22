# Math-Side Operators, Full Ranking, CABs, and Compiler-in-One-Fabric

**Created:** 2026-09-18
**Status:** synthesis + 2 verification searches (Givens-analog, lattice-neural — no counterexamples found)
**Questions answered:** (1) what can be done on the math side — piecewise operators and beyond; (2) final ranking of every idea across all reports; (3) what CABs are; (4) compiler explained simply, mapped onto one physical fabric, confirming/correcting the overcomplete→spec→deactivate→run mental model.

---

## 1. The math side — yes, it is under-exploited, and your piecewise instinct is good

Everything so far was synergy (combine known blocks) or physics-native features. The math side asks a different question: **what dynamical operator should the fabric implement in the first place?** The operator is pure math — basis, structure, parameterization — and different math has radically different analog cost. This is the cheapest place to innovate (no tapeout, no cells, just the twin simulator) and it multiplies every downstream idea, because the compiler's job gets easier or harder depending on the operator you hand it.

### 1a. Your piecewise question, answered directly

Yes — piecewise / regime-switched operators are both mathematically fresh for memory and analog-cheap, because a comparator + switched bias current *is* a piecewise dynamical system. Concrete form ("gearbox memory"):

- Partition by input activity: when |u| is small, run a long-window, low-leak operator (remember far); when |u| is large, switch to a short-window, fast operator (track, don't saturate). Classical control calls this **gain scheduling**; hybrid-systems theory calls it **switched dynamics**.
- Partition by state energy: if total state RMS exceeds a bound, shorten all time constants (shed old history instead of clipping) — a *graceful-forgetting* schedule. This is a dynamical answer to vii that composes with AGC rather than duplicating it.
- Cost: one comparator per regime boundary + switched bias codes you already plan to have (quantized OTAs). The math to verify: dwell-time / average-dwell-time stability conditions (Liberzon textbook) so switching doesn't chatter or destabilize — checkable in the twin before any silicon.

Prior art status: gain-scheduled control is 1990s-classical; *learned* regime boundaries + *learned* per-regime memory operators for a recurrent neural memory do not surfacely exist. The learning question (differentiate through switches — straight-through estimator territory) is the research slice.

### 1b. Math-side shortlist (operator innovations, analog-cost-aware)

**M1. Modal / biquad realization as the compilation target.** Eigen-decompose the trained A; implement each complex-conjugate pole pair as one OTA-C biquad (the canonical analog block), each real pole as one integrator. Classical control (modal decomposition); *using it as the compilation target for gradient-trained RNN memory* did not surface. Composes with per-mode AGC (F3), per-mode companding (F2), per-mode mismatch analysis. This may be the highest-ROI math idea: it converts "arbitrary dense A" into "a shopping list of biquads," which is exactly the language analog designers already speak.

**M2. Lattice-form learned memory.** Adaptive lattice filters (Burg, gradient-lattice/GAL) are classical DSP: identical stages, reflection coefficients bounded by |k|<1 which *guarantees* stability, order-recursive structure. The Laguerre-lattice combo even exists (Fejzo/Lev-Ari 1997, gradient-adaptive). What does not exist: a lattice-structured *neural* memory trained by gradient descent and mapped to an analog fabric. Why it matters for you: identical stages → best-case matching; bounded coefficients → quantize-friendly (your discrete biases); stability by construction → answers reviewer question C3 for free. Check Harris 1999 (adaptive Laguerre in analog VLSI, gradient-based) as the closest ancestor and differentiate on * programmability + learned (not adaptive-filter) training*.

**M3. Givens-structured orthogonal memory.** Orthogonal/unitary RNNs are a mature digital lineage (Arjovsky uRNN; Jing EUNN; Lezcano-Casado; Foucault 2024 quantized ORNN). A Givens rotation is a pairwise coupling — one differential pair with matched gm. So: parameterize memory as products of learned Givens rotations = norm-preserving (no vanishing), each rotation = one analog primitive, quantization = angle quantization. No analog implementation surfaced. Bonus: rotations are *insensitive* to common-mode mismatch (both sides of the pair drift together), which turns your mismatch story from defense into offense.

**M4. Liquid / input-dependent timescales fused with LDN structure.** Hasani LTCs/NCPs (MIT 2020–22) made input-dependent τ respectable; your `--dynamic-leak` spike already built the mechanism. The math-side move: *structured* liquid memory — LDN coupling matrix with gated leaks, i.e., "input-gated Legendre memory." Neither pure-LTC (unstructured) nor pure-LDN (fixed τ) literature owns this intersection. Your dyn-leak audit trail is already half of this paper.

**M5. Selective / learned-measure HiPPO (Mamba-style selectivity on fabric).** S4/S5/Mamba's learned Δ *is* a learned time-warping of the HiPPO measure. The math idea: input-dependent measure + LDN basis, compiled to modulated bias currents. Digital selectivity needs per-step Δ updates (cheap digitally); analog needs a modulation path per timescale (a real wire-cost question — which is itself a publishable cost analysis). Position against Mamba explicitly: same math, different substrate, honest wire-cost accounting.

**M6. Koopman lift: linear evolution in analog, nonlinear only at boundaries.** Learn observables that linearize the channel dynamics; implement the *linear* Koopman evolution in the fabric (linear = precise, low-power, mismatch-tolerant) and keep the nonlinear lifting at input/output edges. Elegant division of labor; Koopman autoencoders exist digitally, analog Koopman equalizer does not surface. Risk: finding good finite observables is the whole hard problem — scope to channels where physics suggests them (your CTLE models may already hint at the right lift).

**M7. Multi-window mixture (multi-θ LDN + learned mixing).** Block-diagonal A, one θ per block, learned readout mixing — S5 already uses block-diagonal HiPPO-N init (verified), so the *math* exists; the *analog* version (one capacitor size per block, free) plus learned mixing under quantization does not. Cheap to test in the twin; do it as part of E1 rather than as its own project.

**M8. Volterra-readout co-design.** An LDN + polynomial readout *is* a Volterra series. Jointly learn basis order and Volterra readout order against the task's sufficient statistics (NARMA needs exactly order-2 with specific lags). Turns your E2-arm-B overkill question into architecture search with a principled stop criterion. Small, sharp, publishable as a methods note.

**M9. Hybrid integrate-and-reset states.** Continuous integration + comparator-triggered reset to a learned value — graded-state memory with event-gated resets (spiking's cousin, but states stay graded). Very analog-native (comparator + reset switch); as *learned memory with learned reset maps*, thin literature. Pairs with F11 event-driven thinking.

**M10. Free-Wiener SDE states (minor).** Thermal noise *is* a Wiener process; digital SDE-RNNs must simulate what you get from physics. Stochastic-resonance-flavored memory is a small, weird, cheap leg — run it only if M1–M4 stall.

**M11. Parametric pumping (exotic, one line).** Refresh decaying states by pumping a parameter at 2× resonance instead of burning gain power. Classical physics, essentially absent from ML memory. Mention in future-work sections to signal ambition; do not schedule.

### 1c. How the math side composes with everything else

The operator is upstream of all of it: the compiler (layer 2) searches *realizations of a given operator* — a better-structured operator (lattice, modal, Givens) shrinks the search and improves every downstream number (edges, sensitivity, quantizability). Recommended order: **M1 (biquad target) + M4 (gated LDN) first**, because both plug directly into the existing twin + E-plan; M2/M3 second as parallel low-cost legs; M5–M9 opportunistically.

---

## 2. Final ranking of ALL ideas discussed (four reports)

Tiered (tiers honest; order within tier matters). Cost = simulator weeks unless noted. "You" = your setting: analog-in/analog-out equalization, no converter tax, KNet twin + fabric existing.

### Tier S — do these; each is a paper unit or the foundation

1. **Compiler middle layer: realization search under analog cost functional (M1 as first target).** The thesis engine. Everything else is a passenger. Cost: 2–3 months. *This is the bet.*
2. **F7 analog neural DFE (slicer-in-the-loop, no converters).** The application paper; owns the converter-tax sentence nobody else can say. Start behavioral models now, in parallel.
3. **F5 learned-pole Laguerre + same-fabric Legendre-vs-Laguerre mismatch comparison.** Decides the basis question before tapeout/pruning; one scalar to learn; clean paper unit. Weeks.
4. **M4 input-gated LDN (productize your dyn-leak spike).** Mechanism built, audited; needs GPU verdicts + task metrics. The math-side flagship.

### Tier A — high ROI, schedule next

5. **F2 companding states + F3 per-state AGC.** Direct vii answer; relaxes OTA-linearity (iii); subthreshold OTAs nearly there. 1–2 months.
6. **M2 lattice-form learned memory.** Identical stages + guaranteed stability + quantize-friendly; closest ancestor Harris 1999 to differentiate against. Weeks in twin.
7. **M3 Givens orthogonal memory.** Norm-preserving + common-mode-mismatch-immune; turns mismatch into offense. Weeks in twin.
8. **Your quantized-OTA + PVT-injected BPTT + device-constrained pruning pipeline.** The fabric program; runs on the M1/M4/F5 winners, not on arbitrary operators. 2–3 months.
9. **F1 trained delay-loop virtual nodes (fully-analog delay).** Deletes the matrix structurally; the second answer to the cost question alongside aux-core. Needs analog delay model. Medium.
10. **F6 mismatch-as-projection + cross-die single readout (F6c).** Kills programming/calibration/drift in one measurement; PVT suite falls out. Low cost (Monte-Carlo harness).

### Tier B — strong supporting / service contributions

11. **B1/B2 multi-input benchmark + NARMA-C robustness suite.** Cheap, cited by everyone including you. Do continuously.
12. **F9 passive SC/bucket-brigade bank.** Cheapest experiment; sets the honest cost floor (Phase-11 fallback, quantified). Days–weeks.
13. **M7 multi-window mixture.** Fold into E1; don't standalone it.
14. **M8 Volterra-readout co-design.** Small methods note; answers the overkill question with a stop criterion.
15. **C2 slicer-in-the-loop (STE) training.** Required sub-task of F7; list separately so it doesn't get forgotten.
16. **M6 Koopman lift.** Elegant; gated on finding good observables — time-box it.
17. **D1 TinyTapeout analog test chip.** The moat; schedule when Tier-S/A converge (tapeout too early = taping out the wrong circuit).

### Tier C — real but deferred

18. **F4 fractance memory.** Highest novelty, real channel-physics match (skin effect ~ half-order); needs ladder models + validation. High-risk parallel thread only.
19. **F8 cellular fabric; F10 bilinear states; M9 integrate-and-reset.** Narrower; promote on evidence.
20. **F12 dissipative learning rule (theory).** Position via the twin (physics-aware training, engineering answer); don't compete with the HEB/RHEL theory crowd.
21. **F11 event-driven updates; M10 free-Wiener; M11 parametric pumping.** Future-work-section material unless blocked elsewhere.
22. **D3 mixed-signal DFE+reservoir hybrid.** Sensible product architecture; research-novelty thin — keep as the industry-facing slide, not the thesis.

### Explicitly deprioritized (occupied — cite, don't attack)

- "First reconfigurable fabric," "first mismatch-aware," "first LDN pruning," "RNN that learns its own memory," new HDL for analog, general analog synthesizer, out-theory-ing HEB/RHEL.

---

## 3. What are CABs?

**CAB = Configurable Analog Block.** The repeating tile of an FPAA — the analog equivalent of an FPGA's CLB (Configurable Logic Block). One CAB is a small cluster of analog primitives (OTAs/opamps, capacitors, resistors, switches, local routing muxes) whose function is set by configuration bits. Examples: Anadigm AN231E04 = 2×2 CABs; RASP 2.8 = 32 CABs. So "place/route onto CABs" means: decide which CAB holds which integrator/biquad/VGA, which internal switches close, and how signals route between CABs — exactly like FPGA place-and-route, except the "logic elements" are continuous-time analog stages and the "timing closure" is pole locations and stability instead of clock slack.

---

## 4. Compiler-in-one-fabric, simply — and yes, your mental model is right

Your guess — **start overcomplete → spec in Python → compiler analyzes → deactivate connections → run** — is essentially correct. Here it is cleaned up, end to end, on one physical fabric.

### The setup (one chip, three kinds of software)

- **The fabric:** one reconfigurable OTA chip, deliberately built *overcomplete* (more OTAs, switches, and caps than any single design needs — same philosophy as an FPGA having more LUTs than your circuit uses). Unused cells get powered down; unused switches stay open. Overcompleteness is not waste, it is the thing that makes one chip serve many operators.
- **The spec (Python, runs on your workstation):** "Implement LDN order-8, window 12, driven by this input range, 6-bit bias codes allowed, at most 40 OTAs, survive ±10% gm spread." Plus the trained weights from your normal PyTorch training. No new language.
- **The compiler (software, runs on your workstation):** takes the spec, searches candidate realizations *in the twin simulator* (prune here? balance there? biquad or lattice form? which 40 of the 200 OTAs? what bias codes?), scores each against the analog cost model, and emits a *configuration*: a list of which cells are on with which codes + which switches are closed + readout weights.
- **The runtime (tiny software + on-chip helpers):** programs the chip, probes it with a known input, fits what the chip *actually* does, nudges a handful of bias codes to close the gap, then runs the real task and re-nudges on drift.

### Concrete walkthrough (numbers illustrative)

1. You train an order-8 Laguerre operator in PyTorch. It uses dense math — 64 couplings — and floats.
2. Spec: "fit on fabric v1 (200 OTAs, 6-bit biases), ≤40 OTAs, fan-in ≤ 8, all states within ±400 mV."
3. Compiler, in the twin: tries dense / pruned / biquad-modal / lattice forms × balanced / unbalanced × 6-bit quantized; simulates each under sampled mismatch; picks lattice-balanced-quantized: 34 OTAs, worst state 380 mV, MC within 5% of float reference across the PVT samples.
4. Emission: "OTAs {3,7,12,…} ON with bias codes {41,22,…}; switches {…} closed; readout weights {…}; OTAs unused → power-gated." This is the bitstream — the analog equivalent of an FPGA bitfile.
5. Program the chip. Probe with a chirp; fit the realized impulse response; 3 bias codes off by 2 LSB → correct. Run equalization. Once an hour, re-probe and re-nudge (drift loop).
6. Next month, a different operator (order-6 Legendre DFE post-filter) compiles to a *different* 29-OTA subset of the *same* chip. The fabric didn't change; the bitstream did. That is what "reconfigurable" buys you, and it is the sentence FPAA vendors can't say about trained dynamics and PAT labs can't say about fixed physics.

### Map back to the advisor's pitch (clause by clause)

- *"Reconfigurable analog fabrics exist (FPAA)"* → the physical chip class; you build (or emulate-then-tapeout) one instance of it.
- *"Trained physical networks exist (PAT)"* → your training happens in PyTorch + twin, PAT-style; the differentiation is that you compile to *any instance* of the fabric, not absorb *one* instance.
- *"Optimal memory operators exist"* → these are the *inputs* to the compiler (LDN/Laguerre/gamma math), not the output. You don't invent memory; you compile it.
- *"Prunes under analog cost constraints"* → the middle-layer search (step 3 above).
- *"Maps onto a mismatch-tolerant reconfigurable OTA fabric"* → steps 4–5: the bitstream + PVT-robust selection.
- *"Per-chip closed-loop retuning as final layer"* → step 5's probe-fit-nudge loop; the runtime from §4 of the previous report.

### Two corrections to the naive version of your model

1. **It is not one-shot.** "Analyze → deactivate → run" suggests compile-once. Reality is compile → program → *measure the actual chip* → retune → run → re-tune on drift. The measure-and-retune loop is load-bearing (it's where instance-specific mismatch gets absorbed) — budget for it as a first-class layer, not a debug step.
2. **"Deactivate" is a power feature, not leftover waste.** Powered-down OTAs and open switches are how the fabric meets the power budget; the compiler's edge count *is* the power number. Report "34/200 OTAs active" the way digital reports gate count — it makes the overcomplete starting point a virtue (headroom for harder operators later) rather than an inefficiency to apologize for.

---

## 5. Caveats (unchanged policy)

Givens-analog and lattice-neural novelty: *searches didn't surface* implementations — treat as standing-until-falsified, keep an eye out during the reading pass. Piecewise/gated-memory learning (differentiating through regime switches) is standard STE territory but the *memory-operator* application is the claim — verify against switched-systems ML literature ("switched neural networks," "mixture of SSMs") before writing. All rankings assume your stated setting; if the target venue is circuits rather than ML, promote D1/F9 and demote M-items one tier.
