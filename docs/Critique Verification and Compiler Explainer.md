# Verification of the Pushback Critique + What "Compiler" Means Here

**Created:** 2026-09-18
**Status:** 8 web searches (4 MiniMax + 2 websearch/fast + 2 websearch/deep-equivalent); all anchor claims checked against live sources
**Context:** a critique challenged the earlier niche reports' claims (FPAA history, mismatch literature, LDN sparsity, drift, balancing) and proposed re-anchoring the program around a "compiler" stack. This document verifies each counter-claim, then explains the compiler concept in plain terms.

---

## 1. Claim-by-claim verification verdict

| # | Critique's claim | Verdict | Evidence |
|---|---|---|---|
| (i)-a | RASP 2.8: 32-CAB floating-gate FPAA, >50k programmable FG elements | **CONFIRMED** | Basu CICC 2008 ("over fifty thousand programmable floating gate elements"); Schlottmann 2012 TVLSI ("RASP 2.8 ... 50,000 switches and 32 CABs") |
| (i)-b | RASP 2.8 in JSSC 2010; <1% current accuracy 6 pA–20 µA; Simulink→SPICE→P&R toolchain; 23 µW VMM classifiers | **CORE CONFIRMED, NUMBERS UNVERIFIED** | Chip + toolchain confirmed via Hasler FPAA papers page and Simulink-tool 2012 TVLSI paper; exact accuracy/range/wattage figures not individually checked — look them up before citing |
| (i)-c | RASP 3.0 SoC FPAA: on-chip MSP430, DACs/ADCs, digital CLBs, partial rapid reconfig via shift-register T-gate row | **PLAUSIBLE, DETAILS UNVERIFIED** | SoC FPAA line confirmed (Okika "System-on-Chip FPAAs" page; Hasler SoC-FPAA program); MSP430/T-gate specifics not checked |
| (i)-d | Anadigm/Okika AN231E04: shipping, dynamically reconfigurable, background-loaded config, single-clock-cycle swap | **CONFIRMED, STRONGER THAN STATED** | Datasheet + User Manual verify: shadow SRAM, Shadow→Config transfer in a single clock cycle, *partial* reconfiguration supported, host-driven on-the-fly updates; now sold by Okika Devices as FlexAnalog (ex-Anadigm dpASP) |
| (i)-e | "Nobody has compiled a trained recurrent dynamical operator (LDN-class + nonlinear core) onto one with gradient-trained parameters" | **NO COUNTEREXAMPLE FOUND** | Everything surfaced (VMM classifiers, filters, AM receivers, AnadigmDesigner2 CAM library, RASP Simulink flow) is hand-designed signal-processing blocks, not gradient-trained dynamics. Negative is hard to prove; treat as "standing until falsified" and keep searching |
| (ii)-a | Hasler 2017 TVLSI calibration flow; σ 14.3 mV → 1.04 mV; "program without dealing with mismatches" | **PAPER CONFIRMED, NUMBERS UNVERIFIED** | Kim/Hasler "Calibration of Floating-Gate SoC FPAA System," IEEE TVLSI July 2017 (cited 38); abstract confirms VT0-mismatch calibration flow for large-scale FG SoC FPAA. Sigma figures not checked — verify before citing |
| (ii)-b | Wright et al. Nature 2022, Physics-Aware Training = mismatch-aware training through measured device response | **CONFIRMED** | Nature s41586-021-04223-6 (cited 1071+): "hybrid in situ–in silico algorithm called physics-aware training"; public repo mcmahon-lab/Physics-Aware-Training. This is the single closest paradigm to your program — must-cite, must-differentiate |
| (ii)-c | Memristor reservoirs (Du et al. Nat. Comm. 2017), BrainScaleS-2, DYNAP-SE do calibration/in-situ training | **CONFIRMED** | Du et al. s41467-017-02337-y (cited 1283): dynamic WOx memristors as reservoir; Moon 2019 follow-up (cited 745); 2024 review literature active. BrainScaleS-2/DYNAP-SE calibration confirmed in earlier searches |
| (iii) | RASP CABs contain "wide-linear-range Gm blocks"; linearized transconductors are decades old | **PLAUSIBLE, WORDING UNVERIFIED** | Linearized-OTA subfield confirmed old (source degen, translinear, Indiveri 2011 divider chains); RASP CAB contents not individually checked. The *inversion* (train around narrow OTAs rather than linearize) is yours to argue — and it overlaps PAT, so cite Wright 2022 there |
| (v)-a | HiPPO is NPLR; S4D/S5 use diagonal state matrices approximating HiPPO-LegS | **CONFIRMED IN DEPTH** | S5 paper (Smith et al. 2022) states it explicitly: HiPPO-LegS "cannot be diagonalized in a numerically stable manner" → NPLR/DPLR form → HiPPO-N diagonal approximation "achieves comparable performance"; DSS + S4D same finding. Reviewer *will* ask "why not S4D" — have the answer ready (see §3) |
| (v)-b | Phase-11 tapsVD4 tension: delay operators resist low-rank compression; measure before promising pruning dividend | **CONFIRMED (YOUR OWN DATA)** | Your harness: tapsVD4 MC 9.21 vs 11.47 at half edges, k locked, never swept. The critique's warning is your own result quoted back — E0/E3 legs first, novelty claims second |
| (vi) | FG re-programmable in-loop; in-situ chips re-absorb drift; pole-drift closed loop unconfirmed | **AGREE WITH THE HEDGING** | FG re-programmability confirmed (Anadigm shadow-SRAM mechanism; Hasler programming infrastructure); pole-specific drift-retuning loop not found — remains your unconfirmed-but-plausible slice |
| (vii) | State-amplitude equalization = Moore 1981 balanced realizations + Mullis-Roberts 1976 optimal digital-filter structures | **CONFIRMED** | Mullis "Synthesis of minimum roundoff noise fixed point digital filters," IEEE Trans. Circuits Syst. CAS-23 (1976), pp. 551–562 (cited 1098). The framing "Mullis-Roberts retargeted from roundoff to rail/device-noise/mismatch" is genuinely new-as-far-as-verified for analog neuromorphic memory |
| Missed-1 | PAT + descendants = closest existing paradigm | **CONFIRMED, UPGRADE TO MUST-READ** | 1071+ citations; instructional repo exists — read the repo, not just the paper |
| Missed-2 | Calibrate-vs-absorb hybrid (statistical device model + quantized per-chip re-tune) unowned | **NO COUNTEREXAMPLE FOUND** | Plausible; this is your sharpest positioning sentence candidate |
| Missed-3 | Memristor/dynamic-device RC 2017–2025; their weakness = fixed dynamics + readout-only learning | **CONFIRMED** | Large active literature (Du 1283 cites; Moon 745; 2024 reviews). Positioning line confirmed usable: trainable dynamics vs fixed dynamics |

**Bottom line on the critique:** its verifiable claims check out. Nothing it asserted with specifics was wrong; where it hedged (vi, vii-thesis-details, RASP numbers), the hedging was honest. Treat it as a reliable related-work skeleton, not gospel — the ~8 anchor papers still need a real reading pass.

---

## 2. What "compiler" means here, in simple terms

**It does NOT mean inventing a new programming language like SystemVerilog.** That reading is understandable but wrong, and pursuing it would waste a year. Here is the plain version.

### The one-paragraph version

You already describe *what behavior you want* in Python/PyTorch (matrices A/B/C, time constants, tanh). A chip fabric only understands *device settings* (which OTA, which bias-current code, which switches closed, which capacitor). A **compiler** is the automatic toolchain that translates the first into the second — choosing *which realization* of your operator to build (pruned? diagonal? balanced? quantized to what?) so that it fits the fabric's constraints and survives its imperfections. Think of it less like "a new language" and more like **the backend of PyTorch for an analog chip**: you keep writing models the way you do now; the compiler worries about bias codes, routing, and mismatch.

### Why the SystemVerilog analogy breaks

- **SystemVerilog describes digital hardware for synthesis into gates.** That works because digital has standard cells (NAND, NOR, flip-flop) with characterized timing/power, and synthesis is a solved combinatorial mapping problem.
- **Analog has no standard cells and no gate-level synthesis.** Every transistor's behavior depends on its sizing, bias, layout, neighbors, temperature. There is nothing to "synthesize to" — which is exactly why analog design is still manual.
- **Existing analog HDLs (Verilog-A / Verilog-AMS) are for modeling and simulation, not synthesis.** They let you *simulate* a circuit, not *generate* one from behavior. Writing a new one would give you a nicer simulator input language and zero mapping capability.
- **The closest existing things to "analog synthesis" are FPAA toolchains** — AnadigmDesigner2 (drag-and-drop CAM blocks → config bitstream) and the RASP Simulink→SPICE→place-and-route flow. Note what they compile: *hand-designed blocks* (filters, VGAs), chosen by a human, with parameters set by a human. Nobody's toolchain compiles a *trained* operator, because the mapping search (which pruned/balanced/quantized realization?) is the research, not the plumbing.

### The four layers of the stack (what you would actually build)

1. **Frontend — operator spec (you already have this).** A trained dynamical operator: state matrices, input/output maps, cell nonlinearities, plus constraints ("fit in 32 OTAs," "bias codes are 6-bit," "fan-in ≤ 8"). Language: Python/PyTorch objects. No new language needed. *If* you later want a small IR (intermediate representation) for operators — analogous to XLA/HLO for accelerators — that is a convenience, not the contribution.
2. **Middle — realization search (this is the thesis).** Given the operator + an *analog cost model* (area, power, wire length, mismatch sensitivity, quantized settings, state dynamic range), search over realizations: prune, diagonalize, balance, quantize, reallocate sensitivity to well-matched devices. Output: a fabric-ready netlist + predicted performance distribution across PVT. Your existing pieces (BO/pruning machinery, twin simulator, MC/PR diagnostics) are the raw materials of this layer.
3. **Backend — place, route, emit codes (engineering).** Map the netlist onto CABs, route switches, emit bias DAC codes + switch settings. AnadigmDesigner2 and the RASP tools prove this layer is buildable; yours differs only in that its input comes from layer 2 instead of a human.
4. **Runtime — per-chip calibrate/retune (your niches ii + vi).** Measure the fabricated instance, close the residual gap with quantized re-tuning, track pole drift over time. The AN231E04's shadow-SRAM single-cycle swap is the existence proof that the hardware side of this loop is feasible.

### The one sentence to put in a paper

> "We present a compiler stack that maps trained continuous-time memory operators onto a quantized, mismatch-tolerant OTA fabric: a realization search under an analog cost functional (locality, fan-in, quantized gm, state dynamic range), a place-and-route backend emitting device codes, and a per-chip closed-loop retuning runtime."

### What NOT to build

- A new HDL or programming language. (Reviewers will ask "why not Python + a backend?" and you will have no answer.)
- A general analog synthesizer. (Scope to *your* fabric and *your* operator class — memory operators + readout. Generality is what killed every previous attempt.)
- The backend before the middle. The place-and-route is plumbing; the realization search under the analog cost functional is the publishable research. Build the cost model + search first, in simulation, using the twin you already have.

---

## 3. Ready-made answers the critique hands you (use them)

- **"Why not S4D?"** → Diagonal SSMs are sparse in a way analog doesn't care about: each state still needs its own precision time constant plus global input/output fan. Analog wants *local* connectivity, bounded fan-in, matched element counts, quantized settings. Nobody has studied which sparse realization of the delay operator minimizes an *analog* cost functional. (Plus your tapsVD4 data point: delay operators resist naive compression, so the realization question is empirical, not settled by S4D.)
- **"Isn't this PAT?"** → PAT absorbs *one* physical instance's quirks through measured-response backprop. The compiler targets a *reconfigurable* fabric with quantized elements, where the mapping must generalize *across* instances; per-chip quantized re-tuning closes only the residual. Per-instance absorption vs. instance-agnostic compiled robustness — different claims, cite and differentiate.
- **"Isn't this balanced realizations / Mullis-Roberts?"** → Yes, deliberately: Mullis-Roberts-style realization optimization retargeted from roundoff noise to rail swing, device noise, and mismatch. The digital-filter version optimizes the wrong cost function for your substrate; porting the *method* with an *analog* cost functional is the contribution.
- **"Isn't this just FPAA configuration?"** → FPAA toolchains map human-designed blocks. The compiler maps *trained dynamics* through a realization search no human performs by hand. Different input, different machinery, same fabric family.

---

## 4. Anchor reading list (verified existence; read before any claim enters a document)

1. Basu et al., "RASP 2.8" (CICC 2008) + JSSC follow-up — fabric baseline
2. Schlottmann et al., Simulink FPAA tool (TVLSI 2012) — closest existing "compiler"
3. Kim/Hasler, "Calibration of Floating-Gate SoC FPAA System" (TVLSI 2017) — calibration baseline
4. Wright et al., "Deep physical neural networks trained with backpropagation" (Nature 2022) + PAT repo — paradigm to differentiate against
5. Voelker thesis (LMU) — check scaling/balancing discussion first (critique's own caveat)
6. Gu et al., S4D (2022); Smith et al., S5 (2022) — diagonalization precedent + "why not S4D" answer
7. Mullis & Roberts (1976) + Moore (1981) — realization-theory foundation for vii
8. de Vries & Principe (1992) gamma memory; Juan/Harris/Principe (1997–99) Laguerre chip — memory-operator lineage
9. Du et al. (Nat. Comm. 2017) + Moon (2019) — memristor-RC positioning
10. AN231E04 datasheet + User Manual (Okika) — runtime-reconfiguration existence proof

## 5. Caveats

8 searches is not a literature review. "No counterexample found" (items i-e, Missed-2) means *searches didn't surface one*, not that none exists — the recommended week with Google Scholar ("FPAA + neural network," "analog state-space realization sensitivity," "physics-aware training analog") still stands. Exact RASP/Hasler numbers (accuracy range, sigma values, 23 µW) must be pulled from the papers before citation. And the critique's two warnings are endorsed: run E0/E3 (pruning dividend) before writing sparsity novelty claims, and do the reading pass before any claim enters the LaTeX template now open.
