# Analog-Native RNN Features & Gap List for State-Space / RNN Analog Hardware

**Created:** 2026-09-18
**Status:** research synthesis, 10 web searches, verified against live sources
**Premise:** reframe from "defend occupied niches" to "find features only analog can do cheaply." Plus: every gap that can be exploited, ranked for the neural-equalization setting where input and output stay analog (no data-converter tax).

---

## 0. What "analog-native" means (the design rule)

A feature is analog-native if the physics does the work for free:

- **Summation:** KCL at a node. Zero devices, zero power. (This is already KirchhoffNet's founding trick.)
- **Integration:** charge on a capacitor. A timescale is a capacitor; multi-timescale is free.
- **tanh/saturation:** a diff pair saturates; rails clip. The nonlinearity is not computed, it happens.
- **Delay:** propagation through a wire, transmission line, or bucket-brigade. Memory without feedback.
- **Noise/mismatch:** thermal noise and Pelgrom mismatch arrive with the fab. They can be signal (dither, free random projections) instead of error.
- **Multiplication:** translinear/Gilbert cells multiply currents; log-domain turns products into sums.
- **Settling/equilibrium:** the circuit finds its own fixed point at the speed of physics; no clock, no iteration counter.

Any RNN feature that maps onto one of these lines is cheap in analog and expensive in digital. Everything below is scored against this rule.

---

## 1. New RNN features that are naturally analog-native

### F1. Time-multiplexed delay-loop virtual nodes (electronic analog, trained, at line rate)

**Idea:** one physical nonlinear node + one delay loop emulates N virtual nodes by time-multiplexing. No dense A matrix exists at all — topology cost collapses from O(N^2) to O(1) plus delay.

**Prior art (verified):** delay-reservoir computing is well-established but almost entirely photonic (Borghi 2021 silicon microring, Tesfaye 2026, Jeon 2025 Nature single-node-to-10-virtual-nodes at 89%). Electronic versions exist but are toy-scale or mixed-signal: Appeltant's thesis used an analog Mackey-Glass node with a *digitally implemented* delay; Soriano/Ortin mixed analog node + digital delay; Zhao 2016 built an analog delay node for spikes; AIP Advances 2023 demoed Mackey-Glass on Arduino. All use a *fixed* Mackey-Glass nonlinearity.

**The gap:** (a) a *fully-analog integrated* electronic delay reservoir where the delay itself is analog (transmission line, SAW, switched-capacitor, bucket-brigade); (b) a *trained/learned* delay operator instead of fixed Mackey-Glass — the mask, the node bias, and the feedback gain learned end-to-end; (c) applied to a real signal-rate task (equalization), not classification toys.

**Why it fits you:** it deletes your dense-mixer problem structurally rather than by pruning. Your aux-core program ("learn the LDN cheaply") and this ("don't build the matrix at all") are the two complementary answers to the same cost question, and a paper containing both — learned-operator vs no-operator — is a strong story.

**Risk:** delay must equal N x theta with precision; temperature drift of delay moves virtual-node alignment. This couples to drift calibration (your niche vi), which becomes a feature of the story, not a side quest.

### F2. Companding / log-domain state elements (direct answer to your vii)

**Idea:** make each state variable a log-domain integrator (Adams/Frey class). Compression happens in the device physics: large signals are squeezed logarithmically into the linear range, then expanded at readout. State SNR balancing across polynomial orders becomes a circuit property, not a training objective.

**Prior art (verified):** log-domain/companding filter literature is mature circuits work — Frey 2001 (syllabic companding, wide dynamic range at low power), Kolios 2010 (maximized-DR log-domain), Serra-Graells 2004 (1-V log-companding), Kircay 2006 (allpass log-domain). All of it is *filters*, none of it is *recurrent memory operators* or RNN states.

**The gap:** nobody has built LDN/HiPPO-style polynomial memory out of companding integrators, and nobody has reported state-SNR-vs-order measurements for a companding reservoir. Your vii observation (x_5, x_6 collapse vs x_0, x_1) is the motivation section; the companding state element is the answer.

**Why it fits you:** subthreshold OTAs (which you already use) *are* the log-domain device. You may already be one bias regime away from this feature. And it composes with F5 (Laguerre): an allpass cascade in log-domain is wide-DR *and* matching-friendly.

### F3. Per-state AGC loops = analog LayerNorm (second answer to vii)

**Idea:** hang a slow AGC loop (peak detector + VGA, textbook SerDes/AGC blocks) on each state variable or each polynomial-order group, regulating every order to the same RMS. This is LayerNorm/RMSNorm implemented as continuous-time feedback — no digital statistics, no batch, no clock.

**Prior art (verified):** AGC is everywhere (Jung 2022 neural-recording IC with AGC AFE; endless SerDes AGC loops; 5-bit digitally-controlled AGC with 64 dB range). Per-neuron AGC *as a normalization layer inside a recurrent network* is not a known ML feature.

**The gap:** formalize it (loop dynamics vs state dynamics timescale separation, stability of the coupled system), show it fixes vii on LDN orders, and report what it costs (one peak detector + VGA per regulated group — cheap against the OTA array). Bonus: the AGC control voltages are themselves slow features of signal statistics — free auxiliary outputs.

### F4. Fractional-order (fractance) memory — power-law history in few states

**Idea:** replace integer-order integrators with fractional-order ones (order alpha in (0,1)). A fractional integrator has a power-law memory kernel t^(-alpha) — long memory with *few* states, natively. In analog this is a fractance (Carlson/Valsa RC ladder approximations) — a passive network, no active devices for the memory itself.

**Prior art (verified):** fractional-order ESNs exist *algorithmically* — Yao 2024 (fractional-integer-order ESN, cited 26), Dong 2024 (fractional-order fuzzy ESN), SP-FIO-ESN 2026. All are discrete-time digital algorithms. Analog fractional-order circuits exist (fractance ladders) but have never been used as RNN reservoirs.

**The gap:** an analog fractance reservoir — the memory kernel is physics, the readout is trained. This is arguably the most "new feature" candidate on this list: a memory mechanism with no integer-order equivalent, unrealizable efficiently in digital (fractional differences need full history buffers — O(T) memory!), but cheap in analog (an RC ladder). The digital Mustererkennung: fractional needs history; analog gets it from the ladder's physics. That asymmetry *is* the paper.

**Why it fits equalization:** channels have long tails (reflections, skin effect ~ fractional-order physics themselves — skin effect is literally half-order!). A fractional memory kernel may match channel physics better than integer taps. There is a beautiful symmetry argument here: the channel is fractional-order, so the equalizer should be too.

### F5. Laguerre/allpass-cascade basis with learned pole (matching-friendly LDN)

**Idea:** Laguerre memory = first low-pass + cascade of identical all-pass sections. One repeated pole. In analog, identical repeated structures match far better than Legendre's distinct-pole coupling matrix. Learn the pole location (single scalar!) plus readout.

**Prior art (verified):** Juan/Harris/Principe 1997-1999 *fabricated* adaptive gamma and Laguerre filters in analog VLSI (4-tap, system ID task). Quote: "we believe that a cascade of all-pass filters (called the Laguerre filter) will potentially outperform the gamma." Gamma/Laguerre analog is done at 4-tap scale for system ID — not as trained RNN memory inside a programmable fabric, not for equalization, not at LDN orders.

**The gap:** learned-pole Laguerre memory at d = 6-12 inside your fabric, with mismatch-sensitivity comparison against Legendre-LDN on the same fabric. Prediction: Laguerre degrades more gracefully under mismatch because the error source is one pole, replicated. That measurement (same fabric, two bases, mismatch sweep) is publishable on its own.

### F6. Mismatch-as-projection: the fab gives you W for free

**Idea:** don't program the recurrent weights at all. Fabricate the array, *measure* the mismatch-realized matrix once per die (or accept it blind), and train only biases + readout. Fixed random projections are provably good enough for reservoirs (ESN theory); mismatch is a free RNG with zero programming cost, zero DACs, zero drift-with-reprogramming.

**Prior art:** physical-reservoir concept generally; photonic people exploit fabrication variation. For OTA fabrics specifically with *measured-then-frozen* mismatch matrices + trained readout, prior art is thin.

**The gap:** (a) quantify how good mismatch matrices are as reservoirs (MC/PR vserc drawn digital matrices); (b) per-die readout training (calibration-as-training — merges your ii and vi); (c) cross-die generalization: one readout for all dies (calibration-free deployment). (c) is the money result: if a single trained readout works across 100 Monte-Carlo dies, you have killed programming, calibration, *and* drift-tracking in one move.

### F7. Analog neural DFE: slicer-in-the-loop recurrent structure, no converters

**Idea:** a DFE is already a recurrent architecture (decisions fed back through post-cursor taps). Replace/augment the tap-delay feedback with a small analog recurrent network operating on slicer decisions and soft residuals — fully analog, decisions from comparators (1-bit, not ADCs), feedback analog. This is *the* native RNN topology of the SerDes world.

**Prior art (verified):** Kim 2023 (neural equalizer "inspired by forward DFE"), Zhang 2023 (nonlinear DFE DNN), Roshan-Zamir 2017 (PAM4 DFE with FIR+IIR taps) — all assume ADC+DSP or are digital. ML-for-SerDes literature is modeling/simulation (2020 paper) or DSP-based receivers (112G DSP SerDes dominates). No-converter fully-analog neural DFE: gap confirmed by absence.

**The gap:** close the loop in analog — comparator decisions feeding an analog recurrent post-cursor canceller, trained with the slicer in the loop (constellation-aware training, straight-through estimator for the slicer). Your CTLE work + twin methodology ports directly.

### F8. Cellular/local connectivity (Chua CNN paradigm) for the fabric

**Idea:** constrain the fabric to locally-coupled cells (nearest-neighbor + global readout). Local wires are short, matched, low-parasitic — the fabric *wants* this. Memory comes from wave propagation across the array (coupled-map lattice dynamics), not dense matrices.

**Prior art:** Chua-Yang 1988 CNN, many chips. As SSM/RNN *memory* fabric with learned local templates + LDN-style readout: underexplored.

**The gap:** learned space-time memory — the array holds history as propagating waves; readout taps space-time points. Compare MC/edges against dense LDN on the same task. Prediction: wins on wire cost, loses on compactness; the Pareto curve is the contribution.

### F9. Passive switched-capacitor / bucket-brigade delay bank (no OTAs for memory)

**Idea:** your 8-tap bank costs 8M routed active voltages. A passive SC delay line (or bucket-brigade) holds the same history with switches + caps — no static OTA power, no mismatch-sensitive active gain in the memory path. OTAs only at the mixer/readout.

**Prior art:** SC delay lines and BBDs are textbook; using one as the *memory bank for a trained analog RNN equalizer* with learned sparse mixer taps is an integration, not an invention.

**The gap:** quantify it: same task, active-bank vs passive-bank vs aux-core, on edges/power/mismatch-sensitivity. This is the cheapest experiment on this list and it directly answers your "make the bank cheaper" fallback from Phase 11.

### F10. Bilinear/second-order state interactions via Gilbert cells

**Idea:** NARMA's sufficient statistics are products (u(n-9)*u(n)). A Gilbert multiplier makes analog products cheap. Add explicit second-order state interactions (bilinear RNN: x' = Ax + N(x,u)) instead of hoping tanh generates them.

**Prior art:** bilinear systems / second-order RNNs (Giles) are known mathematically; analog bilinear RNN reservoirs are thin.

**The gap:** trained bilinear reservoir in analog for a product-heavy task (NARMA, cross-NARMA M=2). The E2-arm-B question ("are product features overkill?") gets an architectural answer: if products are first-class analog citizens, their ROI changes.

### F11. Event-driven / level-crossing sparse-time updates

**Idea:** no clock. States update on level-crossing events; quiet channel segments cost nothing. Level-crossing sampling preserves information with far fewer samples than uniform sampling for bursty signals.

**Prior art:** level-crossing ADCs, event-driven SNNs. As a continuous-time analog RNN with event-gated integration: thin.

**The gap:** energy-proportional-to-activity analog memory. Fits equalization marginally (line is always active) but fits sensor-front-end + wake-up + your "always-on" story well. Lower priority for SerDes; note and defer.

### F12. Learning rule for dissipative dynamics (the RHEL hole)

**Idea:** RHEL/HEB (Pourcel/Ernoult NeurIPS 2025 oral; Lopez-Pastor/Marquardt 2023 PRX; 2026 unification) give exact physical gradients — but only for *non-dissipative Hamiltonian* systems. OTA-C circuits are dissipative. The RHEL authors flag dissipative credit assignment as open (Appendix A.3.1: reversed dissipative trajectories don't encode error signals).

**Prior art (verified):** hot and crowded on the theory side; dissipative case explicitly open.

**The gap:** a learning rule that works on *your* dissipative fabric — e.g., equilibrium-propagation-style contrastive rules on the dissipative steady state, or embedding dissipative dynamics in a larger conservative scaffolding per HEB prescription, or physics-aware training where forward is physical and backward is the twin (you already have the twin!). Your twin methodology *is* a defensible position in this debate: "we do physics-aware training with a calibrated device-physics twin" — cite HEB/RHEL, position the twin as the engineering answer while the theory matures. Do not try to out-theory this crowd; out-measure them.

---

## 2. The full gaps/holes list (features + benchmarks + methodology + hardware path)

**A. Feature gaps (from Section 1):** F1 delay-loop virtual nodes (trained, fully-analog, at rate); F2 companding states; F3 per-state AGC; F4 fractance memory; F5 learned-pole Laguerre at order; F6 mismatch-as-projection + cross-die readout; F7 analog neural DFE; F8 cellular fabric memory; F9 passive SC bank; F10 bilinear states; F11 event-driven updates; F12 dissipative learning rule.

**B. Benchmark gaps:**
- B1. No multi-input analog-RNN benchmark suite (you noted this). Cross-NARMA M=2 is one line of generator code; MIMO-channel models from your CTLE domain are the in-domain version. Nobody has published an analog-RNN benchmark with M>1 + PVT + mismatch reporting.
- B2. No PVT/mismatch robustness benchmark for analog RNNs. Digital ML has ImageNet-C; analog RNNs have nothing. A "NARMA-C" (corrupted/PVT-shifted evaluation suite with pinned seeds, washout protocol, held-out scoring) would be used by everyone and cited by everyone.
- B3. In-sample MC reporting is rampant (you noted). A methods paper enforcing held-out MC + cost-matched baselines would be a service contribution with teeth.

**C. Methodology gaps:**
- C1. Cost-matched baselines (your E0 — still unrun). Nobody in reservoir computing compares against equal-cost compressed banks. This alone is a methods contribution.
- C2. Constellation-aware / slicer-in-the-loop training for analog equalizers. Digital ML equalizers train on MSE; the true objective is BER after slicing. Training through the slicer (STE) with analog dynamics is thin.
- C3. Stability certification for trained analog RNNs (Lyapunov/passivity via circuit structure). Reviewers of analog RNN papers always ask "but is it stable?"; a certificate from port-Hamiltonian structure preempts the question.
- C4. Calibration-free cross-die generalization measurement (see F6c). Nobody reports "one readout, N dies" for analog RNNs.

**D. Hardware-path gaps:**
- D1. Open PDK tapeout path exists NOW (verified: TinyTapeout analog on sky130A + ihp-sg13g2 + gf180; TT05 had "neural network on chip"). Most analog-ML papers never tape out. A small test chip (one OTA array + SC bank + readout, ~1-2 mm²) through TinyTapeout analog would separate you from every simulation-only paper in this space. Cost is ~hundreds of dollars, not hundreds of thousands.
- D2. Discrete-quantized biasing as a fabric design point (your proposal): neither floating-gate-continuous (Hasler) nor fixed-bias (BrainScaleS-ish) but DAC-quantized + trained-through-quantization (STE). Unoccupied as a *studied* design point.
- D3. Mixed-signal DFE + analog reservoir hybrid: comparators + digital tap feedback for the linear post-cursor part (where digital is exact), analog reservoir for the nonlinear residual. Best-of-both partitioning is an architecture paper.

---

## 3. Exploitability ranking for YOUR setting (analog-in/analog-out equalization, no converter tax)

| Rank | Item | Why it wins here | Cost to test |
|---|---|---|---|
| 1 | F7 analog neural DFE | Native topology of the domain; slicer decisions are 1-bit (no ADC); closes your CTLE loop into a full receiver story | Medium (needs slicer-in-loop training infra) |
| 2 | F2 companding states + F3 per-state AGC | Directly answers vii; subthreshold OTAs already log-domain; AGC blocks are textbook SerDes | Low (simulator + one cell redesign) |
| 3 | F5 learned-pole Laguerre | One scalar to learn; matching-friendly; same-fabric Legendre-vs-Laguerre mismatch comparison is a clean paper unit | Low (basis swap in existing harness) |
| 4 | F1 trained delay-loop virtual nodes | Deletes the dense matrix structurally; pairs with aux-core work as the two answers to one cost question | Medium (needs analog delay model) |
| 5 | F6 mismatch-as-projection + F6c cross-die readout | Kills programming + calibration + drift in one measurement; PVT benchmark B2 falls out as byproduct | Low (Monte-Carlo harness exists in spirit) |
| 6 | F4 fractance memory | Highest novelty; channel physics (skin effect ~ half-order) matches kernel physics; digital can't do it cheaply | Medium-high (needs fractance ladder models + validation) |
| 7 | B2 NARMA-C robustness suite + B1 multi-input bench | Service contribution; cited by all later work including yours; cheap | Low |
| 8 | D1 TinyTapeout test chip | Separates you from all simulation-only work; forces honest area/power numbers | High (layout, but shuttle is cheap) |
| 9 | F9 passive SC bank | Cheapest experiment; answers Phase-11 fallback quantitatively | Very low |
| 10 | F12 dissipative learning rule | Crowded theory space; your twin is already the engineering answer — position, don't compete | Low (writing + citation posture) |
| 11 | F8 cellular fabric; F10 bilinear; F11 event-driven | Real but narrower; defer until 1-6 land | Varies |

---

## 4. Recommended attack sequence (combines your quantized-OTA + pruning plan with the above)

1. **F9 first (weeks, simulation-only):** passive-SC-bank vs active-bank vs aux-core on the pinned twin. Sets the honest floor for every memory-cost claim. If the passive bank wins, your pruning program inherits a cheaper reference — good either way.
2. **F5 second (weeks):** learned-pole Laguerre at d = 6-12 in the existing harness + mismatch sweep vs Legendre-LDN. Decides the basis question before you tape out or prune anything.
3. **F2/F3 third (1-2 months):** companding state cell + per-order-group AGC in simulation; state-SNR-vs-order measurement is the vii deliverable. If it works, the OTA-linearity requirement (your iii) relaxes — re-scope the fabric cell.
4. **Your core program fourth (2-3 months):** discrete-quantized OTA biasing + PVT-injected BPTT + device-constrained pruning, applied to the F5 winner basis. This is the train-prune-realize pipeline from the previous report, now with the basis and cell decisions made by data, not taste.
5. **F7 fifth (parallelizable):** slicer-in-the-loop analog neural DFE on the CTLE channel models. This is the application paper; it needs 1-4's fabric but can start with behavioral models now.
6. **B1/B2 throughout:** publish the benchmark + robustness suite as you go; every leg above contributes rows to it.
7. **F4 as the high-risk parallel thread:** one fractance-leg prototype in simulation; promote only if it beats F5 on long-tail channels.
8. **D1 when 2-4 converge:** TinyTapeout analog test chip of the pruned fabric. The tapeout is the moat — nobody else in the ML-analog-memory conversation has silicon on an open PDK.

## 5. What to stop saying, what to start saying

- Stop: "first reconfigurable fabric," "first mismatch-aware," "first LDN pruning," "RNN that learns its own memory."
- Start: "the first *train-prune-realize* pipeline for continuous-time analog memory operators under device physics and discrete quantization"; "companding/AGC-regulated polynomial states"; "learned-pole Laguerre memory with same-fabric mismatch comparison"; "fully-analog neural DFE with no data-converter tax"; "fractional-order analog memory matching channel physics."
- The converter-tax point is genuinely yours to own: everyone else's neural equalizer pays ADC/DSP power; an analog-in/analog-out fabric sidesteps the dominant power term of the DSP-based SerDes receiver. Lead with system power, not with operator novelty.

---

## 6. Verification caveats

10 web searches is not a literature review. Every prior-art claim above is a lead: confirm paper titles, authors, and venues before citing. In particular verify: Jeon 2025 Nature single-node reservoir; Yao 2024 fractional ESN; Kim 2023 neural wireline equalizer; Juan/Harris/Principe 1997-1999 adaptive Laguerre chip; TinyTapeout analog PDK availability at time of writing; Jung 2022 AGC neural-recording IC. The *structure* of the gaps (delay-loop, companding, AGC-norm, fractance, Laguerre-pole, mismatch-projection, analog-DFE, cellular, SC-bank, bilinear, event-driven, dissipative-rule) is robust to citation corrections; individual attributions are not.
