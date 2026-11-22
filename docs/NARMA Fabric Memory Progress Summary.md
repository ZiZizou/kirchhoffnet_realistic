# NARMA Fabric Memory Progress Summary
Created: 2026-09-18, 3:11:12 p.m.\
Last Updated: 2026-09-18, 5:04:49 p.m.

---

### User

Problem statement: learning memory instead of being handed it
The 8-tap delay bank works — that is not in dispute. Across two campaigns and two maps, removing it collapses memory capacity from ~9–13 to ~3–5 and fails every gate, while keeping it passes. It earns its keep.
The problem is what it costs when reality scales. The bank's implementation complexity explodes with input count: M inputs demand 8M full-rate routed voltages plus a nodes×8M dense mixer — static, unprunable, unoptimizable wiring that grows multiplicatively while contributing zero intelligence. It is memory you are handed, not memory the network holds.
That distinction matters because a recurrent network that cannot generate and store its own memory is not really recurrent — it is a feedforward expander riding on an external shift register.  Our measurements say exactly this is the current state: the network supplies a modest ~3–5 units of genuine recurrent memory; the bank supplies the rest. As long as theInput history arrives pre-packaged, we cannot tell whether the core is weak or merely unemployed — and no amount of tuning the core will answer that while the bank does its remembering for it.
So the task: replace the static bank with a small learnable aux core — input→aux→main, strictly feedforward — that generates its own temporal features from raw inputs at sub-bank edge cost, prunable where the bank is not. Success is matching the bank's memory with fewer edges and features the bank cannot express; failure modes are pre-registered (delay-line copy, dead aux, no gain over scalar drive). If the aux cannot beat being handed the past, the program learns that the hard way early and cheaply — before any silicon is drawn. Ask me questions if unclear about any details

[narma-progress-summary.md](https://upoevdcxa3.ufs.sh/f/IN4OjmY4wMHBy5AjHWEh29deW38Po5BzZFxwZOJELBU9dzQyAcE7z9BT6hpUX4vpSn0)
[README.md](https://upoevdcxa3.ufs.sh/f/IN4OjmY4wMHB39HQtKtqDAT7k50Hyn19gjaZMegRXqkSVlDLAT5jB1be2tUCpI48mAV)

---

### Assistant (kimi-k3)

<details>
<summary>Reasoning</summary>

The user presents a problem statement about replacing the static 8-tap delay bank with a learnable aux core. But the documents they just pasted show Phase 10 and Phase 11 already did exactly this: the aux-core-drive probe tested echo_state, shift_register, slow_feature, residual_distilled aux cores — all KILL. The delay bank stays. Phase 11 even extended with concat readout, sparse fan-out, tapsVD4, D4 distilled.

So the "problem statement" describes a task that has already been executed and closed. The user says "Ask me questions if unclear about any details." My instruction says cut fluff, challenge, point out deficiencies.

Key point: The problem statement is stale relative to the progress summary. The aux-core-drive probe (Phase 10) and follow-up (Phase 11) already ran exactly this experiment at the twin level and all flavors KILLED:
- shift_register matched taps8 MC (11.40 vs 11.47) at edge factor 1.17 but was tap-subspace copy R²=1.000 → it's just an expensive taps8- echo_state borderline tap-copy, MC insufficient
- slow_feature underperformed
- D1 concat readout gains evaporated held-out
- D2 sparse fan-out32 edges still copy
- tapsVD4 incompressible below 8 taps
- D4 residual distilled: what main misses is more delays

So the productive output was negative knowledge. The recommendation was: make bank cheaper (fewer taps, sparser fan-out — both unrun), not replace it.

I should ask pointed questions:

1. This task was already run and killed — Phase 10/11. Is the user aware? Is this problem statement from before those results, or do they dispute the KILL verdicts? The main dispute avenue: nothing was ever actually *learned* (all aux params frozen at hand-picked values / selected by MC or teacher R²). The spec said "learnable aux core" but only offline selection was done — no gradient training of the aux on the actual task. That's the genuine gap. The user's problem statement says "learnable aux core" — the probe tested frozen aux cores. So the real question: is the intended task a *trained* aux core (end-to-end gradient through aux→main→readout, or aux with its own objective)? That was explicitly named as gap (a) in Phase 11's "why a follow-up was opened" but D1-D4 addressed other things; D4 used teacher-R² selection, still not gradient training on task loss.

2. Edge cost accounting: the statement says "sub-bank edge cost" — but the shift-register already beat the bank on edges (1.17×) and MC, and died on the copy rule. So the pre-registered failure mode "delay-line copy" kills the very thing that would work. Is that rule right? If the network *learns* a delay line, that's learned memory — arguably success, not failure. The copy rule conflates "the bank is needed" with "the aux is worthless." A learned shift register at32 edges vs 200 static edges is a win on the stated cost metric. Why is copying a kill? Because it's "parameterised taps8 with extra cost" — but D2-SR was 32 edges, *cheaper*. Hmm, the document says D2-SR at 32 edges "dies on rule (i) exactly as pre-registered." So the copy rule killed a cheaper learned delay line. That contradicts the stated goal ("memory the network holds" vs "handed"). A learned delay line IS held memory. The kill rule seems philosophically inconsistent with the problem statement. Worth challenging.

Actually wait — the tap-subspace R² measures whether aux states are predictable from the taps bank. R²=1.0 means the aux is a linear copy of the bank. But the goal was to replace the bank. If the replacement is functionally identical but cheaper and learnable/prunable, that's the win condition per the cost argument. The kill rule exists to prevent "expensive taps8" — but at 0.16× edge cost it's not expensive. There's tension.

Counter-consideration: a shift-register aux doesn't demonstrate the *recurrent core* generating memory — it's still a delay line, just a smaller learned one. The user's framing: "memory you are handed, not memory the network holds" — a learned shift register is held, not handed (weights learned). But it doesn't show nonlinear temporal feature generation "features the bank cannot express" — success criterion includes "features the bank cannot express." A pure delay line expresses nothing the bank can't. So rule (i) encodes that half of the success criterion. Fair — but then the success criteria are jointly nearly impossible: match bank MC AND be non-copy. Only something with genuine nonlinear memory could pass, and echo/slow-feature failed on MC.

3. Training: the biggest gap. All aux cores were frozen/eval-only with ridge readouts. The problem statement says "learnable." Nothing was learned via gradient descent on task loss. An aux trained end-to-end (or with auxiliary predictive objective like next-input prediction) might develop non-delay-line features (e.g., nonlinear products, the u(n-9)·u(n) term precursors). That's the honest open experiment. Question: is the intent to now train the aux with gradients? What objective — task loss through main (BPTT through Heun twin?) or an aux-local objective?

4. The "unemployed core" argument: "no amount of tuning the core will answer that while the bank does its remembering for it." Actually scalar-drive runs did answer it: core holds ~3–5 genuine MC. So the core's genuine memory is measured. Question: what's wrong with that answer?

5. Multi-input scaling claim: 8M routed voltages and nodes×8M mixer — true, but the aux alternative has its own scaling: aux of width W per input = M×W aux nodes plus recurrent edges. Is the budget formula fair at M>1? tapsVD4 showed incompressibility to 4 PCs — that was for M=1; cross-input structure might compress better with multiple inputs, or worse.

6. Held-out evaluation: Phase 11 noted all MC numbers are in-sample fit-and-scored, optimistic. Any future probe should score held-out. Question: will the new probe fix that?

7. What specifically failed and what would be different this time? The problem statement reads like pre-Phase-10 motivation. If this is Phase 12 (trained aux), the delta over Phase 10/11 must be stated: gradient training, held-out scoring, maybe revised copy rule.

So my response: point out that the problem statement describes a closed experiment; ask the sharp questions. Keep it tight, plain, challenging. No need for tools. Format: short intro line identifying the staleness, then numbered questions grouped. Keep reasonably concise.
</details>

This problem statement describes an experiment that has already run and closed — twice. Phases 10 and 11 built exactly this probe (echo-state, shift-register, slow-feature, residual-distilled aux cores, strictly feedforward, edge-budgeted, with the pre-registered kill rules you list verbatim), and every flavor emitted KILL. The bank stays. So before questions on details, the framing needs to catch up with your own results. What follows assumes you're reopening it deliberately.

**1. What is the delta over Phase 10/11?**

The one genuine gap those phases named and did not fill: nothing was ever *trained*. All aux cores were frozen at hand-picked hyperparameters or selected by offline teacher-R². Your statement says "learnable aux core," but no gradient has ever touched an aux on task loss. If Phase 12 is "trained aux," say so explicitly, because that is the entire scientific content of the rerun. If it isn't training, you're re-running a closed experiment.

**2. What objective does the aux train on?**

Three options, and they are not equivalent:
- End-to-end task loss through aux→main→readout (BPTT through the Heun flow — the aux's gradient signal is then filtered by whatever the frozen main core passes; a weak main corrupts the aux's training signal).
- An aux-local objective (next-input prediction, or the NARMA-relevant sufficient statistics: `u(n−k)` and product terms like `u(n−9)·u(n)`).
- Distillation against the bank (but that's just learning to be taps8 — which your own rule (i) kills).

Which one? The choice determines whether "learned memory" is even measurable.

**3. The copy rule contradicts your cost argument — resolve it.**

Your statement's complaint is that the bank is *static, unprunable, unoptimizable* and scales as 8M. But Phase 11's D2-SR was a 32-edge learned shift register — 0.16× the bank's edge cost, prunable, trainable — and it died on rule (i) (tap-subspace R² = 1.000). By the cost logic in your own problem statement, a *learned* delay line at 32 edges is a win: it is memory the network holds, not memory it's handed. Rule (i) was written to reject "an expensive taps8," but it fires identically on a *cheap, learned* taps8. Either:

- amend rule (i) to kill only copies at ≥ some edge factor (a copy at 0.16× cost with pruning headroom passes), or
- admit the real success criterion is "features the bank cannot express" and drop the cost framing, because you cannot have both.

Right now the pre-registration is internally inconsistent, and the next probe will inherit that.

**4. What did scalar-drive already answer?**

You write "we cannot tell whether the core is weak or merely unemployed." The scalar-harness runs already told you: the core supplies ~3–5 units of genuine recurrent memory in both maps, both campaigns. The core is weak *and* measured. The open question is narrower: can a trained aux turn raw input into features worth more than ~5 MC units without recapitulating a delay line? State it that way or the motivation oversells the ignorance.

**5. Held-out scoring.**

Phase 11's own caveat: every MC number in the document, including taps8's 11.47, is fit-and-scored in-sample and therefore optimistic, and the D1 concat "gain" evaporated held-out. Will the new probe score ridge/MC on a held-out stream as the primary metric? If not, you'll be comparing a trained aux against an inflated bank reference — or worse, beating it by overfitting.

**6. The echo-family threshold fragility.**

Echo rows land at tap-R² 0.86–0.92 on the autocorrelated stream, so the 0.90 KILL line cuts through the family and flips by seed. A trained echo-state aux will start somewhere in that band. Are you keeping the 0.90 line, moving it, or replacing it with something seed-stable (e.g., R² under white-noise input, where the measured value was 0.76)?

**7. Multi-input scaling claim.**

The 8M/nodes×8M explosion is the cost motivation, but it was never tested — every campaign was M=1. An aux alternative has its own M-scaling (aux nodes and recurrent edges per input). Do you have any M>1 measurement, or is the scaling argument purely analytical? If analytical, the probe should include at least one M=2 or M=4 leg, because that's the regime where the bank supposedly loses and the aux supposedly wins — and it's currently unmeasured on both sides.

In short: the experiment you're describing is worth doing only if it is (a) trained, (b) held-out scored, and (c) run under a copy rule that doesn't kill the cheapest passing answer. Which of those three are you changing?


---

### User

Do you think training might fix the gap ? I guess the point was to only train if we were able to close the gap through init within a reasonable range. Yes, I think the aux needs its own objective - it needs to be able to store some low rank approximation of history - I am not sure that simple backpropagation to reduce the error of the final output is enough to encourage it to learn anything different from the main core. I think an aux local objective where we try to predict only past outputs from its nodes might be enough encouragement for it to store information. Yes, we must amend the rules that allows passing a network that is cheaper than basic taps. I didnt really test it on any synthetic test with more than 1 input because the MC itself was not comparable (also I do not know any synthetic rnn benchmarks that have more than 1 input) - and I feel like the shift register is not "different"/"innovative" enough from a tapped delay line - it wasnt low rank enough for my liking - maybe we can do a sweep where we intentionally also sweep number of nodes in the aux core shift-register/ring and see when performance collapses (min nodes required to hold 8 nodes of information)? I will try training and implement the better held out scoring.


---

### Assistant (kimi-k3)

<details>
<summary>Reasoning</summary>

The user is responding to my questions. Let me parse their answers:

1. Training might fix the gap — but their E0 philosophy is screen-at-init, train only if init is close. They're now willing to try training anyway.
2. Aux needs its own objective — they propose an aux-local objective: predict past inputs from aux nodes. They doubt end-to-end task loss would differentiate aux from main.
3. Amend copy rule to allow cheaper-than-bank copies.
4. Multi-input: they don't know multi-input RNN benchmarks; MC not comparable.5. Shift register not "low-rank enough" — propose sweeping aux node count (ring/shift-register width) to find minimum nodes to hold 8 taps of information.They plan to: train, implement held-out scoring, sweep aux width.

I should respond critically and efficiently, per their persona instructions (challenge, no fluff, point out deficiencies).

Key technical points to make:

**A. The aux-local objective idea — "predict past inputs from aux nodes" is just memory capacity as a loss.** That's literally training the aux to maximize MC of its own states. Two problems:
- This objective's optimum IS a delay line (or a tapped basis of past inputs). If you train aux states to linearly reconstruct u(t−k) for k=1..K, you're training it to be a bank. So rule (i) amendment is not just needed — the objective and the copy-rule are in direct tension. You'd be training toward the thing you're killing. Need to think about what objective encourages "features the bank cannot express."
- Alternative: objective = predict past inputs AND nonlinear products of them (the NARMA-sufficient statistics: u(t−9)·u(t), IPC-style Legendre/product terms). That forces the aux to store history in a *mixed, compressed* form that a pure delay line can't achieve with few nodes — actually wait, a delay line CAN express products downstream via readout, but if the aux itself must be linearly read to produce products, then the aux needs nonlinearity. So: aux-local objective = linear ridge from aux states to {u(t−k), u(t−i)u(t−j)} targets. A shift register achieves the linear part with zero error but can't express the products linearly. This gives the aux a reason to be more than a bank. This is a concrete, cheap proposal. The readout for the objective is training-only; the aux→main path stays feedforward.

Also worth noting: their instinct that end-to-end BPTT won't differentiate is reasonable but the literature (e.g., reservoir computing with trained input masks, "conceptors," deep reservoir training) — the standard finding is that gradients through a fixed reservoir do adapt input layers usefully. But aux-local objectives are cheaper and more interpretable. Actually there's a cleaner argument: end-to-end loss only rewards the aux for information the main core *doesn't already extract* — which is exactly what they want ("different from the main core"). The concern that backprop won't encourage it to learn anything different from main core is actually backwards: end-to-end training automatically pushes aux toward residual information (what main misses). Phase 11's D4 (residual-distilled teacher) showed what main misses is... more delays. So both objectives may converge to delay storage. That's a prediction, not a dealbreaker.

Hmm, actually their concern: "not sure simple backprop to reduce final error is enough to encourage it to learn anything different from the main core" — mathematically, gradient descent on final loss gives the aux exactly zero gradient for duplicating main's features (they contribute nothing to loss reduction), so it does encourage difference. The real worry is the opposite: weak gradient signal / poor conditioning through25-node tanh core + Heun steps. And D4's evidence suggests the residual IS delays.

**B. "Train only if init closes the gap" philosophy — inconsistent application?** They screened frozen auxes and found KILL, and now will train anyway. That's fine but they should note the E0-gate logic from the NARMA campaign: training historically bought ~0.05-0.085 NRMSE via saturation. The aux case is different though — aux training is unconstrained digital training (no analog device params yet), cheap, and the frozen probe was the screen. Actually wait — the frozen screen said KILL. By their own E0 philosophy, a failed init screen means "do not burn GPU." They're now overriding that. Worth flagging: either the E0-screen philosophy is wrong for auxes (justify why), or this is a violation of their own decision rule. I think the honest justification: the Phase10/11 screen tested *frozen* auxes against a *bank that is itself the optimal frozen aux* — the screen was structurally unwinnable for anything except a cheaper copy. Training is the only axis left untested. But then the pilot gates from Phase 8B style apply: bounded pilot, pre-registered promotion criteria.

**C. Minimum-nodes-to-hold-8-taps sweep.** Their proposal: sweep aux ring/shift-register width to find when performance collapses. Issues:
- The answer is already known analytically for a linear shift register: to reconstruct u(t−k) for k=1..8 with ridge from n node states, you need n ≥ 8 (rank argument — controllability/Krylov span). A linear recurrent system driven by scalar input has state spanning at most the Krylov subspace of dimension min(n, controllability rank); to hold 8 independent delayed copies you need ≥8 states. Below8, MC over the 1..8 window degrades gracefully by filter shape. So the sweep will just rediscover rank. UNLESS the aux is nonlinear, in which case "information" isn't rank-limited the same way — but for linear readout of linear targets it still is (readout can only extract what's linearly present).
- So the interesting sweep is not "min nodes to hold 8 taps" (answer: 8, by linear algebra) but "min nodes to hold the *NARMA-sufficient statistics*" — the product term compresses: you don't need all 8 taps, you need u(t), u(t−9), and their product-ish features, plus the y-history which the main core's output feedback... actually NARMA-10 y depends on y(n)...y(n−9) too, which is a recurrent target history. Hmm, in their harness the target is generated autonomously; the net only sees u. The y-history dependence means the reservoir must integrate u history in a specific nonlinear way — the 0.3 y(n) term etc. This is why NARMA needs MC ~12 (they say task needs MC ~12, ESN 0.459... whatever, their numbers).
- Point: frame the width sweep around *task-sufficient compression*, not tap storage. If aux width 4 with trained nonlinear dynamics reaches MC ≥ ~9 total on task stream, that beats the bank's cost by a lot even if it can't reconstruct all 8 taps. The right metric is held-out task NRMSE + MC, not "holds 8 nodes of information."

**D. Multi-input benchmarks.** They said they don't know any multi-input synthetic RNN benchmarks. I can name some: 
- Multi-input NARMA (drive with vector u, target depends on multiple lags of each and cross-products — trivially constructible; NARMA generalizes).
- Sunspot/multivariate time series (real data).
- Lorenz/Santa Fe laser (scalar though).
- Mackey-Glass multi-tap.
- Sum-of-sines / wireless channel equalization (MIMO!) — actually CTLE/SerDes equalization is inherently multi-input multi-tap (their own domain! CTLE with PAM-4, crosstalk aggressors = multiple inputs).
- Also simple: independent NARMA per input, cross-product NARMA (y depends on u1(t−9)·u2(t)).
- Copying task / adding task (classic Hochreiter) can be made multi-input: adding task is 2-input (marker channel + value channel)! The adding problem is literally a 2-input benchmark. Also the multiplication task. Those are canonical RNN benchmarks with M=2.

So: "I don't know any multi-input benchmarks" is a gap — the adding task, the multiplication task, MIMO channel equalization (their own CTLE context), and cross-NARMA all exist. But also validate their instinct: MC itself is scalar-input-defined; for M inputs you'd define per-input MC or joint MC. That's fine — Jaeger's MC generalizes per-channel.

**E. Held-out scoring** — good, endorse, but note specifics: washout alignment between fit/score streams, same washout=200, and refit protocol (fit ridge on train stream, score on fresh stream with its own washout). Also per-seed R² reporting for the copy rule as I noted before.

**F. Rule amendment specifics.** Concrete proposal: kill rule (i) becomes "copy at cost": if tap-subspace R² ≥ 0.90 AND edge factor ≥ 1.0 → KILL; R² ≥ 0.90 with edge factor < 1.0 → CONDITIONAL PASS if task metrics hold. Actually cleaner: rule (i) should be reformulated as "the aux provides nothing the bank provides at lower cost" — i.e., compare against the *equal-cost bank*, not the200-edge bank. An aux at 32 edges should be compared to a 32-edge compressed bank (tapsVD4 at 100 edges failed the 0.85 gate; what's the equal-32-edge bank's MC? unmeasured — this is a hole: they compared auxes against the200-edge reference but the fair baseline for cost claims is the *cost-matched* bank, e.g.,32-edge sparse-fanout taps8 or tapsVD-k at k≈1-2). D2-SR at 32 edges died on copy-ness, but was it compared against 32-edge taps? No. That's a real experimental deficiency: the cost argument requires cost-matched bank baselines (fewer taps, sparser fan-out, SVD compression) — which Phase 11 itself listed as the unrun productive direction! "Next memory work should target making the bank cheaper (fewer taps, sparser fan-out — both unrun)". So before training auxes, they owe themselves: taps4, tapsVD-k at matched edges, sparse-fanout taps8. If a 50-edge taps variant holds MC ~10, then the "8M explosion" motivation shrinks a lot, and the aux bar becomes "beat the cost-matched bank," not "beat taps8-200."

That's a strong point to lead with: the stated problem is bank cost, but the cheapest bank-variants have never been measured, and Phase 11 explicitly deferred them. Any trained-aux victory over the full bank is meaningless if a 4-tap bank at 100 edges matches it. tapsVD4 got MC 9.21 vs11.47 at half the edges — that already halves the "explosion" claim and it wasn't even swept (k locked at 4).

**G. Will training fix the gap?** My honest assessment: 
- The gap is bank-supplied memory (~4-7 units of the9-13). Training an aux to store history — a linear dynamical system with trained weights can perfectly implement a near-delay-line (this is trivially learnable: Legendre Delay Networks / LDNs! HiPPO! LMU — Maththew Voelker/Eliasmith work: optimal recurrent compression of a delay window into n dimensions with exact continuous-time dynamics). Important prior art: the *optimal* answer to "minimum recurrent state to hold a delay window" is known — HiPPO/Legendre Delay Network: n states optimally approximate a window of length θ with error decaying in n, and it's linear, provable. The shift register with gain<1 is a crude version; LDN is the principled one. So "learned memory" has a known optimum and it's a linear recurrent system — meaning trained aux will converge toward something LDN-like, i.e., a *better delay line*, not something categorically different. This reframes everything: the aux program is "learn the LDN," and the copy rule should be amended to compare against LDN cost/quality, not to kill delay-likeness. The genuinely-non-bank features (products, nonlinear mixtures) come from the main core anyway — that's the "fabric as nonlinear expander, not memory" verdict of Phase 9. So training will likely close the gap *by becoming a cheaper delay operator*; whether that satisfies them depends on whether they accept "learned LDN at 0.2× edges" as a win.
- So answer: yes, training likely closes MC gap (frozen random aux → trained near-optimal linear memory is a well-understood problem; LDN/HiPPO says the ceiling is high), but the result will look like a delay line because delay-line-like operators ARE the optimal linear memory. The "features the bank cannot express" hope should be downgraded — the main core owns nonlinearity.

**H. Their aux-local objective "predict only past outputs from its nodes"** — wait, they wrote "past outputs" — do they mean past inputs? "an aux local objective where we try to predict only past outputs from its nodes" — ambiguous. In context ("store information", "low rank approximation of history") they mean past *inputs* u(t−k). But "outputs" could mean the main core's outputs y? I'll interpret as inputs and note the ambiguity briefly, or actually — predicting past *outputs* y(t−k) would be interesting: NARMA y depends on y-history, so aux storing y-history... but aux doesn't see y (feedforward input-only). So it must be inputs. If they meant past inputs: that's MC-as-loss, and I covered it. Add: include product targets (u(t−i)u(t−j), especially lag-9 product) to push beyond pure delay storage; a pure linear aux can hit the linear targets perfectly (LDN), so products are the only part of the objective that forces nonlinearity. And note: with a *linear* aux + ridge decoder objective, the optimum is exactly the LDN/HiPPO solution — so the sweep and the objective connect.

Also technical caution: aux-local MC objective trained by backprop through the aux dynamics — the aux states are recurrent; training to reconstruct past inputs is exactly what orthogonal/identity-init RNNs and LDNs solve analytically. Gradient will find it or you can just *initialize* at LDN and fine-tune. Cheaper: closed-form LDN init exists (no training needed!) — which again pressures the "learning" framing: the optimal linear memory is known analytically, so a learned aux that converges to it is validating, not discovering.

**I. The contradiction "screen at init, train only if gap closes" vs "I will try training".** Their stated E0 philosophy: failed start = don't burn GPU. Phase 10/11 aux screen failed. Now they train anyway. Defensible only if: the screen was structurally unwinnable (frozen aux vs bank — the bank IS the optimal frozen feedforward memory for this metric, so no random aux passes; the screen tested the wrong hypothesis). Articulate that: the init-screen logic applies when the frozen system class could plausibly contain a good member; for aux-vs-bank the only good frozen members are near-copies, which rule (i) kills — so the screen was guaranteed to return KILL regardless of trainability. Hence training is the only informative test and skipping it on screen grounds is a category error. This gives them the principled override they need. But pair it with a bounded pilot + promotion gates (like Phase 8B), not a campaign.

**Structure of reply:**

1. Answer "will training fix the gap": yes for the MC half, no for the "features the bank can't express" half — because the optimal recurrent memory for reconstructing a delay window is known analytically (HiPPO/LDN); training converges toward a cheaper delay operator. That's a win under the amended cost rule, not under the old copy rule. Cite LDN: n continuous-time neurons compress window θ optimally, closed-form A/B matrices; shift register with gain is its crude ancestor. So the trained aux's ceiling is LDN, and the floor is the Phase-10 frozen echoes.
2. Screen-philosophy inconsistency + why override is justified here (screen was structurally unwinnable) + bounded pilot requirement.
3. Objective design: MC-as-loss = training toward the thing rule (i) kills. Fix: (a) amend rule to cost-conditional copy kill; (b) add nonlinear/product targets to the aux objective, or accept linear aux = LDN; note D4 evidence that what main misses is more delays, so even end-to-end training would push toward delays. Their claim that end-to-end won't differentiate aux from main: gradient is zero for duplicated features — the concern is backwards; the real issue is signal quality through the frozen core, and D4 predicts the residual is delays anyway.
4. Width sweep: rank argument — linear aux needs ≥8 states for 8 taps (Krylov/controllability); sweep will rediscover linear algebra unless targets are the task-sufficient statistics. Reframe: sweep width against *held-out task NRMSE + total MC*, expecting graceful degradation and a knee; LDN theory predicts the knee location (error ~ exp(−c·n) per window length). So the sweep has a quantitative prediction to check — good. Also note LDN stores window in n<window-length? LDN: to represent window of length θ with n dims, error decreases as n grows; n=8 states for 8-sample window is roughly matched; a shift register needs exactly 8 taps for 8 delays, LDN with n=8 can represent the whole window functionally. The interesting regime is n < taps (compression).
5. Missing cost-matched bank baselines — tapsVD-k sweep, fewer-taps bank, sparse fan-out — Phase 11's own deferred direction; without these, beating the 200-edge bank proves nothing about cost. This should be *first*, before training, because it's eval-only cheap and sets the bar.
6. Multi-input benchmarks: adding task (2-input, canonical), multiplication task, MIMO channel equalization (their CTLE domain has crosstalk aggressors — natural), cross-NARMA (y depends on u1·u2 cross-lag products), multivariate NARMA trivially constructible. MC generalizes per-channel. So "no multi-input benchmarks exist" is wrong; the harder truth is MC comparability needs a per-channel definition, which is standard.
7. Held-out: concrete protocol note (separate washout on score stream; report per-seed).

Order by importance: I think (1) LDN/optimal-linear-memory reframe, (5) cost-matched bank baselines, (3) objective tension, (4) sweep reframe, (6) multi-input, (2) pilot gating.

Keep it direct, challenge-y, no fluff. Use headings. Moderate length — this is dense technical feedback; maybe 600-900 words. No horizontal rules. Use minimal math formatting.

Let me double-check the LDN claim: Legendre Delay Networks (Voelker & Eliasmith, NeurIPS 2019): recurrently compress a time window of length θ into n dimensions; the transfer function approximates e^{−θs} optimally (Padé-ish/Legendre); error for window length θ with n dimensions scales favorably; used in LMU. HiPPO (Gu et al.) is the continuous generalization. Yes, this is solid prior art: "learning memory" for a delay window has a known closed-form optimum, and it's linear recurrent. A discrete-time version: the optimal linear recurrence for storing a window of W samples in n state dims is given by the best rank-n approximation of the shift operator's Krylov structure — roughly, you can hold W > n with error growing in W/n; exact for W ≤ n.

For the rank argument: a linear time-invariant aux with scalar input and n states, read out linearly, can produce any impulse response in the span of the Krylov sequence — the set of reconstructible delayed inputs u(t−k) for k in window W requires the observability of those delays; generically n ≥ W for exact reconstruction of W independent delays. With noise-robustness, n somewhat > W. Compression below W is possible approximately (LDN-style, smooth input statistics help — NARMA's u is i.i.d. uniform, white — which is the *hardest* case: no temporal correlation to exploit, so compression must store the full window; white input kills the low-rank-of-signal loophole). Important: because u is white, "low-rank approximation of history" can only mean low-rank in the *operator* sense (LDN), not in the signal sense. The signal history is full-rank. So their phrase "low rank approximation of history" — with white u, the history matrix is full rank; the compressibility comes from basis choice (Legendre-like), giving graceful而非exact reconstruction. Good nuance to state.

The NARMA product term u(t−9)·u(t): with W=10 window, exact linear storage needs 10 taps; the bank has 8 (delays 0..7)! Wait — taps8 = u[t..t−7], but NARMA-10 needs u(t−9). So the bank doesn't even directly supply the lag-9 product input; the *core* must hold t−8, t−9 itself. Interesting — actually that's why lag-9 product R² is only ~0.5 and why beyond-tap MC matters. So the aux objective including u(t−9) reconstruction is a genuine beyond-bank requirement. Actually check: tap 0 = current input, 8 taps → delays 0..7. NARMA-10 product:1.5·u(n−9)·u(n). So u(n−9) is NOT in the bank. The recurrent core must supply delays 8–9. And MC beyond tap 8 was the discriminator in Phase 7/8 (banded leaks improved lag-8 memory 0.15→0.25-0.29). So the genuine recurrent-memory burden is precisely the tail beyond the window. An aux trained to reconstruct k=1..12 would shift this burden off the core entirely. Good point to include: size the objective window to 10-12, not 8, else you're training the aux to do what the bank already does while leaving the actual bottleneck (lag-9) untouched.

MC definition sums k=1..20. Bank covers 1..7 "free". So genuine memory = beyond 7.

Also their claim "min nodes required to hold 8 nodes of information" — for white u and exact ridge reconstruction of 8 delays: n=8 linear states suffice exactly (shift register), n<8 cannot be exact (rank), but approximately yes with error growing as W/n. The sweep will show: n≥8 → near-perfect, n<8 → graceful loss concentrated at large k. Prediction given; if the sweep shows otherwise, something's wrong with the harness. To make the sweep informative, measure reconstruction quality per delay k, not just total.

One more challenge: "trainable aux, prunable" — pruning a trained recurrent aux changes its dynamics (temporal, not just spatial) — pruning recurrent weights is riskier than pruning static fan-out; the bank is unprunable because it's wiring, but the bank's *fan-out mixer* (nodes×8M) IS prunable (it's just a matrix). Actually they called the whole bank "unprunable, unoptimizable" — but tapsVD and sparse fan-out ARE optimizations of the bank's mixer. The truly static part is the 8M routed voltages. Fine.

Format: headings + tight paragraphs. Maybe a small ordered list for the recommended sequence. Aim for clarity and directness per persona.
</details>

Yes — training will probably close the MC gap. But it will close it by turning the aux into a better delay line, not into a new kind of memory, and you should decide now whether that counts as a win. Several things in your plan need tightening first.

## 1. "Learned memory" has a known closed-form optimum, and it's a delay operator

You're proposing to learn "a low-rank approximation of history" recurrently. That problem is solved: Legendre Delay Networks / HiPPO give the optimal linear recurrent compression of a length-θ window into n state dimensions, analytically, no training required. A trained linear aux will converge toward an LDN-like operator because that's the ceiling of its function class. A shift register with gain 0.9 is already a crude member of that family — Phase 10 measured it at MC 11.40 vs taps8's 11.47 at 0.16× the fan-out edges once you sparsify. So the realistic outcome of training is: a learned, prunable, ~LDN-quality delay operator at a fraction of the bank's cost. Under your amended cost rule, that's a pass. Under the old copy rule, it's a KILL. The amendment isn't a loophole — it *is* the result this program can deliver. If "the aux recapitulates delay structure, but learned and cheap" still feels like failure to you, the program is asking for something the linear-memory literature says doesn't exist, and you should say that explicitly in the spec.

One sharpening: NARMA's u is white. The history matrix is full-rank — there is no signal-side redundancy to exploit. All compression must come from the operator basis (Legendre-like), which degrades gracefully rather than exactly below n = window length. Set expectations accordingly.

## 2. Your aux-local objective trains toward the thing rule (i) kills

"Predict past inputs from aux nodes" is MC-as-loss. Its optimum is a delay operator — the exact structure that trips the tap-subspace diagnostic. You can't run that objective and keep an unconditional copy KILL; the objective and the gate are adversarial. Two consistent options:

- **Accept the delay nature.** Objective = reconstruct u(t−k), k=1..W. Rule (i) becomes cost-conditional: copy at ≥1× bank edges = KILL; copy at <1× with held-out task parity = PASS. The science claim is then purely about cost/prunability.
- **Force non-bank features.** Add product targets (u(t−i)·u(t−j), especially the lag-9 pair) to the aux readout objective. A linear aux can nail the linear taps and gets zero on products; only a nonlinear aux moves them. This is the only objective variant that could produce "features the bank cannot express" — but note the bank can't express products either, so you'd be asking the aux to do the *main core's* job (Phase 9: fabric as nonlinear expander). Think about whose job you're reassigning.

Also, size the window to at least 10–12, not 8. Taps8 covers delays 0–7; the NARMA-10 product needs u(n−9), which the bank *doesn't supply* — that's why lag-9 product R² sits at ~0.5 and why beyond-tap MC was the discriminator in Phases 7–8. An aux objective that stops at k=8 trains the aux to duplicate the bank while leaving the actual bottleneck (lags 8–9) on the core, untouched.

## 3. Your worry about end-to-end loss is backwards, but D4 supports the conclusion anyway

Gradient through the main core gives the aux *exactly zero* signal for duplicating main's features — duplication contributes nothing to loss reduction. End-to-end training is structurally biased toward residual information, i.e., precisely "different from the main core." The real problems are gradient quality through 25 tanh nodes × Heun steps, and Phase 11's D4 finding: what main misses is more delays. So both objectives — local MC and end-to-end — plausibly converge to delay storage. Run the aux-local objective because it's cheap and interpretable, not because end-to-end would fail to differentiate.

## 4. The width sweep will rediscover linear algebra unless you change its target

"Minimum nodes to hold 8 taps": for a linear aux with scalar white input and linear readout, exact reconstruction of W independent delays requires n ≥ W (Krylov/controllability rank). Below that, error concentrates at large k, LDN-style. That's a prediction, not a question — if your sweep shows anything else, suspect the harness. The informative version: sweep n against *held-out task NRMSE and per-delay reconstruction*, and look for the knee against the task-sufficient statistics, not against "8 taps." n < W with graceful degradation at long lags is the interesting regime; it tells you the cost floor of learned memory.

## 5. You owe the cost-matched bank baselines before any aux training

This is the biggest hole. Phase 11 said it yourself: "making the bank cheaper (fewer taps, sparser fan-out — both unrun)." Your entire motivation is bank cost, but every aux has only been compared against the full 200-edge taps8. tapsVD4 already got MC 9.21 at 100 edges — half the cost, and k was locked, never swept. If taps4-plus-sparse-fanout at ~64 edges holds MC ~10, a trained aux beating the 200-edge bank proves nothing. The fair bar is the *equal-edge bank*. Run tapsVD-k over k=2..6, taps4/taps6, and sparse fan-out on taps8 first — it's eval-only, minutes of compute, and it sets the number the trained aux must beat.

## 6. Multi-input benchmarks exist; you were looking in the wrong shelf

The adding task and multiplication task (classic long-term-memory RNN benchmarks) are inherently 2-input: marker channel + value channel. MIMO channel equalization is multi-input multi-tap and sits inside your own CTLE domain — crosstalk aggressors are literally the M>1 case. Cross-NARMA (y depends on u₁(n−i)·u₂(n−j)) is a one-line generator change. MC generalizes per-channel; you report a per-input MC vector and sum or gate per channel. "MC wasn't comparable" is a definition problem with a standard fix, not an absence of benchmarks.

## 7. On overriding your own screen rule

Your E0 philosophy says a failed init screen means don't burn GPU. The aux screen deserves an explicit exemption, and here's the principled version: the frozen-aux screen was structurally unwinnable — the bank is the optimal *frozen* feedforward memory for this metric, so the only frozen auxes that pass are near-copies, which rule (i) killed. The screen tested "does a random frozen aux beat the optimal frozen aux," which is KILL by construction, independent of trainability. So training is the only informative test, and skipping it on screen grounds is a category error. But honor the spirit with a bounded pilot, Phase-8B style: one flavor (I'd take echo_state, the only family that wasn't a hand-built delay line), 3 seeds, fixed epoch budget, held-out scoring, and pre-registered promotion gates against the cost-matched bank from §5 — not against taps8-200.

On held-out scoring: fit ridge on the train stream, score on a fresh stream with its own 200-sample washout, and report per-seed tap-subspace R² (the echo family sits at 0.86–0.92 and flips by seed — the winner's R² alone is not a stable basis for a KILL).

Suggested order: §5 baselines (cheap, sets the bar) → amended rule (i) + window-12 objective with product terms → bounded trained-aux pilot → width sweep against task metrics → M=2 cross-NARMA leg only if the pilot passes.


---

### User

Generate a document outlining the sequence of next experiments. I think in terms of it acting like an RNN - we want this system to generate its own memory without an explicit tapped delay line - so regardless of whether it copies/emulates a delay line (which seems like it is a good/best architecture) - it is fine (as long as it atleast does it cheaper and has a plausible circuit implementation). As far as generating more useful features - there is a question of overkill - are they really necessary and is the ROI justified ? Additionally it seems like I (we) are rediscovering stuff like CTRNN reservoir computing from first principles - I need to make sure that my work is unique (heun stepping simulations ; parametrizing rnn as an analog circuit ; a framework for training and pruning connections to find the most simple analog implementation of LDN/Laguerre & Chebyshev Networks ; application to neural equalization as a closed loop optimizer - all could be novel enough but I am not sure). Help me brainstorm


---

### Assistant (kimi-k3)

<details>
<summary>Reasoning</summary>

The user wants: (1) a document outlining the sequence of next experiments; (2) brainstorming on novelty/uniqueness of their work relative to CTRNN reservoir computing, LDN/Laguerre/Chebyshev networks, etc.

Novelty claims need checking against literature — that's a good use of web search. I have max 2 web searches. I should search for the relevant prior art: LDN/HiPPO, Laguerre filters in neural nets (e.g., De Vries & Principe gamma model), CTRNN reservoirs, analog/photonic delay-based reservoir computing, neuromorphic RC surveys. Let me plan queries.

Query set 1 (broad): "Legendre Delay Network HiPPO" / "Laguerre network neural delay memory" / "gamma memory De Vries Principe" / "analog circuit reservoir computing delay line silicon" / "continuous-time recurrent neural network reservoir computing". 

Query set 2: "trainable delay line neural network 2023 2024 2025" / "photonic reservoir computing analog" / "OTA-C filter neuromorphic reservoir" / "hardware-aware neural architecture search analog".

Let me do one search call with multiple queries first, then maybe one follow-up for the most relevant hits.

Then produce the document: experiment sequence (E1: cost-matched bank baselines; E2: analytical LDN/Laguerre aux insertion; E3: trained aux pilot with aux-local objective, held-out scoring; E4: width sweep vs task metrics; E5: pruning/anneal to simplest circuit; E6: M=2 cross-NARMA / equalization application; E7: analog realization study). With pre-registered gates, edge budgets, verdicts.

Plus a novelty-positioning section: what's rediscovered vs plausibly novel, and how to check (references to cite, disclaimers).

Format: a document they can save. Use headings, no horizontal rules. Should be reasonably thorough but not bloated. Include file naming conventions consistent with their repo style? They use docs/, outputs_alliance_can/, spec docs. I can write it as a spec-like markdown document.

Let me do the web searches first.
</details>

<details>
<summary>Reasoning</summary>

Good. The search confirms: HiPPO/LMU/LDN literature, gamma model (de Vries & Principe 1992), analog VLSI implementations of gamma/Laguerre filters (Harris, Juan, Principe). This is highly relevant — the user's "aux core that learns memory" idea has direct prior art: gamma memory = cascade of leaky integrators with adaptive μ, in analog VLSI, no less. The multi-scale gamma (ms-gamma) with exponentially spreading time constants is literally their Phase-7 banded leaks idea.

I have one more search available. Should I search for analog reservoir computing / photonic delay-based RC to check the novelty of "parametrizing RNN as analog circuit" and "neural equalization"? Let me use the second call to check: (a) delay-based/photonic reservoir computing implementations, (b) analog neural network equalizers / neural equalization in wireline links, (c) CTRNN neuromorphic. That's important for the novelty positioning section.
</details>

<details>
<summary>Reasoning</summary>

I have enough. Now write the document. Key points to weave in:

Prior art found:
- HiPPO (Gu et al. 2020), LMU (Voelker et al. 2019), Delay Network (Voelker & Eliasmith 2018) — LDN deployed on analog neuromorphic hardware (Braindrop!) and Loihi. So "LDN on analog hardware" exists.
- Gamma model (de Vries & Principe 1992) — cascade of leaky integrators with adaptive μ, trained by RTRL/BPTT, memory depth/resolution tradeoff. Analog VLSI implementations of gamma and Laguerre filters (Harris, Juan, Principe) — fabricated chips, adaptive Laguerre in analog VLSI, multi-scale gamma with exponentially spreading time constants (this is literally Phase-7 banded leaks!). This is the closest prior art and must be cited.
- Legendre-SNN on Loihi-2 (Gaurav2024).
- The "fabric as nonlinear expander + memory preprocessor" = "focused gamma network" architecture (Mozer focused backprop; de Vries focused gamma net: memory plane → nonlinear feedforward mapper). Their taps8+main core is exactly a focused architecture.

Novelty assessment:
- Rediscovered: gamma memory≈ banded leaks + aux core; LDN/LMU = optimal delay operator; focused architecture = taps8 + expander; analog gamma/Laguerre VLSI = analog delay memory.
- Plausibly novel: (1) the KNet OTA-fabric parameterization with device-realistic cells (compliance gating, rail clip, gm/Isat bounds) trained/pruned toward minimal analog realization — device-aware co-design rather than idealized circuit; (2) train-and-prune framework mapping learned memory operators onto minimal OTA topologies (structured pruning under edge budget with device constraints); (3) closed-loop equalizer application (CTLE) — neural equalization exists (lots of NN equalizer papers), but analog-fabric recurrent equalizer co-optimized with the equalizer circuit might be distinctive; (4) the measurement methodology (pre-registered gates, MC/PR diagnostics, held-out discipline) — engineering contribution, not novelty.
- Heun stepping: not novel (fixed-step neural ODE).
- Parametrizing RNN as analog circuit: not novel per se (many analog RNN/RC works), but the specific OTA-fabric with compliance/rail physics + BPTT + pruning may be.

Caveat: I searched but can't guarantee exhaustive; recommend a proper related-work pass, cite the found items.

Now the experiment sequence document. Structure:

# Phase 12: Learned Memory Aux Core — Experiment Sequence
- Preamble: standing decisions (rule (i) amended, held-out scoring, cost-matched baselines)
- E0: cost-matched bank baselines (tapsVD-k sweep, taps4/6, sparse fan-out) — sets the bar
- E1: analytical aux insertion (LDN/LMU, gamma, Laguerre, ms-gamma) frozen — sanity + ceiling measurement, no training
- E2: trained aux pilot (echo_state flavor, aux-local objective, window 12, product terms optional arm, 3 seeds, held-out) with pre-registered promotion gates
- E3: width/node sweep vs task metrics (knee finding)
- E4: prune/anneal to minimal circuit + analog realizability check (map to OTA cells, edge budget)
- E5: M=2 cross-NARMA leg (only if pilot passes)
- E6: fabric integration decision- Novelty positioning section
- Open questions/risks

Include pre-registered gates per leg, edge accounting, verdicts. Keep it consistent with their doc style (plain language, gates, KILL rules). Also note gamma memory prior art means E1 should include gamma filter as a flavor — and cite that ms-gamma ≈ their banded leaks.

Also address their ROI question on "more useful features": the product-term arm answers it empirically — if lag-9 product R² doesn't move held-out NRMSE, overkill confirmed. Frame as a cheap ablation.

Length: this is a document request, so it can be long, but keep it tight. Use headings, tables where useful. No horizontal rules.
</details>

Before the document: your novelty worry is well-founded, and the search results are blunt about it. The two closest hits you need to read before writing anything:

1. **De Vries & Principe, "The gamma neural net" (Neural Networks, 1992)** and **Harris/Juan/Principe's analog VLSI gamma/Laguerre filter chips (mid-90s)**. The gamma memory is a cascade of leaky integrators with a *trainable* memory-depth parameter μ, trained by RTRL/BPTT, that interpolates between a pure delay line (μ=1) and dispersive memory — and it was fabricated in analog VLSI, with an adaptive Laguerre variant and a **multi-scale gamma with exponentially spreading time constants down the cascade**. That last one is your Phase-7 banded leaks, published ~30 years ago. The "focused gamma network" (memory plane → nonlinear feedforward mapper) is your taps8 + main-core architecture.
2. **Voelker, Kajić & Eliasmith, LMU (NeurIPS 2019)** and **Gu et al., HiPPO (NeurIPS 2020)** — the LDN has already been deployed on **analog neuromorphic hardware (Braindrop)** and Loihi, and Legendre-SNN reservoirs exist on Loihi-2.

So "LDN/Laguerre in analog" is taken. What's plausibly left is in the document below. One caveat: two searches is not a literature review; treat the novelty section as a hypothesis to verify with a real related-work pass.

---

# Phase 12: Learned-Memory Aux Core — Experiment Sequence and Novelty Position

**Status:** planning document, 2026-09-18. Supersedes the closed `aux-core-drive` / `aux-core-drive-2` plans. Inherits all Phase 10/11 instruments and audits.

## 0. Standing decisions (locked before any leg runs)

1. **Rule (i) amended — cost-conditional copy rule.** An aux whose states are reconstructible from the taps8 bank at R² ≥ 0.90 is a KILL **only if** its total edge cost ≥ 1.0 × the cost-matched bank reference (E0 below). A copy at strictly lower edge cost with held-out task parity is a PASS. Rationale: the program's goal is memory the network holds, cheaply and realizably — not novelty of operator form. A learned 32-edge delay operator is a win; a frozen 200-edge one is not.
2. **Held-out scoring is primary.** All ridge/MC/NRMSE numbers reported fit-on-train, score-on-fresh-stream (own 200-sample washout). In-sample numbers may be reported but are labeled diagnostic. Phase 11 showed in-sample concat gains evaporate held-out; every legacy MC (taps8's 11.47 included) is optimistic.
3. **Per-seed tap-subspace R² reported for every row.** The echo family sits at 0.86–0.92 on the autocorrelated stream; single-winner R² is not a stable KILL basis.
4. **The bar is the cost-matched bank, not taps8-200.** No aux result is interpretable until E0 lands.
5. **Frozen-aux screens are structurally unwinnable and no longer gate training.** The bank is the optimal frozen feedforward memory for this metric; the only frozen auxes that pass are near-copies. Training is the only informative test. The E0-gate philosophy is honored via a bounded pilot with pre-registered promotion gates (E2), not via init screening.

## E0 — Cost-matched bank baselines (eval-only, ~1 GPU-hour, runs first)

**Question:** what does the bank itself cost when it's allowed to be cheap?

Legs, all on the pinned Gate-1 flow-twin main core (r=1.0, λT=1.0, scale 0.2), held-out scoring:

- `tapsVD-k`, k ∈ {2, 3, 4, 5, 6}: SVD-compressed bank, 25k fan-in edges. (k=4 already banked: MC 9.21 in-sample at 100 edges.)
- `taps4`, `taps6`: literal short banks, 100/150 edges.
- `taps8-sparse`: full bank, Gate-3 sparse fan-out k ∈ {3, 6} per column (75/150 edges).

**Output:** the Pareto frontier MC-vs-edges of the bank family. The aux must beat the *frontier*, not the 200-edge point. **Verdict:** none — this leg defines the reference curve `R(edges)` used by every later gate. If tapsVD-6 at 150 edges holds MC ≥ 10.5 held-out, the effective bar drops and every subsequent edge budget tightens accordingly.

## E1 — Analytical aux insertion (eval-only, ~1 GPU-hour)

**Question:** what do the known optimal memory operators score in this harness, frozen?

New flavors, same probe machinery:

- `ldn`: Legendre Delay Network, orders d ∈ {4, 6, 8, 12}, window θ ∈ {8, 12} samples. Closed-form A/B matrices, no training. This is the theoretical ceiling for linear recurrent window memory and the direct LMU/LDN comparison the write-up will need.
- `gamma`: de Vries–Principe gamma cascade, K ∈ {4, 8}, μ ∈ {0.3, 0.6, 0.9}. Note: μ=1 is the shift register (already banked, KILL under old rule, PASS-candidate under amended rule at sparse fan-out).
- `laguerre`: first low-pass + all-pass cascade, K ∈ {4, 8}.
- `ms_gamma`: exponentially spreading time constants (this is the analog of the Phase-7 banded core, included deliberately to make the prior-art mapping explicit in the paper).

**Verdicts:** informational only, but two pre-registered readings: (a) if `ldn` d=8/θ=12 does not clearly beat taps8 held-out at equal edges, the entire "optimal memory" framing needs recalibration against this task's actual sufficient statistics; (b) the gamma/LDN rows anchor the related-work claims with in-harness numbers rather than citations alone.

## E2 — Trained-aux pilot (the experiment; ~1 GPU-day, bounded)

**Question:** can a trained aux generate its own history features from raw input, cheaper than the bank frontier?

- **Architecture:** one flavor only — `echo_state` (8-node tanh aux, radius init 0.95, leak init 1.0). It is the only Phase-10 family that wasn't a hand-built delay line. Main core frozen at the Gate-1 corner. Strictly feedforward input→aux→main.
- **Objective (aux-local, primary):** ridge/BPTT-trained to reconstruct `u(t−k)`, k = 1..12, from aux states. Window 12, not 8: taps8 already covers 0–7; the NARMA-10 product needs u(n−9); an 8-window objective trains the aux to duplicate the bank and leaves the actual bottleneck untouched.
- **Objective arm B (the ROI probe, one leg):** add product targets `u(t−i)·u(t−j)` for (i,j) pairs around lag 9. This is the empirical answer to "are non-bank features overkill": a linear aux gets zero on products; if arm B moves held-out NRMSE, nonlinearity in the aux earns its edges; if not, overkill confirmed and the aux stays linear. Cheap, decisive, pre-registered.
- **Budget:** 3 seeds, fixed epoch cap (suggest 30), train aux weights + aux leaks + main fan-in; ridge readout refit for reporting.
- **Edge accounting:** aux_recurrent + aux_fan_in + main_fan_in, dense and sparse (k=3) fan-out variants.

**Promotion gates (all three seeds, held-out):**

1. MC ≥ 0.85 × the E0 frontier value at equal edges;
2. held-out NRMSE within 0.02 of the equal-edge bank reference;
3. activity/rail guards inherited from Phase 8;
4. per-seed tap-R² reported (not gated — the amended rule (i) applies only at the fabric-integration decision).

**Failure handling:** a pilot that fails gates 1–2 is a negative result, not a reason to sweep initializations. One allowed follow-up: end-to-end task-loss arm (aux trained through the frozen main), which tests whether the local objective — not the architecture — was the binder. Note the asymmetry: end-to-end loss gives zero gradient for duplicating main's features, so it is structurally biased toward residual information; D4 says that residual is more delays.

## E3 — Width sweep (only if E2 promotes; ~0.5 GPU-day)

**Question:** where is the knee of aux size vs task-sufficient memory?

n_aux ∈ {2, 3, 4, 6, 8, 12}, trained as in E2. Score against **held-out NRMSE and per-delay reconstruction**, not against "8 taps." Prediction (pre-register it): for a linear aux with white input, exact reconstruction of W delays needs n ≥ W; below that, error concentrates at large k, LDN-style. The interesting regime is n < W with graceful long-lag degradation — that is the cost floor of learned memory, and it is the number the analog realization (E4) sizes against. If the sweep shows anything other than Krylov-rank behavior, suspect the harness before celebrating.

## E4 — Prune-and-realize (only if E3 finds a knee; ~1 GPU-day)

**Question:** what is the simplest OTA-fabric circuit that implements the learned operator?

- Anneal-prune the trained aux under an edge budget (dense-start, magnitude + device-L2, the existing `--sparsity`/`--device-l2-lambda` machinery), tracking held-out MC vs edges.
- Map the surviving structure onto `tanh_free` cells: each aux node = one OTA integrator stage; check gm/Isat bounds, compliance margins, rail usage, and sensitivity to ±10% device mismatch (the mismatch gate machinery already exists from `post_twin_gates`).
- **Verdict:** PASS requires a pruned aux at ≤ 0.5 × bank-frontier edges holding ≥ 0.85 × frontier MC under mismatch. This is the leg that produces the paper's figure: "learned memory operator → minimal analog circuit."

## E5 — Multi-input leg (only if E2 promotes; ~0.5 GPU-day)

**Question:** does the cost argument survive M > 1, where it actually lives?

Cross-NARMA: y depends on u₁(n−i)·u₂(n−j), M=2. Bank cost 16 taps / 400 fan-in edges; aux cost 2× aux or one shared aux — that comparison is the point. Per-channel MC vector reported. This is the only leg where the bank's 8M scaling is measured rather than asserted. (Benchmarks exist: adding/multiplication tasks are inherently 2-input; MIMO equalization is the in-domain version.)

## E6 — Fabric integration decision

Earned only by E2→E4 passing. The aux replaces the boundary delay bank in the real KNet delay-bank harness (Phase-6 architecture), paired seeds 0/1/2, against the Phase-8 activity-restored banded baseline. Pre-registered win: held-out NRMSE within 0.02 of the banked baseline at ≤ 0.5 × boundary edge cost, activity 0.45–0.55, rails ≤ 0.20.

## Dependency summary

E0 → E1 (parallel OK) → E2 → {E3, arm-B reading} → E4 → E5 → E6. Any KILL stops the chain; the document records the negative result and the program falls back to "make the bank cheaper" (E0's own outputs are then the deliverable).

---

## Novelty position (honest version)

**Rediscovered, cite it, don't claim it:**

| Your element | Prior art |
|---|---|
| Cascade of leaky integrators as trainable memory, depth/resolution tradeoff | Gamma memory, de Vries & Principe 1992; trained by RTRL/BPTT |
| Exponentially spread time constants (Phase-7 banded leaks) | Multi-scale gamma, Harris/Juan/Principe mid-90s |
| Optimal recurrent window memory | LDN/LMU (Voelker et al. 2018/2019), HiPPO (Gu et al. 2020) |
| Memory plane → nonlinear expander (taps8 + main core) | Focused gamma network; Mozer focused backprop |
| LDN on analog neuromorphic hardware | Braindrop deployment (Voelker thesis); Loihi LMU; Legendre-SNN on Loihi-2 (Gaurav 2024) |
| Analog gamma/Laguerre filters in silicon | Harris, Juan & Principe, fabricated adaptive chips |
| Heun-stepped neural ODE, "RNN as analog circuit" generally | Neural ODE literature; extensive analog reservoir computing work |

**Plausibly novel (verify with a real related-work pass):**

1. **Device-realistic co-design of the memory operator.** Prior analog memory work used idealized integrator stages; prior LDN hardware work used digital/spiking substrates. Training and pruning a memory operator *inside* a device-faithful OTA fabric — with gm/Isat bounds, compliance gating, rail clipping, and mismatch sensitivity as first-class constraints — and reporting the minimal realizable circuit is a co-design contribution neither lineage has.
2. **The train-prune-map pipeline as a method.** Not "an RNN" but a framework: learn the operator, anneal to minimal edges under a device budget, emit a buildable analog spec. The deliverable is the smallest circuit, not the best score.
3. **Closed-loop equalizer application.** Neural equalizers are a crowded field, but a recurrent analog-fabric equalizer co-optimized with the CTLE it lives inside — where the network *is* the analog front-end's adaptation — is a stronger claim than "we also ran NARMA." This is likely your best application story; it is also where M>1 is native rather than synthetic.
4. **The measurement discipline** (pre-registered gates, held-out MC, cost-matched baselines, KILL rules) is a contribution to reservoir-computing methodology, where in-sample MC reporting is rampant. Frame it as such; don't frame it as a new architecture.

**Strategic recommendation:** stop positioning against "reservoir computing" and position against **gamma/LDN memory + analog implementation** specifically. Your pitch becomes: "The optimal linear memory operator has been known since 1992 (gamma) and 2019 (LDN). What nobody has done is learn it under analog device physics and prune it to a minimal, mismatch-tolerant OTA circuit — and show when the learned version beats being handed the past." That is a defensible thesis. "We built an RNN that learns its own memory" is not — it's a 1992 paper.

**On the overkill question:** E2 arm B settles it empirically for this task. But note the prior art cuts both ways — gamma memory's whole selling point was that the *task* should choose the memory shape. If arm B shows products don't pay, that's not a failure; it's the gamma conclusion replicating in your harness, which strengthens the paper's story rather than weakening it.


---

### User

Let us take a step back and do some research - The way I see it there are a few niches to attack - (i) No one has made a reconfigurable general fabric - all of the circuits (like analogssn) are fixed once designed and not reprogrammable. (ii) Nobody has demonstrated mismatch/pvt aware performance ? (iii) everybody needs large input range linear otas they havent demonstrated any architecture that allows you to use more normal OTAs (v) LDN matrices are dense and tightly coupled - maybe via learned pruning we could remedy this (vi) on chip drift calibration (vii) Dynamic Range Compression & State SNR Balancing [For polynomial orders $d > 6$, higher-order state variables ($x_5, x_6, \dots$) drop dramatically in signal amplitude compared to low-order states ($x_0, x_1$)]. You can find out more. I feel like perhaps attacking the reconfigurable general chip using programmable OTAs (discrete quantized settings) and the dense connections (pruning pipeline) might be my best shot  - but do some research and point out anything that I have missed or got wrong and potential options


---

### Assistant (kimi-k3)

<details>
<summary>Reasoning</summary>

The user wants research on their claimed niches. Let me plan two web searches (limit 2 per response). I should cover: reconfigurable analog neuromorphic fabrics (FPAA — field programmable analog arrays! That's the big miss — FPAAs are exactly reconfigurable general analog fabrics), mismatch/PVT-aware training (noise-aware training literature exists), large input range OTAs (linearization techniques), LDN pruning (maybe), on-chip drift calibration (exists — e.g., calibration in BrainScaleS, mixed-signal neuromorphic calibration), dynamic range compression in LDN states.

Key things I suspect the user got wrong:
- (i) FPAAs (Field Programmable Analog Arrays) are a decades-old field: Anadigm, LATICE ispPAC, RASP (Jennifer Hasler's group, Georgia Tech — FPAA SoC, reconfigurable analog fabric with OTAs, floating-gate programmable). Also "field-programmable analog arrays for neural networks" work. Hasler's group has large-scale reconfigurable analog fabrics with programmable OTAs. So niche (i) as stated is wrong — FPAAs exist and are reconfigurable; the gap might be "FPAA trained as a reservoir/RNN with learned memory operators" but even that has prior art (FPAA neural networks, Hasler's work).
- (ii) mismatch/PVT-aware training: noise-aware training, device-variation-aware training is a big literature (in-memory computing, analog AI — IBM analog AI chip, ambiq, etc.). Hardware-aware training exists.
- (iii) linearized OTAs: tons of circuit techniques (source degeneration, etc.). But the claim "nobody has demonstrated an architecture that allows normal OTAs" — the KNet tanh_free cell with compliance gating is somewhat related.
- (v) LDN dense matrices — sparse LDN approximations? There might be work on sparse/structured state space (S4 uses HiPPO with diagonalization — S4D diagonal! S4D-Lin shows diagonal approximations of HiPPO work). So "dense LDN matrices" — the S4 literature already diagonalized HiPPO. That's relevant.
- (vi) on-chip drift calibration — exists (floating-gate programming compensates mismatch — Hasler's floating-gate FPAAs do exactly this: programmability used to cancel offset/mismatch).
- (vii) LDN state amplitude scaling — Voelker's thesis discusses scaling; LDN state magnitudes decrease with order; there may be work on balanced realizations.

So the big misses: FPAA literature (Hasler's RASP), floating-gate programmability (which gives reconfigurability + mismatch cancellation + drift recalibration), S4/S4D diagonal HiPPO (sparse LDN), noise/variation-aware training literature, analog in-memory computing chips (IBM HERMES, etc.) with PVT variation handling.

Let me do two searches to verify: one on FPAAs/reconfigurable analog fabrics + floating-gate, one on variation-aware training / LDN sparsity / S4D diagonal HiPPO / analog AI chips.

Search 1 queries:
- "field programmable analog array FPAA reconfigurable OTA fabric Hasler RASP"
- "floating-gate programmable analog array mismatch calibration offset cancellation"
- "FPAA neural network reservoir computing analog"

Search 2 queries:
- "variation-aware training analog neural network mismatch PVT noise-aware"
- "S4D diagonal HiPPO sparse approximation state space model"
- "IBM analog AI chip phase change memory drift compensation mixed-signal"
- "LDN Legendre delay network state amplitude scaling higher order"

Let me make these natural-language queries. Then synthesize.
</details>

Two searches, then the assessment.<details>
<summary>Reasoning</summary>

Both searches returned empty. I can't retry more than 2 calls per response — I've used both. So I must answer from knowledge, flagging that the searches returned nothing and the user should verify. Actually the instruction says only say "couldn't find" if literally 0 results — I got 0 results. I should be transparent: searches returned nothing, so this is from my training knowledge, which is decent for these established literatures (FPAAs, floating-gate, S4/S4D, analog AI chips are all pre-2024 established).

Let me write the assessment from knowledge, clearly flagging verification needs.

Key points:

(i) Reconfigurable general fabric — WRONG as stated. FPAAs exist: Anadigm dpASP/FPAA (commercial, 2000s), Lattice ispPAC, and most importantly Jennifer Hasler's group at Georgia Tech: RASP (Reconfigurable Analog Signal Processor), FPAA SoCs with floating-gate programmable OTAs, large-scale field-programmable analog arrays. Floating-gate FPAAs are: reconfigurable (routing + device parameters programmable), mismatch-compensating (floating-gate charge programmed to cancel offsets), used for neural networks, vector-matrix multiplication, and even reservoir-like computing. Also: mixed-signal neuromorphic chips — BrainScaleS (Heidelberg, analog neurons, digital config), DYNAP-SE (SynSense/Indiveri, programmable analog neurons/synapses), Loihi (digital though), TrueNorth (digital). Braindrop (analog, but fixed). So "no one has made a reconfigurable general fabric" is false — the gap is narrower: reconfigurable fabric *trained as a reservoir/memory system with device-realistic co-design* maybe, but even FPAA neural nets exist (Hasler group did FPAA-based neural/learning systems, "large-scale field programmable analog arrays", and there's work on FPAAs for physical reservoir computing — I recall some FPAA reservoir computing papers). The honest gap: FPAAs were programmed/configured, not *trained end-to-end via BPTT with pruning to minimal circuits*; and FPAA field died commercially (Anadigm gone) — opportunity and warning.

(ii) Mismatch/PVT-aware performance — partially wrong. Noise/variation-aware training is a mature literature: hardware-aware training for analog in-memory computing (IBM's analog AI, PCM drift), noise injection during training, Bayesian/device-variation-aware training, "closed-loop" device-aware training (Rasch, Gokmen, IBM analog AI group). Neuromorphic: BrainScaleS has extensive calibration literature (each analog neuron calibrated; "in-the-loop training" for BrainScaleS-2). SpiNNaker/Loihi digital. DYNAP-SE mismatch characterization papers exist (mismatch is often *exploited* as heterogeneity in neuromorphic literature). So mismatch-aware training exists; what's less common: PVT-aware training for *continuous-time analog reservoir/memory operators* with drift recalibration. But BrainScaleS-2 in-the-loop training is close prior art.

(iii) Large input range linear OTAs — the claim that "nobody demonstrated an architecture allowing normal OTAs" needs checking; circuit-level linearization is old (source degeneration, etc.), but at architecture level, KNet's compliance-gated tanh_free cell is somewhat novel-ish. However, translinear circuits, Gilbert multipliers, and "tanh synapse" circuits are standard. The niche may be real but narrow.

(v) LDN dense matrices — partially wrong: S4/S4D literature already showed HiPPO matrices can be diagonalized (S4D-Lin: diagonal approximation works nearly as well). Diagonal = decoupled = sparse. So "LDN matrices are dense and tightly coupled, maybe pruning can fix" — the S4D result says diagonal HiPPO-LegS initialization works; that's exactly a decoupled realization. Your contribution would be: pruning under *device constraints* (OTA fabric) rather than mathematical diagonalization. Also Voelker's thesis has the LDN as coupled ODE; on Braindrop it was implemented with the dense matrix. Sparse LDN realizations — I'm not aware of explicit "pruned LDN" work, but S4D is the key citation that weakens the novelty.

(vi) On-chip drift calibration — exists: floating-gate programming (Hasler) inherently does this; IBM analog AI deals with PCM drift (drift compensation techniques); BrainScaleS calibration. Gap: *online* drift recalibration of a trained memory operator on a reconfigurable OTA fabric — plausible niche but must cite the calibration literature.

(vii) Dynamic range / state SNR balancing in LDN — real observation; Voelker's thesis discusses scaling; balanced realizations are classic control theory (Moore1981 balanced truncation). Applying balanced realization to LDN for analog SNR — plausible small niche. Also S4D/normal forms relate.

Biggest miss overall: the FPAA literature (Hasler's RASP, floating-gate) directly occupies niche (i) and partially (vi). And BrainScaleS-2 in-the-loop training occupies (ii). And S4D occupies (v).

What remains defensible:
- The combination: a *trainable* (BPTT through device-realistic simulation) reconfigurable OTA fabric, pruned to minimal circuits, with mismatch/PVT-aware training and on-chip recalibration, targeting learned memory operators (LDN-like) — the *pipeline* is novel even if each component has prior art.
- FPAAs were configured, not trained end-to-end with device-physics-aware gradients + pruning. But check Hasler's "learning" FPAA work — they did on-chip learning too (floating-gate learning rules, e.g., "floating-gate based learning"). Hmm.

Also warn: Anadigm FPAA commercial failure — why did reconfigurable analog die? Area overhead of switches, parasitics, performance penalty vs custom analog. That's the objection reviewers will raise: reconfigurability tax. The answer could be: for *memory/reservoir* workloads at moderate speeds, the tax is acceptable; or the fabric is a research vehicle.

Also mention: mixed-signal compute-in-memory (Mythic, IBM HERMES core) — reconfigurable-ish analog matrix engines, though fixed function (VMM only).

Also: "analogSSN" the user mentioned — analog spiking? Probably "analog SNN" chips like DYNAP-SE. DYNAP-SE is reconfigurable in connectivity (digital routing, analog neurons). So even in SNN space, reconfigurability exists.

Structure the answer:
1. Note searches returned nothing; this is from training knowledge; verify.
2. Niche-by-niche verdict: what's wrong, what survives, key citations to check.
3. The FPAA problem — biggest miss, detailed.
4. What the defensible position looks like (combination/pipeline, plus specific gaps).
5. Suggested verification steps (related-work pass targets).
</details>

Both searches returned zero results, so I can't verify against live sources this turn — what follows is from training knowledge on established (pre-2024) literatures, which is actually most of what matters here. Treat every citation as a lead to verify, not a confirmed fact. With that caveat: several of your niches are occupied, and one of them is occupied by a literature you appear to have missed entirely.

## (i) "No one has made a reconfigurable general fabric" — this is wrong, and it's your biggest blind spot

**Field-Programmable Analog Arrays (FPAAs)** are exactly this, and the field is decades old:

- **Jennifer Hasler's group (Georgia Tech)** built large-scale reconfigurable analog fabrics — the RASP line and FPAA SoCs — with **floating-gate programmable OTAs**: routing is reconfigurable, device parameters (gm, bias currents, offsets) are programmable per-element, and the floating-gate charge is used to *cancel mismatch and offset at programming time*. They demonstrated neural networks, vector-matrix multiplication, filters, and on-chip learning rules on these fabrics. This single literature occupies your niches (i), (


---
