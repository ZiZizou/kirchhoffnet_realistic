# Revised NARMA Plan: Measurement-First Ordering (supersedes §11 ordering)

**Date:** 2026-09-12 (Round-1 results + Round-2 plan recorded 2026-09-13)
**Status:** Steps 1–4 implemented, audited, and run on Alliance GPU (see §12). Round-2 gated (sparse-drive E0 + VCA-boundary) built, audited, and run on Alliance GPU 2026-09-13 (job 59587991, see §14). Sparse drive killed, VCA-1 negative (+0.10); §13.5 fork resolves to Step 6 (forget gate).
**Supersedes:** the S0–S5 ordering in `narma-fabric-memory-problem-statement.md` §11.7. That section remains the brainstorm record; **this document is the execution order.** S0 (Sugiura control) is deferred per §9; two-core is dropped per §8.
**Advisor input incorporated:** mental-model corrections (§2), five-requirements decomposition (§3), VCA-as-product-path (§6), ESN-as-control (§7), feedback deferral (§8), second benchmark (§10), forget-gate open question (§11).

---

## 1. Execution order (the whole plan in seven lines)

1. **S1-extended** (CPU, hours): frozen k2 states → u-MC curve, y-MC curve, IPC-2. Tells you which deficit dominates.
2. **Tuned-ESN calibration** (GPU-minutes): settles the 0.15–0.29 band. Not optional.
3. **Multi-tap trained readout** on the existing architecture: dense linear over hidden (+accumulator). Zero dynamics risk.
4. **S2 E0 grid**: hetero-leak (log-uniform τ ∈ [1,40]) × random-sign sparse fan-out × small-world. One graph, no new machinery.
5. **VCA-boundary leg** on the best S2 corner: the input-gate / product-path test.
6. **Forget-gate leak** (state/input-modulated retention): first genuine architectural contribution. Design open (§11).
7. Two-core / systolic / feedback / Sugiura: only on failure of the above, each reframed (§§8–9).

Each step gates the next. Kill criteria are pre-registered per step.

---

## 2. Two corrections to the working mental model

**C1. The output accumulator carries across samples.** E2 says so directly: reset-hidden (accumulator kept) gives NRMSE 0.87, reset-all gives 1.02 (`carry_ablation.csv` legs). The accumulator alone is worth ~0.15 NRMSE. It is not "outputs store nothing" — it is that the accumulator is a 1-D leaky integrator, so it holds a blurred trace and nothing else. Any readout redesign must preserve-or-beat this channel, not discard it unknowingly.

**C2. Input-vs-output storage is testable today, for free.** `memory_capacity()` (post-v5 Jaeger convention) regresses states against past *inputs*. Changing the regression target to past *outputs* (`y(n-k)` instead of `u(n-k)`) yields an output-trace MC probe on the frozen k2 checkpoint — minutes of CPU, no new machinery. Folded into S1 (§4) so S1 measures all three NARMA requirements before any architecture is designed around a guess.

---

## 3. First principles: what the fabric needs as an RNN (status per requirement)

| # | Requirement | Mechanism in healthy systems | Fabric status |
|---|---|---|---|
| 1 | Persistent state | carryover across steps | **Has it** (E2 — carryover load-bearing) |
| 2 | Stability / echo-state property | contractive dynamics, spectral radius ≲ 1 | **Has it** (leak pins max\|J\| ≈ 0.94; C1b hit 0.995). The one thing done right |
| 3 | Timescale diversity spanning task lags | ESN: random-W eigenvalue spread; LSTM: per-unit forget gates; clockwork RNN: explicit leak draws | **Lacks it — deficit #1.** Uniform leak init; C1b proved 50 identical slow integrators are *worse* than weak random mixing |
| 4 | Input separability / dimensionality | ESN: sparse random connectivity | **Lacks it — deficit #2.** Torus + identical fan-out + weak gm → PR ≈ 1 |
| 5 | Nonlinear feature expansion | ESN: tanh of mixed signals; LSTM: gating (multiplication *is* the mechanism) | **Lacks it at init; training can't reach it — deficit #3.** gm ≈ 0.077 near-linear; C3 path to gain crosses the saturation cliff |

Mechanism borrows (sequenced in §§6–7 of the execution order):
- **From LSTM — input gate ≈ VCA on boundary edges** (u-dependent multiplicative write control; built, never tested in NARMA) and **forget gate ≈ input/state-dependent leak** (does not exist; novel for KNet; §7).
- **From ESN — full-state readout** (funnel-25→1-accumulator is the current design; frozen full-hidden ridge 0.6434 already beats trained 0.676).

---

## 4. Step 1 — S1-extended (CPU): u-MC curve, y-MC curve, IPC-2 on frozen k2 states

**Rationale.** Every verdict so far is a *linear* measurement on a task whose hard term is *quadratic* (`1.5·u(n-9)·u(n)`). S1-extended measures all three §3 deficits on the frozen k2 checkpoint (epoch 54) before anything is built.

**Work items** (all in `narma_experiment.py` or alongside `memory_capacity()`):
- **(a) u-MC per-delay curve.** Already exists — re-run on the full stream with washout and report the *curve* (delays 1–20), not just the total. Reference: C0 textbook decay vs C1 flat 0.32→0.07.
- **(b) y-MC per-delay curve (new, trivial).** Call the same ridge machinery with regression target `y(n-k)` (standardized test targets) instead of `u(n-k)`. Measures output-trace memory (§2-C2): is the accumulator channel + hidden states reconstructing the Σy functional the target needs?
- **(c) IPC-2 per-delay (new, small).** Regression targets become Legendre `P_2(u(n-k))` (and, if cheap, products `u(n-i)·u(n-j)` for a few (i,j) pairs around lag 9). Same ridge solve, same R² scoring, summed over delays. Reference: run the identical instrument on C0-ESN states — the fabric number is only interpretable against it (Dambre bound: total IPC ≤ node count).

**Inputs:** frozen k2 checkpoint states on the full 2500-sample stream, washout 200 (same provenance as C0/C1). No training, no GPU.

**Pre-registered reading:**
- y-MC decent + u-MC taps missing → timescale diversity (deficit #3) is the whole game; prioritize step 4.
- IPC-2 ≈ 0 while C0 shows nonzero → the lag-9-product path is provably absent; no memory fix alone closes NARMA; a multiplicative mechanism (step 5 VCA or cell change) is mandatory.
- IPC-2 present but NRMSE plateaued → information exists; fault is readout rank; step 3 becomes the fix.

---

## 5. Step 2 — Tuned-ESN calibration (GPU-minutes, not optional)

**Rationale.** This is a **control group, not a design direction**. Every "the fabric lacks X" claim is anchored to a 0.15–0.29 band the harness has never reproduced (in-harness ESN references disagree: 0.5596 local vs 0.771 full-stream, both undertuned). If a tuned ESN in this pipeline cannot break ~0.4, the task instance (stream length, washout, scaling) caps everyone and the fabric has been judged against a phantom — possibly retiring NARMA-10 as the gate.

**Work items** (extend `run_baselines()` ESN leg):
- Grid: spectral radius {0.7, 0.9, 1.0, 1.1}, leak {0.3, 0.5, 0.8, 1.0}, size {25, 50, 100}, input_scaling {0.2, 0.5, 1.0}, ridge λ {1e-4, 1e-2}. Keep seed-0 stream + washout-200 protocol identical to fabric runs.
- Report NRMSE/R²/MC/PR per corner (validates MC/PR instruments on a second healthy reference alongside C0).

**Gate:** tuned ESN clearly below ~0.4 → band stands, all fabric verdicts hold. Tuned ESN stuck ≥ ~0.4 → band miscalibrated for this pipeline; re-anchor every verdict and revisit whether NARMA-10 remains the gate before any fabric build.

**Cost:** minutes of GPU. Run before any fabric training.

---

## 6. Step 3 — Multi-tap trained readout (existing architecture, zero dynamics risk)

**Rationale.** Highest value per effort in the document. Evidence already owned: frozen full-hidden ridge 0.6434 beats the trained single-accumulator readout 0.676, and the 5-lag stack hit 0.586 — the trained readout is *worse* than linear probes of the states it learned to produce. If a trained dense readout lands ~0.58–0.60, H3 was bigger than credited and every subsequent dynamics fix gets a cleaner signal.

**Implementation shape** (config/plumbing only, no dynamics change):
- Build the NARMA net with `enable_temporal_readout=False` + dense `read_mode` (num_proj=0 → reads all 25 hidden) → plain `OutputMapper(node_dim=25, out_dim=1)` (26 params — *fewer* than the mesh + OutputAffine it replaces). New flag through `_build_fabric_net` (e.g. `--readout {temporal,dense}`) + preset override; no stage/rhs changes.
- Fast-path compatibility: `train_fabric` inner loop (`narma_experiment.py:920`) and `_evaluate_fabric_direct` (line 1227) slice via `net.read_slice` generically — verify the slice covers hidden (not the absent accumulator tail) under the dense build; `collect_fabric_states` / MC paths are readout-agnostic already.
- Readout form: dense linear over **hidden + accumulator** (preserve the §2-C1 y-channel while adding taps). Pure-hidden variant as a secondary leg. Lag-tap readout (seeing `x(n-k)` in the forward pass) is explicitly out of scope — it needs a state-history buffer the per-sample forward pass does not have.
- Note: `validate_topology_degrees` write→read hop check is silent when read_idx is None (dense path) — no conflict with full fan-out.

**Gate:** dense trained readout clearly below the 0.6434 frozen-ridge reference (target band ~0.58–0.60) → H3 promoted, dynamics work continues on a cleaner signal. No movement → readout rank is not the binder; proceed to step 4 regardless (S1 will already have said which deficit dominates).

---

## 7. Step 4 — S2 E0 grid (first build; one graph, no new machinery)

**Rationale.** Deficits #1 (timescale spread) and #2 (separability) tested at E0 — minutes per corner, no training — inside a single graph. Multi-core is not built until this falsifies itself (§8).

**Grid axes** (all init-only changes):
- **Hetero-leak:** per-node time constants log-uniform τ ∈ [1, 40] samples (clockwork-RNN style) → `raw_leak_init` vector, not scalar. Half the nodes become fast taps, half slow traces.
- **Random-sign sparse fan-out:** per-node input gains/signs, sparse masks (ESN-style `W_in`, breaking the identical-drive symmetry). Pairs with:
- **Small-world core:** flip `make_narma_preset` `hidden_family` torus → small_world (builder already supports it).
- Hetero-C explicitly deferred: per-node C scales all currents including effective leak, nearly redundant with hetero-leak for this purpose — and unlike leaks it needs an `rhs()` code change. Revisit only if leaks show promise.

**Instruments per corner:** ridge / MC total + per-delay curve / PR / rail% (existing E0 harness; C0-anchored gates).

**Gate:** any corner with PR ≥ 3, MC ≥ 4, ridge clearly < 0.6, rails sane → single short k=2 train on that corner only. 20+ corners all at PR ~ 1–2 → within-graph heterogeneity insufficient; *then* partitioned/structural options (§8) become interesting.

---

## 8. Step 5 — VCA-boundary leg (input gate / product path)

**Rationale.** VCA on the boundary is not just a symmetry breaker — it is mechanistically pointed at deficit #5. The gate `2σ(b + (u·W)·v_e)` multiplies a boundary current depending on `u(n)` and `x(n)`; if `x` carries any `u(n-9)` trace, the injected current contains `u(n)·u(n-9)`-ish cross terms. **VCA-boundary is the candidate for the missing product path** — the thing IPC-2 (§4c) is designed to detect as absent. Sequence it *after* the memory lever lands: E4d already showed stacked levers confound attribution. One flag per leg.

**Work items:** plumb `vca_enabled` (+ rank/bias) through `_build_fabric_net` with CLI flags (defaults stay off; `topology.py:1920-1925` already supports it). Run on the best S2 corner only, not from scratch.

**Caveats (pre-registered):** scalar input → one shared time-varying gate direction across edges (per-edge embeddings only: weak symmetry breaking); GLN modulates boundary/readout gm only, never the core — neither fixes PR≈1 mixing. GLN deferred: a fancier version of the same u-conditioned injection; only if VCA fires.

**Gate:** IPC-2 moves and/or NRMSE moves on the S2-winning corner → product path confirmed, promote to full training. No movement → the multiplicative mechanism must come from cells (S3-bistable revival) or the forget gate (§7 of execution order).

---

## 9. Deferred / reframed items (do not build; conditions to revive)

- **Two-core input/output split: dropped in the literal form.** Blockers: (a) the model never receives y — an "output memory core" has nothing to store without parallel/autoregressive feedback (error compounding, breaks literature comparability) or teacher-forced y-input (changes the benchmark, voids the band). Owner must pick autonomous / teacher-forced / y-from-u-reconstruction before this lives (§11-Q1). (b) Symmetry problem: identical drive + identical init + identical family → identical gradients → two copies of one blurry integrator, not an input-head and output-head. Attention-head diversity comes from independent random init projections, and even then heads go redundant. Per-core symmetry breaking (topology seeds, leak draws, fan-out signs) is prerequisite — at which point two-core = partitioned heterogeneity, and partitioning defends against over-mixing, a disease the fabric does not have (C1: mixing too *weak*). Revive only if step 4 passes E0 but pools interfere (heterogeneity works yet re-homogenizes). Warning: two weakly-coupled cores in series = variant B with extra steps — do not rebuild the systolic chain while believing it rejected.
- **Systolic (variant B): reframed as ceiling diagnostic, still deferred.** Not a nonlinearity mechanism — a memory-structuring one (analog tapped delay line; raw-delay taps already 0.38 in-harness). Value: if B works it proves the harness end-to-end and sets the structural target; if B fails the harness is broken and everything else is moot. Keep, do not prioritize.
- **Own-prediction (ŷ) feedback: deferred.** Makes the model a generator, breaks band comparability, adds scheduled-sampling stability work on top of three unfixed deficits. Note the latent y-channel already exists (accumulator ≈ 0.15 NRMSE); the y-MC probe (§4b) measures whether it is used well before a second channel is built.
- **S0 Sugiura control: deferred, superseded by step 2 for now.** Tuned-ESN calibration answers most of the H5 question at minutes-of-GPU cost vs S0's build + N=50k training. Revive S0 if step 2 fails weirdly (tuned ESN also off-band in a way that implicates the harness rather than the band) or as the fixed-reservoir + big-MLP reference when the fabric approaches the band.

---

## 10. Second benchmark (anti-overfit)

Passing NARMA-10 must not overfit the design to 10-lag products. Pick one probing a *different* memory axis: **Mackey-Glass-17** (continuous chaotic dynamics, longer horizon — the standard reservoir complement to NARMA; a continuous-time fabric should have a native advantage here, and lacking it is diagnostic too). Alternatives recorded but not picked: copy task (exact discrete retention, brutal on leaky analog state) and NARMA-20 (same family, longer lags — checks timescale-spread scaling). Adopt Mackey-Glass when step 4 passes; before that it is scope creep.

---

## 11. Open questions (author decisions recorded 2026-09-12)

1. **Forget-gate drivers (step 6 design input): DECIDED 2026-09-12 — LSTM-faithful (both u and x).** `u` = scalar input drive per sample (`V_u ∈ [−3,+3]`, shared by all nodes); `x` = 26-D ODE state (per-node voltages, history-bearing). Gate form: `λ_j = σ(a_j·V_u + c_j·x_j + b_j)` per node (x-sense path feeds back into the `rhs()` leak term). Scale-down option recorded: GRU-tied variant (couple input/forget, fewest params) if the full gate proves too costly. Detailed design deferred until step 6 approaches.
2. **Two-core feedback choice** (revive condition for §9-dropped item): autonomous / teacher-forced / y-from-u-reconstruction? Still open; only matters if step 4 fails in the re-homogenization way.
3. **GPU budget: CONFIRMED.** Steps 1–2 proceed immediately; GPU available for step 3 trains and post-gate short trains.
4. **S0 disposition: CONFIRMED deferred (ESN-first).** Revive only if step 2 implicates the harness rather than the band.
5. **Second benchmark: CONFIRMED Mackey-Glass-17.** Adopt when step 4 passes.

---

## 12. Round-1 results (Alliance H100, jobs 59367932 / 59444668 / 59492665, 2026-09-12–13)

Code provenance: `narma_revised_plan.py` (new), `_build_fabric_net(readout={temporal,dense})`, `e0_sweep` hetero-leak token + small-world flags, `probe_audit_revised.py` (13 checks, ALL GREEN; `probe_audit1/2` still green). Batch: `narma_revised_plan_batch.sh` (4 h allocations; E0 legs later raised to 3 h caps after the first timeout round).

### S1-extended on frozen checkpoints (k2 epoch-54 + k8, seed 0, full 2500-sample streams)

| | k2 | k8 |
|---|---|---|
| ridge-from-states | 0.641 | 0.645 |
| u-MC (per-delay decay) | 2.56 (0.61→0.05 textbook) | 2.80 |
| y-MC | 7.10 | 6.79 |
| IPC-2 total | 0.39 (smeared, no lag-9 peak) | 0.47 |
| NARMA product pairs (all 7) | 0.01–0.03 | 0.01–0.03 |

Reading (pre-registered §4): y-MC decent + u-MC present-but-small + IPC-2 diffuse + product pairs absent → **the lag-9-product path is provably absent; no memory fix alone closes NARMA; a multiplicative mechanism is mandatory.** y-MC ≫ u-MC confirms C1/C2 (the accumulator output-trace channel is real and load-bearing).

### S2a tuned-ESN calibration (288 corners)

Best 0.2964 < 0.4 → **BAND-STANDS.** The 0.15–0.29 band is real in-harness; the fabric is guilty, not the task setup. (ESN MC/PR on the same corners: 3–5 / ~1.5 — healthy memory at PR values the old spec would have called collapsed; PR stays demoted to diagnostic.)

### S3 dense trained readout (k=2, 80 ep, early-stop ep49, fp32)

Test **NRMSE 0.6773 / R² 0.5408 / MC 3.46**, 1080 params, best val 0.6601 (ep34). Untrained dense ridge 0.7194 = untrained temporal ridge 0.7194 exactly. **0.677 ≈ temporal 0.676 → H3 demoted: readout rank is not the binder.** Side observations: rails stayed ≤0.3% (vs 9.9% temporal — the accumulator path is what slams rails), trained MC rose 2.56→3.46. Implementation note: the leg first died at epoch 0 on non-finite BPTT gradients under CUDA AMP (dense 25-dim read overflows fp16 where the 1-D accumulator read does not; the finiteness guard fires before GradScaler can skip-and-recover). fp32 verified locally on identical seed/streams, `--no-amp` adopted for dense trains, `--compile` kept.

### S2b E0 torus grid (48 corners, 3989 s ≈ 83 s/corner)

Gain × {slow-fixed, randomized, 0.15, hetero:1.0:40.0} × drive. **0/48 pass** (12 rail-clean, all gm=-5):

| Regime | Ridge | MC | PR | Rail |
|---|---|---|---|---|
| gm=-5 | 0.66–0.77 | 0.9–2.0 | ~1.0–1.2 | 0% |
| gm=-2 (best memory) | 0.64–0.80 | peaks 2.6–2.75 | 1.1–1.75 | 31–42% ✗ |
| gm≥0 | 0.80–0.88 | 0.2–0.8 | 2.5–7.8 (garbage variance) | 40–56% ✗ |

Hetero ≈ randomized ≈ fixed-0.15 at gm=-2: **timescale diversity via leaks buys nothing.** The crossover is a cliff (memory arrives hand-in-hand with rails), never a sweet spot — same physics as C3. New evidence: **gate-zero frequently beats the core** (e.g. hetero gm=-2/drive-0.5: ridge 0.76→gzR 0.64, MC 2.75→gzMC 3.01) — the core contributes interference on top of a working boundary→accumulator feedforward channel.

### S2c E0 small-world grid (48 corners, 3940 s)

**0/48 pass.** Same shape, marginally better peaks: randomized gm=-2/drive-0.25 reaches **MC 3.19, ridge 0.64** (best of all 96). Anomaly: slow-fixed gm=0/drive-0.5 scores ridge 1.04 (worse than predicting the mean — instability blowup at high gain). Gate-zero-beats-core pattern repeats. **Verdict: small-world ≈ torus to the second decimal; torus is retained as the single canonical topology.**

### §7 gate verdict

"20+ corners all at PR ~1–2 → within-graph heterogeneity insufficient" — confirmed over 96 corners (PR ≤1.75 at every usable gain; no rail-sane corner above MC 2.0). Per the execution order, work proceeds to Step 5 (VCA-boundary, the product-path candidate) — with the S1 finding (product absent) as its mandate.

### Batch-ops lessons (protocol, not science)

- E0 pace ≈ 80 s/corner (gate-zero double-pass + Jacobians dominate) → size E0 legs at ~1.1 h per 48 corners; 3 h caps.
- `e0_sweep` now flushes per-corner progress (`e0_progress.json`, atomic writes) and resumes by `config_tag`; any kill costs only the in-flight corner.
- CLI fragility found twice: comma-grids starting with `-` need `--flag=value` form (argparse negative-number regex); hetero token must avoid commas (`hetero:LO:HI` colon form). Both covered by audit CLI smokes.

---

## 13. Round-2 execution plan (planned 2026-09-13, built + run as `narma_round2_gated`, job 59587991 — results in §14)

Sequencing decision (author): sparse-drive E0 and VCA plumbing proceed **in parallel**; VCA trains run after, on both corners sequentially; ablations isolate each lever (one enabled / other off).

### 13.1 Sparse-drive E0 grid (tests the forceful-drive hypothesis)

**Hypothesis.** The NARMA preset fans input 0 into all 25 nodes (`{0: list(range(hidden_dim))}`) — an unablated default, not a finding (regression tasks use 2 nodes/input; the E4b screen tested drive *strength*, never *sparsity*). Under identical strong forcing all node voltages track the input, so core edges see only tiny voltage differences and carry no signal — the core is decorative *because* there is nothing to mix. Sparse drive inverts this structurally: undriven nodes feel only leak + recurrent arrivals, voltages differ by construction, recurrent currents grow meaningful, and driven/input-timescale vs free-running/leak-timescale nodes manufacture diversity without hetero machinery.

**Grid (torus only).** Boundary fan-out `{0: [0, 5, 10, 15, 20]}` (5 evenly spaced nodes — maximal inter-drive distance, simplest to reason about) × gain {-5,-2,0} × leak {slow-fixed, randomized, hetero:1.0:40.0} × drive {0.5, 1.0} = **18 corners** (~25 min). The S2b full-fan-out 48 serve as the control arm — no recompute.

**Build spec.** `_build_fabric_net(..., boundary_fan_out=None)` + CLI `--boundary-fan-out '<json>'`, validated against hidden_dim; `None` = canonical full fan-out (zero behavior change). `e0_sweep` + e0 CLI take the same flag for both torus and small-world factories. Tag hygiene: canonical runs keep **byte-identical** tags (`probe_audit1` golden untouched); sparse runs append `_fan0-5-10-15-20`. Resume build-key gains the fan-out map.

**Telltale (pre-registered).** The gate-zero delta: sparse drive flipping it (gzMC ≪ MC, gzR ≫ ridge) confirms full fan-out was suppressing the core; gz still ≥ everywhere exonerates the core independent of drive. Either outcome is decisive.

**Kill criteria.** Sparse best-MC ≤ full-fan-out best-MC *and* no gate-zero flip → drive sparsity is not a lever; VCA reads proceed as planned and sparse nets are dropped. Sparse best clearly above full-fan-out best *with* a gate-zero flip → the suppressor was the drive map; VCA-2 (below) becomes the primary leg and full-fan-out trains stop.

### 13.2 VCA-boundary plumbing spec

`_build_fabric_net(..., vca_enabled=False, vca_rank=None)` forwarded to `build_net_from_config` (stage machinery already exists), hardcoded to **boundary-only** (`vca_core`/`gate_shunt`/`separate_core_bus` off) per §6. CLI: `--vca-boundary`, `--vca-rank`, plus init-override flags `--gain-init` (gm/isat fill) and `--leak-init` (E0-token-compatible: slow-fixed/randomized/numeric/`hetero:LO:HI`, reusing `apply_gain_override` via lazy import — probes already imports experiment, so eager import would cycle; drive axis reuses `--input-scale`). `_vca` cond-name suffix (checkpoint-safe), CSV column, param accounting. Audit: VCA params present when on / absent when off, forward finite, init-override values land, 1-epoch train smoke. No dynamics-code changes.

### 13.3 VCA legs (k=2, 80 ep, early-stop 15, fp32, temporal readout)

Isolation matrix {full, sparse} × {VCA off, VCA on}:

1. **VCA-1** (full drive, best-corner init: gain-init -2, randomized leak, drive 0.25 — MC 2.59 / ridge 0.71; **fallback to canonical gm=-5 init if day-0 rails crash it**). Comparator exists: temporal k2 0.676.
2. **Plain-sparse** (no-VCA control for VCA-2): sparse fan-out + canonical init, same recipe.
3. **VCA-2** (sparse fan-out + VCA on, sparse-winner init).

**Attribution (pre-registered, answers the VCA-entanglement question).** VCA contains static per-edge gains (learned separability, present even at fixed input), dynamic u-dependent multiplication (the product path), but no true zeros (unlike ESN masks — all edges stay, reweighted). Post-hoc read: product-pairs/IPC-2 jump with PR flat → product path did it; PR jumps too → separability contributed. Static sparse masks get built **only** if that verdict demands them.

### 13.4 Batch + resourcing

New `narma_vca_sparse_batch.sh`: preflight → sparse E0 (~0.5 h) → three trains (~10–25 min each) → summary extended with gate-zero deltas and trained VCA-vs-control deltas. One 4 h allocation; every leg independently skippable.

### 13.5 Fork outcomes (pre-registered)

- Sparse E0 flips gate-zero + VCA-2 wins → drive map was the suppressor; full-fan-out work stops.
- VCA moves NRMSE/product-pairs with PR flat → product path confirmed; separability work stops.
- VCA moves nothing on either corner → multiplicative mechanism must come from cells (S3-bistable revival) or the forget gate — Step 6 design activates.

### 13.6 Explicitly NOT built

Static random-sign sparse `W_in` (conditional on VCA attribution); forget-gate leak (parked behind VCA/cells); hetero-capacitance (nearly redundant with hetero-leak, needs `rhs()` surgery — revive only if leaks show promise); GLN (fancier same-injection, behind VCA); small-world and all multi-topology grids (dropped); Mackey-Glass-17 (still scope creep until a leg passes).

---

## 14. Round-2 gated results (Alliance H100, job 59587991, 2026-09-13)

Batch: `narma_round2_gated_batch.sh` (stdout: `outputs_alliance_can/narma_round2_gated-59587991.out`).
Preflight: sparse+VCA build smoke OK; audits A–R3 ALL GREEN. E0 legs 18 corners each
(~25 min each); trains k=2, 80 ep, early-stop 15, fp32 (`--no-amp`), temporal readout.

### E0-dense (full fan-out control, 18 corners): 0/18 pass

Best rail-sane: gm=-5 / hetero:1.0:40.0 / drive 1.0 → ridge 0.6606, MC 1.98,
PR 1.17, 0% rails. Same dead shape as S2b — the best corner reproduces the S2b
best (ridge 0.66055, MC 1.97775) to 5 decimals (determinism check). gm=-2 corners
peak MC 2.7 but railed 31–42%; gm=0 railed 36–54%.

### E0-sparse (fan-out {0:[0,5,10,15,20]}, 18 corners): 0/18 pass, worse

Best rail-sane MC 1.24 (randomized, gm=-5, drive 1.0); best ridge 0.6452
disqualified at 46% rails. No gate-zero flip under the verdict definition →
**drive sparsity is not a lever; sparse nets dropped.** Plain-sparse and
VCA-2-sparse trains skipped (their `results_table.csv` files absent).

### Gate verdict

`winner=dense`: sparse best-MC 1.24 ≤ dense 1.98. Stage-B fanout: dense.

### Plain-full train (control, canonical init)

Test **NRMSE 0.6834 / R² 0.5323 / MC 2.73**, 1261 params, early-stop ep54
(best val 0.6779). Third reproduction of the ~0.676 plateau. Healthy curve
(loss ↓, h_rms ↑, rails 0→9%).

### VCA-1-full-vca (VCA boundary, gain-init -2, randomized leak, drive 0.25)

Test **NRMSE 0.7860 / R² 0.3814 / MC 2.47**, 1363 params, early-stop ep44
(best val 0.7582). **Δ +0.1026 vs control — VCA hurt.** Day-0 rails 36% (the
§13-planned fallback to canonical gm=-5 init materialized but was not taken);
never recovered. Product path NOT confirmed via VCA-boundary on this corner.

### §13.5 fork resolution

VCA moved NRMSE the wrong way on a railed init → per the pre-registered rule the
multiplicative mechanism must come from cells (S3-bistable revival) or the forget
gate — **Step 6 activates** (plan `narma-forget-gate` registered 2026-09-13:
softplus-unbounded gate `leak_eff_j(t) = softplus(a_j·V_u + c_j·x_j + b_j)`,
spec `forget-gate-leak`; design + audit now, E0/train gated). Caveat: VCA-1 ran
on a railed init, so "VCA moves nothing" is confounded by init — a canonical-init
VCA rerun remains a cheap open leg before cells.

### Open issue (2026-09-13): rail-pinned input drive

Targets are standardized (zero-mean/unit-variance, MSE in normalized space,
denormalized only at eval — no rail pressure on the output side). Inputs are
not: raw `u ∈ [0, 0.5]` is affine-stretched to the full bipolar swing
`V_u ∈ [−3, +3]` (`scale_input_to_rails`, `x_max = 3.0`) by construction, with
`--input-scale` as a post-hoc multiplier. Effective forcing is gain × swing
with both defaulting to max, and the E0 grids show the memory/rails cliff
exactly there (gm=-2 buys MC, 30–45% rails; gm=-5 rail-clean, MC ~1–2).
Round-2 killed drive *sparsity* as a lever but never isolated drive *amplitude*
at fixed gain (E4 screened 0.25/0.5/1.0 mixed with gain/leak). Candidate legs
(not built): E0 swing sweep at fixed gm=-5; statistically standardized `u`
(z-scored, then scaled to a chosen swing) instead of the fixed affine-to-rails
map; joint swing × polarity (unipolar shifts the operating point).

---

## 15. Discussion decisions log (2026-09-13)

1. **Torus-only.** Small-world ≈ torus to the second decimal (best MC 3.19 vs 2.75, identical verdict). Core-graph sparsity is not a lever; the sparsity that matters is input-map sparsity (§13.1).
2. **VCA-first, attribution post-hoc.** VCA-on-boundary already contains learned separability (per-edge embeddings differ from day 0), but weak at init, sharing one time-varying direction, and providing reweighting rather than true zeros. Running VCA before static masks is cheaper (plumbing only) and the instruments (product-pairs/IPC-2 vs PR) disentangle the mechanisms afterward.
3. **Forceful-drive suppressor hypothesis.** Full fan-out may leave recurrence nothing to mix (identical forcing → correlated states → difference-driven core currents carry noise). Sparse drive gives undriven nodes a recurrence-only information path; the gate-zero flip is the discriminator.
4. **VCA-1 init: best-corner** (gain -2 / randomized / drive 0.25), canonical fallback on day-0 rail crash.
5. **Parallel tracks, sequential VCA trains** (VCA-1 full-drive, then plain-sparse control, then VCA-2), so each lever is isolated by an on/off ablation.
6. **Sparse drive dropped (post-59587991).** Sparse best-MC 1.24 ≤ dense 1.98 with no gate-zero flip; sparse trains gated out. Full fan-out retained as canonical.
7. **VCA-1 negative on railed init (post-59587991).** Δ +0.1026 vs control; day-0 36% rails, fallback not taken. Product path unconfirmed; canonical-init VCA rerun stays a cheap open leg.
8. **Step-6 trigger met (post-59587991).** §13.5 fork resolves to forget gate; `narma-forget-gate` + `forget-gate-leak` registered (softplus-unbounded, build gated).
9. **Rail-pinned input drive flagged (2026-09-13).** Inputs stretch to full ±3 V rails by construction while targets are standardized; gain × swing both default to max at the memory/rails cliff. Amplitude-at-fixed-gain never isolated; candidate E0 swing-sweep / standardized-`u` legs recorded in §14, not built.
