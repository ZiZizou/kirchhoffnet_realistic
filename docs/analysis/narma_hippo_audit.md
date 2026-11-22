# Audit: NARMA Memory Campaign and the HiPPO / Legendre (LDN / LMU) Thread

**Audit date:** 2026-11-21
**Repo:** `C:\Users\Atharva\Documents\kirchhoffnet_realistic`
**Branch audited:** `simplified` (current). Branches existing: `simplified`, `remotes/origin/main`, `remotes/origin/deq`, `remotes/origin/simplified`.
**Commits in history:** 265 (`git rev-list --all --count`), earliest `e177b0b` (2026-06-14, "first commit, working 3 stage pipeline for smooth2d_grid problem").
**Audit mode:** read-only. No tracked file was modified. The only files written are under `tmp/analysis/` (this report and the extracted PDF text).

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Did NARMA succeed? | **No.** Best KNet NRMSE on NARMA-10 = **0.537** (Phase-7/8 banded-leak core) vs a tuned in-harness ESN reference of **0.2964** and a published success band of 0.15–0.29. The pre-delay-bank fabric plateaued at 0.657–0.79. |
| What was the root cause? | **Not nonlinearity — memory persistence, then (after the delay bank was added) driven-state collapse.** Unsaturated fabric holds memory capacity (MC) ≈ 2–3 max; the task needs MC ≈ 12; NRMSE floor ≈ 0.67 corresponds exactly to MC ≈ 2. Memory and rail saturation were coupled. The final Phase-9 finding reframed it: the fabric supplies only **~3–5 units of genuine recurrent memory**; an explicit **8-tap delay bank supplied the rest**. |
| Which core code exists only for NARMA? | 5 clear families (listed in §3.3): `core_refresh_interval`, `node_activation` + `allow_experimental_cells`/`no_clip`, `allow_boundary_target_overlap`, `dynamic_leak`/`dynamic_leak_input_proj`, `TapRails`/delay-line rail gating, plus the `make_narma_preset` preset. `enable_ref_edges` and persistent drive **predate** NARMA and are not NARMA features. |
| Was HiPPO/Legendre/LDN/LMU ever implemented? | **No. Nothing was implemented.** Zero occurrences of `hippo`, `legendre`, `LDN`, `LMU` in **any of the 265 revisions** on **any** branch (verified with `git grep -a -l -i -E "hippo|legendre|\bLDN\b|\bLMU\b" $(git rev-list --all)` → exit 1, no output). No `.py` defines an LDN/LMU/HiPPO/Legendre/gamma/Laguerre operator. No run outputs exist. |
| Where does the HiPPO thread live? | In **planning documents only**: `docs/NARMA Fabric Memory Progress Summary.md` (a chat transcript that ends mid-sentence at line 536 — this is literally where the thread was lost), `docs/ideas.md`, `docs/Niche Research - ...LDN Pruning....md`, `docs/Math-Side Operators....md`, `docs/Analog-Native RNN Features and Gap List.md`. Plus two **untracked** root files: `how_to_train_your_hippo.pdf` and `run_knet_HiPPO.sh`. |
| Is `run_knet_HiPPO.sh` runnable? | **No.** It is a copy of the canonical Phase-A CTLE BO script with the body replaced by `python test_memory_knet.py`, and `test_memory_knet.py` **does not exist** in the tree and **never existed** in any revision. It never ran. |
| What is salvageable? | The NARMA *measurement methodology* and the *negative knowledge* (memory–rail coupling law; the "fabric = nonlinear expander, not memory" verdict; the tap-bank cost frontier). The HiPPO/Legendre thread is salvageable as a **plan** (Phase 12 E0–E6) but not as code. |

**Important repo-hygiene finding (affects durability, not science):** `.gitignore` contains `*.md`, `*.sh`, `*.out`, `*.log`, `docs`, `output`, `results`. Therefore **almost the entire NARMA documentation and all ~18 `narma_*_batch.sh` scripts have never been committed**. `git ls-files docs/` returns exactly one file (`docs/gln-learned-sharpness.md`). The 8 tracked `narma_*.py` files are the only durable NARMA artifact. `git status --porcelain` shows `M ctle_kirchhoff_bo.py` plus untracked `catboost_info/`, `how_to_train_your_hippo.pdf`, `kirchhoffnet/`, `tmp/`, `tree_rematch/`.

---

# PART 1 — NARMA: chronological account

## 1.0 Metric conventions used throughout

From `docs/narma-progress-summary.md:328-331` and `docs/narma-fabric-memory-problem-statement.md:28-30`:

- **NRMSE** = RMSE / target-std. 0 = perfect, **1.0 = mean predictor**. Lower is better.
- **R²** = 1 − SSE/SST.
- **MC (memory capacity, Jaeger)** = Σ_{k=1..20} R² of a ridge reconstructing `u(n−k)` from states. Healthy 5–20+; gate ≥ 4; C0-ESN reference 8.07; task needs ≈ 12.
- **PR (participation ratio)** = (Σλ)²/Σλ² of the washed-state covariance. ≈1 = all nodes move together. **Demoted to diagnostic** after a healthy-MC ESN scored PR ≈ 3.
- **rail_frac** = fraction of state entries with |x| > 0.9·x_max (gate ≤ 5 %; later relaxed to ≤ 0.20 for the delay-bank phases).
- **activity** = mean |x| / x_max. Target band 0.45–0.55 (Phase 6/7/8).
- **IPC-2** = quadratic information-preservation capacity; implemented in `narma_revised_plan.py:155` (`ipc2_curve()`), regressing against the centered `u(n−k)²` (the `P₂` Legendre identity in practice).

**Success band:** 0.15–0.29 NRMSE (standard-RNN parity), set in `docs/narma-fabric-memory-problem-statement.md:12`.

## 1.1 Phase 0 — first architecture, first plateau (2026-09-03 → 2026-09-06)

**Architecture** (`docs/narma-progress-summary.md:362-394`): single stage, 25 hidden nodes on a directed 5×5 torus, 100 edges, `tanh_free` cells, ~1261 params, Heun `t_span = 1.0`, `num_steps = 8` per sample (dt = 0.125), state (26-D: 25 hidden + 1 output accumulator) carried across samples. Node ODE: `dx_j/dt = Σ_in I_edge − Σ_out I_edge + I_boundary + I_readout − leak_j·x_j − clip(x_j)`. Init `gm_raw = isat_raw = −5.0` → gm ≈ 0.077 (deliberately weak/near-linear), `g_raw = −5.0` → G ≈ 0.0067, `raw_leak = −3.0` → leak ≈ 0.05 (≈0.97 retention/sample), `x_max = 3.0`. Input enters **only** through 25 boundary OTA edges, affine-stretched to the full bipolar ±3 V swing.

**Runs:**
- `narma_exp.log` (root; 12 481 lines) — the very first run, `core_refresh_interval=4`, 4 streams × 2500 samples, TBPTT chunk 25, 80 epochs. Ended `*** JOB 58026190 ON fc10719 CANCELLED AT 2026-09-03T20:33:08 DUE TO TIME LIMIT ***`. No final NRMSE.
- `narma_diag_latest_log.out` (root) — the diagnostic batch. `E3-k8`: `RESULT: fabric_refresh_k8 seed=0 NRMSE=0.6783 R^2=0.5394 MC=0.19 params=1261 epochs=80`; untrained ridge diagnostic `NRMSE=0.7194 R^2=0.4824`.

**Phase-0 results table** (`docs/narma-fabric-memory-problem-statement.md:61-70`):

| leg | NRMSE | R² | MC | note |
|---|---:|---:|---:|---|
| k2 baseline (80 ep, early-stop ep54) | 0.6762 | 0.5421 | 0.23 | loss 0.99→0.58, flat from ~ep40 |
| k4 | ~0.67 | — | — | killed by 2 h limit |
| **E3 k8** | **0.6783** | 0.5394 | **0.19** | frozen baseline; rail 0 %→11.2 %, max\|x\|/x_max → 1.31 |
| untrained ridge (k8) | 0.7194 | 0.4824 | — | training buys ≈0.05 |
| E4a mapper LR ×10 (35 ep) | 0.6710 | 0.5492 | 0.22 | nothing |
| E4b drive ×0.5 (35 ep) | 0.6726 | 0.5471 | 0.18 | half swing, rail 1.3 %: nothing |
| **E4c leak 0.15 (35 ep)** | **0.6571** | 0.5677 | 0.19 | best by 0.02 = single-seed noise |
| E4d combo (35 ep) | 0.6650 | 0.5572 | 0.21 | no synergy |

**Carry-mask ablation (E2)**, same file lines 72-80: full 0.6762 / MC 0.23 → reset-hidden (keep accumulator only) **0.8703 / MC 0.00** → reset-all **1.0197 / MC −0.04**. So state is load-bearing but carries no task-useful memory.

**Ruled out** (lines 84-91): recurrence granularity (k = 2/4/8 identical), readout LR, drive strength, leak timescale, resetting-as-feature, under-training.

**Best Phase-0 NRMSE = 0.6571** vs target band 0.15–0.29. **Fails.**

## 1.2 Phase 1 — measurement-first ordering (`docs/narma-revised-plan.md`, 2026-09-12/13)

Execution order S1–S7 (`docs/narma-revised-plan.md:10-20`). Results in §12 of the same file and `docs/narma-progress-summary.md:435-461`.

| step | Result | Citation |
|---|---|---|
| S1-extended (frozen ckpt, u-MC / y-MC / IPC-2 / product pairs) | u-MC 2.56 (k2) / 2.80 (k8); **y-MC 7.10 / 6.79**; IPC-2 0.39 / 0.47; **all 7 product pairs 0.01–0.03 → lag-9 product path provably absent** | `narma-progress-summary.md:446`, `:660-661` |
| S2a tuned ESN (288 corners) | **0.2964** → band stands; C0-ESN MC 8.07 validates the meter | `narma-progress-summary.md:447`, `:662` |
| S3 dense trained readout | 0.6773 / R² 0.5408 / MC 3.46 (1080 params) ≈ temporal 0.676 → **H3 (readout rank) demoted** | `narma-progress-summary.md:448` |
| S2b torus E0 (48 corners, gm × leak × drive) | **0/48 pass**; best rail-sane gm=−5/hetero/drive-1.0 = 0.6606 / MC 1.98 / PR 1.17 / 0 % rails; gm=−2 peaks MC 2.6–2.75 at 31–42 % rails | `narma-progress-summary.md:449`, `:664-665` |
| S2c small-world (48 corners) | **0/48**; best MC 3.19 / ridge 0.64 → topology ≈ torus | `narma-progress-summary.md:450` |
| Round-2 E0 dense (18) / sparse (18) | dense best MC 1.98; sparse best MC 1.24 with **no gate-zero flip** → **sparse drive killed** | `narma-progress-summary.md:451` |
| Round-2 trains | plain-full 0.6834 (3rd reproduction of plateau); **VCA-1 0.7860 (Δ +0.10, day-0 rails 36 %)** → VCA hurt | `narma-progress-summary.md:452` |

## 1.3 Phase 2 — the gating campaign (`knet-gated-memory`)

| leg | Result | Citation |
|---|---|---|
| **Split** `narma_split-59625759.out` | base 0.7327 / MC 1.58 / PR 1.06 / rails 15.2 % / 5 loud nodes; **arm A** (cut drains) 0.7985 / rails 19.1 %; **arm B** (cut amplifiers) 0.8129 / MC 0.39 / rails 0 %; **arm C** (both) 0.8122 → **B == C, A ≈ base ⇒ all memory lives in the tanh amplifier branch; the shunt carries nothing** | `narma-progress-summary.md:470-484` |
| **Transplant** (54 corners, `59667050`) | **DECISION: TRANSPLANT-DEAD**, 0/54. Best rail-free 0.6186 / MC 2.90 / PR 1.30 / 0 % rails. Highest MC anywhere **3.50 at 34.5 % rails** (disqualified). **isat is the memory axis**: every high-MC corner uses isat=−2; every isat=−6 corner collapses (MC 0.1–0.8, PR 1.00) | `narma-progress-summary.md:501-508` |
| **Phase-1 E0** (48 corners, `59691417`) | **0/48**; best gm−1/isat−5 = 0.6347 / MC 2.20–2.26 / PR 1.31–1.49 / 0 % rails. Clean gradient (hotter = better ridge + more MC) but PR never > ~1.5 | `narma-progress-summary.md:515-517` |
| **Hotter extension** (18 corners, `59712084`) | first MC > 3 (3.14) at **30.7 % rails**; rail-free cap stays MC ≈ 2.3 | `narma-progress-summary.md:519-522` |
| **Spike trains** A/B/C/D × 20 ep (`59691420`) | A 0.7549 / MC 3.03 / 0 % rails; **B/C/D all 0.6697 / MC 1.9–2.1 / 15–18 % rails**; **B/C/D trajectories mutually digit-identical ⇒ dynamic leak and hidden-VCA never engaged as gates** | `narma-progress-summary.md:524-529` |
| **Rung-1 native-linear** (`59724426`) | **MC 0.003–0.005, J max\|λ\| 0.0046** (0.0014 at 16 steps), state RMS ≈ 3e-4, vs **ESN ref MC 12.01 / 0.4588** on identical streams. Finer steps and 7× hotter gm move nothing. The per-sample map forgets ~99.9 % (leak 6.9) | `narma-progress-summary.md:531-537` |
| **Depth** (1/2/3-stage, `59719147`) | s1 leg **rc=124 TIMEOUT**; s2 at ep40 tracks s1 (loss ≈0.60, NRMSE 0.68, rails 2→4 % climbing) → **no depth dividend** | `narma-progress-summary.md:539-544` |

**The 0.67 floor** (`narma-progress-summary.md:546-549`): *"best NRMSE anywhere is 0.6697 and every gain arrives through saturation while unsaturated MC caps at ~3. The floor IS the memory ceiling in target units — fitting NARMA-10 (which needs MC ~12) with MC ≈ 2 caps at ~0.67 regardless of lever."*

## 1.4 Phase 3 — the causal delay bank (the architectural pivot)

### Phase 5 (delay bank + 200 boundary OTAs)
`docs/narma-phase6-core-eigen-problem-statement.md:68-79`:
- A deliberately **nonphysical dense recurrent crossbar** with the same 8 delays (ceiling control, not an allowed implementation): MC 8.77, state PR 5.66, NRMSE 0.555.
- The **physical KNet delay-bank** experiment: **MC 8.09, NRMSE 0.586** — vs ridge on the raw 8 delay taps **0.812**. So the KNet core contributes real computation. But it ran hot: amplitude 0.689, rails 0.284; boundary-injection PR 6.04, state PR 1.89.

### Phase 6 (calibration + topology sweep) — the "central diagnostic paradox"
`docs/narma-phase6-core-eigen-problem-statement.md:81-127`. Boundary `isat_raw` mean −4.4 → amplitude 0.496, rails 0.161. Five topologies, three seeds each:

| Core topology | edges | NRMSE | MC | state PR | amplitude | rails |
|---|---:|---:|---:|---:|---:|---:|
| Directed 5×5 torus | 100 | 0.598 | 7.90 | 1.73 | 0.496 | 0.161 |
| Bidirectional 5×5 torus | 200 | 0.602 | 7.83 | 1.77 | 0.489 | 0.156 |
| Small-world k=8 p=.25 | 100 | 0.600 | 7.88 | 1.77 | 0.486 | 0.156 |
| Small-world k=8 p=.50 | 100 | 0.604 | 7.80 | 1.74 | 0.499 | 0.162 |
| Small-world k=8 p=.25 bidirectional | 200 | 0.598 | 7.88 | 1.79 | 0.493 | 0.158 |

Zero-point diagnostics (lines 105-114): all 25 Jacobian eigenvalue magnitudes > 0.8; eigenvalue-magnitude PR ≈ 24.96; input-Jacobian spatial PR ≈ 5.44; finite-horizon (20-sample) controllability-Gramian PR ≈ 5.9–6.2; driven local Jacobians ≈24.8/25 magnitudes > 0.8, mean spectral radius ≈ 0.96; nonnormality proxy ≈ 1e-4. **Yet actual washed state covariance PR = 1.7–1.8.**

### Phase 7 (interleaved three-band leaks) — `docs/narma-progress-summary.md:204-303`
Baseline = existing log-uniform leaks ≈ 0.06–0.16. Banded = spatially interleaved nodes (0,3,…,24 slow leak 0.051; 1,4,… medium 0.223; 2,5,… fast 0.511). Uniform control = leak 0.18.

| Condition | NRMSE | R² | MC | raw PR | corr PR | lag-9 product R² | activity | rails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5981 ±0.0069 | 0.6421 | 7.904 | 1.728 | 2.322 | 0.4667 | 0.496 | 0.161 |
| **Banded leaks** | **0.5372 ±0.0057** | **0.7113** | **8.737** | 1.945 | **3.465** | **0.5198** | 0.401 | 0.103 |
| Uniform leak 0.18 | 0.5785 ±0.0123 | 0.6650 | 8.112 | **2.131** | 2.774 | 0.4761 | 0.378 | 0.082 |

Realized poles: banded slow 0.93–1.00, medium 0.75–0.83, fast 0.57–0.62; uniform narrowed to 0.80–0.88. Boundary net-current PR ≈ 6.01 unchanged in every condition ⇒ the gain is a core-timescale effect, not richer injection. **Verdict (line 288): "promising mechanism demonstrated; predeclared screening win not yet earned"** — it failed the raw state-PR ≥ 2.5 screen and ran outside the 0.45–0.55 activity band (0.401).

### Phase 8 (activity-restored banded) — `docs/narma-progress-summary.md:45-112`
One-dimensional rescale `leak_j(α) = α·leak_j(banded)`; α calibrated on an **input-only** stream to 0.490 ± 0.010 mean |x|/rail (targets and task scores not used). Uniform control calibrated the same way.

| Condition | NRMSE | MC | raw PR | corr PR | lag-9 product R² | activity | rails |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-6 baseline | 0.598 | 7.90 | 1.73 | 2.32 | 0.467 | 0.496 | 0.161 |
| Banded α=1 (banked ref) | **0.537** | **8.74** | 1.94 | **3.46** | **0.520** | 0.401 | 0.103 |
| Banded, activity-restored | **0.556** | **8.48** | 1.88 | **3.24** | **0.496** | 0.453 | 0.131 |
| Uniform, activity-matched | 0.603 | 7.82 | 1.77 | 2.35 | 0.466 | 0.460 | 0.137 |

**Mechanism verdict: passed** (all three paired screening decisions are wins). **System-level verdict: not yet ESN-competitive** — tuned in-harness ESN ≈ 0.296, so the gap is ~0.24 NRMSE. `docs/narma-phase8-temporal-diversity-problem-statement.md:8-13` states this as the bottom line: *"Phase 8 is a mechanism win, not an ESN-parity win."*

### Phase 8B (tied causal tap-gate random existence probe) — `docs/narma-progress-summary.md:116-200`
`tap_rails.py` (`TapRails`): B shared rails from the 8 present/past taps; `z_j = sigmoid(β_j + Σ_b w_jb r_b)`, `m_j = 2(1−z_j)`; `m` multiplies the node's complete post-KCL current and its leak, clip stays outside.

| random mix scale | NRMSE | MC | corr PR | void-node fraction | result |
|---:|---:|---:|---:|---:|---|
| identity | 0.5560 | 8.478 | 3.235 | 1.00 | Phase-8 reference |
| 0.02 / 0.05 / 0.10 | 0.5562 / 0.5560 / 0.5566 | 8.473 / 8.468 / 8.436 | ~3.235 | 1.00 | effectively off |
| 0.30 | 0.5597 | 8.340 | 3.237 | 1.00 | worse, still void |
| 1.00 | 0.5638 | 8.278 | 3.242 | 0.77 | gates begin to move, but worse |

Void rule `std_t(z_j) < 0.02`; largest observed node gate std at scale ≤ 0.30 was 0.0155. **Decision: do not launch the 3 seeds × 3 rail-counts × all-controls gate-training matrix on this probe alone** (line 179-200).

### Phase 9 — digital node-tanh twin (architecture inversion) — `docs/narma-progress-summary.md:747-889`
`twin_node_tanh()` in `narma_linear_controls.py`: dense linear mixing → sum → **per-node** tanh, once-per-sample discrete map, no Heun, no stages, no training, ridge readout.

| sweep | harness | NRMSE mean / best | MC mean / max | verdicts |
|---|---|---|---|---|
| coarse (144) | taps8 | 0.647 / 0.505 | 8.10 / 10.29 | 0 pass, 0 fail |
| coarse | scalar | 0.688 / 0.547 | 4.65 / 8.94 | 7 fail* |
| **fine (369)** | taps8 | **0.536 / 0.3926** | **9.35 / 13.11** | **78 pass**, 0 fail |
| fine | scalar | 0.632 / 0.547 | 7.35 / 8.94 | 0 pass |

Best corner: taps8 r=1.0/a=1.0/is=0.1 → **0.3926 / MC 12.45 / PR 7.17** (parity ESN 0.5219 same stream). Clean gradients: quieter drive monotonically better; a=1.0 maximizes memory; radius nearly flat.

**Recontextualization (2026-09-18), lines 834-889 — this is the most important paragraph in the whole campaign:**
- Good two-bar performance appears **only with the explicit 8-tap delay bank**. Flow twin, config-matched: taps8 MC **8.27** vs scalar **4.33** (Δ +3.94), scalar verdict `DIGITAL_ONLY`, taps8 `FLOW_VIABLE`. Discrete twin, matched: **8.90** vs **4.53** (Δ +4.37).
- The taps8 drive hands the net `u[t..t−7]` directly (`flow_twin_drive` at `post_twin_gates.py:155`; `twin_delay_bank` at `narma_linear_controls.py:3916-3930`), so **delays 1–7 are reconstructible from the input alone — up to +7 MC points are free**. MC > 8 therefore proves only ~1+ unit of genuine recurrent memory.
- **ESN-parity correction:** `ne.ESN` is scalar-only by construction (`self.W_in = torch.randn(n_reservoir, 1, ...)` at `narma_experiment.py:262`), so the earlier "beats parity in virtually every group mean" line compares a twin with 8 taps of free history against a tap-less ESN — **flattered by the free window**. Drive-matched ranking is ESN (all genuine) > scalar twin (all genuine).
- **Standing:** *"the fabric as nonlinear expander, not memory"*; the net contributes ~3–5 units of genuine recurrent memory; the delay bank contributes the rest.

### Phase 10 — aux-core-drive (eval-only twin) — `docs/narma-progress-summary.md:893-1023`
Question: can a learnable feedforward aux core (input→aux→main) replace the static 8-tap bank at sub-bank edge cost? Flavors: **echo_state** (8-node ESN, radius 0.95, leak ∈ {0.3, 1.0}), **shift_register** (exact chain, gain ∈ {0.7, 0.9}), **slow_feature** (8 decoupled leaky integrators, log-spread τ ∈ [1, 40]).

| Flavor | aux MC | taps8 MC | scalar MC | edge factor | tap-subspace R² | verdict |
|---|---:|---:|---:|---:|---:|---|
| echo_state (best) | 8.13 | 11.47 | 5.53 | 1.49 | 0.902 | **KILL** |
| shift_register (best) | 11.40 | 11.47 | 5.53 | 1.17 | **1.000** | **KILL** |
| slow_feature | 6.40 | 11.47 | 5.53 | 1.04 | <0.9 | **KILL** |

The shift register already satisfies SUCCESS path 1 (MC 11.40 ≥ 0.85 × 11.47, edge 1.17 ≤ 1.50) but dies on the tap-subspace rule: *"the aux IS the taps bank, just stored more expensively."* All three emit **KILL**; no `DifferentialStage` surgery started.

### Phase 11 — aux-core-drive-2 — `docs/narma-progress-summary.md:1027-1139`
- **D1 concat readout** `[main; aux]` (33 features): in-sample ridge 0.528→0.453 and MC 11.38→**12.39** for SR. **Held-out the gains scatter ±0.05 around zero** (SR ≈ +0.01, SF ≈ −0.03) ⇒ overfitting, not information.
- **D2 sparse main fan-out** k ∈ {3,6}: SR at k=3 totals **32 edges vs 200** — still dies on rule (i) (confirmatory, fan-in-independent copy-ness).
- **D3 tapsVD4** (SVD top-4 PCs of the tap bank): MC **9.21** vs 11.47 (0.80 < 0.85 gate) ⇒ **the bank is not compressible to 4 PCs at preserved MC**; the full 200-edge reference stands.
- **D4 residual-distilled aux**: teacher-selected echo lands at tap-R² ≈ 0.90 → KILL. *"What main misses turns out to be more delays."*
- **Standing (lines 1129-1139):** all flavors all legs **KILL**. *"The bank's memory load is real; it is incompressible below ~8 taps at preserved MC; miniature learnt replacements either recapitulate it or underperform it; direct aux readout adds fittable parameters, not transferable memory."* Next memory work should target **making the bank cheaper** (fewer taps, sparser fan-out — both unrun), not replacing it. Every absolute MC in the document (taps8's 11.47 included) is fit-and-scored in-sample and therefore optimistic.

### 1.5 Root-cause diagnosis — what actually went wrong

The campaign went through **four successive root-cause hypotheses**, each falsifying the previous one:

**RC-1 (Phase 0–2): "reservoir is near-linear and weak at init" (H1).** gm ≈ 0.077 at init ⇒ ~100 edges couple nodes through a tiny-gain, almost-linear map — a slow linear integrator, not a reservoir. Training then grows gains straight into compliance shutdown. Evidence: untrained ridge ≈ trained score (0.7194 vs 0.6783); E4b (gentler drive) can't fix what init lacks; the split experiment localized all memory in the tanh amplifier branch (arm B ≈ arm C), the shunt carries nothing. **Status: confirmed as a contributing cause but not sufficient.**

**RC-2 (Phase 2): "memory persistence, not nonlinearity."** Five independent screens agree the **unsaturated** fabric holds **MC ≈ 2–3 max**, while the task needs **MC ≈ 12** and the NRMSE floor sits at **0.67 exactly where an MC ≈ 2 fit caps out** (`docs/narma-progress-summary.md:686-692`). The coupling law: **every corner with MC ≳ 2.5 rails at ~15–35 %; every rail-free corner caps at MC ≈ 2.3.** "Unsaturated = amnesic, memory = saturated" is the standing description of the failure mode (`docs/narma-progress-summary.md:694-699`). **Status: confirmed and is the primary root cause of the 0.67 plateau.**

**RC-3 (Phase 6): "too few long-lived modes" is rejected.** With the delay bank, the core had 25 near-unit Jacobian eigenvalues (eigen-PR ≈ 24.96), input-Jacobian PR ≈ 5.44, and controllability PR ≈ 5.9–6.2 — and *still* driven state PR only 1.7–1.8. Bidirectional and small-world topologies changed nothing (all 0.598–0.604). The missing property is **separable, driven temporal states** — Phase 6 gave many modes with *similar* retention times, so shared delayed drive created similar histories. **Status: confirmed; motivates Phase 7/8.**

**RC-4 (Phase 7–8): the fix partly works but not enough.** Deliberate slow/medium/fast leak bands raised correlation PR 2.32 → 3.46, MC 7.90 → 8.74, lag-9 product R² 0.467 → 0.520, and dropped NRMSE 0.598 → 0.537 (α=1) / 0.556 (activity-restored), beating an activity-matched uniform control in every paired seed. **Status: mechanism confirmed, ESN parity NOT achieved.**

**RC-5 (Phase 9–11, the true reframing): the delay bank was doing the remembering.** The 8-tap bank contributes **up to 7 "free" MC points** (delays 0–7 are reconstructible from the input alone); the fabric itself contributes only **~3–5 units of genuine recurrent memory** (scalar-drive MC in both campaigns, both maps). Every attempt to replace the bank with a learned aux either (a) *recapitulated* it (shift-register tap-R² = 1.000; D4 residual-distilled ≈ 0.90) or (b) underperformed it (echo 8.13, slow-feature 6.40 vs taps8 11.47). And the bank is **incompressible below ~8 taps** (tapsVD4 → MC 9.21, ratio 0.80 < 0.85 gate).

**Instrument bugs that caused early mis-readings** (worth fixing in the simplification pass):
1. The legacy `memory_capacity` convention paired past-states with future-targets (forward prediction); a noiseless delay line scored MC ≈ 0.02 under it vs 1.00 after the fix. All pre-v5 "MC ≈ 0.2" values were artifacts (`docs/narma-progress-summary.md:595-599`).
2. The MC collector originally started from zero state with **no washout discard**, underestimating true MC (`docs/narma-fabric-memory-problem-statement.md:29`).
3. The ±0.05 determinism drift gate on the 5-number split verdict was too tight for a 20-fit-sum MC meter and produced a spurious "split-verdict-void" (`docs/narma-progress-summary.md:486-490`).
4. In-sample fit-and-score inflates every MC number, including the taps8 reference 11.47 (`docs/narma-progress-summary.md:1136-1139`).
5. ESN parity comparisons were apples-to-oranges (scalar-only ESN vs taps8-driven twin) until corrected at Phase 9 (`docs/narma-progress-summary.md:870-880`).

## 1.6 Scoreboard — best NRMSE per phase vs targets and baselines

| Phase | Configuration | Best NRMSE | MC | Target | Reference |
|---|---|---:|---:|---|---|
| 0 | baseline k8/k2/k4, tanh_free, 25 nodes | 0.6762 (k2) | 0.23 | 0.15–0.29 | ESN band uncalibrated |
| 0 | E4c leak 0.15 | **0.6571** | 0.19 | 0.15–0.29 | ridge-30-tap 0.38 |
| 1 | S3 dense readout | 0.6773 | 3.46 | 0.15–0.29 | — |
| 1 | S2b torus E0 best (rail-sane) | 0.6606 | 1.98 | 0.15–0.29 | ESN (untuned) 0.5596; LSTM 0.2965 |
| 2 | Phase-1 E0 best (0 % rails) | 0.6347 | 2.20–2.26 | 0.15–0.29 | — |
| 2 | Transplant best rail-free | 0.6186 | 2.90 | 0.15–0.29 | ESN **0.2964** (tuned, 288 corners) |
| 2 | Spike train B/C/D (20 ep, railed) | **0.6697** | 1.93–2.06 | 0.15–0.29 | best *trained* fabric number; rails 15–18 % |
| 3 | Phase 5 physical KNet + 8-tap bank | 0.586 | 8.09 | 0.15–0.29 | nonphysical dense control 0.555; ridge-on-taps 0.812 |
| 3 | Phase 6 directed torus (calibrated) | 0.598 | 7.90 | 0.15–0.29 | ESN 0.2964 |
| 4 | Phase 7 banded leaks | **0.5372** | 8.737 | 0.15–0.29 | uniform-leak control 0.5785 |
| 4 | Phase 8 activity-restored banded | 0.556 | 8.48 | 0.15–0.29 | ESN **0.2964** |
| 4 | Phase 8B tied gate (best) | 0.5560 (identity) | 8.478 | — | all gated variants ≥ 0.5597 (worse) |
| 5 | Phase 9 twin, fine taps8 (mean / best) | 0.536 / **0.3926** | 9.35 / 13.11 | NRMSE ≤ 0.50 & MC ≥ 5 → **78 PASS** | parity ESN 0.660–0.763 (flattered) |
| 6 | Phase 10 aux cores | — | aux 6.40–11.40 | ≥ 0.85 × taps8 at ≤ 1.5× edges | **all KILL** |
| 7 | Phase 11 aux-core-drive-2 | — | tapsVD4 9.21 | 0.85 ratio | **all KILL** |

**Overall:** best *fabric* result = **0.537** (Phase 7/8 banded-leak KNet, which is memory-assisted by an external 8-tap bank). Best *digital twin* result = **0.3926** (Phase 9), still behind the tuned ESN 0.2964 and the 0.15–0.29 band.

## 1.7 Root-cause code features added to the core model **solely** for NARMA

### 1.7.1 Core model (`config.py`, `differential_stage.py`, `topology.py`, `cell_library.py`) — removal candidates

| # | Feature | Exact locations | Added | Commit | Verdict / removal rationale |
|---|---|---|---|---|---|
| 1 | **`core_refresh_interval`** — recompute the frozen core current every *k* Heun steps | `config.py:1184` (preset kwarg), `config.py:1286`; `differential_stage.py:95` (docstring), `:238` (ctor), `:281-283` (validation), `:1763`; `topology.py` (4 sites) | 2026-09-03 | **`1164950`** "added infrastructure for narma experiment with variable refresh rate" | **Ruled out** — k = 2/4/8 all plateau identically (`narma-fabric-memory-problem-statement.md:33`). Self-described in commit message as "for narma". **Strong removal candidate.** |
| 2 | **`node_activation`** — bounded node broadcast `Y = x_max·tanh(x/x_max)` read by all state-side edges | `differential_stage.py:239` (ctor), `:275-280` (validation), `:1150` (use), `_node_broadcast()` helper; `topology.py` (5 sites) | 2026-09-14 | **`a02de37`** "Added new interactions and cell types" | Code comments literally say *"narma-node-activation plan"*. **Prime removal candidate.** |
| 3 | **`allow_experimental_cells`** + **`no_clip`** — the guard that keeps `node_activation` out of the static training path | `topology.py:1977-1978`, `:2062`, `:2099`, `:2109-2112`, `:2605` | 2026-09-14 | **`a02de37`** | Docstring: *"narma-cell-isolation spec"*. Exists only to fence the NARMA experimental cell. **Prime removal candidate** (goes with #2). |
| 4 | **`allow_boundary_target_overlap`** — several delayed boundary terminals may inject into the same hidden node | `topology.py:1759`, `:1908`, `:1949`, `:2015` (doc), `:2220` (guard) | 2026-09-14 | **`4e4751b`** "added machinery for delayed inputs" | Required only by the multi-tap delay bank. Docstring: *"multi-tap boundary banks may opt in to overlapping terminals"*. **Removal candidate if the delay bank goes.** |
| 5 | **`dynamic_leak` + `dynamic_leak_input_proj`** — input/state-modulated retention (the "forget gate"/"knet-gated-memory" mechanism) | `config.py:1186`, `:1301`; `differential_stage.py:211-212`, `:292-296`, `:405-411`, `:889-900`, `:1763`; `topology.py:1354-1355`, `:1701-1702`, `:1973-1974`, `:2096-2097`, `:2602-2603` | 2026-09-13 | **`e357c6c`** "Added ESN based rung tests" | The spike trains' legs **B/C/D were digit-identical to their ablations** ⇒ the gate never engaged (`narma-progress-summary.md:527-529`). Mechanically dead. **Strong removal candidate** — but note `docs/ideas.md` §2.4 and `docs/Math-Side Operators...` M4 want to *productize* it as "input-gated LDN" (Tier S). **Decision needed before removal.** |
| 6 | **`TapRails`** — low-rank causal tap rails for the Phase-8 tied write/retain probe | `tap_rails.py` (whole file, 62 lines; `TapRails` class, `tap_rails_param_count`); `differential_stage.py` (+38 lines in `8aff5c2`) | 2026-09-15 | **`8aff5c2`** "Added new delay line rail gating feature for narma" | Probe-only. Phase 8B found all nodes void for mix scales ≤ 0.30 and every gated variant worse than identity. **Strong removal candidate.** |
| 7 | **mix → bound dynamics / new cell types** | `cell_library.py` (+50), `topology.py` (+40), `differential_stage.py` (+15), `config.py` (+1) | 2026-09-14 | **`43d6be9`** "Added mix->bound dynamics" | Added for NARMA interaction experiments. **Candidate.** |
| 8 | **`make_narma_preset()`** + `PRESET_NARMA10` / `PRESET_NARMA20` + `narma10`/`narma20` registry entries + `narma_order`/`narma_x_max_rail` config keys | `config.py:1159-1321` (preset builder 1175-1303; presets 1306-1307; registry 1319-1320) | — | (NARMA campaign) | A whole NARMA-only preset inside the core config module. **Removal candidate if NARMA is dropped.** |

### 1.7.2 Features that **predate** NARMA — NOT NARMA features, do not remove on this basis

| Feature | Location | Added | Commit | Note |
|---|---|---|---|---|
| `enable_ref_edges` (reference nodes) | `differential_stage.py:133-145`, `:220`, `:522-526`; `topology.py:1064`, `:1420`, `:1521`, `:1710`, `:1760`, `:1847`, `:1909`, `:1950`, `:2043`, `:2584` | **2026-07-16** | `f097500` "Added reference nodes --enable-ref-edges" | Predates NARMA by ~8 weeks. |
| persistent drive (`DRIVE` dict, `drive_isat`, `raw_drive_g_init`, `drive_scales`) | `config.py:629-634` | pre-NARMA | `3db50bc` "Added persistent drive" | Predates NARMA. |

### 1.7.3 NARMA-only *probe* modules (tracked in git; separate from the core model)

`narma_experiment.py`, `narma_linear_controls.py`, `narma_revised_plan.py`, `narma_advisor_probes.py`, `narma_phase7_leak_bands.py`, `narma_phase8_tied_gate_probe.py`, `narma_phase8_tied_gates_audit.py`, `narma_spectral_norm.py` (8 tracked `narma_*.py` files), plus `post_twin_gates.py`, `audit_post_twin_gates.py`, `aux_core_probe.py`, `audit_aux_core.py`, `audit_twin_standalone.py`, `tap_rails.py`, `probe_audit1.py`, `probe_audit2.py`, `probe_audit_nodeact.py`, `probe_audit_revised.py`, `probe_audit_split.py`, `probe_audit_transplant.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `smoke_audit3.py`, `test_audit.py`.

Gitignored / untracked (never committed — durability risk): ~18 `narma_*_batch.sh` scripts, `narma_exp.log`, `narma_diag_latest_log.out`, `narma_debug.md`, all of `docs/` except `docs/gln-learned-sharpness.md`, and `outputs_alliance_can/` (21 `narma_*.out` job logs + `narma_phase7_leak_bands/`, `narma_phase8_activity_leaks/`, `post_twin_gates/`, `linear_controls/`).

### 1.7.4 What is worth keeping from the NARMA program (recommendation, see also §3)

1. The **measurement methodology** — E0 frozen-core screening, gate-zero / carry-mask ablations, MC + PR + lag-product probes, pre-registered gates with explicit KILL rules, held-out discipline. This is the reusable asset (`docs/Analog-Native RNN Features and Gap List.md:140` calls the "held-out MC + cost-matched baselines" methods contribution B3/C1).
2. The **negative knowledge**, which is sharp and citable: the memory–rail coupling law; "fabric = nonlinear expander, not memory"; the bank cost frontier (taps8 = 200 edges; tapsVD4 = 100 edges → MC 9.21); "the bank is incompressible below ~8 taps".
3. The **leak-band mechanism** (Phase 7/8) — a real, paired, activity-matched win (NRMSE 0.598 → 0.537) and the only core-level improvement that survived.

---

# PART 2 — HiPPO / Legendre (LDN / LMU): every trace

## 2.1 Summary table of all traces

| # | Artifact | Location | Tracked? | Nature | Status |
|---|---|---|---|---|---|
| 1 | `how_to_train_your_hippo.pdf` | repo root, 6.55 MB | **No** — `git status` shows `?? how_to_train_your_hippo.pdf`; `git log --all -- how_to_train_your_hippo.pdf` → empty | The paper | Reference reading; extracted to `tmp/analysis/how_to_train_your_hippo.txt` |
| 2 | `run_knet_HiPPO.sh` | repo root, 30 lines | **No** — gitignored (`.gitignore:20:*.sh`, confirmed by `git check-ignore -v`) | SLURM stub | **Never ran**; references a script that has never existed |
| 3 | Phase-12 plan (E0–E6 incl. E1 `ldn`/`gamma`/`laguerre`/`ms_gamma` flavors) | `docs/NARMA Fabric Memory Progress Summary.md:332-442` | No (`docs` is gitignored) | Plan | **Never started** |
| 4 | LDN/HiPPO prior-art reasoning | `docs/NARMA Fabric Memory Progress Summary.md:163-168`, `:203-207` | No | Reasoning | — |
| 5 | Truncated niches research (ends mid-sentence) | `docs/NARMA Fabric Memory Progress Summary.md:446-536` | No | Research | **This is where the thread was lost** |
| 6 | Living ideas doc, HiPPO/LDN/LMU idea families | `docs/ideas.md:12`, `:171-238`, `:268`, `:311-312`, `:406-409`, `:436-438`, `:449`, `:465`, `:484` | No | Ideas | Live; E1 leg still planned |
| 7 | Niche audit §(v) "LDN pruning", §(vii) state-SNR balancing | `docs/Niche Research - Reconfigurable Analog Fabric, OTA Linearization, LDN Pruning, Drift Calibration, State SNR Balancing.md:113-205` | No | Prior-art audit | — |
| 8 | Math-side operators M1/M4/M5/M7/M8 | `docs/Math-Side Operators, Full Ranking, CABs, Compiler-in-One-Fabric.md:25-45`, `:57-70` | No | Ranking | M4 (gated LDN) is Tier S |
| 9 | Analog-native features F2/F3/F5/F8/F9/F10 | `docs/Analog-Native RNN Features and Gap List.md:39-113`, `:157-182` | No | Gap list | F5 (Laguerre vs Legendre mismatch comparison) ranked #3 |
| 10 | Critique verification ("HiPPO is NPLR; S4D/S5 use diagonal …") | `docs/Critique Verification and Compiler Explainer.md:22`, `:83` | No | Verification | — |
| 11 | "Legendre" in the NARMA *measurement* code (IPC-2 detector) | `docs/narma-progress-summary.md:412-413`; implemented as `ipc2_curve()` at `narma_revised_plan.py:155` | code: yes (tracked) | **Diagnostic, not a memory operator** | Active and used |
| 12 | Any `.py` defining an LDN/LMU/HiPPO/Legendre/gamma/Laguerre operator | — | — | — | **not found** |

## 2.2 `how_to_train_your_hippo.pdf` — what it is

Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, Christopher Ré, *"How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections"*, arXiv:2206.12037v2 [cs.LG], 5 Aug 2022 (extracted text: `tmp/analysis/how_to_train_your_hippo.txt`, 1 991 lines; abstract at lines 21-36).

Summary of content:
- It is the **theoretical exposition of the HiPPO initialization** used by S4. S4 initializes the SSM state matrix `A` to a HiPPO matrix derived in earlier work for a *time-varying* system, with no known mathematical interpretation as a time-invariant SSM. This paper derives a general formulation with a clean interpretation: **S4's `A` produces Legendre polynomials under an exponential re-scaling** (HiPPO-**LegS**), i.e. S4 decomposes the input onto exponentially-warped Legendre polynomials (lines 25-35, 61-71).
- It generalizes to **HiPPO-LegT** (Legendre on *sliding* windows, a time-invariant operator / TOSSM) and **FouT** (truncated Fourier basis), and explains the initialization of the timescale parameter Δ (lines 61-71, 153-154, 275-294).
- **Crucially for this audit: the paper itself identifies the naming bridge** — the Legendre system *"has also been variously called the **Legendre Delay Network (LDN)** or **Legendre Memory Unit (LMU)**"* (extracted text lines 283-285). HiPPO-LegS is the *time-varying* original; HiPPO-LegT is the LTI version that is the LDN/LMU.
- Relevance to the user's thread: it is the canonical citation for *why* a Legendre/HiPPO operator is a good linear memory basis and how to build its `A`/`B` matrices. It is **not** a circuit paper and contains nothing about analog realizations, OTA fabrics, or pruning.

## 2.3 `run_knet_HiPPO.sh` — a dead stub

Full file (30 lines) read at `run_knet_HiPPO.sh:1-30`. Findings:
- Lines 1-15 are a **verbatim copy** of the canonical Phase-A CTLE KNet BO header (same `#SBATCH` block, same *"Canonical Phase-A KNet BO (plan canonical-ctle-unify)"* comment, same `--output=%x-%j.out`), i.e. copied from `run_phase_a_knet_bo.sh:1-15`.
- Lines 17-28 are the same module load / venv activation block as `run_phase_a_knet_bo.sh:17-28`.
- **Line 29 is the entire "body": `python test_memory_knet.py`.** (Compare `run_phase_a_knet_bo.sh:30-59`, which builds the canonical dataset and runs `kn_bayes_opt.py`.)
- **`test_memory_knet.py` does not exist** in the working tree (`Test-Path` → False) and **never existed in any of the 265 revisions** (`git log --all --oneline -- test_memory_knet.py` → empty). It is not in `git ls-files` and not in `git ls-tree` for `origin/main` or `origin/deq`.
- The file is **untracked and gitignored** (`.gitignore:20:*.sh`; `git check-ignore -v run_knet_HiPPO.sh` → `.gitignore:20:*.sh:run_knet_HiPPO.sh`).
- **It never ran.** There is no `run_knet_HiPPO-*.out` job log anywhere (not in `outputs_alliance_can/`, which contains 21 `narma_*.out` logs but nothing HiPPO-related), no `study.db` for it, no output directory.

**Conclusion:** `run_knet_HiPPO.sh` is an *aspirational filename on a copy-pasted CTLE BO script*. It is the only file in the repo whose name promises a HiPPO run, and it contains zero HiPPO content. It should be treated as noise and deleted (or, if a HiPPO leg is genuinely wanted, rewritten from scratch with real content).

## 2.4 What was planned for HiPPO/Legendre

### 2.4.1 Phase 12 — "Learned-Memory Aux Core" (`docs/NARMA Fabric Memory Progress Summary.md:332-442`)
Standing decisions (lines 336-342): rule (i) amended to a **cost-conditional copy rule**; **held-out scoring is primary**; per-seed tap-R² reported; the bar is the **cost-matched bank**, not taps8-200; frozen-aux screens are structurally unwinnable and no longer gate training.

| Leg | Content | Cost | Status |
|---|---|---|---|
| **E0** | Cost-matched bank baselines: `tapsVD-k` k ∈ {2..6}; `taps4`, `taps6`; `taps8-sparse` fan-out k ∈ {3,6}. Produces the MC-vs-edges Pareto frontier `R(edges)` that every later gate references. | ~1 GPU-h, eval-only | **not run** |
| **E1** | **Analytical aux insertion (the HiPPO/LDN/Legendre leg)**: `ldn` = Legendre Delay Network, orders d ∈ {4,6,8,12}, window θ ∈ {8,12}, **closed-form A/B matrices, no training**; `gamma` = de Vries–Principe gamma cascade K ∈ {4,8}, μ ∈ {0.3,0.6,0.9}; `laguerre` = first low-pass + all-pass cascade K ∈ {4,8}; `ms_gamma` = exponentially spreading time constants (the deliberate Phase-7 analog). Pre-registered readings: (a) if `ldn` d=8/θ=12 doesn't beat taps8 held-out at equal edges, the "optimal memory" framing needs recalibration; (b) the gamma/LDN rows anchor the related-work claims with in-harness numbers. | ~1 GPU-h, eval-only | **not run** |
| **E2** | Trained-aux pilot: one flavor only (`echo_state`, 8-node tanh, radius 0.95, leak 1.0); aux-local objective = reconstruct `u(t−k)`, k = 1..**12** (not 8 — taps8 already covers 0–7 while NARMA-10's product needs u(n−9)); arm B adds product targets `u(t−i)·u(t−j)` (the ROI probe); 3 seeds, ~30 epochs. Promotion gates: MC ≥ 0.85 × E0 frontier at equal edges, held-out NRMSE within 0.02 of the equal-edge bank, activity/rail guards. | ~1 GPU-day | **not run** |
| **E3** | Width sweep n_aux ∈ {2,3,4,6,8,12} vs **held-out NRMSE and per-delay reconstruction**. Pre-registered prediction: exact reconstruction of W delays needs n ≥ W (Krylov/controllability rank); below that error concentrates at large k, LDN-style. | ~0.5 GPU-day | **not run** |
| **E4** | Prune-and-realize: anneal-prune under an edge budget (`--sparsity`/`--device-l2-lambda`), map survivors onto `tanh_free` cells, check gm/Isat bounds, compliance margins, rail usage, ±10 % mismatch. PASS = pruned aux at ≤ 0.5 × frontier edges holding ≥ 0.85 × frontier MC under mismatch. | ~1 GPU-day | **not run** |
| **E5** | Multi-input: cross-NARMA M=2, `y` depends on `u₁(n−i)·u₂(n−j)`; bank cost 16 taps / 400 fan-in edges vs aux cost. | ~0.5 GPU-day | **not run** |
| **E6** | Fabric integration decision: aux replaces the boundary delay bank in the real Phase-6 KNet harness, paired seeds 0/1/2, vs the Phase-8 banded baseline. | — | **not run** |

The **novelty position** from the same document (lines 418-439) is the most important output for the user:

| Element | Prior art (cite, don't claim) |
|---|---|
| Cascade of leaky integrators as trainable memory (depth/resolution tradeoff) | gamma memory, de Vries & Principe 1992 (RTRL/BPTT) |
| Exponentially spread time constants (**= Phase-7 banded leaks**) | multi-scale gamma, Harris/Juan/Principe mid-90s |
| Optimal recurrent window memory | LDN/LMU (Voelker et al. 2018/2019), **HiPPO (Gu et al. 2020)** |
| Memory plane → nonlinear expander (= taps8 + main core) | focused gamma network; Mozer focused backprop |
| LDN on analog neuromorphic hardware | Braindrop deployment; Loihi LMU; Legendre-SNN on Loihi-2 |
| Analog gamma/Laguerre filters in silicon | Harris, Juan & Principe, fabricated adaptive chips |
| Heun-stepped neural ODE, "RNN as analog circuit" | neural ODE literature; analog reservoir computing |

### 2.4.2 Later elaborations (`docs/ideas.md`, `docs/Math-Side Operators...`, `docs/Niche Research...`, `docs/Analog-Native RNN Features...`)
- `docs/ideas.md:171-183` **§2.4 Input-gated LDN / liquid timescales** — *"LDN coupling matrix with input-gated leaks ('liquid Legendre memory'): structured memory + data-dependent timescales. Your `--dynamic-leak` mechanism + audit already built the circuit half."* Gap claimed as standing: neither liquid-unstructured (Hasani LTC/NCP) nor fixed-structured (pure LDN/LMU, Voelker 2019) literature owns gated LDN.
- `docs/ideas.md:185-192` **§2.5 Selective / learned-measure HiPPO** — Mamba-style input-dependent Δ (learned time-warping of the HiPPO measure) compiled to modulated bias currents; the claim is that **analog modulation paths cost wires + DACs**, and that wire-cost accounting is itself unpublished.
- `docs/ideas.md:203-207` **§2.7 Multi-window mixture multi-θ LDN** — block-diagonal A, one θ per block; "Math exists (S5 block-diagonal HiPPO-N init); analog + quantized + mismatch version does not. Cheap twin leg — do not standalone."
- `docs/ideas.md:226-238` **§2.10 Basis choice: Legendre vs Laguerre vs gamma** — learned-pole Laguerre (one repeated pole → best matching) at d = 6–12 vs Legendre-LDN on the *same* fabric under mismatch sweep; prediction that Laguerre degrades more gracefully because it has one error source.
- `docs/Math-Side Operators...:25` **M1 modal/biquad realization** — eigen-decompose the trained `A`; implement each complex-conjugate pole pair as **one OTA-C biquad** (the canonical analog block), each real pole as one integrator. *"the highest-ROI math idea: it converts 'arbitrary dense A' into 'a shopping list of biquads'."*
- `docs/Math-Side Operators...:31` **M4** — *"your `--dynamic-leak` spike already built the mechanism"*; the math-side flagship (Tier S #4).
- `docs/Math-Side Operators...:33` **M5** — selective/learned-measure HiPPO, with honest analog wire-cost accounting.
- `docs/Niche Research...:113-142` **§(v)** — *"LDN matrices are dense and tightly coupled — maybe via learned pruning we could remedy this — PARTIALLY WRONG; S4D already diagonalized HiPPO."* Cites Gu/Goel/Ré S4D (889 cites), Gupta/Gu/Ré (671 cites), Mamba unstructured pruning (Shihab et al. EMNLP 2025, 70 % param reduction), SparseSSM (ICML 2026). What survives: **device-constrained** pruning of LDN operators (not mathematical diagonalization), structured sparsity under mismatch sensitivity, and a hybrid of both.
- `docs/Niche Research...:174-205` **§(vii)** — Dynamic Range Compression & State SNR Balancing for polynomial orders d > 6. Cites Park et al. (arXiv 2024) on HiPPO-LegS numerical conditioning, Stöckel 2021, HiPPO 2020, balanced truncation (Moore 1981), Legendre-SNN on Loihi-2. What survives: **balanced realization of LDN operators for analog realization**, per-order gain tuning, compliance-aware scaling.
- `docs/Analog-Native RNN Features...:39-47` **F2 companding states** — *"nobody has built LDN/HiPPO-style polynomial memory out of companding integrators"*; subthreshold OTAs are nearly log-domain already.
- `docs/Analog-Native RNN Features...:67-73` **F5 learned-pole Laguerre** — Juan/Harris/Principe 1997-99 fabricated adaptive gamma and Laguerre filters (4-tap, system ID); the gap is learned-pole Laguerre at d = 6-12 **inside the fabric, with a same-fabric mismatch comparison against Legendre-LDN** — *"publishable on its own."*
- `docs/Analog-Native RNN Features...:175-182` — the recommended attack sequence: **F9 (passive SC bank) first → F5 (Laguerre vs Legendre) second → F2/F3 → core program → F7 (analog DFE) → B1/B2 → F4 → D1 (TinyTapeout)**.

## 2.5 Was anything implemented? — **No. Verdict: not found.**

Exhaustive evidence:

1. **Content search across all history.**
   - `git log --all -S "hippo" -i --oneline` → **empty**.
   - `git log --all -S "legendre" -i --oneline` → **empty**.
   - `git log --all -S "ldn" -i --oneline` → 11 hits, **all false positives** from the substring in `fieldnames` (e.g. `narma_advisor_probes.py:2000: fieldnames: list[str] | None = None`). Verified by inspecting one diff.
   - `git log --all -S "lmu" -i --oneline` → 1 hit (`52ac97c`), **a false positive**: the base64 of an embedded PNG in a notebook (`"image/png": "iVBORw0KGgoAAAANSUhEUg…"` contains the substring "lmu").
   - `git grep -a -l -i -E "hippo|legendre|\bLDN\b|\bLMU\b" $(git rev-list --all)` (all 265 revisions, binary-inclusive) → **no output, exit 1**.
   - `git grep -n -i -E "hippo|legendre|\bLDN\b|\bLMU\b" -- ':!.gitignore'` (working tree, tracked files) → **no output, exit 1**.
2. **No implementation file.** A tree-wide scan of 2 337 `.py/.md/.txt/.sh/.json/.yml/.ipynb/.out/.log` files (excluding `.git`, `venv`, `.venv_simple_ctle`, `__pycache__`, `catboost_info`, `node_modules`) found **no** `def`/`class` named for ldn/lmu/hippo/legendre/gamma/laguerre. The only filename matches are `how_to_train_your_hippo.pdf`, `run_knet_HiPPO.sh`, and the `docs/Niche Research - …LDN Pruning….md` title.
3. **No run artifacts.** `outputs_alliance_can/` contains 21 `narma_*.out` job logs and directories `narma_phase7_leak_bands/`, `narma_phase8_activity_leaks/`, `post_twin_gates/`, `linear_controls/`, `narma_phase_5_drive_rank/`, `narma_refresh2/`. **Nothing** HiPPO/LDN/LMU/gamma/Laguerre-related. There is no `run_knet_HiPPO-*.out` job log, no LDN `study.db`, no LDN output directory anywhere in the repo.
4. **Other branches.** `git ls-tree -r --name-only origin/main` → 25 files, `origin/deq` → 35 files; **neither contains any `narma*`, `hippo`, `ldn`, or `lmu` file.** The entire NARMA program lives on `simplified` only, and much of even that is untracked/ignored.
5. **The one genuine "Legendre" in code** is a *measurement instrument*, not a memory operator: the IPC-2 nonlinear detector regresses states against the centered quadratic `u(n−k)²` (`narma_revised_plan.py:155` `ipc2_curve()`; docstring at `narma_revised_plan.py:17-18`), described in prose as "Legendre `P_2(u(n−k))`" (`docs/narma-progress-summary.md:412-413`). It was used to prove the **absence** of the lag-9 product path (all 7 product pairs 0.01–0.03, `docs/narma-progress-summary.md:446`).

**Conclusion for part 2:** the HiPPO/Legendre/LDN/LMU thread is a **plan and a reading list**, never a program. It was designed in `docs/NARMA Fabric Memory Progress Summary.md` (Phase 12, E0–E6) and then **abandoned mid-thought**: the file's last line (536) ends in the middle of a sentence — *"This single literature occupies your niches (i), ("* — inside the niches research section. That truncated file is the literal end of the thread.

---

# PART 3 — RECOMMENDATION ONLY

> Everything in this part is advisory. Nothing here was implemented, and no file was changed.

## 3.1 What is salvageable from NARMA

**Salvage (high value, low cost):**
1. **The measurement methodology and its instruments.** E0 frozen-core screening, gate-zero, carry-mask, MC/PR/lag-product probes, per-seed pairing, pre-registered KILL rules, and (after the Phase-9 correction) **held-out scoring**. This is the one genuinely reusable asset and is itself a citable methods contribution (`docs/Analog-Native RNN Features and Gap List.md:140`, B3/C1).
2. **The leak-band mechanism.** Phase 7/8 is a clean, paired, activity-matched win: NRMSE 0.598 → 0.537, MC 7.90 → 8.74, correlation PR 2.32 → 3.46, lag-9 product R² 0.467 → 0.520, realized into three separated pole bands. It is a *core-only* intervention with no new devices. Keep it.
3. **The negative knowledge** — the memory–rail coupling law, "fabric = nonlinear expander, not memory," the tap-bank cost frontier (taps8 = 200 edges; tapsVD4 = 100 edges → MC 9.21), and "the bank is incompressible below ~8 taps." These are citable findings, not dead ends.

**Salvage with caveats:** the *twin* methodology (`twin_node_tanh`, `post_twin_gates`) — it is what made Phase 9's architecture inversion possible and it is the substrate on which any HiPPO/LDN leg should first be run.

**Do not salvage:** the 0.67-floor framing as a *goal* (it was a consequence of the no-delay-bank configuration, not a physical law), and the "training buys ~0.05 through saturation" conclusion as a general claim (it was true of the specific weak-init core; the Phase-8 banded core was never given a paired rail-managed training run).

## 3.2 Recommended code simplification (from §1.7)

**Remove (high confidence):** `core_refresh_interval` (ruled out by construction — k = 2/4/8 identical), `node_activation` + `allow_experimental_cells` + `no_clip` (self-described "narma-node-activation" / "narma-cell-isolation"; the compliant/bounded-tanh cell already covers the physics), `TapRails` + the delay-line rail gating (Phase 8B: every gated variant worse than identity), `make_narma_preset` + the `narma10`/`narma20` registry entries.

**Decide explicitly:** `dynamic_leak` / `dynamic_leak_input_proj`. As a NARMA artifact it is dead (legs B/C/D were digit-identical to their ablations). But `docs/ideas.md` §2.4 and `docs/Math-Side Operators...` M4 (Tier S #4) want to productize exactly this mechanism as **input-gated LDN**. Either keep it and re-purpose it deliberately, or remove it and re-derive it inside a proper LDN implementation — but do not leave it as a half-built flag.

**Keep:** `enable_ref_edges` and persistent drive — both predate NARMA and are used by the static CTLE/BO pipeline.

**Also fix before any simplification:** the `.gitignore` patterns `*.md`, `*.sh`, `*.out`, `*.log`, `docs`. These currently guarantee that the NARMA evidence base is unversioned and will be lost.

## 3.3 How external analog memory (HiPPO-LegS/LegT, LDN) maps onto circuits — and how it would connect to the KNet fabric

**Math.** HiPPO-LegS/LegT and the LDN/LMU are all continuous-time linear state-space systems
`ẋ = A x + B u`, `y = C x`, with `A` a (in the LDN/LegT case) Legendre-coupling matrix, and in HiPPO-LegS a *time-varying* `A(t)`. The LDN's special property is that `A` is **fixed in time** (LegT) and is the solution of an orthogonality condition against a uniform measure on a sliding window `[t−θ, t]`, so the state `x(t)` is the projection of the *entire* window of `u` onto truncated Legendre polynomials. That is exactly a **linear optimal-order compressor of a delay window into `n` states**, with approximation error decaying in `n`. LegS is the same construction against an exponentially-weighted measure with a "scaled" (growing) window.

**Circuit mapping — this is the good news.** `ẋ = A x + B u` is *literally* the KNet node ODE without the nonlinearity:
- `A` = the **core OTA adjacency/conductance matrix**: node *i*'s capacitor receives `Σ_j g_ij (V_j − V_i)`. The existing `linear_ota` core cell already implements exactly this KCL term; a HiPPO/LDN `A` is just a *particular choice* of `g_ij`.
- `B` = the **boundary OTA matrix** — exactly the 200 boundary OTAs of the 8-tap delay-bank architecture, with the taps replaced by `u` directly.
- `C` = the **readout** — a linear map over node voltages; the existing ridge/dense readout already is one.
- Therefore an LDN/HiPPO operator is a **G_m-C filter bank**: `n` integrators (capacitor + OTA pair per node) whose inter-node coupling conductances *are* the entries of `A`. A **diagonal** `A` (S4D-style) is `n` **independent leaky integrators** — i.e. exactly the Phase-7/8 "banded leak" core, with per-node time constants. A **dense** `A` (true Legendre coupling) needs cross-coupled OTAs.
- The natural analog decomposition of `A` is **modal**: eigen-decompose `A`; each real pole → one OTA-C integrator; each complex-conjugate pole pair → **one OTA-C biquad** (the canonical second-order analog block). This is `docs/Math-Side Operators...:25` (M1), and it is the correct bridge between "math operator" and "shopping list of analog blocks."

**How it would connect to the KNet fabric:**
1. **As the core.** Replace the random-sign torus `A` with a HiPPO/LDN `A` (or its modal/biquad realization), keep the existing boundary OTAs as `B`, and use the existing ridge readout as `C`. This directly attacks the diagnosed defect: Phase 6 showed the core has 25 near-unit modes but the *driven* state collapses to ~2 effective dimensions because the modes share nearly identical retention times; an LDN basis supplies a *designed* spread of time constants with a principled orthogonalization property rather than a hoped-for one.
2. **As a bank replacement (the E1/E2 route).** Insert the LDN between the input and the main core — exactly the "aux core" slot that Phase 10/11 left empty. Phase 11's evidence (`tapsVD4` incompressible below ~8 taps; shift-register = the bank at R² 1.000) says a *frozen* aux will recapitulate the bank; an LDN is the one aux flavor that is **provably not a copy of a tap bank** (it projects onto a polynomial basis, not onto raw delayed samples), so the tap-subspace R² kill rule should not fire spuriously the way it did for echo/shift-register.
3. **As a gate-rail modulator.** The LDN/LMU timescale `θ` and the Mamba-style learned `Δ` map naturally onto **bias currents**: a per-timescale modulation path per node (M5). This is where the analog wire-cost question becomes real and where the user's "selective HiPPO" idea lives.
4. **As a basis competitor.** Legendre vs Laguerre vs gamma on the *same* fabric under the *same* mismatch sweep (F5) is a clean single-paper unit — Laguerre's single repeated pole should mismatch more gracefully than Legendre's distinct-pole coupling matrix.

**Pitfalls (all of these are already flagged in the repo's own documents):**
- **State amplitude collapse with polynomial order.** HiPPO-LegS conditioning degrades with `d` (Park et al. 2024, cited at `docs/Niche Research...:180`); states `x_5, x_6…` drop far below `x_0, x_1`. In an OTA fabric that means the high-order states sit in the noise floor *and* the high-order entries of `A` are small, so those channels contribute nothing. Remedies: **balanced realization** (Moore 1981), per-order gain tuning, companding/log-domain state elements (F2), or per-state AGC (F3) — and each of these must be measured as state-SNR-vs-order, not assumed.
- **HiPPO-LegS is time-varying.** `A(t)` cannot be realized as static OTA conductances. Only the **LegT/LDN** form (time-invariant) is directly realizable. This is the single most important practical constraint and it is why the PDF's distinction between LegS and LegT matters.
- **Density / edge cost.** A dense Legendre `A` is `O(n²)` OTAs; at `n = 8..12` that is 64–144 cross-couplings *plus* `n` integrators, competing against the 200-edge bank it is meant to replace. S4D's diagonalization is the cheap alternative — but then you are back to independent leaky integrators, i.e. the Phase-7 banded core, and the orthogonalization benefit is lost.
- **Mismatch.** A dense `A` with distinct poles is exactly the structure that mismatches badly; the Laguerre/all-pass alternative exists precisely because one repeated pole is match-friendly. Also: mismatch must be *injected in training*, not just measured afterwards.
- **Nonlinearity is not in the operator.** An LDN is linear; NARMA's sufficient statistic is the *product* `1.5·u(n−9)·u(n)`. Phase 9's verdict stands: the fabric's job is nonlinear expansion, the memory operator's job is linear storage. So an LDN does not replace the tanh core — it replaces the tap bank.
- **The "free window" trap.** With a bank of `T` taps, MC gains up to `+T` are free (measured: +3.9 flow / +4.4 discrete). Any LDN comparison must be **drive-matched and edge-matched**, or it will be flattered exactly as the scalar ESN parity was.
- **Analog memory is not digital memory.** An LDN is optimal for *white* input statistics (NARMA's `u` is i.i.d. uniform ⇒ the history matrix is full-rank, so all compression must come from the operator basis, not from signal redundancy). On a real channel (CTLE) with colored input the optimal operator differs — a mismatch that a NARMA-only study will not reveal.
- **Practical:** the "frozen aux screen" (E0 philosophy) is structurally unwinnable against a bank, as argued in `docs/NARMA Fabric Memory Progress Summary.md:170` — so do not let a failed frozen-LDN leg be read as "LDN does not work."

## 3.4 Recommended order of work (if the thread is revived)

1. **E0 first (cheap, eval-only):** build the cost-matched bank frontier (`tapsVD-k` k = 2..6, `taps4`/`taps6`, `taps8-sparse`) **with held-out scoring**. Every later claim is uninterpretable without this number. This is `docs/ideas.md:395` and `docs/NARMA Fabric Memory Progress Summary.md:226-228`.
2. **Then E1, the HiPPO/LDN leg, at twin level** (not in the fabric): implement `ldn` (closed-form A/B, orders d ∈ {4,6,8,12}, θ ∈ {8,12}), `gamma`, `laguerre`, `ms_gamma`. Report **MC-vs-edges**, **held-out NRMSE**, **per-delay reconstruction**, **state-SNR vs order**, and the **tap-subspace R²** (the last one is the discriminator: an LDN should *not* be a copy of the bank — if it is, the basis is not doing anything). Also implement the **modal/biquad** realization of the same `A` and compare it to the dense `A` — this is the analog-realization question, and it is the thing that makes the work a fabric contribution rather than an SSM library call.
3. **Only then decide whether to train** anything (E2). If E1's frozen LDN already beats the equal-edge bank held-out, the training question is narrowed to "can we prune this to fewer edges?" (E4) rather than "can we discover memory?".
4. **Keep the fabric integration (E6) gated** on beating the *cost-matched* bank at ≤ 0.5× edges, with rails ≤ 0.20 and activity 0.45–0.55, paired seeds.

## 3.5 Bottom line

The NARMA campaign did not fail because the fabric is bad; it failed because (i) memory and rail saturation were coupled, (ii) the eventual fix — a static 8-tap delay bank — handed the network its memory rather than letting it hold it, and (iii) every learned replacement for that bank either copied it or underperformed it. The HiPPO/Legendre thread is the natural next move *because* it is the one operator family whose tap-subspace R² will not be ≈ 1.0 — but it exists **only as a plan**, it was never implemented, and the plan's own E0 (cost-matched baselines) leg — the cheapest and most necessary step — has still not been run.

---

## Appendix A — Files read (audit evidence)

**Docs** (all under `docs/`, all gitignored except where noted):
`narma-progress-summary.md` (1 139 lines) · `NARMA Fabric Memory Progress Summary.md` (536) · `narma-revised-plan.md` (319) · `narma-fabric-memory-problem-statement.md` (172) · `narma-phase6-core-eigen-problem-statement.md` (195) · `narma-phase8-temporal-diversity-problem-statement.md` (175) · `narma-analog-temporal-diversity-research.md` (176) · `Analog-Native RNN Features and Gap List.md` (194) · `ideas.md` (601) · `Niche Research - Reconfigurable Analog Fabric, OTA Linearization, LDN Pruning, Drift Calibration, State SNR Balancing.md` (301) · `Math-Side Operators, Full Ranking, CABs, Compiler-in-One-Fabric.md` (141) · `Critique Verification and Compiler Explainer.md` (grepped) · `docs/plans/` (54 entries) · `docs/specs/` (54 entries).

**Root:** `narma_debug.md` (693) · `narma_exp.log` (12 481) · `narma_diag_latest_log.out` (27 804 bytes) · `run_knet_HiPPO.sh` (30) · `run_phase_a_knet_bo.sh` (59) · `tap_rails.py` (62) · `how_to_train_your_hippo.pdf` → `tmp/analysis/how_to_train_your_hippo.txt` (1 991 lines).

**Code inspected:** `config.py` (`make_narma_preset` 1175-1303, `PRESET_NARMA10/20` 1306-1307, registry 1319-1320, `DRIVE` 629-634) · `differential_stage.py` (`core_refresh_interval` 95/238/281-283/1763, `node_activation` 239/275-280/1150, `dynamic_leak` 211-212/292-296/405-411/889-900, `enable_ref_edges` 133-145/220/522-526) · `topology.py` (`allow_boundary_target_overlap` 1759/1908/1949/2015/2220, `allow_experimental_cells` 1977/2062/2109-2112, `no_clip` 1978/2605, `dynamic_leak` 1354-1355/2096-2097, `enable_ref_edges` 1064/1420/1521/1710/1760/1847/1909/1950/2043/2584) · `narma_revised_plan.py` (`ipc2_curve` 155, thresholds 92-95) · `post_twin_gates.py`, `aux_core_probe.py`, `audit_aux_core.py`, `audit_post_twin_gates.py`, `audit_twin_standalone.py` (tracked NARMA-probe modules, presence confirmed).

**Git (read-only):** `git branch -a` · `git log --all --oneline` (265) · `git log --all -S <keyword> -i` for `hippo`/`legendre`/`ldn`/`lmu`/`enable_ref_edges`/`dynamic_leak`/`node_activation`/`core_refresh_interval`/`allow_boundary_target_overlap`/`tap_rails`/`no_clip`/`allow_experimental_cells`/`drive_isat` · `git grep -a -l -i -E "hippo|legendre|\bLDN\b|\bLMU\b" $(git rev-list --all)` · `git ls-files` (83 tracked) · `git ls-tree -r origin/main` (25) / `origin/deq` (35) · `git show --stat` on 10 NARMA commits · `git check-ignore -v`.

## Appendix B — Commit hashes referenced in this report

| Hash | Date | Subject | Relevance |
|---|---|---|---|
| `f097500` | 2026-07-16 | Added reference nodes `--enable-ref-edges` | **pre-NARMA** feature |
| `3db50bc` | pre-NARMA | Added persistent drive | **pre-NARMA** feature |
| `1164950` | 2026-09-03 | added infrastructure for narma experiment with variable refresh rate | adds `core_refresh_interval` |
| `e357c6c` | 2026-09-13 | Added ESN based rung tests | adds `dynamic_leak`, `dynamic_leak_input_proj` |
| `4356a86` | 2026-09-13 | added esn copy test and multi stage test | ESN/rung + multistage audit |
| `e357c6c`/`4356a86` | 2026-09-13 | — | ESN parity harness |
| `9817ae4` | 2026-09-13 | questioning/testing physical constraints for narma experiments | split-leg probes |
| `a02de37` | 2026-09-14 | Added new interactions and cell types | adds `node_activation`, `allow_experimental_cells`, `no_clip` |
| `4e4751b` | 2026-09-14 | added machinery for delayed inputs | adds `allow_boundary_target_overlap` |
| `43d6be9` | 2026-09-14 | Added mix->bound dynamics | new cell interactions |
| `f0e5e4a` | 2026-09-14 | bugfix for multi delayed inputs | `narma_spectral_norm.py` |
| `8aff5c2` | 2026-09-15 | Added new delay line rail gating feature for narma | adds `tap_rails.py` + `TapRails` |
| `87a251d` | 2026-09-15 | added leak sweep and added more diagnostics | Phase-7 leak bands |
| `92cb560` | 2026-09-16 | final probe digital node tanh twin | Phase-9 twin |
| `6054f36` | 2026-09-17 | Investigating new architecture with heun | — |
| `1812651` | 2026-09-18 | Added original kirchhoff bo and some probes | `post_twin_gates`, `aux_core_probe` era |
| `e11e2b0` | 2026-09-19 | Updated to be exact copy of losses and eval of kn_bayes_opt for ctle problem | HEAD-side CTLE work |

## Appendix C — "Not found" register (explicit non-claims)

- No LDN / LMU / HiPPO / Legendre / gamma / Laguerre **implementation** — in the tree or in any of the 265 revisions. **Not found.**
- No HiPPO/Legendre/LDN **run output, log, `study.db`, or output directory**. **Not found.**
- No `test_memory_knet.py` (the script `run_knet_HiPPO.sh` invokes) in the tree or in any revision. **Not found.**
- No commit message, branch, or doc claiming a HiPPO/LDN *result*. **Not found.**
- No `origin/main` or `origin/deq` NARMA/HiPPO content (both branches contain 25 and 35 files respectively, none NARMA-related). **Confirmed absent.**
- Phase-12 legs **E0–E6** (including the `ldn` / `gamma` / `laguerre` / `ms_gamma` analytical insertion) — **never run**; no output artifacts exist.
- A paired, rail-managed training run from the Phase-8 banded core — **never run** (`docs/narma-phase8-temporal-diversity-problem-statement.md:88-103` states this as an untested hypothesis).
- Taps8-vs-taps8 (matrix-input) ESN parity — **still open** (`docs/narma-progress-summary.md:888-889`).
