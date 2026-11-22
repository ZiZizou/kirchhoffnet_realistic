# Phase 3b: time-multiplexed fabric, implementation plan, retraining

Scope: documentation only. Nothing in the source repo was modified and nothing was run on a GPU.
Source repo (read-only): `ssr0:~/Documents/kirchhoffnet_refactor`, branch `hw-refactor`, tag `pre-surgery` at 2568705. Paths below are relative to it.
Citations are `file:line` in that checkout and describe **current** behaviour. The surgery worktree (`~/Documents/tron_eq/kn_surgery`, branch `surgery`, 6 commits) has different line numbers; implementation sites are therefore also named by function.
Unit layer, constraint formulas and flag names (`hw_ratio_ab`, `hw_slope_bound`, `hw_compliance`, ...) are defined in `phase3a_units_and_constraints.md` (Vu = 0.07 V, Cu = 50 fF, Tu = 75.1 ns, Iu = 46.7 nA). This file does not repeat them.
Items marked **[ASSUMPTION]** are 22FDX or circuit estimates that I could not verify from the repo and that need a Spectre/PDK check.

**New fact versus Phase 3a §1.0.** A trained checkpoint of the 7-stage paper model does exist on ssr0: `~/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/knet_model_files_manual_trial_0011/phase_a_best.pt` (345 state-dict keys; `run_command.txt` next to it matches the paper command, and the same directory holds `phase_a_last.pt`, `phase_a_history.json` and the `59014723.out` log, so it is the paper run, trial_0011). I loaded it with the repo venv (`torch.load`, CPU) and quote statistics from it as "trial_0011 checkpoint". It is not in the repo checkout (`pipeline_report.md:19`). Phase 3a §1.0 has been updated to say so; the trained statistics quoted in Phase 3a are still from `fixed_mlp_knet`, and Phase 3a §2.3 now also quotes the paper checkpoint's gm·A and Isat.

---

## 1. Time-multiplexed fabric

### 1.1 What crosses a stage boundary (verified from code and checkpoint)

| Item | Count | Evidence |
|---|---|---|
| Node voltages written by the stage-end state and rewritten by `StageTransfer` | **18** = 11 hidden + 7 accumulators | Transfer is sized `len(active_nodes) + output_ode_count` (`topology.py:2634-2642`); applied to the whole state vector (`kirchhoff_net.py:173-174`); no `drive_mask` in the CTLE model, so all 18 nodes are transformed (`circuit_model_report.md:169`) |
| Number of boundaries | **6** for 7 stages (no transfer after the last stage) | `len(transfers) == len(stages) - 1` (`kirchhoff_net.py:112-114`); readout is `x[:, 11:18]` of the last stage (`kirchhoff_net.py:645-654`) |
| Initial state of stage 1 | all 18 nodes = 0 (hidden and accumulators) | `kirchhoff_net.py:593-616` |
| Input `u` (4 spec voltages) | not carried: constant terminals for the whole inference, in every stage | `circuit_model_report.md:140` |
| VCA gains `g_vca = 2σ((u·W)·vᵀ)` | not carried: recomputed per stage from `u` only, because `vca_use_hidden` defaults to `False` and the dagger harness has no flag for it | `differential_stage.py:1088-1125` (hidden term only if `W_hidden` and `x` given); `topology.py:1380` |
| GLN rails `z = tanh(α(a·u + c))` | not carried: function of `u`, one set shared by all stages | `gln_rails.py:139-148`; `net.gln_rails.*` keys have no stage index |
| `core_refresh_interval` | 0 (no mid-stage refresh) | `circuit_model_report.md:176` |

So the carried analog state is exactly **18 voltages**. Nothing else (no hidden gate state, no velocity term) crosses a boundary. The handoff guess "18 S&H = 11 + 7" is right as a count of values that cross, but not as a count of persistent holds (next section).

Per-stage graph and parameter sharing, read from the trial_0011 checkpoint:

- `src`, `dst`, `boundary_src`, `boundary_dst`, `readout_sense_src` are **identical in all 7 stages**. One physical interconnect works for every stage; no switch matrix has to be reconfigured.
- The 22 core edges are 11 distinct (src,dst) pairs, each present twice (`edge_repeats = 2`). The 22 senses are 2 per hidden node. The two copies of a pair carry independent parameters.
- All 7 boundary-OTA cell parameters (`a_raw, b_raw, s_raw, gm_raw, isat_raw, theta_raw, g_resistive_raw`) are **identical across stages** (tied). Only `boundary_z_logits` and `vca_v_boundary` differ per stage. Those 8 OTAs need no per-stage reload of their cell bias.
- Core and sense OTA parameters, crossbar `W`, `raw_vref_sense`, `clip_sharpness_raw`, VCA taps and the 6 transfers differ per stage.

### 1.2 `freeze_read` as a sample-and-hold of OTA inputs

Code behaviour:

- At stage entry `_compute_i_edge_const(x0)` evaluates the core tanh branch once from the stage-entry state and holds it for all 10 Heun steps (`differential_stage.py:1885-1886`, body `:1296-1346`). It reads **both** edge terminals, `x[src]` and `x[dst]`, which are hidden nodes only (`src` max index 8, `dst` max index 10 in the checkpoint). Accumulators are never core-OTA inputs.
- The paper run sets `freeze_read=True` (`dagger-nuance-distillation-kirchhoffnet.py:622`, passed at `:1889`).
- What stays **live** (reads the evolving node): the shunt `G·(Vs − Vd)` (`differential_stage.py:1486-1492`, `cell_library.py:402-414`), leak and soft clip (`:1761-1768`), the 8 boundary OTAs (`freeze_boundary=False`, `:100-107`, `circuit_model_report.md:146`) and the 22 sense OTAs (`freeze_temporal_read=False`, `circuit_model_report.md:162`). The shunt is counted exactly once in every path (fixed in commit `7cda290`: the dynamic `freeze_read=False` path used to count it twice).

Hardware mapping:

| Model element | Circuit |
|---|---|
| Held core tanh current | OTA input pair driven from a **hold capacitor**, not from the node. The OTA output leg still lands on the evolving node (ideal current source in the model; finite `g_out` and compliance in §3.4/3.5 of Phase 3a) |
| Stage-entry state `x0` (hidden part) | 11 hold caps `C_h`, loaded with the **post-transfer** hidden voltages and frozen for the whole stage |
| Live shunt, sense OTA, leak, clip | connected directly to the node (gate loading only) |
| Accumulators | no hold: nothing in the core reads them |

Counts:

- **Persistent holds during evolve: 11** (not 18).
- **Transient holds across the transfer: 18.** The transfer block reads each node, forms `f(x)`, and writes it back. Capture of the pre-transfer value must not be destroyed by the write, so each of the 18 needs a temporary copy. Practical choice: the 11 hidden transfer outputs are written into the 11 persistent hold caps and onto the node caps in the same step; the 7 accumulators use 7 transient caps (or 1-2 shared ones used serially at +2 ns each).
- Stage 1 starts from x = 0, so its core OTAs see Vs = Vd = 0 and act as constant offset currents (`tanh(gm·θ)`); stage-1 core A and B stayed at softplus(0) = 0.693 in the checkpoint (never received gradient), consistent with this.

### 1.3 Phase sequence per boundary

```
 evolve(s)            hold            transfer+write         reprogram             evolve(s+1)
 ──────────┬─────────────┬────────────────────┬──────────────────────┬──────────────────
 OTAs on   │ P0: isolate │ P1: f(x) per node  │ P3: swap shadow →    │ P4: outputs back
 50 ns     │ nodes, sample│ P2: write node cap │     active bank,     │ onto nodes,
           │ 18 values    │ + load 11 holds    │     local DACs settle│ tails keep running
           │              │                    │ (overlaps P1/P2)     │
```

1. **P0 hold.** Disconnect OTA outputs from the nodes (steer them into a Vcm-clamped dummy sink; **do not switch the tails off**, see §1.5). Sample the 18 node voltages through unity buffers into transient caps.
2. **P1/P2 interstage transfer and write-back.** `x' = w1·x + w2·tanh x + w3·ReLU(x − vth)` per node (`stage_transfer.py:161-169`). The result is written onto the node cap and, for the 11 hidden nodes, onto the persistent hold cap.
3. **P3 reprogram.** Load stage s+1 into the active bank: a single clocked latch from shadow registers that were filled **during stage s** (see §1.4 budget). Local current DACs and OTA internal nodes then settle.
4. **P4 evolve.** Reconnect OTA outputs to the nodes. Core OTA inputs are static for the next 50 ns.

Because the core currents are frozen, each core OTA is a once-per-stage DC current source. Its bandwidth requirement is one step response per stage, not tracking of a live signal. Sense and boundary OTAs are live.

### 1.4 Hardware simplifications of the interstage transfer

The checkpoint uses all three terms in every transfer (min / max over the 18 nodes):

| Transfer | w1 | w2 | w3 | vth | nodes with \|w3\| > 0.05 |
|---|---|---|---|---|---|
| 0 | 0.43 .. 1.52 | −0.52 .. 0.58 | −0.62 .. 0.20 | −0.12 .. 0.21 | 8/18 |
| 1 | 0.72 .. 1.38 | −0.23 .. 0.44 | −0.41 .. 0.26 | −0.21 .. 0.49 | 11/18 |
| 2 | 0.67 .. 1.51 | −0.28 .. 0.58 | −1.28 .. 0.48 | −0.43 .. 0.22 | 12/18 |
| 3 | 0.52 .. 1.38 | −0.43 .. 0.55 | −0.73 .. 0.56 | −0.33 .. 0.44 | 15/18 |
| 4 | 0.57 .. 1.57 | −0.18 .. 0.56 | −0.88 .. 0.54 | −0.31 .. 0.31 | 13/18 |
| 5 | 0.62 .. 1.28 | −0.32 .. 0.26 | −1.36 .. 1.00 | −1.26 .. 0.92 | 18/18 |

Mean |w3| for hidden nodes grows from 0.12 (transfer 0) to 0.45 (transfer 5); accumulators 0.06 .. 0.35. Dropping ReLU or tanh therefore needs a retrain, not a swap. Candidate simplifications, cheapest first. Each is a retrain flag (`hw_tm_xfer_mode`), compared against the full form with the 10% rule:

| Mode | Change | Circuit saving | Risk |
|---|---|---|---|
| `full` | current form | one gain stage + tanh OTA + one-sided rectifier OTA per node | reference |
| `shared_vth` | one `vth` per transfer (not per node) | removes 17 of 18 vth DACs per boundary (6 × 17 values) | small: per-node vth spread is 0.1-1.3 |
| `quant` | 4-5 bit `w1,w2,w3,vth` with STE (`hw_tm_param_bits`) | smaller DACs | small |
| `kick` | `x' = x + Δ`, `Δ = (w1−1)·x + w2·tanh x + w3·ReLU(x−vth)` realised as a **short current pulse from OTAs onto the existing node cap** (`Δx = I·t_x/C`). The 18 held values drive the pulse OTAs for t_x ≈ 3 ns | no unity-buffer write-back path and no full-range reset of the node cap; reuses OTA style blocks | pulse currents µA-scale for \|Δ\| ≈ 0.5 unit on 50 fF: Δx = 0.5·0.07 V = 35 mV, Q = 1.75 fC, I = 0.6 µA over 3 ns. Energy 0.8 V × 0.6 µA × 3 ns = 1.4 fJ per node, 18 × 6 = 0.16 pJ per inference: negligible |
| `no_tanh` | w2 ≡ 0 | one OTA less per node | needs retrain: w2 is 0.2-0.5 in magnitude |
| `no_relu` | w3 ≡ 0 | rectifier removed | largest accuracy risk (w3 used on 8-18 nodes) |
| `clamp` | output clamp to ±x_hard (Phase 3a `hw_clamp_transfer`) | avoids writing a value above the rail | required for hardware: the soft clip cannot confine states (`circuit_model_report.md:27`) |

Recommendation: start with `full + shared_vth + quant + clamp`, then try `kick` if the write-back buffers dominate P1/P2.

### 1.5 Quantitative budget inside ~350 ns

Baseline: Vu = 0.07 V, node C = Cu = 50 fF, Tu = 75.1 ns, stage evolve = 0.6658 Tu = 50 ns, Iu = 46.7 nA (Phase 3a §2.4). All 350 ns is evolve time. Every nanosecond of boundary overhead is extra unless Tu is reduced: **Tu = (350 ns − 6·t_ovh) / 4.660583** and Iu = Cu·Vu/Tu.

**S&H acquisition.** Path: node → unity buffer → switch Ron → hold cap `C_h`. Take Ron = 2 kΩ (transmission gate, thick-oxide, boosted) and a buffer with gm_b = 100 µS (1/gm_b = 10 kΩ) **[ASSUMPTION]**. τ = (Ron + 1/gm_b)·C_h. Settling to 8 bits (0.4% of the ±4 range = 0.03 unit = 2.2 mV) takes ln 256 = 5.5 τ.

| C_h | τ | acquire to 8 bit | kT/C rms (300 K) | in units | charge injection raw (see below) | droop at 1 pA over 50 ns |
|---|---|---|---|---|---|---|
| 50 fF | 0.6 ns | 3.3 ns | 0.288 mV | 0.0041 | ~10 mV (0.14 unit) | 1.0 mV (0.014 unit) |
| 100 fF | 1.2 ns | 6.6 ns | 0.204 mV | 0.0029 | ~5 mV | 0.5 mV |
| 200 fF | 2.4 ns | 13.2 ns | 0.144 mV | 0.0021 | ~2.5 mV | 0.25 mV |

- **Charge injection** ΔV = (½·W·L·Cox·(Vgs − Vt) + W·Cov·V_clk)/C_h **[ASSUMPTION]**. For a thick-oxide switch with W/L = 0.5 µm/0.15 µm, Cox ≈ 8 fF/µm², V_clk = 1.8 V, Cov ≈ 0.3 fF/µm: ≈ 0.5 fC, i.e. ~10 mV on 50 fF. This is the dominant S&H error at 50 fF. **Bottom-plate sampling plus a half-size dummy switch** typically cancels 10×, leaving ~1 mV (0.014 unit), of which ~15% is signal dependent (a gain error). The constant part can be trained around, so it is modelled as a per-node offset.
- **Droop**: ΔV = I_leak·t/C_h, t = 50 ns hold. Needs thick-oxide (EG) or stacked switches. A regular-Vt core switch at 85 °C leaks ~nA and would droop 1 V, which is unusable. 1 pA is a **[ASSUMPTION]**. Droop is a one-sided offset, so it is trainable-around when deterministic.
- **kT/C accumulation.** Two sampling events per node per boundary (capture, write-back) and 6 boundaries: rms ≈ √12 × 0.288 mV = **1.0 mV = 0.014 unit** at C_h = 50 fF. Against the ±4 range this is ≈ 11 bits. The summed random plus systematic S&H error (kT/C 1.0 mV, droop 1.0 mV, residual injection 1.0 mV) ≈ 1.7 mV = 0.024 unit, which is 0.8 LSB of an 8-bit grid. **At 50 fF the S&H works only with bottom-plate sampling, a dummy switch, and EG switches.** Otherwise use C_h = 100-200 fF; the hold caps are 18 of ~600 cap instances and are cheap in area, but they lengthen P0.

**Bias reprogramming: what settles.** Two things, with very different time scales.

1. Digital shadow-register swap and local current-DAC steering: ~1 ns (clocked latch, unit-cell switching). Not a limit.
2. **OTA internal poles at nA bias** (the real limit). A mirror or tail node with C_int = 2-3 fF carrying I = Isat·Iu has τ_e = C_int·n·U_T / I. In model units, τ_e = κ_int / Isat with κ_int = C_int·nU_T/(Cu·Vu) = (3 fF · 33.6 mV)/(50 fF · 70 mV) = 0.029 Tu = **2.2 ns · unit/Isat** **[ASSUMPTION: C_int = 3 fF, n = 1.3, 300 K]**. So τ_e = 18 ns at Isat = 0.12, 3.6 ns at 0.6, 1.1 ns at 2.0.
   - The trial_0011 checkpoint has core Isat 0.18-0.64 in stage 0 and 0.21-2.69 in stage 6 (σ(z) and g_vca not included), so many edges sit at τ_e of 3-12 ns.
   - After a swap, a held-input OTA's current relaxes: `I(t) = I_new + (I_old − I_new)·e^(−t/τ_e)`. `I_old` is the same slot's frozen current from the previous stage. The extra charge is `(I_old − I_new)·τ_e = C_int·nU_T·(Isat_old/Isat_new − 1)`: for a ratio of 10 that is 0.9 fC = 18 mV = **0.26 unit** on a 50 fF node. If the tail is **switched off** during the boundary, then `I_old = 0`, the loss is `C_int·nU_T` = 0.1 fC (2 mV, 0.029 unit) per edge, but at Isat = 0.12 the stage contributes only 5.6 nA × 50 ns / 50 fF = 5.6 mV, so the edge loses 36% of its effect. **Keep tails running** (dummy-load the outputs during P0-P3).
   - Mitigations, in order: (a) model it (`hw_tm_bias_settle`, §2.4), (b) floor Isat_eff ≥ 0.5 unit, which bounds τ_e ≤ 4.4 ns, (c) tail boost pulse, (d) two ping-pong fabrics (the unused fabric's bias settles during the other's stage; held-input steps still cannot be hidden).

**Per-boundary overhead (shadow-bank design, bias values fetched during the previous stage):**

| Phase | Time | Basis |
|---|---|---|
| P0 isolate + acquire (C_h = 50 fF, 8 bit) | 3.5 ns | 5.5 τ + margin |
| P1+P2 transfer + write (current-mode f-block driving node cap, ~5 µA for 14 fC) | 5 ns | 14 fC / 5 µA = 2.8 ns plus f-block settle |
| P3 swap + local DAC (overlaps P1/P2; residual) | 1 ns | latch |
| P4 reconnect | 1 ns | switch + clock |
| **t_ovh** | **10.5 ns** | |

**What this does to the 350 ns budget:**

| t_ovh | 6 boundaries | Tu for 350 ns total | Iu = Cu·Vu/Tu | total at Tu = 75.1 ns |
|---|---|---|---|---|
| 5 ns | 30 ns | 68.7 ns | 51.0 nA | 380 ns |
| **10.5 ns (shadow bank)** | **63 ns** | **61.6 ns** | **56.8 nA** | **413 ns** |
| 14.5 ns (in-place, 144 DAC channels) | 87 ns | 56.4 ns | 62.0 nA | 437 ns |
| 24.5 ns (in-place, 48 DAC channels) | 147 ns | 43.6 ns | 80.3 nA | 497 ns |

Shrinking Tu raises Iu by 22% (shadow bank) and raises average power by the same factor, but it also speeds the internal poles by 1/Tu. In-place reprogramming costs `ceil(720 / N_dac) × 1 ns` and is dominated by DAC channel count. **A shadow bank loaded during the previous stage is required to stay near 350 ns.**

**Programmable values per stage** (after Phase 3a folding of σ(z) into Isat and G):

| Group | Count | Bits each | Bits |
|---|---|---|---|
| 52 OTAs × (Isat 6, g 4, Â/B̂ ratio 3, s 1, θ 4, G 4) | 312 values | 22 per OTA | 1144 |
| VCA taps: `vca_v_core` 66, `vca_v_boundary` 24, `vca_v_readout` 66, `vca_W` 12, `vca_W_core` 12 (`circuit_model_report.md:72`) | 180 | 6 | 1080 |
| Crossbar `W` 7 × 22 | 154 | 5 | 770 |
| `raw_vref_sense`, `clip_sharpness_raw` | 2 | 6 + 4 | 10 |
| Transfer (per boundary): 18 × (w1, w2, w3, vth) | 72 | 5 | 360 |
| **Per stage** | **720 values** | | **3364 bits (≈ 0.42 kB)** |
| **All 7 stages** (transfer × 6) | | | **≈ 23.2 kbit (≈ 2.9 kB)** |
| Static, not reloaded: GLN (≈ 100 values), tied boundary cell params (8 OTAs) | | | ≈ 0.6 kbit + 1 kbit |

6-bit log-spaced Isat covers 80:1 in 63 steps (7% per LSB). That is a starting point; the QAT sweep (`hw_tm_param_bits`, 5/6/7) decides.

Memory and reload cost **[ASSUMPTION]**:

- SRAM at 22FDX bitcell ≈ 0.12 µm², ~50% macro efficiency: 23.2 kbit ≈ 0.006 mm².
- A 3.4 kbit fetch in ≤ 40 ns needs a 512-bit-wide read at 200 MHz (7 words).
- Fetch energy at 1-5 fJ/bit: 3.4 kbit × 7 stages = 24-120 pJ per inference, against the analog energy of 4.1 pJ at 2 units per edge (Phase 3a §3.9) or ~1 pJ for the trial_0011 checkpoint (stage Σ Isat ≈ 18-30 units at 46.7 nA × 0.8 V × κ=3 → 2-3.4 µW per stage). **Parameter fetch can exceed analog energy by 10-100×.** Memory energy, not OTA power, is then the dominant energy term.

**Time-multiplexed fabric (1 set) versus 7 physical copies** (order of magnitude, ±2×; 22FDX unit areas are **[ASSUMPTION]**: 50 fF MOM cap 30 µm², OTA + trims + local registers 120 µm², transfer block 150 µm², hold cap + switch 40 µm², crossbar cell 40 µm²):

| Item | 1 fabric | 7 copies |
|---|---|---|
| OTAs: core 22, boundary 8, readout 22 per stage | **52** | **364** (154 + 56 + 154) |
| Node caps (18 per stage) | 18 | 126 |
| Crossbar cells (7 × 22) | 154 | 1078 |
| Transfer blocks (18 per boundary) | 18 | 108 |
| Hold caps | 11 + 7 (transient) | 6 × 18 = 108 |
| Bias values live in analog | 720 active + 720 shadow | 5040 static |
| Digital bank | 23.2 kbit SRAM | 23.2 kbit (static trim, no fast fetch) |
| Parameter reload | 24 kbit moved per inference | none |
| Area (sum of the above + SRAM ~0.006 mm²) | **≈ 0.03 mm²** | **≈ 0.07 mm²** (2-3× smaller, not 7×, because storage and VCA generators do not shrink) |
| Average analog power | OTAs active in one stage at a time | 7× if un-gated; ≈ 1× if each copy is power-gated outside its window |
| Mismatch | one set of devices shared by all stages (correlated across stages, as Phase 3a §3.9 models) | independent per stage |
| Latency | 413 ns at Tu = 75 ns (or 350 ns at Iu +22%) | ≈ 350-380 ns (no reprogram dead time; hold/transfer still needed) |

### 1.6 When to fall back to physical stage copies

Fall back (fully or partly) when **any** of these holds after a Spectre check of the hold + transfer + settling path:

1. Total overhead needed exceeds **~20% of 350 ns** (t_ovh > ~12 ns per boundary), so Tu would have to drop by more than ~20% (Iu up more than ~25%).
2. The deterministic S&H, transfer and bias-settling error, once included in training (§2.4), still degrades **≥ 2 of 4 specs by ≥ 10%** on the validation set (the 10% rule: `MIN_DEGRADED_DIMS = 2` and `ERROR_THRESHOLD = 0.10`, `ctle_dagger_common.py:116`, `:123`; the Cadence scorer uses the same default, `spice_failure_check.py:262-265`). Include the baseline at the same threshold for comparison (§3.1).
3. The settle charge `C_int·nU_T·(Isat_old/Isat_new − 1)` stays above ~1 LSB (0.03 unit) on more than ~10% of edges at an Isat floor of 0.5, i.e. the floor needed to make it small costs more power than the copies' static bias.
4. Parameter fetch energy exceeds the energy budget (24-120 pJ vs ~4 pJ analog, ±5× uncertain): measure the real SRAM macro before deciding.
5. Area is not the constraint: copies cost 2-3× and avoid reload logic, reload bandwidth and settling dead time.
6. Partial option, tried before copies: **two ping-pong fabrics.** Fabric B settles its bias during stage s on fabric A. Overhead falls to hold + transfer ≈ 8.5 ns (P3 hidden), area ≈ 2× fabric.

## 2. Implementation plan

The code is written **after** the surgery. The surgery worktree (`kn_surgery`, 6 commits) has already removed legacy cell libraries, DEQ, SimContext/noise paths, TapRails and the MoE flags, and has uncommitted edits in `config.py`, `differential_stage.py` and `kirchhoff_net.py`. The hardware code goes on a new branch cut from the finished surgery branch (suggested name `hw-constraints`), not on `hw-refactor`. Line numbers below are from the main checkout and will move; the function names will not.

### 2.1 Module layout

```
hw/
  __init__.py      HWConfig (aggregate), from_json / from_cli / to_json / hash(); parse_hw_args(parser)
  units.py         HWUnits (frozen dataclass) - Vcm, Vu, Cu, Tu, Iu, derived G_u, slope_max, x_hard, kT/C, conversions
  constraints.py   HWConstraints (flag dataclass) + pure torch functions, one per constraint
  timemux.py       TimeMuxConfig + StageBoundary module (S&H, charge injection, droop, kT/C, bias settle, transfer variants)
tests/
  test_hw_units.py  test_hw_constraints.py  test_hw_timemux.py  test_hw_off_is_golden.py  test_hw_learning.py (slow)
```

`hw/units.py`. `HWUnits` is exactly the Phase 3a §2.5 sketch (primary fields `vdd, vcm, vu, cu, t_total_phys, t_total_units, n_slope, temp_c, x_max`; derived `tu, iu, gu, u_t, slope_max, x_hard, ktc_rms_units`; conversions `v_to_x, x_to_v, isat_to_amps, g_to_siemens, t_to_s, power_w`). It imports nothing from the model. Add `t_ovh_per_boundary` (default 0.0, so Tu is unchanged unless the time-multiplexed run sets it) and `tu_effective = (t_total_phys − 6·t_ovh)/t_total_units` so the §1.5 table is computed, not typed in.

`hw/constraints.py`. Pure functions, no module state, each takes tensors plus `HWUnits`. One function per Phase 3a constraint:

| Function | Constraint (Phase 3a section) | Replaces |
|---|---|---|
| `ratio_ab(r_raw, r_max)` → `(A, B)` | bounded A/B (3.1) | `softplus(a_raw), softplus(b_raw)` |
| `slope_bound(g_raw, units, g_min)` | gm·A cap from `slope_max` (3.2) | `gm_min + (gm_max−gm_min)·σ(gm_raw)` |
| `ic_factor(isat, units, i_spec)` | gm–Isat coupling (3.3) | independent gm |
| `compliance_legs(i_tanh, x_d, x_s, ...)` | hard-rail output compliance, per leg (3.4) | input gate σ(...), soft clip role |
| `g_out_term`, `tanh_ste` | g_out and STE floor (3.5) | `ALPHA·u` |
| `vref_map(raw, units, bits)` | Vref symmetric, quantised (3.6) | `σ(raw)·x_max` |
| `theta_bound(theta, units)` | trimmed offset (3.6) | `theta_raw` |
| `xbar_map(W, mode, bits)` | crossbar bound/quantisation (3.7) | `readout_crossbar_W` |
| `c_node_vector(graph, units, mode)` | fixed + parasitic C (3.8) | scalar `c_eff` |
| `power_penalty(isat, sigma_z, g_vca, units)` | `P_avg` penalty (3.9) | `LAMBDAS["power"]` count |
| `mismatch_draw(shape, sigma, gen)` + `apply_mismatch(...)` | per-slot mismatch (3.9) | dead `SimContext` path |
| `quantize_ste(x, bits, lo, hi)` | DAC/code quantisation, shared helper | - |
| `gln_tail(delta, delta_max)` | GLN as tail-current modulation (3.10) | `gln_rails.modulate_gm_z` clamp |
| `isat_eff_clamp(isat, sigma_z, g_vca)` | VCA × Isat range (3.10) | - |

`HWConstraints` is a plain dataclass with the flag names from Phase 3a §3.11 and the defaults from that table (everything `False`/`None`/`0.0`). `HWConfig` is `{units: HWUnits, constraints: HWConstraints, timemux: TimeMuxConfig}` and `HWConfig.is_off()` returns `True` only when every flag is at its default. The builder uses `is_off()` to pass `hw=None` into the model.

`hw/timemux.py`. `TimeMuxConfig` (default OFF; `enable=False`):

```
enable: bool = False
c_hold_ff: float = 100.0        # hold cap per hidden node
inject_mv: float = 1.0          # residual (post dummy-switch) charge-injection offset, constant part
inject_gain_err: float = 0.0    # signal-dependent part (fraction), e.g. 0.002
droop_mv: float = 1.0           # worst-case droop over a stage (deterministic, sign fixed per node)
ktc: bool = True                # add kT/C noise at each sampling event (training only)
settle: bool = False            # bias-settle model on frozen core currents (needs previous-stage currents)
c_int_ff: float = 3.0           # internal node cap used for tau_e
isat_floor: float = 0.0         # 0 = off; 0.5 recommended with settle
xfer_mode: str = "full"         # full | shared_vth | quant | kick | no_tanh | no_relu
param_bits: int | None = None   # QAT on stage parameters (Isat, g, ratio, theta, G, W, transfer)
```

`StageBoundary(nn.Module)` has two methods:

- `capture(x_pre, stage_idx, gen)` → `x_hold` (used only for the frozen core read) and `x_to_transfer`. Applies in this order: kT/C Gaussian (`σ = ktc_rms_units`), constant offset `inject_mv/Vu` (fixed random sign per node, seeded buffer so it is reproducible), gain error `x·(1 + inject_gain_err)`, droop (a fixed-sign offset per node, applied to `x_hold` only because the hold cap is what droops; the transfer reads the node, not the hold cap), `quantize_ste(x, 8 bits)` optional.
- `settle_core(i_core_new, i_core_prev_slot, tau_e_units, t_stage_units)` → effective frozen core current `I_new + (I_prev − I_new)·(τ_e/T)·(1 − e^(−T/τ_e))`, where `τ_e = κ_int/Isat_eff` with `κ_int = C_int·nU_T/(Cu·Vu)·Tu`-consistent units (§1.5). This needs the previous stage's frozen currents. The current `KirchhoffNet.forward` loop does not pass anything between stages except `x`; the hook returns `i_edge_const` per stage from `DifferentialStage` and keeps the last one in the loop (see §2.3). Marked **phase 2**: first land S&H, injection, droop and kT/C.

All noise is generated from an explicit `torch.Generator` and is a no-op when `model.training` is `False` and `hw_noise_eval` is `False`. Deterministic parts (offset, droop, gain error) stay on at eval so that the evaluated model is the one that gets built.

### 2.2 Config and CLI (all OFF by default)

One JSON file plus a few explicit flags. This keeps plumbing through the BO driver to a single passthrough argument.

| Flag | Default | Meaning |
|---|---|---|
| `--hw-config PATH` | none | JSON with `units`, `constraints`, `timemux` sections; unspecified keys keep their defaults |
| `--hw-preset NAME` | none | named bundle written into `hw/presets/*.json`: `phase3a_core` (ratio_ab, slope_bound, ic_coupling, no_alpha), `phase3a_full` (core + compliance + c_node physical + gln tail + vref + xbar), `phase3b_tm` (full + timemux) |
| `--hw-<flag>` | per Phase 3a table | any individual constraint flag, e.g. `--hw-slope-bound`, `--hw-compliance`, `--hw-c-node physical`, `--hw-gln-mode tail`; overrides the JSON |
| `--hw-tm`, `--hw-tm-xfer-mode M`, `--hw-tm-param-bits N`, `--hw-tm-c-hold-ff F` | off | time-multiplexing model |
| `--hw-noise-eval` | off | keep mismatch and kT/C noise on during evaluation |
| `--zig-degrade-thr T` | 0.10 | **optional** override of the failure threshold. The surgery unifies the rule on a single constant, `DEGRADE_REL_THRESHOLD = 0.10` (`ctle_dagger_common.py:122`), so the retrain does **not** need this flag; it exists only to run an experiment at another threshold (§3.1) |

Precedence: CLI flag > `--hw-config` > preset > default. With none of these given, `HWConfig.is_off()` is true and the code takes the legacy branch everywhere.

Where the flags are consumed:

- `dagger-nuance-distillation-kirchhoffnet.py`: argparse block at `:684-795` (`parse_hw_args(_bo_parser)`); `LocalKirchhoffStudentWrapper.__init__` (`:1713-1895`) builds `HWConfig` and passes `hw=` to `build_net_from_config` (the `core_refresh_interval` kwarg already threads the same way through `topology.py:1381, 1728, 1781, 1930, 1971`). The optional `--zig-degrade-thr` override is applied in `setup` (`ctle_dagger_common.py:3584-3587` loops over `HPARAMS_DEFAULTS`); with the flag absent the unified constant `DEGRADE_REL_THRESHOLD = 0.10` applies.
- `kn_bayes_opt.py`: add `--hw-config` (plus the optional `--zig-degrade-thr` override); append them to the Phase-A trial `cmd` list (built at `:1396-1485`, next to `--kn-gln-rails`); include `HWConfig.hash()` in the study fingerprint (`:1059-1070`, "fingerprint-guarded" resume) so a study cannot silently mix hardware and non-hardware trials.
- `topology.py: build_net_from_config` gets `hw: HWConfig | None = None`; `is_off()` → `None`.

### 2.3 Hook points

| Where | Function (current lines) | What the hook does when `hw` is not `None` |
|---|---|---|
| Cell | `FreeTanhLibrary._tanh_core` (`cell_library.py:344-358`) | A, B from `ratio_ab`; gm from `slope_bound` (+ `ic_factor`); θ bound; drop `ALPHA·u` when `hw_no_alpha`; STE floor; mismatch multipliers on Isat, A, B, θ |
| Cell | `forward_tanh` (`:399-401`) | input compliance gate → leg-wise output compliance (`compliance_legs`) when `hw_compliance`; returns the two leg currents so the stage can scatter them separately |
| Cell | `resistive_current` (`:413`) | G mismatch; unchanged otherwise |
| Stage | `rhs` (`differential_stage.py:1768-1779`) | divide by `c_node` `[1, N]` instead of `self.c_eff`; soft clip stays (it is cheap and still bounds extremes), gets a smaller `clip_current` role when `hw_compliance` is on |
| Stage | KCL scatter (`:1508-1514`, `:1343-1345`) | leg-wise `index_add_`: destination gets the P-leg, source gets the N-leg |
| Stage | frozen precompute call (`_forward_heun`, `:1885-1886`) | `x0_hold = boundary.capture(x0)`; `i_edge_const = _compute_i_edge_const(x0_hold, ...)`. Shunt, sense, boundary and leak still read the live `x` |
| Stage | `_compute_frozen_sense` / sense rhs (`:1263`, `:1703`) | Vref map; crossbar `xbar_map` at `:1698` |
| Boundary | `KirchhoffNet.forward` (`kirchhoff_net.py:145-174`) | between the stage call and `self.transfers[i](x)`: `x_hold, x = boundary.capture(x, i)`; stage `i+1` receives `x_hold` for the frozen read; `x` goes through the transfer. Also the sequence path `forward_sequence` at `:256-290` if it is ever used for CTLE (it is not) |
| Transfer | `StageTransfer.forward` (`stage_transfer.py:161-169`) | `xfer_mode` variants, optional output clamp to ±`x_hard`, param quantisation |
| GLN | `GLNRails.modulate_gm_z` (`gln_rails.py:150-168`) | `hw_gln_mode="tail"`: return a multiplier on Isat, not a gm |
| VCA | `_compute_vca_gate` (`differential_stage.py:1059-1125`) | `hw_vca_bits` STE on `ζ`; `isat_eff_clamp` |
| Loss | regulariser accumulation (`train.py:247-304`) | `power_penalty` when `hw_power_lambda > 0` |

Guarantee: every hook is written as `if self.hw is None: <the existing lines, byte-for-byte>` first, then the hardware branch. No refactoring of the legacy path, so the golden suite is unchanged by construction.

One decision to take at implementation time: the leg-wise compliance (`compliance_legs`) makes charge non-conserving at the rail (the supply covers the difference, Phase 3a §3.4). `index_add_` of two separate tensors costs one extra scatter per rhs call. With 10 Heun steps × 2 rhs calls and the core current frozen, only the boundary, sense and shunt paths are live, so the extra cost is small.

### 2.4 Tests

All tests run on ssr0 CPU with the repo venv, float64 where gradients are compared. Existing commands: `venv/bin/python -m pytest tests/test_golden.py -q` (~20 s, `tests/golden/README.md`).

**(a) Unit tests, one per constraint** (`tests/test_hw_constraints.py`, `tests/test_hw_units.py`, `tests/test_hw_timemux.py`):

| Test | Assertion |
|---|---|
| `HWUnits` defaults | `tu = 75.10 ns`, `iu = 46.6 nA`, `gu = 0.6658 µS`, `slope_max = 0.872`, `x_hard = 5.714` (Phase 3a values), conversions round-trip |
| ratio A/B | for random `r_raw ∈ [−10, 10]`: `A + B = 2`, `A/B ∈ [1/r_max, r_max]`, `r_raw = 0` → `A = B = 1`; gradient finite at `r_raw = ±10` |
| slope bound | for random raw params and A, B: `gm_eff·max(Â, B̂) ≤ slope_max·(1 + 1e-9)`; `g ∈ [g_min, 1]`; monotone in `gm_raw` |
| IC coupling | `ic_factor → 1` as `isat → 0`; `ic_factor(isat)` monotone decreasing; matches the closed form `2/(1 + √(1 + 4·IC))` |
| compliance | at `x_n = 0`: leg currents equal `Isat·(1 ± tanh u)/2` within 1e-9; at `x_n = ±x_hard`: the leg that would push outward is < 1e-3 of its unconstrained value; Heun integration of a single driven node never exceeds `x_hard` by more than 2% |
| g_out / STE | forward equals `tanh(u)` exactly; backward gradient at `|u| = 6` equals `ALPHA_BWD` within 1e-9; with `ALPHA_BWD = 0` gradient equals `1 − tanh²` |
| Vref / θ / crossbar | with `bits = 6`: at most 64 distinct values; STE gradient identity; `|W| ≤ w_max`; offset-binary mode returns the same product as signed mode within the quantisation bound |
| `c_node` | `"uniform"` → all ones; `"physical"` → per-node values equal terminal-count formula (hidden 22 fF parasitic, accumulator 26 fF for the paper graph); shape `[1, N]` |
| power | `power_penalty` equals `κ·VDD·Iu·mean_s Σ Isat σ(z) g_vca` on a hand-built case; gradient > 0 w.r.t. Isat |
| mismatch | shared across the 7 stages (same ε for the same slot); mean of 1000 draws of the multiplier within 1% of `exp(σ²/2)`; off in eval unless `hw_noise_eval` |
| time-mux capture | zero config → identity; kT/C std over 20000 draws equals `ktc_rms_units` within 3%; droop sign fixed per node; STE quantiser gradient is 1 |
| settle (phase 2) | `Isat_prev = Isat_new` → factor 1; `Isat_prev → 0` → factor `(τ_e/T)(1 − e^(−T/τ_e))` |
| transfer modes | `full` equals current `StageTransfer`; `shared_vth` has exactly 6 vth parameters over the 6 transfers; `kick` equals `full` numerically at `t_x → 0` |

**(b) "Flags off ≡ golden".** `tests/test_hw_off_is_golden.py` rebuilds every golden case with `hw=HWConfig()` (all defaults) and also with `hw=None`, and asserts the existing `tests/test_golden.py` comparisons (state-dict keys, forward, per-stage states, loss, per-parameter gradients, 30-step Adam trajectory) at `RTOL, ATOL = 1e-6, 1e-7` in float64 for all 17 cases listed in `tests/golden/README.md` (`paper`, `paper_rawinit`, `friedman2`, `toy_mimo`, `fixed_mlp_knet`, and the 12 paper variants). It additionally asserts `set(state_dict.keys())` is unchanged, i.e. the off path adds no parameters and no buffers. Do not regenerate the golden files (`tests/golden/README.md`).

**(c) Small CPU training** (`tests/test_hw_learning.py`, marked `slow`, run on ssr0 CPU; the 48 logical cores make the 4→1 and 4→3 toy sizes cheap). Specs come from `tests/golden/make_golden.py` (`FRIEDMAN2_NET` at `:127-153`, 2 stages, 16 nodes, 4 → 1; `TOY_NET` at `:156-185`, 2 × 8 nodes, 4 → 3, shared-sense, GLN, clip, VCA all on). Use `_build_net` with `hw=HWConfig(preset="phase3a_core")` and `phase3a_full`. Train Adam on the same synthetic data as `make_golden.py`'s trajectories (`n = 256`), but for 400 full-batch steps, 3 seeds. Run the unconstrained model with the same budget as the reference.

Proposed acceptance criteria **(thresholds to calibrate once the unconstrained baseline numbers exist; I have not run them)**:

1. No NaN or Inf in loss or gradients over all steps; every trainable parameter receives a finite gradient on at least one step (except parameters documented as dead, e.g. `u_logits` if it still exists).
2. Final training loss ≤ 0.25 × initial loss for both tasks and every seed (it learns).
3. Constrained final validation MSE ≤ 1.5 × unconstrained MSE for `phase3a_core`, ≤ 2.0 × for `phase3a_full`, and ≤ 2.5 × for `phase3b_tm` (the cost of hardware effects is bounded). On Friedman-2 with 256 train samples, the unconstrained R² on a held-out 256 must be ≥ 0.8 before the comparison counts; otherwise re-tune lr first.
4. After training, the hardware invariants still hold on the trained parameters: `gm_eff·max(Â,B̂) ≤ slope_max`, `|x| ≤ 1.05·x_hard` for hidden and accumulator nodes over the validation set, Heun stability `dt·λ_max ≤ 2` (`hw_check_stiffness`), all quantised parameters take only code values.
5. Seed spread: the three seeds' final constrained loss within a factor of 2 of each other.

## 3. Retraining (SLURM only)

Nothing in this section was run. `sbatch` is not installed on ssr0 (`which sbatch` fails there), so jobs are submitted from the Alliance login node (the repo root holds a file named `annaik@fir.alliancecan.ca`). The two scripts below are to be created in `scripts/slurm/` after the `hw/` code exists; they are documented here in full. Submit from the repo root so that `$SLURM_SUBMIT_DIR` is the repo root (as in the existing scripts): `sbatch scripts/slurm/retrain_hw_phase_a.sh`.

### 3.1 Inputs

| Input | Location | Notes |
|---|---|---|
| Teacher: trial_0019 PlainMLP | `assets/teachers/trial_0019/dagger_student_plain.pt` in the repo (404 KB dir; also `teacher_config.json`, `scaler_X.pkl`, `flow_scaler_C.pkl`) | The existing scripts point at `outputs/generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019/dagger_student_plain.pt` (`run_train_knet_single_shot_distillation_2.sh:35`); the repo copy is the same teacher. Passed as `--phase-a-mlp-teacher-ckpt` |
| Canonical dataset (schema 2, 20k specs, seed 100) | ssr0: `~/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz` (1.46 MB) | Copy to the cluster as `outputs/canonical_ctle_phase_a_v2.npz`. Do **not** reuse the schema-1 `canonical_ctle_phase_a.npz` (`run_train_knet_single_shot_distillation_2.sh:33-34`). It holds the flow labels, the frozen min-power tie-break, the teacher logits column `mlp_logits_trial0019` and the input log bounds (`pipeline_report.md:59`), so every run sees byte-identical data. If it is missing the scripts rebuild it with `dagger-nuance-distillation-kirchhoffnet.py --prepare-canonical-dataset` (`:55-65`), which needs the ZIG model and the CTLE data dir below; a rebuild is not guaranteed bit-identical, so copying is preferred |
| ZIG surrogate | `~/Documents/improved-zig-nf-spline-pytorch-default-v2/` (7.2 MB: `ctle_conditional_flow.pt`, `hybrid_hurdle_ctle_model.pt`, scalers, configs) | Passed as `--teacher-dir`. This is the model that turns the 7 decoded parameters into the 4 specs for the failure rule (`pipeline_report.md:195-196`). Must exist at the same path on the cluster, or set `ZIG_DIR` |
| CTLE training data | `~/Documents/augmented-cvae-ctle` | `--data-dir`, only needed for the prepare step and for the harness `setup()` |
| Paper hyper-parameters | `run_command.txt` next to `phase_a_best.pt`; same as `docs/logs/run_train_knet_single_shot_distillation_2.sh:69-77` | 7 stages, 11 hidden, k = 2, p = 0.2, VCA rank 3, lr 7.069359e-3, wd 6.087302e-3, batch 512, `shared-x2` readout, learnable clip, GLN B = 4 rank 2 on `boundary,readout`, t_span 4.660583, fan-out {0:[0,4], 1:[1,5], 2:[2,6], 3:[3,7]} |
| Paper result (reference only) | `phase_a_history.json`; log tail of `...-59014723.out`: train 7.67%, val 8.05%, test 8.15%, all at the **10%** rule (§3.1) | directly comparable to the hardware rungs; no re-measurement needed |

**The 10% failure rule.** A design fails if **≥ 2 of the 4 specs (power, jitter, eye height, eye width) degrade by ≥ 10%** in the bad direction relative to the target. The 2-of-4 count is `MIN_DEGRADED_DIMS = 2` (`ctle_dagger_common.py:123`; `dagger-nuance-distillation-kirchhoffnet.py:539`) and the threshold is `ERROR_THRESHOLD = 0.10` (`ctle_dagger_common.py:116`). At 5c6a4f4 the two are used together: the test/val metric calls `evaluator.identify_failures(..., threshold=ERROR_THRESHOLD)` (`ctle_dagger_common.py:3888`, `:4075`; also `:3739`, `:3815`), `identify_failures` forwards it as `default_threshold=threshold` (`:2268`), and `compute_forward_errors` then uses `degrade_thr = 0.10` (`:2499-2502`). The Cadence scorer defaults to the same 0.10 (`spice_failure_check.py:262-265`; `pipeline_report.md:206`).

The separate `DEGRADE_REL_THRESHOLD = 0.20` at 5c6a4f4 (`ctle_dagger_common.py:122`) only reached the **log-only** breakdowns that pass no explicit threshold: the per-spec `dim_fail_rates` (`:2291`), `compute_forward_errors_relaxed` (`:2543`) and the phase-A history bundle (`:3248`). It did **not** set the reported train/val/test failure numbers. The surgery removes the split by unifying everything on the single constant `DEGRADE_REL_THRESHOLD = 0.10`, so the retrain needs **no new flag**; `--zig-degrade-thr` (§2.2) is at most an optional override that defaults to that constant.

Consequences:

1. **The paper's numbers are already 10%-rule numbers.** The paper run reports train 7.67 %, val 8.05 %, test 8.15 % (`...-59014723.out`, final line: `final failure: train= 7.67% val= 8.05% test= 8.15%`), all evaluated with `threshold=ERROR_THRESHOLD` = 0.10 at 5c6a4f4. There is nothing to re-measure: the paper baseline and the hardware rungs are on the same rule, so rung R0 only has to reproduce 8.05 % (val) / 8.15 % (test).
2. Training, early stopping and BO scoring all use the same 10 % rule, so there is no second metric and no second threshold to choose. What is left to decide is the acceptance numbers (open question 6).

### 3.2 Script: `scripts/slurm/retrain_hw_phase_a.sh` (one trial, fixed topology)

SBATCH header copied from `run_train_knet_single_shot_distillation_2.sh:2-7` (account, GPU slice, CPUs, memory, 03:00:00, output pattern).

```bash
#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=03:00:00             # Time limit (paper trial: 300 ep x 19-21 s = 1.6-1.75 h; hw flags add overhead)
#SBATCH --output=%x-%j.out          # Output file name

# HR-KNet hardware-constrained retrain, fixed paper topology (7x11, trial_0011 hyper-parameters).
# One rung of the ladder in docs: pick the rung with HW_PRESET (none = baseline at the 10% rule).
#   sbatch --export=ALL,HW_PRESET=phase3a_core,SEED=100 scripts/slurm/retrain_hw_phase_a.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc
source "${SLURM_SUBMIT_DIR}/ENV/bin/activate"

HW_PRESET="${HW_PRESET:-phase3a_full}"      # none | phase3a_core | phase3a_full | phase3b_tm
HW_CONFIG="${HW_CONFIG:-}"                  # optional JSON overriding preset keys
HW_EXTRA="${HW_EXTRA:-}"                    # extra --hw-* flags, e.g. "--hw-tm --hw-tm-c-hold-ff 100"
DEGRADE_THR="${DEGRADE_THR:-}"             # optional override; empty = unified DEGRADE_REL_THRESHOLD = 0.10
SEED="${SEED:-100}"
RUN_TAG="${RUN_TAG:-${HW_PRESET}_s${SEED}}"

TEACHER_CKPT="${TEACHER_CKPT:-${SLURM_SUBMIT_DIR}/assets/teachers/trial_0019/dagger_student_plain.pt}"
CANONICAL="${CANONICAL:-${SLURM_SUBMIT_DIR}/outputs/canonical_ctle_phase_a_v2.npz}"
ZIG_DIR="${ZIG_DIR:-/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2}"
DATA_DIR="${DATA_DIR:-/home/annaik/Documents/augmented-cvae-ctle}"
OUT="${SLURM_SUBMIT_DIR}/outputs/hw_retrain/${RUN_TAG}"
PREPARE_SCRATCH="${SLURM_SUBMIT_DIR}/outputs/phase_a_prepare_scratch_v2"

test -f "$TEACHER_CKPT" || { echo "ERROR: teacher ckpt missing: $TEACHER_CKPT"; exit 1; }
test -d "$ZIG_DIR"      || { echo "ERROR: ZIG model dir missing: $ZIG_DIR"; exit 1; }
test -d "$OUT" && { echo "ERROR: $OUT exists; refusing to warm-start from a stale checkpoint"; exit 1; }

# Loss weights as in the C-full run (single_shot_distillation_2.sh:44-46)
MLP_W=1.0; FWD_W=1.0; PWR_W=0.0005

# Canonical dataset: copy the paper's file when possible; rebuild only if absent
if [ ! -f "$CANONICAL" ]; then
  python dagger-nuance-distillation-kirchhoffnet.py \
    --prepare-canonical-dataset "$CANONICAL" \
    --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
    --phase-a-n-samples 20000 --phase-a-boundary-ratio 0.5 --phase-a-power-norm train_mean \
    --seed 100 --teacher-dir "$ZIG_DIR" --data-dir "$DATA_DIR" \
    --device cuda --output "$PREPARE_SCRATCH"
fi

# Hardware flags (all default OFF; HW_PRESET=none gives the unconstrained baseline)
HW_ARGS=()
KN_ISAT_MAX=2.868855e+01                    # paper search-bound value
if [ "$HW_PRESET" != "none" ]; then
  HW_ARGS+=(--hw-preset "$HW_PRESET")
  KN_ISAT_MAX=16                            # hardware tail ceiling (Phase 3a section 3.10)
fi
if [ -n "$HW_CONFIG" ]; then HW_ARGS+=(--hw-config "$HW_CONFIG"); fi
if [ -n "$HW_EXTRA" ]; then read -r -a _extra <<< "$HW_EXTRA"; HW_ARGS+=("${_extra[@]}"); fi

# Failure threshold: the surgery's unified DEGRADE_REL_THRESHOLD = 0.10 is the rule the paper
# run already used, so no flag is passed unless DEGRADE_THR is set explicitly (an experiment).
THR_ARGS=()
if [ -n "$DEGRADE_THR" ]; then THR_ARGS+=(--zig-degrade-thr "$DEGRADE_THR"); fi

python dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \
  --phase-a-epochs 300 --kn-num-stages 7 --kn-num-hidden 11 --kn-small-world-k 2 --kn-small-world-p 0.200000 --kn-vca-rank 3 \
  --kn-x-max 4.000000 --kn-gm-max 4.241970e+00 --kn-isat-max "$KN_ISAT_MAX" \
  --kn-mapper-lr-scale 1.0 --kn-struct-lr-scale 4.0 --kn-dyn-lr-scale 1.0 \
  --lr 7.069359e-03 --weight-decay 6.087302e-03 --batch-size 512 --earlystop-eval-every 5 \
  --phase-a-validity-weight 0.300000 --phase-a-validity-ramp-start 10 --phase-a-validity-ramp-epochs 30 \
  --phase-a-mlp-weight "$MLP_W" --phase-a-fwd-weight "$FWD_W" --phase-a-power-weight "$PWR_W" \
  --phase-a-power-norm train_mean --kn-readout shared-x2 \
  --kn-learnable-clip-sharpness --kn-gln-rails --kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout \
  --t-span 4.660583 --boundary-fan-out '{"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}' \
  --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --teacher-dir "$ZIG_DIR" --data-dir "$DATA_DIR" \
  ${THR_ARGS[@]+"${THR_ARGS[@]}"} \
  --device cuda --seed "$SEED" --output "$OUT" \
  "${HW_ARGS[@]}"

# Post-train hardware audit (hw/audit.py, ~20 lines: reload phase_a_best.pt, check slope bound,
# rail excursion, Heun stiffness, code-value quantisation, SPICE-export parameter ranges).
if [ -f hw/audit.py ] && [ "$HW_PRESET" != "none" ]; then
  python -m hw.audit --checkpoint "$OUT/phase_a_best.pt" --hw-preset "$HW_PRESET" ${HW_CONFIG:+--hw-config "$HW_CONFIG"}
fi
```

Differences from the paper command, all deliberate:

- `--kn-moe-num-experts 3 --kn-moe-gate-rank 2 --kn-moe-top-k 1` are **dropped**: the surgery removed those inert flags from the argparse and the command builders (commit 6/6 in `kn_surgery`). They no longer exist and must not be passed.
- `--zig-degrade-thr` is **not passed by default**: the failure threshold is the surgery's single constant `DEGRADE_REL_THRESHOLD = 0.10`, which is the rule the paper run already used (§3.1), so no new flag is needed. Set `DEGRADE_THR=0.20` in the environment to run the paper's *stated* 20 % rule as an experiment; the flag is only worth implementing as the optional override of §2.2.
- The `--freeze-read` behaviour needs no flag: `KN_FREEZE_READ = True` is the harness default (`dagger-nuance-distillation-kirchhoffnet.py:622`).
- Paths come from `$SLURM_SUBMIT_DIR` and env vars, not the hard-coded `kirchhoffnet_realistic` paths of the 2026 run (`single_shot_distillation_2.sh:69-77`).
- `--kn-isat-max 16` replaces 28.69 when hardware flags are on. `--kn-gm-max` stays at the paper value for traceability; with `hw_slope_bound` it no longer sets the tanh slope (Phase 3a §3.2) and bounds only the GLN `slope` mode.
- `ENV/` does not exist in the repo checkout (`ls ENV` fails there). The existing scripts assume it on the cluster (`source "${SLURM_SUBMIT_DIR}/ENV/bin/activate"`); the 2026 single-shot run used `kirchhoffnet_realistic/ENV`. Confirm it exists in the clone that holds `hw/` (open question 8).

### 3.3 Script: `scripts/slurm/retrain_hw_bo.sh` (30-trial BO)

SBATCH header copied from `run_phase_a_knet_bo.sh:2-7` (24:00:00 wall time).

```bash
#!/bin/bash
#SBATCH --account=def-shiyusu       # Allocation account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6           # CPU cores
#SBATCH --mem=32G                   # Memory
#SBATCH --time=24:00:00             # Time limit (30 trials x 300 ep; requeue/resume safe: kn_bayes_opt auto-resumes the study.db)
#SBATCH --output=%x-%j.out          # Output file name

# Hardware-constrained Phase-A MLP-guided BO (30 trials). One study dir per HW_PRESET; rerun or chain:
#   j1=$(sbatch --parsable scripts/slurm/retrain_hw_bo.sh)
#   j2=$(sbatch --parsable --dependency=afterany:$j1 scripts/slurm/retrain_hw_bo.sh)
#   sbatch --dependency=afterany:$j2 scripts/slurm/retrain_hw_bo.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load gcc
source "${SLURM_SUBMIT_DIR}/ENV/bin/activate"

HW_PRESET="${HW_PRESET:-phase3a_full}"
HW_CONFIG="${HW_CONFIG:-}"
HW_EXTRA="${HW_EXTRA:-}"
DEGRADE_THR="${DEGRADE_THR:-}"              # optional override; empty = unified DEGRADE_REL_THRESHOLD = 0.10
N_TRIALS="${N_TRIALS:-30}"
SEED="${SEED:-100}"
TEACHER_CKPT="${TEACHER_CKPT:-${SLURM_SUBMIT_DIR}/assets/teachers/trial_0019/dagger_student_plain.pt}"
CANONICAL="${CANONICAL:-${SLURM_SUBMIT_DIR}/outputs/canonical_ctle_phase_a_v2.npz}"
ZIG_DIR="${ZIG_DIR:-/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2}"
DATA_DIR="${DATA_DIR:-/home/annaik/Documents/augmented-cvae-ctle}"
OUT="${SLURM_SUBMIT_DIR}/outputs/hw_retrain_bo/${HW_PRESET}"   # same dir on every resubmission
PREPARE_SCRATCH="${SLURM_SUBMIT_DIR}/outputs/phase_a_prepare_scratch_v2"

test -f "$TEACHER_CKPT" || { echo "ERROR: teacher ckpt missing: $TEACHER_CKPT"; exit 1; }
test -f "$CANONICAL"    || { echo "ERROR: copy canonical_ctle_phase_a_v2.npz from ssr0 first"; exit 1; }

HW_ARGS=(--hw-preset "$HW_PRESET")
if [ -n "$HW_CONFIG" ]; then HW_ARGS+=(--hw-config "$HW_CONFIG"); fi
if [ -n "$HW_EXTRA" ]; then read -r -a _extra <<< "$HW_EXTRA"; HW_ARGS+=("${_extra[@]}"); fi

# Failure threshold: unified DEGRADE_REL_THRESHOLD = 0.10 unless DEGRADE_THR overrides it.
THR_ARGS=()
if [ -n "$DEGRADE_THR" ]; then THR_ARGS+=(--zig-degrade-thr "$DEGRADE_THR"); fi

python kn_bayes_opt.py \
  --dataset ctle --ctle-phase-a \
  --ctle-canonical-dataset "$CANONICAL" --ctle-objective validation \
  --ctle-phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --ctle-phase-a-mlp-weight 1.0 --ctle-phase-a-fwd-weight 1.0 --ctle-phase-a-power-weight 0.0005 \
  --ctle-phase-a-validity-weight 0.3 --ctle-phase-a-validity-ramp-start 10 --ctle-phase-a-validity-ramp-epochs 30 \
  --ctle-phase-a-power-norm train_mean \
  --readout shared-x2 --learnable-clip-sharpness \
  --gln-rails --gln-B 4 --gln-rank 2 --gln-families boundary,readout \
  --param-budget 7000 --param-tolerance 0.15 \
  --epochs 300 --n-trials "$N_TRIALS" --seed "$SEED" --device cuda \
  --no-seed-trial \
  ${THR_ARGS[@]+"${THR_ARGS[@]}"} \
  "${HW_ARGS[@]}" \
  --output "$OUT"
```

Notes:

- The flag set is the union of `run_phase_a_knet_bo.sh:50-59` (dataset, objective, budget, epochs, trials, seed) and the MLP-guided options of the C-full header (`single_shot_distillation_2.sh:9-17`; options at `kn_bayes_opt.py:891-918`), plus the study-level readout / clip / GLN toggles (`kn_bayes_opt.py:732-765`). The paper's BO flags for readout, clip and GLN are not in `run_phase_a_knet_bo.sh`; they come from the trial command shape at `kn_bayes_opt.py:1426-1479`.
- `--no-seed-trial` is needed: the seed trial is a temporal-readout anchor (`kn_bayes_opt.py:1426-1430`) and would not satisfy the shared-sense hardware readout.
- The canonical-dataset build is deliberately **not** repeated here (it fails fast instead), unlike `run_phase_a_knet_bo.sh:37-48`, because every BO resubmission must use the same npz.
- Code changes needed in `kn_bayes_opt.py` (see §2.2): `--hw-config/--hw-preset` passthrough (plus the optional `--zig-degrade-thr` override, defaulting to the unified constant); include `HWConfig.hash()` in the study fingerprint (`:1059-1070`); stop sampling `gm_max` and `isat_max` (`:1380-1388`) when `hw_slope_bound` is on and fix `isat_max = 16`; build `HWUnits` per trial from the sampled `ctle_t_span` (`:1375`), because with `Tu·t_span = 350 ns` the unit current `Iu = Cu·Vu/Tu` moves with `t_span`; give `knet_param_count` (`bo_param_sampling.py:215-308`) the `hw` config so that the parameter budget is evaluated on the hardware model. `ratio_ab` removes one parameter per OTA (52 × 7 = 364 of 5,649 in the paper config, about 6%), inside `--param-tolerance 0.15`.
- A 30-trial study at 24 h per job needs about 3 chained submissions (§3.5). The existing scripts rely on `study.db` resume; I did not test resume with the hardware fingerprint.

### 3.4 Two ways to run

**Option A: fixed paper topology (ladder).** Use trial_0011 exactly (7 × 11, k = 2, rank 3, shared-x2, GLN, same lr and wd) and add hardware constraints one rung at a time, as the Phase 3a enable order recommends. One SLURM job per rung via `retrain_hw_phase_a.sh`. Each rung is compared with R0 (same seed, same 10% rule):

| Rung | `HW_PRESET` (extra flags) | Adds | Question it answers |
|---|---|---|---|
| R0 | `none` | nothing; the unified 10% rule | what does the paper network score at 10%? The paper run already reports 8.05% val / 8.15% test at this rule (§3.1), so R0 is a reproduction check, not a new measurement |
| R1 | `phase3a_core` | ratio A/B, slope bound, IC coupling, no α, θ bound | does the unbounded slope matter? (Phase 3a: init gm·A of 2.9+ against 0.87 hardware max, `circuit_model_report.md:224-228`) |
| R2 | `phase3a_full` | + compliance rail, `c_node` physical, GLN tail, Vref/crossbar quantisation | does the network survive a hard rail and real caps? |
| R3 | `phase3a_full` + `HW_EXTRA="--hw-mismatch-sigma isat=0.05,g=0.10,a=0.01,theta=0.1 --hw-power-lambda 1e-3"` | + mismatch, power penalty | robustness and power |
| R4 | `phase3b_tm` (`HW_EXTRA="--hw-tm --hw-tm-c-hold-ff 100 --hw-tm-xfer-mode shared_vth --hw-tm-param-bits 6"`) | + S&H, injection, droop, kT/C, quantised parameters, simplified transfer | does one time-multiplexed fabric hold the 10% rule? |

Example submissions (a ladder can run in parallel: the rungs do not depend on each other, only the comparison does):

```bash
sbatch --export=ALL,HW_PRESET=none,SEED=100                scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3a_core,SEED=100        scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3a_full,SEED=100        scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3b_tm,SEED=100,HW_EXTRA="--hw-tm --hw-tm-c-hold-ff 100 --hw-tm-xfer-mode shared_vth --hw-tm-param-bits 6" scripts/slurm/retrain_hw_phase_a.sh
# final rung: two more seeds
for s in 101 102; do sbatch --export=ALL,HW_PRESET=phase3b_tm,SEED=$s,HW_EXTRA="--hw-tm ..." scripts/slurm/retrain_hw_phase_a.sh; done
```

R3 is shown as a separate `HW_EXTRA` row because `--export` with spaces in values needs quoting; put it in a JSON file (`HW_CONFIG=hw/presets/r3.json`) in practice.

Cost: one trial is 19-21 s/epoch × 300 = 1.6-1.75 h on one H100 slice (`pipeline_report.md:766`). Hardware code adds work per step (leg-wise scatter, `c_node` vector, S&H noise, quantisation STE, mismatch draws). I have not measured it. Assuming overhead factors 1.0, 1.1, 1.3, 1.3, 1.4 for R0-R4 **[ASSUMPTION]**: 1.7 h × 6.1 = **≈ 10 GPU-h** for the ladder, plus 2 extra seeds on R4 (2 × 1.7 × 1.4 = 4.8 h) = **≈ 15 GPU-h**, plus about 30% for failed or repeated runs = **≈ 20 GPU-h**. A rung at 1.4× overhead is 2.4 h, close to the 03:00:00 limit; above about 1.7× it will time out. The header time comes from the existing scripts as requested. Raise it if the first hardware run measures more than 1.6× the baseline epoch time (open question 9).

**Option B: 30-trial BO.** `retrain_hw_bo.sh`, same BO machinery as the paper (TPE over topology, solver, lr, wd and batch size; objective lexicographic: validation failure, valid mean power, validation MLP MSE). It lets the optimizer re-pick hidden count, stages, k, rank and t_span under the hardware constraints, which the fixed ladder cannot.

| | Option A: fixed topology ladder | Option B: 30-trial BO |
|---|---|---|
| Trials | 5 rungs + 2 seeds = 7 (+ retries) | 30 |
| GPU-h | ≈ 10-15 (≈ 20 with retries) | **≈ 50-55 at baseline cost** (30 × 1.6-1.8 h, `pipeline_report.md:21`); **≈ 60-75** with the 1.2-1.4× hardware overhead assumed above |
| Wall time | one 3 h job per rung; the rungs run in parallel, so about 3 h if 5-7 slices are free | 24 h per job: 3 chained jobs at baseline cost, 3-4 with overhead |
| Output | a clean per-constraint attribution (which constraint costs how much) | the best hardware-aware topology, but confounded: the topology moves with the constraints |
| Risk | the paper topology may be a poor fit for the constraints | stages change the number of S&H boundaries (`6 × t_ovh`, §1.5); `t_span` changes `Tu` and `Iu`; study resume with the new fingerprint is untested |

Recommendation: run Option A first (R0 and R1 give an answer within one day), then run Option B with the preset that passed, restricted so that the number of stages stays at or below 7. The Option B study should start from the paper point if possible (open question 7), because 30 TPE trials from random starts rarely reach the paper's 5,649-parameter solution (the paper's study was 30 trials with a seeded start).

### 3.5 Acceptance for a retrained model

Proposed, to be confirmed by the user (open question 6). All on the 10% rule (fail if ≥ 2 of the 4 specs degrade by ≥ 10%), validation and test splits of the canonical dataset, mean of ≥ 3 seeds for the final rung:

1. Hardware model val failure ≤ R0 val failure + 3 percentage points (absolute), and test failure within 1 point of validation.
2. The per-spec degrade rates (`dim_fail_rates`, `dagger-nuance-distillation-kirchhoffnet.py:2567`; a log-only breakdown that used the 0.20 constant at 5c6a4f4 and uses the unified 0.10 after the surgery) show no single spec above 2× its R0 value.
3. `hw/audit.py` passes: `gm_eff·max(Â, B̂) ≤ slope_max`; |x| ≤ 1.05·`x_hard` for all nodes over the validation set; Heun stability `dt·λ_max ≤ 2`; quantised parameters take only code values.
4. A Spectre run on the exported parameters via `spice_failure_check.py` (default threshold already 0.10) with a gap to the surrogate number of no more than 5 points. This is the only check that closes the loop; it is outside this plan.

## 4. Open questions

1. **Boundary budget.** Is a total of about 413 ns acceptable (Tu = 75.1 ns kept, shadow-bank design, ~10.5 ns per boundary, §1.5), or must 350 ns hold with Iu raised by ~22% (power up by the same factor)?
2. **Reload hardware.** Is an SRAM plus shadow latch bank plus per-OTA current DACs the intended design (about 3.4 kbit per stage, 23 kbit total, fetch energy possibly 10-100× the analog energy)? Or are ping-pong fabrics, or 7 physical copies for a subset of stages, acceptable?
3. **S&H precision.** At C_h = 50 fF the hold needs bottom-plate sampling, a dummy switch and thick-oxide switches. May hold caps be 100-200 fF (18 caps, cheap in area, +3-10 ns of acquisition per boundary)?
4. **Isat floor.** Is a floor of Isat_eff ≥ 0.5 unit (≈ 23 nA at Iu = 46.7 nA) acceptable? It bounds the bias settling time to ~4 ns but costs power on edges that the trial_0011 checkpoint lets sit at 0.18-0.3.
5. **Transfer simplification.** May the transfer lose per-node `vth` (one per boundary), have parameters quantised to 5 bits and be written as a current pulse (`kick` mode), accepting a retrain? Dropping ReLU or tanh entirely looks too costly: w3 is used on 8-18 of 18 nodes in the checkpoint.
6. **Acceptance numbers.** The rule itself is settled: one 10% rule everywhere (fail if ≥ 2 of the 4 specs degrade by ≥ 10%) — training, early stopping, BO scoring and reporting (§3.1). The only remaining choice is the acceptance threshold (the plus-3-point rule above is a placeholder). Should the paper's own 8.05% (val) / 8.15% (test), which are already 10%-rule numbers, be the reference the rungs are compared against?
7. **BO starting point.** Enqueue the trial_0011 hyper-parameters as the first BO trial (it needs a code path, since `--no-seed-trial` removes the temporal anchor), and constrain `num_stages` to ≤ 7 (or exactly 7)?
8. **Environment and data.** Which clone holds the `ENV/` that the scripts source on the cluster (`ENV/` is absent from the repo checkout; the 2026 run used `kirchhoffnet_realistic/ENV`)? Can `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2` and `augmented-cvae-ctle` be mirrored to the cluster at the same paths, and can the canonical npz be copied from ssr0?
9. **Time limit.** Raise the 3 h limit of the single-trial script if the measured hardware overhead exceeds 1.6× (paper trial 1.6-1.75 h)?
10. **GPU accounting.** Are the "50 GPU-h" figures wall-hours of one `3g.40gb` slice, and what is the allocation budget? The Option B estimate (60-75 h with hardware overhead) may exceed it.
11. **Warm start.** Retrain from scratch (assumed; the ratio parameter `r_raw` replaces `a_raw, b_raw`, so `phase_a_best.pt` cannot be loaded), or add a conversion that initialises `r_raw = ln(A/B)/…` and the new `g` from trial_0011 to save epochs?
12. **Seed policy.** One seed per rung and three for the final rung (assumed), or three throughout?
13. **Phase 3a questions that still decide the retrain** (Vu = 0.07 versus 0.1, slope attenuation scheme, single-ended versus differential, crossbar sign scheme, mismatch σ and corners, power and area targets) are listed in `phase3a_units_and_constraints.md` §4. The presets in §3.4 assume the Phase 3a recommended values.
