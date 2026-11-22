# Phase 3a: unit layer and retrain constraints for HR-KNet in GF 22FDX

Scope: documentation only. Nothing in the source repo was modified.
Source repo: `ssr0:~/Documents/kirchhoffnet_refactor`. Paths below are relative to it.
Citations are `file:line` in that repo. Where a number comes from an audit document instead of code, the audit file and line are cited.
Items marked **[ASSUMPTION]** are 22FDX or design estimates that I could not verify from the repo and that need a PDK/Spectre check.

Target: GF 22FDX, VDD = 0.8 V, single-ended nodes around Vcm = 0.4 V, one fabric time-multiplexed over 7 stages, total inference about 350 ns, small caps.

---

## 1. The current dimensionless model

### 1.0 What is and is not available

- The model is fully dimensionless: `C_eff = 1`, arbitrary voltage, current and time units (`config.py:11-16`, `config.py:305`; `docs/analysis/circuit_model_report.md:11`).
- A trained checkpoint of the paper model (7 stages, 11 hidden nodes) **does exist**, but outside the repo checkout: `ssr0:~/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/knet_model_files_manual_trial_0011/phase_a_best.pt`. That directory is the paper run (trial_0011): it also holds `phase_a_last.pt`, `phase_a_history.json`, `run_command.txt` (the paper command) and the `59014723.out` log. The checkpoint is not in the repo checkout (`circuit_model_report.md:21`, `pipeline_report.md:19`). Phase 3b loads it and quotes its fabric-related statistics; see `phase3b_fabric_impl_retrain.md` §1 and the paper-checkpoint numbers quoted in §2.3 below.
- **Every trained statistic quoted in this document is from `fixed_mlp_knet`**, an older model with 4 stages, 14 hidden nodes, k = 4, VCA rank 2 and no GLN (`trained_ctle_audit.md:23-24`). The tables of §1.4 are labelled as such: they are *not* the paper model's numbers. For the paper model the bounds and init values were known, and now the trained values can be read from the trial_0011 checkpoint.
- Paper-model configuration (`circuit_model_report.md:12-21`): 7 stages, 11 hidden + 7 accumulator nodes per stage, 22 core edges + 8 boundary edges + 22 sense edges per stage (52 OTAs/stage, 364 over 7 stages), `gm_max = 4.24197`, `isat_max = 28.68855`, `x_max = 4.0`, `t_span = 4.660583` total (0.6658 per stage), 10 Heun steps per stage (dt = 0.06658), 5,649 parameters.

### 1.1 Node ODE (all 18 nodes per stage)

```
C_eff * dx_j/dt = Σ_in I_edge − Σ_out I_edge + I_boundary + I_sense − g_leak * x_j − I_clip(x_j)
```

| Term | Code | Parameter and parametrisation |
|---|---|---|
| Division by capacitance | `differential_stage.py:1779` (`(write_retain − clip_term) / self.c_eff`) | `C_eff = 1`, fixed (`config.py:305`, `differential_stage.py:313`) |
| KCL scatter, conservative two-terminal | `differential_stage.py:1508-1514` (`index_add_(dst, +i)`, `index_add_(src, −i)`) | none |
| Leak | `differential_stage.py:1761-1766`, constant branch `:881-933` (`l = lf + self.leak_constant`) | fixed 0.0486 = softplus(−3) (`config.py:641`, `differential_stage.py:395`); not trainable. τ_leak = 20.6 units against a 4.66-unit horizon |
| Soft clip | `differential_stage.py:1376-1386`: `I_clip = clip_current·(σ((x−x_max)/s) − σ((−x−x_max)/s))` | `clip_current = 0.05` (`config.py:307`), `s` learnable per stage in [1e-3, 0.2], init 0.02 (`config.py:315-317`, `differential_stage.py:341-349`) |
| Rail | `x_max` | 4.0 for CTLE (`config.py:304` default is 3.0; BO range 0.5-8, `circuit_model_report.md:37`) |
| Integrator | `differential_stage.py:1914-1920`: `x_pred = x + dt·k1; x = x + 0.5·dt·(k1+k2)` | fixed-step Heun, dt = 0.06658 |

The clip current is 0.05 units against OTA currents up to 28.7. It cannot confine the nodes. Observed excursions in the paper-model logs are 4.85-6.43 × x_max (`circuit_model_report.md:27`). In `fixed_mlp_knet`, stage-3 hidden nodes reach 5.35 and 165 of 256 rows leave the rail; the stage-3 entry state is already 4.94 (`trained_ctle_audit.md:34`, `:276-287`).

### 1.2 Core edge (FreeTanhLibrary)

Per directed edge from node `s` (source) to node `d` (destination); positive current flows s → d.

```
A     = softplus(a_raw)                        cell_library.py:344    A > 0, no upper bound
B     = softplus(b_raw)                        cell_library.py:345    B > 0, no upper bound, independent of A
sgn   = sign(s_raw), straight-through grad     cell_library.py:346-347
gm    = gm_min + (gm_max − gm_min)·σ(gm_raw)   cell_library.py:351-352 (gm_min = 0.01)
Isat  = clamp_min(isat_min + (isat_max − isat_min)·σ(isat_raw), 1e-6)   cell_library.py:354-355 (isat_min = 0.01)
θ     = theta_raw                              cell_library.py:356 (zero init; used when bias enabled)
u     = gm·( sgn·(A·Vs − B·Vd) + θ )           cell_library.py:357
I_tanh= Isat·( tanh(u) + 0.01·u )              cell_library.py:358, ALPHA = 0.01 at :260
gate_c= σ((x_max−|Vs|)/0.02)·σ((x_max−|Vd|)/0.02)   cell_library.py:399-401, beta_softness 0.02 (config.py:306)
I_res = softplus(g_res_raw)·(Vs − Vd)          cell_library.py:413-414 (shunt, no compliance gate)
I_edge= gate_c·I_tanh·σ(z_e)·g_vca + I_res·σ(z_e)      differential_stage.py:1490-1545
```

Notes that matter for hardware:

- `gm` is the **argument slope of the tanh in 1/(volt unit)**. It is not a transconductance. The small-signal transconductance is `Isat·gm·A` (`circuit_model_report.md:107`), in current-unit per volt-unit.
- θ sits inside the gm product (`cell_library.py:357`). Sign `sgn` multiplies the differential part only, not θ.
- The 0.01·u term is a linear function of the OTA input (`cell_library.py:358`). It has no hardware meaning (`feature_surgery_audit.md:210`).
- With `freeze_read=True` (paper config) the whole tanh branch is evaluated once at stage entry and held for the 10 Heun steps. Only shunt, leak, clip, boundary and sense currents see the evolving state (`differential_stage.py:1880-1886`, `circuit_model_report.md:105`).
- σ(z_e) multiplies both the tanh and the shunt (`differential_stage.py:1527-1545`), so it can be folded into Isat and G at export. It is fractional (0.39-0.83) in the trained model (`trained_ctle_audit.md:35`, `:255`).
- The parallel resistive shunt of the core family is counted **exactly once** in every path: the frozen path (`freeze_read=True`), the dynamic path (`freeze_read=False`), the 3.4 compliance path and the boundary / sense / mesh families. The dynamic path used to add it twice (once inside the cell forward `i_tanh + i_res` and once in the `_has_resistive` block of `rhs`); that was a bug and was fixed in commit `7cda290` (the core cell current there is now the tanh-only `forward_tanh`). The core VCA gate therefore scales the tanh part only -- the shunt is scaled by `g_vca` only with `--vca-gate-shunt`.

Initial values (`cell_library.py:290-313`): `a_raw = b_raw = 0` (A = B = 0.693), `gm_raw = isat_raw = −5` (σ = 0.0067; gm and Isat ≈ 0.077 for ceiling 10, ≈ 0.04 and 0.20 for the paper ceilings 4.24 and 28.7), `g_res_raw = −5` (G = 0.0067), `z = 0` (σ = 0.5).

### 1.3 Boundary, sense, VCA, GLN, Vref

| Block | Equation | Code |
|---|---|---|
| Boundary OTA (input injection, 8/stage) | same cell as core with `Vs = u` (ideal terminal, never drained), then `×σ(zb)×g_vca,b`; gm may be GLN-modulated | `differential_stage.py:1547-1609` |
| Sense OTA (22/stage) | cell with `Vs = x_j`, `Vd = Vref`; currents mixed by signed crossbar `W[7×22]` into the 7 accumulators | `differential_stage.py:1688-1759`; `readout_crossbar_W` at `:688`, `:1698` |
| Vref | `Vref = σ(raw_vref_sense)·x_max`, in [0, x_max], init x_max/2 (`raw_vref_init = 0`) | `differential_stage.py:1263`, `:1703`; `config.py:330-333` |
| VCA | `g_vca = 2σ((u·W)·v_eᵀ)` in (0, 2), rank r (paper: 3), DC per sample; multiplies core tanh only (shunt only with `--vca-gate-shunt`), boundary and sense tanh + shunt | `differential_stage.py:1059-1125`, `:1527-1545`; `circuit_model_report.md:117-122` |
| GLN | `gm_e = clamp(gm0_e·exp(z·Wᵀ), gm_min, gm_max)`, B = 4 rails, tied across stages; boundary and sense families only | `gln_rails.py:167-168` (`gm = gm0 * exp(delta); clamp`), `:11`, `:32` |
| Interstage transfer | `x' = w1·x + w2·tanh x + w3·ReLU(x − vth)` per node, all 18 nodes, no clamp on the output | `stage_transfer.py:98-102`; `circuit_model_report.md:166-170` |

### 1.4 Parameter ranges and trained statistics

Bounds and init (paper model):

| Quantity | Bound / init | Source |
|---|---|---|
| gm | [0.01, 4.24197], init ≈ 0.04 | `circuit_model_report.md:14`, `:106` |
| Isat | [0.01, 28.68855], init ≈ 0.20 | same |
| A, B | softplus, > 0, unbounded, init 0.693 | `cell_library.py:344-345` |
| G (shunt) | softplus, init 0.0067 | `cell_library.py:413`, `:312` |
| t_span | 4.660583 total, 0.6658 per stage | `circuit_model_report.md:19` |
| x_max | 4.0 | `circuit_model_report.md:15` |

Trained `fixed_mlp_knet` (4 stages; bounds [0.01, 10] for gm and Isat; min / median / max over edges; `trained_ctle_audit.md:247-257`):

| Quantity | Core | Boundary | Readout |
|---|---|---|---|
| gm | 0.077 / 0.12-0.13 / 0.33 | 0.078 / 0.10 / 0.35 | 0.059 / 0.10-0.11 / 0.24 |
| Isat | 0.077 / 0.12-0.13 / 0.39 | 0.078 / 0.10 / 0.35 | 0.058 / 0.10-0.11 / 0.24 |
| A | 0.49 / 0.9-1.0 / 2.35 | 0.69 / 0.85 / 1.43 | 0.25 / 0.67-0.79 / 2.65 |
| B | 0.36 / 0.83-0.93 / 1.64 | 0.28 / 0.86 / 1.11 | 0.43 / 0.82-0.85 / 1.29 |
| gm·A | 0.048 / 0.088-0.135 / 0.53 | 0.053 / 0.085 / 0.38 | 0.019 / 0.065-0.075 / 0.53 |
| Isat·gm·A (small-signal) | 0.004 / 0.011-0.018 / 0.135 | 0.004 / 0.009 / 0.13 | 0.001 / 0.007 / 0.12 |
| θ | −0.73 / ±0.2 / +0.73 | −0.66 / −0.09 / +0.34 | −0.39 / −0.15 / +0.23 |
| G | 0.0016 / 0.0035-0.0045 / 0.013 | 0.005 / 0.007 / 0.020 | 0.002 / 0.005 / 0.026 |
| σ(z) | 0.39 / 0.58-0.64 / 0.83 | 0.47 / 0.52-0.53 / 0.82 | 0.40 / 0.57-0.60 / 0.77 |

Operating point (`trained_ctle_audit.md:36`, `:289`): no device is near its ceiling; the tanh argument has max |u| ≈ 2 (stage-3 core 1.98) and median < 1; the fraction with |u| > 2 is 0. The trained network is nearly linear. Scaled inputs are within ±0.91 of a 4-unit rail (`trained_ctle_audit.md:278`). Sign: `sgn = +1` on only 7-12 % of core edges in stages 1-3 (`trained_ctle_audit.md:258`).

Dependence on current features (`trained_ctle_audit.md:30-33`): live core OTAs instead of frozen → failure 9.0 % → 26.7 %; VCA gains forced to 1 → 99.3 %; G → 0 → 69.6 %. These are the audit's `fixed_mlp_knet` numbers at the **20 %** degrade gate (`trained_ctle_audit.md:11-14`, `:27`; the same model fails 22.6 % at the 10 % gate, `:29`) — they are not 10 %-rule numbers and must not be read as such. The failure rule used for the retrain, and quoted throughout Phase 3b, is the **10 %** gate: a design fails if ≥ 2 of the 4 specs (power, jitter, eye height, eye width) degrade by ≥ 10 % in the bad direction relative to the target.

The paper's own maxima (gm 4.24, Isat 28.7) therefore describe the search bounds. They are not values observed in any trained model: the `fixed_mlp_knet` numbers above top out at gm 0.35 / Isat 0.39, and the paper checkpoint (trial_0011) reaches gm 0.37 / Isat 2.69 on core edges (§2.3).

---

## 2. The unit layer

### 2.1 Definitions

The ODE is scale-free (`circuit_model_report.md:49`). Three scales are free; the other two follow.

```
V_phys = Vcm + Vu · x            (Vu  [V/unit], Vcm = 0.4 V)
I_phys = Iu · i                  (Iu  [A/unit])
t_phys = Tu · t                  (Tu  [s/unit])
C_u    = Iu · Tu / Vu            (physical capacitance of C_eff = 1)
G_u    = Iu / Vu = C_u / Tu      (physical conductance of 1 unit)
```

Mapping of model quantities: `C_phys = C_eff·C_u`; shunt `G_phys = G·G_u`; leak `g_leak·G_u`; `Isat_phys = Isat·Iu`; tanh slope `gm_phys [1/V] = gm/Vu`; small-signal OTA transconductance `gm_ss,phys = Isat·gm·A·G_u`.

### 2.2 Numerical check of the worked example

Inputs: ±4 units ↔ ±0.4 V so Vu = 0.1 V; C_u = 50 fF; 4.660583 units ↔ 350 ns (`circuit_model_report.md:19`).

| Quantity | Formula | Value | Matches the proposal? |
|---|---|---|---|
| Tu | 350 ns / 4.660583 | 75.10 ns | yes (75 ns) |
| Iu | C_u·Vu/Tu = 50 fF · 0.1 V / 75.1 ns | 66.58 nA | yes (≈67 nA) |
| G_u | Iu/Vu = C_u/Tu | 0.6658 µS | yes (0.67 µS) |
| Isat = 28.68855 | × Iu | 1.910 µA | yes (1.9 µA) |
| Isat = 0.12 (trained median) | × Iu | 7.99 nA | |
| Isat = 0.39 (trained max) | × Iu | 26.0 nA | |
| leak 0.0486 | × G_u | 32.36 nS = 30.9 MΩ | yes (32 nS, 31 MΩ) |
| shunt G = 0.0016 / 0.004 / 0.0067 / 0.026 | × G_u | 1.07 / 2.66 / 4.46 / 17.3 nS | 58-940 MΩ |
| Heun dt = 0.06658 | × Tu | 5.00 ns | |
| per-stage time 0.6658 | × Tu | 50.0 ns | |
| kT/C, C = 50 fF, 300 K | √(kT/C) | 0.288 mV (0.0029 units at Vu = 0.1) | yes (0.29 mV) |

All numbers in the example are arithmetically correct. kT/C is negligible: 0.29 mV against a 0.2-0.3 V swing is 57-60 dB. OTA thermal noise and mismatch dominate, not the capacitor.

A 31 MΩ leak and 59-940 MΩ shunts are not passive resistors in 22FDX. They must be active (weak-inversion MOS in triode, pseudo-resistor, or an OTA). The leak is the easy one: at 32 nS it needs about 1-2 nA of weak-inversion bias (g ≈ I/(n·U_T) → 32 nS × 1.3 × 25.9 mV = 1.1 nA). The trained shunts (1-17 nS) need a sub-nA device. See open question 6.

### 2.3 Critique of the example

**(a) Single-ended headroom.** ±4 units × 0.1 V = ±0.4 V is the entire 0 to 0.8 V supply. No device works with zero Vds. Estimate **[ASSUMPTION, check with Spectre]**:

- Output mirrors need Vdsat ≈ 4-5·U_T ≈ 0.10-0.13 V in weak/moderate inversion, so an output node stays within about 0.12-0.68 V.
- An NMOS-only input pair in weak inversion needs Vgs (about 0.25-0.3 V for an LVT-class device at nA) plus a tail source Vds (about 0.1 V), so its input floor is about 0.35 V. That is almost exactly Vcm. Hence the node-as-input needs a complementary (N+P) input pair or a level shifter. A complementary pair has a gm bump where both halves conduct. That bump is a distortion the model does not have.
- Usable swing is therefore about ±0.25-0.3 V. Two consistent choices:
  - x_max ≈ 2.5-3 at Vu = 0.1 V. This changes x_max from the current 4 (`config.py:304` default 3.0, CTLE 4.0) and changes every learned excursion.
  - **Keep x_max = 4 and take Vu = 0.07 V** (±0.28 V, nodes 0.12-0.68 V). Recommended: it keeps the trained dynamic range in units, the golden baseline stays comparable, and only the unit scale changes.
- The trained models do not stay inside x_max (`fixed_mlp_knet`): stage-3 hidden nodes reach 5.35 and 165/256 rows leave the rail (`trained_ctle_audit.md:34`, `:287`). With Vu = 0.07 that is 0.37 V above Vcm = 0.77 V, essentially at VDD. A hard rail (§3.4) will change the trained solution. Plan to retrain, not to re-scale.

**(b) Subthreshold tanh slope.** A source-coupled pair gives I = I_B·tanh(ΔV/(2nU_T)). With U_T = 25.85 mV at 27 °C:

| n | slope [1/V] | per unit at Vu = 0.1 V | per unit at Vu = 0.07 V |
|---|---|---|---|
| 1.0 | 19.3 | 1.93 | 1.35 |
| 1.2 | 16.1 | 1.61 | 1.13 |
| 1.3 | 14.9 | 1.49 | 1.04 |
| 1.4 | 13.8 | 1.38 | 0.97 |

At 85 °C (U_T = 30.9 mV, n = 1.3) the slope is 12.5 /V: 1.25 per unit at Vu = 0.1 V and **0.87 per unit at Vu = 0.07 V**. So a plain pair gives gm·A ≤ about 1-1.9 per unit, with no way to go steeper.

- The paper's bound gm_max = 4.24 (`circuit_model_report.md:14`) with unbounded A (`cell_library.py:344`, init 0.693) allows gm·A of 2.9 at init and more once A grows. That is above the hardware limit by 2-3×, so the **bound** is infeasible.
- The trained `fixed_mlp_knet` values are not: gm·A is 0.019-0.53 (median 0.065-0.135) (`trained_ctle_audit.md:251`). The paper-model checkpoint exists (trial_0011, §1.0), and its trained gm·A is of the same kind: **0.014-0.93 on the core edges** (per-stage median 0.046 at stage 0 rising to 0.116 at stage 6) and 0.005-0.18 on the sense edges. (Computed from the checkpoint with `gm = gm_min + (gm_max − gm_min)·σ(gm_raw)`, `gm_max = 4.24197`, `A = softplus(a_raw)`; the static gm map, so the GLN modulation of the boundary/sense families is not included.)
- **The feasibility conclusion is unchanged.** The *bound* (init gm·A ≈ 2.9) stays infeasible, as above; the *trained* values of the paper model are not: they sit below the 0.87 hardware maximum, and only the single worst core edge (0.927, stage 6) exceeds it, by 6 %. So the slope bound of §3.2 clips one edge, while the median (0.046-0.116) is 8-19× shallower than the hardware maximum (0.87) — the attenuation problem, not the upper bound, is what the paper model runs into.
- The trained values sit far *below* the hardware maximum. A median gm·A of 0.1 is 10× shallower than 1.04. Reaching it needs input attenuation (capacitive divider on the pair inputs) or source degeneration. A bias increase does not lower the slope in weak inversion. This is the harder design problem than the upper bound.

**(c) Per-unit currents, inversion, bandwidth.** **[ASSUMPTION]** I_spec ≈ 2nµCox·U_T²·(W/L) ≈ 0.3-0.5 µA per square for 22FDX NMOS; Cgg ≈ 1-2 fF for a minimum-ish OTA input device; check against the PDK.

- Trained-like edges (Isat 0.12-0.39 → 8-26 nA at Iu = 66.6 nA): each pair branch carries 4-13 nA. Inversion coefficient IC = I/I_spec ≈ 0.01-0.03 at W/L = 1: deep weak inversion. The 1/(2nU_T) slope applies fully.
- Paper-ceiling edges (Isat = 28.7 → 1.9 µA): each branch 0.95 µA, IC ≈ 2-3 at W/L = 1, moderate inversion. Slope is about 0.5× the weak-inversion value (§3.3, f at IC = 2.4). Power: 52 physical OTAs × 1.9 µA × 0.8 V = 79 µW worst case. With the trained medians: 52 × 8 nA × 0.8 V = 0.33 µW.
- Bandwidth of the *modeled* poles is fine. At Isat = 1.9 µA, gm_ss = 1.9 µA × 14.9 /V = 28 µS and ω = gm_ss/C_u = 5.7×10⁸ rad/s (τ = 1.8 ns = 0.024 Tu). This equals Isat·gm·A·(1/Tu) in the model, which the Heun step (dt = 0.0666) handles only if dt·ω ≲ 0.5 (see §3.3). At the paper bounds (gm_max·Isat_max = 121) it would not.
- Bandwidth of the *unmodeled* internal poles is the real risk. A mirror node in weak inversion has τ_int = C_int·n·U_T/I_branch. With C_int = 2 fF and I_branch = 4 nA: τ = 2 fF × 33.6 mV / 4 nA = 17 ns = 0.22 Tu. The model has no such pole. Requiring τ_int ≤ Tu/16 (4.7 ns) gives I_branch ≥ 14 nA, so **Isat ≥ ≈28 nA ≈ 0.6 units at Iu = 46.7 nA**, which is above the trained median (0.12) and the trained maximum (0.39) of `fixed_mlp_knet`. It is also above the core minimum of the paper checkpoint (trial_0011: core Isat 0.18-2.69, per-stage median 0.37-0.67; Phase 3b §1.5 quotes the same range), so this floor is binding on the paper model too. Treat as a design floor `isat_min_hw`.

**(d) Other issues.**

- kT/C: fine (above). Larger C buys current for the same Tu (Iu = C·Vu/Tu): C = 100 fF → Iu = 93 nA, floor 0.3 units, at 2× cap area.
- Parasitics are 30-60 % of 50 fF (§3.8). The "50 fF" is the total node capacitance, so the *drawn* fixed cap is about 30 fF.
- Time multiplexing over 7 stages: every edge's Isat, gm trim, A/B, θ, s, G, σ(z) must be reloaded each 50 ns. The unit layer does not change, but reload settling (DAC) must be much faster than 50 ns (open question 4).

### 2.4 Recommended parameter set

| Parameter | Value | Reason |
|---|---|---|
| VDD / Vcm | 0.8 V / 0.4 V | given |
| **Vu** | **0.07 V** | keeps x_max = 4 and ±0.28 V swing (0.12-0.68 V) |
| x_max | 4.0 (unchanged) | stays comparable to the golden baseline |
| x_hard (physical limit) | VDD/2/Vu = 5.714 | hard-rail point for §3.4 |
| C_u | 50 fF | total node capacitance, parasitics included |
| **Tu** | **75.1 ns** (use 75 ns → 349.5 ns total) | 4.660583 units ↔ 350 ns |
| Iu = C_u·Vu/Tu | 46.7 nA (at Tu = 75 ns) | |
| G_u = C_u/Tu | 0.667 µS | |
| Isat range | [0.6, 16] units = [28 nA, 0.75 µA] | lower: internal pole floor; upper: power (52 × 0.75 µA × 0.8 V = 31 µW) |
| gm·A range | [s_max/16, s_max] with s_max = Vu/(2nU_T), n = 1.3, 85 °C → [0.055, 0.87] | steepest = plain pair; 16:1 attenuation range |
| A, B | A = 2r/(1+r), B = 2/(1+r), r ∈ [1/4, 4] → A, B ∈ [0.4, 1.6] | §3.1 |
| g_out | λ·Vu = 0.0105 per unit Isat (λ = 0.15 /V) | §3.5 |
| Leak | 0.0486 → 32 nS (active) | unchanged |
| Vref | per stage, [−x_max, x_max] | §3.6 |
| Heun | dt = 0.0666 unit = 5 ns; require dt·(Σ small-signal conductance at node) ≲ 0.5 | accuracy, not stability |

### 2.5 `HWUnits` sketch

```python
from dataclasses import dataclass
import math

K_B, Q_E = 1.380649e-23, 1.602176634e-19

@dataclass(frozen=True)
class HWUnits:
    # --- primary (the only knobs) ---
    vdd: float = 0.8            # V
    vcm: float = 0.4            # V
    vu: float = 0.07            # V per unit
    cu: float = 50e-15          # F, physical capacitance of C_eff = 1
    t_total_phys: float = 350e-9    # s, whole inference
    t_total_units: float = 4.660583 # model t_span (circuit_model_report.md:19)
    n_slope: float = 1.3        # subthreshold slope factor (ASSUMPTION)
    temp_c: float = 85.0        # worst case for slope bound
    x_max: float = 4.0          # soft/design rail in units (config.py:304, CTLE 4.0)

    # --- derived ---
    @property
    def tu(self) -> float:      return self.t_total_phys / self.t_total_units
    @property
    def iu(self) -> float:      return self.cu * self.vu / self.tu
    @property
    def gu(self) -> float:      return self.cu / self.tu          # = iu / vu
    @property
    def u_t(self) -> float:     return K_B * (self.temp_c + 273.15) / Q_E
    @property
    def slope_max(self) -> float:
        """Max tanh slope per unit: vu / (2 n U_T). 0.87 at vu=.07, 85C, n=1.3."""
        return self.vu / (2.0 * self.n_slope * self.u_t)
    @property
    def x_hard(self) -> float:  return self.vdd / 2.0 / self.vu   # 5.714
    @property
    def x_swing_v(self) -> float: return self.x_max * self.vu     # 0.28 V
    @property
    def ktc_rms_units(self) -> float:
        return math.sqrt(K_B * (self.temp_c + 273.15) / self.cu) / self.vu

    # --- conversions ---
    def v_to_x(self, v):   return (v - self.vcm) / self.vu
    def x_to_v(self, x):   return self.vcm + self.vu * x
    def isat_to_amps(self, i): return i * self.iu
    def g_to_siemens(self, g): return g * self.gu
    def t_to_s(self, t):   return t * self.tu
    def power_w(self, isat_sum_units: float) -> float:
        return isat_sum_units * self.iu * self.vdd
```

Checks against the defaults: tu = 75.10 ns, iu = 46.6 nA, gu = 0.6658 µS, slope_max = 0.872 (85 °C), x_hard = 5.714. This is the single source of truth. Every constraint in §3 reads from it.

---

## 3. Retrain constraints

Common rules:

- Every new behaviour sits behind a flag in a single `HWConstraints` config (table at the end of this section). **All flags default to `False`/`None`**, and with all flags off the forward pass is bit-identical to today's (golden baseline).
- All unit-dependent constants come from `HWUnits` (§2.5).
- `x` is the model state in units; `x = (V − Vcm)/Vu`.
- "Edge terminals": the device is conservative in the model (`differential_stage.py:1508-1514`), but a real edge has two output legs, one on each node. Each leg is supply-referenced (§3.4).

### 3.1 Bounded A/B (ratio form)

- **Current code:** `A = softplus(a_raw)`, `B = softplus(b_raw)`, independent and unbounded (`cell_library.py:344-345`). Trained `fixed_mlp_knet` ranges: A 0.25-2.65, B 0.28-1.64 (`trained_ctle_audit.md:249-250`). Stage-0 core edges have A = B = 0.693 exactly (they never received gradient, `trained_ctle_audit.md:260`).
- **Change:** one ratio parameter per edge instead of two:
  ```
  r = exp( ln(r_max) · tanh(r_raw) ),   r ∈ [1/r_max, r_max],  r_max = 4,  r=1 at r_raw=0
  A = 2r/(1+r),   B = 2/(1+r)            →  A + B = 2,  A/B = r,  A,B ∈ [0.4, 1.6]
  ```
  `r_raw` initialises at 0 (A = B = 1). The old init (0.693) is not reproduced; it is a different model, so this flag is a retrain flag, not an epoch-0 identity.
- **Mapping:** A and B are the input attenuations of the two pair inputs. A passive capacitor divider cannot exceed 1. Realised as `Â = A/max(A,B)`, `B̂ = B/max(A,B)` (so max = 1), with the common gain folded into the overall slope factor of §3.2: `u = s_max·g·(sgn·(Â·x_s − B̂·x_d) + θ)`. Â, B̂ are set by a 3-4-bit capacitor-ratio bank per input, or by a W-ratio / degeneration ratio. In Verilog-A: `V(pin_p) = Vcm + Â·(V(s)−Vcm)`; `V(pin_n) = Vcm + B̂·(V(d)−Vcm)`.
- **Effect:** removes the unbounded slope `gm·A`. A ≠ B gives a common-mode term `(A−B)·x_cm` that a real pair does not have; the ratio form bounds A−B = 2(r−1)/(r+1) to ±1.2 rather than unbounded. Trained readout edges with r outside [1/4, 4] (e.g. A = 2.65 against B = 0.43 is r ≈ 6) are clipped to the bound. Expect a small accuracy cost; measure it before enabling other flags.
- Flag: `hw_ratio_ab`. Code site: `_tanh_core` (`cell_library.py:344-345`).

### 3.2 gm·A bound from the subthreshold slope

- **Current code:** `gm = gm_min + (gm_max − gm_min)·σ(gm_raw)` with `gm_max = 4.24197` (`cell_library.py:351-352`), `u = gm·(sgn·(A·Vs − B·Vd) + θ)` (`:357`). Effective slope `gm·A` ≈ 4.2 × A, unbounded because of A.
- **Change:** replace the free gm by a bounded attenuation `g ∈ [g_min, 1]` and a fixed hardware slope:
  ```
  s_max = Vu / (2·n·U_T)                          [per unit; 0.87 at Vu=0.07 V, n=1.3, 85 °C]
  gm_eff = s_max · f_IC(Isat) · g                  (f_IC = 1 in weak inversion; see §3.3)
  g = g_min + (1 − g_min)·σ(gm_raw),  g_min = 1/16
  u = gm_eff · ( sgn·(Â·x_s − B̂·x_d) + θ )        with max(Â, B̂) = 1
  ```
  So the slope per input is `gm_eff·Â ≤ s_max`. The raw parameter `gm_raw` keeps its name and its init (−5), so existing optimizer state shape is unchanged.
- **Mapping:** `g` is the attenuation (capacitor divider, or source degeneration `1/(1+gm_d·R_s)`). `s_max` is the pair itself: `I = I_B·tanh(ΔV/(2nU_T))`. In Verilog-A: `I(d,s) <+ Isat*tanh(slope*ΔV)` with `slope = g/(2*n*$vt)` (`$vt` makes the temperature dependence explicit).
- **Effect (quantitative):** the paper's bound gm·A of 2.9+ at init is 3.3× above the hardware 0.87. The trained `fixed_mlp_knet` gm·A is 0.019-0.53, median 0.065-0.135 (`trained_ctle_audit.md:251`): those are 0.075-0.155 of s_max, which sits at g ≈ 0.1. That means about 10:1 attenuation (divider or degeneration) on most edges. The bound itself will not bite; the lower bound `g_min = 1/16` will not either for the median, but the small-signal floor `Isat_min·s_max·g_min = 0.6 × 0.87/16 = 0.033` is 2-5× above the trained median Isat·gm·A (0.007-0.018, `trained_ctle_audit.md:252`). Retraining must compensate through the shunt G and the VCA, or `g_min` must go to ≈ 1/50 (open question 2).
- Flag: `hw_slope_bound` (uses `HWUnits.slope_max`). Replaces `gm_max` for the tanh slope; `gm_max` still bounds the GLN clamp when GLN is on.

### 3.3 gm–Isat coupling via the inversion coefficient

- **Current code:** `gm` and `Isat` are two independent sigmoids (`cell_library.py:351-355`, init both −5). In the trained model they are nearly identical numerically (medians 0.12-0.13 for both, `trained_ctle_audit.md:247-248`), which is a coincidence of the shared init.
- **Physics:** for a source-coupled pair with tail current `I_B = Isat·Iu` and each device at `IC_d = (I_B/2)/I_spec`:
  ```
  slope_phys = 1/(2nU_T) · f(IC),   f(IC) = 2 / (1 + sqrt(1 + 4·IC)),   IC = Isat·Iu / (2·I_spec)
  ```
  `f → 1` in weak inversion (IC ≪ 1) and `f ≈ 1/√IC` in strong inversion. Here IC is per device, with each device carrying half the tail current. `I_spec = I_spec□ · (W/L)` with `I_spec□ ≈ 0.5 µA` **[ASSUMPTION]**; add `i_spec` to `HWUnits` as a per-design-class field.
- **Change:** in `_tanh_core` replace the gm lookup by `gm_eff = s_max · f(IC(Isat)) · g` (the §3.2 formula with `f` active). `Isat` remains the only free current. `f` is smooth, so the gradient flows to `isat_raw` through the slope as well as through the amplitude.
- **Numbers:** with W/L giving `I_spec` = 2 µA: Isat = 0.6 units (28 nA) → IC = 0.007, f = 0.99; Isat = 16 units (0.75 µA) → IC = 0.19, f = 0.86. In this design range the weak-inversion slope holds within 14 %. A design with I_spec = 0.5 µA (W/L = 1) gives IC = 0.75 and f = 0.67 at 0.75 µA. **The coupling matters only if the OTAs are sized small.** Hence it is optional once the bound in §3.2 is on.
- **Heun step limit:** `dt·λ` for a node is `dt·Σ_edges Isat·gm_eff·Â`. With Isat = 16, s_max = 0.87, 3 live edges on a node: λ = 42 → dt·λ = 2.8 at dt = 0.0666. Heun is unstable above 2. Today's trained model has λ ≈ 0.1 and no problem (`trained_ctle_audit.md:252`). If `freeze_read` is off, or the ceiling approaches 16, use `num_steps` ≥ 40 per stage (dt = 0.017) for λ ≲ 40 → dt·λ ≤ 0.7. Add a check `hw_check_stiffness` that prints `max(dt·λ)`.
- **Mapping:** the pair's W/L and tail mirror ratio set `I_spec` and hence the slope at a given Isat.
- Flag: `hw_ic_coupling`.

### 3.4 Hard physical rails: output-compliance roll-off

- **Current code:**
  - The soft clip `I_clip = 0.05·(σ((x−x_max)/s) − σ((−x−x_max)/s))` at `differential_stage.py:1376-1386`, applied at `:1768`: 0.05 units against OTA currents up to 28.7. It cannot confine the state; observed nodes at 4.85-6.43 × x_max (`circuit_model_report.md:27`) and 5.35 at stage 3 (`trained_ctle_audit.md:34`).
  - The **input** compliance gate `σ((x_max−|Vs|)/0.02)·σ((x_max−|Vd|)/0.02)` multiplies the tanh branch (`cell_library.py:399-401`). When a node exceeds the rail by > 0.04 units, its OTA currents turn off and the node stays outside (`circuit_model_report.md:135`).
  - The interstage transfer can write any value (`stage_transfer.py:98-102`; post-readout 7.65, `trained_ctle_audit.md:288`).
- **Physics:** the OTA is a pair whose two branch currents `I_± = Isat·(1 ± tanh u)/2` are mirrored to the node by a P leg (sources current from VDD) and an N leg (sinks to ground). Each leg stops working when its Vds falls below ≈ 4U_T. With `x_hard = VDD/(2Vu)` = 5.714 at Vu = 0.07:
  ```
  c_top(x) = 1 − exp( − softplus_k( (x_hard − x)·Vu/U_T − V_off ) )      (P leg, near +x_hard)
  c_bot(x) = 1 − exp( − softplus_k( (x_hard + x)·Vu/U_T − V_off ) )      (N leg, near −x_hard)
  softplus_k(y) = ln(1 + e^{k y})/k,  k = 1;  V_off = 0   (V_off shifts the knee by a Vdsat)
  I_into_node(x_n) = Isat·[ (1 + tanh u)/2 · c_top(x_n)·(1 + λv·(x_hard − x_n)·Vu)
                          − (1 − tanh u)/2 · c_bot(x_n)·(1 + λv·(x_hard + x_n)·Vu) ]
  ```
  `λv` is the channel-length modulation per volt (0.15 /V **[ASSUMPTION]**).
- **Per leg:** the destination leg uses `x_n = x_d` with sign +; the source leg uses `x_n = x_s` with sign −. The currents at the two ends are no longer equal and opposite: the supply makes up the difference. **Charge is conserved only when both legs are in compliance.** This is the physical behaviour.
- **Code change:**
  1. Add `forward_legs(x_src, x_dst, ...)` to `FreeTanhLibrary` returning `(I_pos, I_neg)` pre-compliance (`Isat·(1 ± tanh u)/2`).
  2. In `rhs` apply `c_top/c_bot` per node when scattering (`differential_stage.py:1508-1514`). The current `index_add_` pair becomes two scatters with different multipliers.
  3. Set `clip_term = 0` (`:1768`) and skip `gate_c` (`cell_library.py:399-401`) when the flag is on.
  4. Clamp the interstage transfer output to `[−x_hard, x_hard]` (`stage_transfer.py`).
- **`freeze_read` interaction:** the frozen quantity is `u` (and so `I_±`). `c_top/c_bot` depend on the evolving `x_n`. They must be applied **live** at every Heun substep on top of the frozen `I_±`, otherwise the rail has no effect for 10 steps.
- **Verilog-A:** `I(d) <+ Ip*(1-exp(-vtop/$vt)) - In*(1-exp(-vbot/$vt))` where `vtop = V(vdd) − V(d) − voff`, floored with `max(…, 0)` (or a smoothed `limexp`). Transistor: PMOS and NMOS cascode-less mirrors; add cascodes only if `λ` is too high, which costs 0.1 V of headroom.
- **Effect:**
  - Nodes cannot leave `±x_hard` (±0.4 V) because the leg that would push outward turns off and the opposite leg restores.
  - At rest (`tanh u = 0`) the two legs give `−Isat·λv·Vu·(x_n − 0)` : an automatic **output conductance of λv·Vu·Isat to Vcm** (0.0105·Isat per unit at λv = 0.15 /V, Vu = 0.07 V). This is §3.5's `g_out` for free.
  - Inside `|x| < ~3.5`, `c ≈ 1` to within 0.3 % (`exp(−(5.714−3.5)·0.07/0.0259)` = `exp(−6.0)` = 0.0025), so the trained solution is changed only at the excursions (stage 3: 165/256 rows reach > 4, `trained_ctle_audit.md:34`). **This is the largest single behaviour change in the list; retrain with it from an early stage, not as a last fix-up.**
- Flags: `hw_compliance` (replaces both the soft clip and the input gate), `hw_clamp_transfer`.

### 3.5 Gradient floor 0.01·u → g_out = λ·Isat

- **Current code:** `I_tanh = Isat·(tanh(u) + ALPHA·u)`, `ALPHA = 0.01` (`cell_library.py:260`, `:358`). The class docstring states the same form (`cell_library.py:231`). Value at `|u| = 2`: `0.01·2 = 0.02` against `tanh 2 = 0.96`.
- **Change (two parts):**
  1. Forward: drop the term (`ALPHA = 0`). Finite output conductance is modelled by `g_out`: when `hw_compliance` is on it is automatic (§3.4). When it is off, add `−g_out·Isat·(x_n)·1` per leg: `g_out = λv·Vu = 0.0105` per unit of Isat at Vu = 0.07 V, connecting node `n` to Vcm. This is a conductance of `0.0105·Isat` units (e.g. Isat = 2 → 0.021 units = 14 nS).
  2. Backward: optional straight-through floor, so saturated edges still receive gradient:
     ```python
     t = torch.tanh(u)
     t_ste = t + ALPHA_BWD * (u - u.detach())      # forward value = tanh(u); d/du = 1 − tanh² + ALPHA_BWD
     ```
- **Mapping:** `g_out` is the channel-length modulation of the output mirror (`gds = λ·I_D`), seen by Spectre automatically. In Verilog-A it is an explicit `I(d,vcm) <+ gout*Isat*V(d,vcm)`.
- **Effect:** removes a fictitious linear branch that gives each OTA a hidden extra gain of `0.01·gm·A·Isat`. For trained edges (`gm·A ≈ 0.1`, Isat ≈ 0.12) that was `1.2e-4` units, so numerically negligible; its role was gradient flow in the saturated regime, which the straight-through surrogate keeps (the trained model has `|u| > 2` on 0 % of samples, `trained_ctle_audit.md:289`, so the loss of gradient is rarely visible).
- Flags: `hw_no_alpha` (forward), `hw_alpha_bwd` (float, default 0.0).

### 3.6 Vref, and `s` as a switch bit

**Vref**

- **Current code:** `Vref = σ(raw_vref_sense)·x_max`, one scalar per stage, range [0, x_max], init `raw_vref_init = 0` → 2.0 units (`differential_stage.py:1263`, `:1703`; `config.py:330-333`). In hardware 2.0 units = Vcm + 0.14 V = 0.54 V, which is not the natural reference.
- **Change:** `Vref = x_ref_max · tanh(raw_vref)`, range [−x_max, x_max], optional DAC quantisation to `n_bits` (e.g. 6: LSB = 2·4/63 = 0.127 units = 8.9 mV at Vu = 0.07). Straight-through rounding. Init: keep `raw_vref_init` unchanged so Vref = 0 (= Vcm) in the new parametrisation (not identical to the old 2.0; flag-gated).
- **Mapping:** one voltage DAC (resistor ladder from VDD) per stage, buffered, shared by all 22 sense OTAs: stage-multiplexed, 7 codes.
- **Effect:** the sense currents acquire a symmetric offset range. The trained Vref of the 7-stage model is not quoted in Phase 3b (the trial_0011 checkpoint does exist, §1.0); if the optimum was near the old init (2.0) the new range covers it.
- Flag: `hw_vref_symmetric`, `hw_vref_bits`.

**`s` as a switch bit**

- **Current code:** `s = sign(s_raw)`, straight-through (`cell_library.py:346-347`); it already is binary in the forward pass. It multiplies the differential part only: `u = gm·(s·(A·Vs − B·Vd) + θ)` (`:357`).
- **Change:** none in the forward. Document and enforce: (a) s is a per-edge, per-stage **stored bit** reloaded at each stage; (b) θ is not affected by s. Observed: only 7-12 % of core edges (stages 1-3) and 18-20 % of readout edges have s = +1 (`trained_ctle_audit.md:258`). So −1 is the common state and the hardware default should be the swapped connection (no switch in the common path). Setting `s_raw` init to negative with `hw_s_default_neg` is an optional change of init.
- **Mapping:** a 1-bit transmission-gate swap of the two pair inputs. It adds `Ron·C_gate` to the input path (negligible: Ron ~ 5 kΩ × 2 fF = 10 ps = 1.3e-4 Tu) and ~1 fF of junction cap per input (§3.8). Alternative: swap the output legs, which is current-mode and adds no input cap but needs 2× mirrors.
- **θ:** an input-referred offset `gm_eff·θ` with |θ| up to 0.73 units (0.051 V at Vu = 0.07; `trained_ctle_audit.md:253`). Realise it as a current-imbalance trim: `I_off = Isat·tanh(gm_eff·θ)`; 4-bit signed DAC per edge per stage. Constraint `|θ| ≤ θ_max = 1` via `θ = θ_max·tanh(theta_raw)` is optional (`hw_theta_bound`).

### 3.7 Signed crossbar: offset-binary vs differential columns (single-ended)

- **Current code:** `readout_crossbar_W` is `[7×22]`, signed, init N(0, 1/√22) ≈ 0.213 (`differential_stage.py:688`, `:1698`; apply at `:1735-1743`, `circuit_model_report.md:160`). 154 weights per stage.
- **Single-ended implications:** the 22 sense currents `i_k` are push-pull (signed, as in §3.4). Accumulator `i` receives `Σ_k W_ik·i_k`. Three options:

| Option | Realisation | Cost per cross-point | Resolution |
|---|---|---|---|
| (A) sign-magnitude | mirror ratio `|W|` + 1 polarity bit selecting the P or N leg | 1 mirror (binary-weighted, n bits) + 1 swap | n-bit magnitude + sign |
| (B) differential columns | `W = W⁺ − W⁻`, two unipolar mirrors (one to each leg) | 2 mirrors | n-bit each |
| (C) offset-binary | `W_ik = m_ik − m_0`, `m_ik ∈ [0, 2m_0]` unipolar; the shared subtraction `m_0·Σ_k i_k` is one extra mirror column per accumulator (not per cross-point) | 1 mirror | n+1 bits, zero at mid-code |

  Recommendation: **(C) offset-binary** with a replica column per accumulator. It halves the mirror count against (B), and a mid-code zero is the natural reset. It needs the replica column matched to the cross-point mirrors, otherwise a weight of 0 leaks `σ(Δm_0)·Σ i_k` (a systematic error).
- **Constraint:** `W = W_max·tanh(W_raw/W_max)`, `W_max = 4` (mirror ratio ≤ 4 at nA currents without cascodes) **[ASSUMPTION]**; optional n-bit uniform quantisation with straight-through rounding (`hw_xbar_bits`, e.g. 5). At N(0, 0.213) the init sits at 5 % of W_max, so the bound does not matter at init.
- **Effect:** the paper model's trained W is not quoted in Phase 3b (the trial_0011 checkpoint does exist, §1.0). The cap prevents arbitrarily large readout gains; the accumulators are the 7 outputs and see the full `Σ|W|·|i_k|` ≤ `22·4·Isat_sense`, which is large against `x_hard`. The compliance of §3.4 will then dominate the readout.
- Flags: `hw_xbar_bound`, `hw_xbar_bits`, `hw_xbar_mode` ∈ {"signed" (default, current), "offset_binary", "differential"}. The last two only change the quantisation and the mirror count, not the maths (they are equivalent to sign-magnitude in the ideal case); they matter for mismatch (§3.9) and for the parasitic count (§3.8).

### 3.8 Fixed C plus parasitic C_node

- **Current code:** `C_eff = 1` scalar, divides every node (`differential_stage.py:313`, `:1779`). No parasitics (`feature_surgery_audit.md:212`).
- **Change:** a per-node vector `c_node[j] = (C_fixed + C_par[j]) / C_u` (≥ 1 unit when `C_u` is the target).
  ```
  C_par[j] = C_wire + Σ_{edge terminals at j} C_t
  C_t = C_gate (input pin; includes the divider)  or  C_leg (output mirror drain)
  ```
  **[ASSUMPTION] 22FDX numbers:** `C_gate` = 1.5-2 fF for the pair input plus its divider and swap switch (the divider capacitors are the dominant part; at g ≈ 0.1 attenuation the divider adds ≈ 5-10 fF, which pushes this to the top of the range); `C_leg` = 0.5-1 fF per mirror drain with its junction and wire; `C_wire` = 2-5 fF per node.
- **Terminal count (from the paper config, `circuit_model_report.md:13`):**
  - Hidden node (average): 22 core edges × 2 ends / 11 nodes = 4 core terminals, each with an input gate and an output leg; 8 boundary edges / 11 = 0.73 boundary terminals (gate + leg, the source is an ideal terminal); 22 sense edges / 11 = 2 sense gates (the sense output leg lands on an accumulator). Total 6.7 gates + 4.7 legs.
  - Accumulator node: receives 22 crossbar legs with option (A)/(C), 44 with option (B).
  - Hidden: with `C_gate = 2`, `C_leg = 1`, `C_wire = 4` fF, `C_par = 6.7·2 + 4.7·1 + 4 = 22 fF` (44 % of 50 fF).
  - Accumulator: `22·1 + 4 = 26 fF` (52 %), or 48 fF (96 %) with differential columns.
  - So the drawn `C_fixed` is about 28 fF (hidden) and 24 fF (accumulator, offset-binary), and only 2 fF with differential columns, which is not buildable at 50 fF. Differential columns require a larger `C_u` on the accumulator nodes.
- **Training:** `c_node` is a registered buffer built from the graph (`topology.py` edge lists) and `HWUnits`; `rhs` divides by `c_node` (a `[1,N]` tensor) instead of the scalar at `differential_stage.py:1779`. It is non-trainable (§2.2 mentions C is chosen from area, not searched).
- **Heun dt:** the stiffest node now is the hidden node with the smallest `c_node`.
- **Effect:** with `"physical"` the node capacitances differ by node (22 fF vs 26 fF of parasitic, plus whatever fixed C is drawn), so hidden and accumulator time constants shift relative to today's `C_eff = 1`. With `"uniform"` the fixed C is padded so that `c_node = 1` everywhere, which keeps today's dynamics. `"uniform"` pads `C_fixed = C_u − C_par` per node and errors if that is negative.
- Flag: `hw_c_node` ∈ {None (current), "uniform", "physical"}.

### 3.9 Power penalty (optional) and mismatch injection (optional)

**Power**

- **Current code:** `LAMBDAS["power"] = 1e-6` counts gm, not current (`config.py:390`; `feature_surgery_audit.md:213`).
- **Formula:** with the time-multiplexed fabric the physical OTA slot `e` carries stage `s`'s Isat only during stage `s`:
  ```
  P_avg = κ · VDD · Iu · (1/S) Σ_s Σ_e Isat_{s,e} · σ(z_{s,e}) · g_vca,e      [W]
  P_peak = κ · VDD · Iu · max_s Σ_e Isat_{s,e} · σ(z) · g_vca
  L_pow = λ_P · P_avg / P_ref
  ```
  `κ ≈ 3` = tail + both mirror legs + bias overhead **[ASSUMPTION]**. `S = 7`, 52 slots. Plain `Σ Isat` (without κ·VDD·Iu) is equivalent up to scale.
- **Numbers (Iu = 46.7 nA):** all 52 OTAs at 2 units: `κ·0.8·46.7e-9·104 = 11.7 µW`; energy 4.1 pJ per 350 ns inference. All at the ceiling 16: 93 µW (32 pJ). The paper checkpoint is far below both: its Σ Isat is 18.5-30.3 per stage over the 52 OTAs (core + boundary + sense, without the σ(z)·g_vca folding), i.e. 2.1-3.4 µW per stage and about 1 pJ per 350 ns inference summed over the 7 stages — the same figure Phase 3b §1.5 quotes.
- **Effect:** biases Isat down toward the floor 0.6 units, balancing against accuracy. Choose `λ_P` so the penalty is ≈ 1 % of the task loss at init (needs a measurement).
- Flag: `hw_power_lambda` (0.0 = off).

**Mismatch**

- **Current code:** `VARIATION` dict (`config.py:652-665`: gm 0.05, Isat 0.03 log-std) feeds `SimContext`, which no stage consumes (`circuit_model_report.md:183`). Dead.
- **Change:** multiplicative log-normal noise on the **physical slot**, shared across the 7 stages (one fabric):
  ```
  Isat'  = Isat · exp(σ_I · ε_I[e])     σ_I = 0.05
  G'     = G    · exp(σ_G · ε_G[e])     σ_G = 0.10
  A'     = A    · exp(σ_A · ε_A[e])     σ_A = 0.01   (B likewise, independent ε_B)
  θ'     = θ + σ_θ · ε_θ[e]             σ_θ = 0.1 units (σ_Vos ≈ 7 mV at Vu = 0.07)
  ε ~ N(0,1), shape [E] (core+boundary+sense = 52), redrawn per batch, frozen across stages
  ```
  **[ASSUMPTION]** Pelgrom A_VT ≈ 1.5 mV·µm; a pair of 0.5 × 0.2 µm devices gives `σ_Vos = √2·1.5/√0.1 = 6.7 mV` = 0.095 units. A weak-inversion single device mirrors with `σ_I/I = σ_VT/(nU_T)` = 4.7/(1.3·25.9) = 14 % at 0.1 µm² and 4.5 % at 1 µm². Pick σ_I = 5 % for mirrors sized ≥ 1 µm² and say so in the design notes.
- **Note:** σ_θ = 0.1 units is of the order of the trained median |θ| (0.1-0.2 units, `trained_ctle_audit.md:253`); the retrained θ will be partly noise.
- **Effect:** costs a few % accuracy when on. Evaluate with ≥ 30 samples of ε. Never apply at eval unless `hw_mismatch_eval = True`.
- Flags: `hw_mismatch_sigma = dict(isat=…, g=…, a=…, b=…, theta=…)` (None = off), `hw_mismatch_eval`.

### 3.10 VCA and GLN as bias generators (both KEPT)

**VCA** (`differential_stage.py:1059-1125`; `g = 2σ((u·W)·v_eᵀ)` in (0, 2), r = 3, DC per sample; multiplies the tanh current (core), tanh + shunt (boundary, sense))

- **Mapping — a current-mode multiplier on the tail current.** `g·Isat = 2·Isat·σ(ζ_e)` with `ζ_e = (u·W)·v_eᵀ`. In weak inversion a differential pair steers its tail current with `σ(ΔV/(nU_T))`, so a pair fed with `ΔV = nU_T·ζ_e` and a tail of `2·Isat` delivers `2·Isat·σ(ζ_e)` on one branch. That branch is mirrored into the OTA's tail. The pair input `ΔV_e` is `nU_T·Σ_r bus_r·v_{e,r}`: three bus voltages (r = 3) formed once per sample by a 4→3 current-mode mixer, and a 3-term weighted sum per edge (switched-capacitor or resistive).
- **It multiplies Isat, not the slope.** In weak inversion that matches the model (`g_vca` multiplies the output current, not `gm`). Good.
- **Range / constraint:** hardware range of the tail current is [0, I_tail,max]. Effective `Isat_eff = Isat·g_vca·σ(z)` must satisfy `Isat_eff ≤ Isat_hw_max = 16` and `≥ 0.6 · (small value)` when active; clamp `Isat_eff` (`hw_isat_eff_clamp`). With `g_vca ≤ 2`, base `Isat ≤ 8` keeps the clamp inactive; set `isat_max = 8` when VCA is on.
- **Precision:** `ζ_e` bits: 0.1 in ζ = 10 % in gain; 5-6 bit DAC resolution on the bus values. `hw_vca_bits`.
- **Trained importance:** forcing all VCA gains to 1 raises failure from 9.0 % to 99.3 % (`trained_ctle_audit.md:31`), so VCA precision directly drives accuracy.

**GLN** (`gln_rails.py:167-168`; `gm_e = clamp(gm0_e·exp(z·Wᵀ), gm_min, gm_max)`; 4 tanh rails shared across stages; boundary and sense families only)

- **Mismatch between the model and a tail-current DAC.** The GLN modulates the **tanh slope** gm. A tail-current DAC (the natural implementation of an `exp` control: `I = I_0·exp(ΔV_gs/(nU_T))`) changes **Isat** and `gm_ss` together and leaves the slope unchanged in weak inversion. The two are not equal. Without a change the GLN cannot be built as a tail DAC.
- **Options:**
  1. **`hw_gln_mode = "tail"` (recommended):** GLN multiplies Isat instead of gm: `Isat_e' = Isat_e·exp(clamp(z·Wᵀ, −δ_max, δ_max))`, slope fixed at `s_max·g`. The small-signal gain `Isat·slope` still scales by `exp(δ)`, so for the *small-signal* part (trained model is near-linear: median |u| < 1) it is the same function; only the saturation level changes with it. Retrain needed (`gln_rails.py:167-168` plus the consumers at `differential_stage.py:1559-1569`, `:1709-1719`).
  2. `"slope"` (current): GLN switches a divider/degeneration bank (`g_e·exp(δ)`) — more switches, more parasitic cap on the input node (§3.8).
- **Realisation of "tail":** per edge a 6-bit exponential (log-step) current DAC. A mixing step `δ = Σ_b W_eb·z_b` is a weighted sum of 4 rail voltages; `δ_max = ln 4 = 1.39` (±47 mV at the tail gate: `nU_T·δ`) covers a 16:1 range. The 4 rails `z_b = tanh(α_b(a_b·u + c_b))` are an analog function of the 4 inputs, evaluated once per sample; 4 analog computations of one tanh each. `P@Q` factorised mix (E×2 · 2×4) is the same DAC code set per edge (static, one table).
- **Constraint:** `δ_max` bound replaces `gm_min/gm_max` in the clamp (`gln_rails.py:168`); it must be applied after the exp.
- **Time-multiplexing cost:** the parameter inventory lists the VCA taps `vca_v_boundary` 24, `vca_v_readout` 66, `vca_v_core` 66 and `vca_W` 12 outside the "per stage (×7)" group (`circuit_model_report.md:72`), so they look shared across stages, as the GLN `P`, `Q` are (`gln_rails.py:32`). I did not verify this in the stage code. If it holds, the VCA and GLN tables are static for the whole inference and only the per-edge device parameters (Isat, g, Â/B̂, s, θ, G, σ(z)) are reloaded per stage. If not, 7× as many VCA taps must be reloaded every 50 ns (open question 4).
- Flags: `hw_gln_mode` ∈ {None, "tail", "slope"}, `hw_gln_delta_max`, `hw_isat_eff_clamp`.

### 3.11 Flag summary (all default off)

| Flag | Default | Section | Replaces / touches |
|---|---|---|---|
| `hw_ratio_ab` | False | 3.1 | `cell_library.py:344-345` |
| `hw_slope_bound` | False | 3.2 | `cell_library.py:351-352`, `:357` |
| `hw_ic_coupling` | False | 3.3 | `cell_library.py:351-355` |
| `hw_compliance` | False | 3.4 | `cell_library.py:399-401`, `differential_stage.py:1376-1386`, `:1508-1514`, `:1768` |
| `hw_clamp_transfer` | False | 3.4 | `stage_transfer.py:98-102` |
| `hw_no_alpha`, `hw_alpha_bwd` | False, 0.0 | 3.5 | `cell_library.py:260`, `:358` |
| `hw_vref_symmetric`, `hw_vref_bits` | False, None | 3.6 | `differential_stage.py:1263`, `:1703` |
| `hw_theta_bound` | False | 3.6 | `cell_library.py:356-357` |
| `hw_xbar_bound`, `hw_xbar_bits`, `hw_xbar_mode` | False, None, "signed" | 3.7 | `differential_stage.py:688`, `:1698` |
| `hw_c_node` | None | 3.8 | `differential_stage.py:313`, `:1779` |
| `hw_power_lambda` | 0.0 | 3.9 | `config.py:390` |
| `hw_mismatch_sigma`, `hw_mismatch_eval` | None, False | 3.9 | replaces dead `SimContext` path |
| `hw_vca_bits`, `hw_isat_eff_clamp` | None, False | 3.10 | `differential_stage.py:1059-1125` |
| `hw_gln_mode`, `hw_gln_delta_max` | None, ln 4 | 3.10 | `gln_rails.py:167-168` |
| `hw_check_stiffness` | False | 3.3 | diagnostics only |

Suggested enable order (each step gets its own retrain and a regression check against the previous one): units/`HWUnits` only (no model change) → 3.1+3.2 → 3.4 → 3.5 → 3.8 → 3.10 (tail) → 3.7 → 3.6 → 3.9.

---

## 4. Open questions for the user

1. **Vu choice.** Is the real usable swing ±0.28 V (Vu = 0.07, x_max = 4) acceptable, or do you prefer Vu = 0.1 with x_max ≈ 2.5-3? The second forces a retrain from scratch (the trained model uses up to 5.35 units); the first changes only the unit layer. Also: which input device flavour (LVT/RVT, NMOS or complementary), since the input range and the 0.35 V input floor depend on it.
2. **Slope range.** Trained gm·A is 0.065-0.135 median against a hardware maximum of 0.87 (about 10:1 attenuation needed), and the trained Isat·gm·A (0.007-0.018) is 2-5× below the §2.4 floor of 0.033. Do you accept (a) capacitor dividers, (b) source degeneration, or (c) a wider `g_min` (1/50)? Each costs a different parasitic or mismatch penalty.
3. **`freeze_read`.** The trained model depends on it (failure 9.0 % → 26.7 % when off, `trained_ctle_audit.md:30`), and with a physical OTA the input is the live node. Are you willing to retrain with `freeze_read = False`, or will the fabric sample-and-hold the node at each stage entry (a S&H per node, 18 nodes)?
4. **Time-multiplexing reload.** Per stage the reload set is 52 OTAs × (Isat, g, Â/B̂, s, θ, G, σ(z)) plus VCA taps, in under ~5 ns of the 50 ns stage. Is a digital register file plus current DACs (≈ 52 × 7 × ~30 bits) the intended design, or are 7 physical copies on the table for part of the network? (`feature_surgery_audit.md:217`.)
5. **Single-ended vs differential.** The note above assumes single-ended with a Vcm-referenced mirror. Doubling the nodes (differential) would give twice the swing (±0.5 V diff) and the cancelled common-mode term `(A−B)`, at 2× cap and 2× OTA count. Is single-ended final?
6. **Leak and shunt devices.** The leak (32 nS ≈ 31 MΩ) and the trained shunts (1-17 nS) need active conductors. Is an active (tail-biased weak-inversion) leak OK, and is a per-edge programmable G required (it is a per-edge tuning knob: set to 0 → failure 9.0 % → 69.6 %, `trained_ctle_audit.md:32`)?
7. **Crossbar sign scheme** (§3.7 A/B/C) and its bit width; and whether the replica column is acceptable.
8. **Which model to retrain.** A trained 7-stage, 11-hidden paper checkpoint now exists on ssr0 (trial_0011, §1.0), but it is outside the repo checkout, and every trained statistic quoted in this document is still from `fixed_mlp_knet` (4 stages, 14 hidden, k = 4). Do you want to re-quote the paper checkpoint's full parameter statistics (gm, Isat, A, B, gm·A, θ, G, σ(z), per stage) and rebuild §1.4 from it, or keep the `fixed_mlp_knet` tables as the constraint source? The paper-checkpoint Isat and gm·A numbers quoted in §2.3 do not change the feasibility conclusions.
9. **Mismatch and PVT.** Which σ values (Pelgrom, mirror sizes) and which corners (temperature 27 °C vs 85 °C for the slope, supply ±10 %) should the retrain target?
10. **Power and area targets.** A rough P_avg budget (the 52-OTA example is 12 µW at 2 units per edge) and a total C area budget decide `C_u`, `Iu` and the floor in §2.4. Vu/Tu follow from the 350 ns budget.
11. **I/O.** The log10, min-max and sigmoid-then-10ˣ steps stay off-chip (`feature_surgery_audit.md:216`). Confirm, and confirm the 4 input voltages stay within ±0.91 units ≈ ±64 mV, which is small against the ±0.28 V swing and costs input SNR.
