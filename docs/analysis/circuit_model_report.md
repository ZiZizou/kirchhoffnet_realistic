# HR-KNet circuit-model spec sheet (paper → code), for Verilog-A / Spectre / 22FDX

Scope: read-only analysis. Repo: `C:\Users\Atharva\Documents\kirchhoffnet_realistic`. Paper: `tmp/analysis/paper_raw.txt` (read in full).
All `file:line` refs are to the files in this repo as found today. "not found" means I searched and did not find it.
Everything marked **[RUN]** I verified by instantiating the code (read-only, no files written, bytecode disabled).

---------------------------------------------------------------------------------------------------

## 0. Key findings (read this first)

1. **The model is fully dimensionless.** No SI units anywhere in the dynamics. `C_eff = 1`, time is in arbitrary "time units", voltages in arbitrary "volt units" with rail `x_max = 4.0`, currents/conductances in arbitrary units. The code itself says so: `config.py:11-16` ("NOT calibrated to a consistent timescale … No pretence is made that ω = g/C is a real analog time constant"), and paper §IV-A ("No quantity is tied to SI units … C_eff, t_span, gm, conductances, and Isat are all continuous"). The comments "μS-scale / μA-scale" at `config.py:25-27` apply to the legacy cell library, not to the `tanh_free` library used for CTLE.
2. **The paper's CTLE model (5,649 params, Table II) is reproduced exactly [RUN]** by this configuration (taken from `spice_failure_check.py:138-203` defaults, whose comment at `:137-139` says it is verified against `knet_model_files_manual_trial_0011`; I rebuilt it from `topology.build_net_from_config` and got `TOTAL 5649`):
   - 7 stages, 11 hidden nodes/stage + 7 accumulator nodes/stage (18 states), Watts–Strogatz k=2, p=0.2, seed=1, `edge_repeats=2` → 22 core OTA edges/stage
   - cell library `tanh_free`, `gm_max=4.24197`, `isat_max=28.68855` (gm_min = isat_min = 0.01)
   - `x_max=4.0`, `C_eff=1`, fixed leak 0.0486, `clip_current=0.05`, learnable clip sharpness (init 0.02)
   - `freeze_read=True` (core frozen), boundary & sense **live**
   - VCA rank 3 (core on, separate core bus, no bias, no shunt gating), GLN B=4 rank 2 on `boundary,readout`
   - readout `shared-x2` (2 sense OTAs / hidden node → 22 senses, crossbar W[7×22])
   - total horizon `t_span = 4.660583` (0.6658/stage), 10 Heun steps/stage (70 total, dt = 0.06658)
   - interstage `residual-relu-tanh`, 4 boundary terminals each fanned to 2 hidden nodes: `{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}`
   - **No trained checkpoint of this model exists in the repo** (only `outputs/kn_bayes_opt/friedman2_knet_e1/trial_0000/model.pt` = Friedman-2, and a PlainMLP teacher). Learned per-edge values are therefore not available; only init values and bounds are.
3. **Paper vs code mismatches that matter for Verilog-A** (details in §3):
   - θ is *inside* the gm product in code: `u = gm·(s·(A·Vsrc − B·Vdst) + θ)` (`cell_library.py:357`); paper Eq.(3) writes `gm·s·(…) + θ`.
   - The shunt is also multiplied by σ(z) (`differential_stage.py:1527-1529`); paper Eq.(5) leaves `G·(Vsrc−Vdst)` outside σ(z).
   - A and B are independent `softplus` values with **no upper bound and no A+B=1** (`cell_library.py:344-345`), so the effective differential slope is `gm·A`, unbounded above.
   - The compliance gate is `σ((x_max−|V|)/0.02)` on **both** terminals, tanh branch only (`cell_library.py:399-401`); transition width 0.02 units = 0.5 % of the rail.
   - The soft rail clip is only 0.05 current units vs OTA currents up to ~29, so nodes are **not** actually confined to ±x_max. Logged runs show max|x|/x_max = 4.85–6.43 in late stages (`dagger_distillation_trial_0002_recreation_vca_gate_shunt.out:801-808, 1003-1010`).
4. **Non-physical/hard items:** per-stage frozen core currents (sample-and-hold of the whole state at each stage entry); an instantaneous learned per-node nonlinear interstage re-write of node voltages (W1·x + W2·tanh x + W3·ReLU(x−Vth)); stage-specific parameters on the same nodes (time-multiplexed reprogramming or 7 physical copies); signed dense crossbar (7×22); signed polarity `s=±1` via `sign()`; tanh-with-0.01-leak (gradient-floor term) in the I–V; digital log10/min-max input scaling and sigmoid→10^x output decoding; weight-less "dead" params (u_logits). See §6.
5. **Variation/noise code is effectively dead for this model:** `SimContext` is never consumed by `differential_stage.py`/`cell_library.py`/`topology.py` (grep for `ctx`: no hit in those files; `kirchhoff_net.py` only mentions it in a docstring, `:93`), `kirchhoff_noise.py:172-177` calls `base(h, ctx=…, tau=…)` which `KirchhoffNetWithIO.forward` (`kirchhoff_net.py:555-565`) does not accept, `analog_noise.py` wraps MLPs. Only `stage_noise_std` in `KirchhoffNet.forward` (`kirchhoff_net.py:163-172`) is live and defaults to 0. Paper §II-F-c / §V also say mismatch/noise/PVT are future work.

---------------------------------------------------------------------------------------------------

## 1. Global numerical conventions

| Item | Value | Source |
|---|---|---|
| Voltage unit | arbitrary; rail `x_max` | `config.py:303-318` (`PHYS["x_max"]=3.0` default); CTLE uses **4.0** (`dagger-nuance-distillation-kirchhoffnet.py:632`, `kn_bayes_opt.py:260,1387`, `spice_failure_check.py:175`); BO range [0.5, 8.0] (`kn_bayes_opt.py:830-835`) |
| Capacitance | `C_eff = 1.0` for every node, fixed, not trainable | `config.py:305`, `differential_stage.py:313`; `rhs` divides by `c_eff` at `:1779` |
| Time | arbitrary; per-stage `t_span`, `num_steps` | `config.py:531-535` (global 7.0 / 70); CTLE total span BO'd in [3,7] (`kn_bayes_opt.py:1375`), per-stage = total / num_stages (`dagger-nuance…py:1842-1843`, `spice_failure_check.py:926-928`) |
| Solver | **fixed-step Heun** (RK2 predictor–corrector), explicit, no adaptivity, backprop through unrolled steps | `differential_stage.py:1906-1920` (`x_pred = x + dt·k1; x = x + 0.5·dt·(k1+k2)`); also `_heun_steps` `:2097-2158`. DEQ option exists (`config.py:544-554`) but is not used for CTLE (`solver="heun"`, `dagger-nuance…py:1955`) |
| dt (paper CTLE model) | 4.660583/7/10 = 0.06658 | [RUN] `net.core.stage_times`, `stage_steps` |
| Leak | `g_leak = 0.0486` per node (`softplus(-3)`), fixed (`leak_mode="non-programmable"`) | `config.py:641`, `differential_stage.py:395,933` |
| Soft rail | `I_clip = 0.05·(σ((V−x_max)/s) − σ((−V−x_max)/s))`, `s` learnable per stage in [1e-3, 0.2], init 0.02 | `differential_stage.py:1361-1386`; `config.py:306-317`; paper Eq.(2) matches |
| Initial state | all nodes (hidden **and** accumulators) = 0 in boundary mode | `kirchhoff_net.py:593-616` |
| Precision | float32 | defaults |

Implied time constants (still arbitrary units): leak-only τ = C/g = 20.6; per-edge small-signal conductance of a tanh edge ≈ `Isat·gm·A·σ(z)·gate` (up to ~120·A at the BO bounds); shunt `G = softplus(g_raw)` init 0.0067. Total horizon 4.66 ≪ 20.6, so the leak barely acts within one inference.

Free scale factors if you map to SI (the ODE is scale-free): `V_phys = Vu·V`, `I_phys = Iu·I`, `t_phys = Tu·t` ⇒ `C_phys = C_eff·Iu·Tu/Vu`, `G_phys = G·Iu/Vu`, slope inside tanh `gm_phys = gm/Vu`. Only three scales are free; see §7 for what 22FDX forces.

---------------------------------------------------------------------------------------------------

## 2. Reference architecture (what actually runs for the CTLE result)

Data path (wrapper `LocalKirchhoffStudentWrapper`/`KirchhoffNetStudent`, `dagger-nuance-distillation-kirchhoffnet.py:~1700-1990`; `spice_failure_check.py:833-1060` is an equivalent rebuild):

```
specs (power, jitter, height, width)  [digital]
  -> u_i = clamp( 2*(log10(spec_i)-lo_i)/(hi_i-lo_i) - 1 , -4, 4 )          scale_input, :1916-1937   [digital/analog log]
  -> KirchhoffNetWithIO(u)  (kirchhoff_net.py:555-665):
       x0 = 0 (18 nodes)                                                    :593-616
       for stage s in 0..6:                                                 KirchhoffNet.forward :145-174
           Heun-integrate 10 steps of  C dx/dt = KCL(x_frozen, u, x) ...    differential_stage.rhs :1388-1779
           x <- StageTransfer_s(x)      (not after last stage)              :173-174
       x_read = x[:, 11:18]  (7 accumulators, final-time sample)            :456-459, :645-654
       y = OutputAffine(x_read)         gain*x+bias+tg*tanh x+rg*relu(x-thr)  io_mapper.py:605-616
       y = post_readout_transfer(y)     w1*y+w2*tanh y+w3*relu(y-vth)       kirchhoff_net.py:494-499, :662
  -> p_i = 10**( lo_i + (hi_i-lo_i)*sigmoid(y_i) )                          predict, :1958-1966  [digital]
```
Output bounds (`ctle_dagger_common.py:89-97`): fW [1e-6, 10], current [5e-4, 2.5], ind [1e-12, 3.0], Rd [10, 1500], Cs [1e-15, 1e-9], Rs [10, 1500], VDD [0.6, 1.2] (log-uniform bounds; units not stated in code, Rd/Rs read as Ω, Cs as F, VDD as V). Input log bounds are data-derived with 5 % pad (`kn_input_log_pad_frac`, logged at `dagger_distillation_trial_0002…out:47-50`).

Parameter budget **[RUN]** (5,649 total): per stage (×7) `cell_lib` 7×22, `boundary_cell_lib` 7×8, `readout_sense_cell_lib` 7×22 (a,b,s,gm,isat,θ,g_res raw), `z_logits` 22, `boundary_z_logits` 8, `readout_sense_z_logits` 22, `readout_crossbar_W` 7×22 = 154, `raw_vref_sense` 1, `clip_sharpness_raw` 1, `vca_W` 12, `vca_W_core` 12, `vca_v_boundary` 24, `vca_v_readout` 66, `vca_v_core` 66, **`u_logits` 18 (dead, deprecated)**; plus `core.transfers` 432 (6 transfers × 18 nodes × 4), `output_mapper` 35, `post_readout_transfer` 28, `gln_rails` 100.

---------------------------------------------------------------------------------------------------

## 3. Component-by-component spec

Notation: `Vs`, `Vd` = source / destination node voltage (state `x`), `u` = boundary terminal voltage, `σ` = logistic, `sp` = softplus. All formulas are what the code computes. "Trainable" = `nn.Parameter` with `requires_grad`.

### C1. Node (capacitor + leak + soft rail clip)
- **Paper:** §II-C Eq.(1)–(2): `C_eff dV/dt = Σin − Σout − V/R_leak − I_clip(V)`.
- **Code:** `DifferentialStage.rhs` tail, `differential_stage.py:1761-1779`; `soft_clip` `:1376-1386`; leak `:392-395, 881-933`.
- **Math:** `dx_j/dt = ( acc_j + i_drive_j − g_leak·x_j − I_clip(x_j) ) / C_eff`. Leak and clip read the raw state `x` (not any bounded broadcast).
- **Params:** `C_eff=1` fixed (`:313`); `g_leak=0.0486` fixed (`INIT["leak_constant"]`, `config.py:641`); `clip_current=0.05` fixed (`config.py:307`); clip sharpness `s`: trainable scalar per stage, `s = 1e-3 + (0.2−1e-3)·σ(clip_sharpness_raw)`, init 0.02 (`differential_stage.py:341-349,1369-1374`); `x_max=4`.
- **Applies to:** all 18 nodes per stage (11 hidden + 7 accumulators). Accumulators get the same leak/clip (paper §II-D-c says so).
- **Verilog-A notes:** capacitor + conductance + two sigmoid current sources (clip is a pair of steep sigmoids: ~±0.05·unit current once |V|>x_max, width 0.02 units). Because the clip is weak, expect |V| ≫ x_max in practice (§0 item 3). Nodes need a common-mode: model is symmetric about 0 with ±x_max; on a single 0.8 V supply this means Vcm = VDD/2 offsets (not in code).

### C2. Core edge = differential OTA (tanh) ∥ linear shunt, two-terminal conservative
- **Paper:** §II-D-a Eq.(3)–(5), Fig. 2(c,d).
- **Code:** `FreeTanhLibrary` `cell_library.py:224-435` (selected by `kn_cell_library="tanh_free"`, `make_cell_library` `:641-652`; `CELL_LIBRARIES["tanh_free"]` has `BIAS_ENABLED=True`, `config.py:261`); assembled in `DifferentialStage.rhs` `differential_stage.py:1490-1545`, frozen variant `_compute_i_edge_const` `:1293-1346`.
- **Math (per directed edge e from src to dst):**
  ```
  A = sp(a_raw) ; B = sp(b_raw)                         (independent, A,B>0, no upper bound)   cell_library.py:344-345
  s = sign(s_raw)  (±1 forward, straight-through grad)                                       :346-347
  gm   = gm_min + (gm_max-gm_min)*σ(gm_raw)   in [0.01, gm_max]                               :352
  Isat = isat_min + (isat_max-isat_min)*σ(isat_raw), clamp_min 1e-6                            :354-355
  u    = gm * ( s*(A*Vs - B*Vd) + θ )                                                          :357   (θ = theta_raw, init 0)
  I_tanh = Isat*( tanh(u) + 0.01*u )                                                           :358, ALPHA=0.01 at :260
  gate_c = σ((x_max-|Vs|)/0.02) * σ((x_max-|Vd|)/0.02)     (compliance, tanh branch only)      :399-401
  I_res  = softplus(g_res_raw)*(Vs - Vd)                   (shunt, no compliance gate)         :413-414
  I_edge = [ gate_c*I_tanh ] * σ(z_e) * g_vca,e(core)  +  I_res * σ(z_e)                       differential_stage.py:1496-1502, 1527-1545
  ```
  KCL scatter: `acc[dst] += I_edge; acc[src] −= I_edge` (`:1512-1514`; `read_only_source=False`), i.e. a conservative two-terminal device, positive current flows src→dst. With `s=−1` the device pushes current the other way (polarity swap, Fig. 2(d) "Swap polarity").
  - `g_vca,e(core)` only multiplies the tanh branch unless `--vca-gate-shunt` (False in the paper config, `:1534-1535`).
  - **freeze_read=True (paper CTLE):** `I_tanh·gate_c·σ(z)·g_vca` is evaluated **once per stage** from the state at stage entry (after the interstage transfer) and held constant for all 10 Heun steps (`:1884-1886`, `:1522`); only the shunt `I_res·σ(z)`, leak, clip, boundary and sense currents see the evolving state. In stage 1 the entry state is 0, so the frozen core current reduces to the θ-only constant `Isat(tanh(gm·s·θ)+…)`.
- **Params per edge (all trainable):** `a_raw`(init 0 → A=0.693), `b_raw`(0 → 0.693), `s_raw`(randn·0.1 → random ±1), `gm_raw`(−5 → σ=0.0067 → gm≈0.038 for gm_max 4.24), `isat_raw`(−5 → Isat≈0.20 for isat_max 28.7), `theta_raw`(0), `g_resistive_raw`(−5 → G=0.0067), `z_logits`(0 → σ=0.5). Init at `cell_library.py:292-313`, `differential_stage.py:434`. Fixed: gm_min=isat_min=0.01 (`config.py:270-273`), gm_max/isat_max set per trial (BO log-uniform [1, 50], `kn_bayes_opt.py:836-849`; paper config 4.24197 / 28.68855).
- **Units:** `gm` here is the **tanh slope in 1/(volt unit)**, not a transconductance; the small-signal transconductance is `Isat·gm·A` (current-unit/volt-unit). `Isat` is the max OTA output current in current units.
- **Trainable vs searched:** per-edge values trainable; `gm_max, isat_max` (and `x_max`, topology, `t_span`) searched by Optuna; for CTLE `x_max` fixed 4.0.
- **Hard/non-physical:** (i) `0.01·u` linear leak on the tanh (gradient floor) — physically an extra 1 % linear conductance; (ii) A, B unbounded and independent (real pair: ratio from W_A/W_B, common-mode gain `A−B≠0` is not a diff pair); (iii) `s=±1` hard sign; (iv) θ is an input-referred offset scaled by gm; (v) the shunt gated by σ(z); (vi) compliance gate is a 2-input product of steep sigmoids (not a real circuit); (vii) frozen-current (S&H) semantics; (viii) the paper Fig. 2(d) OTA has I+ and I− outputs, matching the code's "drain src / source dst".

### C3. Structural edge gate σ(z)
- **Paper:** §II-D-b ("Structural gates σ(z_e) (init 0.5) scale every edge current").
- **Code:** `z_logits` `[E]`, `differential_stage.py:434`; applied `:1480,1496,1529`. Same for `boundary_z_logits` (`:500-502`, `:1554,1574`) and `readout_sense_z_logits` (`:685-687`, `:1699,1724`).
- **Math:** `gate = σ(z)`, init z=0 → 0.5. Degree-budget competition (`:935-1044`) is **off** (`budget_enabled=False`, `:451`; only enabled by `set_budget_frac` from the older training schedule). Pruning code exists (`topology.py:710-1295`) but is not part of the CTLE path.
- **Verilog-A:** a trainable analog multiplier/trim per edge, or fold into Isat at export (σ(z) is a static per-edge scalar once trained → merge into `Isat_eff = Isat·σ(z)` and `G_eff = G·σ(z)`). Exact only for the shunt and tanh when the VCA gate is 1.

### C4. VCA gate (input-conditioned edge current scale)
- **Paper:** §II-F-a Eq.(7)–(8): `g_vca = 2σ(inmix·v_e + b_e)`, `inmix = W·u`, W zero-init → g=1.
- **Code:** `_compute_vca_gate` `differential_stage.py:1059-1087`; core `_compute_core_gate` `:1089-1125`; params `vca_W [4,r]`(boundary+readout), `vca_W_core [4,r]` (separate core bus), `vca_v_{boundary,readout,core} [E,r]`, biases off; init `:765-796`.
- **Math:** `g = 2σ( (u @ W) @ v_e^T [+ b_e] )` ∈ (0,2), `r=3`. `u` = the 4 normalized spec voltages, constant over the whole inference (per sample), so each gate is evaluated **once per sample** (`:1864`, folded into `i_edge_const` for core). Gate multiplies: core → tanh branch only; boundary and readout/sense → **tanh + shunt together** (`:1574-1578`, `:1724-1728`).
- **Params:** `vca_W` zero-init, `vca_v` ~N(0,0.01) (`config.py:603-608`); rank searched 1–8 (paper config 3). `vca_use_hidden=False` (gate does not depend on hidden state).
- **Verilog-A:** a (4→r) static linear mix of the input terminals → r control buses; per-edge tap `v_e` → 2σ → multiplies the OTA output current (current-mode multiplier / tail-current scaling). Because `u` is DC for the evaluation, it can be a static trim computed once (a table of 4 inputs); implement as a Verilog-A `analog function`.

### C5. GLN rails (input-trimmed gm)
- **Paper:** §II-F-b Eq.(9)–(10).
- **Code:** `gln_rails.py:74-175` (`GLNRails`, `_EdgeMix` `:51-71`); applied in `differential_stage.py:1559-1569` (boundary), `:1709-1719` (sense), frozen variants `:1153-1165, 1265-1276`; built at `topology.py:2511-2555`.
- **Math:** `z_b = tanh(α_b·(a_b·u + c_b))`, `α=sp(alpha_raw)+1e-6`, B=4 rails shared by the whole net (tied across stages); per-family mix `W = P@Q`, `P[E,2]~N(0,0.01)`, `Q[2,4]=0`; `gm_e = clamp(gm0_e·exp(z @ W^T), 0.01, gm_max)`. Only the `boundary` (8 edges) and `readout` (22 senses) families are gated; core edges and shunts are never touched (v1). Same `P,Q` used in all 7 stages (tied).
- **Params:** 100 trainable (`gln_rails.py:191-203` formula). Note: `exp(·)` log-domain multiplier → a bias-current-controlled gm; it is `exp` of a linear combination of tanh outputs.
- **Verilog-A:** straightforward to express as an `analog function` of the 4 input voltages (static, DC). Hardware: B=4 tanh "rails" + per-edge mixing + exponential bias control (exp is natural for a sub-threshold current mirror).

### C6. Compliance (rail) gate
- **Paper:** §II-D-a ("rail-compliance gate shuts off only the OTA branch near the rails, leaving the shunt active").
- **Code:** `cell_library.py:399-401` (`beta_softness=0.02`, `config.py:306`). Applied to the tanh branch of **every** library call (core, boundary, sense).
- **Math:** `gate_c = σ((x_max−|Vs|)/0.02)·σ((x_max−|Vd|)/0.02)`. For boundary edges `Vs=u` (so `|u|` is also compliance-checked, but |u|≲1 ≪ 4); for sense edges `Vd=Vref`.
- **Consequence:** when a node exceeds the rail by >0.04 units its OTA currents (both as source and destination) turn off; only shunt + leak + 0.05 clip remain, so nodes can stay beyond the rail (observed 4.85–6.43×).

### C7. Input injection ("boundary OTAs") — no input DAC/sampling switch in the model
- **Paper:** §II-D-c; Fig. 3 (Vin1/Vin2 inject).
- **Code:** boundary list built `topology.py:2191-2288` (flat `(input i, hidden j)` pairs; 4 inputs × 2 targets = **8 boundary OTAs** per stage); stage ctor `differential_stage.py:467-513`; RHS `:1547-1609`; input net `NullInputMapper` (`io_mapper.py:278-293`, used at `topology.py:2437`).
- **How inputs are injected:** the 4 normalized spec voltages `u` are **ideal, constant voltage terminals held for the entire evolve phase, in every stage** (`boundary_first_stage_only=False` default `topology.py:1975`, so all 7 stages have their own 8 boundary OTAs; this is not an "initial-condition write" — nodes start at 0). Not sampled-and-held, not time-varying, no DAC.
- **Math (per boundary edge):**
  ```
  I_b = [ gate_c(u,Vd)*Isat*(tanh(gm_g*(s*(A*u - B*Vd)+θ)) + 0.01*u') + G*(u - Vd) ] * σ(zb) * g_vca,b(u)
  acc[Vd_node] += I_b            (one-sided: ideal terminal never drained)       differential_stage.py:1549-1588
  ```
  with `gm_g = clamp(gm0·exp(z@W_b^T), 0.01, gm_max)` when GLN on. Live (not frozen) because `freeze_boundary=False`.
- **Params:** `boundary_cell_lib` (8 edges × 7 raw params), `boundary_z_logits` 8, VCA/GLN taps per above, per stage.
- **Input range:** `u ∈ ±~1` nominal (min-max in log10 domain with 5 % pad) clamped ±4 (`:1925-1937`); i.e. ≈ ¼ of the rail.
- **Digital/analog preprocessing not in the paper's "no converter" claim:** `log10`, min–max scaling to [−1,1], clamp. A hardware realization needs analog log-amp or off-chip DAC with log-coded values.

### C8. Shared-sense readout: sense OTAs, Vref, crossbar, accumulators
- **Paper:** §II-D-d Eq.(6), Fig. 4(b).
- **Code:** stage ctor `differential_stage.py:608-710`; builder `topology.py:2396-2411`; sense jitter for `s=2` (`gm_raw[1::2] += 0.5`) `topology.py:1606-1611`; frozen `_compute_frozen_sense` `:1233-1291`; RHS `:1688-1759`.
- **Structure:** one sense OTA per (hidden node, k∈{0,1}) = 22 senses reading `x_j` vs a **per-stage learnable scalar Vref**; sense currents are mixed by a signed dense crossbar `W[7×22]` and injected into the 7 accumulator nodes (indices 11..17). Source nodes are never drained.
- **Math:**
  ```
  Vref   = σ(raw_vref_sense) * x_max     in [0, x_max]  (init 2.0 for x_max=4)     :1263, 1703
  i_k    = [ gate_c(x_j, Vref)*Isat_k*(tanh(gm_k*(s*(A*x_j-B*Vref)+θ)) + 0.01*·) + G_k*(x_j - Vref) ] * σ(z_k) * g_vca,r(u)
  gm_k   = clamp(gm0_k*exp(z@W_r^T), 0.01, gm_max)      (GLN readout family)
  acc[11+i] += Σ_k W[i,k]*i_k            (W signed, init N(0,1/√22))                :1735-1743
  ```
  Live (not frozen: `freeze_temporal_read=False`).
- **Accumulator dynamics:** the standard node ODE (C1) with this KCL — a leaky integrator over the whole evolve horizon; **readout = voltage of the accumulator at t_end of the last stage** (not an integral of a separate signal, not a time-average), passed through the interstage transfers at each stage boundary (the accumulators are among the 18 states transformed by `StageTransfer`, see C9).
- **Verilog-A:** crossbar = linear VCCS array; a resistive realization needs differential columns for signed `W` (paper says so). Vref in [0,x_max] ≠ 0 means the reference is not the symmetric zero of the rail.

### C9. Interstage transfer (instantaneous re-write of node voltages)
- **Paper:** §II-D-e.
- **Code:** `stage_transfer.py:34-192` (`residual-relu-tanh`, `:98-102,161-169`); built `topology.py:2639-2642`; used `kirchhoff_net.py:173-174`.
- **Math:** `x' = w1⊙x + w2⊙tanh(x) + w3⊙ReLU(x − vth)` per node (w1=1, w2=w3=vth=0 at init). Applied to **all 18 nodes including the 7 accumulators** (no `drive_mask` in CTLE: `enable_drive=False`; the paper's "driven nodes pass through unchanged" is moot). 6 transfers × 18 × 4 = 432 trainable params.
- **Verilog-A / circuit:** needs a sample step, an analog function block per node (gain, tanh, ReLU), and a reset/re-initialization of each node capacitor to the transformed value at each stage boundary. In Verilog-A this can be done with `idt(expr, ic, assert)` (resettable integrator) or event-driven `@(timer)` updates; in transistors it is the hardest block (S&H + nonlinear buffer per node, 6 times, with ~96 ns-level parameter changes if time-multiplexed).

### C10. Output stage and decode
- **Code:** `OutputAffine` `io_mapper.py:577-616` (gain, bias, tanh_gain, relu_gain, relu_threshold per output → 35 params; init 1,0,0,0,0); then `post_readout_transfer` (`kirchhoff_net.py:494-499`, 28 params, same residual-relu-tanh form); then `sigmoid` and log-bound mapping (`dagger-nuance…py:1958-1975`).
- **Digital post-processing:** sigmoid + `10**` decode to physical CTLE parameters (7 values) is in software, not in the analog netlist.

### C11. Time/solver structure ("freeze read" and stages)
- Stage loop `kirchhoff_net.py:145-174`; per-stage Heun `differential_stage.py:1844-1925`. `core_refresh_interval=0` (no periodic refresh, `:1885`). For Verilog-A: (a) 7 sequential stages of length 0.6658 time units; (b) at each stage entry sample `V_node` into S&H (`Vsh`), compute core tanh currents from `Vsh` only (those are the OTA inputs for that stage), (c) apply C9, (d) reset node capacitors.

### C12. Present in code but **not used** for the CTLE paper model
- Persistent drive (`drive_current` `differential_stage.py:1046-1057`, `enable_drive=False`), reference edges (`enable_ref_edges=False`), node activation `x_max·tanh(x/x_max)` (`:1348-1359`, NARMA-only, requires `allow_experimental_cells`), dynamic leak (`:406-420`), `TapRails` (`tap_rails.py:23-62`, attached only by the NARMA probe: `differential_stage.py:363`), skip-linear (`kirchhoff_net.py:523-538`), DEQ solver, degree budget, `SimpleEdgeLibrary/RealisticTanh*/LinearOTA/AntiParallel` libraries (`cell_library.py:51-591`), legacy L/S/P/Z cell tables (`config.py:29-251`). `ctle_kirchhoff_bo.py` is a **different, adjoint/dopri5 bare-repo model** with its own bounds table (`:41-43` ≠ `ctle_dagger_common.py:89-97`); not the paper's HR-KNet.

### C13. Noise / variation / quantization (as found)
- `sim_context.py:29-177`: `SimContext` (global gm/Isat log shifts 0.05/0.02, per-edge mismatch std 0.05/0.03 from `config.py:652-665`), not consumed by any stage code → no effect.
- `kirchhoff_noise.py:80-190`, `analog_noise.py:67-105`: ADC/DAC quantization (4 or 6 bit) + Gaussian state noise; incompatible call signature with the current `KirchhoffNetWithIO.forward`; the CTLE scripts do not use it.
- Live: `KirchhoffNet.forward(stage_noise_std=0.0)` additive Gaussian on stage-end states (`kirchhoff_net.py:163-172`).
- Not found: thermal noise model for OTAs/resistors, kT/C noise, mismatch model tied to transistor geometry.

---------------------------------------------------------------------------------------------------

## 4. Topology and export

**Generator:** `topology.small_world_graph` (`topology.py:208-307`): ring lattice with `k/2` neighbours each side → undirected pairs `(min,max)`; each pair rewired with probability p by replacing the high endpoint with a random distinct node, using `random.Random(seed)`; emitted as one directed edge per pair, direction **low index → high index** (`src.append(a); dst.append(b)`, `:294-299`). Then `repeat_edges(topo, edge_repeats=2)` (`:378-434`) duplicates each hidden edge (independent parameters, same direction, currents add in KCL) ⇒ "two devices per graph edge" (paper). Edge ordering comes from iterating a Python `set` of int tuples (deterministic in CPython but implementation-dependent), so use the stored buffers, not the seed, as the source of truth.

**Seeded:** yes, `small_world_seed=1` (`dagger-nuance…py:617`; paper "seed 1"). The same graph is built for every stage (all stages share `src/dst`: [RUN] `same src all stages? True`).

**Stored in checkpoint:** `src`, `dst`, `boundary_src`, `boundary_dst`, `readout_sense_src`, `output_ode_*` are `register_buffer` (persistent) in every `DifferentialStage` (`differential_stage.py:372-373, 493-498, 680-683`) and are saved in `student.state_dict()` (`dagger-nuance…py:4123`). Hyper-parameters (stages, k, p, x_max, gm_max, isat_max, t_span, readout, VCA rank, GLN settings) are **not** in the state dict; they come from the run command (`run_train_knet_single_shot_distillation.sh:69-77`) or the `spice_failure_check.py` defaults.

**Counts, paper CTLE model [RUN]:** per stage 18 nodes (11 hidden `0..10`, 7 accumulators `11..17`), 22 core edges (11 undirected pairs ×2), 8 boundary OTAs, 22 sense OTAs, crossbar 7×22. Whole net: 7 stages × (22 + 8 + 22) = 364 OTA instances, 126 nodes, 7 per-stage Vref, 7 per-stage rail sharpness. Example [RUN] stage-0 buffers: `src=[1,1,0,0,3,3,0,0,7,7,2,2,6,6,4,4,8,8,5,5,7,7]`, `dst=[2,2,4,4,4,4,10,10,9,9,3,3,7,7,5,5,9,9,6,6,8,8]` (pairs 1-2, 0-4, 3-4, 0-10, 7-9, 2-3, 6-7, 4-5, 8-9, 5-6, 7-8), `boundary_src=[0,0,1,1,2,2,3,3]`, `boundary_dst=[0,4,1,5,2,6,3,7]`. For comparison (my replica of the generator, same seed/p): n=25,k=4 → 50 pairs (paper "≈50 graph edges → 100 core edges"); n=14,k=4 → 28; n=12,k=4 → 24.

**Export recipe (untested sketch; no files written by me):**
1. Rebuild the net with the exact hyper-parameters (see §0 item 2) via `topology.build_net_from_config` and `load_state_dict`.
2. Per stage `s`: nodes 0..17; core edges `(src[e],dst[e])` with `A=sp(a_raw)`, `B=sp(b_raw)`, `s=sign(s_raw)`, `gm=0.01+(gm_max−0.01)σ(gm_raw)`, `Isat=0.01+(isat_max−0.01)σ(isat_raw)`, `θ`, `G=sp(g_res_raw)`, `σ(z)`, `g_vca,e = 2σ(((u@vca_W_core)@vca_v_core^T)[e])` (computed once for given u). Boundary edges same with `(boundary_src→terminal index, boundary_dst)`; sense edges with `(readout_sense_src, Vref_s)` and crossbar `W`.
3. Global: leak 0.0486, `C_eff=1`, `x_max`, `clip_current=0.05`, `s_clip`, `gln_rails.{a,c,alpha_raw}`, `gln_rails.families.{boundary,readout}.{P,Q}` (tied across stages), 6 `core.transfers.*.{residual_w1,w2,w3,vth}`, `output_mapper.{gain,bias,tanh_gain,relu_gain,relu_threshold}`, `post_readout_transfer.*`.
4. Since inputs are DC, **precompute per-input** the VCA/GLN scalars (they depend only on `u`) and emit static per-edge parameters; only the 7 core-edge S&H states and node ODEs are dynamic.

---------------------------------------------------------------------------------------------------

## 5. How "realistic" device constraints are currently set

| Constraint | Current setting | Source | Comment |
|---|---|---|---|
| gm bounds | gm ∈ [0.01, gm_max], gm_max BO'd in [1,50] log-uniform; paper CTLE 4.24197; alt run 1.35 | `config.py:270-271`, `kn_bayes_opt.py:836-842`, `run_train_knet_single_shot_distillation.sh:71` | "gm" = tanh slope in 1/volt-unit; real small-signal gm = Isat·gm·A (A unbounded) |
| Isat bounds | [0.01, isat_max], BO'd [1,50]; paper 28.69 | `config.py:272-273`, `kn_bayes_opt.py:843-849` | current units arbitrary |
| Output resistance / shunt | per-edge `G=softplus(raw)`, init 0.0067, unbounded above, **no OTA r_out** other than this and the global leak | `cell_library.py:313,413` | "Rout" of Fig.2(c) = 1/G; the paper says R_leak but the code has fixed g_leak=0.0486 |
| Capacitance | 1 for all nodes (no parasitic/charge-sharing model) | `config.py:305` | paper Fig. 2(b) "charge-sharing parasitics" is not modeled; `ARCHITECTURE.md:51` confirms "omits common-mode dynamics, non-diagonal capacitance, sample-and-hold circuitry, and post-layout parasitics" |
| Rails | ±x_max = ±4 (soft clip 0.05, compliance 0.02), BO [0.5, 8] | above | bipolar, centered on 0; not tied to VDD |
| Vdd | **not found** in the KNet model. (`VDD` appears only as one of the 7 *predicted CTLE parameters*, bounds 0.6–1.2 V, `ctle_dagger_common.py:96`) | — | no supply/headroom/compliance in volts |
| Input range | ~±1 (≤ ±4 clamp) | `dagger-nuance…py:630,1937` | |
| Temperature | `temp_c` deprecated/ignored | `sim_context.py:10-14` | |
| Quantization of device levels | none ("continuous" per paper) | paper §IV-A | |

**Consistency with 22 nm (sub-1 V core): not consistent as-is.**
- A ±4-unit symmetric rail needs a physical mapping; the only physically sensible mapping to a 0.8 V core supply is `Vcm=0.4 V, ±0.4 V swing` ⇒ `Vu = 0.1 V/unit` (or `x_max` retrained smaller with larger `Vu`). The model has no common-mode, headroom, or body-bias concept.
- Slope limit: a differential pair's tanh slope is at most `1/(2·n·Vt)` ≈ 15–19 /V (n≈1–1.3, weak inversion, Vt=25.9 mV). With `Vu=0.1 V/unit` the max physical `gm·A` is ≈ **1.5–1.9 per unit**; the CTLE config uses `gm_max=4.24` (and A up to unbounded) ⇒ requires `Vu ≳ 0.22 V/unit` or source degeneration/attenuation (A<1). The 1.35 run (`…single_shot_distillation.sh:71`) is in the physical range.
- Currents: Isat ≤ 28.7 units; for `Iu = 1 µA` that's ≤ ~29 µA/edge ×(22+8+22) edges/stage ⇒ ~1.5 mA/stage worst case (static power would need to be budgeted; the code has no power model besides a `power` regularizer weight, `config.py:390`, and a count of OTAs).
- Time-scale example (proposal, not in code): `Vu=0.1 V, Iu=1 µA, Tu=1 ns` ⇒ `C_eff=1 → 10 fF`, `G=1 → 10 µS`, `g_leak 0.0486 → 0.49 µS (R ≈ 2 MΩ)`, horizon 4.66 ns, dt 66 ps. Node time constants spanning ~10 ps…20 ns result from the learned `Isat·gm·A` spread; this must be re-checked after retraining with explicit bounds.

---------------------------------------------------------------------------------------------------

## 6. Non-physical or hard-to-realize items (checklist)

| # | Item | Where | Severity for Spectre |
|---|---|---|---|
| 1 | Dimensionless units; Isat·gm·A coupling; A,B unbounded | `cell_library.py:344-358` | must re-parameterize with bounds |
| 2 | Signed polarity `s=±1` from `sign()` | `:346-347` | needs swap switches/cross-coupling per edge (or fix s at export) |
| 3 | Extra `0.01·u` linear term | `:358` | small; implement as 1 % parallel linear gm |
| 4 | θ scaled by gm (offset inside tanh) | `:357` | needs input-referred offset DAC |
| 5 | Core current frozen at stage entry (S&H of all 18 states × 7 stages) | `differential_stage.py:1884-1886, 1522` | needs 18 S&H per stage or equivalent; fidelity-critical |
| 6 | Interstage learned nonlinear re-write (W1,W2,W3,Vth per node) incl. accumulators, 6 times | `stage_transfer.py:161-169` | hardest block; requires per-node nonlinear buffer + capacitor reset |
| 7 | Stage-specific parameters on shared nodes (or 7 physical copies) | per-stage `ModuleList` | time-multiplexed bias reprogramming or 7× hardware |
| 8 | Signed dense crossbar 7×22 | `:688-690, 1735` | differential columns / mixed-sign resistive weights |
| 9 | VCA gate (2σ), GLN rails (tanh + exp) | `:1059-1125`, `gln_rails.py` | analog multipliers; static given DC inputs → can be precomputed at export |
| 10 | Compliance gate as product of steep sigmoids; weak clip (0.05) allows 6× rail excursions | `cell_library.py:399-401`, `config.py:307` | no real circuit does this; actual rail clamp/saturation behaviour must be defined |
| 11 | σ(z) edge gate on tanh and shunt | `:1480,1496,1529` | merge into Isat/G at export |
| 12 | Boundary shunt `G(u−Vd)` loads the input terminal (resistor to the source) | `:1567-1569` | affects input source drive; terminals are ideal in the model |
| 13 | Vref per stage learnable ∈[0,x_max] | `:691-693` | per-stage reference DACs |
| 14 | Digital pre/post-processing: log10, min–max, clamp; output sigmoid + log-bound decode + 10^x | `dagger-nuance…py:1916-1975` | not "converter-less" |
| 15 | Dead params (`u_logits` 126, deprecated) and 0.01 gm_min floor | `differential_stage.py:440`, `config.py:270` | cosmetic |
| 16 | No mismatch/noise/PVT/thermal; noise wrapper broken | §3-C13 | Monte-Carlo must be new work |
| 17 | Single-ended-looking state with negative voltages | everywhere | needs Vcm or true differential nodes (ARCHITECTURE.md:51 says differential-ness/common-mode not modeled) |

---------------------------------------------------------------------------------------------------

## 7. Suggested Verilog-A component list (phase 1, behavioral)

1. `knet_node` — capacitor (C), leak conductance, soft clip (I_clip), optional integrator reset (`idt(·,ic,assert)`).
2. `knet_ota_diff` — two-terminal tanh OTA with params `(A,B,s,gm,Isat,theta,G,sigz,gvca,gate_c enable)` and I± outputs; implements `I = sigz·gvca·gate_c·Isat(tanh(u)+0.01u) + sigz·G(Vs−Vd)`.
3. `knet_ota_bnd` — input OTA with ideal terminal (one-sided output), params identical.
4. `knet_sense` + `knet_xbar` — one-sided sense OTA against Vref, VCCS crossbar to accumulator nodes.
5. `knet_sh` — S&H (sample at stage start, hold until next).
6. `knet_xfer` — analog function `w1·x + w2·tanh x + w3·relu(x−vth)` + capacitor re-write.
7. `knet_vca`, `knet_gln` — `analog function`s of the DC input terminals, evaluated once.
8. `knet_affine` — OutputAffine + post transfer (or leave in software).
9. Clock/sequencer: `@(timer)` stage boundaries at multiples of `t_span/7`.
Replace in order of risk: OTA → node/C → S&H/transfer → crossbar/readout → VCA/GLN.

---------------------------------------------------------------------------------------------------

## 8. Open questions for the user

1. Is the reference to reproduce the **7-stage, 11-node, k=2 manual_trial_0011-like model (5,649 params)**, and where is its trained checkpoint? None is in the repo, so per-edge learned values are unavailable.
2. Stages: 7 physical copies of the fabric, or one fabric re-programmed in time? This decides S&H/reset hardware and area.
3. Which Verilog-A fidelity do you want for the frozen-core (S&H) semantics, and the learned interstage transfer? (Exact replication vs retraining without freeze and with a hardware-friendly transfer.)
4. Voltage mapping: Vcm and `Vu` (e.g. 0.1 V/unit with ±0.4 V swing on 0.8 V core) and whether to retrain with `x_max`, `gm_max`, bounded A/B and a real rail clamp before building transistors?
5. Differential nodes vs single-ended with Vcm? (Code is symmetric about 0; no common-mode.)
6. Are input terminals driven by ideal sources in Spectre (log-scaled specs by a DAC/stimulus) and is the digital sigmoid/10^x decode acceptable outside the netlist?
7. Time scale: pick `Tu`/`Iu` (suggest ~1 ns, 1 µA) or keep `C` as a free retraining knob (ODE is scale-free)?
8. Should signed polarity `s`, θ offsets and the `0.01·u` term be kept, or fixed/removed at retraining?
9. Is the 22FDX transistor-level target the OTA only first (with ideal capacitors/S&H), and which flavor (SLVT/LVT, thick-oxide 1.8 V for the input stage)?
10. The CTLE itself (fW, ind, Rd, Cs, Rs, VDD 0.6–1.2 V): which PDK/node was used for the paper's Cadence runs (`spice_failure_check.py:1377` shows `design_wrapper.lib.scs tt_pre`)? Is it 22FDX?
