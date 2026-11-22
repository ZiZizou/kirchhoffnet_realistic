# HR-KNet pre-surgery feature audit (read-only)

Repo `C:\Users\Atharva\Documents\kirchhoffnet_realistic`, branch `simplified`, HEAD `e11e2b0`. Only tracked change: `ctle_kirchhoff_bo.py` (not an HR-KNet module). No existing file was modified. This report is the only file I created (plus `_smoke_*.txt`/`_train_*.txt` scratch logs in this folder).
Background: `circuit_model_report.md` (CM) and `pipeline_report.md` (PR). I only repeat their content where needed. All `file:line` refs are to today's files.

Rating key (Verilog-A first / 22FDX transistor): **Easy**, **Moderate**, **Hard**, **Not physical** (no circuit does this as written; it must be redefined).

---------------------------------------------------------------------------------------------------

## 0. Headline findings

1. **The code base is one big flag matrix, not a model.** `differential_stage.py` (2609 lines) and `topology.py` (2873) carry every experiment that was ever run (legacy L/S/P/Z cell tables, DEQ, degree budget, pruning, drive, ref edges, node activation, dynamic leak, NARMA sequences, GLN, VCA). A crude keyword count (gln|vca|freeze|budget|deq|node_activation|dynamic_leak|refresh|drive|ref_edge|prune|moe|skip_linear|ctx|mismatch|read_only_source|ota_mesh|legacy|experimental) hits 540 lines in `differential_stage.py` (21 %), 375 in `topology.py` (13 %), 132 in `kirchhoff_net.py` (16 %). `cell_library.py` (666 lines) has 7 classes, but only `FreeTanhLibrary` (`cell_library.py:224-435`) is used by the CTLE paper model and by every NARMA/Friedman HR-KNet run I found.
2. **The CTLE model and the Friedman KNet runs share one core recipe** (hard-coded in `kn_bayes_opt.py:481-529` `_build_*_command`): `tanh_free`, `--leak non-programmable`, `--freeze-read`, `--vca --vca-core --vca-separate-core-bus`, `--interstage-activation residual-relu-tanh`, `--solver heun`, Watts-Strogatz graph, `edge_repeats=2`. CTLE adds readout `shared-x2`, GLN (B=4, rank 2, boundary+readout), learnable clip sharpness (CM §0.2). Friedman adds `--target-noise-std 1.0` and, in the one run I can see (`outputs/kn_bayes_opt/friedman2_knet_e1/trial_0000/config_snapshot.txt`), `readout=temporal` default (`kn_bayes_opt.py:465`), no GLN, `C_eff=9.06`, `x_max=6.49`, `gm_max=21.2`, `isat_max=12.2` (BO also searches `C_eff`; the paper CTLE fixes `C_eff=1`).
3. **MoE does not exist in the KNet.** `kn_bayes_opt.py:187-194, 576-581, 1348-1371` states the `--kn-moe-*`/`moe_*` arguments are "inert schema labels" ignored by `parse_known_args`. All real MoE code is in the teacher/baseline side (`ctle_dagger_common.py`, `spice_failure_check.py` `MoeStudent`, `train_ctle.py`, `generative-distillation-*`). Safe to ignore in the KNet surgery; clean the BO command builder only.
4. **Tests are mostly stale.** `test_smoke.py` (8421 lines, 246 tests): 158 pass / 88 fail under pytest (§3). The only suite that exercises the paper-config features (GLN, VCA, learnable clip, shared-sense, freeze flags, interstage) is `test_f1_f2.py` (15 tests, all pass, 4 s). `test_training.py` 39/40 pass. A regression harness for the surgery must be built on `test_f1_f2.py`-style fixtures, not on `test_smoke.py`.
5. **The noise/variation stack is dead for HR-KNet** (CM §0.5): `SimContext` is consumed by no `differential_stage/cell_library/topology` code (grep `\bctx\b|SimContext|edge_mismatch` in those three files: 0 hits), and `KirchhoffNetWithIO.forward` rejects `ctx` (11 smoke failures: "unexpected keyword argument 'ctx'").

---------------------------------------------------------------------------------------------------

## 1. Feature inventory

Columns: where / default / who uses it / paper-CTLE (CTLE) / Friedman KNet (FR) / hardware rating. "BO" = `kn_bayes_opt.py` command builder. "TS" = `train_script.py`. "DAG" = `dagger-nuance-distillation-kirchhoffnet.py` + `dagger-mlp-…` + `fixed-mlp-…`. "NARMA" = `narma_*.py`, `probe_*`, `smoke_audit3.py`, `knet_gating_audit1.py`.

### 1.1 Edge device (cell library) — `cell_library.py`

| Feature | What / where | Default | Used by | CTLE / FR | HW rating and why |
|---|---|---|---|---|---|
| `FreeTanhLibrary` (`tanh_free`) | `I = Isat·(tanh(u)+0.01u)`, `u = gm·(s·(A·Vs − B·Vd)+θ)`; `cell_library.py:224-435`, formula `:344-358`; registered `config.py:261` | selected via `--cell-library`/`kn_cell_library` | BO, TS, DAG, spice_failure_check, all NARMA | yes / yes | **Moderate/Moderate.** Diff pair + tail source is the textbook tanh. The issues are the parametrisation, not the shape (§4). |
| gm, Isat per edge, bounded sigmoid | `:317-325, 352-355`, bounds `config.py:270-273` (0.01..10 default), BO ranges `config.py:325-326` | gm_min=isat_min=0.01 | all | yes / yes | **Moderate/Moderate.** Isat = tail current (programmable via DAC/mirror). `gm` here is the tanh *slope in 1/V-unit*, not a transconductance: a diff pair's slope is fixed by 1/(2nVt) or by source degeneration, and cannot be set by the tail current in weak inversion. So a free per-edge `gm` needs a degeneration resistor DAC or an input attenuator. |
| Independent unbounded A, B (softplus) | `:344-345` | init 0 → 0.693 | all | yes / yes | **Not physical** as written (A, B unbounded, A−B ≠ 0 gives common-mode gain). Real: W_A/W_B ratio, bounded, and then differential only if A=B. |
| Signed polarity `s=sign(s_raw)` with STE | `:346-347, 357` | randn·0.1 | all | yes / yes | **Moderate/Hard.** Needs a per-edge polarity swap (cross-coupled switch pair on the OTA output) or two mirrored OTAs. Fixable at export (trained `s` is static), costs a 2:1 switch per edge. |
| θ offset (`bias_enabled`, `theta_raw`) | `:302-303, 356`; `config.py:261` `BIAS_ENABLED=True` | 0 | all `tanh_free` | yes / yes | **Hard.** θ is multiplied by gm (offset inside the tanh); needs an input-referred offset DAC per edge. Could be replaced by an unbalanced pair (A/B ratio) or a current-mirror offset into the output. |
| Leaky `0.01·u` (ALPHA) | `:260, 358` | 0.01 | all | yes / yes | **Not physical** as a gradient floor. Physically ~1 % extra linear conductance, which could be modelled as finite `r_out`/an explicit shunt. |
| Resistive shunt `G·(Vs−Vd)` | `:403-414`, init softplus(−5)=0.0067 | on | all | yes / yes | **Easy/Moderate.** MOS-resistor or poly R per edge; trainable G needs a programmable R. Not compliance-gated, bypasses `freeze_read`. |
| Compliance gate `σ((x_max−|V|)/0.02)` on both terminals | `:399-401`, `config.py:306` | always | all | yes / yes | **Not physical.** No circuit multiplies a current by a steep function of terminal voltage. Real OTAs have output-swing limits (saturation of the tail/mirror): model as soft output clamp, not as a gate on inputs. |
| `parallel_tanh_mult` branch | `:270, 289, 304-309, 378-397`; `config.py:261` `False` | off | none (`grep parallel_tanh` in tests = 0) | no / no | **Remove.** Dead experimental branch. |
| `SimpleEdgeLibrary` (relu/tanh), `RealisticTanhLibrary`, `RealisticTanhUpgradeLibrary`, `LinearOTALibrary`, `AntiParallelFreeTanhLibrary` | `cell_library.py:51-222, 438-591` | n/a | `knet_gating_audit1`, `probe_audit*`, `narma_linear_controls` (linear OTA), tests (35 smoke refs) | no / no | **Remove** (keep linear OTA only if a linear-control is needed). `SimpleEdgeLibrary` has no `num_cells` etc.; 32 smoke failures are from it. |
| Legacy L/S/P/Z/v15/v2 tables | `config.py:29-264` | n/a | tests, `ctle_dagger_common`, `train_ctle`, `train.py` | no / no | **Remove.** |

### 1.2 Node / stage dynamics — `differential_stage.py`, `config.py`

| Feature | Where | Default | Used by | CTLE / FR | HW rating |
|---|---|---|---|---|---|
| `C_eff` | `config.py:305`, `differential_stage.py:313,1779` | 1.0 (CTLE); BO searches it for Friedman (`c_eff` 9.06 in snapshot) | all | fixed 1 / searched | **Easy.** Pure scale; map to a real capacitor (the user wants it small). |
| Leak (programmable vs non-programmable) | `:286-288, 395, 881-933`; `config.py:641` (`0.0486`) | `programmable` in code, `non-programmable` in BO/DAG | all | non-prog / non-prog | **Easy.** Fixed conductance per node. `programmable` leak adds a per-node trained value: remove. |
| Dynamic leak (`dynamic_leak`, `dynamic_leak_input_proj`) | `:211-212, 295-296, 405-420, 889-930`; `config.py` preset flags `:1186,1301` | off | `knet_gating_audit1`, `narma_experiment` | no / no | **Not physical** (leak = softplus(a·x+b·u+c)). Remove. |
| Soft rail clip, fixed or learnable sharpness | `:341-349, 1361-1386`; `config.py:306-317` | fixed 0.02; learnable via `--learnable-clip-sharpness` | BO (`:522`), TS, DAG (`KN_LEARNABLE_CLIP_SHARPNESS=False` `:640`), `test_f1_f2` | CTLE: learnable / FR: optional | **Moderate/Moderate.** Clip current 0.05 vs OTA currents up to Isat≈29 means no real rail (nodes reach 4.85-6.43× x_max, CM §0.3). Learnable sharpness = per-stage trained slope of a clamp: not a tunable circuit parameter. Replace with a physical rail (ESD/diode-like or OTA output saturation). |
| `x_max` | `config.py:304` (3.0), CTLE 4.0, BO 0.5-8 | | all | yes / yes | **Easy** as a unit scale but must be tied to VDD/2 − headroom (§4). |
| Heun fixed-step solver | `:1906-1920, 2097-2158` | `heun` | all | yes / yes | **Easy.** Spectre will use its own integrator; only dt choice affects equivalence. |
| DEQ solver (`forward_equilibrium`, `deq_cfg`, `leak_floor`) | `:2188-2305`; `config.py:544-554` | off | TS `--solver deq` (`:2641`), 15 deq smoke tests (13 fail on missing `deq_solver` module) | no / no | **Not physical** (steady-state fixed point is a different computation). Remove. |
| Degree budget (`set_budget_frac`, `_compute_budget_gate`, `_budget_group_mask`) | `:844-1044`; `config.py:568-576` | off | TS `--budget*` (`:2828-2854`), `train_ctle`, 11 smoke tests | no / no | **Not physical.** Training-only soft top-k. Remove or move to a pruning script. |
| Edge gates σ(z) (`z_logits`) on tanh and shunt | `:434, 1480, 1496, 1529` | z=0 → 0.5 | all | yes / yes | **Easy at export** (fold into Isat/G: `Isat_eff=Isat·σ(z)`, `G_eff=G·σ(z)`; exact when VCA gate=1). In training it is a redundant second multiplier of Isat. |
| Node gate `u_logits` | `:430-440` | DEPRECATED, dead params (CM §2: 18/stage) | none | dead / dead | **Remove** (also `LAMBDAS["node_gate"]`, `PRUNE["node_threshold"]`). |
| `read_only_source` | `:213, 244, 1344, 1513, 1543` | `False` for core, one-sided for boundary/sense | all (boundary/sense implicit) | | **Easy.** Defines ideal-terminal one-sided OTAs. |
| `node_activation` (`none/tanh/identity`) | `:239, 275-280, 1150, 1212, 1257, 1323, 1357, 1461-1463, 1552`; `topology.py:2102-2115` (needs `allow_experimental_cells`) | `none` | `narma_experiment`, `narma_phase7_leak_bands`, `narma_spectral_norm`, `probe_audit_nodeact`, 4 `.sh` | no / no | **Moderate** (buffered tanh node), but experimental. Isolate or remove. |
| Persistent drive (`drive_current`, `enable_drive`, `drive_isat`) | `:385-390, 1046-1057`; `config.py:631-635`; `kirchhoff_net.py:359-417` | off | TS, `train_ctle`, 50 test refs | no / no | **Moderate**, unused. Remove. |
| Ref edges (`enable_ref_edges`, `ref_cell_lib`) | `:220-221, 522-526`; `config.py:332-334` | off | TS `:…`, 2 files, 6 smoke tests | no / no | **Moderate**, unused. Remove. |
| `output_ode_*` / `ota_mesh` temporal readout | `:222-224, 225-229`; `topology.py` builder; `kirchhoff_net.py:428-459` | `ota_mesh` | BO default `readout="temporal"` (`kn_bayes_opt.py:465`), DAG `KN_READOUT="temporal"` (`:624`) | no (CTLE uses `shared-x2`) / yes (`temporal`) | **Moderate/Hard.** h×d_out OTA mesh with per-edge gate+VCA. Pick one readout (§4). |
| `shared-sense` readout (`readout_mode="shared_sense"`, `senses_per_node ∈ {1,2}`, per-stage Vref, crossbar W[7×22]) | `:608-710, 1233-1291, 1688-1759`; `topology.py:1572-1611, 2396-2411`; TS `--readout shared|shared-x2` (`train_script.py:2742`) | off | BO/DAG/spice_failure_check/test_f1_f2 | yes / optional | **Moderate/Hard.** Sense OTAs are fine; signed crossbar needs differential columns; Vref per stage in [0,x_max] is a DAC. The `[1::2] += 0.5` gm jitter (`topology.py:1606-1611`) is a training trick, not hardware. |

### 1.3 Time / freezing semantics

| Feature | Where | Default | Used by | CTLE / FR | HW |
|---|---|---|---|---|---|
| `freeze_read` (core currents evaluated once at stage entry from the stage-entry state and held for all Heun steps) | `:214, 245, 1293-1346, 1884-1886`; flag `train_script.py:2868` | off in code; **on** in BO (`kn_bayes_opt.py:497`) and DAG (`KN_FREEZE_READ=True`, `:622`) | 29 files | yes / yes | **Hard / Hard.** Needs a sample-and-hold on every node feeding the core OTA input (one S&H per hidden/accumulator *node*, not per edge: 18 per stage; every OTA edge then reads the node's held copy). The core current then does not depend on the evolving state, so the core is effectively a feed-forward "instantaneous map from stage-entry state to a constant current into each node" and only shunt/leak/clip/boundary/sense are dynamic. Cheapest hardware reading is **per node S&H plus per-edge OTA**. Fidelity-critical for the current numerics. Whether to keep it is the biggest architectural decision (§4). |
| `freeze_boundary` / `freeze_temporal_read` | `:215-216, 1127-1230`; TS `:2874,2884`; BO `:515-518` | off | `kn_bayes_opt`, `narma_advisor_probes`, `test_f1_f2` | off / off | **Hard**, same as above for other families. Remove. |
| `core_refresh_interval` (recompute frozen current every k Heun steps) | `:238, 281-285, 1885, 2097-2158`; `topology.py:2091-2119`; NARMA presets | 0 | 23 NARMA/probe files, no CTLE/FR | 0 / 0 | **Hard** (periodic S&H). Remove, or keep as the only general form of freeze (k=∞ ≡ freeze_read; k=1 ≡ no freeze) and delete the boolean. |
| `stage_noise_std` (additive Gaussian between stages) | `kirchhoff_net.py:163-172` | 0 | `train.py`, `train_script.py` | off / off | **Easy** in Spectre via `noise`; remove from model. |
| NARMA streaming (`forward_sequence`, `_forward_heun_sequence`, `carry_keep`) | `kirchhoff_net.py:182-297`; `differential_stage.py:1927-2050` | | NARMA only | no / no | Not in scope for CTLE/Friedman. Isolate or remove. |

### 1.4 Gating and input-conditioned features

| Feature | Where | Default | Used by | CTLE / FR | HW |
|---|---|---|---|---|---|
| **VCA gate** `g_e = 2σ((u·W)·v_e [+b_e])`, rank r | `_compute_vca_gate :1059-1087`, `_compute_core_gate :1089-1125`; params `vca_W, vca_W_core, vca_v_{boundary,readout,core}` init `:732-796`; `config.py:603-608` | off in code; **on** (rank 3) in BO/DAG | 27 files | yes (r=3, core+boundary+readout, separate core bus, no bias, no shunt gating) / yes (r from BO) | see §1.5 |
| `vca_core`, `vca_separate_core_bus`, `vca_gate_shunt`, `vca_bias`, `vca_use_hidden` | `:233-237, 306-311, 1089-1125`; TS `:2787-2806` | off/off/off/off/off | BO sets core+separate bus always; `vca_use_hidden` only `narma` | CTLE: core+separate bus / same | `vca_use_hidden` (gate depends on node voltages, `:300-306`) **Not physical**, remove. `vca_gate_shunt`, `vca_bias`: remove (off in all live configs). |
| **GLN rails** | `gln_rails.py:74-175` (`GLNRails`, `_EdgeMix`); stage `:206, 360, 1153-1161, 1265-1276, 1473-1475, 1559-1569`; build `topology.py:1966-1970, 2082-2086, 2511-2555`; TS `--gln-*` `:2418-2439`; BO `:524-526`; DAG `KN_GLN_RAILS=False` default (`:644`) | off | BO, TS, DAG, spice_failure_check, `test_f1_f2` (67 refs) | yes (B=4, rank 2) / no (not in the Friedman snapshot) | see §1.5 |
| `interstage_activation` `residual-relu-tanh` (+ `relu`, `residual`, `residual_mixing`, `drive_mask`, `residual_rank`) | `stage_transfer.py:31-192`; built `topology.py` (build_net_from_config `interstage_*`); used `kirchhoff_net.py:173-174`; TS `:2896-2907` | `none` | 13 files | yes / yes | see §1.5 |
| Post-readout transfer | `kirchhoff_net.py:487-499, 662` always on (28 params) | identity at init | all via `KirchhoffNetWithIO` | yes / yes | digital post-processing (outside the netlist) |
| `OutputAffine` | `io_mapper.py:577-616`; 5 params/output | on for temporal/shared readout | all | yes / yes | digital post-processing |

### 1.5 Hardware-focused assessment of the specifically requested features

**GLN rails.** Math: `z_b = tanh(α_b(a_b·u + c_b))`, `δ_e = Σ_b W[e,b]·z_b` with `W = P@Q`, `gm_e = clamp(gm0_e·exp(δ_e), 0.01, gm_max)` (`gln_rails.py:146-168`). Applies to boundary (8/stage) and readout-sense (22/stage) families only, 30 edges/stage; parameters tied across stages (100 params total).
- *Semantics matter:* `gm` is the tanh *argument slope* (CM §3-C2), so GLN = input-dependent slope control, **not** tail-current control. Tail current sets Isat, not slope. A hardware slope control needs variable source degeneration (switched resistor ladder) or a programmable input attenuator in front of a fixed-slope pair. That is the circuit that "multiplies by exp(δ)". The exp is easy only in sub-threshold current-mode designs (translinear), which control a *current*, i.e. Isat, not a voltage-domain slope.
- *Circuit shape:* (i) 4 rail generators: a 4-input linear combiner + a tanh (diff pair), shared net-wide; (ii) per edge, a 4-way weighted sum of the rails with programmable weights (`W[e,:]` is a trained row, rank 2): ~4 multipliers or one 4-input programmable transconductor per edge; (iii) exp (translinear/log-domain) per edge; (iv) a clamp to [0.01, gm_max]; (v) the exponentiated result must drive a degeneration DAC or a slope control. Per edge that is roughly ≥ 4 weight cells + 1 exp + 1 slope DAC, evaluated for 30 edges per stage × 7 stages (the *values* are tied across stages, but each stage has its own OTAs, so the control paths are replicated or time-multiplexed).
- *Verilog-A:* **Easy** (an `analog function` of the 4 input pin voltages; because `u` is DC per inference it can be a static table). *22FDX:* **Hard** (≈ 8-10 transistor-level analog blocks per edge, nonlinear in the input; accuracy and mismatch of exp and tanh blocks dominate). It only touches 30 of 52 edges per stage and never the core (22 edges). Your doubt is justified at transistor level; it is implementable in Verilog-A, and it costs roughly as much silicon as the edges it modulates.

**VCA.** Math: `g_e = 2σ((u·W)ᵀv_e)`, `u` 4-dim input, `W` 4×r (shared "control buses"), `v_e` r-dim per edge, `g ∈ (0, 2)`; multiplies the tanh current of core (per `vca_separate_core_bus`) and boundary/readout (tanh *and* shunt together, `:1574-1578, 1724-1728`).
- *Circuit shape:* r shared control voltages (4→r linear combiner, no nonlinearity); per edge a dot-product of r with programmable `v_e`, then a sigmoid. The sigmoid is natural: a diff pair steering the tail current gives `I1/(I1+I2) = σ(v)`; the factor 2 is a current mirror ratio. The gain multiplies the OTA output: a Gilbert-style current steering stage or tail-current scaling (Isat·g). So per edge: r-weight programmable dot product (r transconductors or resistive DAC) + one steering diff pair + one mirror. That is "tail-current steering driven by a shared analog bus". No exp, no per-edge nonlinear function other than a diff pair.
- *Verilog-A:* **Easy.** *22FDX:* **Moderate** (one extra differential steering stage and r weight cells per gated edge; range 0-2 relies on matching of the steering and mirror). Cost per edge ≈ 1 diff pair + 1 mirror + r weight cells, for core (22), boundary (8), readout (22) edges = all 52 edges/stage in the paper config. For r=3 this is ~3× the weight cells of the base OTA. Compared with GLN: simpler, but ubiquitous. Since `u` is DC per inference, the rank-r structure could be replaced at export by a per-edge DC bias set once per sample (a per-sample DAC, no analog dot product).
- *Alternative that deletes it:* if the input is needed only for training-time capacity, remove VCA and GLN and let boundary OTAs (which already see `u` with trained gm, Isat) carry all input dependence. The paper's ablations should say how much accuracy VCA/GLN buy; I did not find an ablation table in `paper_raw.txt` (not verified here).

**`freeze_read`.** S&H **per node** (not per edge). The frozen core current of every core edge is `f(x_src_held, x_dst_held)`; both terminals are node voltages, so 18 node S&H (per stage) feed ~22 OTAs. Edge-level S&H would be 22 per stage. Alternatives: (a) keep the *core* instantaneous (no freeze) and retrain (drops 18 S&H and the stage-entry sampling logic, adds true feedback dynamics through the core, which may be stiffer: integrator is `Isat·gm·A` up to ~120 per node vs C=1); (b) keep freeze but stage = clocked sample phase. Option (a) is the "natural" circuit (an autonomous nonlinear RC network), but `core_refresh_interval=1` ≡ no freeze, and nothing proves the CTLE accuracy survives it; **must be tested by retraining**.

**Learned interstage transfer** `x' = w1·x + w2·tanh x + w3·ReLU(x − vth)` per node, on all 18 nodes (including accumulators), 6 transfers (CTLE) = 432 params (`stage_transfer.py:161-169`; applied `kirchhoff_net.py:173-174`). Needs per node: sample the node, a nonlinear block (gain, tanh, ReLU with programmable threshold), and re-initialisation of the node capacitor to the output. That is a clocked analog block per node per stage boundary. *Verilog-A:* **Moderate** (analog function + `idt` reset or `@timer` event). *22FDX:* **Hard** (S&H, buffer, nonlinear transform, cap reset per node, and 6 stage boundaries). If 7 stages are physically 7 copies, the transfer is a wire with a nonlinear buffer; if time-multiplexed on one fabric, it is also a bias-reload.

**Signed polarity `s`, A/B ratios, θ.** See table 1.1. All three are per-edge static programmable parameters in the model; in hardware they are 3 extra programmable dimensions per OTA, plus Isat, gm, G, σ(z) = 8 per edge. Note `s=±1` cannot be learned smoothly in hardware (it's a switch), and the code already uses STE. For retraining, consider fixing `s` by the sign of an initial weight, or replace by `{A, B}` swap = `s` (since `s·(A·Vs − B·Vd)` with A,B both free and positive already implements a swap when A=B; it is only needed to flip the output current direction).

**Shared-sense readout + crossbar.** Sense OTAs against a per-stage learnable Vref and signed dense `W[7×22]` (`differential_stage.py:688-690, 1735-1743`). Realizable as a VCCS array; signed weights need differential columns (paper says so). Vref in [0, x_max], not at zero rail: per-stage DAC. *Verilog-A:* **Easy**; *22FDX:* **Moderate**.

**Output affine (gain, bias, tanh_gain, relu_gain, relu_threshold) and post-readout transfer.** Digital/off-chip (the CTLE decode `10**(lo+(hi-lo)σ(y))` is also off-chip: `dagger-nuance-…py` predict). Keep outside the analog netlist.

**MoE/experts.** None in the KNet (finding 3).

**Pruning gates σ(z).** Training-time structural sparsity: `prune_stage/prune_network` (`topology.py:710-1295`, ~590 lines), `train_script.py:2535-2556`, `post_twin_gates.py`, smoke tests (293 "prune" refs). Not used in the CTLE paper path. For hardware, pruned edges become absent OTAs, so keep only an export-time mask; remove the training machinery and the 3-/4-phase schedules (`config.py:430-528`).

**Degree budget** (§1.2): remove. **`core_refresh_interval`** (§1.3): remove or reduce to a single freeze-period parameter. **DEQ path**: remove. **Noise models:** `kirchhoff_noise.py` (354 lines), `analog_noise.py` (568 lines), `sim_context.py` (177 lines) and `VARIATION` (`config.py:652-665`): dead for HR-KNet; replace with Spectre Monte-Carlo and `noise` analysis after the Verilog-A stage. `analog_noise.py` is still imported by the MLP benchmarks (`mlp_benchmark*.py`).

---------------------------------------------------------------------------------------------------

## 2. Entry-point map and dead code

Size of core modules (lines): `differential_stage.py` 2609, `topology.py` 2873, `config.py` 1321, `kirchhoff_net.py` 803, `io_mapper.py` 688, `cell_library.py` 666, `analog_noise.py` 568, `kirchhoff_noise.py` 354, `stage_transfer.py` 204, `gln_rails.py` 203, `sim_context.py` 177, `tap_rails.py` 62. Big functions: `DifferentialStage.__init__` 192-830 (640 lines), `rhs` 1388-1779 (390), `topology_to_stage` 1339-1735 (400), `build_net_from_config` 1934-2873 (940), `prune_stage/prune_network` 710-1295 (585), degree budget 844-1044 (200), `forward_equilibrium` 2188-2305 (120). `io_mapper.py` has 13 classes; the HR-KNet needs `NullInputMapper` (`:278`) and `OutputAffine` (`:577`); the other 11 are legacy input/output maps (only 1 feature-flag hit).

### 2.1 Live entry points (the user should keep)

| Script | Role | Core features it needs |
|---|---|---|
| `dagger-nuance-distillation-kirchhoffnet.py` (205 KB) | Builds the paper CTLE HR-KNet student: Phase-A single-shot, canonical dataset (`PR §2.1`). Constants block `:621-660` (`KN_FREEZE_READ=True`, `KN_VCA_CORE=True`, `KN_INTERSTAGE_ACTIVATION`, `KN_GLN_*` default off, `KN_LEARNABLE_CLIP_SHARPNESS=False`). Wrapper `LocalKirchhoffStudentWrapper` `:1700-1990`. | all CTLE features |
| `kn_bayes_opt.py` (111 KB) | Optuna driver for CTLE/Friedman/smooth2d; builds subprocess commands for `train_script.py` (Friedman/smooth2d, `:481-529`) and the DAgger script (CTLE, `:532-…`). Not imported by anyone. | spawns the above; moe args inert |
| `train_script.py` (251 KB) + `train.py` (58 KB, loss/regulariser/loop) | Static-regression trainer (Friedman, housing, sinx, smooth2d, narma presets via `config.PRESETS`). 137 `add_argument`s. | every flag in §1 (GLN, VCA, prune, DEQ, budget, drive, ref, skip-linear, 3/4-phase schedules) |
| `ctle_dagger_common.py` (202 KB) | Canonical dataset, ZIG surrogate, distillation losses, failure metric. Not model code. | none of the model flags |
| `spice_failure_check.py` (90 KB) | Rebuilds the 5,649-param model from defaults (`:138-203`) and runs Cadence; `KNetDistillWrapper :828-1060` | builds CTLE config via `topology.build_net_from_config` |
| `fixed-mlp-distillation-kirchhoffnet.py`, `fixed-distillation-bayes-opt.py`, `run_phase_a_knet_bo.sh`, `run_train_knet_single_shot_distillation.sh` | Phase-A BO wrapper chain | same as DAG |
| `bo_param_sampling.py` | BO search space, imported by BO and tests | `tanh_free`, interstage, readout, gln |
| `test_f1_f2.py` | Only live HR-KNet feature test | GLN, VCA, learnable clip, freeze_*, shared-sense, interstage |

### 2.2 One-off / experimental (NARMA stream, 8 files `narma_*.py` + ~15 probes + 22 `.sh` batch scripts)

`narma_experiment.py` (147 KB), `narma_linear_controls.py` (218 KB), `narma_advisor_probes.py` (122 KB), `narma_revised_plan.py` (43 KB), `narma_phase7_leak_bands.py`, `narma_spectral_norm.py`, `narma_phase8_tied_gate_probe.py`, `narma_phase8_tied_gates_audit.py`, `post_twin_gates.py` (82 KB), `aux_core_probe.py` (79 KB), `audit_aux_core.py`, `audit_post_twin_gates.py`, `audit_twin_standalone.py`, `probe_audit1/2/_nodeact/_revised/_split/_transplant.py`, `probe_study_db.py`, `smoke_audit3.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `vca_sanity.py`, `run_vca_grid.py`, `dry_run_knet_input_clipping.py`, `_test_step.py`, `_count_params.py`, `gen_network_images.py`. Features only these use: `core_refresh_interval` (23 files), `node_activation` (8), `dynamic_leak` (3), `TapRails` (4: `narma_phase8_*`, `narma_advisor_probes`), NARMA sequence forward, `vca_use_hidden`, linear-OTA control library, `narma_*` presets (`config.py:1175-1307`).

### 2.3 Dead or effectively dead

- `tap_rails.py` (62 lines): only NARMA phase-8 probes. `differential_stage.py:363` attaches `self.tap_rails = None`.
- `sim_context.py`, `kirchhoff_noise.py`, `analog_noise.py` (HR-KNet side): not consumed by stage/cell/topology; imported only by `train.py`, `train_script.py`, MLP benchmarks, tests, `gen_network_images.py`.
- `kn_benchmark_friedman2.py` (31 KB): imports `utils.model` / `utils.topology` which do not exist in the repo (`Test-Path utils` = False). It benchmarks the *original CircuitNet* from `../kirchhoffnet_original`, not HR-KNet. Broken here.
- `ctle_kirchhoff_bo.py` (modified, uncommitted): different adjoint/dopri5 model with its own bounds (CM §C12). Not HR-KNet.
- `train_ctle.py` (44 KB): old 3-stage grid KirchhoffNet distilled from a RegimeAwareMoE teacher (`train_ctle.py:1-5`); uses legacy features (budget, drive, prune).
- Baselines unrelated to the surgery: `mlp_*`, `xgb_*`, `tree_benchmark.py`, `generative-distillation-*`, `plot_*`, `generate_report.py`, `summarize_fixed_bo.py`, `compare-fixed-distillation-bo.py`, `check_optuna_versions.py`.
- Dead parameters: `u_logits` (`differential_stage.py:440`), `LAMBDAS["node_gate"/"capacitance"]`, `PRUNE["node_threshold"]`.
- Dead import paths from tests: `deq_solver`, `deq_diagnostics`, `sparse_solver_data`, `joblib`, `compute_solidification_metrics`, `_stage_soft_weights` (missing from `train.py`).

---------------------------------------------------------------------------------------------------

## 3. Test coverage

Run notes: `venv\Scripts\python.exe` has torch 2.13 CPU but **no pytest**, so pytest was run with system Python 3.13 (`C:\Python313\python.exe`, torch 2.14 CPU, pytest 9.0.3), `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`. Every file finished in < 30 s. Tests have no CUDA dependency here.

| File | Tests | Result | Features covered |
|---|---|---|---|
| `test_smoke.py` (8421 lines, 375 KB, script-style `main()` plus 246 `test_*` functions) | 246 | **158 pass, 88 fail** (26 s) | prune (20 test names, 293 keyword hits), DEQ (15), budget (11), lrp (11), repeat-edges/bidir (19), legacy v15/v2 libraries (18), SimContext (72 hits), analog noise (22), drive (27), ref edges (6), small-world (33), tau schedule, mappers. **Zero** references to GLN, VCA, freeze_read, core_refresh, node_activation, dynamic_leak, learnable_clip, OutputAffine. |
| `test_training.py` (1954 lines) | 40 | **39 pass, 1 fail** (`test_no_adc_flag`: `FileNotFoundError`, subprocess launches a script/interpreter that is not on PATH) | OutputAffine (17 hits), drive (23), interstage/StageTransfer (8), NARMA preset forward/state carry, edge repeats, LR scheduler. |
| `test_f1_f2.py` (326 lines) | 15 | **15 pass** (4 s) | GLN (67 hits: identity at init, shared params, families, param count), learnable clip sharpness (13), VCA, freeze_read/boundary/temporal, shared-sense, interstage, small-world. **The only suite aligned with the paper config.** |
| `test_bo_guard.py` (485 lines) | 12 | **10 pass, 2 fail**: `test_metrics_parse_failures_yield_penalty_not_fail` (PermissionError deleting a temp dir: sandbox artefact) and `test_sampler_rev_bumped` (asserts `SAMPLER_REV = 2` in source; stale) | BO sampling/guards, VCA arg plumbing. |
| `test_audit.py`, `test_friedman2_small_world.py`, `_test_step.py` | 0 `test_` functions | script-style, not collected | — |

The 88 `test_smoke.py` failures cluster in four buckets (counts are repeated "E" lines in the log, not unique tests): `NameError: ring_graph/line_graph not defined` 34+ (a missing import inside tests, e.g. test file imports in functions but calls without import), `AttributeError: 'SimpleEdgeLibrary' object has no attribute 'num_cells'` 32 (legacy library API removed), `AttributeError: 'DifferentialStage' object has no attribute 'logits'` 14 (old API), `ModuleNotFoundError` for `sparse_solver_data`/`joblib`/`deq_solver`/`deq_diagnostics` 28, `ValueError: Unknown cell library 'legacy'` 8, `TypeError … unexpected keyword 'ctx'/'cell_mode'` 10. Running `python test_smoke.py` (script mode) stops with `NameError: line_graph` at Test 3 and reports 8 stale `[FAIL]` config assertions (e.g. `LAMBDAS has rail=1.0`, `PRESETS has sinx,…`). Conclusion: this file documents a previous era of the repo (legacy L/S/P/Z cells and a `ctx` variation path) and cannot be used as a regression guard. The surgery should delete the stale half and keep only the passing 158 that exercise the code that survives (a triage of which of the 158 survive was not done; most are legacy-prune/DEQ/budget tests that will be deleted with the feature).

No tests exist for: paper-config numerics (5,649 params is only verified by CM's manual run), `core_refresh_interval`, `node_activation`, `dynamic_leak`, TapRails, heun-vs-reference-ODE agreement, checkpoint load/save of the CTLE model, export to Verilog-A.

---------------------------------------------------------------------------------------------------

## 4. Proposed minimal Cadence-realizable core

### 4.1 KEEP (core)

- `FreeTanhLibrary`-style OTA: `I = gated Isat·tanh(gm·(A·Vs − B·Vd))` in parallel with a shunt `G`, with *bounded, hardware-meaningful* parameters (§5). `tanh_free` only.
- Node: `C`, fixed leak, one physical rail clamp.
- Graph: Watts-Strogatz seeded, `edge_repeats`, stages, boundary OTAs, shared-sense readout + signed crossbar (or a single readout choice), Heun solver for training, per-stage `t_span/num_steps`.
- Interstage transfer: keep only if retraining shows it matters; otherwise it is the first thing to remove (§4.3).
- Edge-gate σ(z): keep in training as a structural mask (it exports into Isat/G).
- `s` sign: keep as exported static switch bit.
- Output affine and post-readout transfer: keep (digital).

### 4.2 OPTIONAL but isolated (separate module / flag, off by default, not imported by the core)

- VCA gate (core, boundary, readout): behind a `gating/vca.py` with a single entry `edge_gain(u) -> [E]` (the stage multiplies `i_edge` by that vector; nothing else knows about VCA). This is already the structure of `_compute_vca_gate`.
- GLN rails: same isolation as an `edge_gm_scale(u) -> [E]`. This is already what `GLNRails.modulate_gm_z` returns. Candidate to remove entirely if the user decides it is not implementable.
- `freeze_read` as a single "core evaluation period" (`core_refresh_interval`) with `0 ≡ continuous`.
- Learnable clip sharpness: keep only as a trained constant; fix before export.
- `node_activation`, `read_only_source`, NARMA sequence forward: move to an `experimental/` package used by NARMA scripts only.

### 4.3 REMOVE

`FreeTanh` `parallel_tanh_mult`; all legacy libraries (`SimpleEdge`, `RealisticTanh*`, `LinearOTA`, `AntiParallel`, legacy L/S/P/Z/v15/v2 tables and `cells_to_tensor_dict`); DEQ; degree budget; pruning machinery and the 3/4-phase schedules (keep an export-time mask); node gates; drive; ref edges; skip-linear; dynamic leak; `vca_use_hidden`, `vca_gate_shunt`, `vca_bias`; programmable leak; `output_ode_*` `ota_mesh` mesh readout (keep shared-sense); `SimContext` / `kirchhoff_noise` / `analog_noise` (HR-KNet side); `stage_noise_std`; `TapRails`; `freeze_boundary`/`freeze_temporal_read`; 10 non-CTLE presets; `train_ctle.py` and the legacy `train.py` regularisers that serve the removed features (`edge_gate`, `node_gate`, `capacitance`, `entropy`, `rail` stay only if used by the retrain). MoE flags in BO/DAG command builders.

### 4.4 Surgery plan (ordered; risks)

1. **Freeze a numerical baseline before touching code.** Commit the stale state tag. Build a script (new file, outside the repo or in `tests/`) that, for the CTLE config (`spice_failure_check.py:138-203` defaults, 5,649 params, seed fixed) and the Friedman-2 BO config (`friedman2_knet_e1`), saves: initial state_dict (seeded), forward output on a fixed batch (e.g. 64 inputs, float64 if possible), per-stage states, and one training step's loss and gradient norms (`.pt` golden file). Also run the same with `freeze_read=False`, GLN off, VCA on and off, learnable clip on/off to cover each branch. Risk: `s_raw = randn·0.1` and `edge repeat` order depend on the RNG stream; seeding order must be preserved (`topology.py:208-307` uses `random.Random(seed)`, edges come from a Python `set` iteration: CM §4 says use the stored `src/dst` buffers, so also store them).
2. **Turn `test_f1_f2.py` style checks into the regression suite**, add a "feature-off ≡ baseline" test per removal (compare against golden). Delete the 88 failing smoke tests outright, keep the 158 only where the feature survives. Risk: low; these tests currently provide no protection.
3. **Delete dead modules and scripts (no model impact):** `tap_rails.py`, `kirchhoff_noise.py`, `analog_noise.py` (only after MLP benchmarks import their own copy), `sim_context.py`, `kn_benchmark_friedman2.py`, `train_ctle.py`, `dry_run_*`, `_test_step.py`, `_count_params.py`, NARMA probes/batch `.sh` (or move to `legacy/`). Risk: `train.py`/`train_script.py` import `sim_context` and `analog_noise` (`train_script.py` top imports), so remove those imports and `--no-adc`/ctx code paths together (also fixes `test_no_adc_flag`).
4. **Remove legacy libraries/tables from `cell_library.py`/`config.py`** and make `make_cell_library` return `FreeTanhLibrary` only. Risk: `bo_param_sampling.py`, `ctle_dagger_common.py`, `fixed-*` reference library names via strings; 35 files mention legacy names, mostly in comments/choices. Update choices and run the golden test.
5. **Strip DEQ, budget, drive, ref-edge, dynamic-leak, node-gate, skip-linear, pruning, 3/4-phase scheduler** from `DifferentialStage`/`topology`/`train_script`. Do it one feature per commit; after each, rerun the golden tests. Risk: `parameter_breakdown`, `state_dict` keys: `u_logits` removal changes the key set (old checkpoints would fail `load_state_dict(strict=True)`; the only checkpoints in the repo are Friedman-2 `trial_0000/model.pt` and the PlainMLP teacher, and the CTLE checkpoint is not in the repo, so the impact is small). Add a `strict=False` loader for `u_logits` only.
6. **Collapse the freeze flags** into `core_refresh_interval` (0 = continuous; k = hold for k Heun steps; ≥ num_steps ≡ freeze_read=True). Delete `freeze_boundary`/`freeze_temporal_read`. Risk: `kn_bayes_opt.py:497` and DAG hard-code `--freeze-read`; keep `--freeze-read` as an alias for one release. The equivalence `k≥num_steps ≡ freeze_read` is documented at `differential_stage.py:96-98, 250-258`; verify it by test (compare outputs) before relying on it.
7. **Isolate VCA and GLN** behind small interfaces (`edge_gain`, `edge_gm_scale`), so the stage `rhs` has a single multiply per family rather than 7 branches (`:1153-1161, 1176, 1227, 1265-1272, 1287, 1473-1475, 1559-1575, 1602, 1709-1728`). Then decide: keep VCA (optional), drop GLN. Risk: GLN removal changes the 5,649 → 5,549 param count of the paper model and makes it a different model; do that only after the user chooses to retrain.
8. **Add the exporter** (state_dict + graph buffers → Verilog-A netlist/parameter table). This is the real deliverable; it also validates that the simplified model has no feature left that the exporter cannot express (the exporter is a test: any parameter in `state_dict` not mapped to a Verilog-A parameter should fail the build).
9. **Refactor builders** (`build_net_from_config` 940 lines, `topology_to_stage` 400): replace kwarg pass-through with one `HRKNetConfig` dataclass (the CTLE config is 20 numbers). Risk: `kn_bayes_opt.py`, `bo_param_sampling.py`, `spice_failure_check.py` and 22 `.sh` scripts build CLI strings: keep the CLI names stable during this step.
10. **Re-run the golden tests and a short CTLE train** (a few epochs) to confirm the loss trajectory matches the pre-surgery run within float tolerance.

How to keep the "simplified ≡ current when removed features are OFF" check: the golden file approach above, applied at each step with all removed features at their *current default values*: `drive=False, ref_edges=False, budget=False, dynamic_leak=False, node_activation='none', core_refresh_interval=0, enable_skip_linear=False, deq=False`, fixed 0.0486 leak. Compare `net.state_dict()` key sets (minus the deleted ones), forward output with `torch.allclose(rtol=1e-6, atol=1e-7)` in float64, and the first-step gradient. Keep the pre-surgery tag so a bisect is possible. Note the seeded `randn`-based init means initial parameter values must be loaded from the golden state_dict, not regenerated.

---------------------------------------------------------------------------------------------------

## 5. Hardware-motivated changes to consider for the retrain (recommendations only)

1. **Bounded A and B, preferably A+B fixed.** Replace unbounded `softplus(a_raw)`, `softplus(b_raw)` (`cell_library.py:344-345`) by `A = a_max·σ(a_raw)`, `B = b_max·σ(b_raw)` (a W-ratio is typically within ~0.25-4), or a single ratio `r` with `A=2r/(1+r)`, `B=2/(1+r)` so the differential part is constant and only common-mode changes. Prevents the unbounded effective slope `gm·A`.
2. **Couple gm to Isat through overdrive/degeneration** instead of two independent trained values. For a diff pair in strong inversion, slope ≈ `1/(Vov)` with `Isat = gm_phys·Vov`-type relation; in weak inversion slope = `1/(2nVt)` ≈ 15-19 /V, independent of Isat. Parametrise `gm_slope ≤ 1/(2 n Vt)·Vu` (with `Vu` the V-per-unit scale) and make Isat the only free current. Remove independent `gm` or keep it as a bounded degeneration factor `1/(1+Rs·gm_phys)`.
3. **Hard physical rails** instead of the soft 0.05 clip and the compliance gate. The CTLE logs show |x| up to 6.4× x_max (CM §0.3), which means the model relies on node voltages no 0.8 V supply can produce. Options: (a) clamp via a differentiable `clamp`/`softclamp` on the node *voltage* in the RHS (`dx/dt` zero when pushing outward), (b) make OTA output current go to 0 smoothly as the output node approaches the rail (a real OTA's output compliance), and delete the input-voltage compliance gate (`cell_library.py:399-401`). Pick (b) with an explicit 22FDX headroom (e.g. usable swing ~0.2-0.6 V of 0.8 V).
4. **Remove the 0.01·u gradient floor or model it as finite output conductance.** If kept, replace by an explicit per-OTA output conductance `g_out = Isat·λ` (small, positive), so it is a real parameter that Spectre also sees (and `0.01·u` as a *linear* term in the tanh argument has no hardware meaning). Training can use a gradient surrogate (e.g. straight-through `tanh`) without changing the forward model.
5. **Physical unit scaling (V, I, T).** Fix `Vu` (volts per unit), `Iu` (amps per unit), `Tu` (seconds per unit) in `config.py` and train in units where `C_eff·Iu·Tu/Vu` is the intended small capacitor (CM §7 sketch: `Vu=0.1 V`, `Iu=1 µA`, `Tu=1 ns` → `C=10 fF`). Add `Vcm = VDD/2 = 0.4 V` and map `x ∈ [−x_max, x_max]` to `0.4 V + Vu·x`. Fix `x_max` to a value consistent with headroom (e.g. ~2-3 units at `Vu=0.1`) rather than searching 0.5-8.
6. **Capacitance choice.** Do not search `C_eff` as a free BO parameter (the Friedman BO does: 9.06). Choose it from area: with small C (≈ 10-50 fF), parasitics (wire + OTA input caps ≈ 2-10 fF per edge terminal in 22FDX) become 10-50 % of C. Add a parasitic term `C_node = C_wire + Σ C_in(edges)` to the model (currently "no charge-sharing parasitics", CM §5) before retraining, otherwise `C_eff=1` will not hold after layout. Also make `t_span`, `num_steps` consistent with the smallest physical time constant (`C/(Isat·slope)`).
7. **Bound `Isat` by power.** Add a differentiable `Σ Isat·VDD` penalty (the existing `LAMBDAS["power"]` is a count of gm, not current).
8. **Reduce what is trained per edge.** Per edge today: A, B, s, gm, Isat, θ, G, σ(z), plus VCA/GLN taps (8-11 numbers). Candidates to freeze or drop at retrain: θ (or fold into A≠B), σ(z) (fold into Isat), `G` (set to a fixed ratio of Isat or to the OTA output conductance), gm (fix at the device-limited slope).
9. **Differential vs single-ended.** The model is symmetric about 0 and has no common-mode. Decide whether nodes are fully differential pairs (doubles every node/cap/OTA pair) or single-ended around Vcm (then OTA outputs need a Vcm-referenced mirror). This is not in the code at all and affects the A/B/θ meaning.
10. **Input scaling in the analog domain.** Inputs are log10 + min-max + clamp ±4 off-chip (`dagger-nuance-…:1916-1937`). Keep them off-chip but restrict `u` to the range the boundary OTA can accept (±0.3 V around Vcm). Same for the output decode.
11. **Time-multiplexing vs 7 copies.** Decide before retraining: 7 stages × (22+8+22) = 364 OTAs; if time-multiplexed, parameters must be reloadable between stages (bias DACs per edge), which makes GLN/VCA much costlier.
12. **Mismatch awareness.** Retrain with injected per-edge multiplicative Gaussian on Isat/G/A/B (simple, in the model, replacing the dead `SimContext` path) at levels taken from 22FDX Monte-Carlo, so the exported design survives Spectre variation.

---------------------------------------------------------------------------------------------------

## 6. Caveats

- I read `config.py:255-1321`, `cell_library.py:224-438`, `kirchhoff_net.py:86-215, 348-557`, `gln_rails.py`, `stage_transfer.py:30-204`, `topology.py:1934-2193` and the constructor/signature layers of `differential_stage.py`; the rest of `differential_stage.py`/`topology.py` relies on CM's line-level reading (CM §3).
- Feature-usage lists are from regex greps over `*.py`/`*.sh` (counts overlap with comments). "Used by" means "mentions the flag", not "runs with it on".
- The CTLE checkpoint is not in the repo (CM §0.2). Feature importance for accuracy (VCA, GLN, freeze, interstage) is **not measured here**; ablations require retraining.
- Pytest ran on system Python 3.13 / torch 2.14, not the venv (torch 2.13, no pytest). Failure causes are API/import errors, not numerics, so I expect the venv to give the same result, but did not verify.
- `test_bo_guard` and `test_no_adc_flag` failures include sandbox artefacts (temp-dir delete, subprocess lookup).
