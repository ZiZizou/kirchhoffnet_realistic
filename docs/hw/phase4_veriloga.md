# Phase 4 — Verilog-A behavioral implementation

## Status

This is an initial behavioral primitive library, not yet a validated checkpoint-to-Spectre flow. The canonical constants and block-level modules are committed on `phase4-veriloga`. Task A wiring is pending: `ota_edge.ab_mode` reserves 0=legacy, 1=ratio, 2=differential, 3=cap-divider; mode 2 forces A=B=1 and mode 3 consumes Ahat/Bhat. The circuit-level scaling/validation still needs review against the model before use for signoff.

## Unit contract

Single-ended core supply is 0.8 V, Vcm=0.4 V, Vu=0.07 V/unit, C_u=50 fF, T_u=75.1 ns/model-time-unit, I_u=C_u Vu/T_u, G_u=C_u/T_u. Model state is x=(V−Vcm)/Vu; currents in model units are scaled by I_u and conductances by G_u. `HWUnits` is authoritative for configurable conversions.

## Behavioral modules

| File / module | Ports | Principal parameters |
|---|---|---|
| `ota_edge.va` / `ota_edge` | src,dst electrical | mode,A,B,Ahat,Bhat,ab_mode,s,theta,gm,Isat,z_gate,vca,gln |
| `shunt.va` / `shunt` | src,dst electrical | G,z_gate,vca,vca_gate_shunt |
| `node_cap_leak.va` / `node_cap_leak` | node electrical | c_eff,c_phys,g_leak,Vcm |
| `rail_clip.va` / `rail_clip` | node electrical | x_max,sharp,Iclip,Vcm,Vu |
| `sample_hold.va` / `sample_hold` | in,out,clk electrical | C_hold,R_on,R_off,inject_v,droop_tau |
| `boundary_sense_ota.va` / `boundary_sense_ota` | src,dst electrical | gm,Isat,A,B,theta,s,z_gate |
| `crossbar_accumulator.va` / `crossbar_accumulator` | sense,acc electrical | weight,Vref,Gunit,Cacc |
| `interstage_transfer.va` / `interstage_transfer` | in,out,enable electrical | w1,w2,w3,vth,x_hard |
| `vca_bias.va` / `vca_bias` | u0..u3,gain electrical | weights,bias |
| `gln_bias.va` / `gln_bias` | u0..u3,trim electrical | weights,bias,delta_max |
| `phase_ctrl.va` / `phase_ctrl` | clk,phase,stage electrical | stages and phase durations |

## Fidelity and gaps

`ota_edge` contains legacy and hardware parameter branches and the A/B extension hook. Legacy implementation is intended to represent softplus A/B values, signed differential tanh, edge/VCA multiplier, compliance gating and the linear floor term. Hardware branch exposes slope attenuation and A/B normalized inputs. These are primitive implementations; exact end-to-end legacy reproduction has not been established. Hardware output-leg compliance, parameter-bank scheduling, exact shared-sense wiring, and nonideal S&H parameter conversion remain to be completed/verified. OutputAffine and input log/min-max scaling are digital/off-chip by design.

## Exporter interface and resolved table

The Python behavioral exporter is available as `python -m export.spectre_export --checkpoint ... [--hw-config ... | --hw-preset NAME] --mode legacy|hw --fabric timemux|copies --specs N --out DIR`; library callers can use `export_model(model, ...)` for an already-built KirchhoffNetWithIO. `params.json` contains `resolved` (schema `kirchhoffnet-resolved-params-v1`) with input scaling, topology, stage settings/cells/gates/readout/VCA/GLN, transfers and output mapping. Dimensionless values are the interpreter's source of truth. In hw mode each stage additionally has a `physical` view (V, A, S, F, s) and cap-divider integer codes. `HWUnits` provides the conversions: x→V uses Vcm+Vu*x, currents use Iu, conductances use Gu, capacitances use Cu, and time uses Tu.

The state-dict map is deliberately strict: each serialized key must be classified as a topology entry, a mapped behavioral parameter, or an explicit digital/off-chip key (`DIGITAL_OFFCHIP_KEYS`; input log scaling and output affine/post-transfer are off-chip). Unknown keys fail export. This mapping is being tightened to verify actual table consumption, not merely recognize leaf names.

Hw export resolves A/B modes (including cap-divider codes), slope/IC coupling, no-alpha, theta bound, Vref and crossbar transforms from the model's resolved module methods. It refuses stochastic mismatch, training-only power, VCA-logit quantization, output compliance, hard transfer clamp, node-capacitance modes, GLN tail mode, Isat-eff ceiling, and active timemux/S&H; these paths are not yet represented by the standalone interpreter and are rejected rather than silently approximated. Validate flags using `HWConfig` before export. `phase3a_core_diff` and cap-divider core presets are candidate supported configurations; `phase3a_full` and `phase3b_tm` currently include refused dynamics.

## Validation

The interpreter has no model imports and the legacy paper-checkpoint path is tested against forward/logits and per-stage dynamics. The unit-view round-trip test checks Isat, gm, G and C conversions. Spectre/ahdlcmi availability and Verilog-A simulation are environment dependent; these Python tests do not establish analog signoff or CTLE closure. Full hardware-preset toy round trips, consumption-based mapping coverage, and measured per-preset error bounds remain outstanding.
