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

## Intended exporter interface

Target CLI: `python -m export.spectre_export --checkpoint ... [--hw-config ...] --mode legacy|hw --fabric timemux|copies --specs N --out DIR`. No exporter, strict state-dict mapping, JSON parameter table, or netlist interpreter has been completed in this change. Do not use the current modules for tapeout or claim model equivalence.

## Validation

The environment's non-interactive shell has no `spectre` or `ahdlcmi` on PATH (checked 2026-11-22). `iverilog` is present but is not a Verilog-A compiler. No Spectre transient, syntax compilation, interpreter-vs-PyTorch comparison, or CTLE closure has been run. Required next steps: source the Cadence setup as described in `spice_failure_check.py`, check executable/license; implement strict exporter and torch interpreter; run paper checkpoint legacy validation first; then a hardware-flagged toy; finish syntax/Spectre and full pytest validation. Task A wiring and hardware checkpoint validation remain pending.
