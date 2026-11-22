# Signal-Path Equalization Experiment

**Status:** Contingent — ONLY proceed if NARMA-20 shows `freeze_read=False` clearly beats `freeze_read=True` (NRMSE ~0.1–0.2 vs ~0.3+)  
**Priority:** MEDIUM (blocked on NARMA)  
**Cost:** ~weeks  

---

## Contingency

| NARMA outcome | Action |
|--------------|--------|
| freeze_read ≈ no-freeze | **STOP.** Evolving core is dead. Do not proceed. |
| no-freeze clearly wins | Proceed. The evolving nonlinear core has computational value. |

---

## What this is

Replace the CTLE directly in the signal path with the evolving-core fabric. The fabric's state runs free across symbols (state-carryover, no resets). The continuous-time filtering IS the equalization — analogous to how nobody asks whether a CTLE "settles within 1 UI."

```
RX waveform (post fixed VGA) → fabric boundary terminals → fabric state (free-running) → output → slicer
```

---

## Prerequisites

| Code | From | Status |
|------|------|--------|
| P1: State-carryover | NARMA plan | Must be working |
| P2: Boundary mode | NARMA plan | Must be working |
| P4: Link sim (`link_sim.py`) | Control-path plan | Can reuse directly |
| P5: Time-constant rescaling | This experiment | New — see below |

---

## Implementation plan

### Step 1: Time-constant rescaling

The current config has `leak ≈ 0.0486` and `t_span = 7.0` → dominant τ ~20 time units. For a signal-path fabric, τ must be ~0.5–5 UI.

**What to rescale:**

| Parameter | Current | Target range |
|-----------|---------|-------------|
| `INIT.leak_constant` | 0.0486 | 0.2–2.0 (faster decay) |
| `gm` ranges | 0.2–3.0 | 1.0–10.0 (higher transconductance) |
| `C_eff` | 1.0 | 0.1–0.5 (lower capacitance) |
| `t_span` per UI | 7.0 | Scale to match UI × steps |
| `num_steps` per UI | 70 | Keep or reduce |

Sanity-check implied `gm/C` against target process node.  
**Honest baud-rate claim: 2–20 GBd**, not 100 GBd. Document the cap.

Config changes in `config.py`: add a `SIGNAL_PATH` scaling dict that multiplies leak, gm, C_eff.

### Step 2: Network as cascaded physical stages

**Stages = physical CTLE sections, not time-multiplexing.** 3–5 stages with independent parameters.

Each stage:
- Independent `t_span`, `num_steps`, `leak_floor`
- Independent cell library parameters (gm, Isat per edge)
- Boundary input from the RX waveform (post VGA)
- Output drives the next stage input + the slicer

This is exactly how real multi-stage CTLEs are built. The existing multi-stage `KirchhoffNet` infrastructure already supports this — just configure each stage with its own parameters.

### Step 3: Link pipeline

Reuse `link_sim.py` from control-path experiment. The only difference:
- Instead of feature extractors → fabric → knobs, the RX waveform (post fixed VGA) drives the fabric boundary terminals directly
- Fabric state runs free across symbols (state-carryover, no resets)
- Fabric output node(s) drive the slicer directly
- No CTLE block in the RX (fabric replaces it)

### Step 4: Evaluation

**BER vs SNR curves** for:
| Condition | Note |
|-----------|------|
| No equalizer | The floor — raw channel ISI |
| Optimized fixed CTLE | The ceiling — golden per-channel CTLE tuning |
| Fabric (evolving core) | The experiment — one fabric per channel, trained |
| Fabric (one-shot) | Single fabric trained across ensemble, no per-channel tuning |

**Per-channel adaptability:** can one fabric instance handle multiple channel conditions without recalibration? This is the key advantage over a fixed CTLE.

### Step 5: Power analysis

Use Murmann ADC FoM data to estimate ADC+DSP power at the claimed baud rate (2–20 GBd).  
Estimate fabric power from cell library parameters:
```
P_fabric ≈ Σ_e (m_e · V_dd · I_sat_q · p(e,q)) per stage  
         + Σ_node (V_dd² · g_leak_j)
```

| Component | Power estimate | Notes |
|-----------|---------------|-------|
| Fabric (this work) | From cell library params | Per-edge gm, Isat, multiplicity |
| ADC (alternate) | Murmann FoM | Resolution × baud rate |
| DSP (alternate) | Gate count × energy/gate | FFE/DFE tap count |
| **Delta** | **Fabric − (ADC+DSP)** | Must be negative to win |

### Step 6: Decision gate

| Outcome | Interpretation |
|---------|---------------|
| Fabric beats fixed CTLE on per-channel adaptability | Signal-path has value — evolve core can adapt where CTLE cannot |
| Fabric matches CTLE at comparable power | Neutral — dynamic equalization without ADC is still interesting |
| Fabric loses to CTLE | Signal-path branch dead even if NARMA was positive |

---

## Script structure

Reuses `link_sim.py` from control-path plan. New files:

```
sig_path_experiment.py   — main entry point
  ├── build_fabric()      — 3-5 stage cascaded fabric with independent params
  ├── train_fabric()      — backprop through link into fabric
  ├── evaluate_ber_snr()  — sweep SNR, compute BER for each condition
  ├── power_estimate()    — fabric power from cell library params
  └── main()              — run all conditions, plot curves
```

Config additions in `config.py`:
- `SIGNAL_PATH` — time-constant rescaling multipliers
- Signal-path preset with per-stage parameter overrides

---

## Relationship to other plans

```
01-narma-discriminator  ──provides──►  state-carryover (P1)
                                        boundary mode (P2)
                                        experiment pattern
                       ──decides──►    whether this plan runs at all

02-control-path-equalization ──provides──►  link_sim.py (P4)
                                             channel ensemble tools
                                             evaluation infrastructure

03-signal-path-equalization ──uses──►  P1, P2, P4
                                       adds time-constant rescaling (P5)
```
