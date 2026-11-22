# Control-Path Equalization Experiment

**Status:** Active (depends on P1–P3 from NARMA plan — state-carryover, boundary mode, shared infrastructure)  
**Priority:** HIGH — this is your likely paper  
**Cost:** ~a few weeks  

---

## What this is

A simulated wireline link where a **conventional analog front-end stays in the data path** (parametric CTLE + VGA + optional 1-tap DFE), and your frozen-core KirchhoffNet replaces the **adaptation engine** that sets the control knobs.

The fabric never touches the baud-rate signal. It runs at adaptation bandwidth (kHz–MHz). The frozen core — your best, most manufacturable configuration — is sufficient.

---

## Prerequisites from NARMA plan

| Code | Needed for | Status |
|------|-----------|--------|
| P1: State-carryover | N/A — frozen core doesn't need sample-to-sample carryover | Optional |
| P2: Boundary mode | Feature extractors → boundary terminals | Done after NARMA |
| P3: Preset infrastructure | ctrl_eq preset pattern | Done after NARMA |

The main thing that carries over is boundary mode (P2) and the general experiment-script pattern from NARMA.

---

## System architecture

```
Channel → VGA(g1) → CTLE(g2, g3) → [DFE summer(g4)] → Slicer → data
                                        ↑
  analog feature extractors → KirchhoffNet (frozen core) → g1..g4
```

The differentiable link sim is built in PyTorch so gradients flow through the whole link → fabric.

---

## Implementation plan

### Step 1: Differentiable link simulation (`link_sim.py`)

**TX:**
- NRZ or PAM4, sps = 32 samples/UI
- Gaussian or raised-cosine bandlimit, -3 dB at 0.5–0.75× baud
- Optional TX jitter: phase-modulate symbol clock with ~0.01 UI rms random walk
- Optional TX nonlinearity: third-order compression (reserve as test case)

**Channel:**
- Frequency-domain loss: `IL(f) = a·√f + b·f` dB (skin + dielectric)
- Apply via FFT
- **Ensemble:** 50–200 channels with random a,b → 10–35 dB loss at Nyquist
- Temperature corners scaling the loss

**RX front-end (parametric):**

| Component | Knob | Range |
|-----------|------|-------|
| VGA | g₁ | [0.5, 2.0] |
| CTLE zero frequency | g₂ | parametric |
| CTLE DC gain | g₃ | parametric |
| 1-tap DFE | g₄ | subtrains previous symbol |

CTLE transfer function:
```
H(s) = (s + 2π·f_z1)(s + 2π·f_z2) / ((s + 2π·f_p1)(s + 2π·f_p2))
```

with f_z, f_p functions of g₂, g₃.

**Slicer + metrics:**
- Sample at symbol centre (or simple CDR model)
- Compute eye margin (vertical eye opening at sampling instant) and BER

### Step 2: Analog feature extractors

4–8 features, all ADC-free, continuous-time:

| Feature | How to measure | Why |
|---------|---------------|-----|
| Total received power | Rectify + RC lowpass | Signal strength baseline |
| HF power content | Signal − RC lowpass → rectify → lowpass | Channel-loss sensor — ratio encodes equalization needed |
| Edge rate | CDR bang-bang phase detector early/late | Transition density |
| Slicer error | Analog integrator accumulating |d − d̂| over a window | Adaptation steering signal |

These become boundary-terminal inputs to the fabric.

### Step 3: Frozen-core fabric network

- `enable_boundary=True`, `boundary_fan_out` from feature extractors to hidden nodes
- `freeze_read=True` (only boundary + temporal-readout OTA edges trainable)
- Temporal readout accumulators → linear → g₁..g₄ control voltages
- Outputs rail-limited by fabric's output compliance (no extra clamping needed)

### Step 4: Training

**Loss:** differentiable BER surrogate:
```
L = mean(softplus(-signed_distance_to_threshold / T))
```
or negative eye margin.

**Optimization:** backprop through entire link into fabric. Train across channel ensemble (random channel per batch element). This is a control/regression problem — structurally identical to the Friedman pipeline.

### Step 5: Baselines

| Baseline | What it represents | Implementation |
|----------|-------------------|----------------|
| Fixed nominal settings | No adaptation | Static g₁..g₄ mid-range |
| Golden per-channel calibration | Ceiling | Grid search over knobs per channel |
| SAR ADC + FSM + DACs | Incumbent | 8-bit 1 MS/s (~µW from Murmann) + LUT + DACs |
| Fabric (frozen core) | You | Same network as above |

### Step 6: PVT robustness

- Inject 1–5% mismatch into fabric via SimContext
- Recalibrate readout only (frozen-core calibration story)
- Show margin retention across corners

### Step 7: Metrics and win condition

| Metric | Target |
|--------|--------|
| Eye margin / BER across ensemble | Match golden calibration within small delta |
| Adaptation-engine power | Orders below ADC+FSM+DAC stack |
| Convergence / robustness | Works across 50–200 channels + PVT |

**Win condition:** match golden-calibration margin at adaptation-engine power orders below the incumbent, with a **no-ADC-at-all** story. Even a tie on margin with a cleaner power/PVT story is a paper.

---

## Script structure

```
link_sim.py             — differentiable TX/Channel/RX pipeline
  ├── TX               — NRZ/PAM4, bandlimit, jitter
  ├── Channel          — FFT-based loss, ensemble generation
  ├── RX               — VGA/CTLE/DFE param knobs
  └── Slicer           — eye margin, BER surrogate

ctrl_eq_experiment.py   — main entry point
  ├── build_fabric()    — frozen-core fabric from preset
  ├── generate_ensemble() — 50-200 random channels
  ├── train()           — backprop through link into fabric
  ├── evaluate()        — eye margin, BER across ensemble
  ├── baselines()       — fixed, golden, digital
  └── pvt_analysis()    — mismatch injection + recalibration
```

---

## Dependencies

- `link_sim.py` is standalone PyTorch (no external SerDes library needed)
- `ctrl_eq_experiment.py` imports from `link_sim.py`, `topology.py`, `kirchhoff_net.py`, `train.py`
- Channel ensemble is generated programmatically (random a,b)
- Eye diagrams via matplotlib
