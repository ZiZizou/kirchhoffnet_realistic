# Spec: ctrl-path-sim

Scope: feature

# Control-Path Equalization Experiment Protocol

## System Architecture

A simulated wireline link where a conventional analog front-end stays in the data path (parametric CTLE + VGA + optional 1-tap analog DFE, all with control-voltage knobs), and the frozen-core KirchhoffNet replaces the adaptation engine that sets those knobs:

```
Channel → VGA(g1) → CTLE(g2, g3) → [DFE summer(g4)] → Slicer → data
                                        ↑
  analog feature extractors → KirchhoffNet (frozen core) → control voltages g1..g4
```

The fabric never touches the baud-rate signal. It runs at adaptation bandwidth (kHz–MHz), which is why the frozen core is sufficient.

## Link Simulation (Differentiable PyTorch Pipeline)

### TX
- NRZ or PAM4, sps = 32 samples/UI
- Bandlimit with Gaussian or raised-cosine filter, -3 dB at 0.5–0.75× baud
- Optional TX jitter: phase-modulate symbol clock with ~0.01 UI rms random walk
- Optional TX nonlinearity: third-order compression (reserve for test case where multiplier branch earns its keep)

### Channel
- Frequency-domain loss model: IL(f) = a*sqrt(f) + b*f dB (skin + dielectric)
- Apply via FFT
- **Ensemble**: 50–200 channels with a, b randomized to give 10–35 dB loss at Nyquist
- Temperature corners that scale the loss

### RX Front-End with Knobs
- VGA: gain g1 in [0.5, 2]
- CTLE: two-zero/two-pole filter with peaking knob g2 (zero frequency) and DC-gain knob g3
  - H(s) = (s + 2*pi*f_z1)(s + 2*pi*f_z2) / ((s + 2*pi*f_p1)(s + 2*pi*f_p2))
  - f_z, f_p are functions of the knobs
- Optional 1-tap DFE: subtract g4 * d_{n-1} before slicing

### Slicer + Metrics
- Sample at symbol center (or with simple CDR model)
- Compute eye margin (vertical eye opening at sampling instant) and BER

## Analog Features (ADC-free, Continuous-time)

1. **Total received power**: rectify + RC lowpass (one diode + one RC)
2. **High-frequency power content**: signal minus its own RC lowpass, rectify + lowpass (classic analog channel-loss sensor — ratio encodes how much equalization is needed)
3. **Edge rate / transition density**: from CDR's bang-bang phase detector early/late outputs (standard in any real receiver)
4. **Slicer error magnitude**: analog integrator accumulating |d - d_hat| over a window (needs slicer, which exists anyway)

4–8 features total. These become boundary-terminal inputs.

## Fabric Outputs
- g1 (VGA gain), g2 (CTLE peaking), g3 (CTLE DC gain), g4 (DFE tap weight)
- Rail-limited — fabric's output compliance already enforces this

## Training

### Loss
- Differentiable BER surrogate: softplus(-signed distance to threshold / T)
- Or negative eye margin (mean over slicer instants)

### Optimization
- Backprop through entire link into the fabric
- Train across channel ensemble (random channel per batch element)
- Control/regression problem — structurally identical to Friedman pipeline

### Frozen Core Configuration
- `freeze_read=True`
- Boundary-fan-out from features
- Temporal readout accumulators for output integration
- Only boundary and temporal-readout OTA edges trainable

## Baselines

| Baseline | What it represents |
|---|---|
| Fixed nominal settings (no adaptation) | The floor |
| Golden per-channel exhaustive calibration (grid search) | The ceiling |
| Digital: 8-bit 1 MS/s SAR ADC (~uW) + LUT/FSM + DACs | Incumbent adaptation engine |
| Fabric (frozen core) | The proposed solution |

## Evaluation Metrics
- Eye margin / BER across the channel ensemble
- Power overhead of adaptation engine itself (USP story)
- Convergence/robustness across corners
- PVT: inject 1–5% mismatch into fabric, recalibrate readout only, show margin retention

## Win Condition
- Match golden-calibration margin within small delta
- Adaptation-engine power orders below ADC+FSM+DAC stack
- No-ADC-at-all story
- Even a tie on margin with cleaner power/PVT story is a paper

## Files to Create
- `link_sim.py`: the differentiable SerDes link simulation pipeline
- `ctrl_eq_experiment.py`: training and evaluation script
- Channel ensemble generation utilities
- Analog feature extractor functions