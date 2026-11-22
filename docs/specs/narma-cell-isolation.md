# Spec: narma-cell-isolation

Scope: feature

# Feature Spec: NARMA Cell Isolation (plan narma-node-activation)

## Purpose
Guarantee every symbol introduced by the mix-first bounding repair (node-activation flag, spectral-norm util, new cells, new topologies, NARMA CLI options) is unreachable from the static-regression training path (`train_script.py`, `kn_bayes_opt.py`, friedman/housing benchmarks). The NARMA fabric is the experimental substrate; the CTLE/static path must not shift behavior, param counts, or checkpoint compatibility.

## Contract

### 1. Flag storage (default-off, param-neutral)
- `node_activation` (`{"none","tanh","identity"}`, default `"none"`) and any clip-bypass are plain attributes / kwargs, never `nn.Parameter`. Default construction yields a byte-identical forward and identical `state_dict` keys vs pre-change.
- `normalize_spectral_radius` lives in a NARMA-side module (never `topology.py` / `config.py`); it mutates a built net in place only when explicitly called.

### 2. Factory closure
- `make_cell_library` gains no new names; `CELL_LIBRARIES` gains no entries.
- New cells (`LinearOTA`, `DeadZoneOTA`, `AsymmetricOTA` in `narma_cells.py`) are `FreeTanhLibrary`-signature-compatible (`forward(x_src, x_dst, x_max)`, optional `resistive_current` / `forward_tanh`) and imported only by `narma_experiment.py`, `narma_advisor_probes.py`, `narma_linear_controls.py`, and the node-activation audit. Never by `train_script.py`, `kn_bayes_opt.py`, or benchmarks.

### 3. Builder guard
- `build_net_from_config` / `make_narma_preset` accept experimental keys (`node_activation`, `narma_cell`, `narma_topology`, clip-bypass) only when `allow_experimental_cells=True`, which solely the NARMA `_build_fabric_net` path passes. Any other caller passing them gets `ValueError`.
- `train_script.py` receives no new CLI flags and never passes the kwarg, so it cannot opt in by accident.

### 4. Topology closure
- New constructors (`esn_sparse_random`, `hierarchical_multiscale`, `random_dag`) live in `narma_topologies.py`; `topology.py` primitives and the `hidden_family` string namespace are unchanged. `small_world k=6` is a parameter value, not new code.

### 5. Audit invariants (`probe_audit_nodeact.py`)
- (a) Default train-path build before/after the change is forward-identical to 1e-6 on a fixed seed (param count identical).
- (b) Passing any experimental key through the train path (without the guard kwarg) raises `ValueError`.
- (c) `make_cell_library` rejects the new cell names with `ValueError`.
- (d) Epoch-0 equality: `node_activation='tanh'` at gm=isat=0 matches the legacy baseline to 1e-6 (small-signal `tanh(x)≈x`); `'none'` vs `'identity'` identical at small signal.

## Non-goals
- No `cell_library.py` default edits; no `CELL_LIBRARIES` additions.
- No `train_script.py` / BO / benchmark CLI or behavior changes.
- No training-time spectral adaptation (init-only hook).