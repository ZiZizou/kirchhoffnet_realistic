# Spec: narma-experiment-fixes

Scope: feature

# Spec: narma-experiment-fixes

Scope: feature

# NARMA Experiment Fix Requirements

## P0 — Result-altering bugs

### 1. Apply input scaling to fabric data pipeline
`scale_input_to_rails()` is defined at `narma_experiment.py:104` but **never called**. Both `run_baselines()` and `run_fabric_condition()` pass raw narma output (range [0, 0.5]) to the fabric, making the boundary OTA current 6x weaker than the spec's `V_u(n) = 3.0 * u(n)/0.5 V`.

**Fix:** Call `u = scale_input_to_rails(u)` on both train and test splits before any fabric operation.

### 2. Increase bptt_window to 50 (configurable via CLI)
Spec says "Backprop through ~50–100 samples of history" but code uses `bptt_window=15`. With only 15-step context, the network cannot learn NARMA-20's 20-step memory.

**Fix:** Default `bptt_window=50`. Add `--bptt-window` CLI argument.

### 3. Memory Capacity targets u not y
`run_fabric_condition()` at line 577 passes `y_train` to `memory_capacity()`. Spec says "reconstruct u(n-k) for k=1..20".

**Fix:** Change to `memory_capacity(states_tr, u_train, ...)`.

## P1 — Code quality / correctness

### 4. Remove dead parameters
`run_fabric_condition()` signature has `hidden_dim` and `num_steps_per_sample` parameters that are never used. The actual values come from the preset.

**Fix:** Remove both parameters from the signature.

### 5. Gate decision rule on NARMA-20
Spec says the pre-registered rule applies to NARMA-20 specifically. Currently runs for both orders.

**Fix:** Add `if args.order == 20` guard around the decision logic block.

### 6. Add absolute-NRMSE check to decide()
`decide()` only checks the difference between conditions, ignoring that both might have poor absolute NRMSE. Spec implies COMMIT requires reasonable performance from both.

**Fix:** If `fr_off_mean > 0.5`, return "NEUTRAL" (neither condition is working well enough to draw conclusions).

## P2 — Polish

### 7. Fix docstring (narma_curves.png)
Docstring at line 46 promises a PNG output but no plotting code exists.

**Fix:** Remove the line from the docstring.

### 8. ESN hyperparameter grid search
Spec says "Input scaling, leak rate tuned" but the ESN uses fixed values.

**Fix:** Grid search over `input_scaling=[0.2, 0.5, 1.0]` and `leak=[0.3, 0.5, 1.0]`, selecting the combination with best validation NRMSE (last 500 of train set).