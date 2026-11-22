# Spec: lstm-baseline-complexity

Scope: repo

# Spec: LSTM + complexity reporting for NARMA experiment

## LSTM Baseline Requirements

### 1. LSTMRegressor class
- Single-layer LSTM with 1D scalar input, configurable hidden_dim, batch_first=True
- Forward returns (predictions, final_state) for state carryover
- Matches user-provided snippet structure

### 2. train_and_eval_lstm() function
- Training: truncated BPTT with bptt_window=50, gradient clipping (max_norm=1.0), AdamW optimizer
- Evaluation: pass full test sequence as single batch (1, T, 1) with state carryover
- Discard washout=200 eval steps before computing NRMSE/R^2
- Grid search over hidden_dim=[16, 25, 32] and lr=[1e-3, 5e-4] using last 500 train samples as validation (same pattern as ESN grid search)
- Return {"nrmse": ..., "r2": ..., "n_params": ...}

### 3. Integration into run_baselines()
- LSTM is the 4th baseline condition (named "lstm") alongside "ridge", "esn", "mlp"
- Uses same narma() data, same train/test splits, same washout convention
- Results dict includes "n_params" for all four baselines

## Complexity Reporting Requirements

### 4. Per-method complexity in run_baselines() return
Each baseline returns a dict with at least: {"nrmse", "r2", "n_params"}.
- Ridge: n_taps weights + 1 bias = 31
- ESN: "26 trained / 651 total" — both numbers tracked
- MLP: sum(p.numel() for p in mlp.parameters()) = 801
- LSTM: sum(p.numel() for p in lstm.parameters()) = ~2826

### 5. Results table enhancement
The summary table in main() prints complexity for every condition alongside NRMSE and R^2.
- For ridge: "params=31"
- For esn: "params=26/651t" (trained/total)
- For mlp: "params=801"
- For lstm: "params=N"
- For fabric: already prints "params=N"

CSV output extended with additional columns for trained_params and total_params where applicable.