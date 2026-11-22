---
plan name: lstm-baseline-complexity
plan description: Add LSTM, complexity, fix device bug
plan status: done
---

## Idea
Add an LSTM recurrent baseline to the NARMA experiment, add a large-capacity variant of both MLP and LSTM matched to KNet's ~7,600 params for fair comparison, add complexity reporting (params/nodes) to all method outputs, and fix the device mismatch bug in memory_capacity(). The LSTM uses truncated BPTT with state carryover, matching the fabric's streaming inference pattern. Both small (25 hidden) and large (237 MLP / 43 LSTM hidden) variants are included so the user can decide which comparison is fairest.

## Implementation
- Fix device bug in run_fabric_condition(): add u_train = u_train.to(device) before memory_capacity() call
- Add LSTMRegressor class: single-layer LSTM, batch_first=True, scalar input, state carryover
- Add train_and_eval_lstm() with truncated BPTT, grad clipping, grid search (hidden_dim=[16,25,32], lr=[1e-3,5e-4])
- Integrate LSTM as 4th baseline condition 'lstm' in run_baselines(), using same narma() data splits and washout=200
- Add MLP_large with hidden_dim=237 (7,585 params ≈ KNet) and LSTM_large with hidden_dim=43 (7,784 params ≈ KNet) as conditions 'mlp_large' and 'lstm_large'
- Add complexity reporting: n_params to all baseline dict returns; print params alongside NRMSE/R^2 in summary table
- Extend CSV output with trained_params, total_params, hidden_dim columns
- Ensure NARMA-20 gating on decision rule is preserved and LSTM conditions don't participate in it
- Run syntax check: python -c "import ast; ast.parse(open('narma_experiment.py').read())"

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- lstm-feature-implementation
<!-- SPECS_END -->