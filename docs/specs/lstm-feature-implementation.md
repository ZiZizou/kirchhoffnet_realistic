# Spec: lstm-feature-implementation

Scope: feature

# FEATURE: LSTM + complexity reporting

## Implementation Plan

### Step 1: Fix device bug (run_fabric_condition, line ~669)
Change:
```python
mc_per_delay, mc_total = memory_capacity(
    states_tr, u_train, max_delay=20, ridge_l2=1e-2,
)
```
To:
```python
u_train = u_train.to(device)
mc_per_delay, mc_total = memory_capacity(
    states_tr, u_train, max_delay=20, ridge_l2=1e-2,
)
```

### Step 2: Add LSTMRegressor class
Insert after MLPRegressor class (~line 243):
```python
class LSTMRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 25) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, u_seq, state=None):
        if u_seq.dim() == 2:
            u_seq = u_seq.unsqueeze(-1)
        out, final_state = self.lstm(u_seq, state)
        preds = self.fc(out).squeeze(-1)
        return preds, final_state
```

### Step 3: Add train_and_eval_lstm() function
After make_mlp_features (~line 252), add the training function:
- Accepts: order, seed, hidden_dim, bptt_window=50, epochs=200, washout=200, lr=1e-3
- Training: truncated BPTT, random window starts, grad clipping max_norm=1.0
- Evaluation: full sequence, washout discard
- Returns: {"nrmse", "r2", "n_params", "hidden_dim"}

### Step 4: Modify run_baselines() to include all baselines
Extended signature or internal logic:
- Keep existing ridge/esn/mlp logic (unchanged)
- Add LSTM with grid search (hidden_dim=[16,25,32], lr=[1e-3,5e-4]) → condition "lstm"
- Add MLP_large with hidden_dim=237 → condition "mlp_large"
- Add LSTM_large with hidden_dim=43, grid search lr only → condition "lstm_large"
- All baselines return dict with nrmse, r2, n_params (and hidden_dim for LSTM variants)
- ESN returns: nrmse, r2, trained_params=26, total_params=651

### Step 5: Update main() summary table
Each condition line prints e.g.:
```
         ridge  NRMSE=0.4231  R^2=0.8210  params=31
           esn  NRMSE=0.3512  R^2=0.8766  params=26t/651T
           mlp  NRMSE=0.2813  R^2=0.9201  params=801
     mlp_large  NRMSE=0.2651  R^2=0.9288  params=7585
          lstm  NRMSE=0.1856  R^2=0.9651  params=2726 (hid=25)
    lstm_large  NRMSE=0.1621  R^2=0.9733  params=7784 (hid=43)
 fabric_fr_off  NRMSE=0.2134  R^2=0.9538  MC=8.23  params=7601
  fabric_fr_on  NRMSE=0.2876  R^2=0.9166  MC=7.41  params=7601
```

### Step 6: CSV output
Columns: condition, seed, nrmse, r2, mc_total, trained_params, total_params, hidden_dim

### Step 7: Verify
- `import ast; ast.parse(open('narma_experiment.py').read())` must pass
- NARMA-20 gating on decision rule preserved
- LSTM conditions excluded from decision rule