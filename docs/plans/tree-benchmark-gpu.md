---
plan name: tree-benchmark-gpu
plan description: GPU auto-detect for tree benchmark
plan status: done
---

## Idea
Modify tree_benchmark.py to automatically use GPU when available (target: Kaggle, which has GPUs; local machine has none). Add a --device auto|cpu|cuda CLI flag (default auto, matching mlp_benchmark.py:267 / kn_benchmark_friedman2.py:542 convention). GPU-capable methods: xgboost (device='cuda' + tree_method='hist'), lightgbm (device='gpu' with graceful CPU fallback since pip builds usually lack GPU), catboost (task_type='GPU'). sklearn methods (gradient_boosting, hist_gradient_boosting, random_forest, extra_trees) have no GPU support — log a note and run on CPU when a GPU is requested/present. Device detection via torch.cuda.is_available() (torch already imported at tree_benchmark.py:81). Record the resolved device in config_snapshot.txt and final_metrics.txt.

## Implementation
- Add --device CLI arg (choices auto/cpu/cuda, default auto) to _build_parser() in tree_benchmark.py near the seed arg, with help text explaining auto-detection and per-method GPU support.
- Add a device-resolution helper (e.g. _resolve_device(device_choice) -> str 'cpu'|'cuda') in tree_benchmark.py: returns 'cuda' if choice=='cuda', 'cpu' if choice=='cpu', and auto = 'cuda' if torch.cuda.is_available() else 'cpu'. Print resolved device in main() header line.
- Modify _train_xgboost: add device to params dict as device='cuda' when gpu_available else 'cpu', keeping tree_method='hist'. Pass gpu_available (or device string) into trainer via args/parameter.
- Modify _train_lightgbm: add device='gpu' when gpu requested, wrapped in try/except around lgb.train so a build without GPU support falls back to device='cpu' with a printed warning (retry train on CPU).
- Modify _train_catboost: add task_type='GPU' (+ devices='0') when gpu requested, with try/except around CatBoostRegressor fit to fall back to task_type='CPU' with a warning.
- Handle sklearn methods: when resolved device is cuda, print a note in each _train_sklearn_* trainer (or once in main) that the method has no GPU support and will run on CPU.
- Record resolved device in config_snapshot.txt (device: line) and final_metrics.txt (device: line) via the existing args/extra mechanism.
- Verify: py_compile tree_benchmark.py, then run a quick smoke test on CPU (e.g. .\venv\Scripts\python.exe tree_benchmark.py --method xgboost --dataset friedman1 --n-estimators 5 --validate-every 1) to confirm nothing breaks and device=cpu is recorded.
- Verify GPU path cannot be tested locally (no GPU) — confirm --device cuda on a GPU-less machine still resolves to cpu only via torch.cuda.is_available()==False OR errors cleanly if explicit cuda requested; decide: explicit --device cuda with no GPU should warn and fall back to cpu (graceful) rather than crash.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
<!-- SPECS_END -->