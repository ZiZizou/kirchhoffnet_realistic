---
plan name: narma-tanh-free
plan description: Run NARMA-20 full comparison with KNet tanh_free
plan status: done
---

## Idea
Add --cell-library CLI flag to narma_experiment.py to enable tanh_free cell class for fabric conditions. Run full comparison (baselines + fabric) on GPU with torch.compile for NARMA-20, 200 epochs, 4 streams, 2500 samples, freeze_read=False (evolving core only), seeds 0,1,2 on Kaggle.

## Implementation
- Add --cell-library CLI argument to parse_args() with choices matching CELL_LIBRARIES keys (tanh, tanh_free, tanh_realistic, tanh_realistic_upgrade, tanh_anti, relu)
- Modify run_fabric_condition() signature to accept cell_library parameter
- Update run_fabric_condition() to use passed cell_library instead of hardcoded 'tanh'
- Update main() to pass args.cell_library to run_fabric_condition calls
- Add optional --freeze-read flag to control which freeze_read conditions to run (default: both OFF/ON for backward compat, but allow --freeze-read false for evolving core only)
- Verify syntax with py_compile
- Run on Kaggle GPU with: python narma_experiment.py --order 20 --device cuda --compile --cell-library tanh_free --freeze-read false --epochs 200 --n-streams 4 --train-samples 2500 --seeds 0,1,2 --output ./output/narma20_tanh_free

## Required Specs
<!-- SPECS_START -->
- narma-experiment
- narma-experiment-fixes
- narma-tanh-free
<!-- SPECS_END -->