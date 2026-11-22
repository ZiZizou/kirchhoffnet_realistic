---
plan name: update-comparison-report
plan description: Add 1200-epoch MLP rows to report
plan status: done
---

## Idea
Update generate_report.py so comparison_report.html includes BOTH the original 600-epoch MLP results (from C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb) AND the new 1200-epoch MLP results (from C:\Users\Atharva\Downloads\results_mlp_benchmarks_extra_epochs), keeping the 8 KNet rows unchanged. The 8 new MLP directories share identical names with the originals (e.g. friedman1_mlp_huber), so collect_results() must iterate both roots, tag each record by source, and disambiguate display labels (e.g. directory suffix " (1200 ep)"). Metrics stay sourced from the min-val row of loss_history.txt per existing methodology. Also update hardcoded notes (ratio baseline note "MLP = 600 epochs", methodology, meta Source line) to reflect dual MLP training budgets, and make the ratio baseline explicitly the 600-epoch MLP-Huber row rather than first-by-iteration-order. Regenerate the HTML with the venv python and verify 24 directories parsed (8 MLP-600 + 8 MLP-1200 + 8 KNet).

## Implementation
- Add EXTRA_EPOCHS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_mlp_benchmarks_extra_epochs") constant in generate_report.py near RESULTS_ROOT.
- Refactor collect_results() to iterate both roots (original first, extra second), tagging each record with a source root tag so identical MLP directory names from the two roots remain distinct.
- Add a display helper (e.g. dir_label(rec)) that appends an epoch-count suffix like ' (1200 ep)' to directory labels for records from the extra-epochs root, and apply it everywhere a directory is rendered (summary table, per-problem tables, Huber section, verification section).
- Update _render_val_loss_section() so prob_huber includes both MLP rows and the KNet/MLP ratio baseline is explicitly the 600-epoch MLP-Huber row (filter by epochs_run==600), not the first row by iteration order; update its ratio note text to mention MLP budgets of both 600 and 1200 epochs.
- Update the meta Source line in build_html() to list both results roots, and update _render_methodology_note() to document the dual MLP sources and that MLP metrics are verified min-val rows from each loss_history.txt.
- Regenerate the report with .\venv\Scripts\python.exe generate_report.py and verify: 24 directories parsed, two MLP rows per problem (600 + 1200), 8 KNet rows unchanged, and best-cell/ratio highlighting reflects the new rows.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
<!-- SPECS_END -->