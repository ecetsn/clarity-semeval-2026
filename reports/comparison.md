# Experiment vs Paper Baseline

The latest fusion run (`experiments/prediction_test_experiment_1_base_20251213.json`) was evaluated with the same metrics defined in *"I Never Said That": A dataset, taxonomy and baselines on response clarity classification* (Thomas et al., 2024). The table below compares our current fusion stack against the strongest model reported in the paper (instruction-tuned Llama-2-70B with evasion-based clarity prompting).

| Metric | Ours – Fusion (Exp. 1, base) | Thomas et al. 2024 (Llama-2-70B) |
| --- | --- | --- |
| Accuracy | 0.666 | 0.713 |
| Macro Precision | 0.525 | 0.743 |
| Macro Recall | 0.404 | 0.713 |
| Macro F1 | 0.403 | 0.720 |
| Weighted F1 | 0.591 | 0.752 |

- Our figures are computed on the QEvasion test split using the saved predictions, mirroring the paper's accuracy/precision/recall/F1 reporting scheme (macro and weighted variants).
- Paper metrics are extracted from Table 8 in `semval_2026_task6.pdf` (the evasion-based clarity setup that achieved the best scores).

Re-running `python src/cli/run_fusion.py --config config/ensemble.yaml` will append a new `metrics_experiment_<n>_<base|zero-shot>_<date>.json` / `prediction_test_experiment_<n>_<...>.json` pair under `experiments/`; update this document after each run to keep the comparison in sync.
