# Clarity–SemEval 2026 (Group 2) — Representation-Fusion Branch (İhsan)

This branch implements a **late/representation-level fusion** system for the QEvasion benchmark (SemEval 2026 Task: Clarity & Evasion). It includes:
- A **YAML-driven ablation pipeline** (single entrypoint) for controlled experimentation.
- A **final, single-script run** for training on the official train split and evaluating on the official test split (including the test split’s special evasion labeling scheme).

---

## 1) Repository layout (what matters)

### Experiments & artifacts
All runs write into `PROJECT_ROOT/experiments/`.

- `PROJECT_ROOT/experiments/logs/`
  - Timestamped run logs for each experiment (`.log`)
  - Use these to debug configuration, dataset splits, training dynamics, truncation rates, etc.

- `PROJECT_ROOT/experiments/representation/`
  - Each experiment has its own folder containing:
    - `config.yaml` or copied config snapshot
    - `summary.json` (includes `best_macro_f1`, `best_epoch`, etc.)
    - `val_epoch_<k>.json` (per-epoch evaluation outputs: macro/weighted metrics, per-class breakdown, confusion matrix, debug heads)
    - model checkpoints (e.g., `best.pt`)

> Convention used in analysis: **use `summary.json -> best_macro_f1` to select which `val_epoch_k.json` is the “best run”** for reporting.

---

## 2) Core code (where to look)

Everything relevant to the representation-fusion pipeline lives under:

- `PROJECT_ROOT/src/representation/`

Key subfolders:
- `src/representation/training/`
  - `train_from_yaml.py`  
    Main ablation runner: reads a YAML config, builds models, trains, evaluates per epoch, saves `summary.json`, `val_epoch_k.json`, checkpoints, and logs.

- `src/representation/final_script/`
  - `final_script.py`  
    Final-run entrypoint: trains on **official QEvasion train split** and evaluates on **official QEvasion test split**, writing all outputs to:
    - `PROJECT_ROOT/experiments/representation/FINAL_RUN/`

- `src/representation/configs/`
  - All experiment YAMLs used in ablations (grouped by family/run_group).

---

## 3) Dataset assumptions (QEvasion specifics)

Dataset: `ailsntua/QEvasion` from Hugging Face.

Splits:
- `train` (≈ 3.4k rows): contains both `clarity_label` and `evasion_label`.
- `test` (308 rows):
  - has `clarity_label`
  - **does not provide a single `evasion_label`**; instead it provides `annotator1`, `annotator2`, `annotator3` (each is a valid ground truth)

### How we evaluate evasion on test
We report:
1) Annotator-wise metrics: evaluate predictions against each annotator column separately  
2) Any-match accuracy: a prediction is correct if it matches **any of the 3 annotators**

This matches the task FAQ: any annotator response is considered correct.

---

## 4) How to run

### 4.1 Run ablation experiments (YAML-driven)
From `PROJECT_ROOT`:

```bash
python -m src.representation.training.train_from_yaml src/representation/configs/<family>/<experiment>.yaml
