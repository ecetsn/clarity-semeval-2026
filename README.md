# CLARITY-SemEval-2026

### Two-Stage Evasion-to-Clarity Modeling for Ambiguity Detection in Question–Answer Interactions

![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)

---

## Overview

This repository implements the full pipeline for the **CLARITY (SemEval-2026)** shared task:

* **Task 1:** Predict whether an interview response is a *Clear Reply*, *Ambiguous*, or *Clear Non-Reply*.
* **Task 2:** Detect the *evasion technique* used in the response (9-class taxonomy).

Our approach integrates **instruction-tuned large language models** (Llama/Mistral) with **linguistically guided preprocessing** to achieve high Macro-F1 across both subtasks.

It builds upon the benchmark introduced by [Papantoniou et al., 2024](https://arxiv.org/abs/2409.13879), extending it through:

* A **two-stage evasion→clarity pipeline** trained via LoRA fine-tuning.
* **Named-entity scrubbing** and **question decomposition** to reduce truncation bias.
* **Dual-pass prompting** (original vs grounded) for stable performance on long, multi-part inputs.
* **Ensemble calibration** for final Macro-F1 optimization.

---

## Repository Structure
```
clarity-semeval-2026/
│
├── config/               # YAML configs for tasks, LoRA, and ensemble
│   ├── task1_clarity.yaml
│   ├── task2_evasion.yaml
│   └── ensemble.yaml
│
├── data/                 # Dataset storage (raw / processed / cache)
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── src/
│   ├── data/             # Preprocessing, NE-scrubbing, question splitting
│   ├── prompts/          # Prompt templates and few-shot examples
│   ├── models/           # LoRA modules, fusion heads, encoders
│   ├── training/         # Training utilities and run scripts
│   ├── evaluation/       # Metrics, calibration, ensemble logic
│   └── utils/            # Logging, seeding, HF helper functions
│
├── notebooks/            # Data exploration and analysis notebooks
├── experiments/          # Logs, checkpoints, and results
└── reports/              # Figures and result summaries
```

---

## Installation

Clone the repository and set up the environment using Conda.
```bash
git clone https://github.com/ecetsn/clarity-semeval-2026.git
cd clarity-semeval-2026

conda env create -f environment.yml
conda activate clarity
```

---

## Data Preparation

1. Download the official dataset **QEvasion** from Hugging Face:
```python
   from datasets import load_dataset
   load_dataset("ailsntua/QEvasion")
```

2. Run preprocessing (named-entity scrubbing, question decomposition, and train/dev split):
```bash
   python src/data/preprocess.py
```

   This creates:
```
   data/processed/train_clarity.json
   data/processed/dev_clarity.json
   data/processed/train_evasion.json
   data/processed/dev_evasion.json
```

---

## Training

### 1. Task 2 – Evasion Technique Classification

Train the evasion detection model using LoRA fine-tuning.
```bash
python src/cli/run_lora.py --config config/task2_evasion.yaml
```

This produces a fine-tuned model under:
```
experiments/results_dev/evasion/checkpoint-best/
```

---

### 2. Task 1 – Clarity Classification

Use the evasion model's logits as auxiliary features for clarity prediction.
```bash
python src/cli/run_lora.py --config config/task1_clarity.yaml
```

This trains the clarity model via the **two-step evasion→clarity pipeline** and saves it under:
```
experiments/results_dev/clarity/checkpoint-best/
```

---

## Inference

Run deterministic decoding for evaluation on the development or test sets.
```bash
python src/evaluation/run_inference.py --task clarity --checkpoint experiments/results_dev/clarity/checkpoint-best/
```

Predictions are stored as:
```
experiments/results_dev/clarity/predictions.json
```

---

## Ensembling

Optionally combine multiple models to boost Macro-F1.
```bash
python src/evaluation/ensemble.py --config config/ensemble.yaml
```

Example `ensemble.yaml`:
```yaml
models:
  - experiments/results_dev/clarity/checkpoint-best
  - experiments/results_dev/evasion/checkpoint-best
  - experiments/results_dev/encoder_roberta
fusion_method: logistic_regression
metric: macro_f1
```

---

## Key Components

| Component               | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Dual-pass prompting** | Runs two inference passes per sample (original and NE-scrubbed/grounded).         |
| **Soft-label training** | Incorporates multiple annotator labels for evasion detection.                     |
| **Class-balanced loss** | Mitigates label imbalance in clarity categories.                                  |
| **Calibration**         | Temperature scaling for stable Macro-F1.                                          |
| **Slice evaluation**    | Evaluates performance separately for single vs multi-part, short vs long samples. |

---

## Evaluation Metric

The official competition metric is **Macro-F1**:

* Task 1: across 3 clarity classes.
* Task 2: across 9 evasion techniques (any annotator label considered correct).
