# CLARITY-SemEval-2026

### Decision-Level Late Fusion for Ambiguity Detection

![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)

---


## Overview

This repository implements a **decision-level ensemble (late fusion)** pipeline for the CLARITY (SemEval-2026) shared task:

* **Task 1:** Categorise an interview answer as *Clear Reply*, *Ambivalent*, or *Clear Non-Reply*.
* **Task 2:** Detect the *evasion technique* expressed by the answer (9 labels).

The final prediction layer fuses heterogeneous models that operate on the same input text:

1. **TF-IDF + Logistic Regression** – fast lexical baseline.
2. **OpenRouter Embedding Classifiers** – dense representations from OpenAI (`text-embedding-3-large`) and Qwen (`Qwen3-Embedding-8B`).
3. *(Optional)* **Zero-shot NLI (BART MNLI)** – enable in the config if you want an extra semantic expert. When enabled, predictions are cached under `experiments/cache/zero_shot/` so future runs reuse the scores.

Per-model probability vectors are concatenated and passed to a **meta logistic regression head**, yielding a calibrated decision-level ensemble.

---

## Quickstart

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/ecetsn/clarity-semeval-2026.git
   cd clarity-semeval-2026
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure secrets**
   ```bash
   cp .env.example .env
   # Fill OPENROUTER_API_KEY with the key you received from OpenRouter
   ```

3. **Run the late-fusion experiment**
   ```bash
   python src/cli/run_fusion.py --config config/ensemble.yaml
   ```

The script automatically downloads the QEvasion dataset from Hugging Face, trains all base models, fits the fusion head, and writes artefacts inside `experiments/` as:

* `metrics_experiment_<n>_label-<label>_models-<m1+m2>_res-<type>_seed-<seed>_tfidf-rep-<r>_<YYYYMMDD>.json`
* `prediction_test_experiment_<n>_label-<label>_models-<m1+m2>_res-<type>_seed-<seed>_tfidf-rep-<r>_<YYYYMMDD>.json`

Default configs cap the splits (≈900 train / 256 val / 256 test) so runs finish quickly on CPU. Set the `max_*_samples` fields to `null` once you are ready for full-dataset experiments.

---

## Configuration

Everything is driven by `config/ensemble.yaml`:

```yaml
dataset:
  dataset_name: ailsntua/QEvasion     # Hugging Face identifier
  label_column: clarity_label         # Use evasion_label to target Task 2
  text_column: interview_answer
  question_column: question
  validation_size: 0.2
  random_seed: 42
  calibration_fraction: 0.2           # portion of dev reserved for temperature scaling
  max_train_samples: null             # set to null for full-data runs
  max_val_samples: null
  max_test_samples: null

base_models:
  - name: tfidf_baseline
    type: tfidf_logreg
    params:
      ngram_range: [1, 2]
      max_features: 25000
  - name: openrouter_embedder
    type: openrouter_logreg
    params:
      model_name: text-embedding-3-large
      batch_size: 6
  - name: openrouter_qwen8b
    type: openrouter_logreg
    params:
      model_name: Qwen/Qwen3-Embedding-8B
      batch_size: 4
  # Uncomment to re-enable the cached zero-shot expert
  # - name: zero_shot_bart
  #   type: zero_shot
  #   params:
  #     model_name: facebook/bart-large-mnli
  #     batch_size: 4

fusion:
  type: logistic_regression
  params:
    C: 2.0

output_dir: experiments

paper_reference:
  description: "Llama-2-70b (evasion-based clarity) from Thomas et al., 2024"
  source: semval_2026_task6.pdf
  metrics:
    accuracy: 0.713
    precision: 0.743
    recall: 0.713
    macro_f1: 0.720
    weighted_f1: 0.752
```

To target the evasion taxonomy, simply change `label_column` to `evasion_label`. Further base models can be added by specifying their `type` (`tfidf_logreg`, `openrouter_logreg`, or `zero_shot`) and parameters.

---

## Components

| Path | Description |
| --- | --- |
| `src/data/datamodule.py` | Loads QEvasion from Hugging Face and composes question + answer text. |
| `src/models/tfidf_classifier.py` | TF-IDF features + multinomial logistic regression baseline. |
| `src/models/openrouter_classifier.py` | Fetches dense embeddings from OpenRouter and trains a logistic head. |
| `src/models/zero_shot.py` | Zero-shot classifier powered by Hugging Face's MNLI models. |
| `src/evaluation/late_fusion.py` | Coordinates base-model training, fusion fitting, metric computation, and artifact logging. |
| `src/utils/env.py` | Loads `.env` secrets (OpenRouter API key/url). |

---

## Outputs

Running the pipeline produces timestamped artifacts directly under `experiments/`, e.g.:

* `metrics_experiment_<n>_label-<label>_models-<m1+m2>_res-<type>_seed-<seed>_tfidf-rep-<r>_<date>.json` – validation/test accuracy, macro/weighted precision/recall/F1 for every base model and the fusion head. Filenames carry the label column, model list, resampling strategy, seed, and the tf-idf baseline replica count for easier tracking. If `paper_reference` is defined, the same file embeds the paper values for quick comparison.
* `prediction_test_experiment_<n>_label-<label>_models-<m1+m2>_res-<type>_seed-<seed>_tfidf-rep-<r>_<date>.json` – fused predictions with calibrated probabilities for each test sample.
* Optional zero-shot cache files under `experiments/cache/zero_shot/` so repeated runs can reuse previous MNLI scores.
* If `calibration_fraction > 0`, the metrics file also records the learned temperature used to calibrate the fusion logits on the held-out calibration slice.

## Running TF-IDF/Fusion sweeps

A lightweight grid runner lives at `src/cli/run_sweep_experiments.py` to sweep TF-IDF + fusion settings (ngram ranges, `min_df`, `max_features`, `c`, resampling strategies, and fusion `C`). Example:

```bash
python -m src.cli.run_sweep_experiments \
  --config config/ensemble.yaml \
  --output-dir experiments/sweeps \
  --replicas 5 \
  --max-runs 10
```

Defaults cover `(1,2)/(1,3)` ngrams, `min_df` in {2,3,5}, `max_features` in {20k, 30k, 40k}, TF-IDF `c` in {2,3,4}, resampling {over, under, none}, and fusion `C` in {1,2,4}. Use `--max-runs` to cap the grid if you want a quick scan.

These files are ready to submit/evaluate or to serve as inputs to downstream calibration/analysis notebooks. See `reports/comparison.md` for an example comparison against the SemEval baseline.

---

## Extending the Ensemble

* Add new base models in `config/ensemble.yaml` with their `type` and hyperparameters.
* Implement additional model wrappers under `src/models/` and register them in `src/models/factory.py`.
* Switch the `fusion.type` to `weighted_average` for quick experiments or keep `logistic_regression` for stacked generalisation.

---

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Secret key for OpenRouter's embedding endpoint. |
| `OPENROUTER_API_URL` | (Optional) Override for the embeddings URL. Defaults to `https://openrouter.ai/api/v1/embeddings`. |

Store them in `.env` (ignored by git) as shown in `.env.example`.

---

## Paper Reference & Metrics Alignment

We evaluate using the exact metrics described in *“I Never Said That”* (Thomas et al., 2024): accuracy, macro/weighted precision, macro/weighted recall, and macro/weighted F1. Each config’s `paper_reference` block stores the published scores (instruction-tuned Llama-2-70B with evasion-based clarity prompting), and every `metrics_experiment_*.json` automatically embeds those reference values for side-by-side comparison. The reference PDF is bundled as `semval_2026_task6.pdf`, and the latest comparison snapshot is maintained in `reports/comparison.md`.
