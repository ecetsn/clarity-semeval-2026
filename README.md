# SemEval-2024 Task 1: Question-Answer Evasion Detection

Machine learning pipeline for detecting question-answer evasion strategies using 25 Context Tree features extracted from multiple transformer models. This repository implements three evaluation methodologies for the SemEval-2024 Task 1 competition.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [The 25 Context Tree Features](#the-25-context-tree-features)
4. [Three Evaluation Methodologies](#three-evaluation-methodologies)
5. [Project Structure](#project-structure)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Results and Storage Locations](#results-and-storage-locations)
8. [Core Modules](#core-modules)
9. [Usage](#usage)
10. [Requirements](#requirements)

---

## Overview

This project implements a pipeline for SemEval-2024 Task 1: Question-Answer Evasion Detection. The pipeline extracts 25 Context Tree features from question-answer pairs using six transformer models and performs classification on two main tasks:

### Tasks

1. **Clarity Task** (3 classes):
   - `Clear Reply`: The answer directly addresses the question
   - `Ambivalent`: The answer is ambiguous or partially addresses the question
   - `Clear Non-Reply`: The answer does not address the question

2. **Evasion Task** (9 classes):
   - `Claims ignorance`: "I don't know"
   - `Clarification`: Asking for clarification
   - `Declining to answer`: Explicit refusal
   - `Deflection`: Redirecting to another topic
   - `Dodging`: Avoiding the question
   - `Explicit`: Direct answer
   - `General`: Vague or general response
   - `Implicit`: Indirect answer
   - `Partial/half-answer`: Incomplete answer

### Models

- **BERT** (`bert-base-uncased`)
- **BERT-Political** (`bucketresearch/politicalBiasBERT`)
- **BERT-Ambiguity** (`Slomb/Ambig_Question`)
- **RoBERTa** (`roberta-base`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **XLNet** (`xlnet-base-cased`)

### Classifiers

- **LogisticRegression** (with StandardScaler)
- **LinearSVC** (with StandardScaler)
- **RandomForest**
- **MLP** (Multi-Layer Perceptron, with StandardScaler)
- **XGBoost**
- **LightGBM**

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for transformer models)
- Google Colab account (for Drive integration) or local setup with Google Drive API

### Setup Instructions

#### Option 1: Google Colab (Recommended)

1. Open Google Colab and create a new notebook
2. Clone the repository:
   ```python
   !git clone https://github.com/EonTechie/semeval-context-tree-modular.git
   ```
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install dependencies (each notebook includes installation cells):
   ```python
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install transformers datasets scikit-learn pandas numpy xgboost lightgbm tqdm matplotlib seaborn
   ```

#### Option 2: Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/EonTechie/semeval-context-tree-modular.git
   cd semeval-context-tree-modular
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers datasets scikit-learn pandas numpy xgboost lightgbm tqdm matplotlib seaborn
   ```

4. Configure Google Drive API (if using local storage):
   - Follow [Google Drive API setup guide](https://developers.google.com/drive/api/quickstart/python)
   - Place credentials in the project root

### Dependencies

The project requires the following Python packages:

- **PyTorch** (>=1.12.0): Deep learning framework for transformer models
- **transformers** (>=4.20.0): HuggingFace transformers library
- **datasets** (>=2.0.0): HuggingFace datasets library
- **scikit-learn** (>=1.0.0): Machine learning classifiers and utilities
- **pandas** (>=1.3.0): Data manipulation
- **numpy** (>=1.21.0): Numerical computations
- **xgboost** (>=1.6.0): XGBoost classifier
- **lightgbm** (>=3.3.0): LightGBM classifier
- **tqdm** (>=4.64.0): Progress bars
- **matplotlib** (>=3.5.0): Plotting
- **seaborn** (>=0.12.0): Statistical visualizations

---

## The 25 Context Tree Features

The pipeline extracts 25 Context Tree features from each question-answer pair. These features are divided into 7 model-dependent features (require transformer model) and 18 model-independent features (text-based only).

### Model-Dependent Features (7 features)

These features require a transformer model to extract attention patterns and tokenization information.

#### 1. `question_model_token_count`
- **Description**: Number of tokens in the question after tokenization by the model
- **Formula**: `count(question_tokens)` where tokens are model-specific (e.g., BERT uses WordPiece)
- **Range**: Integer ≥ 0

#### 2. `answer_model_token_count`
- **Description**: Number of tokens in the answer after tokenization
- **Formula**: `count(answer_tokens)`
- **Range**: Integer ≥ 0

#### 3. `attention_mass_q_to_a_per_qtoken`
- **Description**: Average attention mass from question tokens to answer tokens, normalized by question token count
- **Formula**: 
  ```
  attention_mass_q_to_a_per_qtoken = Σ(attention[q_i, a_j]) / |Q|
  ```
  where `attention[q_i, a_j]` is the attention weight from question token `q_i` to answer token `a_j`, summed over all question-answer token pairs, and `|Q|` is the number of question tokens.
- **Range**: [0, 1]

#### 4. `attention_mass_a_to_q_per_atoken`
- **Description**: Average attention mass from answer tokens to question tokens, normalized by answer token count
- **Formula**:
  ```
  attention_mass_a_to_q_per_atoken = Σ(attention[a_i, q_j]) / |A|
  ```
  where `attention[a_i, q_j]` is the attention weight from answer token `a_i` to question token `q_j`, and `|A|` is the number of answer tokens.
- **Range**: [0, 1]

#### 5. `focus_token_to_answer_strength`
- **Description**: Average maximum attention strength from focus tokens (top-k most central question tokens) to answer tokens
- **Formula**:
  ```
  focus_tokens = top_k(centrality_score, k=min(8, |Q|))
  centrality_score[q_i] = Σ(attention[q_i, :]) + Σ(attention[:, q_i])
  focus_token_to_answer_strength = mean(max(attention[focus_i, a_j] for all a_j))
  ```
  where focus tokens are the top-8 most central question tokens (by incoming + outgoing attention).
- **Range**: [0, 1]

#### 6. `answer_token_to_focus_strength`
- **Description**: Average maximum attention strength from answer tokens to focus tokens
- **Formula**:
  ```
  answer_token_to_focus_strength = mean(max(attention[a_i, focus_j] for all focus_j))
  ```
- **Range**: [0, 1]

#### 7. `focus_token_coverage_ratio`
- **Description**: Fraction of focus tokens that have strong attention (>0.08 threshold) to at least one answer token
- **Formula**:
  ```
  focus_token_coverage_ratio = count(focus_i where max(attention[focus_i, a_j]) > 0.08) / |focus_tokens|
  ```
- **Range**: [0, 1]

---

### Model-Independent Features (18 features)

These features are extracted from text only and are shared across all models.

#### 8. `tfidf_cosine_similarity_q_a`
- **Description**: TF-IDF cosine similarity between question and answer
- **Formula**:
  ```
  tfidf_q = TFIDFVectorizer.transform(question)
  tfidf_a = TFIDFVectorizer.transform(answer)
  similarity = cosine_similarity(tfidf_q, tfidf_a)[0, 0]
  ```
  Uses unigrams and bigrams, English stopwords removed, min_df=2.
- **Range**: [0, 1]

#### 9. `content_word_jaccard_q_a`
- **Description**: Jaccard similarity of content words (non-stopwords) between question and answer
- **Formula**:
  ```
  content_words_q = {w for w in question_words if w not in stopwords}
  content_words_a = {w for w in answer_words if w not in stopwords}
  jaccard = |content_words_q ∩ content_words_a| / |content_words_q ∪ content_words_a|
  ```
- **Range**: [0, 1]

#### 10. `question_content_coverage_in_answer`
- **Description**: Fraction of question content words that appear in the answer
- **Formula**:
  ```
  coverage = |content_words_q ∩ content_words_a| / |content_words_q|
  ```
- **Range**: [0, 1]

#### 11. `answer_content_word_ratio`
- **Description**: Ratio of content words (non-stopwords) to total words in the answer
- **Formula**:
  ```
  ratio = count(content_words) / count(all_words)
  ```
- **Range**: [0, 1]

#### 12. `answer_digit_groups_per_word`
- **Description**: Number of digit groups (consecutive digits) per word in the answer
- **Formula**:
  ```
  digit_groups = count(re.findall(r"\d+", answer))
  words = count(re.findall(r"[A-Za-z']+", answer))
  ratio = digit_groups / max(1, words)
  ```
- **Range**: [0, ∞)

#### 13. `refusal_pattern_match_count`
- **Description**: Count of refusal pattern matches in the answer (case-insensitive regex)
- **Formula**: Count of matches for patterns like:
  - `r"\bI (can't|cannot|won't) (comment|answer|say|discuss)\b"`
  - `r"\bI (don't|do not) (know|have information)\b"`
  - `r"\bno comment\b"`
  - `r"\bI (decline|refuse)\b"`
  - (8 total patterns)
- **Range**: Integer ≥ 0

#### 14. `clarification_pattern_match_count`
- **Description**: Count of clarification request patterns in the answer
- **Formula**: Count of matches for patterns like:
  - `r"\b(can|could|would) you clarify\b"`
  - `r"\bwhat do you mean\b"`
  - `r"\b(i )?(don'?t|do not) understand\b"`
  - (25+ total patterns)
- **Range**: Integer ≥ 0

#### 15. `answer_question_mark_count`
- **Description**: Number of question marks in the answer
- **Formula**: `count("?")` in answer text
- **Range**: Integer ≥ 0

#### 16. `answer_word_count`
- **Description**: Total word count in the answer
- **Formula**: `count(re.findall(r"[A-Za-z']+", answer))`
- **Range**: Integer ≥ 0

#### 17. `answer_is_short_question`
- **Description**: Binary feature: 1 if answer has question mark AND ≤10 words, else 0
- **Formula**:
  ```
  answer_is_short_question = 1.0 if (question_mark_count > 0 and word_count <= 10) else 0.0
  ```
- **Range**: {0, 1}

#### 18. `answer_negation_ratio`
- **Description**: Ratio of negation words to total words in the answer
- **Formula**:
  ```
  negation_words = {"no", "not", "never", "none", "cannot", "don't", "won't", ...}
  ratio = count(negation_words in answer) / count(all_words)
  ```
- **Range**: [0, 1]

#### 19. `answer_hedge_ratio`
- **Description**: Ratio of hedge words (uncertainty markers) to total words
- **Formula**:
  ```
  hedge_words = {"maybe", "perhaps", "probably", "seems", "appears", "roughly", "about", ...}
  ratio = count(hedge_words in answer) / count(all_words)
  ```
- **Range**: [0, 1]

#### 20. `question_sentiment_polarity`
- **Description**: Sentiment polarity of the question (positive - negative)
- **Formula**:
  ```
  sentiment_scores = sentiment_pipeline(question)
  polarity = positive_score - negative_score
  ```
  Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` model.
- **Range**: [-1, 1]

#### 21. `answer_sentiment_polarity`
- **Description**: Sentiment polarity of the answer (positive - negative)
- **Formula**: Same as question sentiment polarity, applied to answer text
- **Range**: [-1, 1]

#### 22. `answer_char_per_sentence`
- **Description**: Average characters per sentence in the answer
- **Formula**:
  ```
  sentences = re.split(r"(?<=[.!?])\s+", answer)
  char_per_sentence = total_characters / max(1, len(sentences))
  ```
- **Range**: [0, ∞)

#### 23. `inaudible`
- **Description**: Binary metadata feature indicating if the question/answer was inaudible
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}

#### 24. `multiple_questions`
- **Description**: Binary metadata feature indicating if the question contains multiple sub-questions
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}

#### 25. `affirmative_questions`
- **Description**: Binary metadata feature indicating if the question is affirmative (yes/no style)
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}

---

## Three Evaluation Methodologies

This repository implements three evaluation methodologies on the test set (held-out, never used for training or development). Each methodology uses different feature selection strategies and ensemble approaches.

**Experimental Setup**: 
- **6 Transformer Models**: BERT, BERT-Political, BERT-Ambiguity, RoBERTa, DeBERTa, XLNet
- **6 Classifiers**: LogisticRegression, LinearSVC, RandomForest, MLP, XGBoost, LightGBM
- **2 Tasks**: Clarity (3 classes), Evasion (9 classes)

---

### Methodology 1: Individual Model Baseline Evaluation

**Notebook**: `04_methodology_1__initial_evaluationon_test_set_1_6_6_36.py`  
**Description**: Baseline evaluation where each of the 6 models is evaluated separately with each of the 6 classifiers, using all 25 features per model. No feature selection or ensemble.

**Experimental Design**: 
- **6 models** × **6 classifiers** × **2 tasks** = **72 model-classifier-task combinations**
- Each model uses its own 25 features (7 model-dependent + 18 model-independent)

**Process**:
1. Extract 25 features for each model on test set
2. Train on Train+Dev combined data (final training)
3. Evaluate each model×classifier combination on test set
4. Generate results tables and visualizations

**Feature Selection**: None (uses all 25 features per model)

**Ensemble Strategy**: None (individual model results only)

**Evaluation Set**: Test set (308 samples for Clarity, 275 samples for Evasion)

**Results Location** (Google Drive):
- **Predictions**: `results/FinalResultsType1/predictions/pred_test_{model}_{classifier}_{task}.npy`
- **Probabilities**: `results/FinalResultsType1/probabilities/probs_test_{model}_{classifier}_{task}.npy`
- **Plots**: `results/FinalResultsType1/plots/{model}_{task}_{classifier}/`
- **Tables**: `results/FinalResultsType1/tables/`
- **Metadata**: `results/FinalResultsType1Results/FINAL_TEST_{model}_{task}.json`

**Total Results**: **72 model-classifier-task combinations**

---

### Methodology 2: Early Fusion with Classifier-Specific Feature Selection

**Notebook**: `0_5_methodology_2_early_fusion_60_feature_6_classifier.py`  
**Description**: Early fusion of 60 features (18 model-independent + 42 model-dependent from 6 models) with classifier-specific feature selection. Each classifier gets 40 features selected via greedy forward selection.

**Prerequisite Step (Development Evaluation)**: 
- **Notebook**: `03_train_evaluate_on_dev_set_6_6_36.ipynb`
- **Purpose**: Development set evaluation used for model selection and to guide feature selection
- **Process**: Train on train set, evaluate on dev set (6 models × 6 classifiers × 2 tasks)

**Experimental Design**:
- **60 Early Fusion Features**: 18 model-independent + 42 model-dependent (6 models × 7 features)
- **40 Features per Classifier**: Selected via greedy forward selection (global top 20 + classifier-specific greedy 20)
- **6 classifiers** × **2 tasks** = **12 classifier-task combinations**

**Process**:

#### Step 1: Development Set Evaluation (Prerequisite)
- Extract 25 features for each model on train and dev sets
- Train on train set, evaluate on dev set
- Compare classifier performance to guide feature selection
- **Output**: `predictions/pred_dev_{model}_{classifier}_{task}.npy`

#### Step 2: Early Fusion Feature Creation (60 features)
- **18 model-independent features** (shared across all models)
- **42 model-dependent features** (6 models × 7 features each)
- Total: **60 features** via concatenation

#### Step 3: Classifier-Specific Feature Selection
- **Global Top 20**: Selected from weighted score ranking (across all models)
- **Classifier-Specific Greedy 20**: Greedy forward selection for each classifier (starts with global top 20, adds up to 20 more)
- **Final Feature Set**: 40 features per classifier (global 20 + classifier-specific greedy 20)

#### Step 4: Training and Evaluation on Test Set
- Train on Train+Dev combined data
- Evaluate on **test set** (final evaluation)
- Each classifier uses its own 40 selected features

**Feature Selection**: 
- **40 features per classifier** (selected via greedy forward selection)
- Global top 20 + classifier-specific greedy 20

**Ensemble Strategy**: Weighted average ensemble from probabilities (weights = Macro F1 scores)

**Evaluation Set**: Test set (final evaluation)

**Results Location** (Google Drive):
- **Predictions**: `results/FinalResultsType2/predictions/{classifier}_{task}_predictions.npy`
- **Probabilities**: `results/FinalResultsType2/probabilities/{classifier}_{task}_probabilities.npy`
- **Ensemble**: `results/FinalResultsType2/predictions/ensemble_hard_labels_from_weighted_proba_{task}.npy`
- **Metrics**: `results/FinalResultsType2/metrics/ensemble_evaluation_metrics_{task}.json`

**Total Results**: **12 classifier-task combinations** (6 classifiers × 2 tasks) + **2 ensemble results** (one per task)

---

### Methodology 3: Early Fusion Baseline Evaluation

**Notebook**: `06_methodology_3_and_4_ablation_and_ensemble_methodologies.ipynb`  
**Description**: Early fusion of all 60 features (18 model-independent + 42 model-dependent from 6 models) evaluated with all 6 classifiers. No feature selection - uses all 60 features.

**Process**:

#### Step 1: Early Fusion Feature Creation
- **18 model-independent features** (extracted once, shared)
- **42 model-dependent features** (6 models × 7 features each)
- Concatenation: `[18 model-independent | 42 model-dependent]` = **60 features**

#### Step 2: Training and Evaluation
- Train on Train+Dev combined data
- Evaluate on **test set** (final evaluation)
- All classifiers use the same 60 features (no feature selection)

**Feature Selection**: None (uses all 60 features)

**Ensemble Strategy**: None (individual classifier results only)

**Evaluation Set**: Test set (final evaluation)

**Results Location** (Google Drive):
- **Test Features**: `results/FinalResultsType3/test/X_test_60feat_{task}.npy`
- **Predictions**: `results/FinalResultsType3/predictions/pred_test_{classifier}_{task}.npy`
- **Probabilities**: `results/FinalResultsType3/predictions/proba_test_{classifier}_{task}.npy`
- **Tables**: `results/FinalResultsType3/tables/`
  - `summary_{task}.csv` (per-task summary)
  - `summary_all_tasks_pivot.csv` (classifier × task pivot table)
  - `final_summary_classifier_wise.csv` (includes hierarchical evaluations)
  - `final_summary_detailed.csv` (all metrics)
- **Results**: `results/FinalResultsType3/final_results_type3.json`

**Experimental Design**: 
- **60 Early Fusion Features**: 18 model-independent + 42 model-dependent (6 models × 7 features)
- **6 classifiers** × **2 tasks** = **12 classifier-task combinations**
- All classifiers use the same 60 features (no feature selection)

**Total Results**: **12 classifier-task combinations** (6 classifiers × 2 tasks)

**Additional Evaluations**:
- **Hierarchical Evasion → Clarity**: Maps evasion predictions to clarity labels
- **Annotator-Based Clarity**: Evaluates against annotator1/2/3 clarity labels (mapped from evasion)

---

## Methodology Comparison

| Methodology | Features | Feature Selection | Ensemble | Evaluation Set | Total Results |
|------------|----------|------------------|----------|----------------|---------------|
| **Method 1: Individual Model Baseline** | 25 per model | None | None | Test | 72 combinations (6 models × 6 classifiers × 2 tasks) |
| **Method 2: Early Fusion with Feature Selection** | 40 per classifier | Greedy (global 20 + classifier-specific 20) | Weighted average | Test | 12 individual + 2 ensemble |
| **Method 3: Early Fusion Baseline** | 60 (early fusion) | None | None | Test | 12 combinations (6 classifiers × 2 tasks) |

**Note**: Development set evaluation (`03_train_evaluate_on_dev_set_6_6_36.ipynb`) is a prerequisite step for Methodology 2, not a separate methodology. It evaluates 6 models × 6 classifiers × 2 tasks on the development set for model selection and feature selection guidance.

---

## Project Structure

```
semeval-context-tree-modular/
├── notebooks/              # Pipeline notebooks
│   ├── 01_data_split.ipynb               # Dataset splitting (Train/Dev/Test)
│   ├── 02_feature_extraction_separate.ipynb  # Feature extraction per model
│   ├── 03_train_evaluate_on_dev_set_6_6_36.ipynb  # Dev set evaluation (for Methodology 2)
│   ├── 03_train_evaluate_on_dev_set_6_6_36.py
│   ├── 04_methodology_1__initial_evaluationon_test_set_1_6_6_36.ipynb  # Methodology 1
│   ├── 04_methodology_1__initial_evaluationon_test_set_1_6_6_36.py
│   ├── 0_5_methodology_2_early_fusion_60_feature_6_classifier.ipynb  # Methodology 2
│   ├── 0_5_methodology_2_early_fusion_60_feature_6_classifier.py
│   └── 06_methodology_3_and_4_ablation_and_ensemble_methodologies.ipynb  # Methodology 3
├── src/                    # Python source code
│   ├── data/               # Dataset loading and splitting
│   │   ├── loader.py
│   │   └── splitter.py
│   ├── features/           # Feature extraction and fusion
│   │   ├── extraction.py  # 25 feature extraction functions
│   │   ├── fusion.py       # Early fusion functions
│   │   └── utils.py        # Feature calculation utilities
│   ├── models/             # Model training and classifiers
│   │   ├── classifiers.py # Classifier definitions
│   │   ├── trainer.py     # Training functions
│   │   ├── hierarchical.py # Evasion → Clarity mapping
│   │   ├── ensemble.py    # Ensemble methods
│   │   ├── inference.py   # Inference functions
│   │   └── final_classification.py
│   ├── evaluation/         # Metrics and visualization
│   │   ├── metrics.py     # Metric computation
│   │   ├── tables.py      # Results table generation
│   │   ├── plots.py       # Plotting functions
│   │   └── visualizer.py  # Visualization utilities
│   ├── storage/            # Data storage management
│   │   └── manager.py     # StorageManager class
│   └── utils/              # Utility functions
│       └── reproducibility.py
└── metadata/               # Feature and result metadata
```

---

## Pipeline Workflow

### 1. Data Splitting (`01_data_split.ipynb`)
- Load QEvasion dataset from HuggingFace (`ailsntua/QEvasion`)
- **Clarity Task**: Use all samples (no filtering)
- **Evasion Task**: Apply majority voting (keep samples with ≥2/3 annotator agreement)
- Split HuggingFace train into Train (80%) and Dev (20%)
- Keep HuggingFace test untouched (only used in final evaluation)

**Outputs**:
- `splits/dataset_splits_{task}.pkl` (task-specific splits)

### 2. Feature Extraction (`02_feature_extraction_separate.ipynb`)
Extract 25 Context Tree features for each transformer model separately.

**Outputs**:
- `features/raw/X_{split}_{model}_{task}.npy` (feature matrices)
- `metadata/features_{split}_{model}_{task}.json` (feature metadata)

### 3. Development Set Evaluation (`03_train_evaluate_on_dev_set_6_6_36.ipynb`) - Prerequisite for Methodology 2
Train and evaluate on development set (for model selection and feature selection guidance). This is not a final evaluation but a prerequisite step.

**Outputs**:
- Predictions: `predictions/pred_dev_{model}_{classifier}_{task}.npy`
- Probabilities: `features/probabilities/probs_dev_{model}_{classifier}_{task}.npy`
- Results: `results/{model}_{task}_separate.json`

### 4. Methodology 1 (`04_methodology_1__initial_evaluationon_test_set_1_6_6_36.py`)
Evaluate all model×classifier combinations on test set.

**Outputs**:
- Test predictions: `results/FinalResultsType1/predictions/`
- Test probabilities: `results/FinalResultsType1/probabilities/`
- Final tables: `results/FinalResultsType1/tables/`

### 5. Methodology 2 (`0_5_methodology_2_early_fusion_60_feature_6_classifier.py`)
Early fusion with classifier-specific feature selection.

**Outputs**:
- Predictions: `results/FinalResultsType2/predictions/`
- Probabilities: `results/FinalResultsType2/probabilities/`
- Ensemble results: `results/FinalResultsType2/predictions/ensemble_*.npy`

### 6. Methodology 3 (`06_methodology_3_and_4_ablation_and_ensemble_methodologies.ipynb`)
Early fusion of 60 features evaluated with all classifiers.

**Outputs**:
- Fused features: `results/FinalResultsType3/test/X_test_60feat_{task}.npy`
- Predictions: `results/FinalResultsType3/predictions/`
- Tables: `results/FinalResultsType3/tables/`

---

## Results and Storage Locations

All results are stored in **Google Drive** at `/content/drive/MyDrive/semeval_data/` (when running in Colab).

### Methodology 1
**Drive Path**: `results/FinalResultsType1/`
- Predictions: `predictions/pred_test_{model}_{classifier}_{task}.npy`
- Probabilities: `probabilities/probs_test_{model}_{classifier}_{task}.npy`
- Plots: `plots/{model}_{task}_{classifier}/`
- Tables: `tables/`
- Metadata: `results/FinalResultsType1Results/FINAL_TEST_{model}_{task}.json`

### Development Set Evaluation (Prerequisite for Methodology 2)
**Drive Path**: `results/` and `predictions/`
- Predictions: `predictions/pred_dev_{model}_{classifier}_{task}.npy`
- Probabilities: `features/probabilities/probs_dev_{model}_{classifier}_{task}.npy`
- Results: `results/{model}_{task}_separate.json`

### Methodology 2
**Drive Path**: `results/FinalResultsType2/`
- Hard labels: `predictions/{classifier}_{task}_predictions.npy`
- Probabilities: `probabilities/{classifier}_{task}_probabilities.npy`
- Ensemble: `predictions/ensemble_hard_labels_from_weighted_proba_{task}.npy`
- Metrics: `metrics/ensemble_evaluation_metrics_{task}.json`

### Methodology 3
**Drive Path**: `results/FinalResultsType3/`
- Test features: `test/X_test_60feat_{task}.npy`
- Predictions: `predictions/pred_test_{classifier}_{task}.npy`
- Probabilities: `predictions/proba_test_{classifier}_{task}.npy`
- Tables: `tables/`
  - `summary_{task}.csv`
  - `summary_all_tasks_pivot.csv`
  - `final_summary_classifier_wise.csv`
  - `final_summary_detailed.csv`
- Results: `final_results_type3.json`

---

## Core Modules

### `src/storage/manager.py` - StorageManager
Manages data storage: GitHub for metadata, Google Drive for large files.

**Key Methods**:
- `save_features()` / `load_features()`: Feature matrices
- `save_splits()` / `load_split()`: Dataset splits (task-specific)
- `save_predictions()` / `load_predictions()`: Hard label predictions
- `save_probabilities()` / `load_probabilities()`: Probability distributions
- `save_fused_features()` / `load_fused_features()`: Early fusion features

### `src/features/extraction.py` - Feature Extraction
Extracts 25 Context Tree features from question-answer pairs.

**Key Functions**:
- `extract_batch_features_v2()`: Extract all 25 features from a batch
- `featurize_hf_dataset_in_batches_v2()`: Process entire dataset in batches
- `featurize_model_independent_features()`: Extract 18 model-independent features
- `featurize_model_dependent_features()`: Extract 7 model-dependent features
- `get_feature_names()`: Returns list of 25 feature names
- `get_model_independent_feature_names()`: Returns 18 model-independent feature names
- `get_model_dependent_feature_names()`: Returns 7 model-dependent feature names

### `src/models/classifiers.py` - Classifiers
Defines classifier instances with proper preprocessing.

**Classifiers**: LogisticRegression, LinearSVC, RandomForest, MLP, XGBoost, LightGBM

**Note**: Uses LabelEncoder to convert string labels to numeric (required for MLP, XGBoost, LightGBM)

### `src/models/hierarchical.py` - Hierarchical Approach
Maps evasion predictions to clarity labels using hierarchical mapping.

**Mapping**:
- **Non-Reply** → "Clear Non-Reply": Claims ignorance, Clarification, Declining to answer
- **Reply** → "Clear Reply": Explicit
- **Others** → "Ambivalent": Implicit, Dodging, General, Deflection, Partial/half-answer

### `src/features/fusion.py` - Feature Fusion
Implements early fusion by concatenating features from multiple models.

**Function**: `fuse_attention_features()` - Horizontal concatenation

---

## Usage

### Running the Complete Pipeline

To reproduce all experiments, run the notebooks in the following order:

1. **Data Splitting**: Run `01_data_split.ipynb` to create train/dev/test splits
2. **Feature Extraction**: Run `02_feature_extraction_separate.ipynb` to extract 25 features for each model
3. **Development Evaluation**: Run `03_train_evaluate_on_dev_set_6_6_36.ipynb` (prerequisite for Methodology 2) for dev set results
4. **Methodology 1**: Run `04_methodology_1__initial_evaluationon_test_set_1_6_6_36.py` for individual model baseline evaluation
5. **Methodology 2**: Run `0_5_methodology_2_early_fusion_60_feature_6_classifier.py` for early fusion with classifier-specific feature selection
6. **Methodology 3**: Run `06_methodology_3_and_4_ablation_and_ensemble_methodologies.ipynb` for 60-feature early fusion baseline

### Running Individual Methodologies

Each methodology can be run independently if the prerequisite steps are completed:

- **Methodology 1**: Requires steps 1-2 (data splitting and feature extraction)
- **Methodology 2**: Requires steps 1-3 (data splitting, feature extraction, and dev evaluation)
- **Methodology 3**: Requires steps 1-2 (data splitting and feature extraction)

### Command Line Execution

For Python scripts (`.py` files), you can run them directly:

```bash
# From the notebooks directory
python 04_methodology_1__initial_evaluationon_test_set_1_6_6_36.py
python 0_5_methodology_2_early_fusion_60_feature_6_classifier.py
```

**CRITICAL**: Test set is **ONLY** accessed in final evaluation notebooks (Methodologies 1, 2, and 3). Do not access test set in development notebooks.

---

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for transformer models)
- Minimum 16GB RAM (32GB recommended)
- Google Colab Pro or local machine with sufficient resources

### Python Packages

Install all dependencies using:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas numpy xgboost lightgbm tqdm matplotlib seaborn
```

Or create a `requirements.txt` file with:

```
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0
transformers>=4.20.0
datasets>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
xgboost>=1.6.0
lightgbm>=3.3.0
tqdm>=4.64.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

### Google Colab Setup

If using Google Colab, each notebook includes setup cells that:
1. Clone the repository
2. Mount Google Drive
3. Install required packages
4. Configure paths

Simply run the setup cells at the beginning of each notebook.

## Key Features

1. **Task-Specific Splits**: Clarity and Evasion have different splits (Evasion is filtered by majority voting)
2. **Test Set Isolation**: Test set is only accessed in final evaluation (prevents data leakage)
3. **TF-IDF Per Task**: Each task gets its own TF-IDF vectorizer (fitted on train, applied to dev/test)
4. **Label Encoding**: String labels converted to numeric (required for some classifiers)
5. **Storage Management**: Large files in Google Drive, metadata in GitHub
6. **Reproducibility**: All random seeds fixed to 42
   - Python random, NumPy, PyTorch, and HuggingFace Transformers seeds set to 42
   - PyTorch deterministic mode enabled for full reproducibility
   - Classifiers use `random_state=42`
   - Data splits use `seed=42`
   - Each notebook has a reproducibility setup cell that must be run first

---

## License

See LICENSE file for details.

---

## Acknowledgments

- SemEval-2024 Task 1 organizers
- QEvasion dataset creators
- HuggingFace for transformer models

---

**Project**: SemEval-2024 Task 1 - Question-Answer Evasion Detection  
**Repository**: https://github.com/EonTechie/semeval-context-tree-modular
