# SemEval-2024 Task 1: Question-Answer Evasion Detection

A modular machine learning pipeline for detecting question-answer evasion strategies using Context Tree features extracted from transformer models.

## 📋 Overview

This project implements a comprehensive pipeline for the **SemEval-2024 Task 1: Question-Answer Evasion Detection** competition. It extracts **19 Context Tree features** from question-answer pairs using multiple transformer models and performs classification on two main tasks:

1. **Clarity Task**: Classify answer clarity (3 classes)
   - Clear Reply
   - Ambivalent
   - Clear Non-Reply

2. **Evasion Task**: Classify evasion strategies (9 classes)
   - Claims ignorance, Clarification, Declining to answer, Deflection, Dodging, Explicit, General, Implicit, Partial/half-answer

## 🏗️ Project Structure

```
semeval-context-tree-modular/
├── notebooks/              # Main pipeline notebooks
│   ├── 00_setup.ipynb                    # Repository setup and Drive mount
│   ├── 01_data_split.ipynb               # Dataset splitting (Train/Dev/Test)
│   ├── 02_feature_extraction_separate.ipynb  # Feature extraction per model
│   ├── 03_train_evaluate.ipynb            # Individual model training
│   ├── 03_5_ablation_study.ipynb          # Feature ablation and selection
│   ├── 04_early_fusion.ipynb              # Multi-model feature fusion
│   └── 05_final_evaluation.ipynb          # Final test set evaluation
├── src/                    # Python source code
│   ├── data/               # Dataset loading and splitting
│   ├── features/           # Feature extraction and fusion
│   ├── models/             # Model training and classifiers
│   ├── evaluation/         # Metrics and visualization
│   └── storage/            # Data storage management
├── scripts/                # Utility scripts
└── metadata/               # Feature and result metadata
```

## 🔄 Pipeline Workflow

### 1. Setup (`00_setup.ipynb`)
- Clone repository from GitHub
- Mount Google Drive for persistent storage
- Initialize StorageManager

### 2. Data Splitting (`01_data_split.ipynb`)
- Load QEvasion dataset from HuggingFace (`ailsntua/QEvasion`)
- **Clarity Task**: Use all samples (no filtering)
- **Evasion Task**: Apply majority voting (keep samples with ≥2/3 annotator agreement)
- Split HuggingFace train into Train (80%) and Dev (20%)
- Keep HuggingFace test untouched (only used in final evaluation)

**Outputs**:
- `splits/dataset_splits_{task}.pkl` (task-specific splits)

### 3. Feature Extraction (`02_feature_extraction_separate.ipynb`)
Extract 19 Context Tree features for each transformer model separately.

**Models**:
- BERT (bert-base-uncased)
- BERT-Political (bert-base-uncased - placeholder)
- BERT-Ambiguity (bert-base-uncased - placeholder)
- RoBERTa (roberta-base)
- DeBERTa (microsoft/deberta-v3-base)
- XLNet (xlnet-base-cased)

**19 Context Tree Features**:

1. **Length Features (2)**:
   - `question_model_token_count`
   - `answer_model_token_count`

2. **Attention Features (2)**:
   - `attention_mass_q_to_a_per_qtoken`
   - `attention_mass_a_to_q_per_atoken`

3. **Focus Features (3)**:
   - `focus_token_to_answer_strength`
   - `answer_token_to_focus_strength`
   - `focus_token_coverage_ratio`

4. **Alignment Features (3)**:
   - `tfidf_cosine_similarity_q_a`
   - `content_word_jaccard_q_a`
   - `question_content_coverage_in_answer`

5. **Answer Surface Features (2)**:
   - `answer_content_word_ratio`
   - `answer_digit_groups_per_word`

6. **Pattern Features (5)**:
   - `refusal_pattern_match_count`
   - `clarification_pattern_match_count`
   - `answer_question_mark_count`
   - `answer_word_count`
   - `answer_is_short_question`

7. **Lexicon Ratios (2)**:
   - `answer_negation_ratio`
   - `answer_hedge_ratio`

**Outputs**:
- `features/raw/X_{split}_{model}_{task}.npy` (feature matrices)
- `metadata/features_{split}_{model}_{task}.json` (feature metadata)

### 4. Training and Evaluation (`03_train_evaluate.ipynb`)
Train multiple classifiers on features from each model separately.

**Classifiers**:
- LogisticRegression (with StandardScaler)
- LinearSVC (with StandardScaler)
- RandomForest
- MLP (Multi-Layer Perceptron, with StandardScaler)
- XGBoost (optional)
- LightGBM (optional)

**Process**:
- Train all classifiers on each model's features
- Evaluate on Dev set
- Implement hierarchical approach: Map evasion predictions to clarity labels
- Generate results tables and visualizations

**Outputs**:
- Predictions: `predictions/pred_{split}_{model}_{classifier}_{task}.npy`
- Probabilities: `features/probabilities/probs_{split}_{model}_{classifier}_{task}.npy`
- Results: `results/{model}_{task}_separate.json`

### 5. Ablation Study (`03_5_ablation_study.ipynb`) ⭐
Comprehensive feature ablation study to identify optimal feature subsets.

**Scope**:
- 3 Tasks: Clarity, Evasion, Hierarchical Evasion → Clarity
- 6 Models × 6 Classifiers = 36 combinations per task
- 19 Features evaluated individually

**Workflow**:

1. **Single-Feature Ablation**: Evaluate each feature individually across all 36 model×classifier combinations
   - Total evaluations: 3 tasks × 6 models × 6 classifiers × 19 features = **2,052 evaluations**
   - Each feature is trained and evaluated separately

2. **Statistical Aggregation**: Compute statistics for each feature:
   - `min_f1`: Minimum Macro F1 (worst-case)
   - `median_f1`: Median Macro F1 (typical)
   - `mean_f1`: Mean Macro F1 (average)
   - `std_f1`: Standard deviation (consistency)
   - `best_f1`: Maximum Macro F1 (best-case)
   - `runs`: Number of evaluations (should be 36)

3. **Weighted Score Calculation**:
   ```
   normalized_std = std_f1 / (mean_f1 + epsilon)
   weighted_score = 0.5 * mean_f1 + 0.3 * best_f1 + 0.2 * (1 - normalized_std)
   ```
   - Balances average performance (50%), peak performance (30%), and consistency (20%)

4. **Feature Ranking**: Rank features by weighted_score (separately for each task)

5. **Top-K Feature Selection**: Select top-K features (default: K=10) for Early Fusion

6. **Greedy Forward Selection** (optional): Iteratively add features that maximize Macro F1
   - Start with top-K features
   - Add best feature at each iteration
   - Continue until no improvement or max_features reached

**Outputs**:
- `results/ablation/single_feature_{task}.csv` (raw ablation results)
- `results/ablation/feature_ranking_{task}.csv` (feature rankings with statistics)
- `results/ablation/selected_features_for_early_fusion.json` (top-K features)
- `results/ablation/greedy_trajectory_{model}_{task}.csv` (greedy selection trajectories)

### 6. Early Fusion (`04_early_fusion.ipynb`)
Combine features from all models using feature-level concatenation.

**Fusion Method**: Horizontal concatenation
- Fused features = [BERT features | RoBERTa features | DeBERTa features | XLNet features]
- Total feature dimension = sum of all model feature dimensions

**Process**:
- Load features from all models
- Concatenate horizontally
- Train classifiers on fused features
- Compare with individual model results

**Outputs**:
- `features/fused/X_{split}_fused_{models}_{task}.npy`
- Fused model predictions and results

### 7. Final Evaluation (`05_final_evaluation.ipynb`)
Evaluate final models on the held-out test set.

**⚠️ CRITICAL**: Test set is **ONLY** accessed in this notebook!

**Process**:
1. Extract test set features (if not already extracted)
2. Combine Train+Dev for final training (maximum data)
3. Evaluate on test set
4. Generate final results tables and visualizations

**Outputs**:
- Test set predictions and probabilities
- Final evaluation metrics
- Competition submission files

## 🧩 Core Modules

### `src/storage/manager.py` - StorageManager
Manages data storage: GitHub for metadata, Google Drive for large files.

**Key Methods**:
- `save_features()` / `load_features()`: Feature matrices
- `save_splits()` / `load_split()`: Dataset splits (task-specific)
- `save_predictions()` / `load_predictions()`: Hard label predictions
- `save_probabilities()` / `load_probabilities()`: Probability distributions

### `src/features/extraction.py` - Feature Extraction
Extracts 19 Context Tree features from question-answer pairs.

**Key Functions**:
- `extract_batch_features_v2()`: Extract features from a batch
- `featurize_hf_dataset_in_batches_v2()`: Process entire dataset in batches
- `get_feature_names()`: Returns list of 19 feature names

### `src/models/trainer.py` - Model Training
Trains classifiers and evaluates on dev set.

**Key Function**:
- `train_and_evaluate()`: Train, evaluate, and visualize results

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

## 📊 Data Flow

```
HuggingFace Dataset (ailsntua/QEvasion)
    ↓
01_data_split.ipynb
    ↓
Train/Dev/Test Splits (task-specific)
    ↓
02_feature_extraction_separate.ipynb
    ↓
19 Context Tree Features (per model)
    ↓
03_train_evaluate.ipynb
    ↓
Model Predictions & Probabilities
    ↓
03_5_ablation_study.ipynb ⭐
    ↓
Feature Rankings & Selected Features
    ↓
04_early_fusion.ipynb
    ↓
Fused Model Predictions
    ↓
05_final_evaluation.ipynb
    ↓
Final Test Set Results
```

## 🔑 Key Features

1. **Task-Specific Splits**: Clarity and Evasion have different splits (Evasion is filtered by majority voting)
2. **Test Set Isolation**: Test set is only accessed in final evaluation (prevents data leakage)
3. **TF-IDF Per Task**: Each task gets its own TF-IDF vectorizer (fitted on train, applied to dev/test)
4. **Label Encoding**: String labels converted to numeric (required for some classifiers)
5. **Storage Management**: Large files in Google Drive, metadata in GitHub
6. **Reproducibility**: All random seeds fixed to 42

## 📝 Usage

1. Run notebooks in order: `00_setup.ipynb` → `01_data_split.ipynb` → ... → `05_final_evaluation.ipynb`
2. Mount Google Drive when prompted
3. Check outputs after each notebook
4. **Do not touch test set** until final evaluation

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy
- XGBoost, LightGBM (optional)
- Google Colab (for Drive integration)

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- SemEval-2024 Task 1 organizers
- QEvasion dataset creators
- HuggingFace for transformer models

---

**Project**: SemEval-2024 Task 1 - Question-Answer Evasion Detection  
**Repository**: https://github.com/EonTechie/semeval-context-tree-modular

