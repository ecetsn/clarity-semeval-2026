# SemEval 2026 Task 6: Modular Context Tree Feature Extraction

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EonTechie/semeval-context-tree-modular/blob/main/notebooks/01_data_split.ipynb)

Click the badge above to open the first notebook directly in Google Colab. Run notebooks sequentially:
1. `01_data_split.ipynb` - Split dataset into Train/Dev/Test
2. `02_feature_extraction_separate.ipynb` - Extract features for each model
3. `03_train_evaluate.ipynb` - Train and evaluate individual models
4. `04_early_fusion.ipynb` - Early fusion experiments
5. `05_final_evaluation.ipynb` - Final evaluation on test set

## 📋 Overview

This repository implements **modular Context Tree feature extraction** for SemEval 2026 Task 6 (CLARITY).

**Key Features:**
- ✅ 19 Context Tree features (attention-based, pattern-based, lexicon-based)
- ✅ Multiple models: BERT, RoBERTa, DeBERTa, XLNet
- ✅ Multiple classifiers: LogisticRegression, LinearSVC, RandomForest, XGBoost, LightGBM
- ✅ Early fusion support (concatenate attention features)
- ✅ Complete evaluation (metrics, plots, confusion matrix, PR/ROC curves)
- ✅ Storage manager (GitHub for metadata, Drive for large data)
- ✅ TEST leakage prevention (Train/Dev/Test split)

**Note:** This is a standalone repository for modular Context Tree feature extraction. You can reference other implementations (Paper, Ihsan, Ece) from the main Question-Evasion repository if needed.

## 🚀 Quick Start (Colab)

Click the Colab badge above to open the first notebook. Each notebook includes automatic setup (clone repo, mount Drive). Run notebooks sequentially: 01 → 02 → 03 → 04 → 05.

## 📁 Repository Structure

```
semeval-modular/
├── src/              # Python modules (importable)
│   ├── data/        # Data loading, splitting
│   ├── features/    # Feature extraction, fusion
│   ├── models/      # Classifiers, fusion models
│   ├── evaluation/  # Metrics, reporting
│   ├── storage/     # Save/load utilities
│   └── utils/       # Helper functions
├── configs/         # Configuration files
├── notebooks/       # Colab notebooks (run these)
├── metadata/        # Metadata JSONs (GitHub)
└── results/         # Results JSONs (GitHub)
```

## 🔧 Requirements

See `requirements.txt` for full list. Main dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0

**Note:** Colab notebooks use pre-installed packages. Requirements.txt is not installed automatically to avoid runtime issues.

## 📦 Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/EonTechie/semeval-context-tree-modular.git
cd semeval-context-tree-modular

# Install dependencies
pip install -r requirements.txt
```

## 🔄 Reproducibility

All random operations use fixed seed `42` (data splitting, classifiers, training). Run notebooks sequentially to reproduce siparismaili01 experiment.

## 🔗 Related Implementations

For reference, you can check other implementations in the main Question-Evasion repository:
- **Paper authors' code**: https://github.com/konstantinosftw/Question-Evasion (root directory)
- **Ihsan's implementation**: Representation-level fusion
- **Ece's implementation**: Decision-level fusion

## 📊 Experiments

This repository implements:
1. **Separate Models Approach**: Each model (BERT, RoBERTa, DeBERTa, XLNet) trained separately
2. **Early Fusion Approach**: Attention features from all models fused together
3. **Late Fusion Approach**: Probability-level fusion (Ece style) - TODO
4. **Representation Fusion**: Representation-level fusion (Ihsan style) - TODO

This is a standalone repository focused on modular Context Tree feature extraction.

