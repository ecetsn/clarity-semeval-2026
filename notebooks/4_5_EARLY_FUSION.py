# KOD HÜCRESİ 1
# ==============
# ============================================================================
# SETUP: Repository Clone, Drive Mount, and Path Configuration
# ============================================================================
# This cell performs minimal setup required for the notebook to run:
# 1. Clones repository from GitHub (if not already present)
# 2. Mounts Google Drive for persistent data storage
# 3. Configures Python paths and initializes StorageManager
# 4. Loads data splits and features created in previous notebooks

import shutil
import os
import subprocess
import time
import requests
import zipfile
import sys
from pathlib import Path
from google.colab import drive
import numpy as np

# Repository configuration
repo_dir = '/content/semeval-context-tree-modular'
repo_url = 'https://github.com/EonTechie/semeval-context-tree-modular.git'
zip_url = 'https://github.com/EonTechie/semeval-context-tree-modular/archive/refs/heads/main.zip'

# Clone repository (if not already present)
if not os.path.exists(repo_dir):
    print("Cloning repository from GitHub...")
    max_retries = 2
    clone_success = False
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['git', 'clone', repo_url],
                cwd='/content',
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("Repository cloned successfully via git")
                clone_success = True
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(3)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
    
    # Fallback: Download as ZIP if git clone fails
    if not clone_success:
        print("Git clone failed. Downloading repository as ZIP archive...")
        zip_path = '/tmp/repo.zip'
        try:
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('/content')
            extracted_dir = '/content/semeval-context-tree-modular-main'
            if os.path.exists(extracted_dir):
                os.rename(extracted_dir, repo_dir)
            os.remove(zip_path)
            print("Repository downloaded and extracted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to obtain repository: {e}")

# Mount Google Drive (if not already mounted)
try:
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass  # Already mounted

# Configure paths
BASE_PATH = Path('/content/semeval-context-tree-modular')
DATA_PATH = Path('/content/drive/MyDrive/semeval_data')

# Verify repository structure exists
if not BASE_PATH.exists():
    raise RuntimeError(f"Repository directory not found: {BASE_PATH}")
if not (BASE_PATH / 'src').exists():
    raise RuntimeError(f"src directory not found in repository: {BASE_PATH / 'src'}")
if not (BASE_PATH / 'src' / 'storage' / 'manager.py').exists():
    raise RuntimeError(f"Required file not found: {BASE_PATH / 'src' / 'storage' / 'manager.py'}")

# Add repository to Python path
sys.path.insert(0, str(BASE_PATH))

# Verify imports work
try:
    from src.storage.manager import StorageManager
    from src.features.fusion import fuse_attention_features
    from src.models.classifiers import get_classifier_dict
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules. "
        f"Repository path: {BASE_PATH}, "
        f"Python path: {sys.path[:3]}, "
        f"Error: {e}"
    )

# Initialize StorageManager
storage = StorageManager(
    base_path=str(BASE_PATH),
    data_path=str(DATA_PATH),
    github_path=str(BASE_PATH)
)

# Data splits will be loaded per-task in the fusion loop
# Clarity and Evasion have different splits (Evasion uses majority voting)

print("Setup complete")
print(f"  Repository: {BASE_PATH}")
print(f"  Data storage: {DATA_PATH}")
print(f"\nNOTE: Data splits will be loaded per-task (task-specific splits)")
print(f"      Clarity and Evasion have different splits due to majority voting")

# KOD HÜCRESİ 2
# ==============
# ============================================================================
# REPRODUCIBILITY SETUP: Set Random Seeds for All Libraries
# ============================================================================
# This cell sets random seeds for Python, NumPy, PyTorch, and HuggingFace
# to ensure reproducible results across all runs.
# 
# IMPORTANT: Run this cell FIRST before any other code that uses randomness.
# Seed value: 42 (same as used in all other parts of the pipeline)

from src.utils.reproducibility import set_all_seeds

# Set all random seeds to 42 for full reproducibility
# deterministic=True ensures PyTorch operations are deterministic (slower but fully reproducible)
set_all_seeds(seed=42, deterministic=True)

print("✓ Reproducibility configured: All random seeds set to 42")
print("✓ PyTorch deterministic mode enabled")
print("\nNOTE: If you encounter performance issues or non-deterministic behavior,")
print("      you can set deterministic=False in set_all_seeds() call above.")

# KOD HÜCRESİ 3
# ==============
# ============================================================================
# CONFIGURE MODELS, TASKS, AND CLASSIFIERS
# ============================================================================
# Defines the models to fuse, tasks to perform, and classifiers to train
# Label mappings are defined for clarity (3-class) and evasion (9-class) tasks

MODELS = ['bert', 'bert_political', 'bert_ambiguity', 'roberta', 'deberta', 'xlnet']
TASKS = ['clarity', 'evasion']

# Label mappings for each task
CLARITY_LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
EVASION_LABELS = ['Claims ignorance', 'Clarification', 'Declining to answer', 
                  'Deflection', 'Dodging', 'Explicit', 
                  'General', 'Implicit', 'Partial/half-answer']

# Initialize classifiers with fixed random seed for reproducibility
classifiers = get_classifier_dict(random_state=42)

print("Configuration:")
print(f"  Models to fuse: {MODELS}")
print(f"  Tasks: {TASKS}")
print(f"  Classifiers: {list(classifiers.keys())}")
print(f"  Fusion method: Early fusion (feature concatenation)")

# KOD HÜCRESİ 4
# ==============
# ============================================================================
# PERFORM EARLY FUSION - Feature Concatenation Only
# ============================================================================
# For each task, loads model-independent features once (18 features) and
# model-dependent features from each model (7 features per model).
# Concatenates: [18 Model-Independent | 6 Models × 7 Model-Dependent] = 60 features
# Saves fused features for Train and Dev splits
# NOTE: Training and evaluation will be done in the next cell (Cell 6) on Test set

from src.features.extraction import get_model_independent_feature_names, get_model_dependent_feature_names

for task in TASKS:
    print(f"\n{'='*80}")
    print(f"TASK: {task.upper()} - EARLY FUSION (60 FEATURES)")
    print(f"{'='*80}")
    
    # Select appropriate label list and dataset key for this task
    if task == 'clarity':
        label_list = CLARITY_LABELS
        label_key = 'clarity_label'
    else:  # evasion
        label_list = EVASION_LABELS
        label_key = 'evasion_label'
    
    # ========================================================================
    # STEP 0: Check if fused features already exist (CHECKPOINT)
    # ========================================================================
    print("\nStep 0: Checking for existing fused features (checkpoint)...")
    try:
        X_train_fused_existing = storage.load_fused_features(MODELS, task, 'train')
        X_dev_fused_existing = storage.load_fused_features(MODELS, task, 'dev')
        print(f"  ✓ Fused features already exist for {task}")
        print(f"    Train: {X_train_fused_existing.shape[0]} samples, {X_train_fused_existing.shape[1]} features")
        print(f"    Dev: {X_dev_fused_existing.shape[0]} samples, {X_dev_fused_existing.shape[1]} features")
        print(f"  SKIPPING fusion (already done)")
        continue
    except FileNotFoundError:
        print(f"  → Fused features not found. Proceeding with fusion...")
        pass
    
    # ========================================================================
    # STEP 1: Load model-independent features (18 features, shared across all models)
    # ========================================================================
    print("\nStep 1: Loading model-independent features (18 features, shared)...")
    try:
        X_train_indep = storage.load_model_independent_features('train', task=task)
        X_dev_indep = storage.load_model_independent_features('dev', task=task)
        print(f"  ✓ Loaded model-independent features: {X_train_indep.shape[1]} features")
        print(f"    Train: {X_train_indep.shape[0]} samples")
        print(f"    Dev: {X_dev_indep.shape[0]} samples")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model-independent features not found for task '{task}'. "
            f"Make sure you have run 02_feature_extraction_separate.ipynb first.\n"
            f"Error: {e}"
        )
    
    # Get model-independent feature names
    indep_feature_names = get_model_independent_feature_names()
    assert len(indep_feature_names) == 18, f"Expected 18 model-independent features, got {len(indep_feature_names)}"
    
    # ========================================================================
    # STEP 2: Load full features from all models and extract model-dependent portion
    # ========================================================================
    print("\nStep 2: Loading model-dependent features from each model (7 features per model)...")
    model_dep_features_train = {}
    model_dep_features_dev = {}
    model_dep_feature_names = {}
    
    # Model-dependent feature names (same for all models)
    dep_feature_names = get_model_dependent_feature_names()
    assert len(dep_feature_names) == 7, f"Expected 7 model-dependent features, got {len(dep_feature_names)}"
    
    for model in MODELS:
        # Load full features (25 features: 7 model-dependent + 18 model-independent)
        X_train_full = storage.load_features(model, task, 'train')
        X_dev_full = storage.load_features(model, task, 'dev')
        
        # Extract model-dependent portion (first 7 features)
        # Feature order: [7 model-dependent | 18 model-independent]
        X_train_dep = X_train_full[:, :7]  # First 7 features
        X_dev_dep = X_dev_full[:, :7]
        
        model_dep_features_train[model] = X_train_dep
        model_dep_features_dev[model] = X_dev_dep
        model_dep_feature_names[model] = [f"{model}_{name}" for name in dep_feature_names]
        
        print(f"  {model}: Extracted {X_train_dep.shape[1]} model-dependent features from {X_train_full.shape[1]} total features")
    
    # ========================================================================
    # STEP 3: Concatenate all features: [18 Model-Independent | 6 Models × 7 Model-Dependent]
    # ========================================================================
    print("\nStep 3: Concatenating features (60 total features)...")
    
    # Concatenate model-dependent features from all models
    model_dep_list_train = [model_dep_features_train[model] for model in MODELS]
    model_dep_list_dev = [model_dep_features_dev[model] for model in MODELS]
    
    X_train_dep_concat = np.hstack(model_dep_list_train)  # (N, 42)
    X_dev_dep_concat = np.hstack(model_dep_list_dev)  # (N, 42)
    
    # Final concatenation: [18 Model-Independent | 42 Model-Dependent]
    X_train_fused = np.hstack([X_train_indep, X_train_dep_concat])  # (N, 60)
    X_dev_fused = np.hstack([X_dev_indep, X_dev_dep_concat])  # (N, 60)
    
    # Build feature names
    fused_feature_names = indep_feature_names.copy()  # 18 model-independent
    for model in MODELS:
        fused_feature_names.extend(model_dep_feature_names[model])  # 7 per model
    
    print(f"  ✓ Fused features: {X_train_fused.shape[1]} features total")
    print(f"    - Model-independent: {X_train_indep.shape[1]} features")
    print(f"    - Model-dependent: {X_train_dep_concat.shape[1]} features (6 models × 7)")
    print(f"    Train: {X_train_fused.shape[0]} samples")
    print(f"    Dev: {X_dev_fused.shape[0]} samples")
    print(f"\n  Feature names (first 5): {fused_feature_names[:5]}")
    print(f"  Feature names (last 5): {fused_feature_names[-5:]}")
    print(f"\n  Model-dependent feature examples:")
    for model in MODELS[:2]:  # Show first 2 models
        print(f"    {model}: {model_dep_feature_names[model][:3]}...")
    
    # Verify feature count
    assert X_train_fused.shape[1] == 60, f"Expected 60 features, got {X_train_fused.shape[1]}"
    assert len(fused_feature_names) == 60, f"Expected 60 feature names, got {len(fused_feature_names)}"
    
    # ========================================================================
    # STEP 4: Save fused features to persistent storage
    # ========================================================================
    print("\nStep 4: Saving fused features to persistent storage...")
    storage.save_fused_features(
        X_train_fused, MODELS, task, 'train',
        fused_feature_names, fusion_method='concat_60'
    )
    storage.save_fused_features(
        X_dev_fused, MODELS, task, 'dev',
        fused_feature_names, fusion_method='concat_60'
    )
    print("  ✓ Fused features saved (Train and Dev)")
    print("  ✓ Ready for Train+Dev training and Test evaluation in next cell")

print(f"\n{'='*80}")
print("Early fusion complete for all tasks (60 features)")
print(f"{'='*80}")
print("\nSummary:")
print("  - 60 features: 18 model-independent + 42 model-dependent (6 models × 7)")
print("  - Fused features saved for Train and Dev splits")
print("  - Next cell will: Train on Train+Dev, Evaluate on Test set")

# KOD HÜCRESİ 5
# ==============
# ============================================================================
# FINAL EVALUATION ON TEST SET (TYPE 3)
# ============================================================================
# This section performs final evaluation on the TEST set using 60-feature early fusion:
# 1. Extract test features (60 features: 18 model-independent + 42 model-dependent)
#    - Checkpoint: Load if exists, extract and save if not
# 2. Combine Train+Dev for final training
# 3. Train all 6 classifiers on Train+Dev (60 features)
# 4. Evaluate on Test set (2 tasks: clarity, evasion)
# 5. Generate summary tables (like notebook 5)
# 6. Save all results to FinalResultsType3 directory structure

**Test Set Sizes:**
- Clarity: 308 samples
- Evasion: 275 samples

**Output Structure:**
- `results/FinalResultsType3/test/` - Test features (60 features)
- `results/FinalResultsType3/predictions/` - Test predictions
- `results/FinalResultsType3/tables/` - Summary tables
- `results/FinalResultsType3/plots/` - Evaluation plots
- `results/FinalResultsType3/` - Other results

# KOD HÜCRESİ 6
# ==============
# ============================================================================
# FINAL EVALUATION ON TEST SET (TYPE 3)
# ============================================================================
# Extract test features (60 features), train on Train+Dev, evaluate on Test

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from src.features.extraction import (
    featurize_model_independent_features,
    featurize_model_dependent_features,
    get_model_independent_feature_names,
    get_model_dependent_feature_names
)
from src.models.classifiers import train_classifiers
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tables import create_final_summary_pivot, style_table_paper
import pandas as pd
from IPython.display import display, HTML
import json

# Model configurations
MODEL_CONFIGS = {
    'bert': 'bert-base-uncased',
    'bert_political': 'bert-base-uncased',  # Fine-tuned version
    'bert_ambiguity': 'bert-base-uncased',  # Fine-tuned version
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-base',
    'xlnet': 'xlnet-base-cased'
}

MODEL_MAX_LENGTHS = {
    'bert': 512,
    'bert_political': 512,
    'bert_ambiguity': 512,
    'roberta': 512,
    'deberta': 512,
    'xlnet': 1024
}

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========================================================================
# STEP 1: Create Type3 output directories (CHECKPOINT)
# ========================================================================
print("\n" + "="*80)
print("STEP 1: CREATE TYPE3 OUTPUT DIRECTORIES")
print("="*80)

# Drive directories
test_features_dir = storage.data_path / 'results/FinalResultsType3/test'
predictions_dir = storage.data_path / 'results/FinalResultsType3/predictions'
tables_dir = storage.data_path / 'results/FinalResultsType3/tables'
plots_dir = storage.data_path / 'results/FinalResultsType3/plots'
results_dir = storage.data_path / 'results/FinalResultsType3'

# Create all directories (always, to prevent FileNotFoundError)
# CRITICAL: mkdir(parents=True, exist_ok=True) creates ALL parent directories recursively
# Example: test_features_dir = 'results/FinalResultsType3/test'
#   - Creates 'results/' if not exists
#   - Creates 'results/FinalResultsType3/' if not exists
#   - Creates 'results/FinalResultsType3/test/' if not exists
# This ensures the entire path exists before any save operation

test_features_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Verify all directories were created successfully
print("✓ Created all Type3 output directories:")
print(f"  - Test features: {test_features_dir} (exists: {test_features_dir.exists()})")
print(f"  - Predictions: {predictions_dir} (exists: {predictions_dir.exists()})")
print(f"  - Tables: {tables_dir} (exists: {tables_dir.exists()})")
print(f"  - Plots: {plots_dir} (exists: {plots_dir.exists()})")
print(f"  - Results: {results_dir} (exists: {results_dir.exists()})")

# Verify parent directories also exist
parent_dir = storage.data_path / 'results'
print(f"\n✓ Parent directory 'results/' exists: {parent_dir.exists()}")
type3_parent = storage.data_path / 'results/FinalResultsType3'
print(f"✓ Type3 parent directory exists: {type3_parent.exists()}")

# ========================================================================
# STEP 2: Extract or load test features (60 features) - CHECKPOINT
# ========================================================================
print("\n" + "="*80)
print("STEP 2: TEST FEATURE EXTRACTION (60 FEATURES)")
print("="*80)

# Load sentiment pipeline for model-independent features
print("Loading sentiment analysis pipeline...")
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True
    )
    print("  ✓ Sentiment pipeline loaded")
except Exception as e:
    print(f"  ⚠ Could not load sentiment pipeline: {e}")
    sentiment_pipeline = None

metadata_keys = {
    'inaudible': 'inaudible',
    'multiple_questions': 'multiple_questions',
    'affirmative_questions': 'affirmative_questions'
}

# Store test features for each task
test_features_60 = {}  # {task: {'train': X_train_60, 'dev': X_dev_60, 'test': X_test_60}}

for task in TASKS:
    print(f"\n{'-'*60}")
    print(f"Task: {task.upper()}")
    print(f"{'-'*60}")
    
    # Load test split
    try:
        test_ds = storage.load_split('test', task=task)
        print(f"  Test set: {len(test_ds)} samples")
    except FileNotFoundError as e:
        print(f"  ⚠ Test split not found for {task}: {e}")
        continue
    
    # ====================================================================
    # 2.1: Extract or load model-independent test features (18 features)
    # ====================================================================
    print(f"\n  2.1: Model-independent test features (18 features)...")
    test_indep_path = test_features_dir / f'X_test_independent_{task}.npy'
    
    if test_indep_path.exists():
        X_test_indep = np.load(test_indep_path)
        print(f"    ✓ Loaded from checkpoint: {X_test_indep.shape}")
    else:
        print(f"    → Extracting model-independent test features...")
        X_test_indep, _ = featurize_model_independent_features(
            test_ds,
            question_key='interview_question',
            answer_key='interview_answer',
            batch_size=32,
            show_progress=True,
            sentiment_pipeline=sentiment_pipeline,
            metadata_keys=metadata_keys,
        )
        # Save to checkpoint (ensure directory exists before saving)
        test_features_dir.mkdir(parents=True, exist_ok=True)
        np.save(test_indep_path, X_test_indep)
        print(f"    ✓ Extracted and saved: {X_test_indep.shape}")
    
    # ====================================================================
    # 2.2: Extract or load model-dependent test features (7 features per model)
    # ====================================================================
    print(f"\n  2.2: Model-dependent test features (7 features × 6 models = 42 features)...")
    model_dep_test_features = {}
    
    for model_key in MODELS:
        model_name = MODEL_CONFIGS[model_key]
        max_seq_len = MODEL_MAX_LENGTHS[model_key]
        
        test_dep_path = test_features_dir / f'X_test_{model_key}_dependent_{task}.npy'
        
        if test_dep_path.exists():
            X_test_dep = np.load(test_dep_path)
            print(f"    {model_key}: ✓ Loaded from checkpoint: {X_test_dep.shape}")
        else:
            print(f"    {model_key}: → Extracting model-dependent test features...")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            
            # Extract model-dependent features only
            X_test_dep, _ = featurize_model_dependent_features(
                test_ds,
                tokenizer,
                model,
                device,
                question_key='interview_question',
                answer_key='interview_answer',
                batch_size=8,
                max_sequence_length=max_seq_len,
                show_progress=True,
            )
            
            # Save to checkpoint (ensure directory exists before saving)
            test_features_dir.mkdir(parents=True, exist_ok=True)
            np.save(test_dep_path, X_test_dep)
            print(f"    {model_key}: ✓ Extracted and saved: {X_test_dep.shape}")
            
            # Free GPU memory after processing each model (prevent CUDA errors)
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"    {model_key}: ✓ GPU memory cleared")
        
        model_dep_test_features[model_key] = X_test_dep
    
    # ====================================================================
    # 2.3: Concatenate test features: [18 Model-Independent | 42 Model-Dependent]
    # ====================================================================
    print(f"\n  2.3: Concatenating test features (60 total)...")
    
    # Concatenate model-dependent features from all models
    model_dep_list = [model_dep_test_features[model] for model in MODELS]
    X_test_dep_concat = np.hstack(model_dep_list)  # (N, 42)
    
    # Final concatenation: [18 Model-Independent | 42 Model-Dependent]
    X_test_60 = np.hstack([X_test_indep, X_test_dep_concat])  # (N, 60)
    
    print(f"    ✓ Test features: {X_test_60.shape} (60 features)")
    print(f"      - Model-independent: {X_test_indep.shape[1]} features")
    print(f"      - Model-dependent: {X_test_dep_concat.shape[1]} features (6 models × 7)")
    
    # Verify feature count
    assert X_test_60.shape[1] == 60, f"Expected 60 features, got {X_test_60.shape[1]}"
    
    # Save complete test features to checkpoint (ensure directory exists before saving)
    test_features_dir.mkdir(parents=True, exist_ok=True)
    test_complete_path = test_features_dir / f'X_test_60feat_{task}.npy'
    np.save(test_complete_path, X_test_60)
    print(f"    ✓ Saved complete test features to: {test_complete_path.name}")
    
    # Store for later use
    test_features_60[task] = {
        'test': X_test_60
    }

print("\n✓ Test feature extraction complete for all tasks")

# KOD HÜCRESİ 7
# ==============
# ========================================================================
# STEP 3: Train on Train+Dev and evaluate on Test (2 tasks, 6 classifiers)
# ========================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN ON TRAIN+DEV AND EVALUATE ON TEST")
print("="*80)

# Store all results for summary tables
all_results_type3 = {}  # {task: {classifier: {metrics, predictions, probabilities}}}

for task in TASKS:
    print(f"\n{'-'*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'-'*80}")
    
    if task not in test_features_60:
        print(f"  ⚠ Skipping {task}: Test features not available")
        continue
    
    # Select appropriate label list and dataset key
    if task == 'clarity':
        label_list = CLARITY_LABELS
        label_key = 'clarity_label'
    else:  # evasion
        label_list = EVASION_LABELS
        label_key = 'evasion_label'
    
    # Load Train+Dev fused features (60 features) - from Cell 4
    # These were already created and saved in Cell 4
    try:
        # Load fused features directly from Cell 4 (no need to reconstruct)
        X_train_60 = storage.load_fused_features(MODELS, task, 'train')
        X_dev_60 = storage.load_fused_features(MODELS, task, 'dev')
        
        print(f"  ✓ Loaded Train fused features: {X_train_60.shape} (60 features)")
        print(f"  ✓ Loaded Dev fused features: {X_dev_60.shape} (60 features)")
        
    except FileNotFoundError as e:
        print(f"  ⚠ Fused features not found for {task}. Make sure Cell 4 completed successfully.")
        print(f"  Error: {e}")
        continue
    except Exception as e:
        print(f"  ⚠ Error loading train+dev features: {e}")
        continue
    
    # Load labels
    train_ds = storage.load_split('train', task=task)
    dev_ds = storage.load_split('dev', task=task)
    test_ds = storage.load_split('test', task=task)
    
    y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
    y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
    y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
    
    # Combine Train+Dev for final training
    X_train_full = np.vstack([X_train_60, X_dev_60])
    y_train_full = np.concatenate([y_train, y_dev])
    
    print(f"  Combined Train+Dev: {X_train_full.shape[0]} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Get test features
    X_test_60 = test_features_60[task]['test']
    
    # Train all 6 classifiers and evaluate on test
    print(f"\n  Training and evaluating {len(classifiers)} classifiers...")
    all_results_type3[task] = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n    Classifier: {clf_name}")
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_full)
        y_test_encoded = le.transform(y_test)
        
        # Train classifier
        clf.fit(X_train_full, y_train_encoded)
        
        # Predict on test
        y_test_pred_encoded = clf.predict(X_test_60)
        y_test_pred = le.inverse_transform(y_test_pred_encoded)
        
        # Get probabilities (if available)
        if hasattr(clf, 'predict_proba'):
            y_test_proba = clf.predict_proba(X_test_60)
        else:
            y_test_proba = None
        
        # Compute metrics
        metrics = compute_all_metrics(
            y_test_encoded,
            y_test_pred_encoded,
            label_list,
            task_name=f"TYPE3_TEST_{task}_{clf_name}"
        )
        
        # Store results
        all_results_type3[task][clf_name] = {
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'metrics': metrics
        }
        
        print(f"      Test Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
        print(f"      Test Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        
        # Save HARD LABELS (predictions) to Type3 folder
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pred_path = predictions_dir / f'pred_test_{clf_name}_{task}.npy'
        np.save(pred_path, y_test_pred)
        print(f"      ✓ Saved HARD LABELS: {pred_path.name}")
        
        # Save SOFT LABELS (probabilities) to Type3 folder
        if y_test_proba is not None:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            proba_path = predictions_dir / f'proba_test_{clf_name}_{task}.npy'
            np.save(proba_path, y_test_proba)
            print(f"      ✓ Saved SOFT LABELS: {proba_path.name} (shape: {y_test_proba.shape})")
        else:
            print(f"      ⚠ SOFT LABELS not available for {clf_name} (classifier does not support predict_proba)")

print("\n✓ Training and evaluation complete for all tasks and classifiers")

# KOD HÜCRESİ 8
# ==============

# ========================================================================
# STEP 4: Generate summary tables (like notebook 5)
# ========================================================================
print("\n" + "="*80)
print("STEP 4: GENERATE SUMMARY TABLES")
print("="*80)

# Create summary tables for each task
for task in TASKS:
    if task not in all_results_type3:
        continue
    
    print(f"\n{'-'*60}")
    print(f"Task: {task.upper()}")
    print(f"{'-'*60}")
    
    # Create summary DataFrame
    summary_rows = []
    for clf_name, result in all_results_type3[task].items():
        metrics = result['metrics']
        summary_rows.append({
            'classifier': clf_name,
            'task': task,
            'macro_f1': metrics.get('macro_f1', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'macro_precision': metrics.get('macro_precision', 0.0),
            'macro_recall': metrics.get('macro_recall', 0.0),
        })
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Display table
    print(f"\nSummary Table for {task.upper()}:")
    display(df_summary.style.format(precision=4))
    
    # Save table (ensure directory exists before saving)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f'summary_{task}.csv'
    df_summary.to_csv(table_path, index=False)
    print(f"  ✓ Saved table: {table_path.name}")
    
    # Save HTML version
    tables_dir.mkdir(parents=True, exist_ok=True)
    html_path = tables_dir / f'summary_{task}.html'
    df_summary.to_html(html_path, index=False, float_format='{:.4f}'.format)
    print(f"  ✓ Saved HTML: {html_path.name}")

# Create combined summary (all tasks)
print(f"\n{'-'*60}")
print("Combined Summary (All Tasks)")
print(f"{'-'*60}")

all_summary_rows = []
for task in TASKS:
    if task not in all_results_type3:
        continue
    for clf_name, result in all_results_type3[task].items():
        metrics = result['metrics']
        all_summary_rows.append({
            'classifier': clf_name,
            'task': task,
            'macro_f1': metrics.get('macro_f1', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
        })

df_all_summary = pd.DataFrame(all_summary_rows)

# Pivot table: Classifier × Task
if len(df_all_summary) > 0:
    df_pivot = df_all_summary.pivot(index='classifier', columns='task', values='macro_f1')
    
    print("\nPivot Table: Classifier × Task (Macro F1)")
    display(df_pivot.style.format(precision=4))
    
    # Save pivot table (ensure directory exists before saving)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pivot_path = tables_dir / 'summary_all_tasks_pivot.csv'
    df_pivot.to_csv(pivot_path)
    print(f"  ✓ Saved pivot table: {pivot_path.name}")
    
    # Save HTML version
    tables_dir.mkdir(parents=True, exist_ok=True)
    html_pivot_path = tables_dir / 'summary_all_tasks_pivot.html'
    df_pivot.to_html(html_pivot_path, float_format='{:.4f}'.format)
    print(f"  ✓ Saved HTML: {html_pivot_path.name}")

# Save complete results to JSON (ensure directory exists before saving)
results_dir.mkdir(parents=True, exist_ok=True)
results_json_path = results_dir / 'final_results_type3.json'
results_dict = {
    'method': 'early_fusion_60feat',
    'n_features': 60,
    'feature_breakdown': {
        'model_independent': 18,
        'model_dependent': 42,
        'models': len(MODELS),
        'features_per_model': 7
    },
    'tasks': TASKS,
    'classifiers': list(classifiers.keys()),
    'results': {
        task: {
            clf_name: {
                'metrics': result['metrics'],
                'n_test': len(test_features_60[task]['test']) if task in test_features_60 else 0
            }
            for clf_name, result in task_results.items()
        }
        for task, task_results in all_results_type3.items()
    }
}

with open(results_json_path, 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)

print(f"\n✓ Saved complete results: {results_json_path.name}")

print(f"\n{'='*80}")
print("FINAL EVALUATION TYPE 3 COMPLETE")
print(f"{'='*80}")
print("\nSummary:")
print("  - 60 features: 18 model-independent + 42 model-dependent (6 models × 7)")
print("  - Trained on Train+Dev combined data")
print("  - Evaluated on Test set (2 tasks: clarity, evasion)")
print("  - All 6 classifiers evaluated")
print("  - Results saved to FinalResultsType3 directory")
print("\nOutput locations:")
print(f"  - Test features: {test_features_dir}")
print(f"  - Predictions: {predictions_dir}")
print(f"  - Tables: {tables_dir}")
print(f"  - Results: {results_dir}")

# KOD HÜCRESİ 9
# ==============
# ============================================================================
# FINAL SUMMARY TABLES: Clarity, EvasionBasedClarity, and Annotator-Based Clarity
# ============================================================================
# Generate comprehensive summary tables similar to 3. and 5. notebooks

from src.models.hierarchical import evasion_to_clarity, evaluate_hierarchical_approach
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*80)
print("FINAL SUMMARY TABLES GENERATION")
print("="*80)

# ========================================================================
# STEP 1: Evaluate Evasion-Based Clarity and Annotator-Based Clarity
# ========================================================================
print("\nStep 1: Evaluating Evasion-Based Clarity and Annotator-Based Clarity...")

# Store all evaluation results (including clarity, evasion_based_clarity, annotator-based)
all_evaluation_results = {}  # {task_name: {classifier: {metrics}}}

# First, add direct clarity results
if 'clarity' in all_results_type3:
    all_evaluation_results['clarity'] = {
        clf_name: {'metrics': result['metrics']}
        for clf_name, result in all_results_type3['clarity'].items()
    }

# For evasion task, create evasion_based_clarity and annotator-based clarity
if 'evasion' in all_results_type3:
    print("\n  Processing evasion task for hierarchical evaluations...")
    
    # Load test split for evasion (to get annotator labels)
    try:
        test_ds_evasion = storage.load_split('test', task='evasion')
        test_ds_clarity = storage.load_split('test', task='clarity')
        
        # Get true clarity labels
        y_clarity_true_test = np.array([test_ds_clarity[i]['clarity_label'] for i in range(len(test_ds_clarity))])
        
        # Encode clarity labels
        le_clarity = LabelEncoder()
        y_clarity_true_encoded = le_clarity.fit_transform(y_clarity_true_test)
        clarity_label_list = CLARITY_LABELS
        
        # ====================================================================
        # 1.1: Evasion-Based Clarity (evasion predictions → clarity)
        # ====================================================================
        print("\n  1.1: Evaluating evasion_based_clarity...")
        all_evaluation_results['evasion_based_clarity'] = {}
        
        for clf_name, result in all_results_type3['evasion'].items():
            # Get evasion predictions (string labels)
            y_evasion_pred = result['predictions']
            
            # Map evasion predictions to clarity using hierarchical mapping
            hierarchical_metrics = evaluate_hierarchical_approach(
                np.zeros(len(y_evasion_pred), dtype=int),  # Dummy evasion_true (not used)
                y_evasion_pred,  # Evasion predictions (string labels)
                y_clarity_true_encoded,  # True clarity labels (encoded)
                EVASION_LABELS,
                clarity_label_list
            )
            
            all_evaluation_results['evasion_based_clarity'][clf_name] = {
                'metrics': hierarchical_metrics['metrics']
            }
            print(f"      {clf_name}: Macro F1 = {hierarchical_metrics['metrics'].get('macro_f1', 0.0):.4f}")
        
        # ====================================================================
        # 1.2: Annotator-Based Clarity (annotator1/2/3 evasion labels → clarity)
        # ====================================================================
        print("\n  1.2: Evaluating annotator-based clarity...")
        
        # Extract annotator labels from test dataset
        try:
            y_annotator1_evasion = np.array([test_ds_evasion[i]['annotator1'] for i in range(len(test_ds_evasion))])
            y_annotator2_evasion = np.array([test_ds_evasion[i]['annotator2'] for i in range(len(test_ds_evasion))])
            y_annotator3_evasion = np.array([test_ds_evasion[i]['annotator3'] for i in range(len(test_ds_evasion))])
            
            # Evaluate each annotator's labels mapped to clarity
            for annotator_name, y_annotator_evasion in [
                ('annotator1_based_clarity', y_annotator1_evasion),
                ('annotator2_based_clarity', y_annotator2_evasion),
                ('annotator3_based_clarity', y_annotator3_evasion)
            ]:
                print(f"\n    Evaluating {annotator_name}...")
                
                # Map annotator's evasion labels to clarity
                y_annotator_clarity_mapped = np.array([
                    evasion_to_clarity(str(ev_label)) for ev_label in y_annotator_evasion
                ])
                y_annotator_clarity_encoded = le_clarity.transform(y_annotator_clarity_mapped)
                
                # For each classifier, evaluate its evasion_based_clarity predictions against annotator's mapped clarity
                annotator_results = {}
                for clf_name, result in all_results_type3['evasion'].items():
                    # Get evasion predictions and map to clarity
                    y_evasion_pred = result['predictions']
                    hierarchical_metrics = evaluate_hierarchical_approach(
                        np.zeros(len(y_evasion_pred), dtype=int),
                        y_evasion_pred,
                        y_annotator_clarity_encoded,  # Compare against annotator's mapped clarity
                        EVASION_LABELS,
                        clarity_label_list
                    )
                    
                    annotator_results[clf_name] = {
                        'metrics': hierarchical_metrics['metrics']
                    }
                    print(f"      {clf_name}: Macro F1 = {hierarchical_metrics['metrics'].get('macro_f1', 0.0):.4f}")
                
                all_evaluation_results[annotator_name] = annotator_results
                
        except KeyError as e:
            print(f"    ⚠ WARNING: Could not find annotator columns in test dataset: {e}")
            print(f"    Skipping annotator-based clarity evaluations...")
            print(f"    Only clarity and evasion_based_clarity will be shown in tables.")

print("\n✓ All evaluations complete")

# ========================================================================
# STEP 2: Create Summary Tables
# ========================================================================
print("\nStep 2: Creating summary tables...")

# Define task order for tables
# If annotator columns exist, show all 5 tasks; otherwise show only 2
if 'annotator1_based_clarity' in all_evaluation_results:
    all_tasks = ['clarity', 'evasion_based_clarity', 'annotator1_based_clarity', 
                 'annotator2_based_clarity', 'annotator3_based_clarity']
    print(f"  Tasks: {len(all_tasks)} tasks (including annotator-based clarity)")
else:
    all_tasks = ['clarity', 'evasion_based_clarity']
    print(f"  Tasks: {len(all_tasks)} tasks (clarity and evasion_based_clarity only)")

# Create summary DataFrame
summary_rows = []
for task in all_tasks:
    if task not in all_evaluation_results:
        continue
    for clf_name, result in all_evaluation_results[task].items():
        metrics = result['metrics']
        summary_rows.append({
            'classifier': clf_name,
            'task': task,
            'macro_f1': metrics.get('macro_f1', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'macro_precision': metrics.get('macro_precision', 0.0),
            'macro_recall': metrics.get('macro_recall', 0.0),
        })

if not summary_rows:
    print("  ⚠ WARNING: No results available for summary tables")
else:
    df_summary = pd.DataFrame(summary_rows)
    
    # Remove duplicates (safety)
    df_summary = df_summary.drop_duplicates(subset=['classifier', 'task'], keep='first')
    
    # ====================================================================
    # Create Pivot Table: Classifier × Tasks
    # ====================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE: Classifier × Tasks (Macro F1)")
    print("="*80)
    
    try:
        df_pivot = df_summary.pivot(index='classifier', columns='task', values='macro_f1')
        
        # Display table
        display(df_pivot.style.format(precision=4))
        
        # Save table (ensure directory exists)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        pivot_csv_path = tables_dir / 'final_summary_classifier_wise.csv'
        df_pivot.to_csv(pivot_csv_path)
        print(f"\n  ✓ Saved CSV: {pivot_csv_path.name}")
        
        # Save HTML
        pivot_html_path = tables_dir / 'final_summary_classifier_wise.html'
        df_pivot.to_html(pivot_html_path, float_format='{:.4f}'.format)
        print(f"  ✓ Saved HTML: {pivot_html_path.name}")
        
    except Exception as e:
        print(f"  ⚠ Error creating pivot table: {e}")
        import traceback
        traceback.print_exc()
    
    # ====================================================================
    # Create Detailed Summary Table (all metrics)
    # ====================================================================
    print("\n" + "="*80)
    print("DETAILED SUMMARY TABLE: All Metrics")
    print("="*80)
    
    display(df_summary.style.format(precision=4))
    
    # Save detailed table
    tables_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = tables_dir / 'final_summary_detailed.csv'
    df_summary.to_csv(detailed_csv_path, index=False)
    print(f"\n  ✓ Saved detailed CSV: {detailed_csv_path.name}")
    
    detailed_html_path = tables_dir / 'final_summary_detailed.html'
    df_summary.to_html(detailed_html_path, index=False, float_format='{:.4f}'.format)
    print(f"  ✓ Saved detailed HTML: {detailed_html_path.name}")

print("\n" + "="*80)
print("FINAL SUMMARY TABLES COMPLETE")
print("="*80)
print(f"\nTables saved to: {tables_dir}")
print(f"  - final_summary_classifier_wise.csv/html (Pivot: Classifier × Tasks)")
print(f"  - final_summary_detailed.csv/html (All metrics)")
print(f"\nTasks included: {', '.join(all_tasks)}")

# KOD HÜCRESİ 10
# ==============