# -*- coding: utf-8 -*-
"""0_5_methodology_2_early_fusion_60_feature_6_classifier.py

Methodology 2: Early fusion with classifier-specific feature selection.
Early fusion of 60 features (18 model-independent + 42 model-dependent) with 40 features per classifier selected via greedy forward selection.
"""

# ============================================================================
# SETUP: Repository Clone, Drive Mount, and Path Configuration
# ============================================================================
# Setup steps:
# 1. Clean GitHub cache (removes old repository to ensure fresh clone)
# 2. Clean PyTorch cache (fixes import errors)
# 3. Clone repository from GitHub (if not already present)
# 4. Mount Google Drive for persistent data storage
# 5. Configure Python paths and initialize StorageManager
# 6. Load data splits and features created in previous notebooks

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

# ========================================================================
# STEP 0: Clean GitHub and PyTorch Cache (Ensure Fresh Environment)
# ========================================================================
print("="*80)
print("STEP 0: CLEANING CACHE (GitHub + PyTorch)")
print("="*80)

repo_dir = '/content/semeval-context-tree-modular'

# Remove old repository if exists (ensures fresh clone)
if os.path.exists(repo_dir):
    print(f"Removing old repository: {repo_dir}")
    try:
        shutil.rmtree(repo_dir)
        print("  ✓ Old repository removed successfully")
    except Exception as e:
        print(f"   Warning: Could not remove old repository: {e}")
        try:
            subprocess.run(['rm', '-rf', repo_dir], check=True, timeout=10)
            print("  ✓ Old repository removed via subprocess")
        except Exception as e2:
            print(f"   Warning: Alternative removal also failed: {e2}")
else:
    print("  ✓ No old repository found (clean state)")

# Clean Python import cache (remove cached modules)
print("\nCleaning Python import cache...")
modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('src.') or key.startswith('torch')]
if modules_to_remove:
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
        except:
            pass
    print(f"  ✓ Removed {len(modules_to_remove)} cached modules (src.* and torch.*)")

# Remove __pycache__ directories
import glob
pycache_dirs = glob.glob('/content/**/__pycache__', recursive=True)
if pycache_dirs:
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
        except:
            pass
    print(f"  ✓ Cleaned {len(pycache_dirs)} __pycache__ directories")

# CRITICAL: Reinstall PyTorch to fix "ValueError: module functions cannot set METH_CLASS or METH_STATIC"
print("\nReinstalling PyTorch to fix potential import errors...")
try:
    # Uninstall PyTorch first
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'],
                   capture_output=True, timeout=30)
    print("  ✓ Uninstalled old PyTorch")

    # Reinstall PyTorch (with CUDA support for Colab)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url',
                    'https://download.pytorch.org/whl/cu118'],
                   capture_output=True, timeout=120)
    print("  ✓ Reinstalled PyTorch with CUDA support")
except Exception as e:
    print(f"   Warning: PyTorch reinstall failed: {e}")
    print("  Continuing anyway (may cause issues if PyTorch is corrupted)")

print("\n✓ Cache cleaning complete")
print("="*80)

# ========================================================================
# STEP 1: Clone Repository from GitHub
# ========================================================================
print("\n" + "="*80)
print("STEP 1: CLONING REPOSITORY FROM GITHUB")
print("="*80)

# Repository configuration
repo_url = 'https://github.com/EonTechie/semeval-context-tree-modular.git'
zip_url = 'https://github.com/EonTechie/semeval-context-tree-modular/archive/refs/heads/main.zip'

# Clone repository (fresh clone after cache cleaning)
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
            print("  ✓ Repository cloned successfully via git")
            clone_success = True
            break
        else:
            print(f"   Git clone attempt {attempt + 1} failed: {result.stderr[:200]}")
            if attempt < max_retries - 1:
                time.sleep(3)
    except Exception as e:
        print(f"   Git clone attempt {attempt + 1} exception: {str(e)[:200]}")
        if attempt < max_retries - 1:
            time.sleep(3)

# Fallback: Download as ZIP if git clone fails
if not clone_success:
    print("\nGit clone failed. Downloading repository as ZIP archive...")
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
        print("  ✓ Repository downloaded and extracted successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to obtain repository: {e}")

# ========================================================================
# STEP 2: Mount Google Drive
# ========================================================================
print("\n" + "="*80)
print("STEP 2: MOUNTING GOOGLE DRIVE")
print("="*80)

# Mount Google Drive (if not already mounted)
try:
    drive.mount('/content/drive', force_remount=False)
    print("  ✓ Google Drive mounted")
except Exception:
    print("  ✓ Google Drive already mounted")
    pass  # Already mounted

# ========================================================================
# STEP 3: Configure Paths and Verify Repository
# ========================================================================
print("\n" + "="*80)
print("STEP 3: CONFIGURING PATHS AND VERIFYING REPOSITORY")
print("="*80)

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

print("  ✓ Repository structure verified")

# Add repository to Python path
sys.path.insert(0, str(BASE_PATH))
print("  ✓ Repository added to Python path")

# ========================================================================
# STEP 4: Import and Initialize (with error handling for PyTorch)
# ========================================================================
print("\n" + "="*80)
print("STEP 4: IMPORTING MODULES AND INITIALIZING")
print("="*80)

# Verify imports work (with PyTorch error handling)
try:
    from src.storage.manager import StorageManager
    print("  ✓ StorageManager imported")
except Exception as e:
    raise ImportError(f"Failed to import StorageManager: {e}")

try:
    from src.features.fusion import fuse_attention_features
    print("  ✓ fuse_attention_features imported")
except Exception as e:
    if "METH_CLASS" in str(e) or "METH_STATIC" in str(e) or "torch._C" in str(e):
        print(f"   PyTorch import error detected: {e}")
        print("  SOLUTION: Please restart runtime (Runtime → Restart runtime)")
        print("  Then run this cell again.")
        raise RuntimeError(
            "PyTorch import error. Please restart runtime and try again.\n"
            "Runtime → Restart runtime, then re-run this cell."
        )
    else:
        raise ImportError(f"Failed to import fuse_attention_features: {e}")

try:
    from src.models.classifiers import get_classifier_dict
    print("  ✓ get_classifier_dict imported")
except Exception as e:
    raise ImportError(f"Failed to import get_classifier_dict: {e}")

# Initialize StorageManager
storage = StorageManager(
    base_path=str(BASE_PATH),
    data_path=str(DATA_PATH),
    github_path=str(BASE_PATH)
)
print("  ✓ StorageManager initialized")

# ========================================================================
# SETUP COMPLETE
# ========================================================================
print("\n" + "="*80)
print("SETUP COMPLETE")
print("="*80)
print(f"  Repository: {BASE_PATH}")
print(f"  Data storage: {DATA_PATH}")
print(f"\nNOTE: Data splits will be loaded per-task (task-specific splits)")
print(f"      Clarity and Evasion have different splits due to majority voting")

# STEP 2
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

# STEP 3
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

# STEP 4
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

# STEP 6
# ==============
# ============================================================================
# FINAL EVALUATION ON TEST SET (TYPE 3)
# ============================================================================
# Extract test features (60 features), train on Train+Dev, evaluate on Test
# CRITICAL: CUDA REQUIRED - No CPU fallback. If CUDA unavailable, raises error.

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
import gc  # For garbage collection
import numpy as np

# ========================================================================
# CRITICAL: CUDA CHECK - NO CPU FALLBACK
# ========================================================================
if not torch.cuda.is_available():
    raise RuntimeError(
        " CUDA is REQUIRED for this cell. GPU runtime is mandatory.\n"
        "Please restart with GPU runtime. CPU fallback is NOT supported."
    )

device = torch.device('cuda')
print(f"✓ Device: {device}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

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

# Create all directories
test_features_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

print("✓ Created all Type3 output directories")

# ========================================================================
# STEP 2: Extract or load test features (60 features) - CHECKPOINT
# ========================================================================
print("\n" + "="*80)
print("STEP 2: TEST FEATURE EXTRACTION (60 FEATURES)")
print("="*80)
print("CRITICAL: If checkpoint exists, loads from Drive. Otherwise extracts on CUDA only.")

# Load sentiment pipeline for model-independent features
print("\nLoading sentiment analysis pipeline...")
sentiment_pipeline = None
try:
    from transformers import pipeline
    # CRITICAL: CUDA only (device=0)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0,  # CUDA only - no fallback
        return_all_scores=True
    )
    print("  ✓ Sentiment pipeline loaded on GPU")
except Exception as e:
    raise RuntimeError(
        f" Failed to load sentiment pipeline on GPU: {e}\n"
        "CUDA is required. Please ensure GPU runtime is active."
    )

metadata_keys = {
    'inaudible': 'inaudible',
    'multiple_questions': 'multiple_questions',
    'affirmative_questions': 'affirmative_questions'
}

# Store test features for each task
test_features_60 = {}  # {task: {'test': X_test_60}}

for task in TASKS:
    print(f"\n{'-'*60}")
    print(f"Task: {task.upper()}")
    print(f"{'-'*60}")

    # CRITICAL: Reset GPU state before each task (prevent CUDA errors)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  ✓ GPU state reset before task")

    # Load test split
    try:
        test_ds = storage.load_split('test', task=task)
        print(f"  Test set: {len(test_ds)} samples")
    except FileNotFoundError as e:
        print(f"   Test split not found for {task}: {e}")
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
        print(f"    → Extracting model-independent test features (GPU required)...")
        X_test_indep, _ = featurize_model_independent_features(
            test_ds,
            question_key='interview_question',
            answer_key='interview_answer',
            batch_size=32,
            show_progress=True,
            sentiment_pipeline=sentiment_pipeline,
            metadata_keys=metadata_keys,
        )
        # Save to checkpoint
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
            print(f"    {model_key}: → Extracting model-dependent test features (GPU required)...")

            # Retry mechanism for CUDA errors
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    # Aggressive GPU cleanup before each attempt
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    if retry_count > 0:
                        print(f"      Retry attempt {retry_count + 1}/{max_retries}...")
                        import time
                        time.sleep(2)  # Wait 2 seconds between retries

                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)

                    # CRITICAL: CUDA only - no CPU fallback
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

                    # Save to checkpoint immediately after successful extraction
                    test_features_dir.mkdir(parents=True, exist_ok=True)
                    np.save(test_dep_path, X_test_dep)
                    print(f"    {model_key}: ✓ Extracted and saved: {X_test_dep.shape}")
                    success = True

                    # Aggressive GPU cleanup after successful extraction
                    del model, tokenizer
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"    {model_key}: ✓ GPU memory cleared")

                except RuntimeError as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "device" in error_str:
                        retry_count += 1
                        # Aggressive cleanup on error
                        if 'model' in locals():
                            del model
                        if 'tokenizer' in locals():
                            del tokenizer
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        if retry_count >= max_retries:
                            raise RuntimeError(
                                f" Failed to extract {model_key} features after {max_retries} retries.\n"
                                f"CUDA error: {e}\n"
                                "Please restart runtime with fresh GPU and try again."
                            )
                        print(f"       CUDA error (attempt {retry_count}/{max_retries}): {str(e)[:100]}...")
                    else:
                        # Non-CUDA error - re-raise immediately
                        raise
                except Exception as e:
                    # Any other error - cleanup and re-raise
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    raise

            if not success:
                raise RuntimeError(
                    f" Failed to extract {model_key} features after {max_retries} retries.\n"
                    "Please restart runtime with fresh GPU and try again."
                )

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

    # Save complete test features to checkpoint
    test_features_dir.mkdir(parents=True, exist_ok=True)
    test_complete_path = test_features_dir / f'X_test_60feat_{task}.npy'
    np.save(test_complete_path, X_test_60)
    print(f"    ✓ Saved complete test features to: {test_complete_path.name}")

    # Store for later use
    test_features_60[task] = {
        'test': X_test_60
    }

    # CRITICAL: Clean GPU after each task (prevent CUDA errors in next task)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  ✓ GPU cleaned after task {task}")

print("\n✓ Test feature extraction complete for all tasks")

# CRITICAL: Clean sentiment pipeline at the end (prevent GPU memory leak)
if sentiment_pipeline is not None:
    del sentiment_pipeline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print("✓ Sentiment pipeline cleaned")

print("\n" + "="*80)
print("STEP 2 COMPLETE: All test features extracted/loaded successfully")
print("="*80)

# STEP 7
# ==============
# ========================================================================
# STEP 3: Train on Train+Dev and evaluate on Test (2 tasks, 6 classifiers)
# ========================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN ON TRAIN+DEV AND EVALUATE ON TEST")
print("="*80)
print("CRITICAL: This cell requires fused features from STEP 4.")
print("          If you see 'Fused features not found' errors,")
print("          please run STEP 4 first to create fused features.")

# Store all results for summary tables
all_results_type3 = {}  # {task: {classifier: {metrics, predictions, probabilities}}}

# Check if test features are available from STEP 6
if 'test_features_60' not in globals() or not test_features_60:
    print("\n ERROR: test_features_60 not found!")
    print("   Please run STEP 6 first to extract test features.")
    raise RuntimeError("test_features_60 not available. Run STEP 6 first.")

for task in TASKS:
    print(f"\n{'-'*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'-'*80}")

    # Check if test features are available for this task
    if task not in test_features_60:
        print(f"   Skipping {task}: Test features not available from STEP 6")
        print(f"     Please ensure STEP 6 completed successfully for {task}.")
        continue

    # Select appropriate label list and dataset key
    if task == 'clarity':
        label_list = CLARITY_LABELS
        label_key = 'clarity_label'
    else:  # evasion
        label_list = EVASION_LABELS
        label_key = 'evasion_label'

    # Load Train+Dev fused features (60 features) - from STEP 4
    # These were already created and saved in STEP 4
    print(f"\n  Loading fused features from STEP 4...")
    try:
        # Load fused features directly from STEP 4 (no need to reconstruct)
        X_train_60 = storage.load_fused_features(MODELS, task, 'train')
        X_dev_60 = storage.load_fused_features(MODELS, task, 'dev')

        # Verify shapes
        if X_train_60.shape[1] != 60:
            raise ValueError(f"Expected 60 features in train, got {X_train_60.shape[1]}")
        if X_dev_60.shape[1] != 60:
            raise ValueError(f"Expected 60 features in dev, got {X_dev_60.shape[1]}")

        print(f"  ✓ Loaded Train fused features: {X_train_60.shape} (60 features)")
        print(f"  ✓ Loaded Dev fused features: {X_dev_60.shape} (60 features)")

    except FileNotFoundError as e:
        print(f"\n   ERROR: Fused features not found for {task}!")
        print(f"  Error details: {e}")
        print(f"\n  SOLUTION:")
        print(f"  1. Go back to STEP 4")
        print(f"  2. Run STEP 4 completely (it will create fused features)")
        print(f"  3. Wait for STEP 4 to finish successfully")
        print(f"  4. Then come back and run STEP 7 again")
        print(f"\n  Expected file locations:")
        print(f"    - Train: {storage.data_path}/features/fused/X_train_fused_{'_'.join(MODELS)}_{task}.npy")
        print(f"    - Dev: {storage.data_path}/features/fused/X_dev_fused_{'_'.join(MODELS)}_{task}.npy")
        print(f"\n   Skipping {task} - cannot proceed without fused features.")
        continue

    except Exception as e:
        print(f"\n   ERROR loading train+dev features for {task}: {type(e).__name__}")
        print(f"  Error: {e}")
        print(f"\n  SOLUTION:")
        print(f"  1. Check if STEP 4 completed successfully")
        print(f"  2. Verify file paths and permissions")
        print(f"  3. Try running STEP 4 again")
        print(f"\n   Skipping {task} - cannot proceed without fused features.")
        continue

    # Load labels
    try:
        train_ds = storage.load_split('train', task=task)
        dev_ds = storage.load_split('dev', task=task)
        test_ds = storage.load_split('test', task=task)

        y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
        y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
        y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])

        # Verify label counts match feature counts
        if len(y_train) != X_train_60.shape[0]:
            raise ValueError(f"Train labels ({len(y_train)}) don't match features ({X_train_60.shape[0]})")
        if len(y_dev) != X_dev_60.shape[0]:
            raise ValueError(f"Dev labels ({len(y_dev)}) don't match features ({X_dev_60.shape[0]})")
        if len(y_test) != test_features_60[task]['test'].shape[0]:
            raise ValueError(f"Test labels ({len(y_test)}) don't match test features ({test_features_60[task]['test'].shape[0]})")

    except Exception as e:
        print(f"   ERROR loading labels for {task}: {type(e).__name__}")
        print(f"  Error: {e}")
        print(f"   Skipping {task} - cannot proceed without labels.")
        continue

    # Combine Train+Dev for final training
    X_train_full = np.vstack([X_train_60, X_dev_60])
    y_train_full = np.concatenate([y_train, y_dev])

    print(f"\n  Combined Train+Dev: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"  Test: {len(y_test)} samples, {test_features_60[task]['test'].shape[1]} features")

    # Get test features
    X_test_60 = test_features_60[task]['test']

    # Verify test features have 60 features
    if X_test_60.shape[1] != 60:
        print(f"   ERROR: Test features have {X_test_60.shape[1]} features, expected 60")
        print(f"   Skipping {task} - feature mismatch.")
        continue

    # Train all 6 classifiers and evaluate on test
    print(f"\n  Training and evaluating {len(classifiers)} classifiers...")
    all_results_type3[task] = {}

    for clf_name, clf in classifiers.items():
        print(f"\n    Classifier: {clf_name}")

        try:
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
                print(f"       SOFT LABELS not available for {clf_name} (classifier does not support predict_proba)")

        except Exception as e:
            print(f"       ERROR training/evaluating {clf_name} for {task}: {type(e).__name__}")
            print(f"      Error: {e}")
            print(f"       Skipping {clf_name} for {task}")
            continue

# Summary
print("\n" + "="*80)
print("STEP 3 SUMMARY")
print("="*80)
if all_results_type3:
    print(f"✓ Successfully completed training/evaluation for {len(all_results_type3)} task(s)")
    for task, results in all_results_type3.items():
        print(f"  - {task}: {len(results)} classifier(s) evaluated")
else:
    print(" WARNING: No tasks were successfully completed!")
    print("   Please check the errors above and ensure:")
    print("   1. STEP 4 completed successfully (creates fused features)")
    print("   2. STEP 6 completed successfully (creates test features)")
    print("   3. All required files are accessible")

print("\n✓ Training and evaluation complete for all tasks and classifiers")

# STEP 8
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

    # Reorder columns: clarity first, then evasion
    desired_order = ['clarity', 'evasion']

    # Only include columns that exist in the pivot table
    available_columns = [col for col in desired_order if col in df_pivot.columns]
    # Add any remaining columns that weren't in desired_order (alphabetically)
    remaining_columns = sorted([col for col in df_pivot.columns if col not in available_columns])
    column_order = available_columns + remaining_columns

    # Reorder columns
    df_pivot = df_pivot[column_order]

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

# STEP 9
# ==============
# ============================================================================
# FINAL SUMMARY TABLES: Clarity → EvasionBasedClarity → Annotator 1/2/3
# ============================================================================

from src.models.hierarchical import evasion_to_clarity, evaluate_hierarchical_approach
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from IPython.display import display
import numpy as np

print("\n" + "="*80)
print("FINAL SUMMARY TABLES GENERATION")
print("="*80)

# ========================================================================
# STEP 1: Collect All Evaluation Results
# ========================================================================
print("\nStep 1: Evaluating all clarity variants...")

all_evaluation_results = {}  # {task_name: {classifier: {metrics}}}

# ------------------------------------------------------------------------
# 1.1 Direct Clarity
# ------------------------------------------------------------------------
if 'clarity' in all_results_type3:
    all_evaluation_results['clarity'] = {
        clf: {'metrics': res['metrics']}
        for clf, res in all_results_type3['clarity'].items()
    }

# ------------------------------------------------------------------------
# 1.2 Evasion → Clarity + Annotator-Based Clarity
# ------------------------------------------------------------------------
if 'evasion' in all_results_type3:
    print("\nProcessing evasion-based evaluations...")

    test_ds = storage.load_split('test', task='evasion')

    y_clarity_true = np.array(
        [test_ds[i]['clarity_label'] for i in range(len(test_ds))]
    )

    le_clarity = LabelEncoder()
    y_clarity_true_enc = le_clarity.fit_transform(y_clarity_true)

    clarity_labels = CLARITY_LABELS
    n_test = len(test_ds)

    # ----------------------------
    # Evasion-Based Clarity
    # ----------------------------
    all_evaluation_results['evasion_based_clarity'] = {}

    for clf, res in all_results_type3['evasion'].items():
        y_evasion_pred = res['predictions']

        if len(y_evasion_pred) != n_test:
            continue

        hierarchical_metrics = evaluate_hierarchical_approach(
            np.zeros(n_test, dtype=int),
            y_evasion_pred,
            y_clarity_true_enc,
            EVASION_LABELS,
            clarity_labels
        )

        # Extract macro precision and recall from classification_report
        classification_report_dict = hierarchical_metrics.get('classification_report', {})
        macro_avg = classification_report_dict.get('macro avg', {})
        macro_precision = macro_avg.get('precision', 0.0)
        macro_recall = macro_avg.get('recall', 0.0)

        # Create metrics dict with all required fields
        metrics_dict = {
            'accuracy': hierarchical_metrics.get('accuracy', 0.0),
            'macro_f1': hierarchical_metrics.get('macro_f1', 0.0),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'weighted_f1': hierarchical_metrics.get('weighted_f1', 0.0),
        }

        all_evaluation_results['evasion_based_clarity'][clf] = {
            'metrics': metrics_dict
        }

    # ----------------------------
    # Annotator-Based Clarity
    # ----------------------------
    annotator_sources = {
        'annotator1_based_clarity': 'annotator1',
        'annotator2_based_clarity': 'annotator2',
        'annotator3_based_clarity': 'annotator3',
    }

    for task_name, col in annotator_sources.items():
        y_ann_evasion = np.array(
            [test_ds[i][col] for i in range(n_test)]
        )

        y_ann_clarity = np.array(
            [evasion_to_clarity(str(x)) for x in y_ann_evasion]
        )
        y_ann_clarity_enc = le_clarity.transform(y_ann_clarity)

        all_evaluation_results[task_name] = {}

        for clf, res in all_results_type3['evasion'].items():
            y_evasion_pred = res['predictions']

            if len(y_evasion_pred) != n_test:
                continue

            hierarchical_metrics = evaluate_hierarchical_approach(
                np.zeros(n_test, dtype=int),
                y_evasion_pred,
                y_ann_clarity_enc,
                EVASION_LABELS,
                clarity_labels
            )

            # Extract macro precision and recall from classification_report
            classification_report_dict = hierarchical_metrics.get('classification_report', {})
            macro_avg = classification_report_dict.get('macro avg', {})
            macro_precision = macro_avg.get('precision', 0.0)
            macro_recall = macro_avg.get('recall', 0.0)

            # Create metrics dict with all required fields
            metrics_dict = {
                'accuracy': hierarchical_metrics.get('accuracy', 0.0),
                'macro_f1': hierarchical_metrics.get('macro_f1', 0.0),
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'weighted_f1': hierarchical_metrics.get('weighted_f1', 0.0),
            }

            all_evaluation_results[task_name][clf] = {
                'metrics': metrics_dict
            }

print("\n✓ All evaluations collected")

# ========================================================================
# STEP 2: Build Summary DataFrame
# ========================================================================
print("\nStep 2: Creating summary tables...")

TASK_ORDER = [
    'clarity',
    'evasion_based_clarity',
    'annotator1_based_clarity',
    'annotator2_based_clarity',
    'annotator3_based_clarity'
]

rows = []
for task in TASK_ORDER:
    if task not in all_evaluation_results:
        continue
    for clf, res in all_evaluation_results[task].items():
        m = res['metrics']
        rows.append({
            'classifier': clf,
            'task': task,
            'macro_f1': m.get('macro_f1', 0.0),
            'accuracy': m.get('accuracy', 0.0),
            'macro_precision': m.get('macro_precision', 0.0),
            'macro_recall': m.get('macro_recall', 0.0),
        })

df_summary = pd.DataFrame(rows).drop_duplicates(
    subset=['classifier', 'task'],
    keep='first'
)

# ========================================================================
# STEP 3: Pivot Table (Classifier × Tasks) — ORDER FIXED
# ========================================================================
print("\n" + "="*80)
print("FINAL SUMMARY TABLE: Classifier × Tasks (Macro F1)")
print("="*80)

df_pivot = (
    df_summary
    .pivot(index='classifier', columns='task', values='macro_f1')
    .reindex(columns=[t for t in TASK_ORDER if t in df_summary['task'].unique()])
)

display(df_pivot.style.format(precision=4))

tables_dir.mkdir(parents=True, exist_ok=True)

df_pivot.to_csv(tables_dir / 'final_summary_classifier_wise.csv')
df_pivot.to_html(
    tables_dir / 'final_summary_classifier_wise.html',
    float_format='{:.4f}'.format
)

print("✓ Saved pivot tables")

# ========================================================================
# STEP 4: Detailed Table
# ========================================================================
print("\n" + "="*80)
print("DETAILED SUMMARY TABLE: All Metrics")
print("="*80)

display(df_summary.style.format(precision=4))

df_summary.to_csv(
    tables_dir / 'final_summary_detailed.csv',
    index=False
)
df_summary.to_html(
    tables_dir / 'final_summary_detailed.html',
    index=False,
    float_format='{:.4f}'.format
)

print("\n" + "="*80)
print("FINAL SUMMARY TABLES COMPLETE")
print("="*80)
print(f"Tasks order: {', '.join([t for t in TASK_ORDER if t in df_summary['task'].unique()])}")

