# KOD HÜCRESİ 1
# ==============
# ============================================================================
# SETUP: Repository Clone, Drive Mount, and Path Configuration
# ============================================================================
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
import pandas as pd

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
    from src.models.classifiers import get_classifier_dict
    from src.features.extraction import get_feature_names
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.base import clone
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

# Create ablation results directory
ablation_dir = DATA_PATH / 'results' / 'FinalResultsType2' / 'ablation'
ablation_dir.mkdir(parents=True, exist_ok=True)

print("Setup complete")
print(f"  Repository: {BASE_PATH}")
print(f"  Data storage: {DATA_PATH}")
print(f"  Ablation results: {ablation_dir}")

# ============================================================================
# KOD HÜCRESİ 2
# ==============
# ============================================================================
# REPRODUCIBILITY SETUP: Set Random Seeds for All Libraries
# ============================================================================
from src.utils.reproducibility import set_all_seeds

# Set all random seeds to 42 for full reproducibility
# deterministic=True ensures PyTorch operations are deterministic (slower but fully reproducible)
set_all_seeds(seed=42, deterministic=True)

print("✓ Reproducibility configured: All random seeds set to 42")
print("✓ PyTorch deterministic mode enabled")
print("\nNOTE: If you encounter performance issues or non-deterministic behavior,")
print("      you can set deterministic=False in set_all_seeds() call above.")

# ============================================================================
# KOD HÜCRESİ 3
# ==============
# ============================================================================
# CONFIGURE MODELS, TASKS, AND CLASSIFIERS
# ============================================================================
# Check if get_classifier_dict is imported (from Cell 1 - Setup)
if 'get_classifier_dict' not in globals():
    raise NameError(
        "get_classifier_dict not found. Please run Cell 1 (Setup) first.\n"
        "Cell 1 imports get_classifier_dict from src.models.classifiers."
    )

MODELS = ['bert', 'bert_political', 'bert_ambiguity', 'roberta', 'deberta', 'xlnet']
# NOTE: Only clarity and hierarchical_evasion_to_clarity for greedy selection
# 'evasion' task is NOT included (it's only used for training in 3. notebook)
# Best classifier selection happens in 4. notebook, not here
TASKS = ['clarity', 'hierarchical_evasion_to_clarity']  # 2 tasks for greedy selection

# Label mappings for each task
CLARITY_LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
EVASION_LABELS = ['Claims ignorance', 'Clarification', 'Declining to answer',
                  'Deflection', 'Dodging', 'Explicit',
                  'General', 'Implicit', 'Partial/half-answer']

# Initialize classifiers with fixed random seed for reproducibility
# Includes MLP (Multi-Layer Perceptron) as requested
classifiers = get_classifier_dict(random_state=42)

print("="*80)
print("CONFIGURATION")
print("="*80)
print(f"  Models: {len(MODELS)} models")
print(f"    {MODELS}")
print(f"  Tasks: {len(TASKS)} tasks")
print(f"    {TASKS}")
print(f"  Classifiers: {len(classifiers)} classifiers")
print(f"    {list(classifiers.keys())}")
print(f"  Total combinations per task: {len(MODELS)} × {len(classifiers)} = {len(MODELS) * len(classifiers)}")
print(f"  Evaluation set: Dev set (not test)")
print("="*80)

# ============================================================================
# KOD HÜCRESİ 4
# ==============
# ============================================================================
# SINGLE-FEATURE ABLATION STUDY
# ============================================================================
def eval_single_feature(X_train, X_dev, y_train, y_dev, feature_idx, clf):
    """
    Evaluate a single feature using a classifier.

    This function trains a classifier using only one feature and evaluates its
    performance on the dev set. StandardScaler is applied to normalize the
    single feature before classification.

    Args:
        X_train: Training feature matrix (N, F) where F is total number of features
        X_dev: Dev feature matrix (M, F)
        y_train: Training labels (N,)
        y_dev: Dev labels (M,)
        feature_idx: Index of the feature to evaluate (0 to F-1)
        clf: Classifier instance (will be cloned to avoid state issues)

    Returns:
        Macro F1 score on dev set (float)
    """
    # Encode labels to numeric (required for MLP, XGBoost, LightGBM)
    # This matches the approach in siparismaili01 notebook
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_dev_encoded = label_encoder.transform(y_dev)

    # Select only the specified feature (single column)
    X_train_f = X_train[:, [feature_idx]]
    X_dev_f = X_dev[:, [feature_idx]]

    # Pipeline with scaling (critical for single features to work properly)
    # StandardScaler normalizes the feature to have zero mean and unit variance
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(clf))  # Clone to avoid modifying the original classifier
    ])

    # Train on single feature and evaluate on dev set
    pipe.fit(X_train_f, y_train_encoded)
    pred = pipe.predict(X_dev_f)
    macro_f1 = f1_score(y_dev_encoded, pred, average='macro')

    return macro_f1

# Check if required variables are defined (from Cell 2 - Configuration)
if 'TASKS' not in globals() or 'MODELS' not in globals() or 'classifiers' not in globals():
    raise NameError(
        "Required variables not defined. Please run Cell 2 (Configuration) first.\n"
        "Cell 2 defines: TASKS, MODELS, CLARITY_LABELS, EVASION_LABELS, and classifiers."
    )

# Check if storage is defined (from Cell 1 - Setup)
if 'storage' not in globals():
    raise NameError(
        "storage not found. Please run Cell 1 (Setup) first.\n"
        "Cell 1 initializes StorageManager as 'storage'."
    )

print("="*80)
print("SINGLE-FEATURE ABLATION STUDY")
print("="*80)
print("Evaluating each feature individually across all model×task×classifier combinations")
print(f"Total evaluations: {len(TASKS)} tasks × {len(MODELS)} models × {len(classifiers)} classifiers × 19 features")
print("This may take 15-30 minutes depending on your hardware...\n")

# Store all ablation results
# Each entry contains: model, task, classifier, feature, feature_idx, macro_f1
ablation_results = []

for task in TASKS:
    print(f"\n{'='*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*80}")

    # Select appropriate label list and dataset key based on task
    if task == 'clarity':
        label_list = CLARITY_LABELS
        label_key = 'clarity_label'
        task_for_split = 'clarity'
    elif task == 'evasion':
        label_list = EVASION_LABELS
        label_key = 'evasion_label'
        task_for_split = 'evasion'
    else:  # hierarchical_evasion_to_clarity
        # For hierarchical task, we need to load evasion dev set to get clarity labels
        # (hierarchical uses evasion predictions mapped to clarity labels)
        label_list = CLARITY_LABELS
        label_key = 'clarity_label'
        # We'll load from evasion dev set (same filtered samples)
        task_for_split = 'evasion'

    # Load task-specific splits
    # For hierarchical task, we load evasion split (which has clarity labels)
    train_ds = storage.load_split('train', task=task_for_split)
    dev_ds = storage.load_split('dev', task=task_for_split)

    # Extract labels
    y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
    y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])

    print(f"  Train: {len(y_train)} samples")
    print(f"  Dev: {len(y_dev)} samples")

    # Get feature names directly from extraction module (same for all models)
    # This avoids dependency on metadata files in GitHub
    # Feature names are the same across all models (19 Context Tree features)
    feature_names = get_feature_names()
    n_features = len(feature_names)

    print(f"  Features: {n_features} features")
    print(f"  Feature names: {feature_names}\n")

    # For hierarchical task, we need to use evasion features
    # (hierarchical approach uses evasion predictions, so we evaluate on evasion features)
    feature_task = 'evasion' if task == 'hierarchical_evasion_to_clarity' else task

    # For each model
    for model in MODELS:
        print(f"  Model: {model}")

        # Load features
        # For hierarchical task, use evasion features (since we're evaluating
        # how well evasion features predict clarity via hierarchical mapping)
        try:
            X_train = storage.load_features(model, feature_task, 'train')
            X_dev = storage.load_features(model, feature_task, 'dev')
        except FileNotFoundError:
            print(f"    ⚠ Features not found for {model} × {feature_task}, skipping...")
            continue

        # Verify feature count matches
        if X_train.shape[1] != n_features:
            print(f"    ⚠ Feature count mismatch: expected {n_features}, got {X_train.shape[1]}, skipping...")
            continue

        # For each classifier
        for clf_name, clf in classifiers.items():
            print(f"    Classifier: {clf_name}")

            # Evaluate each feature individually
            for feature_idx, feature_name in enumerate(feature_names):
                try:
                    macro_f1 = eval_single_feature(
                        X_train, X_dev,
                        y_train, y_dev,
                        feature_idx, clf
                    )

                    ablation_results.append({
                        'model': model,
                        'task': task,
                        'classifier': clf_name,
                        'feature': feature_name,
                        'feature_idx': feature_idx,
                        'macro_f1': float(macro_f1)
                    })
                except Exception as e:
                    print(f"      ⚠ Error evaluating feature {feature_name}: {e}")
                    continue

print(f"\n{'='*80}")
print("SINGLE-FEATURE ABLATION COMPLETE")
print(f"{'='*80}")
print(f"Total evaluations completed: {len(ablation_results)}")
print(f"Expected: {len(TASKS)} tasks × {len(MODELS)} models × {len(classifiers)} classifiers × {n_features} features = {len(TASKS) * len(MODELS) * len(classifiers) * n_features}")

# ============================================================================
# KOD HÜCRESİ 5
# ==============
# ============================================================================
# FEATURE RANKING AND STATISTICAL ANALYSIS
# ============================================================================
# Check if ablation_results exists (from Cell 4 - Single-Feature Ablation)
if 'ablation_results' not in globals():
    raise NameError(
        "ablation_results not found. Please run Cell 4 (Single-Feature Ablation) first.\n"
        "Cell 4 performs the ablation study and creates ablation_results list."
    )

# Check if storage and ablation_dir are defined (from Cell 1 - Setup)
if 'storage' not in globals():
    raise NameError(
        "storage not found. Please run Cell 1 (Setup) first.\n"
        "Cell 1 initializes StorageManager as 'storage'."
    )
if 'ablation_dir' not in globals():
    raise NameError(
        "ablation_dir not found. Please run Cell 1 (Setup) first.\n"
        "Cell 1 creates ablation_dir directory."
    )

df_ablation = pd.DataFrame(ablation_results)

if len(df_ablation) == 0:
    print("⚠ No ablation results found. Make sure Cell 4 completed successfully.")
    print("  You need to run Cell 4 (Single-Feature Ablation) first.")
else:
    print("="*80)
    print("FEATURE RANKING AND STATISTICAL ANALYSIS")
    print("="*80)
    print(f"Total ablation results: {len(df_ablation)} evaluations")
    print(f"Expected per task: {len(MODELS)} models × {len(classifiers)} classifiers × 19 features = {len(MODELS) * len(classifiers) * 19}")

    # Save raw ablation results for each task
    print(f"\n{'='*80}")
    print("SAVING RAW ABLATION RESULTS")
    print(f"{'='*80}")

    for task in TASKS:
        df_task = df_ablation[df_ablation['task'] == task]
        if len(df_task) > 0:
            # Ensure directory exists before saving
            ablation_dir.mkdir(parents=True, exist_ok=True)
            csv_path = ablation_dir / f'single_feature_{task}.csv'
            df_task.to_csv(csv_path, index=False)
            print(f"  Saved {task}: {len(df_task)} evaluations → {csv_path}")

    # ========================================================================
    # STATISTICAL AGGREGATION AND WEIGHTED SCORE CALCULATION
    # ========================================================================
    # Aggregate results across all 36 model×classifier combinations for each feature
    # Compute comprehensive statistics and calculate weighted score for ranking

    print(f"\n{'='*80}")
    print("STATISTICAL AGGREGATION AND FEATURE RANKING")
    print(f"{'='*80}")
    print("Computing statistics across all 36 model×classifier combinations...")

    # Calculate comprehensive statistics for each feature×task combination
    # Using 'median' in addition to mean/std/min/max to get more robust statistics
    df_stats = df_ablation.groupby(['task', 'feature'])['macro_f1'].agg([
        'min',      # Minimum F1 (worst-case)
        'median',   # Median F1 (typical performance)
        'mean',     # Mean F1 (average performance)
        'std',      # Standard deviation (consistency)
        'max',      # Maximum F1 (best-case, same as best_f1)
        'count'     # Number of evaluations (should be 36)
    ]).reset_index()

    # Rename columns for clarity
    df_stats.columns = ['task', 'feature', 'min_f1', 'median_f1', 'mean_f1', 'std_f1', 'best_f1', 'runs']

    # Calculate weighted score
    # Formula: weighted_score = 0.5*mean + 0.3*best + 0.2*(1 - normalized_std)
    # This balances:
    # - Average performance (50% weight)
    # - Peak performance (30% weight)
    # - Consistency (20% weight, lower std = higher score)
    #
    # Normalize std by mean to account for scale differences:
    # normalized_std = std_f1 / (mean_f1 + epsilon)
    # where epsilon prevents division by zero
    EPSILON = 1e-6
    df_stats['normalized_std'] = df_stats['std_f1'] / (df_stats['mean_f1'] + EPSILON)

    # Calculate weighted score
    # Higher is better: we want high mean, high best, and low std (high 1-normalized_std)
    df_stats['weighted_score'] = (
        0.5 * df_stats['mean_f1'] +
        0.3 * df_stats['best_f1'] +
        0.2 * (1 - df_stats['normalized_std'])
    )

    # Sort by weighted_score (descending) for ranking
    # Secondary sort by mean_f1 for tie-breaking
    df_stats = df_stats.sort_values(['weighted_score', 'mean_f1'], ascending=False)

    # ========================================================================
    # DISPLAY AND SAVE RANKINGS FOR EACH TASK
    # ========================================================================

    for task in TASKS:
        print(f"\n{'='*80}")
        print(f"TASK: {task.upper()} - FEATURE RANKING")
        print(f"{'='*80}")

        df_task = df_stats[df_stats['task'] == task].copy()

        if len(df_task) == 0:
            print(f"  ⚠ No results found for task: {task}")
            continue

        # Round all numeric columns for display
        numeric_cols = ['min_f1', 'median_f1', 'mean_f1', 'std_f1', 'best_f1', 'runs', 'normalized_std', 'weighted_score']
        df_task[numeric_cols] = df_task[numeric_cols].round(4)

        # Display top 15 features with all statistics
        print(f"\nTop 15 Features (ranked by weighted_score):")
        print("Weighted Score Formula: 0.5*mean + 0.3*best + 0.2*(1 - normalized_std)")
        print("\nColumns:")
        print("  - min_f1: Minimum Macro F1 across 36 combinations (worst-case)")
        print("  - median_f1: Median Macro F1 (typical performance)")
        print("  - mean_f1: Mean Macro F1 (average performance)")
        print("  - std_f1: Standard deviation (lower = more consistent)")
        print("  - best_f1: Maximum Macro F1 (best-case)")
        print("  - runs: Number of evaluations (should be 36)")
        print("  - normalized_std: std_f1 / mean_f1 (scale-normalized consistency)")
        print("  - weighted_score: Combined score for ranking\n")

        from IPython.display import display
        display(df_task[['feature', 'min_f1', 'median_f1', 'mean_f1', 'std_f1', 'best_f1', 'runs', 'normalized_std', 'weighted_score']].head(15))

        # Save complete ranking to CSV
        # Ensure directory exists before saving
        ablation_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ablation_dir / f'feature_ranking_{task}.csv'
        df_task.to_csv(csv_path, index=False)
        print(f"\n  ✓ Saved complete ranking: {csv_path}")
        print(f"    Total features ranked: {len(df_task)}")
        print(f"    Expected runs per feature: 36 (6 models × 6 classifiers)")

        # Verify data completeness
        incomplete = df_task[df_task['runs'] < 36]
        if len(incomplete) > 0:
            print(f"    ⚠ Warning: {len(incomplete)} features have incomplete data (< 36 runs)")

    # ========================================================================
    # TOP-K FEATURE SELECTION FOR EARLY FUSION
    # ========================================================================
    # Select top-K features for each task to use in Early Fusion
    # These features will be used across all models in Early Fusion

    print(f"\n{'='*80}")
    print("TOP-K FEATURE SELECTION FOR EARLY FUSION")
    print(f"{'='*80}")
    print("Selecting top-K features for each task (to be used in Early Fusion)")

    TOP_K_FEATURES = 10  # Number of top features to select for Early Fusion

    selected_features_for_fusion = {}

    for task in TASKS:
        df_task = df_stats[df_stats['task'] == task].copy()

        if len(df_task) == 0:
            print(f"  ⚠ No ranking data found for task: {task}")
            continue

        # Select top-K features by weighted_score
        top_k_features = df_task.head(TOP_K_FEATURES)['feature'].tolist()

        selected_features_for_fusion[task] = {
            'top_k': TOP_K_FEATURES,
            'features': top_k_features,
            'ranking': df_task.head(TOP_K_FEATURES)[['feature', 'weighted_score', 'mean_f1', 'best_f1', 'std_f1']].to_dict('records')
        }

        print(f"\n  {task.upper()} - Top {TOP_K_FEATURES} Features:")
        for i, feat in enumerate(top_k_features, 1):
            row = df_task[df_task['feature'] == feat].iloc[0]
            print(f"    {i:2d}. {feat}")
            print(f"        weighted_score={row['weighted_score']:.4f}, mean_f1={row['mean_f1']:.4f}, best_f1={row['best_f1']:.4f}")

    # Save selected features for Early Fusion
    import json
    # Ensure directory exists before saving
    ablation_dir.mkdir(parents=True, exist_ok=True)
    fusion_features_path = ablation_dir / 'selected_features_for_early_fusion.json'
    with open(fusion_features_path, 'w') as f:
        json.dump(selected_features_for_fusion, f, indent=2)

    print(f"\n{'='*80}")
    print("FEATURE RANKING COMPLETE")
    print(f"{'='*80}")
    print("Rankings saved separately for each task:")
    for task in TASKS:
        print(f"  - {task}: {ablation_dir / f'feature_ranking_{task}.csv'}")
    print(f"\nTop-K features for Early Fusion saved:")
    print(f"  - {fusion_features_path}")
    print(f"  - Top {TOP_K_FEATURES} features per task (to be used across all models in Early Fusion)")

# ============================================================================
# KOD HÜCRESİ 6
# ==============
# ============================================================================
# GREEDY FORWARD SELECTION (OPTIONAL)
# ============================================================================
import json
from tqdm import tqdm

def greedy_forward_selection(X_train, X_dev, y_train, y_dev, feature_names,
                            seed_features, clf, max_features=None):
    """
    Greedy forward selection: iteratively add best feature

    Args:
        X_train: Training features
        X_dev: Dev features
        y_train: Training labels
        y_dev: Dev labels
        feature_names: List of feature names
        seed_features: Initial feature set (list of feature names)
        clf: Classifier instance
        max_features: Maximum number of features to select (None = all)

    Returns:
        selected_features: List of selected feature names
        trajectory: List of (n_features, macro_f1) tuples
    """
    # Encode labels to numeric (required for MLP, XGBoost, LightGBM)
    # This matches the approach in siparismaili01 notebook
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_dev_encoded = label_encoder.transform(y_dev)

    selected_indices = [feature_names.index(f) for f in seed_features]
    available_indices = [i for i in range(len(feature_names)) if i not in selected_indices]

    trajectory = []

    # Evaluate initial set
    X_train_selected = X_train[:, selected_indices]
    X_dev_selected = X_dev[:, selected_indices]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(clf))
    ])
    pipe.fit(X_train_selected, y_train_encoded)
    pred = pipe.predict(X_dev_selected)
    current_f1 = f1_score(y_dev_encoded, pred, average='macro')
    trajectory.append((len(selected_indices), current_f1))

    # Greedy selection
    max_iter = max_features if max_features else len(available_indices)

    for iteration in tqdm(range(max_iter), desc="Greedy selection"):
        best_f1 = current_f1
        best_idx = None

        # Try each available feature
        for idx in available_indices:
            candidate_indices = selected_indices + [idx]
            X_train_candidate = X_train[:, candidate_indices]
            X_dev_candidate = X_dev[:, candidate_indices]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clone(clf))
            ])
            pipe.fit(X_train_candidate, y_train_encoded)
            pred = pipe.predict(X_dev_candidate)
            candidate_f1 = f1_score(y_dev_encoded, pred, average='macro')

            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_idx = idx

        # If no improvement, stop
        if best_idx is None:
            break

        # Add best feature
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
        current_f1 = best_f1
        trajectory.append((len(selected_indices), current_f1))

    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, trajectory

# Check if required variables are defined
if 'df_stats' not in globals():
    raise NameError(
        "df_stats not found. Please run Cell 5 (Feature Ranking) first.\n"
        "Cell 5 performs statistical analysis and creates df_stats DataFrame."
    )
if 'df_ablation' not in globals():
    raise NameError(
        "df_ablation not found. Please run Cell 5 (Feature Ranking) first.\n"
        "Cell 5 creates df_ablation DataFrame from ablation_results."
    )
if 'TASKS' not in globals() or 'MODELS' not in globals() or 'classifiers' not in globals():
    raise NameError(
        "Required variables not defined. Please run Cell 3 (Configuration) first."
    )
if 'storage' not in globals() or 'ablation_dir' not in globals():
    raise NameError(
        "storage or ablation_dir not found. Please run Cell 1 (Setup) first."
    )

print("="*80)
print("GREEDY FORWARD SELECTION")
print("="*80)
print("Starting with top-K features from weighted score ranking (Cell 5)\n")

# Use global top 10 as seed (same as selected_features_for_early_fusion.json)
TOP_K_SEED = 10  # Start with global top 10 features (from weighted_score ranking)
MAX_FEATURES = 20  # Maximum features to select (global 10 + classifier-specific 10)

selected_features_dict = {}
greedy_trajectories = {}

for task in TASKS:
    print(f"\n{'='*80}")
    print(f"TASK: {task.upper()} - GREEDY FORWARD SELECTION")
    print(f"{'='*80}")

    # Get top-K features for this task (from Cell 5, ranked by weighted_score)
    df_task_stats = df_stats[df_stats['task'] == task].copy()

    if len(df_task_stats) == 0:
        print(f"  ⚠ No ranking data found for task: {task}")
        continue

    # Select top-K features by weighted_score
    top_k_features = df_task_stats.head(TOP_K_SEED)['feature'].tolist()

    print(f"  Top {TOP_K_SEED} seed features (by weighted_score):")
    for i, feat in enumerate(top_k_features, 1):
        row = df_task_stats[df_task_stats['feature'] == feat].iloc[0]
        print(f"    {i}. {feat}")
        print(f"       weighted_score={row['weighted_score']:.4f}, mean={row['mean_f1']:.4f}, best={row['best_f1']:.4f}, std={row['std_f1']:.4f}")

    # Determine which task to use for loading data
    if task == 'hierarchical_evasion_to_clarity':
        # Hierarchical task: use evasion features but evaluate against clarity labels
        # (hierarchical approach maps evasion predictions to clarity)
        data_task = 'evasion'  # Use evasion features and splits
        label_key = 'clarity_label'  # But evaluate against clarity labels
    elif task == 'clarity':
        data_task = 'clarity'
        label_key = 'clarity_label'
    else:  # evasion
        data_task = 'evasion'
        label_key = 'evasion_label'

    # Load task-specific splits
    train_ds = storage.load_split('train', task=data_task)
    dev_ds = storage.load_split('dev', task=data_task)

    # Extract labels
    y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
    y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])

    # Get feature names directly from extraction module (same for all tasks and models)
    # This avoids dependency on metadata files in GitHub
    feature_names = get_feature_names()

    # For each model, use the best classifier for this model×task
    for model in MODELS:
        try:
            X_train = storage.load_features(model, data_task, 'train')
            X_dev = storage.load_features(model, data_task, 'dev')
        except FileNotFoundError:
            print(f"  ⚠ Features not found for {model} × {data_task}, skipping...")
            continue

        # Find all classifiers for this model×task from ablation results
        df_model_task = df_ablation[
            (df_ablation['task'] == task) &
            (df_ablation['model'] == model)
        ]

        if len(df_model_task) == 0:
            continue

        # Get classifier scores (by mean F1 across all features)
        classifier_scores = df_model_task.groupby('classifier')['macro_f1'].mean().sort_values(ascending=False)

        # Run greedy selection for EACH classifier (classifier-specific)
        # IMPORTANT: Don't select best classifier here - save ALL classifier results
        # Best classifier selection happens in 4. notebook based on 3. notebook F1 scores
        print(f"\n  {model.upper()} - Running greedy for {len(classifier_scores)} classifiers:")

        for clf_name, clf_mean_f1 in classifier_scores.items():
            clf = classifiers[clf_name]

            # Run greedy selection (starts with global top 10, adds up to 10 more)
            # Each classifier gets: Global top 10 (seed) + classifier-specific greedy 10 = max 20
            selected_features, trajectory = greedy_forward_selection(
                X_train, X_dev, y_train, y_dev,
                feature_names, top_k_features, clf,
                max_features=MAX_FEATURES  # Max 20: global 10 + classifier-specific 10
            )

            final_f1 = trajectory[-1][1] if trajectory else 0.0
            print(f"    {clf_name}: {len(selected_features)} features, F1={final_f1:.4f}")

            # Save EACH classifier's greedy selection (key: model_task_classifier)
            key = f"{model}_{task}_{clf_name}"
            selected_features_dict[key] = {
                'model': model,
                'task': task,
                'classifier': clf_name,
                'selected_features': selected_features,
                'n_features': len(selected_features),
                'greedy_f1': final_f1
            }

            greedy_trajectories[key] = trajectory

            # Save trajectory for each classifier
            # Ensure directory exists before saving
            ablation_dir.mkdir(parents=True, exist_ok=True)
            df_traj = pd.DataFrame(trajectory, columns=['n_features', 'macro_f1'])
            csv_path = ablation_dir / f'greedy_trajectory_{model}_{task}_{clf_name}.csv'
            df_traj.to_csv(csv_path, index=False)
            print(f"      Saved trajectory: {csv_path.name}")

# Save selected features
if selected_features_dict:
    # Ensure directory exists before saving
    ablation_dir.mkdir(parents=True, exist_ok=True)
    json_path = ablation_dir / 'selected_features_all.json'
    with open(json_path, 'w') as f:
        json.dump(selected_features_dict, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Saved selected features: {json_path}")
    print(f"{'='*80}")

print("\n" + "="*80)
print("ABLATION STUDY COMPLETE")
print("="*80)
print("\nSummary:")
print("  ✓ Single-feature ablation completed")
print("  ✓ Feature rankings generated")
print("  ✓ Greedy forward selection completed")
print("  ✓ All results saved to Google Drive")

# ============================================================================
# KOD HÜCRESİ 7
# ==============
# ============================================================================
# CLASSIFIER-SPECIFIC FEATURE SELECTION: Global Top 20 + Classifier-Specific Greedy 20
# ============================================================================
# This cell performs classifier-specific feature selection on 60 early fusion features:
# - **60 Features**: 18 model-independent + 6 models × 7 model-dependent
# - **Global Top 20**: Selected across all models using weighted_score ranking
# - **Classifier-Specific Greedy 20**: Selected via greedy forward selection for each classifier
# - **Final Feature Set**: 40 features per classifier (global 20 + classifier-specific greedy 20)
# - **Training**: Train+Dev combined for final training
# - **Evaluation**: Test set evaluation
# - **Weighted Ensemble**: Combines probabilities from all classifiers using Macro F1 weights

import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.base import clone
from tqdm import tqdm

from src.features.extraction import (
    get_model_independent_feature_names,
    get_model_dependent_feature_names
)
from src.models.classifiers import get_classifier_dict
from src.evaluation.metrics import compute_all_metrics

# ========================================================================
# STEP 1: Create 60 Feature Names (with model prefixes for model-dependent features)
# ========================================================================
print("\n" + "="*80)
print("STEP 1: CREATE 60 FEATURE NAMES")
print("="*80)

# Get base feature names
indep_feature_names = get_model_independent_feature_names()  # 18 features
dep_feature_names = get_model_dependent_feature_names()  # 7 features

# Create 60 feature names: [18 model-independent | 6 models × 7 model-dependent]
fused_feature_names_60 = indep_feature_names.copy()  # 18 model-independent

# Add model-dependent features with model prefix
MODELS_60 = ['bert', 'bert_political', 'bert_ambiguity', 'roberta', 'deberta', 'xlnet']
for model in MODELS_60:
    for dep_name in dep_feature_names:
        fused_feature_names_60.append(f"{model}_{dep_name}")

print(f"✓ Created 60 feature names:")
print(f"  - Model-independent: {len(indep_feature_names)} features")
print(f"  - Model-dependent: {len(MODELS_60)} models × {len(dep_feature_names)} features = {len(MODELS_60) * len(dep_feature_names)} features")
print(f"  - Total: {len(fused_feature_names_60)} features")
print(f"\n  First 5 features: {fused_feature_names_60[:5]}")
print(f"  Last 5 features: {fused_feature_names_60[-5:]}")

# Create feature name to index mapping
feature_name_to_idx_60 = {name: idx for idx, name in enumerate(fused_feature_names_60)}

# ========================================================================
# STEP 2: Load 60 Features for Train/Dev/Test (from early fusion)
# ========================================================================
print("\n" + "="*80)
print("STEP 2: LOAD 60 FEATURES (EARLY FUSION)")
print("="*80)

# Tasks for this evaluation
TASKS_60 = ['clarity', 'hierarchical_evasion_to_clarity']  # 2 tasks

# Store 60 features for each task
features_60 = {}  # {task: {'train': X_train_60, 'dev': X_dev_60, 'test': X_test_60}}

for task in TASKS_60:
    print(f"\n{'-'*60}")
    print(f"Task: {task.upper()}")
    print(f"{'-'*60}")

    # Determine split task (hierarchical uses evasion splits)
    split_task = 'evasion' if task == 'hierarchical_evasion_to_clarity' else task

    # Load model-independent features
    X_train_indep = storage.load_model_independent_features('train', task=split_task)
    X_dev_indep = storage.load_model_independent_features('dev', task=split_task)
    X_test_indep = storage.load_model_independent_features('test', task=split_task)

    # Load model-dependent features from each model
    model_dep_train_list = []
    model_dep_dev_list = []
    model_dep_test_list = []

    for model in MODELS_60:
        # Load full features and extract model-dependent portion (first 7)
        X_train_full = storage.load_features(model, split_task, 'train')
        X_dev_full = storage.load_features(model, split_task, 'dev')
        X_test_full = storage.load_features(model, split_task, 'test')

        # Extract model-dependent portion (first 7 features)
        model_dep_train_list.append(X_train_full[:, :7])
        model_dep_dev_list.append(X_dev_full[:, :7])
        model_dep_test_list.append(X_test_full[:, :7])

    # Concatenate model-dependent features
    X_train_dep_concat = np.hstack(model_dep_train_list)  # (N, 42)
    X_dev_dep_concat = np.hstack(model_dep_dev_list)  # (N, 42)
    X_test_dep_concat = np.hstack(model_dep_test_list)  # (N, 42)

    # Final concatenation: [18 Model-Independent | 42 Model-Dependent] = 60 features
    X_train_60 = np.hstack([X_train_indep, X_train_dep_concat])  # (N, 60)
    X_dev_60 = np.hstack([X_dev_indep, X_dev_dep_concat])  # (N, 60)
    X_test_60 = np.hstack([X_test_indep, X_test_dep_concat])  # (N, 60)

    features_60[task] = {
        'train': X_train_60,
        'dev': X_dev_60,
        'test': X_test_60
    }

    print(f"  Train: {X_train_60.shape[0]} samples, {X_train_60.shape[1]} features")
    print(f"  Dev: {X_dev_60.shape[0]} samples, {X_dev_60.shape[1]} features")
    print(f"  Test: {X_test_60.shape[0]} samples, {X_test_60.shape[1]} features")

# ========================================================================
# STEP 3: Select Global Top 20 Features (from weighted_score ranking)
# ========================================================================
print("\n" + "="*80)
print("STEP 3: SELECT GLOBAL TOP 20 FEATURES")
print("="*80)

# Load global top features from Cell 5 (weighted_score ranking)
# We need to map 25-feature names to 60-feature names
global_top_20_dict = {}  # {task: [feature_names]}

for task in TASKS_60:
    # Map task name for Cell 5 compatibility
    task_key_35 = 'hierarchical_evasion_to_clarity' if task == 'hierarchical_evasion_to_clarity' else task

    # Get top 20 features from df_stats (weighted_score ranking)
    if 'df_stats' not in globals():
        raise NameError("df_stats not found. Please run Cell 5 (Feature Ranking) first.")

    df_task_stats = df_stats[df_stats['task'] == task_key_35].copy()
    if len(df_task_stats) == 0:
        print(f"  ⚠ No stats found for task '{task_key_35}', using top 20 from all features")
        # Fallback: use first 20 features
        global_top_20_dict[task] = fused_feature_names_60[:20]
        continue

    # Sort by weighted_score (descending)
    df_task_stats = df_task_stats.sort_values('weighted_score', ascending=False)

    # Get top 20 feature names (from 25-feature system)
    top_20_features_25 = df_task_stats.head(20)['feature'].tolist()

    # Map to 60-feature names
    # Strategy: For model-independent features, use as-is
    # For model-dependent features, include all 6 model variants
    global_top_20_mapped = []

    for feat_name_25 in top_20_features_25:
        if feat_name_25 in indep_feature_names:
            # Model-independent feature: use as-is
            if feat_name_25 not in global_top_20_mapped:
                global_top_20_mapped.append(feat_name_25)
        elif feat_name_25 in dep_feature_names:
            # Model-dependent feature: add all 6 model variants
            for model in MODELS_60:
                mapped_name = f"{model}_{feat_name_25}"
                if mapped_name in feature_name_to_idx_60 and mapped_name not in global_top_20_mapped:
                    global_top_20_mapped.append(mapped_name)
                    if len(global_top_20_mapped) >= 20:
                        break
        else:
            # Try to find in 60-feature names directly
            if feat_name_25 in feature_name_to_idx_60 and feat_name_25 not in global_top_20_mapped:
                global_top_20_mapped.append(feat_name_25)

        if len(global_top_20_mapped) >= 20:
            break

    # If we don't have 20 yet, fill with remaining top features
    if len(global_top_20_mapped) < 20:
        remaining = [f for f in fused_feature_names_60 if f not in global_top_20_mapped]
        global_top_20_mapped.extend(remaining[:20 - len(global_top_20_mapped)])

    global_top_20_dict[task] = global_top_20_mapped[:20]

    print(f"\n  {task.upper()} - Global Top 20 Features:")
    for i, feat in enumerate(global_top_20_dict[task][:10], 1):
        print(f"    {i:2d}. {feat}")
    print(f"    ... (showing first 10 of 20)")

# ========================================================================
# STEP 4: Greedy Forward Selection for Each Classifier
# ========================================================================
print("\n" + "="*80)
print("STEP 4: GREEDY FORWARD SELECTION (PER CLASSIFIER)")
print("="*80)

# Greedy selection parameters
GLOBAL_TOP_K = 20  # Start with global top 20
MAX_FEATURES = 40  # Maximum features per classifier (global 20 + greedy 20)

# Load classifiers
classifiers_60 = get_classifier_dict(random_state=42)

# Greedy forward selection function
def greedy_forward_selection_60(X_train, X_dev, y_train, y_dev, feature_names_60,
                                 seed_features, clf, max_features=None):
    """
    Greedy forward selection: iteratively add best feature
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_dev_encoded = label_encoder.transform(y_dev)

    selected_indices = [feature_names_60.index(f) for f in seed_features if f in feature_names_60]
    available_indices = [i for i in range(len(feature_names_60)) if i not in selected_indices]

    trajectory = []

    # Evaluate initial set
    X_train_selected = X_train[:, selected_indices]
    X_dev_selected = X_dev[:, selected_indices]

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
    pipe.fit(X_train_selected, y_train_encoded)
    pred = pipe.predict(X_dev_selected)
    initial_f1 = f1_score(y_dev_encoded, pred, average='macro')
    trajectory.append((len(selected_indices), initial_f1))

    # Iteratively add best feature
    while len(available_indices) > 0:
        if max_features is not None and len(selected_indices) >= max_features:
            break

        best_f1 = initial_f1
        best_idx = None

        for idx in tqdm(available_indices, desc=f"Greedy selection ({len(selected_indices)}/{max_features or len(feature_names_60)})", leave=False):
            test_indices = selected_indices + [idx]
            X_train_test = X_train[:, test_indices]
            X_dev_test = X_dev[:, test_indices]

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
            pipe.fit(X_train_test, y_train_encoded)
            pred = pipe.predict(X_dev_test)
            f1 = f1_score(y_dev_encoded, pred, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_idx = idx

        if best_idx is None or best_f1 <= initial_f1:
            break

        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
        initial_f1 = best_f1
        trajectory.append((len(selected_indices), best_f1))

    selected_features = [feature_names_60[i] for i in selected_indices]
    return selected_features, trajectory

# Create output directories
results_dir_type2 = storage.data_path / 'results/FinalResultsType2/classifier_specific'
results_dir_type2.mkdir(parents=True, exist_ok=True)

predictions_dir = results_dir_type2 / 'predictions'
probabilities_dir = results_dir_type2 / 'probabilities'
metrics_dir = results_dir_type2 / 'metrics'

predictions_dir.mkdir(parents=True, exist_ok=True)
probabilities_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Store results
classifier_specific_results = {}  # {task: {classifier: {features, metrics, predictions, probabilities}}}

for task in TASKS_60:
    print(f"\n{'-'*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'-'*80}")

    # Load features and labels
    X_train_60 = features_60[task]['train']
    X_dev_60 = features_60[task]['dev']
    X_test_60 = features_60[task]['test']

    # Load labels
    split_task = 'evasion' if task == 'hierarchical_evasion_to_clarity' else task
    label_key = 'clarity_label' if task == 'hierarchical_evasion_to_clarity' else ('clarity_label' if task == 'clarity' else 'evasion_label')
    label_list = CLARITY_LABELS if 'clarity' in task else EVASION_LABELS

    train_ds = storage.load_split('train', task=split_task)
    dev_ds = storage.load_split('dev', task=split_task)
    test_ds = storage.load_split('test', task=split_task)

    y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
    y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
    y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])

    # Get global top 20 for this task
    global_top_20 = global_top_20_dict[task]

    # Initialize results for this task
    classifier_specific_results[task] = {}

    # For each classifier
    for clf_name, clf in classifiers_60.items():
        print(f"\n  Classifier: {clf_name}")

        # Run greedy selection (starts with global top 20, adds up to 20 more)
        print(f"    Running greedy selection (starting with global top 20, max 40 features)...")
        selected_features, trajectory = greedy_forward_selection_60(
            X_train_60, X_dev_60, y_train, y_dev,
            fused_feature_names_60, global_top_20, clf,
            max_features=MAX_FEATURES
        )

        final_f1 = trajectory[-1][1] if trajectory else 0.0
        print(f"    ✓ Selected {len(selected_features)} features, Dev F1={final_f1:.4f}")

        # Get feature indices
        selected_indices = [feature_name_to_idx_60[name] for name in selected_features if name in feature_name_to_idx_60]

        # Extract selected features
        X_train_selected = X_train_60[:, selected_indices]
        X_dev_selected = X_dev_60[:, selected_indices]
        X_test_selected = X_test_60[:, selected_indices]

        # Combine train+dev for final training
        X_train_combined = np.vstack([X_train_selected, X_dev_selected])
        y_train_combined = np.concatenate([y_train, y_dev])

        # Train final model
        print(f"    Training on Train+Dev combined ({X_train_combined.shape[0]} samples)...")
        le = LabelEncoder()
        y_train_combined_encoded = le.fit_transform(y_train_combined)
        y_test_encoded = le.transform(y_test)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
        pipe.fit(X_train_combined, y_train_combined_encoded)

        # Predict on test
        y_test_pred_encoded = pipe.predict(X_test_selected)
        y_test_pred = le.inverse_transform(y_test_pred_encoded)

        # Get probabilities if available
        y_test_proba = None
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            try:
                # CRITICAL: Use StandardScaler to transform test features before predict_proba
                X_test_scaled = pipe.named_steps['scaler'].transform(X_test_selected)
                y_test_proba = pipe.named_steps['clf'].predict_proba(X_test_scaled)
            except Exception as e:
                print(f"      ⚠ Warning: Could not get probabilities for {clf_name}: {e}")

        # Compute metrics
        metrics = compute_all_metrics(
            y_test_encoded, y_test_pred_encoded, label_list,
            task_name=f"TEST_{task}_{clf_name}"
        )

        print(f"    Test Macro F1: {metrics.get('macro_f1', 0.0):.4f}")

        # Store results
        classifier_specific_results[task][clf_name] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'metrics': metrics,
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'trajectory': trajectory
        }

        # Save individual classifier results
        # Save predictions
        np.save(predictions_dir / f'{clf_name}_{task}_predictions.npy', y_test_pred)
        
        # Save probabilities if available
        if y_test_proba is not None:
            np.save(probabilities_dir / f'{clf_name}_{task}_probabilities.npy', y_test_proba)
            print(f"    ✓ Saved probabilities: {probabilities_dir / f'{clf_name}_{task}_probabilities.npy'}")
        else:
            print(f"    ⚠ No probabilities available for {clf_name}")

print("\n" + "="*80)
print("GREEDY FORWARD SELECTION COMPLETE")
print("="*80)

# ========================================================================
# STEP 5: Weighted Average Ensemble
# ========================================================================
print("\n" + "="*80)
print("STEP 5: WEIGHTED AVERAGE ENSEMBLE")
print("="*80)

ensemble_results = {}  # {task: {'predictions': ..., 'probabilities': ..., 'weights': ...}}

for task in TASKS_60:
    print(f"\n{'-'*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'-'*80}")

    if task not in classifier_specific_results:
        print(f"  ⚠ Skipping {task}: No results available")
        continue

    label_list = CLARITY_LABELS if 'clarity' in task else EVASION_LABELS

    # Collect probabilities and weights
    probabilities_list = []
    weights_list = []
    classifier_names_list = []

    for clf_name, result in classifier_specific_results[task].items():
        y_proba = result.get('probabilities')
        if y_proba is None:
            print(f"  ⚠ Skipping {clf_name}: No probabilities available")
            continue

        metrics = result.get('metrics', {})
        macro_f1 = metrics.get('macro_f1', 0.0)
        weight = max(macro_f1, 0.0001)  # Use macro_f1 as weight

        probabilities_list.append(y_proba)
        weights_list.append(weight)
        classifier_names_list.append(clf_name)

    if len(probabilities_list) == 0:
        print(f"  ⚠ No probabilities available for {task}. Skipping ensemble.")
        continue

    # Normalize weights
    total_weight = sum(weights_list)
    normalized_weights = [w / total_weight for w in weights_list] if total_weight > 0 else [1.0 / len(weights_list)] * len(weights_list)

    print(f"\n  Normalized weights (based on Macro F1):")
    for clf_name, norm_weight, macro_f1 in zip(classifier_names_list, normalized_weights, weights_list):
        print(f"    {clf_name}: {norm_weight:.4f} (Macro F1: {macro_f1:.4f})")

    # Weighted average ensemble
    ensemble_proba = np.zeros_like(probabilities_list[0])
    for proba, weight in zip(probabilities_list, normalized_weights):
        ensemble_proba += weight * proba

    # Get hard labels from weighted average probabilities
    ensemble_pred_indices = np.argmax(ensemble_proba, axis=1)
    ensemble_pred = np.array([label_list[i] for i in ensemble_pred_indices])

    print(f"    ✓ Ensemble predictions shape: {ensemble_pred.shape}")

    # Store results
    ensemble_results[task] = {
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'weights': {name: weight for name, weight in zip(classifier_names_list, normalized_weights)},
        'classifiers_used': classifier_names_list
    }

    # Save ensemble results
    # Save hard labels (argmax from weighted average probabilities)
    np.save(predictions_dir / f'ensemble_hard_labels_from_weighted_proba_{task}.npy', ensemble_pred)
    print(f"    ✓ Saved ensemble predictions: {predictions_dir / f'ensemble_hard_labels_from_weighted_proba_{task}.npy'}")

    # Save soft labels (weighted average probabilities)
    np.save(probabilities_dir / f'ensemble_weighted_average_probabilities_{task}.npy', ensemble_proba)
    print(f"    ✓ Saved ensemble probabilities: {probabilities_dir / f'ensemble_weighted_average_probabilities_{task}.npy'}")

    # Evaluate ensemble on test set
    split_task = 'evasion' if task == 'hierarchical_evasion_to_clarity' else task
    test_ds = storage.load_split('test', task=split_task)
    label_key = 'clarity_label' if 'clarity' in task else 'evasion_label'
    y_test_true = np.array([test_ds[i][label_key] for i in range(len(test_ds))])

    le = LabelEncoder()
    y_test_true_encoded = le.fit_transform(y_test_true)
    ensemble_pred_encoded = le.transform(ensemble_pred)

    ensemble_metrics = compute_all_metrics(
        y_test_true_encoded, ensemble_pred_encoded, label_list,
        task_name=f"ENSEMBLE_{task}"
    )

    print(f"    Ensemble Test Macro F1: {ensemble_metrics.get('macro_f1', 0.0):.4f}")

    # Save ensemble classifier weights (used for weighted average)
    weights_metadata = {
        'task': task,
        'method': 'weighted_average',
        'weight_metric': 'macro_f1',
        'n_classifiers': len(classifier_names_list),
        'classifiers': classifier_names_list,
        'weights': {name: float(weight) for name, weight in zip(classifier_names_list, normalized_weights)},
        'n_samples': len(ensemble_pred),
        'label_list': label_list
    }

    with open(metrics_dir / f'ensemble_classifier_weights_{task}.json', 'w') as f:
        json.dump(weights_metadata, f, indent=2)
    print(f"    ✓ Saved ensemble weights: {metrics_dir / f'ensemble_classifier_weights_{task}.json'}")

    # Save ensemble evaluation metrics
    with open(metrics_dir / f'ensemble_evaluation_metrics_{task}.json', 'w') as f:
        json.dump({
            'task': task,
            'metrics': {k: float(v) for k, v in ensemble_metrics.items()},
            'n_samples': len(ensemble_pred)
        }, f, indent=2)
    print(f"    ✓ Saved ensemble metrics: {metrics_dir / f'ensemble_evaluation_metrics_{task}.json'}")

print("\n" + "="*80)
print("CLASSIFIER-SPECIFIC FEATURE SELECTION COMPLETE")
print("="*80)
print(f"\nResults saved to: {results_dir_type2}")
print(f"  - Selected features per classifier (40 features each)")
print(f"  - Test predictions per classifier: predictions/{clf_name}_{{task}}_predictions.npy")
print(f"  - Test probabilities per classifier: probabilities/{clf_name}_{{task}}_probabilities.npy")
print(f"  - Ensemble results:")
print(f"    - predictions/ensemble_hard_labels_from_weighted_proba_{{task}}.npy")
print(f"    - probabilities/ensemble_weighted_average_probabilities_{{task}}.npy")
print(f"    - metrics/ensemble_classifier_weights_{{task}}.json")
print(f"    - metrics/ensemble_evaluation_metrics_{{task}}.json")