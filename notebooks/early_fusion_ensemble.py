# ============================================================================
# EARLY FUSION ENSEMBLE: Weighted Average from Probabilities
# ============================================================================
# This script:
# 1. Loads early fusion probabilities from Drive
# 2. Computes weighted average ensemble
# 3. Calculates metrics
# 4. Creates comparison tables

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# Mount Drive
try:
    drive.mount('/content/drive', force_remount=False)
except:
    pass

# Add repository to path
sys.path.insert(0, '/content/semeval-context-tree-modular')

from src.storage.manager import StorageManager
from src.evaluation.metrics import compute_all_metrics

# Initialize StorageManager
BASE_PATH = Path('/content/semeval-context-tree-modular')
DATA_PATH = Path('/content/drive/MyDrive/semeval_data')

storage = StorageManager(
    base_path=str(BASE_PATH),
    data_path=str(DATA_PATH),
    github_path=str(BASE_PATH)
)

# Label lists
CLARITY_LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
EVASION_LABELS = ['Claims ignorance', 'Clarification', 'Declining to answer',
                  'Deflection', 'Dodging', 'Explicit',
                  'General', 'Implicit', 'Partial/half-answer']

TASKS = ['clarity', 'evasion']
CLASSIFIERS = ['LogisticRegression', 'LinearSVC', 'RandomForest', 'MLP', 'XGBoost', 'LightGBM']

# ============================================================================
# STEP 1: Load Early Fusion Probabilities and Compute Individual Metrics
# ============================================================================
print("="*80)
print("STEP 1: LOAD EARLY FUSION PROBABILITIES")
print("="*80)

predictions_dir = storage.data_path / 'results/FinalResultsType3/predictions'
early_fusion_results = {}  # {task: {classifier: {predictions, probabilities, metrics}}}

for task in TASKS:
    print(f"\nTask: {task.upper()}")
    early_fusion_results[task] = {}
    
    label_list = CLARITY_LABELS if task == 'clarity' else EVASION_LABELS
    label_key = 'clarity_label' if task == 'clarity' else 'evasion_label'
    
    # Load test labels
    test_ds = storage.load_split('test', task=task)
    y_test_true = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
    
    le = LabelEncoder()
    y_test_true_encoded = le.fit_transform(y_test_true)
    
    for clf_name in CLASSIFIERS:
        # Load probabilities
        proba_path = predictions_dir / f'proba_test_{clf_name}_{task}.npy'
        
        if not proba_path.exists():
            print(f"  ⚠ {clf_name}: No probabilities found")
            continue
        
        try:
            y_test_proba = np.load(proba_path)
            
            # Load predictions (hard labels)
            pred_path = predictions_dir / f'pred_test_{clf_name}_{task}.npy'
            if pred_path.exists():
                y_test_pred = np.load(pred_path)
            else:
                # Generate from probabilities
                y_test_pred_indices = np.argmax(y_test_proba, axis=1)
                y_test_pred = np.array([label_list[i] for i in y_test_pred_indices])
            
            # Encode predictions
            y_test_pred_encoded = le.transform(y_test_pred)
            
            # Compute metrics
            metrics = compute_all_metrics(
                y_test_true_encoded, y_test_pred_encoded, label_list,
                task_name=f"EARLY_FUSION_{task}_{clf_name}"
            )
            
            early_fusion_results[task][clf_name] = {
                'predictions': y_test_pred,
                'probabilities': y_test_proba,
                'metrics': metrics
            }
            
            print(f"  ✓ {clf_name}: Macro F1={metrics.get('macro_f1', 0.0):.4f}")
            
        except Exception as e:
            print(f"  ❌ {clf_name}: Error loading - {e}")
            continue

# ============================================================================
# STEP 2: Compute Early Fusion Weighted Average Ensemble
# ============================================================================
print("\n" + "="*80)
print("STEP 2: EARLY FUSION WEIGHTED AVERAGE ENSEMBLE")
print("="*80)

early_fusion_ensemble_results = {}  # {task: {predictions, probabilities, metrics}}

for task in TASKS:
    if task not in early_fusion_results:
        continue
    
    label_list = CLARITY_LABELS if task == 'clarity' else EVASION_LABELS
    label_key = 'clarity_label' if task == 'clarity' else 'evasion_label'
    
    # Collect probabilities and weights
    probabilities_list = []
    weights_list = []
    classifier_names_list = []
    
    for clf_name, result in early_fusion_results[task].items():
        y_proba = result.get('probabilities')
        if y_proba is None:
            continue
        
        metrics = result.get('metrics', {})
        macro_f1 = metrics.get('macro_f1', 0.0)
        weight = max(macro_f1, 0.0001)
        
        probabilities_list.append(y_proba)
        weights_list.append(weight)
        classifier_names_list.append(clf_name)
    
    if len(probabilities_list) == 0:
        print(f"  ⚠ {task}: No probabilities available")
        continue
    
    # Normalize weights
    total_weight = sum(weights_list)
    normalized_weights = [w / total_weight for w in weights_list]
    
    print(f"\n  {task.upper()} - Weights:")
    for clf_name, norm_weight, macro_f1 in zip(classifier_names_list, normalized_weights, weights_list):
        print(f"    {clf_name}: {norm_weight:.4f} (Macro F1: {macro_f1:.4f})")
    
    # Weighted average ensemble
    ensemble_proba = np.zeros_like(probabilities_list[0])
    for proba, weight in zip(probabilities_list, normalized_weights):
        ensemble_proba += weight * proba
    
    # Hard labels from ensemble
    ensemble_pred_indices = np.argmax(ensemble_proba, axis=1)
    ensemble_pred = np.array([label_list[i] for i in ensemble_pred_indices])
    
    # Evaluate ensemble
    test_ds = storage.load_split('test', task=task)
    y_test_true = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
    
    le = LabelEncoder()
    y_test_true_encoded = le.fit_transform(y_test_true)
    ensemble_pred_encoded = le.transform(ensemble_pred)
    
    ensemble_metrics = compute_all_metrics(
        y_test_true_encoded, ensemble_pred_encoded, label_list,
        task_name=f"ENSEMBLE_EARLY_FUSION_{task}"
    )
    
    early_fusion_ensemble_results[task] = {
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'metrics': ensemble_metrics,
        'weights': {name: float(weight) for name, weight in zip(classifier_names_list, normalized_weights)}
    }
    
    print(f"  ✓ Ensemble Macro F1: {ensemble_metrics.get('macro_f1', 0.0):.4f}")

# ============================================================================
# STEP 3: Load Classifier-Specific Ensemble Results (if available)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: LOAD CLASSIFIER-SPECIFIC ENSEMBLE RESULTS")
print("="*80)

classifier_specific_ensemble_results = {}  # {task: {metrics}}

# Try to load from ablation results
ensemble_metrics_dir = storage.data_path / 'results/evaluation_a/metrics'
TASKS_CS = ['clarity', 'hierarchical_evasion_to_clarity']

for task in TASKS_CS:
    ensemble_metrics_path = ensemble_metrics_dir / f'ensemble_evaluation_metrics_{task}.json'
    
    if ensemble_metrics_path.exists():
        import json
        with open(ensemble_metrics_path, 'r') as f:
            data = json.load(f)
            metrics = data.get('metrics', {})
            classifier_specific_ensemble_results[task] = metrics
            print(f"  ✓ {task}: Macro F1={metrics.get('macro_f1', 0.0):.4f}")
    else:
        print(f"  ⚠ {task}: No ensemble results found")

# ============================================================================
# STEP 4: Create Comparison Tables
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CREATE COMPARISON TABLES")
print("="*80)

tables_dir = storage.data_path / 'results/FinalResultsType3/tables'
tables_dir.mkdir(parents=True, exist_ok=True)

# Table 1: Early Fusion Individual Classifiers
print("\nTable 1: Early Fusion Individual Classifiers")
table1_rows = []

for task in TASKS:
    if task not in early_fusion_results:
        continue
    
    for clf_name, result in early_fusion_results[task].items():
        metrics = result.get('metrics', {})
        table1_rows.append({
            'methodology': 'Early Fusion 60',
            'classifier': clf_name,
            'task': task,
            'macro_f1': metrics.get('macro_f1', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'macro_precision': metrics.get('macro_precision', 0.0),
            'macro_recall': metrics.get('macro_recall', 0.0),
            'weighted_f1': metrics.get('weighted_f1', 0.0)
        })

df_table1 = pd.DataFrame(table1_rows)
if len(df_table1) > 0:
    print(df_table1.to_string(index=False))
    table1_path = tables_dir / 'early_fusion_individual_classifiers.csv'
    df_table1.to_csv(table1_path, index=False)
    print(f"\n  ✓ Saved: {table1_path.name}")

# Table 2: Early Fusion Ensemble
print("\n\nTable 2: Early Fusion Ensemble (Weighted Average)")
table2_rows = []

for task in TASKS:
    if task not in early_fusion_ensemble_results:
        continue
    
    metrics = early_fusion_ensemble_results[task].get('metrics', {})
    table2_rows.append({
        'methodology': 'Early Fusion 60',
        'classifier': 'Ensemble (Weighted)',
        'task': task,
        'macro_f1': metrics.get('macro_f1', 0.0),
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_precision': metrics.get('macro_precision', 0.0),
        'macro_recall': metrics.get('macro_recall', 0.0),
        'weighted_f1': metrics.get('weighted_f1', 0.0)
    })

df_table2 = pd.DataFrame(table2_rows)
if len(df_table2) > 0:
    print(df_table2.to_string(index=False))
    table2_path = tables_dir / 'early_fusion_ensemble.csv'
    df_table2.to_csv(table2_path, index=False)
    print(f"\n  ✓ Saved: {table2_path.name}")

# Table 3: Classifier-Specific Ensemble
print("\n\nTable 3: Classifier-Specific Ensemble (Weighted Average)")
table3_rows = []

for task in TASKS_CS:
    if task not in classifier_specific_ensemble_results:
        continue
    
    metrics = classifier_specific_ensemble_results[task]
    table3_rows.append({
        'methodology': 'Classifier-Specific 20-40',
        'classifier': 'Ensemble (Weighted)',
        'task': task,
        'macro_f1': metrics.get('macro_f1', 0.0),
        'accuracy': metrics.get('accuracy', 0.0),
        'macro_precision': metrics.get('macro_precision', 0.0),
        'macro_recall': metrics.get('macro_recall', 0.0),
        'weighted_f1': metrics.get('weighted_f1', 0.0)
    })

df_table3 = pd.DataFrame(table3_rows)
if len(df_table3) > 0:
    print(df_table3.to_string(index=False))
    table3_path = tables_dir / 'classifier_specific_ensemble.csv'
    df_table3.to_csv(table3_path, index=False)
    print(f"\n  ✓ Saved: {table3_path.name}")

# Combined Table: All Results
print("\n\nCombined Table: All Methodologies")
df_combined = pd.concat([df_table1, df_table2, df_table3], ignore_index=True)
if len(df_combined) > 0:
    # Sort by methodology, then task, then classifier
    df_combined = df_combined.sort_values(['methodology', 'task', 'classifier'])
    print(df_combined.to_string(index=False))
    combined_path = tables_dir / 'all_methodologies_combined.csv'
    df_combined.to_csv(combined_path, index=False)
    print(f"\n  ✓ Saved: {combined_path.name}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nAll tables saved to: {tables_dir}")

