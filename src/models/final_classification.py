"""
Final classification utilities for competition submission
Supports two types of classification:
1. Type 1: Individual model features (6 classifiers per model)
2. Type 2: Early fusion features (4 classifiers on fused features)
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

from .classifiers import get_classifier_dict
from .inference import predict_batch_from_dataset
from ..storage.manager import StorageManager


def final_classification_type1(
    storage: StorageManager,
    models: List[str],
    tasks: List[str],
    classifiers: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Type 1 Final Classification: Individual model features
    
    For each model×task combination:
    1. Load train+dev features and labels
    2. Train all 6 classifiers on train+dev
    3. Load test features (or extract if not cached)
    4. Predict on test set
    5. Return predictions and probabilities
    
    This matches the approach in 03_train_evaluate.ipynb but:
    - Trains on train+dev (not just train)
    - Evaluates on test (not dev)
    
    Args:
        storage: StorageManager instance
        models: List of model names (e.g., ['bert', 'roberta', ...])
        tasks: List of task names (e.g., ['clarity', 'evasion'])
        classifiers: Dict of classifiers (or None for default 6 classifiers)
        random_state: Random seed
    
    Returns:
        Dictionary: {
            model_name: {
                task_name: {
                    classifier_name: {
                        'predictions': np.ndarray,
                        'probabilities': np.ndarray,
                        'metrics': dict (if test labels available)
                    }
                }
            }
        }
    """
    if classifiers is None:
        classifiers = get_classifier_dict(random_state=random_state)
    
    results = {}
    
    for model in models:
        results[model] = {}
        
        for task in tasks:
            print(f"\n{'='*80}")
            print(f"TYPE 1 FINAL CLASSIFICATION: {model.upper()} × {task.upper()}")
            print(f"{'='*80}")
            
            results[model][task] = {}
            
            # Load train+dev features and labels
            try:
                X_train = storage.load_features(model, task, 'train')
                X_dev = storage.load_features(model, task, 'dev')
                
                train_ds = storage.load_split('train', task=task)
                dev_ds = storage.load_split('dev', task=task)
                
                # Determine label key
                label_key = 'clarity_label' if task == 'clarity' else 'evasion_label'
                label_list = train_ds[0].get('clarity_labels', []) if task == 'clarity' else train_ds[0].get('evasion_labels', [])
                
                y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
                y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
                
                # Combine train+dev for final training
                X_train_full = np.vstack([X_train, X_dev])
                y_train_full = np.concatenate([y_train, y_dev])
                
                print(f"  Training on: {X_train_full.shape[0]} samples (train+dev combined)")
                
            except FileNotFoundError as e:
                print(f"   Error loading features: {e}")
                continue
            
            # Load test dataset
            try:
                test_ds = storage.load_split('test', task=task)
                print(f"  Test set: {len(test_ds)} samples")
            except FileNotFoundError as e:
                print(f"   Error loading test set: {e}")
                continue
            
            # Train each classifier and predict on test
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train_full)
            
            for clf_name, clf in classifiers.items():
                print(f"\n  Training {clf_name}...")
                
                # Train on train+dev
                clf.fit(X_train_full, y_train_encoded)
                
                # Predict on test (features should be cached or will be extracted)
                # Note: For inference, we need model/tokenizer/tfidf_vectorizer
                # This function assumes test features are already extracted
                try:
                    X_test = storage.load_features(model, task, 'test')
                    print(f"    Loaded test features: {X_test.shape}")
                except FileNotFoundError:
                    print(f"     Test features not found. Need to extract first using inference function.")
                    continue
                
                # Predict
                y_test_pred_encoded = clf.predict(X_test)
                y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
                
                try:
                    y_test_proba = clf.predict_proba(X_test)
                except AttributeError:
                    y_test_proba = None
                
                results[model][task][clf_name] = {
                    'predictions': y_test_pred,
                    'probabilities': y_test_proba,
                }
                
                print(f"    ✓ Predictions: {len(y_test_pred)} samples")
    
    return results


def final_classification_type2(
    storage: StorageManager,
    models: List[str],
    tasks: List[str],
    classifiers: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Type 2 Final Classification: Early fusion features
    
    For each task:
    1. Load fused features (all models concatenated) for train+dev
    2. Train 4 classifiers on train+dev (LogisticRegression, LinearSVC, RandomForest, MLP)
    3. Load or extract fused test features
    4. Predict on test set
    5. Return predictions and probabilities
    
    This matches the approach in 04_early_fusion.ipynb but:
    - Trains on train+dev (not just train)
    - Evaluates on test (not dev)
    
    Args:
        storage: StorageManager instance
        models: List of model names (used to construct fused feature path)
        tasks: List of task names (e.g., ['clarity', 'evasion'])
        classifiers: Dict of classifiers (default: 4 classifiers for early fusion)
        random_state: Random seed
    
    Returns:
        Dictionary: {
            task_name: {
                classifier_name: {
                    'predictions': np.ndarray,
                    'probabilities': np.ndarray,
                    'metrics': dict (if test labels available)
                }
            }
        }
    """
    # Default classifiers for early fusion (4 classifiers)
    if classifiers is None:
        default_clfs = get_classifier_dict(random_state=random_state)
        # Use only: LogisticRegression, LinearSVC, RandomForest, MLP
        classifiers = {
            k: v for k, v in default_clfs.items() 
            if k in ['LogisticRegression', 'LinearSVC', 'RandomForest', 'MLP']
        }
    
    results = {}
    
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"TYPE 2 FINAL CLASSIFICATION: EARLY FUSION × {task.upper()}")
        print(f"{'='*80}")
        
        results[task] = {}
        
        # Construct model string for fused features
        model_str = '_'.join(sorted(models))
        
        # Load fused train+dev features
        try:
            # Try to load fused features
            fused_train_path = storage.data_path / f'features/fused/X_train_fused_{model_str}_{task}.npy'
            fused_dev_path = storage.data_path / f'features/fused/X_dev_fused_{model_str}_{task}.npy'
            
            X_train_fused = np.load(fused_train_path)
            X_dev_fused = np.load(fused_dev_path)
            
            train_ds = storage.load_split('train', task=task)
            dev_ds = storage.load_split('dev', task=task)
            
            label_key = 'clarity_label' if task == 'clarity' else 'evasion_label'
            y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
            y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
            
            # Combine train+dev
            X_train_full = np.vstack([X_train_fused, X_dev_fused])
            y_train_full = np.concatenate([y_train, y_dev])
            
            print(f"  Training on: {X_train_full.shape[0]} samples (train+dev fused)")
            
        except FileNotFoundError as e:
            print(f"   Error loading fused features: {e}")
            print(f"    Expected paths:")
            print(f"      {fused_train_path}")
            print(f"      {fused_dev_path}")
            continue
        
        # Load test dataset
        try:
            test_ds = storage.load_split('test', task=task)
            print(f"  Test set: {len(test_ds)} samples")
        except FileNotFoundError as e:
            print(f"   Error loading test set: {e}")
            continue
        
        # Load or extract fused test features
        try:
            fused_test_path = storage.data_path / f'features/fused/X_test_fused_{model_str}_{task}.npy'
            X_test_fused = np.load(fused_test_path)
            print(f"  Loaded fused test features: {X_test_fused.shape}")
        except FileNotFoundError:
            print(f"   Fused test features not found. Need to extract first.")
            continue
        
        # Train each classifier and predict on test
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_full)
        
        for clf_name, clf in classifiers.items():
            print(f"\n  Training {clf_name}...")
            
            # Train on train+dev
            clf.fit(X_train_full, y_train_encoded)
            
            # Predict on test
            y_test_pred_encoded = clf.predict(X_test_fused)
            y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
            
            try:
                y_test_proba = clf.predict_proba(X_test_fused)
            except AttributeError:
                y_test_proba = None
            
            results[task][clf_name] = {
                'predictions': y_test_pred,
                'probabilities': y_test_proba,
            }
            
            print(f"    ✓ Predictions: {len(y_test_pred)} samples")
    
    return results

