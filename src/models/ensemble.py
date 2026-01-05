"""
Ensemble and advanced fusion utilities for combining predictions and features
from multiple models and classifiers.

Supports:
1. Late Fusion (Ensemble): Combine predictions/probabilities from multiple models
2. Greedy-based Early Fusion: Use greedy-selected features per model for fusion
3. Top-K Model Selection: Select best models by macro F1 for ensemble
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from ..storage.manager import StorageManager
from ..features.fusion import fuse_attention_features


def select_top_models_by_f1(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    task: str,
    top_k: int = 10,
    metric: str = 'macro_f1'
) -> List[Tuple[str, str, float]]:
    """
    Select top-K model×classifier combinations by macro F1 score
    
    Args:
        results: Dictionary from final evaluation (e.g., from 05_final_evaluation.ipynb) or similar:
            {model_name: {task_name: {classifier_name: {'metrics': {...}}}}}
        task: Task name (e.g., 'clarity', 'evasion')
        top_k: Number of top models to select (default: 10)
        metric: Metric to use for ranking (default: 'macro_f1')
    
    Returns:
        List of (model_name, classifier_name, score) tuples, sorted by score descending
    """
    model_scores = []
    
    for model_name, model_results in results.items():
        if task not in model_results:
            continue
        
        for clf_name, clf_result in model_results[task].items():
            metrics = clf_result.get('metrics', {})
            score = metrics.get(metric, 0.0)
            
            if score > 0:  # Only include models with valid scores
                model_scores.append((model_name, clf_name, score))
    
    # Sort by score descending
    model_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Return top-K
    return model_scores[:top_k]


def ensemble_predictions_voting(
    predictions_list: List[np.ndarray],
    method: str = 'hard_voting'
) -> np.ndarray:
    """
    Ensemble predictions using voting methods
    
    Args:
        predictions_list: List of prediction arrays, all same length
        method: 'hard_voting' (majority vote) or 'soft_voting' (requires probabilities)
    
    Returns:
        Ensemble predictions
    """
    if len(predictions_list) == 0:
        raise ValueError("predictions_list cannot be empty")
    
    n_samples = len(predictions_list[0])
    for pred in predictions_list:
        if len(pred) != n_samples:
            raise ValueError("All predictions must have same length")
    
    if method == 'hard_voting':
        # Majority voting
        ensemble_pred = []
        for i in range(n_samples):
            votes = [pred[i] for pred in predictions_list]
            # Count votes and select most common
            vote_counts = Counter(votes)
            ensemble_pred.append(vote_counts.most_common(1)[0][0])
        
        return np.array(ensemble_pred)
    else:
        raise ValueError(f"Unknown method: {method}")


def ensemble_probabilities(
    probabilities_list: List[np.ndarray],
    label_lists: List[List[str]],
    method: str = 'mean',
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble probabilities from multiple models
    
    Args:
        probabilities_list: List of probability matrices (N, n_classes)
        label_lists: List of label lists for each model (must be same labels)
        method: 'mean' (average), 'weighted_mean' (weighted by F1), 'max' (max pooling)
        weights: Optional weights for weighted_mean (default: equal weights)
    
    Returns:
        (ensemble_probabilities, ensemble_predictions)
    """
    if len(probabilities_list) == 0:
        raise ValueError("probabilities_list cannot be empty")
    
    # Verify all have same shape
    n_samples, n_classes = probabilities_list[0].shape
    for proba in probabilities_list:
        if proba.shape != (n_samples, n_classes):
            raise ValueError("All probability matrices must have same shape")
    
    # Verify all have same labels
    first_labels = label_lists[0]
    for labels in label_lists[1:]:
        if labels != first_labels:
            raise ValueError("All models must use same label order")
    
    # Normalize weights
    if weights is None:
        weights = [1.0 / len(probabilities_list)] * len(probabilities_list)
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Ensemble probabilities
    if method == 'mean':
        ensemble_proba = np.mean(probabilities_list, axis=0)
    elif method == 'weighted_mean':
        ensemble_proba = np.zeros_like(probabilities_list[0])
        for proba, weight in zip(probabilities_list, weights):
            ensemble_proba += weight * proba
    elif method == 'max':
        ensemble_proba = np.max(probabilities_list, axis=0)
        # Normalize to sum to 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get predictions from probabilities
    ensemble_pred_indices = np.argmax(ensemble_proba, axis=1)
    ensemble_pred = np.array([first_labels[i] for i in ensemble_pred_indices])
    
    return ensemble_proba, ensemble_pred


def load_greedy_selected_features(
    storage: StorageManager,
    task: str
) -> Dict[str, List[str]]:
    """
    Load greedy-selected features from ablation study results
    
    Args:
        storage: StorageManager instance
        task: Task name (e.g., 'clarity', 'evasion')
    
    Returns:
        Dict mapping model_name -> list of selected feature names
        Example: {'bert': ['feature1', 'feature2'], 'roberta': ['feature3', 'feature4']}
    """
    import json
    
    ablation_dir = storage.data_path / 'results/ablation'
    json_path = ablation_dir / 'selected_features_all.json'
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Greedy selected features not found: {json_path}\n"
            f"Make sure you have run 03_5_ablation_study.ipynb (Greedy Forward Selection)"
        )
    
    with open(json_path, 'r') as f:
        selected_features_dict = json.load(f)
    
    # Convert to model_name -> features format
    greedy_features = {}
    for key, value in selected_features_dict.items():
        # Key format: "model_task"
        parts = key.split('_', 1)
        if len(parts) == 2:
            model_name, task_name = parts
            if task_name == task:
                greedy_features[model_name] = value.get('selected_features', [])
    
    return greedy_features


def create_greedy_fused_features(
    storage: StorageManager,
    models: List[str],
    task: str,
    greedy_selected_features: Optional[Dict[str, List[str]]] = None,
    split: str = 'train',
    auto_load_greedy: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Create early fusion features using greedy-selected features per model
    
    This is different from standard early fusion:
    - Standard: Concatenate ALL features from all models
    - Greedy-based: Concatenate only GREEDY-SELECTED features from each model
    
    Args:
        storage: StorageManager instance
        models: List of model names
        task: Task name (e.g., 'clarity', 'evasion')
        greedy_selected_features: Optional dict mapping model_name -> list of selected feature names
            If None and auto_load_greedy=True, will load from ablation results
        split: Data split ('train', 'dev', 'test')
        auto_load_greedy: If True and greedy_selected_features is None, load from saved results
    
    Returns:
        (fused_features, fused_feature_names)
    """
    from ..features.extraction import get_feature_names
    
    # Load greedy features if not provided
    if greedy_selected_features is None and auto_load_greedy:
        try:
            greedy_selected_features = load_greedy_selected_features(storage, task)
            print(f"  Loaded greedy-selected features for {len(greedy_selected_features)} models")
        except FileNotFoundError as e:
            print(f"   Could not load greedy features: {e}")
            print(f"  Using all features instead...")
            greedy_selected_features = {}
    
    # Get all feature names to find indices
    all_feature_names = get_feature_names()
    feature_name_to_idx = {name: idx for idx, name in enumerate(all_feature_names)}
    
    model_features_dict = {}
    model_feature_names_dict = {}
    
    for model in models:
        # Load full features for this model
        try:
            X_full = storage.load_features(model, task, split)
        except FileNotFoundError:
            print(f"   Features not found for {model} × {task} × {split}, skipping...")
            continue
        
        # Get selected features for this model
        if model not in greedy_selected_features:
            print(f"   No greedy features for {model}, using all features")
            selected_features = all_feature_names
        else:
            selected_features = greedy_selected_features[model]
        
        # Find indices of selected features
        selected_indices = []
        selected_names = []
        for feat_name in selected_features:
            if feat_name in feature_name_to_idx:
                idx = feature_name_to_idx[feat_name]
                selected_indices.append(idx)
                selected_names.append(feat_name)
            else:
                print(f"   Feature '{feat_name}' not found in feature list, skipping...")
        
        if len(selected_indices) == 0:
            print(f"   No valid features selected for {model}, skipping...")
            continue
        
        # Extract selected features
        X_selected = X_full[:, selected_indices]
        
        model_features_dict[model] = X_selected
        model_feature_names_dict[model] = [f"{model}_{name}" for name in selected_names]
        
        print(f"  {model}: {len(selected_names)} greedy-selected features")
    
    if len(model_features_dict) == 0:
        raise ValueError("No valid model features found")
    
    # Fuse using standard concatenation
    fused_features, fused_feature_names = fuse_attention_features(
        model_features_dict,
        model_feature_names_dict
    )
    
    return fused_features, fused_feature_names


def create_topk_fused_features(
    storage: StorageManager,
    top_models: List[Tuple[str, str, float]],  # (model_name, classifier_name, score)
    task: str,
    top_k_features: int = 10,
    split: str = 'train'
) -> Tuple[np.ndarray, List[str]]:
    """
    Create early fusion features using top-K models with top-K features each
    
    This selects:
    1. Top-K models by macro F1
    2. Top-K features per model (by weighted_score from ablation)
    
    Args:
        storage: StorageManager instance
        top_models: List of (model_name, classifier_name, score) tuples
        task: Task name
        top_k_features: Number of top features to use per model (default: 10)
        split: Data split ('train', 'dev', 'test')
    
    Returns:
        (fused_features, fused_feature_names)
    """
    from ..features.extraction import get_feature_names
    
    # Load top-K features from ablation results
    try:
        import json
        ablation_dir = storage.data_path / 'results/ablation'
        topk_features_path = ablation_dir / f'selected_features_for_early_fusion.json'
        
        with open(topk_features_path, 'r') as f:
            topk_data = json.load(f)
        
        # Get top-K features for this task
        task_topk = topk_data.get(task, {}).get('features', [])[:top_k_features]
        
    except (FileNotFoundError, KeyError) as e:
        print(f"   Could not load top-K features: {e}")
        print(f"  Using all features instead...")
        task_topk = get_feature_names()[:top_k_features]  # Fallback: first K features
    
    # Get unique models from top_models
    unique_models = list(set([model_name for model_name, _, _ in top_models]))
    
    # Create greedy_selected_features dict (same top-K features for all models)
    greedy_selected_features = {model: task_topk for model in unique_models}
    
    # Use create_greedy_fused_features
    return create_greedy_fused_features(
        storage, unique_models, task, greedy_selected_features, split
    )


def ensemble_from_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    task: str,
    top_k: int = 10,
    ensemble_method: str = 'weighted_mean',
    metric: str = 'macro_f1'
) -> Dict[str, Any]:
    """
    Create ensemble from final evaluation results
    
    Args:
        results: Results dictionary from final evaluation (e.g., from 05_final_evaluation.ipynb)
        task: Task name
        top_k: Number of top models to ensemble (default: 10)
        ensemble_method: 'hard_voting', 'mean', 'weighted_mean', 'max'
        metric: Metric to use for ranking and weighting
    
    Returns:
        Dictionary with ensemble predictions, probabilities, and metrics
    """
    # Select top-K models
    top_models = select_top_models_by_f1(results, task, top_k, metric)
    
    if len(top_models) == 0:
        raise ValueError(f"No models found for task: {task}")
    
    print(f"\nSelected top-{len(top_models)} models for ensemble:")
    for i, (model_name, clf_name, score) in enumerate(top_models, 1):
        print(f"  {i}. {model_name} × {clf_name}: {metric}={score:.4f}")
    
    # Collect predictions and probabilities
    predictions_list = []
    probabilities_list = []
    label_lists = []
    weights = []
    
    for model_name, clf_name, score in top_models:
        if model_name not in results or task not in results[model_name]:
            continue
        
        if clf_name not in results[model_name][task]:
            continue
        
        clf_result = results[model_name][task][clf_name]
        
        # Get predictions
        pred = clf_result.get('predictions')
        if pred is not None:
            predictions_list.append(pred)
        
        # Get probabilities
        proba = clf_result.get('probabilities')
        if proba is not None:
            probabilities_list.append(proba)
            # Get label list (assume same for all, get from first)
            if len(label_lists) == 0:
                # Try to get from metrics
                metrics = clf_result.get('metrics', {})
                # Label list might be in results structure
                # For now, infer from predictions
                unique_labels = sorted(list(set(pred)))
                label_lists.append(unique_labels)
            else:
                label_lists.append(label_lists[0])  # Same labels
        
        # Weight by metric score
        weights.append(score)
    
    if len(predictions_list) == 0:
        raise ValueError("No predictions found in results")
    
    # Ensemble
    if ensemble_method == 'hard_voting':
        ensemble_pred = ensemble_predictions_voting(predictions_list, 'hard_voting')
        ensemble_proba = None
    elif ensemble_method in ['mean', 'weighted_mean', 'max']:
        if len(probabilities_list) == 0:
            # Fallback to hard voting if no probabilities
            print("   No probabilities available, using hard voting instead")
            ensemble_pred = ensemble_predictions_voting(predictions_list, 'hard_voting')
            ensemble_proba = None
        else:
            # Normalize weights for weighted_mean
            if ensemble_method == 'weighted_mean':
                weights_normalized = [w / sum(weights) for w in weights]
            else:
                weights_normalized = None
            
            ensemble_proba, ensemble_pred = ensemble_probabilities(
                probabilities_list,
                label_lists,
                method=ensemble_method,
                weights=weights_normalized
            )
    else:
        raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
    
    return {
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'top_models': top_models,
        'ensemble_method': ensemble_method,
        'n_models': len(top_models)
    }

