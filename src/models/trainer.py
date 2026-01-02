"""
Training utilities: train and evaluate models
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .classifiers import train_classifiers, get_classifier_dict
from ..evaluation.metrics import compute_all_metrics, print_classification_report
from ..evaluation.tables import print_results_table, create_results_table
from ..evaluation.visualizer import visualize_all_evaluation, visualize_comparison


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    label_list: List[Any],
    task_name: str = "",
    classifiers: Dict[str, Any] = None,
    random_state: int = 42,
    print_report: bool = True,
    print_table: bool = True,
    create_plots: bool = True,
    save_plots_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train classifiers and evaluate on dev set
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_dev: Dev features
        y_dev: Dev labels
        label_list: List of label names/values
        task_name: Task name (for display)
        classifiers: Dict of classifiers (or None for default)
        random_state: Random seed
        print_report: Print classification report for each classifier
        print_table: Print results table
    
    Returns:
        Dictionary mapping classifier_name -> {
            'model': trained_model,
            'train_pred': predictions,
            'dev_pred': predictions,
            'train_proba': probabilities,
            'dev_proba': probabilities,
            'metrics': metrics dict
        }
    """
    # Train classifiers
    results = train_classifiers(
        X_train, y_train, X_dev, y_dev,
        classifiers=classifiers,
        random_state=random_state
    )
    
    # Compute metrics for each classifier
    for name, result in results.items():
        metrics = compute_all_metrics(
            y_dev,
            result['dev_pred'],
            label_list,
            task_name=f"{task_name} - {name}"
        )
        result['metrics'] = metrics
        
        # Print classification report
        if print_report:
            print_classification_report(
                y_dev,
                result['dev_pred'],
                label_list,
                task_name=f"{task_name} - {name}"
            )
    
    # Print results table
    if print_table:
        print_results_table(results, task_name=task_name, sort_by="Macro F1")
    
    # Create plots (confusion matrix, PR curves, ROC curves)
    if create_plots:
        print(f"\nCreating evaluation plots for {task_name}...")
        for name, result in results.items():
            if result['dev_proba'] is not None:
                visualize_all_evaluation(
                    y_dev,
                    result['dev_pred'],
                    result['dev_proba'],
                    label_list,
                    task_name=task_name,
                    classifier_name=name,
                    save_dir=save_plots_dir
                )
        
        # Create comparison plots
        visualize_comparison(
            results,
            label_list,
            task_name=task_name,
            save_dir=save_plots_dir
        )
    
    return results

