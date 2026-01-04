"""
Results table creation and printing (like siparismaili01)
"""
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path


def create_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = ""
) -> pd.DataFrame:
    """
    Create results table from multiple classifier results
    
    Args:
        results_dict: Dict mapping classifier_name -> {
            'dev_pred': predictions,
            'dev_proba': probabilities (optional),
            'metrics': metrics dict (optional)
        }
        task_name: Task name for display
    
    Returns:
        DataFrame with results
    """
    rows = []
    
    for classifier_name, result in results_dict.items():
        if 'metrics' in result:
            metrics = result['metrics']
            rows.append({
                'Classifier': classifier_name,
                'Task': task_name,
                'Accuracy': metrics.get('accuracy', 0.0),
                'Macro F1': metrics.get('macro_f1', 0.0),
                'Weighted F1': metrics.get('weighted_f1', 0.0),
                'Macro Precision': metrics.get('macro_precision', 0.0),
                'Macro Recall': metrics.get('macro_recall', 0.0),
            })
        else:
            # If metrics not computed, add placeholder
            rows.append({
                'Classifier': classifier_name,
                'Task': task_name,
                'Accuracy': None,
                'Macro F1': None,
                'Weighted F1': None,
                'Macro Precision': None,
                'Macro Recall': None,
            })
    
    df = pd.DataFrame(rows)
    return df


def print_per_class_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = "",
    label_list: List[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Print per-class metrics table for the best classifier (by Macro F1)
    
    Args:
        results_dict: Results dictionary
        task_name: Task name
        label_list: List of label names (optional, will be extracted from metrics if not provided)
    
    Returns:
        DataFrame with per-class metrics (or None if no metrics available)
    """
    # Find best classifier by Macro F1
    best_classifier = None
    best_f1 = -1
    
    for name, result in results_dict.items():
        if 'metrics' in result:
            f1 = result['metrics'].get('macro_f1', -1)
            if f1 > best_f1:
                best_f1 = f1
                best_classifier = name
    
    if best_classifier is None or 'metrics' not in results_dict[best_classifier]:
        return None
    
    metrics = results_dict[best_classifier]['metrics']
    
    # Get per-class metrics
    if 'per_class' not in metrics:
        return None
    
    per_class = metrics['per_class']
    
    # Extract label list if not provided
    if label_list is None:
        label_list = sorted(per_class.keys())
    
    # Create per-class table
    rows = []
    for label in label_list:
        if label in per_class:
            rows.append({
                'Class': label,
                'Precision': per_class[label]['precision'],
                'Recall': per_class[label]['recall'],
                'F1-Score': per_class[label]['f1'],
                'Support': per_class[label]['support']
            })
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    
    print(f"\n{'='*80}")
    if task_name:
        print(f"Per-Class Metrics: {task_name} - {best_classifier} (Best by Macro F1)")
    else:
        print(f"Per-Class Metrics: {best_classifier}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return df


def print_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = "",
    sort_by: str = "Macro F1",
    show_per_class: bool = True,
    label_list: List[Any] = None
) -> pd.DataFrame:
    """
    Print formatted results table
    
    Args:
        results_dict: Results dictionary
        task_name: Task name
        sort_by: Column to sort by
        show_per_class: Whether to show per-class metrics table
        label_list: List of label names (for per-class table)
    
    Returns:
        DataFrame (for further use)
    """
    df = create_results_table(results_dict, task_name)
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    print(f"\n{'='*80}")
    if task_name:
        print(f"Results Table: {task_name}")
    else:
        print("Results Table")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")
    
    # Print per-class metrics for best classifier
    if show_per_class:
        print_per_class_table(results_dict, task_name, label_list)
    
    return df
