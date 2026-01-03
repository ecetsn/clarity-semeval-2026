"""
Results table creation and printing (like siparismaili01)
"""
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np


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


def create_model_wise_summary(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str]
) -> pd.DataFrame:
    """
    Create model-wise summary table (like siparismaili01)
    
    Args:
        all_results: Dict[model_name][task_name][classifier_name] -> results
        models: List of model names
        classifiers: List of classifier names
        tasks: List of task names
    
    Returns:
        DataFrame with model-wise summary
    """
    rows = []
    
    for model in models:
        if model not in all_results:
            continue
        for classifier in classifiers:
            for task in tasks:
                if task not in all_results[model]:
                    continue
                if classifier not in all_results[model][task]:
                    continue
                
                result = all_results[model][task][classifier]
                if 'metrics' in result:
                    metrics = result['metrics']
                    rows.append({
                        'Model': model,
                        'Classifier': classifier,
                        'Task': task,
                        'Accuracy': metrics.get('accuracy', 0.0),
                        'Macro F1': metrics.get('macro_f1', 0.0),
                        'Weighted F1': metrics.get('weighted_f1', 0.0),
                    })
    
    df = pd.DataFrame(rows)
    return df


def create_final_summary_pivot(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str],
    metric: str = 'macro_f1'
) -> pd.DataFrame:
    """
    Create final summary pivot table: ALL MODELS × CLASSIFIERS × TASKS
    
    Args:
        all_results: Dict[model_name][task_name][classifier_name] -> results
        models: List of model names
        classifiers: List of classifier names
        tasks: List of task names
        metric: Metric to use for pivot (default: 'macro_f1')
    
    Returns:
        Pivot DataFrame with MultiIndex (Model, Classifier) × Tasks
    """
    summary_rows = []
    
    for model in models:
        if model not in all_results:
            continue
        for task in tasks:
            if task not in all_results[model]:
                continue
            for classifier in classifiers:
                if classifier not in all_results[model][task]:
                    continue
                
                result = all_results[model][task][classifier]
                if 'metrics' in result:
                    metrics = result['metrics']
                    value = metrics.get(metric, 0.0)
                    summary_rows.append({
                        'model': model,
                        'classifier': classifier,
                        'task': task,
                        metric: value
                    })
    
    if not summary_rows:
        return pd.DataFrame()
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Create pivot table: (Model, Classifier) × Tasks
    df_pivot = df_summary.pivot_table(
        index=['model', 'classifier'],
        columns='task',
        values=metric
    )
    
    # Sort by tasks (if available)
    if len(tasks) > 0:
        sort_cols = [col for col in tasks if col in df_pivot.columns]
        if sort_cols:
            df_pivot = df_pivot.sort_values(by=sort_cols, ascending=False)
    
    return df_pivot


def create_model_wise_summary_pivot(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str],
    metric: str = 'macro_f1'
) -> pd.DataFrame:
    """
    Create model-wise summary pivot: Classifier × Tasks (grouped by Model)
    
    Args:
        all_results: Dict[model_name][task_name][classifier_name] -> results
        models: List of model names
        classifiers: List of classifier names
        tasks: List of task names
        metric: Metric to use for pivot (default: 'macro_f1')
    
    Returns:
        Pivot DataFrame: Model × (Classifier, Task)
    """
    summary_rows = []
    
    for model in models:
        if model not in all_results:
            continue
        for classifier in classifiers:
            for task in tasks:
                if task not in all_results[model]:
                    continue
                if classifier not in all_results[model][task]:
                    continue
                
                result = all_results[model][task][classifier]
                if 'metrics' in result:
                    metrics = result['metrics']
                    value = metrics.get(metric, 0.0)
                    summary_rows.append({
                        'model': model,
                        'classifier': classifier,
                        'task': task,
                        metric: value
                    })
    
    if not summary_rows:
        return pd.DataFrame()
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Create pivot: Model × (Classifier, Task)
    # Group by model, then pivot classifier × task
    df_pivot = df_summary.pivot_table(
        index='model',
        columns=['classifier', 'task'],
        values=metric
    )
    
    return df_pivot


def create_classifier_wise_summary_pivot(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str],
    metric: str = 'macro_f1'
) -> pd.DataFrame:
    """
    Create classifier-wise summary pivot: Model × Tasks (grouped by Classifier)
    
    Args:
        all_results: Dict[model_name][task_name][classifier_name] -> results
        models: List of model names
        classifiers: List of classifier names
        tasks: List of task names
        metric: Metric to use for pivot (default: 'macro_f1')
    
    Returns:
        Pivot DataFrame: Classifier × (Model, Task)
    """
    summary_rows = []
    
    for classifier in classifiers:
        for model in models:
            if model not in all_results:
                continue
            for task in tasks:
                if task not in all_results[model]:
                    continue
                if classifier not in all_results[model][task]:
                    continue
                
                result = all_results[model][task][classifier]
                if 'metrics' in result:
                    metrics = result['metrics']
                    value = metrics.get(metric, 0.0)
                    summary_rows.append({
                        'classifier': classifier,
                        'model': model,
                        'task': task,
                        metric: value
                    })
    
    if not summary_rows:
        return pd.DataFrame()
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Create pivot: Classifier × (Model, Task)
    df_pivot = df_summary.pivot_table(
        index='classifier',
        columns=['model', 'task'],
        values=metric
    )
    
    return df_pivot


def style_table(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    precision: int = 4,
    task_colors: Optional[Dict[str, str]] = None
) -> 'pd.Styler':
    """
    Style table with color gradients (like siparismaili01)
    Applies comparison-based coloring: Task 1 (clarity) vs Task 3 (hierarchical_evasion_to_clarity)
    - If Task 1 > Task 3: Task 1 = green, Task 3 = red
    - If Task 1 < Task 3: Task 1 = red, Task 3 = green
    
    Args:
        df: DataFrame to style
        metric_cols: List of metric columns to apply gradient (if None, auto-detect numeric)
        precision: Decimal precision for formatting
        task_colors: Dict mapping task names to color names (for task-specific styling)
    
    Returns:
        Styled DataFrame
    """
    if metric_cols is None:
        # Auto-detect numeric columns
        metric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    styled = df.style
    
    # Format numeric columns
    for col in metric_cols:
        if col in df.columns:
            styled = styled.format({col: f"{{:.{precision}f}}"})
    
    # Check if this is a pivot table with task columns (Task 1 vs Task 3 comparison)
    task1_col = 'clarity'
    task3_col = 'hierarchical_evasion_to_clarity'
    
    # Handle MultiIndex columns (if tasks are in column level)
    if isinstance(df.columns, pd.MultiIndex):
        # Extract task names from MultiIndex columns
        task_cols_flat = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
        has_task1 = task1_col in task_cols_flat
        has_task3 = task3_col in task_cols_flat
    else:
        # Simple column names
        has_task1 = task1_col in df.columns
        has_task3 = task3_col in df.columns
    
    if has_task1 and has_task3:
        # Apply comparison-based coloring: Task 1 vs Task 3
        # Create a function to apply colors based on comparison
        def color_comparison(row):
            """Apply green/red based on Task 1 vs Task 3 comparison"""
            colors = [''] * len(row)
            
            # Find column indices for Task 1 and Task 3
            task1_idx = None
            task3_idx = None
            
            for i, col in enumerate(df.columns):
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    col_name = col[-1] if isinstance(col, tuple) else str(col)
                else:
                    col_name = col
                
                if col_name == task1_col:
                    task1_idx = i
                elif col_name == task3_col:
                    task3_idx = i
            
            if task1_idx is not None and task3_idx is not None:
                task1_val = row.iloc[task1_idx] if pd.notna(row.iloc[task1_idx]) else None
                task3_val = row.iloc[task3_idx] if pd.notna(row.iloc[task3_idx]) else None
                
                if task1_val is not None and task3_val is not None:
                    if task1_val > task3_val:
                        # Task 1 wins: green for Task 1, red for Task 3
                        colors[task1_idx] = 'background-color: #90EE90'  # Light green
                        colors[task3_idx] = 'background-color: #FFB6C1'  # Light red
                    elif task1_val < task3_val:
                        # Task 3 wins: red for Task 1, green for Task 3
                        colors[task1_idx] = 'background-color: #FFB6C1'  # Light red
                        colors[task3_idx] = 'background-color: #90EE90'  # Light green
                    # If equal, keep default (no special coloring)
            
            return colors
        
        # Apply comparison-based coloring
        styled = styled.apply(color_comparison, axis=1)
        
        # Also apply gradient to other numeric columns (if any)
        if isinstance(df.columns, pd.MultiIndex):
            other_numeric_cols = [col for col in df.columns if col[-1] not in [task1_col, task3_col]]
        else:
            other_numeric_cols = [col for col in metric_cols if col not in [task1_col, task3_col]]
        
        if other_numeric_cols:
            styled = styled.background_gradient(
                subset=other_numeric_cols,
                cmap='RdYlGn',
                vmin=0.0,
                vmax=1.0
            )
    else:
        # No task comparison - apply standard gradient to all numeric columns
        if metric_cols:
            styled = styled.background_gradient(
                subset=metric_cols,
                cmap='RdYlGn',  # Red-Yellow-Green gradient
                vmin=0.0,
                vmax=1.0
            )
    
    # Add borders and improve readability
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
    ])
    
    return styled

