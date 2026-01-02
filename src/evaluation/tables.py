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
                'Cohen Kappa': metrics.get('cohen_kappa', 0.0),
                'Matthews CorrCoef': metrics.get('matthews_corrcoef', 0.0),
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


def print_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = "",
    sort_by: str = "Macro F1"
) -> pd.DataFrame:
    """
    Print formatted results table
    
    Args:
        results_dict: Results dictionary
        task_name: Task name
        sort_by: Column to sort by
    
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
    print(f"{'='*80}\n")
    
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
    precision: int = 4
) -> pd.io.formats.style.Styler:
    """
    Style table with color gradients (like siparismaili01)
    
    Args:
        df: DataFrame to style
        metric_cols: List of metric columns to apply gradient (if None, auto-detect numeric)
        precision: Decimal precision for formatting
    
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
    
    # Apply background gradient to numeric columns
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

