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
    Style table with custom coloring rules:
    - Default: Gray background (except header row and first column which stay blue)
    - If hierarchical_evasion_to_clarity > clarity: Green background
    - If hierarchical_evasion_to_clarity <= clarity: Gray background
    - Row with highest Macro F1: Bold font for entire row
    
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
    
    # Find clarity and hierarchical_evasion_to_clarity columns
    clarity_col = None
    hierarchical_col = None
    
    for col in df.columns:
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            col_name = col[-1] if isinstance(col, tuple) else str(col)
        else:
            col_name = col
        
        if col_name == 'clarity':
            clarity_col = col
        elif col_name == 'hierarchical_evasion_to_clarity':
            hierarchical_col = col
    
    # Find row with highest Macro F1 (check clarity column first, then hierarchical)
    max_f1_row_idx = None
    max_f1_value = -1
    
    if clarity_col is not None and clarity_col in df.columns:
        # Use clarity column for finding max
        max_f1_value = df[clarity_col].max()
        max_f1_row_idx = df[clarity_col].idxmax()
    elif hierarchical_col is not None and hierarchical_col in df.columns:
        # Fallback to hierarchical column
        max_f1_value = df[hierarchical_col].max()
        max_f1_row_idx = df[hierarchical_col].idxmax()
    
    # Apply styling function
    def apply_cell_styles(row):
        """Apply background colors and bold font"""
        styles = [''] * len(row)
        row_idx = row.name
        
        # Check if this is the row with highest Macro F1
        is_max_row = (row_idx == max_f1_row_idx)
        
        for i, val in enumerate(row):
            # Get column object
            if isinstance(df.columns, pd.MultiIndex):
                col_obj = df.columns[i]
                col_name = col_obj[-1] if isinstance(col_obj, tuple) else str(col_obj)
            else:
                col_obj = df.columns[i]
                col_name = str(col_obj)
            
            # Get clarity and hierarchical values for this row
            clarity_val = None
            hierarchical_val = None
            
            if clarity_col is not None and clarity_col in df.columns:
                clarity_val = row[clarity_col]
            if hierarchical_col is not None and hierarchical_col in df.columns:
                hierarchical_val = row[hierarchical_col]
            
            # Determine background color
            if clarity_val is not None and hierarchical_val is not None and pd.notna(clarity_val) and pd.notna(hierarchical_val):
                if hierarchical_val > clarity_val:
                    # Green background (hierarchical beats clarity)
                    styles[i] = 'background-color: #90EE90'
                else:
                    # Gray background (hierarchical <= clarity)
                    styles[i] = 'background-color: #D3D3D3'
            else:
                # Default gray for all other cells
                styles[i] = 'background-color: #D3D3D3'
            
            # Apply bold font to entire row if it's the max row
            if is_max_row:
                if styles[i]:
                    styles[i] += '; font-weight: bold'
                else:
                    styles[i] = 'font-weight: bold'
        
        return styles
    
    # Apply cell styles
    styled = styled.apply(apply_cell_styles, axis=1)
    
    # Keep header row and first column blue (default pandas styling)
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
        # First column (index) styling - keep blue/white
        {'selector': 'th:first-child', 'props': [('background-color', '#4472C4'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td:first-child', 'props': [('background-color', '#E7F0F8'), ('font-weight', 'bold')]},
    ])
    
    return styled


def style_table_paper(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    precision: int = 4,
    clarity_col_name: str = 'clarity',
    hierarchical_col_name: str = 'hierarchical_evasion_to_clarity',
    apply_column_mapping: bool = False,
    best_direction: str = 'auto',  # 'auto', 'column', 'row'
    table_name: Optional[str] = None  # NEW: Table name for auto-detection (e.g., 'model_wise_bert', 'classifier_wise_LightGBM')
) -> 'pd.Styler':
    """
    Paper-ready table styling: Minimal, professional, academic-friendly
    
    Styling rules:
    1. Best values → Bold + Dark Green
       - Model-wise tables (Classifier × Tasks): Column-wise best (each task's best classifier)
       - Classifier-wise tables (Model × Tasks): Row-wise best (each model's best task)
    2. Hierarchical > Clarity → Italic (no color)
    3. Others → Normal black
    
    Args:
        df: DataFrame to style
        metric_cols: List of metric columns (if None, auto-detect numeric)
        precision: Decimal precision for formatting
        clarity_col_name: Name of clarity column
        hierarchical_col_name: Name of hierarchical column
        apply_column_mapping: If True, apply paper-ready column name mapping
        best_direction: 'auto' (detect from table_name or index), 'column' (column-wise best), 'row' (row-wise best)
        table_name: Optional table name for auto-detection (e.g., 'model_wise_bert' → column-wise, 'classifier_wise_LightGBM' → row-wise)
    
    Returns:
        Styled DataFrame (paper-ready)
    """
    # Clean up column names: Remove MultiIndex names if present (e.g., "task" header)
    df_clean = df.copy()
    
    # Remove completely empty rows and columns FIRST (before other processing)
    df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
    
    # Remove rows with invalid index values (NaN, empty strings, etc.)
    if len(df_clean) > 0:
        valid_indices = []
        for idx in df_clean.index:
            if pd.notna(idx) and str(idx).strip() != '':
                valid_indices.append(idx)
        if len(valid_indices) < len(df_clean.index):
            df_clean = df_clean.loc[valid_indices]
    
    # Remove columns with invalid names
    if len(df_clean.columns) > 0:
        valid_cols = []
        for col in df_clean.columns:
            if pd.notna(col) and str(col).strip() != '':
                valid_cols.append(col)
        if len(valid_cols) < len(df_clean.columns):
            df_clean = df_clean.loc[:, valid_cols]
    
    if isinstance(df_clean.columns, pd.MultiIndex):
        # Flatten MultiIndex: keep only the actual column names, remove the level name
        df_clean.columns = df_clean.columns.get_level_values(-1)
    # Remove column name if it's just "task" or similar (pivot table artifact)
    if df_clean.columns.name in ['task', 'Task', 'TASK']:
        df_clean.columns.name = None
    
    # Column name mapping (paper-ready formatting) - SADECE İSTENİRSE UYGULA
    if apply_column_mapping:
        COLUMN_NAME_MAPPING = {
            'clarity': 'Clarity',
            'evasion': 'Evasion',
            'hierarchical_evasion_to_clarity': 'Hierarchical Mapping to Clarity'
        }
        # Apply mapping: use mapping if exists, otherwise capitalize first letter
        df_clean.columns = df_clean.columns.map(
            lambda x: COLUMN_NAME_MAPPING.get(str(x).lower(), str(x).title()) if isinstance(x, str) else x
        )
    
    if metric_cols is None:
        # Auto-detect numeric columns (AFTER mapping)
        metric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Convert metric_cols to list of strings to avoid Index object issues after mapping
    # This ensures column matching works correctly
    metric_cols_str = [str(col) if not isinstance(col, str) else col for col in metric_cols]
    
    styled = df_clean.style
    
    # Format numeric columns - match by string name
    for col_str in metric_cols_str:
        # Find actual column object by string matching
        for actual_col in df_clean.columns:
            if str(actual_col) == col_str:
                try:
                    styled = styled.format({actual_col: f"{{:.{precision}f}}"})
                except (KeyError, ValueError):
                    pass
                break
    
    # Find clarity and hierarchical columns (after mapping, search in mapped names)
    clarity_col = None
    hierarchical_col = None
    
    for col in df_clean.columns:
        col_name = str(col)
        
        # Search in mapped column names (e.g., "Clarity", "Hierarchical Mapping to Clarity")
        if 'clarity' in col_name.lower() and 'hierarchical' not in col_name.lower():
            clarity_col = col
        elif 'hierarchical' in col_name.lower() or 'mapping' in col_name.lower():
            hierarchical_col = col
    
    # AUTO-DETECT table type from table_name (most reliable) or index name (fallback)
    if best_direction == 'auto':
        # Priority 1: Check table_name (most reliable)
        if table_name:
            table_name_lower = str(table_name).lower()
            if 'model_wise' in table_name_lower:
                best_direction = 'column'  # Model-wise: Classifier × Tasks → column-wise best
            elif 'classifier_wise' in table_name_lower:
                best_direction = 'row'     # Classifier-wise: Model × Tasks → row-wise best
            else:
                # Fallback to index name detection
                index_name = str(df_clean.index.name).lower() if df_clean.index.name else ''
                if 'classifier' in index_name:
                    best_direction = 'column'
                elif 'model' in index_name:
                    best_direction = 'row'
                else:
                    best_direction = 'column'  # Default
        else:
            # Priority 2: Check index name
            index_name = str(df_clean.index.name).lower() if df_clean.index.name else ''
            if 'classifier' in index_name:
                best_direction = 'column'  # Model-wise: Classifier × Tasks → column-wise best
            elif 'model' in index_name:
                best_direction = 'row'     # Classifier-wise: Model × Tasks → row-wise best
            else:
                # Fallback: Check index values (first few) to guess
                sample_indices = [str(idx).lower() for idx in df_clean.index[:3] if pd.notna(idx)]
                classifier_keywords = ['classifier', 'logistic', 'random', 'xgboost', 'lightgbm', 'mlp', 'linear', 'svc']
                if any(any(kw in idx for kw in classifier_keywords) for idx in sample_indices):
                    best_direction = 'column'  # Likely classifier names
                else:
                    best_direction = 'column'  # Default to column-wise (safer)
    
    # Find best values based on direction
    column_best = {}
    column_best_indices = {}
    row_best = {}
    row_best_col_names = {}  # Store column NAME instead of object (safer)
    
    if best_direction == 'column':
        # Column-wise best: Find best value per column (task)
        for col_str in metric_cols_str:
            for actual_col in df_clean.columns:
                if str(actual_col) == col_str:
                    try:
                        col_values = df_clean[actual_col].dropna()
                        if len(col_values) > 0:
                            max_val = col_values.max()
                            unique_vals = col_values.unique()
                            if len(unique_vals) > 1:
                                max_indices = col_values[col_values == max_val].index.tolist()
                                if len(max_indices) == 1:
                                    column_best[actual_col] = max_val
                                    column_best_indices[actual_col] = max_indices[0]
                            elif len(col_values) == 1:
                                column_best[actual_col] = max_val
                                column_best_indices[actual_col] = col_values.index[0]
                    except (KeyError, ValueError):
                        pass
                    break
    
    elif best_direction == 'row':
        # Row-wise best: Find best value per row (model)
        for idx in df_clean.index:
            row_values = []
            row_col_names = []  # Store column names
            for col_str in metric_cols_str:
                for actual_col in df_clean.columns:
                    if str(actual_col) == col_str:
                        try:
                            val = df_clean.loc[idx, actual_col]
                            if pd.notna(val):
                                row_values.append(val)
                                row_col_names.append(str(actual_col))  # Store name
                        except (KeyError, ValueError):
                            pass
                        break
            if row_values:
                max_val = max(row_values)
                max_idx = row_values.index(max_val)
                row_best[idx] = max_val
                row_best_col_names[idx] = row_col_names[max_idx]  # Store column name
    
    # Apply styling function
    def apply_cell_styles(row):
        """Apply paper-ready styling: bold+green for best, italic for hierarchical>clarity"""
        styles = [''] * len(row)
        row_idx = row.name
        
        for i, val in enumerate(row):
            # Get column object
            col_obj = df_clean.columns[i]
            col_name = str(col_obj)
            
            # Check if this is a metric column - use string matching
            is_metric_col = col_name in metric_cols_str
            
            if is_metric_col and pd.notna(val):
                style_parts = []
                
                # Apply best styling based on direction
                if best_direction == 'column':
                    # Column-wise: Mark best value per column (task's best classifier)
                    if col_obj in column_best and col_obj in column_best_indices:
                        if row_idx == column_best_indices[col_obj]:
                            if abs(val - column_best[col_obj]) < 1e-6:
                                style_parts.append('font-weight: bold')
                                style_parts.append('color: #006400')  # Dark green
                
                elif best_direction == 'row':
                    # Row-wise: Mark best value per row (model's best task)
                    if row_idx in row_best and row_idx in row_best_col_names:
                        # Compare by column name (safer than object identity)
                        if col_name == row_best_col_names[row_idx]:
                            if abs(val - row_best[row_idx]) < 1e-6:
                                style_parts.append('font-weight: bold')
                                style_parts.append('color: #006400')  # Dark green
                
                # 3. Check if hierarchical > clarity (italic)
                # FIX: Ensure clarity_col is found correctly (even without column mapping)
                if hierarchical_col is not None and clarity_col is not None:
                    if col_obj == hierarchical_col:
                        hierarchical_val = val
                        # Try to get clarity value from the same row
                        try:
                            clarity_val = df_clean.loc[row_idx, clarity_col] if clarity_col in df_clean.columns else None
                        except (KeyError, IndexError):
                            clarity_val = None
                        
                        if clarity_val is not None and pd.notna(clarity_val) and pd.notna(hierarchical_val):
                            # FIX: Use proper comparison with tolerance
                            if hierarchical_val > clarity_val + 1e-6:  # Clear improvement
                                style_parts.append('font-style: italic')
                
                if style_parts:
                    styles[i] = '; '.join(style_parts)
        
        return styles
    
    # Apply cell styles
    styled = styled.apply(apply_cell_styles, axis=1)
    
    # Column alignment: numbers right, text left
    # Use CSS selectors (like style_table) instead of set_properties to avoid KeyError issues
    # Build alignment styles for numeric columns (right align)
    alignment_styles = []
    for i, col in enumerate(df_clean.columns):
        # FIX: Use string matching to avoid Index object issues
        col_str = str(col)
        if col_str in metric_cols_str:
            # Numeric columns: right align using nth-child selector
            # +2 because: +1 for index column (first-child), +1 for 1-based indexing
            col_selector_idx = i + 2
            alignment_styles.append({
                'selector': f'td:nth-child({col_selector_idx})',
                'props': [('text-align', 'right')]
            })
            alignment_styles.append({
                'selector': f'th:nth-child({col_selector_idx})',
                'props': [('text-align', 'right')]
            })
        # Text columns default to left (handled by table_styles below)
    
    # Note: Index column styling is handled by table_styles below (th:first-child, td:first-child)
    
    # Minimal table styling (clean, professional, NO background colors)
    # Combine with alignment styles
    # FIX: Index column styling - ensure proper vertical alignment and prevent text sliding
    base_styles = [
        {'selector': 'th', 'props': [('color', '#212529'), ('font-weight', 'bold'), ('border', '1px solid #dee2e6'), ('text-align', 'center'), ('vertical-align', 'middle'), ('padding', '8px')]},
        {'selector': 'td', 'props': [('border', '1px solid #dee2e6'), ('padding', '8px'), ('vertical-align', 'middle')]},
        {'selector': 'th:first-child', 'props': [('font-weight', 'bold'), ('text-align', 'left'), ('vertical-align', 'middle'), ('padding', '8px')]},
        {'selector': 'td:first-child', 'props': [('font-weight', 'bold'), ('text-align', 'left'), ('vertical-align', 'middle'), ('padding', '8px')]},
    ]
    
    # Apply all styles together
    styled = styled.set_table_styles(base_styles + alignment_styles)
    
    return styled


def export_table_latex(
    df: pd.DataFrame,
    filepath: Path,
    caption: str = "",
    label: str = "",
    precision: int = 4
) -> None:
    """
    Export DataFrame to LaTeX table format (paper-ready)
    
    Args:
        df: DataFrame to export
        filepath: Path to save .tex file
        caption: Table caption
        label: LaTeX label for referencing
        precision: Decimal precision
    """
    # Format numeric columns
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "")
    
    # Convert to LaTeX
    latex_str = df_formatted.to_latex(
        index=True,
        caption=caption,
        label=label,
        float_format=lambda x: f"{x:.{precision}f}" if pd.notna(x) else "",
        escape=False
    )
    
    # Save to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX table: {filepath}")


def export_table_markdown(
    df: pd.DataFrame,
    filepath: Path,
    precision: int = 4
) -> None:
    """
    Export DataFrame to Markdown table format (clean, readable)
    
    Args:
        df: DataFrame to export
        filepath: Path to save .md file
        precision: Decimal precision
    """
    # Format numeric columns
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "")
    
    # Convert to Markdown
    markdown_str = df_formatted.to_markdown(index=True)
    
    # Save to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_str)
    
    print(f"Saved Markdown table: {filepath}")

