"""
Plotting utilities: confusion matrix, precision-recall curves, ROC curves
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix
)
from typing import List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # For Colab compatibility


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[Any],
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot confusion matrix (heatmap)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_list: List of label names
        task_name: Task name for title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_list,
        yticklabels=label_list,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix{f": {task_name}" if task_name else ""}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_list: List[Any],
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot precision-recall curves for each class (one-vs-rest)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (N, num_classes)
        label_list: List of label names
        task_name: Task name for title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_classes = len(label_list)
    
    # Convert labels to indices
    # Handle label mapping: dataset uses 'Ambivalent' but code expects 'Ambiguous'
    label_mapping = {'Ambivalent': 'Ambiguous'}  # Map dataset label to expected label
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    
    # Map y_true labels if needed
    y_true_mapped = np.array([
        label_mapping.get(str(label), label) if str(label) in label_mapping else label
        for label in y_true
    ])
    
    y_true_idx = np.array([label_to_idx[label] for label in y_true_mapped])
    
    plt.figure(figsize=figsize)
    
    for i, label in enumerate(label_list):
        # One-vs-rest: positive class = i, negative = all others
        y_true_binary = (y_true_idx == i).astype(int)
        y_proba_binary = y_proba[:, i]
        
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary,
            y_proba_binary
        )
        
        # Calculate AUC (Average Precision)
        ap = auc(recall, precision)
        
        plt.plot(
            recall,
            precision,
            label=f'{label} (AP={ap:.3f})',
            linewidth=2
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves{f": {task_name}" if task_name else ""}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved precision-recall curves: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_list: List[Any],
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot ROC curves for each class (one-vs-rest)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (N, num_classes)
        label_list: List of label names
        task_name: Task name for title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_classes = len(label_list)
    
    # Convert labels to indices
    # Handle label mapping: dataset uses 'Ambivalent' but code expects 'Ambiguous'
    label_mapping = {'Ambivalent': 'Ambiguous'}  # Map dataset label to expected label
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    
    # Map y_true labels if needed
    y_true_mapped = np.array([
        label_mapping.get(str(label), label) if str(label) in label_mapping else label
        for label in y_true
    ])
    
    y_true_idx = np.array([label_to_idx[label] for label in y_true_mapped])
    
    plt.figure(figsize=figsize)
    
    for i, label in enumerate(label_list):
        # One-vs-rest: positive class = i, negative = all others
        y_true_binary = (y_true_idx == i).astype(int)
        y_proba_binary = y_proba[:, i]
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr,
            tpr,
            label=f'{label} (AUC={roc_auc:.3f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves{f": {task_name}" if task_name else ""}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curves: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_metrics_comparison(
    results_dict: dict,
    metric_name: str = "macro_f1",
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot bar chart comparing metrics across classifiers
    
    Args:
        results_dict: Dict mapping classifier_name -> {'metrics': {...}}
        metric_name: Metric to plot (e.g., 'macro_f1', 'accuracy')
        task_name: Task name for title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    classifiers = []
    values = []
    
    for name, result in results_dict.items():
        if 'metrics' in result and metric_name in result['metrics']:
            classifiers.append(name)
            values.append(result['metrics'][metric_name])
    
    if not classifiers:
        print(f"⚠️  No {metric_name} found in results")
        return
    
    plt.figure(figsize=figsize)
    bars = plt.bar(classifiers, values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison{f": {task_name}" if task_name else ""}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved metrics comparison: {save_path}")
    else:
        plt.show()
    plt.close()

