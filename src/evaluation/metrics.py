"""
Evaluation metrics and reporting
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    hamming_loss,
    jaccard_score
)
from typing import Dict, List, Any, Union

Label = Union[int, str]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[Label],
    task_name: str = ""
) -> Dict[str, Any]:
    """
    Compute classification metrics including accuracy, macro/weighted metrics, and per-class metrics
    
    Returns:
        Dictionary with accuracy, macro/weighted metrics, per-class metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    
    # CRITICAL FIX: Check if y_true/y_pred are integers or strings
    # If integers, don't pass labels parameter (sklearn auto-detects)
    # If strings, pass label_list as labels
    is_numeric = np.issubdtype(y_true.dtype, np.integer) or np.issubdtype(y_pred.dtype, np.integer)
    
    if is_numeric:
        # y_true/y_pred are integers - don't pass labels, sklearn will auto-detect
        report = classification_report(
            y_true,
            y_pred,
            # labels parameter removed for integer labels
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        
        # Map integer indices to string labels
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        
        # Per-class metrics - map integer indices to string labels
        per_class = {}
        for idx, label in enumerate(label_list):
            # sklearn uses integer keys (0, 1, 2, ...) when y_true/y_pred are integers
            label_key = str(idx)
            if label_key in report:
                per_class[label] = {
                    "precision": float(report[label_key]["precision"]),
                    "recall": float(report[label_key]["recall"]),
                    "f1": float(report[label_key]["f1-score"]),
                    "support": int(report[label_key]["support"]),
                }
                metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])
    else:
        # y_true/y_pred are strings - pass label_list as labels
        report = classification_report(
            y_true,
            y_pred,
            labels=label_list,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        
        # Per-class metrics - use string keys directly
        per_class = {}
        for label in label_list:
            label_key = str(label)
            if label_key in report:
                per_class[label] = {
                    "precision": float(report[label_key]["precision"]),
                    "recall": float(report[label_key]["recall"]),
                    "f1": float(report[label_key]["f1-score"]),
                    "support": int(report[label_key]["support"]),
                }
                metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])
    
    metrics["per_class"] = per_class
    metrics["macro_avg_detail"] = {
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
        "support": int(report["macro avg"]["support"]),
    }
    metrics["weighted_avg_detail"] = {
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "support": int(report["weighted avg"]["support"]),
    }
    
    # Confusion matrix - filter labels to only include those present in y_true or y_pred
    # This prevents ValueError when some labels don't appear in test set (e.g., annotator-based evaluation)
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    all_unique = np.unique(np.concatenate([unique_true, unique_pred]))
    
    # Filter label_list to only include labels that exist in data
    if len(label_list) > 0 and isinstance(label_list[0], str):
        # String labels - filter by what exists in data
        existing_labels = [label for label in label_list if label in all_unique]
    else:
        # Encoded labels - use all_unique
        existing_labels = sorted(all_unique.tolist()) if len(all_unique) > 0 else label_list
    
    if len(existing_labels) == 0:
        # Fallback: use all_unique directly if label_list filtering resulted in empty
        existing_labels = sorted(all_unique.tolist()) if len(all_unique) > 0 else label_list
    
    # Use existing_labels for confusion matrix and jaccard_score
    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=existing_labels).tolist()
    except ValueError as e:
        # If still fails, use labels=None (auto-detect from data)
        print(f"  Warning: Could not create confusion matrix with specified labels: {e}")
        print(f"  Using auto-detected labels from data...")
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    # Specialized metrics
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    
    # Jaccard score (IoU) - average across classes
    try:
        metrics["jaccard_score"] = float(jaccard_score(
            y_true, y_pred, labels=existing_labels, average='macro', zero_division=0
        ))
    except ValueError as e:
        # If still fails, use labels=None (auto-detect from data)
        print(f"  Warning: Could not compute jaccard_score with specified labels: {e}")
        print(f"  Using auto-detected labels from data...")
        metrics["jaccard_score"] = float(jaccard_score(
            y_true, y_pred, average='macro', zero_division=0
        ))
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[Label],
    task_name: str = ""
) -> None:
    """
    Print classification report (same format as sklearn)
    """
    print(f"\n{'='*60}")
    if task_name:
        print(f"Classification Report: {task_name}")
    else:
        print("Classification Report")
    print(f"{'='*60}")
    
    # CRITICAL FIX: Same fix as compute_all_metrics
    is_numeric = np.issubdtype(y_true.dtype, np.integer) or np.issubdtype(y_pred.dtype, np.integer)
    
    if is_numeric:
        # y_true/y_pred are integers - don't pass labels, use target_names for display
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        # Map integer indices to string labels for display
        target_names = [str(label_list[i]) if i < len(label_list) else f"Class {i}" for i in unique_labels]
        print(classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0
        ))
    else:
        # y_true/y_pred are strings - pass label_list as labels
        print(classification_report(
            y_true,
            y_pred,
            labels=label_list,
            digits=4,
            zero_division=0
        ))
    
    print(f"{'='*60}\n")


def compute_all_metrics_any_annotator_match(
    y_pred: np.ndarray,
    y_annotator1: np.ndarray,
    y_annotator2: np.ndarray,
    y_annotator3: np.ndarray,
    label_list: List[Label],
    task_name: str = ""
) -> Dict[str, Any]:
    """
    Compute metrics where a prediction is considered correct if it matches
    ANY of the three annotator labels (annotator1, annotator2, or annotator3).
    
    This is used for test set evaluation where multiple annotators provided
    labels and any match is considered correct.
    
    Args:
        y_pred: Predicted labels (can be encoded integers or strings)
        y_annotator1: Annotator 1 labels
        y_annotator2: Annotator 2 labels
        y_annotator3: Annotator 3 labels
        label_list: List of all possible labels
        task_name: Task name for reporting
    
    Returns:
        Dictionary with metrics computed using "any match" logic
    """
    # Check if predictions match ANY of the three annotators
    matches_any = (
        (y_pred == y_annotator1) | 
        (y_pred == y_annotator2) | 
        (y_pred == y_annotator3)
    )
    
    # Calculate accuracy (fraction of predictions that match at least one annotator)
    accuracy = float(np.mean(matches_any))
    
    # For other metrics, we need to handle them differently
    # We'll use a "best match" approach: for each prediction, use the annotator
    # that matches (if any), or use annotator1 as default for computing class-based metrics
    
    # Create a combined "true" label array: use prediction if it matches any annotator,
    # otherwise use annotator1 (for computing class-based metrics)
    y_true_combined = np.where(matches_any, y_pred, y_annotator1)
    
    # Now compute standard metrics using the combined true labels
    # This gives us class-based metrics while respecting the "any match" accuracy
    metrics = {
        "accuracy": accuracy,  # Use the "any match" accuracy
        "macro_precision": float(
            precision_score(y_true_combined, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_combined, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true_combined, y_pred, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true_combined, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true_combined, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_combined, y_pred, average="weighted", zero_division=0)
        ),
    }
    
    # Check if labels are numeric or strings
    is_numeric = np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(y_true_combined.dtype, np.integer)
    
    if is_numeric:
        report = classification_report(
            y_true_combined,
            y_pred,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        
        unique_labels = sorted(np.unique(np.concatenate([y_true_combined, y_pred])).tolist())
        
        per_class = {}
        for idx, label in enumerate(label_list):
            label_key = str(idx)
            if label_key in report:
                per_class[label] = {
                    "precision": float(report[label_key]["precision"]),
                    "recall": float(report[label_key]["recall"]),
                    "f1": float(report[label_key]["f1-score"]),
                    "support": int(report[label_key]["support"]),
                }
                metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])
    else:
        report = classification_report(
            y_true_combined,
            y_pred,
            labels=label_list,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        
        per_class = {}
        for label in label_list:
            label_key = str(label)
            if label_key in report:
                per_class[label] = {
                    "precision": float(report[label_key]["precision"]),
                    "recall": float(report[label_key]["recall"]),
                    "f1": float(report[label_key]["f1-score"]),
                    "support": int(report[label_key]["support"]),
                }
                metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])
    
    metrics["per_class"] = per_class
    metrics["macro_avg_detail"] = {
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
        "support": int(report["macro avg"]["support"]),
    }
    metrics["weighted_avg_detail"] = {
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "support": int(report["weighted avg"]["support"]),
    }
    
    # Confusion matrix
    unique_true = np.unique(y_true_combined)
    unique_pred = np.unique(y_pred)
    all_unique = np.unique(np.concatenate([unique_true, unique_pred]))
    
    if len(label_list) > 0 and isinstance(label_list[0], str):
        existing_labels = [label for label in label_list if label in all_unique]
    else:
        existing_labels = sorted(all_unique.tolist()) if len(all_unique) > 0 else label_list
    
    if len(existing_labels) == 0:
        existing_labels = sorted(all_unique.tolist()) if len(all_unique) > 0 else label_list
    
    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true_combined, y_pred, labels=existing_labels).tolist()
    except ValueError as e:
        print(f"  Warning: Could not create confusion matrix with specified labels: {e}")
        print(f"  Using auto-detected labels from data...")
        metrics["confusion_matrix"] = confusion_matrix(y_true_combined, y_pred).tolist()
    
    metrics["hamming_loss"] = float(hamming_loss(y_true_combined, y_pred))
    
    try:
        metrics["jaccard_score"] = float(jaccard_score(
            y_true_combined, y_pred, labels=existing_labels, average='macro', zero_division=0
        ))
    except ValueError as e:
        print(f"  Warning: Could not compute jaccard_score with specified labels: {e}")
        print(f"  Using auto-detected labels from data...")
        metrics["jaccard_score"] = float(jaccard_score(
            y_true_combined, y_pred, average='macro', zero_division=0
        ))
    
    # Add information about match statistics
    matches_annotator1 = np.sum(y_pred == y_annotator1)
    matches_annotator2 = np.sum(y_pred == y_annotator2)
    matches_annotator3 = np.sum(y_pred == y_annotator3)
    matches_multiple = np.sum(
        ((y_pred == y_annotator1).astype(int) + 
         (y_pred == y_annotator2).astype(int) + 
         (y_pred == y_annotator3).astype(int)) > 1
    )
    
    metrics["match_statistics"] = {
        "total_samples": len(y_pred),
        "matches_any": int(np.sum(matches_any)),
        "matches_annotator1": int(matches_annotator1),
        "matches_annotator2": int(matches_annotator2),
        "matches_annotator3": int(matches_annotator3),
        "matches_multiple": int(matches_multiple),
    }
    
    return metrics
