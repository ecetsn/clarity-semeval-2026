"""
Hierarchical Evasion -> Clarity mapping
"""
from typing import List, Dict, Any
import numpy as np
from collections import Counter


# Evasion label mappings to Clarity labels
# Based on siparismaili01 notebook mapping
NONREPLY = {"Claims ignorance", "Clarification", "Declining to answer"}
REPLY = {"Explicit"}


def evasion_to_clarity(evasion_label: str) -> str:
    """
    Map evasion label to clarity label (hierarchical approach)
    
    Args:
        evasion_label: Evasion label string
    
    Returns:
        Clarity label: "Clear Non-Reply", "Clear Reply", or "Ambiguous"
    """
    if evasion_label in NONREPLY:
        return "Clear Non-Reply"
    if evasion_label in REPLY:
        return "Clear Reply"
    # Default to "Ambiguous" for all other evasion labels
    return "Ambiguous"


def map_evasion_predictions_to_clarity(
    evasion_predictions: np.ndarray,
    evasion_label_list: List[str]
) -> np.ndarray:
    """
    Map evasion predictions to clarity predictions using hierarchical mapping
    
    Args:
        evasion_predictions: Array of predicted evasion labels (encoded as integers)
        evasion_label_list: List of evasion label names (in order of encoding)
    
    Returns:
        Array of clarity predictions (encoded as integers)
    """
    # Convert integer predictions to label strings
    evasion_pred_labels = [evasion_label_list[int(pred)] for pred in evasion_predictions]
    
    # Map to clarity labels
    clarity_pred_labels = [evasion_to_clarity(ev_label) for ev_label in evasion_pred_labels]
    
    # Convert back to integers (assuming clarity labels are: Clear Reply=0, Ambiguous=1, Clear Non-Reply=2)
    # Note: "Ambiguous" is used instead of "Ambivalent" to match dataset labels
    clarity_label_list = ["Clear Reply", "Ambiguous", "Clear Non-Reply"]
    clarity_pred_encoded = np.array([
        clarity_label_list.index(cl_label) if cl_label in clarity_label_list else 1  # Default to Ambiguous if not found
        for cl_label in clarity_pred_labels
    ])
    
    return clarity_pred_encoded


def evaluate_hierarchical_approach(
    y_evasion_true: np.ndarray,
    y_evasion_pred: np.ndarray,
    y_clarity_true: np.ndarray,
    evasion_label_list: List[str],
    clarity_label_list: List[str]
) -> Dict[str, Any]:
    """
    Evaluate hierarchical approach: evasion predictions -> clarity predictions
    
    Args:
        y_evasion_true: True evasion labels (encoded)
        y_evasion_pred: Predicted evasion labels (encoded)
        y_clarity_true: True clarity labels (encoded)
        evasion_label_list: List of evasion label names
        clarity_label_list: List of clarity label names
    
    Returns:
        Dictionary with metrics for hierarchical approach
    """
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    # Map evasion predictions to clarity
    y_clarity_pred_hierarchical = map_evasion_predictions_to_clarity(
        y_evasion_pred, evasion_label_list
    )
    
    # Compute metrics
    accuracy = float(accuracy_score(y_clarity_true, y_clarity_pred_hierarchical))
    macro_f1 = float(f1_score(y_clarity_true, y_clarity_pred_hierarchical, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_clarity_true, y_clarity_pred_hierarchical, average="weighted", zero_division=0))
    
    # Classification report
    report = classification_report(
        y_clarity_true,
        y_clarity_pred_hierarchical,
        labels=list(range(len(clarity_label_list))),
        target_names=clarity_label_list,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "predictions": y_clarity_pred_hierarchical
    }

