"""
Hierarchical Evasion -> Clarity mapping
"""
from typing import List, Dict, Any
import numpy as np
from collections import Counter


# Evasion label mappings to Clarity labels
# Based on taxonomy diagram and dataset labels
# Dataset evasion labels: 'Claims ignorance', 'Clarification', 'Declining to answer', 
#                          'Deflection', 'Dodging', 'Explicit', 'General', 'Implicit', 'Partial/half-answer'
# Dataset clarity labels: 'Clear Reply', 'Ambivalent', 'Clear Non-Reply'

# Non-Reply labels (map to "Clear Non-Reply")
NONREPLY = {
    "Claims ignorance",
    "Clarification",
    "Declining to answer"
}

# Reply labels (map to "Clear Reply")
REPLY = {
    "Explicit"
}

# All other evasion labels map to "Ambivalent" (Implicit, Dodging, General, Deflection, Partial/half-answer)


def evasion_to_clarity(evasion_label: str) -> str:
    """
    Map evasion label to clarity label (hierarchical approach)
    
    Args:
        evasion_label: Evasion label string
    
    Returns:
        Clarity label: "Clear Non-Reply", "Clear Reply", or "Ambivalent"
    """
    if evasion_label in NONREPLY:
        return "Clear Non-Reply"
    if evasion_label in REPLY:
        return "Clear Reply"
    # Default to "Ambivalent" for all other evasion labels (Implicit, Dodging, General, Deflection, Partial/half-answer)
    return "Ambivalent"


def map_evasion_predictions_to_clarity(
    evasion_predictions: np.ndarray,
    evasion_label_list: List[str]
) -> np.ndarray:
    """
    Map evasion predictions to clarity predictions using hierarchical mapping
    
    Args:
        evasion_predictions: Array of predicted evasion labels (can be integers or strings)
        evasion_label_list: List of evasion label names (in order of encoding, if predictions are integers)
    
    Returns:
        Array of clarity predictions (encoded as integers)
    """
    # Handle both integer and string predictions
    evasion_pred_labels = []
    for pred in evasion_predictions:
        if isinstance(pred, (int, np.integer)):
            # Integer prediction: use as index
            evasion_pred_labels.append(evasion_label_list[int(pred)])
        elif isinstance(pred, str):
            # String prediction: use directly
            evasion_pred_labels.append(str(pred))
        elif hasattr(np, 'str_') and isinstance(pred, np.str_):
            # NumPy string type (NumPy 1.x compatibility)
            evasion_pred_labels.append(str(pred))
        else:
            # Try to convert to string
            evasion_pred_labels.append(str(pred))
    
    # Map to clarity labels
    clarity_pred_labels = [evasion_to_clarity(ev_label) for ev_label in evasion_pred_labels]
    
    # Convert back to integers (assuming clarity labels are: Ambivalent=0, Clear Non-Reply=1, Clear Reply=2)
    # Note: Dataset uses "Ambivalent" (not "Ambiguous") and order matches dataset: ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
    clarity_label_list = ["Ambivalent", "Clear Non-Reply", "Clear Reply"]
    clarity_pred_encoded = np.array([
        clarity_label_list.index(cl_label) if cl_label in clarity_label_list else 0  # Default to Ambivalent if not found
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
        y_evasion_true: True evasion labels (can be encoded integers or strings)
        y_evasion_pred: Predicted evasion labels (can be encoded integers or strings)
        y_clarity_true: True clarity labels (can be encoded integers or strings)
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

