from __future__ import annotations

from typing import Dict, List, Union, Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

Label = Union[int, str]


def compute_classification_metrics(
    y_true: List[Label],
    y_pred: List[Label],
    label_list: List[Label],
) -> Dict[str, Any]:
    """
    Computes accuracy, macro/weighted precision-recall-F1,
    and per-class metrics.

    Fully JSON-serializable.
    """

    metrics: Dict[str, Any] = {
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

    report = classification_report(
        y_true,
        y_pred,
        labels=label_list,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    # -----------------------
    # Per-class metrics
    # -----------------------
    per_class: Dict[Label, Dict[str, float]] = {}

    for label in label_list:
        label_key = str(label)
        per_class[label] = {
            "precision": float(report[label_key]["precision"]),
            "recall": float(report[label_key]["recall"]),
            "f1": float(report[label_key]["f1-score"]),
            "support": int(report[label_key]["support"]),
        }
        metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])

    metrics["per_class"] = per_class

    # -----------------------
    # Aggregate details
    # -----------------------
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

    return metrics
