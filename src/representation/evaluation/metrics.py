from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    label_list: List[str],
) -> Dict[str, float]:
    """
    Computes accuracy, macro/weighted precision-recall-F1,
    and per-class metrics.
    """

    metrics: Dict[str, float] = {
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

    per_class = {}
    for label in label_list:
        per_class[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        metrics[f"{label}_f1"] = float(report[label]["f1-score"])

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

    return metrics
