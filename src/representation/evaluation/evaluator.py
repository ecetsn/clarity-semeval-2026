from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.representation.evaluation.metrics import (
    compute_classification_metrics,
)
from src.representation.evaluation.confusion_matrix import (
    compute_confusion_matrix,
)


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    label_list: List[str],
    *,
    compute_cm: bool = True,
    cm_normalize: Optional[str] = "true",
) -> Dict[str, object]:
    """
    Unified evaluator for classification tasks.

    Returns:
        {
            "metrics": {...},
            "confusion_matrix": np.ndarray | None
        }
    """

    results: Dict[str, object] = {}

    # 1. Scalar metrics
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    results["metrics"] = metrics

    # 2. Confusion matrix
    if compute_cm:
        cm = compute_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            label_list=label_list,
            normalize=cm_normalize,
        )
        results["confusion_matrix"] = cm
    else:
        results["confusion_matrix"] = None

    return results
