from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from src.representation.evaluation.metrics import (
    compute_classification_metrics,
)
from src.representation.evaluation.confusion_matrix import (
    compute_confusion_matrix,
)

Label = Union[int, str]


def evaluate_predictions(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    label_list: Sequence[Label],
    *,
    compute_cm: bool = True,
    cm_normalize: Optional[str] = "true",
    head_k: int = 20,
) -> Dict[str, object]:
    """
    Unified evaluator for classification tasks.

    Returns a fully JSON-serializable dictionary.
    """

    results: Dict[str, object] = {}

    # Scalar metrics
    metrics = compute_classification_metrics(
        y_true=list(y_true),
        y_pred=list(y_pred),
        label_list=list(label_list),
    )
    results["metrics"] = metrics

    # Confusion matrix
    if compute_cm:
        cm = compute_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            label_list=label_list,
            normalize=cm_normalize,
        )
        # Convert to JSON-safe format
        results["confusion_matrix"] = cm.tolist()
    else:
        results["confusion_matrix"] = None

    # Debug heads
    results["y_true_ids_head"] = list(y_true[:head_k])
    results["y_pred_ids_head"] = list(y_pred[:head_k])

    return results
