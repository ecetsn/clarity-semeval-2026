from __future__ import annotations

from typing import Sequence

from .openrouter_classifier import OpenRouterEmbeddingClassifier
from .tfidf_classifier import TfidfLogisticClassifier
from .zero_shot import ZeroShotClassifier
from .xgboost_classifier import TfidfXGBoostClassifier


def create_model(model_config: dict, label_list: Sequence[str]):
    model_type = model_config.get("type")
    params = model_config.get("params", {})

    if model_type == "tfidf_logreg":
        return TfidfLogisticClassifier(label_list=label_list, **params)
    if model_type == "openrouter_logreg":
        return OpenRouterEmbeddingClassifier(label_list=label_list, **params)
    if model_type == "zero_shot":
        return ZeroShotClassifier(label_list=label_list, **params)
    if model_type == "tfidf_xgboost":
        return TfidfXGBoostClassifier(label_list=label_list, **params)

    raise ValueError(f"Unsupported model type: {model_type}")
