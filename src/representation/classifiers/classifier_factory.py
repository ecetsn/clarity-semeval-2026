from typing import Any, Dict

from .mlp_classifier import MLPClassifier
from .base_classifier import BaseClassifier


def build_classifier(cfg: Dict[str, Any]) -> BaseClassifier:
    ctype = str(cfg.get("type", "mlp")).lower().strip()
    params = dict(cfg.get("params", {}))

    # Normalize "linear" into an MLP with no hidden layers
    if ctype == "linear":
        params["hidden_dims"] = []
        return MLPClassifier(**params)

    if ctype == "mlp":
        # hidden_dims must exist; default to linear behavior if omitted
        params.setdefault("hidden_dims", [])
        return MLPClassifier(**params)

    raise ValueError(f"Unknown classifier type: {ctype}")
