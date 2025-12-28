from collections import Counter
import torch
from typing import List, Dict


def build_label_mapping(labels: List[str]) -> Dict[str, int]:
    unique = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(unique)}


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = "inverse"
) -> torch.Tensor:
    """
    Returns a weight tensor of shape (num_classes,)
    """
    counts = Counter(labels)

    if method == "inverse":
        weights = [
            1.0 / counts.get(i, 1) for i in range(num_classes)
        ]
    elif method == "balanced":
        total = sum(counts.values())
        weights = [
            total / (num_classes * counts.get(i, 1))
            for i in range(num_classes)
        ]
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return torch.tensor(weights, dtype=torch.float)
