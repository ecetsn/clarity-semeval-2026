from collections import Counter
import torch
import torch.nn as nn


def build_loss(
    *,
    labels: list[int],
    num_classes: int,
    device: str,
    class_weighted: bool = False,
    scheme: str = "inverse_frequency",
) -> nn.Module:
    """
    Factory for classification losses.
    """

    if not class_weighted:
        return nn.CrossEntropyLoss()

    counts = Counter(labels)
    total = sum(counts.values())

    if scheme == "inverse_frequency":
        weights = torch.tensor(
            [
                total / counts[i] if i in counts else 0.0
                for i in range(num_classes)
            ],
            dtype=torch.float,
            device=device,
        )

        # Normalize for stability
        weights = weights / weights.mean()

    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    return nn.CrossEntropyLoss(weight=weights)
