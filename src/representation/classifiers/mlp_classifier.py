import torch
import torch.nn as nn

from .base_classifier import BaseClassifier


class MLPClassifier(BaseClassifier):
    """
    hidden_dims controls depth/width:
      []            -> Linear(input_dim -> num_classes)
      [256]         -> input_dim -> 256 -> num_classes
      [256, 256]    -> input_dim -> 256 -> 256 -> num_classes
      [256,256,256] -> input_dim -> 256 -> 256 -> 256 -> num_classes

    activation applies to hidden layers only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__(num_classes)

        act = str(activation).lower().strip()
        if act == "gelu":
            act_layer = nn.GELU()
        elif act == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(num_classes)]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers.append(act_layer)
                if dropout and float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
