# src/representation/projections/mlp_projection.py
import torch
import torch.nn as nn
from .base_projection import BaseProjection


class ProjectionMLP(BaseProjection):
    """
    MLP projection with configurable depth.

    n_layers counts Linear layers:
      - n_layers=1: Linear(input_dim -> output_dim)
      - n_layers=2: Linear(input_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> output_dim)
      - n_layers=3: Linear(input_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> output_dim)
      etc.

    dropout is applied after each GELU (i.e., after each hidden layer).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__(output_dim)

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        act = activation.lower()
        if act == "gelu":
            act_layer = nn.GELU()
        elif act == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []

        # n_layers == 1: pure linear projection
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
            self.net = nn.Sequential(*layers)
            return

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_layer)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Middle hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_layer)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
