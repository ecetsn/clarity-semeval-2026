# src/representation/projections/mlp_projection.py
import torch
import torch.nn as nn
from .base_projection import BaseProjection


class ProjectionMLP(BaseProjection):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
