# src/representation/projections/base_projection.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseProjection(nn.Module, ABC):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
