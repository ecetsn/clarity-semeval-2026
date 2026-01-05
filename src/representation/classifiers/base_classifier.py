import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseClassifier(nn.Module, ABC):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
