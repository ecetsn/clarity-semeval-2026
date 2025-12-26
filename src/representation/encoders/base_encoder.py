from __future__ import annotations
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """
    Abstract base class for all encoder models.
    """

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        hidden_states: (B, T, H)
        attention_mask: (B, T)
        """
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        elif self.pooling == "cls":
            return hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
