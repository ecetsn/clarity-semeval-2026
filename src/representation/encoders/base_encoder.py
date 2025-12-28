from __future__ import annotations
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """
    Abstract base class for all encoder models.
    Handles pooling over token-level representations.
    """

    def __init__(self, pooling: str = "masked_mean"):
        super().__init__()
        self.pooling = pooling

    def pool(
        self,
        hidden_states: torch.Tensor,   # (B, T, H)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:

        if self.pooling in {"mean", "masked_mean"}:
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1)
            masked = hidden_states.masked_fill(mask == 0, -1e9)
            return masked.max(dim=1).values

        elif self.pooling == "cls":
            return hidden_states[:, 0]

        elif self.pooling == "last_non_pad":
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
