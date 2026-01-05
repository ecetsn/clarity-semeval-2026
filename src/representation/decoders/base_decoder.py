# src/representation/decoders/base_decoder.py

from __future__ import annotations

import torch
import torch.nn as nn

from src.representation.decoders.lora_config import LoRAConfig


class BaseDecoder(nn.Module):
    """
    Base class for all decoders.

    Contract:
      - forward(texts: list[str]) -> torch.Tensor of shape (B, D)
      - self.output_dim must be set by subclass after model load
      - self.pool(...) supports pooling token-level states -> sentence vector
      - enable_lora(cfg) may be overridden by subclasses that support PEFT/LoRA
    """

    def __init__(self, pooling: str = "masked_mean", device: str | None = None):
        super().__init__()
        self.pooling = str(pooling)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim: int | None = None

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, T, D)
        attention_mask: (B, T) with 1 for tokens, 0 for pad
        returns: (B, D)
        """
        if self.pooling == "masked_mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
            summed = (hidden_states * mask).sum(dim=1)                   # (B, D)
            denom = mask.sum(dim=1).clamp(min=1.0)                       # (B, 1)
            return summed / denom

        elif self.pooling == "last_non_pad":
            # lengths = number of real tokens per sample
            lengths = attention_mask.sum(dim=1)  # (B,)
            # last index = lengths-1, but clamp to 0 to avoid -1 for fully padded rows
            last_idx = (lengths - 1).clamp(min=0).long()
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, last_idx]  # (B, D)

        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked = hidden_states.masked_fill(~mask, -1e9)
            return masked.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def enable_lora(self, cfg: LoRAConfig) -> None:
        """
        Subclasses that support LoRA should override this.
        """
        raise NotImplementedError("This decoder does not implement LoRA injection.")

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
