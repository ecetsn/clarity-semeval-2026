import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, pooling: str = "masked_mean", device: str | None = None):
        super().__init__()
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim: int | None = None

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "masked_mean":
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

        elif self.pooling == "last_non_pad":
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths]

        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1)
            masked = hidden_states.masked_fill(mask == 0, -1e9)
            return masked.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def enable_lora(self, **kwargs):
        raise NotImplementedError("This decoder does not implement LoRA injection.")
