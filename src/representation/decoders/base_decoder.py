import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

    def pool(self, hidden_states, attention_mask):
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == "last":
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
