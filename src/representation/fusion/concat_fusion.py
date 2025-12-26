import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_vec: torch.Tensor, dec_vec: torch.Tensor) -> torch.Tensor:
        return torch.cat([enc_vec, dec_vec], dim=-1)
