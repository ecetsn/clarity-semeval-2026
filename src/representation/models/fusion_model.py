import torch
import torch.nn as nn

from src.representation.projections.mlp_projection import ProjectionMLP
from src.representation.fusion.concat_fusion import ConcatFusion


class RepresentationFusionModel(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        projection_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.encoder_proj = ProjectionMLP(
            input_dim=encoder_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
        )

        self.decoder_proj = ProjectionMLP(
            input_dim=decoder_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
        )

        self.fusion = ConcatFusion()

        self.classifier = nn.Linear(
            in_features=projection_dim * 2,
            out_features=num_classes,
        )

    def forward(
        self,
        encoder_vec: torch.Tensor,
        decoder_vec: torch.Tensor,
    ) -> torch.Tensor:
        enc_p = self.encoder_proj(encoder_vec)
        dec_p = self.decoder_proj(decoder_vec)

        fused = self.fusion(enc_p, dec_p)

        logits = self.classifier(fused)
        return logits
