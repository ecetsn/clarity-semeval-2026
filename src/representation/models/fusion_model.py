import torch
import torch.nn as nn

from src.representation.fusion.concat_fusion import ConcatFusion


class RepresentationFusionModel(nn.Module):
    def __init__(
        self,
        encoder_proj: nn.Module,
        decoder_proj: nn.Module,
        num_classes: int,
    ):
        super().__init__()

        if not hasattr(encoder_proj, "output_dim"):
            raise TypeError("encoder_proj must expose attribute `output_dim`")

        if not hasattr(decoder_proj, "output_dim"):
            raise TypeError("decoder_proj must expose attribute `output_dim`")

        self.encoder_proj = encoder_proj
        self.decoder_proj = decoder_proj

        self.fusion = ConcatFusion()

        self.classifier = nn.Linear(
            encoder_proj.output_dim + decoder_proj.output_dim,
            num_classes,
        )

    def forward(
        self,
        encoder_vec: torch.Tensor,
        decoder_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        encoder_vec: (B, encoder_dim)
        decoder_vec: (B, decoder_dim)
        """
        enc_p = self.encoder_proj(encoder_vec)
        dec_p = self.decoder_proj(decoder_vec)

        fused = self.fusion(enc_p, dec_p)
        logits = self.classifier(fused)
        return logits
