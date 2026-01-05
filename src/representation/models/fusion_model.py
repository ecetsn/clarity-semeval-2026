import torch
import torch.nn as nn

from src.representation.fusion.concat_fusion import ConcatFusion
from src.representation.classifiers.classifier_factory import build_classifier

class RepresentationFusionModel(nn.Module):
    """
    Supports branch contribution ablations via `mode`:

      - "concat":       use encoder + decoder (current behavior)
      - "encoder_only": ignore decoder branch entirely
      - "decoder_only": ignore encoder branch entirely

    The forward signature stays the same, so train_from_yaml does not need
    to change how it calls fusion_model(enc_vec, dec_vec).
    """

    def __init__(
        self,
        encoder_proj: nn.Module,
        decoder_proj: nn.Module,
        num_classes: int,
        mode: str = "concat",
        classifier_cfg: dict | None = None,
    ):
        super().__init__()

        if not hasattr(encoder_proj, "output_dim"):
            raise TypeError("encoder_proj must expose attribute `output_dim`")

        if not hasattr(decoder_proj, "output_dim"):
            raise TypeError("decoder_proj must expose attribute `output_dim`")

        mode = str(mode).lower().strip()
        if mode not in {"concat", "encoder_only", "decoder_only"}:
            raise ValueError(f"Unknown fusion mode: {mode}")

        self.mode = mode
        self.encoder_proj = encoder_proj
        self.decoder_proj = decoder_proj

        self.fusion = ConcatFusion()

        if self.mode == "concat":
            in_dim = encoder_proj.output_dim + decoder_proj.output_dim
        elif self.mode == "encoder_only":
            in_dim = encoder_proj.output_dim
        else:
            in_dim = decoder_proj.output_dim

        if classifier_cfg is None:
            classifier_cfg = {
                "type": "linear",
                "params": {"input_dim": in_dim, "num_classes": num_classes},
            }
        else:
            # fill required params if omitted
            classifier_cfg = dict(classifier_cfg)
            params = dict(classifier_cfg.get("params", {}))
            params.setdefault("input_dim", in_dim)
            params.setdefault("num_classes", num_classes)
            classifier_cfg["params"] = params

        self.classifier = build_classifier(classifier_cfg)

    def forward(self, encoder_vec: torch.Tensor, decoder_vec: torch.Tensor) -> torch.Tensor:
        """
        encoder_vec: (B, encoder_dim)
        decoder_vec: (B, decoder_dim)
        """
        if self.mode == "encoder_only":
            enc_p = self.encoder_proj(encoder_vec)
            logits = self.classifier(enc_p)
            return logits

        if self.mode == "decoder_only":
            dec_p = self.decoder_proj(decoder_vec)
            logits = self.classifier(dec_p)
            return logits

        # concat (default)
        enc_p = self.encoder_proj(encoder_vec)
        dec_p = self.decoder_proj(decoder_vec)
        fused = self.fusion(enc_p, dec_p)
        logits = self.classifier(fused)
        return logits
