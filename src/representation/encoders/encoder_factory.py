# src/representation/encoders/encoder_factory.py

from typing import Dict
from src.representation.encoders.distilbert_encoder import DistilBERTEncoder


def build_encoder(cfg: Dict, device: str):
    enc_type = cfg["type"]

    if enc_type == "distilbert":
        return DistilBERTEncoder(
            model_name=cfg.get("model_name", "distilbert-base-uncased"),
            pooling=cfg.get("pooling", "masked_mean"),
            freeze=cfg.get("freeze", True),
            device=device,
        )

    raise ValueError(f"Unknown encoder type: {enc_type}")
