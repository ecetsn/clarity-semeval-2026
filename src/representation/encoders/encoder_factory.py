# src/representation/encoders/encoder_factory.py

from typing import Dict

from .distilbert_encoder import DistilBERTEncoder
from .roberta_encoder import RoBERTaEncoder
from .deberta_encoder import DeBERTaEncoder


def build_encoder(cfg: Dict, device: str):
    enc_type = cfg["type"]

    if enc_type == "distilbert":
        return DistilBERTEncoder(
            model_name=cfg.get("model_name", "distilbert-base-uncased"),
            pooling=cfg.get("pooling", "masked_mean"),
            freeze=cfg.get("freeze", True),
            device=device,
        )

    if enc_type == "roberta":
        return RoBERTaEncoder(
            model_name=cfg.get("model_name", "roberta-base"),
            pooling=cfg.get("pooling", "masked_mean"),
            freeze=cfg.get("freeze", True),
            device=device,
        )

    if enc_type == "deberta":
        return DeBERTaEncoder(
            model_name=cfg.get("model_name", "microsoft/deberta-v3-base"),
            pooling=cfg.get("pooling", "masked_mean"),
            freeze=cfg.get("freeze", True),
            device=device,
        )

    raise ValueError(f"Unknown encoder type: {enc_type}")
