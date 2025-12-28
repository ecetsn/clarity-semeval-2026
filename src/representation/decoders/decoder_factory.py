# src/representation/decoders/decoder_factory.py
from typing import Dict

from src.representation.decoders.gpt2_decoder import GPT2Decoder
from src.representation.decoders.lora_config import LoRAConfig


def build_decoder(cfg: Dict, device: str):
    dec_type = cfg["type"]

    if dec_type == "gpt2":
        decoder = GPT2Decoder(
            model_name=cfg.get("model_name", "gpt2"),
            pooling=cfg.get("pooling", "masked_mean"),
            freeze=cfg.get("freeze", True),
            device=device,
        )

        lora_cfg = cfg.get("lora", {})
        if lora_cfg.get("enabled", False):
            decoder.enable_lora(
                r=lora_cfg.get("r", 8),
                alpha=lora_cfg.get("alpha", 16),
                dropout=lora_cfg.get("dropout", 0.1),
            )

        return decoder

    raise ValueError(f"Unknown decoder type: {dec_type}")
