from typing import Dict

from src.representation.decoders.gpt2_decoder import GPT2Decoder
from src.representation.decoders.opt_decoder import OPTDecoder
from src.representation.decoders.llama_decoder import LlamaDecoder
from src.representation.decoders.lora_config import LoRAConfig


def build_decoder(cfg: Dict, device: str):
    dec_type = cfg["type"]
    pooling = cfg.get("pooling", "masked_mean")
    freeze = cfg.get("freeze", True)

    # ----------------------------
    # GPT-2 (baseline)
    # ----------------------------
    if dec_type == "gpt2":
        decoder = GPT2Decoder(
            model_name=cfg.get("model_name", "gpt2"),
            pooling=pooling,
            freeze=freeze,
            device=device,
        )

    # ----------------------------
    # OPT (mid-capacity)
    # ----------------------------
    elif dec_type == "opt":
        decoder = OPTDecoder(
            model_name=cfg.get("model_name", "facebook/opt-1.3b"),
            pooling=pooling,
            freeze=freeze,
            device=device,
        )

    # ----------------------------
    # LLaMA (heavy)
    # ----------------------------
    elif dec_type == "llama":
        quant = cfg.get("quantization", {})

        decoder = LlamaDecoder(
            model_name=cfg.get("model_name", "meta-llama/Meta-Llama-3-8B"),
            pooling=pooling,
            freeze=freeze,
            device=device,
            hf_token=cfg.get("hf_token"),
            load_in_4bit=quant.get("load_in_4bit", False),
            load_in_8bit=quant.get("load_in_8bit", False),
            torch_dtype=quant.get("torch_dtype", "bfloat16"),
        )

    else:
        raise ValueError(f"Unknown decoder type: {dec_type}")

    # ----------------------------
    # Optional LoRA (decoder-agnostic)
    # ----------------------------
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        if hasattr(decoder, "unfreeze"):
            decoder.unfreeze()

        lora_config = LoRAConfig(
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
        )

        decoder.enable_lora(lora_config)

    return decoder
