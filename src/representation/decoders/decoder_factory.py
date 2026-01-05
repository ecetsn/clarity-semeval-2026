from typing import Dict

from src.representation.decoders.gpt2_decoder import GPT2Decoder
from src.representation.decoders.opt_decoder import OPTDecoder
from src.representation.decoders.llama_decoder import LlamaDecoder
from src.representation.decoders.lora_config import LoRAConfig
from src.representation.decoders.qwen_decoder import QwenDecoder



def build_decoder(cfg: Dict, device: str):
    dec_type = cfg["type"]
    pooling = cfg.get("pooling", "masked_mean")
    freeze = cfg.get("freeze", True)

    # GPT-2 (baseline)
    if dec_type == "gpt2":
        decoder = GPT2Decoder(
            model_name=cfg.get("model_name", "gpt2"),
            pooling=pooling,
            freeze=freeze,
            device=device,
        )

    # OPT (mid-capacity)
    elif dec_type == "opt":
        decoder = OPTDecoder(
            model_name=cfg.get("model_name", "facebook/opt-1.3b"),
            pooling=pooling,
            freeze=freeze,
            device=device,
        )

    # LLaMA (heavy)
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

    elif dec_type == "qwen":
        if cfg.get("model_name") is None:
            raise ValueError("Qwen decoder requires `model_name` in YAML (e.g., Qwen/Qwen2-1.5B).")
        quant = cfg.get("quantization", {})
        decoder = QwenDecoder(
            model_name=cfg.get("model_name"),
            pooling=pooling,
            freeze=freeze,
            device=device,
            max_length=cfg.get("max_length", 128),
            load_in_4bit=quant.get("load_in_4bit", False),
            load_in_8bit=quant.get("load_in_8bit", False),
            torch_dtype=quant.get("torch_dtype", "bfloat16"),
        )


    else:
        raise ValueError(f"Unknown decoder type: {dec_type}")

    # Optional LoRA (decoder-agnostic)
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        lora_config = LoRAConfig(
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),

        )
        decoder.enable_lora(lora_config)

    return decoder
