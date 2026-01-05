# src/representation/decoders/qwen_decoder.py

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.representation.decoders.base_decoder import BaseDecoder
from src.representation.decoders.lora_config import LoRAConfig


class QwenDecoder(BaseDecoder):
    def __init__(
        self,
        model_name: str,
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: str = "cpu",
        max_length: int = 256,
        trust_remote_code: bool = True,
        # quantization knobs
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(pooling=pooling, device=device)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right padding is fine for masked_mean / last_non_pad pooling
        self.tokenizer.padding_side = "right"

        # Load model
        quant_cfg = None
        device_map = None

        if load_in_4bit or load_in_8bit:
            # Only import when needed
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if str(torch_dtype).lower() == "bfloat16" else torch.float16

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=bool(load_in_4bit),
                load_in_8bit=bool(load_in_8bit),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"  # put quantized model on GPU automatically

        model_dtype = torch.bfloat16 if str(torch_dtype).lower() == "bfloat16" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_safetensors=True,
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype=model_dtype,
        )

        # If not quantized, move explicitly
        if quant_cfg is None:
            self.model = self.model.to(self.device)

        # Training/memory critical
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
            self.model.config.return_dict = True

        # Hidden size
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not infer hidden_size from Qwen config.")
        self.output_dim = int(hidden_size)

        if freeze:
            self.freeze()

    def _base_transformer(self):
        """
        Return the underlying base transformer module that outputs last_hidden_state.
        Works for both plain HF model and PEFT-wrapped model.
        """
        # PEFT wrapper
        if hasattr(self.model, "get_base_model") and callable(self.model.get_base_model):
            base = self.model.get_base_model()
        else:
            base = self.model

        # Qwen2ForCausalLM exposes `.model` as the transformer
        if hasattr(base, "model"):
            return base.model
        return base

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Move inputs to the same device as the model's first param (safe with device_map="auto")
        model_device = next(self.model.parameters()).device
        batch = {k: v.to(model_device) for k, v in batch.items()}

        # Ensure cache off
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        base = self._base_transformer()
        out = base(**batch, return_dict=True)

        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Base transformer did not return last_hidden_state for Qwen.")

        return self.pool(hidden, batch["attention_mask"])

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def enable_lora(self, cfg: LoRAConfig):
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

        # If quantized (k-bit), prepare properly
        self.model = prepare_model_for_kbit_training(self.model)

        peft_cfg = PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=cfg.task_type,
        )
        self.model = get_peft_model(self.model, peft_cfg)

        # keep cache off
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.model.train()
