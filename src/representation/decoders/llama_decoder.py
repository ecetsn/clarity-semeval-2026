from __future__ import annotations
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.representation.decoders.base_decoder import BaseDecoder
from src.representation.decoders.lora_config import LoRAConfig


class LlamaDecoder(BaseDecoder):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(pooling=pooling, device=device)

        self._lora_applied = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        model_kwargs: Dict[str, Any] = {
            "token": hf_token,
            "output_hidden_states": True,
            "torch_dtype": dtype_map[torch_dtype],
        }

        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if not (load_in_4bit or load_in_8bit):
            self.model.to(self.device)

        self.output_dim = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        if not hasattr(self.model, "hf_device_map"):
            inputs = inputs.to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.hidden_states[-1]
        return self.pool(hidden, inputs["attention_mask"])

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def enable_lora(self, cfg: LoRAConfig):
        if self._lora_applied:
            raise RuntimeError("LoRA already applied")

        self._lora_applied = True

        from peft import LoraConfig as PeftConfig, get_peft_model, TaskType

        peft_cfg = PeftConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias=cfg.bias,
        )

        self.model = get_peft_model(self.model, peft_cfg)
        self.model.train()
