# src/representation/decoders/opt_decoder.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.representation.decoders.lora_config import LoRAConfig

from .base_decoder import BaseDecoder


class OPTDecoder(BaseDecoder):
    def __init__(
        self,
        model_name: str = "facebook/opt-350m",
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__(pooling=pooling, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            use_safetensors=True,
        ).to(self.device)

        self.output_dim = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**batch)
        hidden = outputs.hidden_states[-1]  # (B, T, D)

        return self.pool(hidden, batch["attention_mask"])

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True


    def enable_lora(self, cfg: LoRAConfig):
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules or ["q_proj", "v_proj"],
            bias=cfg.bias,
            task_type=cfg.task_type,
        )

        self.model = get_peft_model(self.model, peft_cfg)
        self.model.train()
