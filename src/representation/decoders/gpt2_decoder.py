from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from typing import List, Optional

from src.representation.decoders.base_decoder import BaseDecoder
from src.representation.decoders.lora_config import LoRAConfig


class GPT2Decoder(BaseDecoder):
    def __init__(
        self,
        model_name: str = "gpt2",
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(pooling=pooling, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(self.device)

        self.output_dim = self.decoder.config.hidden_size
        self._lora_applied = False

        if freeze:
            self.freeze()

    def enable_lora(self, cfg: LoRAConfig) -> None:
        if self._lora_applied:
            raise RuntimeError("LoRA already applied")

        peft_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=cfg.task_type,
        )

        self.decoder = get_peft_model(self.decoder, peft_cfg)
        self._lora_applied = True
        self.decoder.train()

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        outputs = self.decoder(**inputs)
        hidden = outputs.hidden_states[-1]
        return self.pool(hidden, inputs["attention_mask"])

    def freeze(self) -> None:
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.decoder.train()
        for p in self.decoder.parameters():
            p.requires_grad = True
