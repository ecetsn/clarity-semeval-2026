# src/representation/encoders/deberta_encoder.py

import torch
from transformers import AutoModel, AutoTokenizer

from .base_encoder import BaseEncoder


class DeBERTaEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__(pooling=pooling)

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

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
        return self.pool(outputs.last_hidden_state, batch["attention_mask"])

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True
