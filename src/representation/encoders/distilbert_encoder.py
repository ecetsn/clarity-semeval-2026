# src/representation/encoders/distilbert_encoder.py

from __future__ import annotations
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

from src.representation.encoders.base_encoder import BaseEncoder


class DistilBERTEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pooling: str = "masked_mean",
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__(pooling=pooling)

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)

        self.output_dim = self.encoder.config.hidden_size  # ✅ ADD THIS

        if freeze:
            self.freeze()

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.encoder(**inputs)
        return self.pool(outputs.last_hidden_state, inputs["attention_mask"])

    def freeze(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
