from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer
from representation.encoders.base_encoder import BaseEncoder
from typing import List


class DistilBERTEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pooling: str = "mean",
        freeze: bool = True,
        device: str | None = None,
    ):
        super().__init__(pooling)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        outputs = self.encoder(**inputs)
        return self.pool(outputs.last_hidden_state, inputs["attention_mask"])
    def freeze(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True
