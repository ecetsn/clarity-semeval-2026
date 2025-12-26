from transformers import AutoTokenizer, AutoModelForCausalLM
from representation.decoders.base_decoder import BaseDecoder
import torch
from typing import List


class GPT2Decoder(BaseDecoder):
    def __init__(
        self,
        model_name="gpt2",
        pooling="mean",
        freeze=True,
        device=None,
    ):
        super().__init__(pooling)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )

        if freeze:
            for p in self.decoder.parameters():
                p.requires_grad = False

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder.to(self.device)

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

