from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.env import load_and_validate_env

from .base import BaseTextClassifier
from .openrouter_client import OpenRouterEmbeddingClient


class OpenRouterEmbeddingClassifier(BaseTextClassifier):
    """
    Classifier that obtains dense embeddings from an OpenRouter model and fits
    a multinomial logistic regression layer on top of those representations.
    """

    def __init__(
        self,
        label_list: Sequence[str],
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        batch_size: int = 8,
        c: float = 1.0,
        max_iter: int = 200,
    ) -> None:
        super().__init__(label_list)
        secrets = load_and_validate_env(["OPENROUTER_API_KEY"])
        key = api_key or secrets["OPENROUTER_API_KEY"]
        url = api_url or secrets["OPENROUTER_API_URL"]

        self.client = OpenRouterEmbeddingClient(
            api_key=key,
            api_url=url,
            model=model_name,
            batch_size=batch_size,
            verbose=True,
        )
        self.classifier = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_dim: Optional[int] = None

    def fit(self, texts, labels) -> "OpenRouterEmbeddingClassifier":
        y = self.encode_labels(labels)
        x = self._get_embeddings(texts)
        self.classifier.fit(x, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        x = self._get_embeddings(texts)
        return self.classifier.predict_proba(x)

    # ------------------------------------------------------------------ #
    def _get_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = self._embedding_dim or 0
            return np.zeros((0, dim))
        to_compute = [text for text in texts if text not in self._embedding_cache]
        if to_compute:
            print(
                f"[OpenRouter] Computing embeddings for {len(to_compute)} new texts "
                f"(cache size={len(self._embedding_cache)}, batch_size={self.client.batch_size})"
            )
            new_embeddings = self.client.embed(to_compute)
            for text, emb in zip(to_compute, new_embeddings):
                self._embedding_cache[text] = emb
                if self._embedding_dim is None:
                    self._embedding_dim = emb.shape[0]

        ordered_embeddings = [self._embedding_cache[text] for text in texts]
        return np.vstack(ordered_embeddings)
