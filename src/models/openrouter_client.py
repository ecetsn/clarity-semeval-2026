from __future__ import annotations

import time
from typing import Iterable, List, Sequence

import numpy as np
import requests


class OpenRouterEmbeddingClient:
    """Thin wrapper around the OpenRouter embedding REST API."""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
        batch_size: int = 8,
        max_retries: int = 5,
        retry_wait: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        batched_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batched_embeddings.append(self._embed_batch(batch))
        if not batched_embeddings:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(batched_embeddings)

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        payload = {"input": list(texts), "model": self.model}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/ecetsn/clarity-semeval-2026",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()["data"]
                return np.array([item["embedding"] for item in data], dtype=float)

            if attempt == self.max_retries:
                raise RuntimeError(
                    f"OpenRouter embedding request failed after {self.max_retries} attempts "
                    f"(status={response.status_code}, body={response.text})"
                )
            time.sleep(self.retry_wait * attempt)

        raise RuntimeError("Unreachable state while requesting OpenRouter embeddings.")

