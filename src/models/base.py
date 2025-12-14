from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

import numpy as np


class BaseTextClassifier(ABC):
    """Minimal interface implemented by all first-level classifiers."""

    def __init__(self, label_list: Sequence[str]) -> None:
        self.label_list = list(label_list)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        return np.array([self.label_to_id[label] for label in labels], dtype=int)

    def decode_predictions(self, indices: Sequence[int]) -> List[str]:
        return [self.label_list[idx] for idx in indices]

    def predict(self, texts: Sequence[str]) -> List[str]:
        probs = self.predict_proba(texts)
        predictions = np.argmax(probs, axis=1)
        return self.decode_predictions(predictions)

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "BaseTextClassifier":
        ...

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        ...

