from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .base import BaseTextClassifier


class TfidfLogisticClassifier(BaseTextClassifier):
    """
    Lightweight baseline that represents the concatenated question/answer text
    with TF-IDF features and optimizes a multinomial logistic regression head.
    """

    def __init__(
        self,
        label_list: Sequence[str],
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 20000,
        c: float = 2.0,
        max_iter: int = 1000,
        min_df: int = 2,
    ) -> None:
        super().__init__(label_list)
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=min_df,
        )
        self.classifier = LogisticRegression(
            C=c, max_iter=max_iter, class_weight="balanced"
        )

    def fit(self, texts, labels) -> "TfidfLogisticClassifier":
        y = self.encode_labels(labels)
        x = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(x)
