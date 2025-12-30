from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from .base import BaseTextClassifier


class TfidfXGBoostClassifier(BaseTextClassifier):
    """
    Text classifier that pairs TF-IDF bag-of-ngrams with an XGBoost
    multi-class head (softprob).
    """

    def __init__(
        self,
        label_list: Sequence[str],
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 30000,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 400,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
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
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(self.label_list),
            random_state=random_state,
        )

    def fit(self, texts, labels) -> "TfidfXGBoostClassifier":
        y = self.encode_labels(labels)
        x = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(x)
