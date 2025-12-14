from __future__ import annotations

from typing import Dict, Sequence

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from transformers import pipeline
from transformers.utils import logging as hf_logging

from .base import BaseTextClassifier


class ZeroShotClassifier(BaseTextClassifier):
    """Uses an instruction-tuned NLI model for zero-shot multi-class scoring."""

    def __init__(
        self,
        label_list: Sequence[str],
        model_name: str = "facebook/bart-large-mnli",
        hypothesis_template: str = "This response is {}.",
        batch_size: int = 4,
        device: int = -1,
        cache_dir: str = "experiments/cache/zero_shot",
        use_cache: bool = True,
    ) -> None:
        super().__init__(label_list)
        hf_logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Length of IterableDataset .+PipelineChunkIterator object",
        )
        self.pipe = pipeline(
            "zero-shot-classification", model=model_name, device=device
        )
        self.hypothesis_template = hypothesis_template
        self.batch_size = max(1, batch_size)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / self._build_cache_filename(model_name)
        self.cache: Dict[str, np.ndarray] = {}
        if self.use_cache and self.cache_path.exists():
            self._load_cache()

    def fit(self, texts, labels):
        # This model is inference-only; fitting is a no-op for API symmetry.
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_list)))

        results = [None] * len(texts)
        uncached_indices = []
        uncached_samples = []

        if self.use_cache:
            for idx, text in enumerate(texts):
                cached = self._cache_lookup(text)
                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_indices.append(idx)
                    uncached_samples.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_samples = list(texts)

        for start in range(0, len(uncached_samples), self.batch_size):
            batch = uncached_samples[start : start + self.batch_size]
            outputs = self.pipe(
                batch,
                candidate_labels=self.label_list,
                hypothesis_template=self.hypothesis_template,
                multi_label=False,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for local_idx, out in enumerate(outputs):
                global_idx = uncached_indices[start + local_idx]
                converted = self._convert_scores(out)
                results[global_idx] = converted
                self._cache_store(texts[global_idx], converted)

        self._flush_cache()
        return np.vstack(results)

    def _convert_scores(self, output) -> np.ndarray:
        label_to_score = dict(zip(output["labels"], output["scores"]))
        return np.array([label_to_score[label] for label in self.label_list])

    # -------------------- Cache helpers -------------------- #
    def _build_cache_filename(self, model_name: str) -> str:
        safe_model = model_name.replace("/", "_")
        label_sig = hashlib.sha256("|".join(self.label_list).encode("utf-8")).hexdigest()[
            :10
        ]
        return f"{safe_model}_{label_sig}.json"

    def _cache_lookup(self, text: str):
        if not self.use_cache:
            return None
        key = self._hash_text(text)
        entry = self.cache.get(key)
        if entry is None:
            return None
        return np.array(entry, dtype=float)

    def _cache_store(self, text: str, scores: np.ndarray) -> None:
        if not self.use_cache:
            return
        key = self._hash_text(text)
        self.cache[key] = scores.tolist()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            meta_labels = payload.get("meta", {}).get("labels")
            if meta_labels == self.label_list:
                self.cache = payload.get("scores", {})
            else:
                self.cache = {}
        except (json.JSONDecodeError, OSError):
            self.cache = {}

    def _flush_cache(self) -> None:
        if not self.use_cache:
            return
        data = {"meta": {"labels": self.label_list}, "scores": self.cache}
        with open(self.cache_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)
