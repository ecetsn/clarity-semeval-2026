# src/representation/data/qev_datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class DataSplit:
    """
    NOTE:
    - On HF test split, evasion_label is empty; use annotator1/2/3 instead.
    """
    ids: List[str]
    texts: List[str]

    clarity_labels: List[str]
    evasion_labels: List[str]

    # Test-time evasion ground truth (3 annotators). These columns exist in both splits,
    # but are typically null in HF train split.
    annotator1: List[Optional[str]]
    annotator2: List[Optional[str]]
    annotator3: List[Optional[str]]


class QEvasionDataModule:
    """
    Loads and prepares the QEvasion dataset for representation learning and evaluation.

    Current behavior:
    - Uses HF "train" split (≈3.4k) and creates local train/validation split via train_test_split.
    - Loads HF "test" split (308).

    IMPORTANT for Task 2 (evasion):
    - HF test split has evasion_label empty.
    - Ground truth is the set {annotator1, annotator2, annotator3} (any is accepted).
    """

    def __init__(
        self,
        dataset_name: str = "ailsntua/QEvasion",
        text_column: str = "interview_answer",
        question_column: str = "question",
        clarity_label_column: str = "clarity_label",
        evasion_label_column: str = "evasion_label",
        validation_size: float = 0.2,
        seed: int = 13,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.question_column = question_column
        self.clarity_label_column = clarity_label_column
        self.evasion_label_column = evasion_label_column

        self.validation_size = float(validation_size)
        self.seed = int(seed)

        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self.dataset: Optional[DatasetDict] = None
        self.splits: Dict[str, DataSplit] = {}

    # Public API
    def prepare(self) -> "QEvasionDataModule":
        dataset: DatasetDict = load_dataset(self.dataset_name)

        if "train" not in dataset:
            raise KeyError(f"Dataset '{self.dataset_name}' has no 'train' split.")
        if "test" not in dataset:
            raise KeyError(f"Dataset '{self.dataset_name}' has no 'test' split.")

        base_train: Dataset = dataset["train"]
        split = base_train.train_test_split(test_size=self.validation_size, seed=self.seed)

        self.splits["train"] = self._convert_split(split["train"], limit=self.max_train_samples)
        self.splits["validation"] = self._convert_split(split["test"], limit=self.max_val_samples)
        self.splits["test"] = self._convert_split(dataset["test"], limit=self.max_test_samples)

        self.dataset = dataset
        return self

    def get_split(self, name: str) -> DataSplit:
        if name not in self.splits:
            raise ValueError(f"Split '{name}' not found. Call prepare() first.")
        return self.splits[name]

    def get_test_evasion_gold_sets(self) -> List[Set[str]]:
        """
        For HF test split, returns per-example acceptable label sets for evasion:
            gold[i] = {annotator1[i], annotator2[i], annotator3[i]} \ {None/""}
        """
        test = self.get_split("test")
        gold_sets: List[Set[str]] = []

        for a1, a2, a3 in zip(test.annotator1, test.annotator2, test.annotator3):
            s: Set[str] = set()
            for v in (a1, a2, a3):
                nv = self._normalize_label(v)
                if nv is not None:
                    s.add(nv)
            gold_sets.append(s)

        return gold_sets

    # Internal helpers
    def _convert_split(self, split: Dataset, limit: Optional[int]) -> DataSplit:
        if limit is not None:
            split = split.select(range(min(int(limit), len(split))))

        ids = [str(i) for i in range(len(split))]

        questions = (
            split[self.question_column]
            if self.question_column in split.column_names
            else [None] * len(split)
        )

        answers = split[self.text_column] if self.text_column in split.column_names else ["" for _ in range(len(split))]
        texts = [self._compose_text(q, a) for q, a in zip(questions, answers)]

        # Labels are STRINGS in the HF dataset (or "" in test evasion_label).
        clarity_raw = split[self.clarity_label_column] if self.clarity_label_column in split.column_names else ["" for _ in range(len(split))]
        evasion_raw = split[self.evasion_label_column] if self.evasion_label_column in split.column_names else ["" for _ in range(len(split))]

        clarity_labels = [self._normalize_label_strict(v) for v in clarity_raw]
        evasion_labels = [self._normalize_label_strict(v) for v in evasion_raw]

        # Annotators: may be present but null in train; present and non-null in test.
        a1 = split["annotator1"] if "annotator1" in split.column_names else [None] * len(split)
        a2 = split["annotator2"] if "annotator2" in split.column_names else [None] * len(split)
        a3 = split["annotator3"] if "annotator3" in split.column_names else [None] * len(split)

        annotator1 = [self._normalize_label(v) for v in a1]
        annotator2 = [self._normalize_label(v) for v in a2]
        annotator3 = [self._normalize_label(v) for v in a3]

        return DataSplit(
            ids=ids,
            texts=texts,
            clarity_labels=clarity_labels,
            evasion_labels=evasion_labels,
            annotator1=annotator1,
            annotator2=annotator2,
            annotator3=annotator3,
        )

    @staticmethod
    def _compose_text(question: Optional[str], answer: str) -> str:
        q = (question or "").strip()
        a = (answer or "").strip()
        if q:
            return f"Question: {q}\nAnswer: {a}"
        return a

    @staticmethod
    def _normalize_label(value: Optional[str]) -> Optional[str]:
        """
        Normalize labels coming from annotator columns:
        - None -> None
        - "" or whitespace -> None
        - else -> stripped string
        """
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        v = value.strip()
        return v if v else None

    @staticmethod
    def _normalize_label_strict(value: Optional[str]) -> str:
        """
        Normalize labels for single-label fields (clarity_label/evasion_label):
        Keep them as strings; empty becomes "" (caller decides how to use it).
        """
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()
