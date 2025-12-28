from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

from datasets import Dataset, DatasetDict, load_dataset


# Data container
@dataclass
class DataSplit:
    ids: List[str]
    texts: List[str]
    clarity_labels: List[int]
    evasion_labels: List[int]


# Data module
class QEvasionDataModule:
    """
    Loads and prepares the QEvasion dataset for representation learning
    and downstream classification.
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
        self.validation_size = validation_size
        self.seed = seed

        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self.dataset: Optional[DatasetDict] = None
        self.splits: Dict[str, DataSplit] = {}

    # Public API
    def prepare(self) -> "QEvasionDataModule":
        dataset = load_dataset(self.dataset_name)

        base_train: Dataset = dataset["train"]

        split = base_train.train_test_split(
            test_size=self.validation_size,
            seed=self.seed,
        )

        self.splits["train"] = self._convert_split(
            split["train"], self.max_train_samples
        )
        self.splits["validation"] = self._convert_split(
            split["test"], self.max_val_samples
        )
        self.splits["test"] = self._convert_split(
            dataset["test"], self.max_test_samples
        )

        self.dataset = dataset
        return self

    def get_split(self, name: str) -> DataSplit:
        if name not in self.splits:
            raise ValueError(f"Split '{name}' not found. Call prepare() first.")
        return self.splits[name]

    # Internal helpers
    def _convert_split(
        self, split: Dataset, limit: Optional[int]
    ) -> DataSplit:
        if limit is not None:
            split = split.select(range(min(limit, len(split))))

        ids = [str(i) for i in range(len(split))]

        questions = (
            split[self.question_column]
            if self.question_column in split.column_names
            else [None] * len(split)
        )

        answers = split[self.text_column]

        texts = [
            self._compose_text(q, a) for q, a in zip(questions, answers)
        ]

        clarity_labels = split[self.clarity_label_column]
        evasion_labels = split[self.evasion_label_column]

        return DataSplit(
            ids=ids,
            texts=texts,
            clarity_labels=clarity_labels,
            evasion_labels=evasion_labels,
        )

    @staticmethod
    def _compose_text(question: Optional[str], answer: str) -> str:
        if question and question.strip():
            return f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        return answer.strip()
