from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class DataSplit:
    """Container that holds the textual inputs and gold labels for a split."""

    ids: List[str]
    texts: List[str]
    labels: List[str]


class QEvasionDataModule:
    """
    Helper that loads the QEvasion dataset from Hugging Face and produces
    train/validation/test splits tailored for clarity/evasion classification.
    """

    def __init__(
        self,
        dataset_name: str = "ailsntua/QEvasion",
        label_column: str = "clarity_label",
        text_column: str = "interview_answer",
        question_column: Optional[str] = "question",
        validation_size: float = 0.2,
        random_seed: int = 13,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        **_: Dict,
    ) -> None:
        self.dataset_name = dataset_name
        self.label_column = label_column
        self.text_column = text_column
        self.question_column = question_column
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self._dataset: Optional[DatasetDict] = None
        self._splits: Dict[str, DataSplit] = {}
        self._label_list: List[str] = []

    def prepare(self) -> "QEvasionDataModule":
        dataset = load_dataset(self.dataset_name)
        base_train: Dataset = dataset["train"]
        validation_fraction = max(0.05, min(0.5, self.validation_size))
        split_dataset = base_train.train_test_split(
            test_size=validation_fraction, seed=self.random_seed
        )

        self._splits["train"] = self._convert_split(
            split_dataset["train"], limit=self.max_train_samples, split_name="train"
        )
        self._splits["validation"] = self._convert_split(
            split_dataset["test"], limit=self.max_val_samples, split_name="validation"
        )
        self._splits["test"] = self._convert_split(
            dataset["test"], limit=self.max_test_samples, split_name="test"
        )
        self._label_list = self._extract_label_list()
        self._dataset = dataset
        return self

    def get_split(self, split_name: str) -> DataSplit:
        if split_name not in self._splits:
            raise ValueError(
                f"Split '{split_name}' is not prepared. Call prepare() first."
            )
        return self._splits[split_name]

    def get_label_list(self) -> List[str]:
        if not self._label_list:
            raise ValueError("Data module has not been prepared yet.")
        return self._label_list

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _extract_label_list(self) -> List[str]:
        labels = self._splits["train"].labels
        # Preserve the natural order from the dataset.
        seen = set()
        ordered = []
        for lbl in labels:
            if lbl not in seen:
                ordered.append(lbl)
                seen.add(lbl)
        return ordered

    def _convert_split(
        self, hf_split: Dataset, limit: Optional[int], split_name: str
    ) -> DataSplit:
        if limit is not None:
            limit = min(limit, len(hf_split))
            hf_split = hf_split.select(range(limit))

        if split_name == "test" and self.label_column == "evasion_label":
            hf_split = self._apply_evasion_test_majority_vote(hf_split)

        if "index" in hf_split.column_names:
            raw_ids = hf_split["index"]
        else:
            raw_ids = list(range(len(hf_split)))
        ids = [str(idx) if idx is not None else str(i) for i, idx in enumerate(raw_ids)]

        questions: List[Optional[str]]
        if self.question_column and self.question_column in hf_split.column_names:
            questions = hf_split[self.question_column]
        else:
            questions = [None] * len(ids)

        answers = hf_split[self.text_column]
        labels = hf_split[self.label_column]
        texts = [
            self._compose_text(question, answer)
            for question, answer in zip(questions, answers)
        ]
        return DataSplit(ids=ids, texts=texts, labels=labels)

    def _apply_evasion_test_majority_vote(self, hf_split: Dataset) -> Dataset:
        """Resolve evasion labels on the test split via 2-of-3 annotator majority vote."""
        required = {"annotator1", "annotator2", "annotator3"}
        missing = required - set(hf_split.column_names)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"Cannot resolve evasion labels for test split; missing columns: {missing_cols}"
            )

        def assign_labels(batch: Dict[str, List[Optional[str]]]) -> Dict[str, List]:
            resolved_labels: List[Optional[str]] = []
            keep_mask: List[bool] = []
            for a1, a2, a3 in zip(
                batch["annotator1"], batch["annotator2"], batch["annotator3"]
            ):
                votes = [
                    str(v).strip()
                    for v in (a1, a2, a3)
                    if v is not None and str(v).strip()
                ]
                counts = Counter(votes)
                if not counts:
                    resolved_labels.append(None)
                    keep_mask.append(False)
                    continue
                top_label, count = counts.most_common(1)[0]
                has_majority = count >= 2
                resolved_labels.append(top_label if has_majority else None)
                keep_mask.append(has_majority)
            batch["_keep_mask"] = keep_mask
            batch[self.label_column] = resolved_labels
            return batch

        updated = hf_split.map(
            assign_labels, batched=True, load_from_cache_file=False, desc="Majority vote"
        )
        filtered = updated.filter(lambda example: example["_keep_mask"])
        return filtered.remove_columns("_keep_mask")

    @staticmethod
    def _compose_text(question: Optional[str], answer: str) -> str:
        if question and question.strip():
            return f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        return answer.strip()
