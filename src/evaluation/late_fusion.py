from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

from src.data.datamodule import QEvasionDataModule, DataSplit
from src.evaluation.metrics import compute_classification_metrics
from src.models.factory import create_model


@dataclass
class BaseModelResult:
    name: str
    val_probs: np.ndarray
    test_probs: np.ndarray
    calib_probs: Optional[np.ndarray] = None


class DecisionLevelFusion:
    """Combines model probabilities through logistic regression or averaging."""

    def __init__(self, label_list: List[str], fusion_cfg: Dict) -> None:
        self.label_list = label_list
        self.label_to_id = {label: idx for idx, label in enumerate(label_list)}
        self.method = fusion_cfg.get("type", "logistic_regression")
        self.params = fusion_cfg.get("params", {})
        self.model = None
        self.weights = None

    def fit(self, per_model_probs: List[np.ndarray], labels: List[str]) -> "DecisionLevelFusion":
        if not per_model_probs:
            raise ValueError("At least one model is required for fusion.")

        if self.method == "logistic_regression":
            x = np.concatenate(per_model_probs, axis=1)
            y = np.array([self.label_to_id[label] for label in labels])
            params = {"max_iter": 500}
            params.update(self.params)
            self.model = LogisticRegression(**params)
            self.model.fit(x, y)
        elif self.method == "weighted_average":
            if "weights" in self.params:
                weights = np.array(self.params["weights"], dtype=float)
            else:
                weights = np.ones(len(per_model_probs), dtype=float)
            if len(weights) != len(per_model_probs):
                raise ValueError(
                    "Number of weights must match the number of models for weighted_average fusion."
                )
            self.weights = weights / weights.sum()
        else:
            raise ValueError(f"Unsupported fusion method: {self.method}")
        return self

    def predict_proba(self, per_model_probs: List[np.ndarray]) -> np.ndarray:
        if self.method == "logistic_regression":
            if self.model is None:
                raise ValueError("Fusion model is not trained.")
            features = np.concatenate(per_model_probs, axis=1)
            return self.model.predict_proba(features)

        if self.method == "weighted_average":
            if self.weights is None:
                raise ValueError("Fusion model is not initialized.")
            stacked = np.stack(per_model_probs, axis=0)  # (m, n, c)
            weighted = np.tensordot(self.weights, stacked, axes=(0, 0))
            return weighted

        raise ValueError(f"Unsupported fusion method {self.method}")

    def predict(self, per_model_probs: List[np.ndarray]) -> List[str]:
        probs = self.predict_proba(per_model_probs)
        predictions = np.argmax(probs, axis=1)
        return [self.label_list[idx] for idx in predictions]


class LateFusionExperiment:
    """End-to-end runner that trains base models and a decision-level fusion head."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.output_dir = Path(config.get("output_dir", "experiments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paper_reference = config.get("paper_reference") or {}
        self.calibration_result: Optional[Dict] = None

    def run(self) -> Dict:
        dataset_cfg = self.config.get("dataset", {})
        print("[LateFusion] Loading dataset and preparing splits...")
        data_module = QEvasionDataModule(**dataset_cfg).prepare()
        train_split = data_module.get_split("train")
        val_split = data_module.get_split("validation")
        test_split = data_module.get_split("test")
        label_list = data_module.get_label_list()
        print(
            f"[LateFusion] Raw splits -> train={len(train_split.texts)}, "
            f"val={len(val_split.texts)}, test={len(test_split.texts)}"
        )
        train_split = self._resample_train_split(
            train_split,
            dataset_cfg.get("resampling", {}),
            seed=dataset_cfg.get("random_seed", 42),
        )
        calib_fraction = dataset_cfg.get("calibration_fraction", 0.0) or 0.0
        val_split, calib_split = self._split_for_calibration(
            val_split, calib_fraction, dataset_cfg.get("random_seed", 42)
        )
        print(
            f"[LateFusion] Dataset ready -> train={len(train_split.texts)}, "
            f"val={len(val_split.texts)}, test={len(test_split.texts)}, labels={label_list}"
        )
        if self.paper_reference:
            desc = self.paper_reference.get("description", "paper baseline")
            print(f"[LateFusion] Paper reference loaded: {desc}")

        model_results = []
        per_model_metrics = {}
        for idx, model_cfg in enumerate(self.config.get("base_models", [])):
            name = model_cfg.get("name", f"model_{idx}")
            print(f"[LateFusion] Training base model '{name}' ({model_cfg['type']})...")
            (
                val_probs,
                test_probs,
                calib_probs,
                val_preds,
                test_preds,
            ) = self._train_with_replicas(
                model_cfg,
                label_list,
                train_split,
                val_split,
                test_split,
                calib_split,
                seed=dataset_cfg.get("random_seed", 42),
            )
            model_results.append(
                BaseModelResult(
                    name=name, val_probs=val_probs, test_probs=test_probs, calib_probs=calib_probs
                )
            )

            per_model_metrics[name] = {
                "validation": compute_classification_metrics(val_split.labels, val_preds, label_list),
                "test": compute_classification_metrics(test_split.labels, test_preds, label_list),
            }
            self._log_metric_summary(
                name,
                per_model_metrics[name],
                label_list,
            )
            print(f"[LateFusion] Finished base model '{name}'.")

        if not model_results:
            raise ValueError("No base models defined. Please add at least one entry under base_models.")

        val_prob_list = [result.val_probs for result in model_results]
        test_prob_list = [result.test_probs for result in model_results]
        calib_prob_list = (
            [result.calib_probs for result in model_results] if calib_split else None
        )

        print("[LateFusion] Fitting fusion head...")
        fusion = DecisionLevelFusion(label_list, self.config.get("fusion", {}))
        fusion.fit(val_prob_list, val_split.labels)

        fusion_val_probs = fusion.predict_proba(val_prob_list)
        fusion_test_probs = fusion.predict_proba(test_prob_list)
        fusion_val_preds = [label_list[idx] for idx in np.argmax(fusion_val_probs, axis=1)]
        fusion_test_preds = [label_list[idx] for idx in np.argmax(fusion_test_probs, axis=1)]
        if calib_split and calib_prob_list:
            fusion_calib_probs = fusion.predict_proba(calib_prob_list)
            best_temp = self._optimize_temperature(fusion_calib_probs, calib_split.labels, label_list)
            fusion_val_probs = self._apply_temperature(fusion_val_probs, best_temp)
            fusion_test_probs = self._apply_temperature(fusion_test_probs, best_temp)
            self.calibration_result = {
                "temperature": best_temp,
                "calibration_size": len(calib_split.labels),
                "calibration_fraction": calib_fraction,
            }
            print(
                f"[LateFusion] Calibration applied -> temperature={best_temp:.3f}, "
                f"calib_size={len(calib_split.labels)}"
            )
        else:
            self.calibration_result = None

        fusion_val_preds = [label_list[idx] for idx in np.argmax(fusion_val_probs, axis=1)]
        fusion_test_preds = [label_list[idx] for idx in np.argmax(fusion_test_probs, axis=1)]
        fusion_metrics = {
            "validation": compute_classification_metrics(val_split.labels, fusion_val_preds, label_list),
            "test": compute_classification_metrics(test_split.labels, fusion_test_preds, label_list),
        }
        self._log_metric_summary("fusion", fusion_metrics, label_list)

        print("[LateFusion] Writing metrics and predictions...")
        self._write_artifacts(
            per_model_metrics,
            fusion_metrics,
            test_split,
            fusion_test_preds,
            fusion_test_probs,
            label_list,
            dataset_cfg.get("label_column"),
            [m.get("name", f"model_{idx}") for idx, m in enumerate(self.config.get("base_models", []))],
            (dataset_cfg.get("resampling", {}) or {}).get("type"),
            dataset_cfg.get("random_seed"),
            self._tfidf_replicas(),
            self.config,
        )
        print(f"[LateFusion] Completed run. Artifacts saved to {self.output_dir.resolve()}.")
        return {"base_models": per_model_metrics, "fusion": fusion_metrics}

    # ------------------------------------------------------------------ #
    def _write_artifacts(
        self,
        base_model_metrics: Dict,
        fusion_metrics: Dict,
        test_split,
        predictions: List[str],
        probabilities: np.ndarray,
        label_list: List[str],
        label_column: Optional[str],
        model_names: List[str],
        resampling_type: Optional[str],
        random_seed: Optional[int],
        tfidf_replicas: Optional[int],
        config_snapshot: Dict,
    ) -> None:
        experiment_idx = self._next_experiment_index()
        date_tag = datetime.now().strftime("%Y%m%d")
        suffix = self._build_run_suffix(
            label_column, model_names, resampling_type, random_seed, tfidf_replicas
        )

        metrics_name = f"metrics_experiment_{experiment_idx}_{suffix}_{date_tag}.json"
        preds_name = f"prediction_test_experiment_{experiment_idx}_{suffix}_{date_tag}.json"
        config_name = f"config_experiment_{experiment_idx}_{suffix}_{date_tag}.json"

        metrics_payload = {
            "base_models": base_model_metrics,
            "fusion": fusion_metrics,
        }
        if self.paper_reference:
            metrics_payload["paper_reference"] = self.paper_reference
        if self.calibration_result:
            metrics_payload["calibration"] = self.calibration_result
        metrics_path = self.output_dir / metrics_name
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics_payload, fp, indent=2)
        config_path = self.output_dir / config_name
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(config_snapshot, fp, indent=2)

        records = []
        for idx, sample_id in enumerate(test_split.ids):
            prob_dict = {
                label: float(probabilities[idx, label_idx])
                for label_idx, label in enumerate(label_list)
            }
            records.append({"id": sample_id, "prediction": predictions[idx], "probabilities": prob_dict})

        pred_path = self.output_dir / preds_name
        with open(pred_path, "w", encoding="utf-8") as fp:
            json.dump(records, fp, indent=2)

    def _next_experiment_index(self) -> int:
        max_idx = 0
        for path in self.output_dir.glob("metrics_experiment_*"):
            parts = path.stem.split("_")
            if len(parts) >= 3 and parts[0] == "metrics" and parts[1] == "experiment":
                try:
                    idx = int(parts[2])
                except ValueError:
                    continue
                max_idx = max(max_idx, idx)
        return max_idx + 1

    def _split_for_calibration(
        self, split: DataSplit, fraction: float, seed: Optional[int]
    ) -> (DataSplit, Optional[DataSplit]):
        if not fraction or fraction <= 0.0:
            return split, None
        n = len(split.labels)
        calib_size = int(round(n * fraction))
        if calib_size == 0 or calib_size >= n:
            return split, None
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        calib_idx = indices[:calib_size]
        remain_idx = indices[calib_size:]
        return (
            self._subset_split(split, remain_idx),
            self._subset_split(split, calib_idx),
        )

    def _subset_split(self, split: DataSplit, indices: np.ndarray) -> DataSplit:
        return DataSplit(
            ids=[split.ids[i] for i in indices],
            texts=[split.texts[i] for i in indices],
            labels=[split.labels[i] for i in indices],
        )

    def _resample_train_split(
        self,
        split: DataSplit,
        resampling_cfg: Dict,
        seed: int,
    ) -> DataSplit:
        if not resampling_cfg:
            return split

        resampling_type = (resampling_cfg.get("type") or "").lower()
        if resampling_type in ("", "none", None):
            return split

        sampler = None
        if resampling_type in ("over", "oversample", "oversampling"):
            sampler = RandomOverSampler(random_state=resampling_cfg.get("random_seed", seed))
        elif resampling_type in ("under", "undersample", "undersampling"):
            sampler = RandomUnderSampler(random_state=resampling_cfg.get("random_seed", seed))
        else:
            raise ValueError(f"Unsupported resampling type: {resampling_type}")

        n = len(split.labels)
        if n == 0:
            return split
        print(f"[Resampling] Strategy={resampling_type} on {n} train rows")
        before_counts = self._label_counts(split.labels)
        x_placeholder = np.arange(n).reshape(-1, 1)
        y = np.array(split.labels)
        resampled_indices, resampled_labels = sampler.fit_resample(x_placeholder, y)
        idx = resampled_indices.ravel()
        after_counts = self._label_counts(resampled_labels)
        added_counts = {
            label: after_counts.get(label, 0) - before_counts.get(label, 0)
            for label in after_counts
        }
        print(
            f"[Resampling] Label counts before={before_counts} "
            f"after={after_counts} added={added_counts}"
        )
        return DataSplit(
            ids=[split.ids[i] for i in idx],
            texts=[split.texts[i] for i in idx],
            labels=list(resampled_labels),
        )

    def _train_with_replicas(
        self,
        model_cfg: Dict,
        label_list: List[str],
        train_split: DataSplit,
        val_split: DataSplit,
        test_split: DataSplit,
        calib_split: Optional[DataSplit],
        seed: int,
    ):
        replicas = int(model_cfg.get("replicas", 1) or 1)
        if replicas <= 1:
            model = create_model(model_cfg, label_list)
            model.fit(train_split.texts, train_split.labels)
            val_probs = self._align_probs(
                model.predict_proba(val_split.texts), getattr(model, "classifier", None), label_list
            )
            test_probs = self._align_probs(
                model.predict_proba(test_split.texts), getattr(model, "classifier", None), label_list
            )
            calib_probs = (
                self._align_probs(
                    model.predict_proba(calib_split.texts),
                    getattr(model, "classifier", None),
                    label_list,
                )
                if calib_split
                else None
            )
            val_preds = model.predict(val_split.texts)
            test_preds = model.predict(test_split.texts)
            return val_probs, test_probs, calib_probs, val_preds, test_preds

        sampling = model_cfg.get("replica_sampling", "partition")
        weights = model_cfg.get("replica_weights")
        indices_per_replica = self._build_replica_indices(
            len(train_split.labels), replicas, sampling, seed
        )
        effective_replicas = len(indices_per_replica)
        print(
            f"[Replicas] {model_cfg.get('name', 'model')} -> replicas={effective_replicas} "
            f"sampling={sampling}"
        )
        if weights is None:
            weights_array = np.ones(effective_replicas, dtype=float)
        else:
            weights_array = np.array(weights, dtype=float)
            if weights_array.shape[0] != effective_replicas:
                raise ValueError(
                    f"replica_weights length ({weights_array.shape[0]}) does not match replicas ({effective_replicas})."
                )
        weights_array = weights_array / weights_array.sum()

        val_prob_list = []
        test_prob_list = []
        calib_prob_list = [] if calib_split else None

        for rep_idx, indices in enumerate(indices_per_replica):
            subset = self._subset_split(train_split, indices)
            print(
                f"[Replicas] Training replica {rep_idx+1}/{effective_replicas} "
                f"on {len(subset.labels)} samples"
            )
            model = create_model(model_cfg, label_list)
            model.fit(subset.texts, subset.labels)
            val_prob_list.append(
                self._align_probs(
                    model.predict_proba(val_split.texts), getattr(model, "classifier", None), label_list
                )
            )
            test_prob_list.append(
                self._align_probs(
                    model.predict_proba(test_split.texts), getattr(model, "classifier", None), label_list
                )
            )
            if calib_split and calib_prob_list is not None:
                calib_prob_list.append(
                    self._align_probs(
                        model.predict_proba(calib_split.texts),
                        getattr(model, "classifier", None),
                        label_list,
                    )
                )

        val_probs = self._weighted_average_probs(val_prob_list, weights_array)
        test_probs = self._weighted_average_probs(test_prob_list, weights_array)
        calib_probs = (
            self._weighted_average_probs(calib_prob_list, weights_array)
            if calib_split and calib_prob_list
            else None
        )
        val_preds = [label_list[idx] for idx in np.argmax(val_probs, axis=1)]
        test_preds = [label_list[idx] for idx in np.argmax(test_probs, axis=1)]
        return val_probs, test_probs, calib_probs, val_preds, test_preds

    def _build_replica_indices(
        self, n_samples: int, replicas: int, sampling: str, seed: int
    ) -> List[np.ndarray]:
        replicas = max(1, min(replicas, n_samples)) if sampling == "partition" else max(1, replicas)
        rng = np.random.default_rng(seed)
        if sampling == "partition":
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            splits = np.array_split(indices, replicas)
            kept = [split for split in splits if len(split) > 0]
            return kept

        # bootstrap: sample (approx) n/replicas per replica with replacement
        base_size = max(1, int(np.ceil(n_samples / replicas)))
        return [rng.choice(n_samples, size=base_size, replace=True) for _ in range(replicas)]

    def _weighted_average_probs(self, prob_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        stacked = np.stack(prob_list, axis=0)  # (m, n, c)
        return np.tensordot(weights, stacked, axes=(0, 0))

    def _align_probs(
        self, probs: np.ndarray, classifier, label_list: List[str]
    ) -> np.ndarray:
        """
        Aligns model probabilities to the full label_list.
        If a replica was trained on a subset of classes, insert zero columns for missing labels
        and renormalize rows to sum to 1 when possible.
        """
        num_labels = len(label_list)
        classes = None
        if classifier is not None and hasattr(classifier, "classes_"):
            classes = classifier.classes_
        if classes is None or len(classes) == num_labels:
            return probs

        # Map classifier classes to indices in the full label_list.
        label_to_id = {lbl: idx for idx, lbl in enumerate(label_list)}
        aligned = np.zeros((probs.shape[0], num_labels), dtype=probs.dtype)
        for src_idx, cls in enumerate(classes):
            dest_idx = None
            if isinstance(cls, (int, np.integer)) and 0 <= cls < num_labels:
                dest_idx = int(cls)
            elif cls in label_to_id:
                dest_idx = label_to_id[cls]
            if dest_idx is not None:
                aligned[:, dest_idx] = probs[:, src_idx]

        row_sums = aligned.sum(axis=1)
        valid = row_sums > 0
        aligned[valid] /= row_sums[valid, None]
        return aligned

    def _log_metric_summary(self, name: str, metrics: Dict, label_list: List[str]) -> None:
        def short_row(split_name: str):
            m = metrics[split_name]
            return (
                f"acc={m['accuracy']:.3f} "
                f"macro_f1={m['macro_f1']:.3f} "
                f"weighted_f1={m['weighted_f1']:.3f}"
            )

        print(f"[Metrics] {name} -> val {short_row('validation')} | test {short_row('test')}")
        for split_name in ("validation", "test"):
            per_class = metrics[split_name].get("per_class", {})
            if not per_class:
                continue
            print(f"[Metrics] {name} per-class ({split_name}):")
            for label in label_list:
                if label not in per_class:
                    continue
                pc = per_class[label]
                print(
                    f"  {label:15s} "
                    f"P={pc['precision']:.3f} "
                    f"R={pc['recall']:.3f} "
                    f"F1={pc['f1']:.3f} "
                    f"N={pc['support']}"
                )

    @staticmethod
    def _label_counts(labels: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def _optimize_temperature(self, probs: np.ndarray, labels: List[str], label_list: List[str]) -> float:
        if probs.size == 0:
            return 1.0
        y = np.array([label_list.index(lbl) for lbl in labels])
        best_T = 1.0
        best_nll = float("inf")
        temperatures = np.linspace(0.5, 3.0, 51)
        for temp in temperatures:
            scaled = self._apply_temperature(probs, temp)
            nll = -np.mean(
                np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, 1.0))
            )
            if nll < best_nll:
                best_nll = nll
                best_T = temp
        return best_T

    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        eps = 1e-12
        log_probs = np.log(np.clip(probs, eps, 1.0))
        scaled = log_probs / max(temperature, eps)
        scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        scaled /= np.sum(scaled, axis=1, keepdims=True)
        return scaled

    def _build_run_suffix(
        self,
        label_column: Optional[str],
        model_names: List[str],
        resampling_type: Optional[str],
        random_seed: Optional[int],
        tfidf_replicas: Optional[int],
    ) -> str:
        label_tag = self._sanitize_tag(label_column or "labels")
        model_tag = "+".join(self._sanitize_tag(name) for name in model_names) or "models"
        resampling_tag = self._sanitize_tag(resampling_type or "none")
        seed_tag = self._sanitize_tag(str(random_seed) if random_seed is not None else "seed")
        tfidf_tag = (
            self._sanitize_tag(str(tfidf_replicas))
            if tfidf_replicas is not None
            else "tfidf"
        )
        return (
            f"label-{label_tag}_models-{model_tag}_res-{resampling_tag}_"
            f"seed-{seed_tag}_tfidf-rep-{tfidf_tag}"
        )

    def _sanitize_tag(self, value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "+-" else "-" for ch in value)
        cleaned = cleaned.strip("-")
        return cleaned or "na"

    def _tfidf_replicas(self) -> Optional[int]:
        for model_cfg in self.config.get("base_models", []):
            if model_cfg.get("name") == "tfidf_baseline":
                return int(model_cfg.get("replicas", 1) or 1)
        return None
