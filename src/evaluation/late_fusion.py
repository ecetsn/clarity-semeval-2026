from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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

        zero_shot_used = any(
            model_cfg.get("type") == "zero_shot" for model_cfg in self.config.get("base_models", [])
        )

        model_results = []
        per_model_metrics = {}
        for idx, model_cfg in enumerate(self.config.get("base_models", [])):
            name = model_cfg.get("name", f"model_{idx}")
            print(f"[LateFusion] Training base model '{name}' ({model_cfg['type']})...")
            model = create_model(model_cfg, label_list)
            model.fit(train_split.texts, train_split.labels)
            val_probs = model.predict_proba(val_split.texts)
            test_probs = model.predict_proba(test_split.texts)
            calib_probs = model.predict_proba(calib_split.texts) if calib_split else None
            model_results.append(
                BaseModelResult(
                    name=name, val_probs=val_probs, test_probs=test_probs, calib_probs=calib_probs
                )
            )

            val_preds = model.predict(val_split.texts)
            test_preds = model.predict(test_split.texts)
            per_model_metrics[name] = {
                "validation": compute_classification_metrics(val_split.labels, val_preds, label_list),
                "test": compute_classification_metrics(test_split.labels, test_preds, label_list),
            }
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
        else:
            self.calibration_result = None

        fusion_val_preds = [label_list[idx] for idx in np.argmax(fusion_val_probs, axis=1)]
        fusion_test_preds = [label_list[idx] for idx in np.argmax(fusion_test_probs, axis=1)]
        fusion_metrics = {
            "validation": compute_classification_metrics(val_split.labels, fusion_val_preds, label_list),
            "test": compute_classification_metrics(test_split.labels, fusion_test_preds, label_list),
        }

        print("[LateFusion] Writing metrics and predictions...")
        self._write_artifacts(
            per_model_metrics,
            fusion_metrics,
            test_split,
            fusion_test_preds,
            fusion_test_probs,
            label_list,
            zero_shot_used,
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
        zero_shot_used: bool,
    ) -> None:
        experiment_idx = self._next_experiment_index()
        date_tag = datetime.now().strftime("%Y%m%d")
        run_tag = "zero-shot" if zero_shot_used else "base"

        metrics_name = f"metrics_experiment_{experiment_idx}_{run_tag}_{date_tag}.json"
        preds_name = f"prediction_test_experiment_{experiment_idx}_{run_tag}_{date_tag}.json"

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
