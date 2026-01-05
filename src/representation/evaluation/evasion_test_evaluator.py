# src/representation/evaluation/evasion_test_evaluator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.representation.data.qev_datamodule import DataSplit
from src.representation.evaluation.evaluator import evaluate_predictions


def evaluate_evasion_on_official_test(
    *,
    y_pred_labels: List[str],
    test_split: DataSplit,
    label_list: List[str],
) -> Dict[str, Any]:
    """

    Computes:
      - annotator-wise classification metrics: A1 vs pred, A2 vs pred, A3 vs pred
        (using your existing evaluate_predictions)
      - any-match accuracy: pred in {annotator1, annotator2, annotator3}

    Notes:
      - Some annotator fields may be None or "" => ignored in any-match set.
      - For annotator-wise metrics, rows with missing annotator label are filtered out.

    y_pred_labels: list of predicted LABEL STRINGS, length == len(test_split.texts)
    label_list: stable ordered list of class names (e.g., ["Deflection","Dodging",...])
    """
    n = len(test_split.texts)
    if len(y_pred_labels) != n:
        raise ValueError(f"y_pred_labels length mismatch: {len(y_pred_labels)} vs test size {n}")

    a1 = test_split.annotator1
    a2 = test_split.annotator2
    a3 = test_split.annotator3

    def _filter_pairs(y_true: List[Optional[str]], y_pred: List[str]):
        yt: List[str] = []
        yp: List[str] = []
        for t, p in zip(y_true, y_pred):
            if t is None or str(t).strip() == "":
                continue
            yt.append(str(t).strip())
            yp.append(str(p).strip())
        return yt, yp

    # Annotator-wise metrics
    a1_true, a1_pred = _filter_pairs(a1, y_pred_labels)
    a2_true, a2_pred = _filter_pairs(a2, y_pred_labels)
    a3_true, a3_pred = _filter_pairs(a3, y_pred_labels)

    annotator_results: Dict[str, Any] = {}
    if len(a1_true) > 0:
        annotator_results["annotator1"] = evaluate_predictions(y_true=a1_true, y_pred=a1_pred, label_list=label_list)
    else:
        annotator_results["annotator1"] = {"error": "No usable annotator1 labels found."}

    if len(a2_true) > 0:
        annotator_results["annotator2"] = evaluate_predictions(y_true=a2_true, y_pred=a2_pred, label_list=label_list)
    else:
        annotator_results["annotator2"] = {"error": "No usable annotator2 labels found."}

    if len(a3_true) > 0:
        annotator_results["annotator3"] = evaluate_predictions(y_true=a3_true, y_pred=a3_pred, label_list=label_list)
    else:
        annotator_results["annotator3"] = {"error": "No usable annotator3 labels found."}

    # Any-match accuracy
    gold_sets: List[Set[str]] = []
    for t1, t2, t3 in zip(a1, a2, a3):
        s: Set[str] = set()
        for v in (t1, t2, t3):
            if v is None:
                continue
            vv = str(v).strip()
            if vv:
                s.add(vv)
        gold_sets.append(s)

    hits = 0
    valid = 0
    for pred, gold in zip(y_pred_labels, gold_sets):
        if not gold:
            # no gold labels available for this row; skip from any-match
            continue
        valid += 1
        if str(pred).strip() in gold:
            hits += 1

    any_match_acc = (hits / valid) if valid > 0 else 0.0

    return {
        "any_match": {
            "accuracy": any_match_acc,
            "hits": hits,
            "valid": valid,
            "skipped_no_gold": n - valid,
        },
        "annotator_wise": annotator_results,
    }
