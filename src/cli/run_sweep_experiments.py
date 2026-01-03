from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.training.train_utils import run_fusion_training

# Modest defaults to keep the sweep tractable.
DEFAULT_TFIDF_GRID: List[Dict] = [
    {"ngram_range": (1, 2), "min_df": 2, "max_features": 20000, "c": 2.0},
    {"ngram_range": (1, 2), "min_df": 3, "max_features": 30000, "c": 3.0},
    {"ngram_range": (1, 3), "min_df": 3, "max_features": 30000, "c": 3.0},
    {"ngram_range": (1, 3), "min_df": 5, "max_features": 40000, "c": 4.0},
]
DEFAULT_RESAMPLING: List[Optional[str]] = ["over", "under", None]
DEFAULT_FUSION_C: List[float] = [1.0, 2.0, 4.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-sweep runner for TF-IDF baselines + fusion."
    )
    parser.add_argument(
        "--config",
        default="config/ensemble.yaml",
        help="Base YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/sweeps",
        help="Directory to write sweep artifacts.",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=5,
        help="Replica count for the TF-IDF baseline during sweeps (reduces runtime).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of sweep runs.",
    )
    return parser.parse_args()


def load_base_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def find_tfidf_index(cfg: Dict) -> int:
    for idx, model in enumerate(cfg.get("base_models", [])):
        if model.get("type") == "tfidf_logreg":
            return idx
    raise ValueError("No tfidf_logreg model found in base_models.")


def apply_tfidf_params(cfg: Dict, tfidf_idx: int, params: Dict, replicas: int) -> None:
    model_cfg = cfg["base_models"][tfidf_idx]
    model_cfg["replicas"] = replicas
    model_cfg.setdefault("params", {})
    model_cfg["params"]["ngram_range"] = list(params["ngram_range"])
    model_cfg["params"]["min_df"] = params["min_df"]
    model_cfg["params"]["max_features"] = params["max_features"]
    model_cfg["params"]["c"] = params["c"]


def apply_resampling(cfg: Dict, strategy: Optional[str]) -> None:
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("resampling", {})
    cfg["dataset"]["resampling"]["type"] = strategy or "none"


def apply_fusion_c(cfg: Dict, c_value: float) -> None:
    cfg.setdefault("fusion", {})
    cfg["fusion"].setdefault("params", {})
    cfg["fusion"]["params"]["C"] = c_value


def format_run_name(tfidf_params: Dict, resample: Optional[str], fusion_c: float) -> str:
    ngram = tfidf_params["ngram_range"]
    parts = [
        f"ng{ngram[0]}-{ngram[1]}",
        f"df{tfidf_params['min_df']}",
        f"mf{tfidf_params['max_features']}",
        f"c{tfidf_params['c']}",
        f"res-{resample or 'none'}",
        f"fC{fusion_c}",
    ]
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    base_cfg = load_base_config(Path(args.config))
    tfidf_idx = find_tfidf_index(base_cfg)

    combos = itertools.product(DEFAULT_TFIDF_GRID, DEFAULT_RESAMPLING, DEFAULT_FUSION_C)
    if args.max_runs:
        combos = itertools.islice(combos, args.max_runs)

    for run_id, (tfidf_params, resample_strategy, fusion_c) in enumerate(combos, start=1):
        cfg = copy.deepcopy(base_cfg)
        cfg["output_dir"] = args.output_dir
        apply_tfidf_params(cfg, tfidf_idx, tfidf_params, replicas=args.replicas)
        apply_resampling(cfg, resample_strategy)
        apply_fusion_c(cfg, fusion_c)

        run_tag = format_run_name(tfidf_params, resample_strategy, fusion_c)
        print(f"\n[Sweep] Run {run_id}: {run_tag}")
        run_fusion_training(cfg)


if __name__ == "__main__":
    main()
