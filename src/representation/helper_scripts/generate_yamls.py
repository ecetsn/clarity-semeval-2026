# scripts/generate_baseline_pooling_yamls.py
# Generates 24 YAMLs by sweeping:
#   encoder.pooling ∈ {masked_mean, max, cls, last_non_pad}   (4)
#   decoder.pooling ∈ {masked_mean, last_non_pad, max}        (3)
#   experiment.task ∈ {clarity, evasion}                      (2)
#
# Names follow:
#   weightedclasses_<task>_<encpool>_<decpool>
#
# Logging/outputs sections are preserved from the template YAML.

from __future__ import annotations

import copy
from pathlib import Path
import yaml


ENCODER_POOLINGS = ["masked_mean", "max", "cls", "last_non_pad"]
DECODER_POOLINGS = ["masked_mean", "last_non_pad", "max"]
TASKS = ["clarity", "evasion"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Template YAML must be a mapping/dict: {path}")
    return obj


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Template should match your example structure (e.g., reproduced_baseline.yaml)
    template_path = repo_root / "configs" / "reproduced_baseline.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    out_dir = repo_root / "configs" / "generated" / "baseline_pooling"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(template_path)

    written = 0
    for task in TASKS:
        for enc_pool in ENCODER_POOLINGS:
            for dec_pool in DECODER_POOLINGS:
                cfg = copy.deepcopy(base_cfg)

                exp_name = f"weightedclasses_{task}_{enc_pool}_{dec_pool}"

                # required edits
                cfg.setdefault("experiment", {})
                cfg["experiment"]["name"] = exp_name
                cfg["experiment"]["task"] = task

                cfg.setdefault("encoder", {})
                cfg["encoder"]["pooling"] = enc_pool

                cfg.setdefault("decoder", {})
                cfg["decoder"]["pooling"] = dec_pool

                # keep weighted classes on (as you stated you will use for remainder)
                cfg.setdefault("loss", {})
                cfg["loss"]["class_weighted"] = True

                out_path = out_dir / f"{exp_name}.yaml"
                dump_yaml(cfg, out_path)
                written += 1

    print(f"Wrote {written} YAML files to: {out_dir}")


if __name__ == "__main__":
    main()
