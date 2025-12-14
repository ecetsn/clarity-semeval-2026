import argparse
import yaml

from src.training.train_utils import run_fusion_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run the late fusion pipeline.")
    parser.add_argument("--config", default="config/ensemble.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    run_fusion_training(config)
