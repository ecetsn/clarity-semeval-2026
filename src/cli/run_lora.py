import yaml, sys
from src.training.train_utils import run_lora_training

if __name__ == "__main__":
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else "config/task1_clarity.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_lora_training(cfg)
