from __future__ import annotations

from typing import Dict
import warnings

from src.evaluation.late_fusion import LateFusionExperiment

# Suppress noisy sklearn FutureWarnings about multi_class deprecation.
warnings.filterwarnings(
    "ignore",
    message=".*'multi_class' was deprecated.*",
    category=FutureWarning,
)


def run_fusion_training(cfg: Dict) -> Dict:
    """Executes the late-fusion training pipeline defined in the config file."""
    experiment = LateFusionExperiment(cfg)
    return experiment.run()


def run_late_fusion_pipeline(cfg: Dict) -> Dict:
    """Backward-compatible alias for run_fusion_training."""
    return run_fusion_training(cfg)
