# src/representation/logging/logger_setup.py

import logging
import sys
from datetime import datetime
from pathlib import Path


def init_logger(
    *,
    script_name: str,
    run_group: str,
    base_dir: Path,
    level: int = logging.INFO,
):
    """
    Initialize a unified experiment logger.

    Logs are stored as:
    <base_dir>/<run_group>/<script_name>/<timestamp>.log
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = base_dir / run_group / script_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("Logger initialized")
    logger.info(f"Script      : {script_name}")
    logger.info(f"Run group   : {run_group}")
    logger.info(f"Log path   : {log_path}")
    logger.info("=" * 80)

    return logger, log_path
