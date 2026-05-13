"""
logger.py — Unified logging for all experiments.

Log format: [timestamp] [exp_name] [seed] [method] [target] metric values
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(exp_name, log_dir="logs"):
    """
    Create a logger that writes to both console and file.

    Args:
        exp_name: str, experiment identifier (e.g., 'exp11_lodo')
        log_dir: str, directory for log files

    Returns:
        logging.Logger: configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{exp_name}_{timestamp}.log"

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        logger.handlers.clear()

    # Console handler — INFO level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter(
        f"[%(asctime)s] [{exp_name}] %(message)s",
        datefmt="%H:%M:%S"
    )
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    # File handler — DEBUG level (captures everything)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter(
        f"[%(asctime)s] [{exp_name}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger
