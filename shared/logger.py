"""logger.py — minimal logging setup.

Reconstructed minimal version used by run_exp101_reproducible.py:
    setup_logger(name, path)
where `path` may be a directory (a `<name>.log` file is created inside it)
or a full file path.
"""
import logging
import os


def setup_logger(name, log_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    if log_path:
        # If the path looks like a directory, place a <name>.log inside it.
        if os.path.isdir(log_path) or not log_path.endswith(".log"):
            os.makedirs(log_path, exist_ok=True)
            file_path = os.path.join(log_path, f"{name}.log")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            file_path = log_path
        fh = logging.FileHandler(file_path, encoding="utf-8")
    else:
        fh = logging.StreamHandler()

    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
    return logger
