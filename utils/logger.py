"""Logging setup."""

import logging
import os
import sys


def setup_logger(name: str = "FEDEATA", level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def add_file_handler(logger: logging.Logger, save_path: str, filename: str = "logfile.log") -> logging.Logger:
    """Add a file handler to an existing logger."""
    file_handler = logging.FileHandler(os.path.join(save_path, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()
