"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("documind")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
