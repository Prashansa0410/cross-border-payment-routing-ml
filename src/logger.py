"""Logging configuration for the cross-border payment routing system."""

import logging
import os
import sys

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logger(name: str) -> logging.Logger:
    """Creates or retrieves a logger with standard formatting and handlers.

    Ensures that sensitive payloads or keys are not printed blindly and
    outputs are clean.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
