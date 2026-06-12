"""Tiny logging helper so every script logs consistently."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "franka_d435_fp", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring a simple stream handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
