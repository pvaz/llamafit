# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""File logging. The console stays quiet; details go to ``llamafit.log``."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "llamafit"


def setup_logging(log_dir: Path, *, verbose: bool = False) -> logging.Logger:
    """Configure the ``llamafit`` logger to write to ``log_dir/llamafit.log``.

    Returns the logger. Safe to call more than once; handlers are not duplicated.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "llamafit.log", maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, for example ``get_logger("hardware.gpu")``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
