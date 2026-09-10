# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""LlamaFit: find, size, install and verify open-weight LLMs for llama.cpp."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llamafit")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
