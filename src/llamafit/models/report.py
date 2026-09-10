# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Top-level result of ``llamafit system``."""

from __future__ import annotations

from pydantic import BaseModel

from llamafit.models.host import Host
from llamafit.models.llamacpp import LlamaCpp


class SystemReport(BaseModel):
    """Host plus llama.cpp status, stamped with the LlamaFit version that produced it."""

    host: Host
    llamacpp: LlamaCpp
    version: str
