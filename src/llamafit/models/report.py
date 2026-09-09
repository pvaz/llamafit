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
