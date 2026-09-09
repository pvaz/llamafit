"""Read a GGUF file's header and derive architecture facts from it, without downloading it."""

from __future__ import annotations

from llamafit.gguf.cache import read_facts
from llamafit.gguf.reader import read_header

__all__ = ["read_facts", "read_header"]
