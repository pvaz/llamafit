"""Read a GGUF file's header and derive architecture facts from it, without downloading it."""

from __future__ import annotations

from llamafit.gguf.cache import read_facts
from llamafit.gguf.facts import bits_per_weight
from llamafit.gguf.reader import merge_shard_headers, read_header

__all__ = ["bits_per_weight", "merge_shard_headers", "read_facts", "read_header"]
