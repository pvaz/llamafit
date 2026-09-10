# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Read a GGUF file's header and derive architecture facts from it, without downloading it."""

from __future__ import annotations

from llamafit.gguf.cache import read_facts
from llamafit.gguf.facts import bits_per_weight
from llamafit.gguf.reader import merge_shard_headers, read_header

__all__ = ["bits_per_weight", "merge_shard_headers", "read_facts", "read_header"]
