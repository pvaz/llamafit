# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What a model needs to run: bytes by component, pool by pool, and whether it fits.

``compute`` is the whole package from outside. Everything else here is one responsibility
of it: :mod:`~llamafit.budget.weights` places the tensors, :mod:`~llamafit.budget.kv`
sizes what the model remembers, :mod:`~llamafit.budget.compute_buffer` models the two
buffers nobody can read off a header, and :mod:`~llamafit.budget.projector` prices vision.
"""

from llamafit.budget.budget import compute, pool_verdict, utilisation
from llamafit.budget.compute_buffer import batch_for, compute_buffer_bytes, output_buffer_bytes
from llamafit.budget.kv import (
    KV_TYPES,
    derived_kv_cache_bytes,
    kv_cache_bytes,
    unaccounted_kv_cache_bytes,
)

__all__ = [
    "KV_TYPES",
    "batch_for",
    "compute",
    "compute_buffer_bytes",
    "derived_kv_cache_bytes",
    "kv_cache_bytes",
    "output_buffer_bytes",
    "pool_verdict",
    "unaccounted_kv_cache_bytes",
    "utilisation",
]
