# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Speed estimation: what a placement will actually run at, and how much to trust it.

:func:`estimate_speed` is the entry point. Everything else is here so an explanation can
be built from the parts rather than from the answer: :func:`per_token_traffic` says which
bytes a token reads and how they are reached, :func:`resolve_bandwidths` says what this
machine's pools achieve, and :func:`formula_estimate` is section 10's arithmetic with no
measurement allowed to correct it.
"""

from llamafit.speed.bandwidths import EffectiveBandwidths, resolve_bandwidths
from llamafit.speed.estimate import Formula, estimate_speed, formula_estimate
from llamafit.speed.traffic import (
    TokenTraffic,
    TrafficLine,
    active_expert_bytes,
    expert_bytes_in_ram,
    kv_bytes_per_token,
    per_token_traffic,
    streamed_expert_fraction,
)

__all__ = [
    "EffectiveBandwidths",
    "Formula",
    "TokenTraffic",
    "TrafficLine",
    "active_expert_bytes",
    "estimate_speed",
    "expert_bytes_in_ram",
    "formula_estimate",
    "kv_bytes_per_token",
    "per_token_traffic",
    "resolve_bandwidths",
    "streamed_expert_fraction",
]
