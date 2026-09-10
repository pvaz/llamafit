# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Four scores, the weights that combine them, and the ranking that comes out.

A recommendation is one ordered list, and this package is what orders it. Each part
answers a different question about the same candidate:

- :mod:`~llamafit.scoring.fit_score` — does it use the machine well, at either end?
- :mod:`~llamafit.scoring.speed_score` — is it fast enough for the job asked of it?
- :mod:`~llamafit.scoring.context_score` — can it hold the context the job needs?
- :mod:`llamafit.quality` — is it good enough to be worth running at all?

:mod:`~llamafit.scoring.weights` combines them, per use case, with numbers a user can
replace from the config file, and :mod:`~llamafit.scoring.rank` produces the board —
including the candidates that were excluded, each carrying the reason it was.
"""

from __future__ import annotations

from llamafit.scoring.context_score import DEFAULT_CONTEXT, context_score, requested_context
from llamafit.scoring.fit_score import fit_score, fit_score_for, worst_pool_utilisation
from llamafit.scoring.rank import evaluate, evaluate_and_rank, rank
from llamafit.scoring.speed_score import TARGET_TPS, prompt_penalty, speed_score, target_tps
from llamafit.scoring.weights import (
    DEFAULT_WEIGHTS,
    PARTS,
    PREFERENCE_SHIFT,
    PREFERENCES,
    USE_CASES,
    shift_preference,
    weights_for,
)

__all__ = [
    "DEFAULT_CONTEXT",
    "DEFAULT_WEIGHTS",
    "PARTS",
    "PREFERENCES",
    "PREFERENCE_SHIFT",
    "TARGET_TPS",
    "USE_CASES",
    "context_score",
    "evaluate",
    "evaluate_and_rank",
    "fit_score",
    "fit_score_for",
    "prompt_penalty",
    "rank",
    "requested_context",
    "shift_preference",
    "speed_score",
    "target_tps",
    "weights_for",
    "worst_pool_utilisation",
]
