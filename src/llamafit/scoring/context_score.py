# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""How much of the context a request wants a candidate can actually hold.

``context_score = 100 x min(1, max_context_fit / requested_context)``

The score is one-sided on purpose, and it is the only one of the four that is. There is no
penalty for holding more context than was asked for: unused context costs nothing at run
time because llama.cpp allocates the cache for the context it is started with, and the
placement planner already charges the memory that context needs to the fit score. A model
that can hold four times what was asked for is not wasteful, it is simply finished being
scored on this axis.

Falling short, on the other hand, is a real loss and is scored in proportion: half the
requested context is half the score, because a model that holds half a repository does
half the job.

The defaults, from section 11.2 of the design specification, are what a request wants when
it does not say: 8K for general and chat, which covers a conversation; 32K for coding and
reasoning, which is a few real source files or a long chain of thought; 16K for multimodal,
where an image is worth hundreds of tokens before any text arrives.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from llamafit.models.plan import Needs
from llamafit.scoring.weights import check_use_case

DEFAULT_CONTEXT: Mapping[str, int] = MappingProxyType(
    {
        "general": 8192,
        "chat": 8192,
        "coding": 32768,
        "reasoning": 32768,
        "multimodal": 16384,
        "embedding": 8192,
    }
)
"""What each use case asks for when the request does not say.

*embedding* is not in the specification's list, which names the other five. It takes the
general default: an embedding request is one document at a time, and the models that serve
it rarely offer more anyway.
"""


def requested_context(needs: Needs) -> int:
    """Return the context this request wants, explicit or by use case.

    Args:
        needs: What the user asked for.

    Returns:
        ``needs.requested_context`` when it was given, and the use case's default
        otherwise, in either case never above ``needs.max_context``.

    Raises:
        ConfigError: If the use case is not one of the six.

    The ceiling applies here and not only to the planner. This figure is the score's
    denominator, and a request that will not run past 16,384 tokens scored against the
    32,768 its use case would have wanted would mark every candidate down by half for
    failing to reach a length the same request forbade.
    """
    wanted = (
        needs.requested_context
        if needs.requested_context is not None
        else DEFAULT_CONTEXT[check_use_case(needs.use_case)]
    )
    return wanted if needs.max_context is None else min(wanted, needs.max_context)


def context_score(max_context_fit: int, requested: int) -> float:
    """Score how much of the requested context a candidate can hold, from 0 to 100.

    Args:
        max_context_fit: The largest context this placement fits, in tokens.
        requested: The context the request wants, in tokens. Above zero; a request for
            no context is not a request.

    Returns:
        100 when the candidate holds everything asked for or more, and the fraction it
        does hold otherwise.

    Raises:
        ValueError: If ``requested`` is not above zero.
    """
    if requested <= 0:
        raise ValueError(f"requested context must be above zero: {requested!r}")
    return 100.0 * min(1.0, max(0, max_context_fit) / requested)
