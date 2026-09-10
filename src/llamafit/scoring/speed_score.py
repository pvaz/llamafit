"""How the estimated speed compares with what this use case actually needs.

``speed_score = 100 x min(1, tokens_per_second / target)``

The score is capped at the target rather than rewarded above it, and that is the whole
idea: past the point where a person stops waiting, more speed buys nothing. Reading is
about fifteen tokens per second and skimming perhaps twice that, so a model generating
forty tokens per second in a chat is already faster than its reader; scoring it above one
generating thirty-two would be scoring a number nobody experiences.

The targets, from section 11.4 of the design specification:

===========  ======  =========================================================
Use case     Target  Why
===========  ======  =========================================================
chat             30  a conversation is read as it arrives, and pauses show
general          25  a mixed workload, mostly read as it arrives
coding           20  output is reviewed, not read aloud; bursts are short
reasoning        15  thinking tokens are skipped past, not read
multimodal       15  an image costs prompt time, and the reply is usually short
embedding       200  nobody reads an embedding; this is a throughput job
===========  ======  =========================================================

Embedding is scored on prompt throughput rather than generation, because an embedding run
generates nothing at all.

Coding and reasoning carry a prompt-processing modifier on top: ten points off below 100
prompt tokens per second, twenty below 40. These are the two use cases that begin by
feeding the model something long — a repository, a document, a chain of earlier thinking —
and a model that generates comfortably while taking a minute and a half to read a
twenty-thousand-token prompt feels slow in a way the generation figure alone never shows.
The modifier replaces itself rather than stacking: below 40 the penalty is twenty, not
thirty.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from llamafit.models.plan import SpeedEstimate
from llamafit.scoring.weights import check_use_case

TARGET_TPS: Mapping[str, float] = MappingProxyType(
    {
        "chat": 30.0,
        "general": 25.0,
        "coding": 20.0,
        "reasoning": 15.0,
        "multimodal": 15.0,
        "embedding": 200.0,
    }
)
"""Tokens per second at which a use case stops benefiting from more speed."""

PROMPT_SENSITIVE = frozenset({"coding", "reasoning"})
"""The use cases that begin with a long prompt and so are scored on reading it, too."""

SLOW_PROMPT = 100.0
"""Prompt tokens per second below which a prompt-sensitive use case loses ten points."""

VERY_SLOW_PROMPT = 40.0
"""Prompt tokens per second below which it loses twenty instead."""

SLOW_PROMPT_PENALTY = 10.0
"""What crossing :data:`SLOW_PROMPT` costs."""

VERY_SLOW_PROMPT_PENALTY = 20.0
"""What crossing :data:`VERY_SLOW_PROMPT` costs. It replaces the ten, never adds to it."""


def target_tps(use_case: str) -> float:
    """Return the tokens per second this use case is scored against.

    Args:
        use_case: What the request asked for.

    Returns:
        The target from :data:`TARGET_TPS`.

    Raises:
        ConfigError: If the use case is not one of the six.
    """
    return TARGET_TPS[check_use_case(use_case)]


def prompt_penalty(pp_tps: float, use_case: str) -> float:
    """Return the points a slow prompt costs this use case.

    Args:
        pp_tps: Prompt tokens per second.
        use_case: What the request asked for.

    Returns:
        Zero for a use case that does not start with a long prompt, and otherwise ten or
        twenty points depending on how slow the prompt is.
    """
    if use_case not in PROMPT_SENSITIVE:
        return 0.0
    if pp_tps < VERY_SLOW_PROMPT:
        return VERY_SLOW_PROMPT_PENALTY
    if pp_tps < SLOW_PROMPT:
        return SLOW_PROMPT_PENALTY
    return 0.0


def speed_score(speed: SpeedEstimate, use_case: str) -> float:
    """Score an estimate against what this use case needs, from 0 to 100.

    Args:
        speed: The estimate for this placement.
        use_case: What the request asked for.

    Returns:
        100 when the model is at least as fast as the target, the proportion of the
        target otherwise, less any prompt-processing penalty, and never below zero.

    Raises:
        ConfigError: If the use case is not one of the six.
    """
    target = target_tps(use_case)
    # An embedding run generates nothing, so the figure that matters is how fast it reads.
    observed = speed.pp_tps if use_case == "embedding" else speed.gen_tps
    reached = 100.0 * min(1.0, max(0.0, observed) / target)
    return max(0.0, reached - prompt_penalty(speed.pp_tps, use_case))
