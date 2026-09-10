# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""How the estimated speed compares with what this use case actually needs.

The score runs between two speeds, and they are different kinds of number.

**The target is a property of the job**, and it is where the score stops. Past the point
where a person stops waiting, more speed buys nothing, so forty tokens per second in a
chat scores exactly what thirty-two does: the difference is a number nobody experiences.

**The floor is a property of the person**, and it is where the score starts. Reading runs
at something like five to seven tokens a second — a couple of hundred words a minute,
three quarters of a word to a token — so a model slower than that cannot keep up with
somebody reading its own output. It is not a slow model; it is a model you wait for, one
sentence at a time, and no use case makes that acceptable. The floor therefore does not
move with the target the way everything else in this table does.

Between the two, the score is **linear in doublings, not in tokens per second**:

``speed_score = 100 x log2(tps / floor) / log2(target / floor)``, clamped to 0 and 100

A person does not experience tokens per second, they experience the multiple of their own
reading rate, and the steps in that multiple are not equal in tokens per second. Going
from just keeping up to twice reading speed is the largest single improvement available
below the target — it is the difference between waiting and not — and a straight line in
tokens per second prices it the same as the step from twenty-one to twenty-five, which
nobody can feel. The composite's other arm already makes this argument: section 11.3's
capacity curve is linear in halvings of the model for the same reason, that a fiftieth of
a machine and a hundredth of a machine are a ratio apart and not a difference apart.

The targets and the floor, from section 11.4 of the design specification:

===========  ======  =====  ====================================================
Use case     Target  Floor  Why the target is there
===========  ======  =====  ====================================================
chat             30      6  a conversation is read as it arrives, and pauses show
general          25      6  a mixed workload, mostly read as it arrives
coding           20      6  output is reviewed, not read aloud; bursts are short
reasoning        15      6  thinking tokens are skipped past, not read
multimodal       15      6  an image costs prompt time, the reply is usually short
embedding       200      0  nobody reads an embedding; this is a throughput job
===========  ======  =====  ====================================================

Embedding is scored on prompt throughput rather than generation, because an embedding run
generates nothing at all — and for the same reason it has no reading floor and keeps the
straight line to zero. There is no person waiting on the tokens, so there is no speed at
which the experience stops existing; there is only more work done or less.

**What the old shape said, and why it could not be defended.** ``100 x min(1, tps /
target)`` sloped politely to zero from the target, so Gemma 3 27B at 2.1 tokens per second
scored 8.4 out of 100 for a general request and finished above Llama 3.1 8B at 9.7 on the
same machine, carried there by quality and fit. Eight percent is not what a person waiting
three and a half minutes for a five-hundred-token answer is getting. They are getting none
of it, and the curve now says so.

**And what that costs.** Every model below the floor scores zero, so on a machine where
nothing reaches six tokens per second the speed column stops separating candidates at all
and the board is ordered by quality, fit and context. That is the right answer to the
wrong question — when nothing is interactive, "which is fastest" is not what a person
needs settling, "which is worth the wait" is — but it is a real change in behaviour on a
slow machine and not a free one.

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
from math import log2
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

READING_TPS = 6.0
"""Tokens per second a person reads at, below which generation scores nothing.

Silent reading of prose runs at roughly 240 words a minute, which is four words a second,
and a token is about three quarters of a word: five to seven tokens per second, and six is
the middle of that band rather than a measurement of anybody in particular.

**It is the one number in this module that describes the reader and not the machine**, so
it does not scale with the use case. A person reading a reasoning trace reads at the same
rate as a person reading a chat reply; what changes between them is how much faster than
that they need it to be, and that is the target's job.
"""

THROUGHPUT_ONLY = frozenset({"embedding"})
"""Use cases nobody reads the output of, which therefore have no reading floor."""

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


def floor_tps(use_case: str) -> float:
    """Return the speed below which this use case scores nothing at all.

    Args:
        use_case: What the request asked for.

    Returns:
        :data:`READING_TPS` for anything a person reads as it arrives, and zero for a
        throughput job, which has no reader to fall behind.

    Raises:
        ConfigError: If the use case is not one of the six.
    """
    return 0.0 if check_use_case(use_case) in THROUGHPUT_ONLY else READING_TPS


def speed_ramp(observed: float, floor: float, target: float) -> float:
    """Score one speed between a floor and a target, from 0 to 100.

    Args:
        observed: Tokens per second the estimate says.
        floor: The speed at which the score reaches zero. Zero itself means there is no
            floor, and the ramp is then the straight line to the origin.
        target: The speed at which the score reaches 100, and above which it stays there.

    Returns:
        100 at or above the target, 0 at or below the floor, and in between the share of
        the doublings from the floor to the target that this speed has covered.
    """
    if observed >= target:
        return 100.0
    if observed <= floor:
        return 0.0
    if floor <= 0.0:
        return 100.0 * observed / target
    return 100.0 * log2(observed / floor) / log2(target / floor)


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
        100 when the model is at least as fast as the target, zero when it is at or below
        the speed its reader reads at, the share of the doublings between them otherwise,
        less any prompt-processing penalty, and never below zero.

    Raises:
        ConfigError: If the use case is not one of the six.
    """
    checked = check_use_case(use_case)
    # An embedding run generates nothing, so the figure that matters is how fast it reads.
    observed = speed.pp_tps if checked in THROUGHPUT_ONLY else speed.gen_tps
    reached = speed_ramp(max(0.0, observed), floor_tps(checked), TARGET_TPS[checked])
    return max(0.0, reached - prompt_penalty(speed.pp_tps, checked))
