# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What each part of the score counts for, per use case, and how a user changes it.

These weights are opinions. They are not measured, they cannot be derived, and two
reasonable people would write different ones, which is exactly why they live in one
readable table with the reasoning attached and why a user can replace them from the
config file. A ranking is only as honest as its willingness to show the numbers that
produced it, and the weights are the most arguable numbers in this program.

The table, from section 11.5 of the design specification:

==========  =======  =====  ===  =======
Use case    Quality  Speed  Fit  Context
==========  =======  =====  ===  =======
general        0.35   0.25  0.25    0.15
coding         0.40   0.20  0.20    0.20
reasoning      0.50   0.15  0.20    0.15
chat           0.25   0.40  0.25    0.10
multimodal     0.40   0.20  0.25    0.15
embedding      0.30   0.45  0.20    0.05
==========  =======  =====  ===  =======

The reasoning, which is the part a bare table loses:

*Coding* needs a model good enough to be trusted with a change and enough context to hold
a real repository, so quality and context are both weighted heavily; a coding session is
read as fast as it is generated and rarely waits on the model, so speed gives ground.

*Reasoning* is dominated by quality because thinking tokens are cheap to wait for and
expensive to get wrong. A slow model that reaches the right answer is worth more than a
fast one that does not, so speed falls to the lowest weight in the table.

*Chat* is judged mostly by how fast it feels. A conversation at fifteen tokens per second
is a worse experience than a slightly less capable one at forty, and the difference is
obvious to a person in a way that two points of benchmark score is not.

*General* is the balanced default, weighted so that no single part can carry a candidate.

*Multimodal* follows general, with a little more weight on fit: a projector and an image
context are what push these models into the pool that is already tight.

*Embedding* is a throughput job. Nobody reads an embedding, so speed dominates and
context barely matters — an embedding request is one short document at a time.

To override them, put the parts you want in the config file under the use case::

    [scoring.weights.coding]
    quality = 0.55
    speed = 0.15
    fit = 0.15
    context = 0.15

A use case that is overridden must give all four parts. A partial set would have to be
merged with the defaults and renormalised, and the result would be neither what the user
wrote nor what the defaults said.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import get_args

from llamafit.errors import ConfigError
from llamafit.i18n import _
from llamafit.models.catalog import UseCase

PARTS: tuple[str, ...] = ("quality", "speed", "fit", "context")
"""The four parts of the composite score, in the order a breakdown shows them."""

PREFERENCES: tuple[str, ...] = ("balanced", "quality", "speed")
"""What ``--prefer`` accepts, from section 12.1's ``Needs``."""

PREFERENCE_SHIFT = 0.10
"""How much weight ``--prefer`` moves between quality and speed (section 12.1).

A tenth is enough to reorder a close board and not enough to overturn a clear answer,
which is what a preference should be: the user leaning on the scales, not replacing them.
Somebody who wants more than a lean writes the four numbers into the config file, where
they can see exactly what they asked for.
"""

USE_CASES: tuple[str, ...] = get_args(UseCase)
"""Every use case the catalog knows, taken from the catalog's own type."""

DEFAULT_WEIGHTS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "general": MappingProxyType({"quality": 0.35, "speed": 0.25, "fit": 0.25, "context": 0.15}),
        "coding": MappingProxyType({"quality": 0.40, "speed": 0.20, "fit": 0.20, "context": 0.20}),
        "reasoning": MappingProxyType(
            {"quality": 0.50, "speed": 0.15, "fit": 0.20, "context": 0.15}
        ),
        "chat": MappingProxyType({"quality": 0.25, "speed": 0.40, "fit": 0.25, "context": 0.10}),
        "multimodal": MappingProxyType(
            {"quality": 0.40, "speed": 0.20, "fit": 0.25, "context": 0.15}
        ),
        "embedding": MappingProxyType(
            {"quality": 0.30, "speed": 0.45, "fit": 0.20, "context": 0.05}
        ),
    }
)
"""The project's default weights, keyed by use case and then by part."""


def check_use_case(use_case: str) -> str:
    """Return the use case unchanged, or explain that it is not one of the six.

    :class:`~llamafit.models.plan.Needs` types its use case as a plain string so that a
    request can be built from a config file or a command line without importing the
    catalog's vocabulary. That leaves this as the place a typo surfaces, and it surfaces
    as a sentence naming the six rather than as a :class:`KeyError` on a table.

    Args:
        use_case: What the request asked for.

    Returns:
        The same string.

    Raises:
        ConfigError: If it is not a known use case.
    """
    if use_case not in USE_CASES:
        raise ConfigError(
            _("unknown use case %(use_case)s") % {"use_case": use_case},
            hint=_("Known use cases: %(known)s.") % {"known": ", ".join(USE_CASES)},
        )
    return use_case


def weights_for(
    use_case: str, overrides: Mapping[str, Mapping[str, float]] | None = None
) -> dict[str, float]:
    """Return the weights for one use case, with any override from the config applied.

    The weights that come back always sum to one, so a total stays on the 0 to 100 scale
    and two use cases stay comparable. An override that sums to something else is
    normalised rather than refused — the ratios between the parts are what a user is
    expressing, and the breakdown carries the normalised numbers, so what was actually
    applied is on screen rather than in this docstring.

    Args:
        use_case: What the request asked for.
        overrides: The config file's weights, keyed by use case and then by part. Only
            the entry for this use case is read, but every key is checked, because a
            user who misspells a use case in a config file has to be told rather than
            left wondering why nothing changed.

    Returns:
        A weight for each of :data:`PARTS`, summing to one.

    Raises:
        ConfigError: If the use case is unknown, or an override names an unknown use
            case or part, omits a part, or holds a negative or zero-summing set.
    """
    check_use_case(use_case)
    if overrides is None:
        return dict(DEFAULT_WEIGHTS[use_case])
    for name in overrides:
        check_use_case(name)
    override = overrides.get(use_case)
    if override is None:
        return dict(DEFAULT_WEIGHTS[use_case])
    return _normalise(use_case, override)


def _normalise(use_case: str, override: Mapping[str, float]) -> dict[str, float]:
    """Check one use case's overridden weights and scale them so they sum to one."""
    unknown = sorted(set(override) - set(PARTS))
    if unknown:
        raise ConfigError(
            _("unknown score part %(part)s in the weights for %(use_case)s")
            % {"part": unknown[0], "use_case": use_case},
            hint=_("The parts are: %(parts)s.") % {"parts": ", ".join(PARTS)},
        )
    missing = [part for part in PARTS if part not in override]
    if missing:
        raise ConfigError(
            _("the weights for %(use_case)s leave out %(part)s")
            % {"use_case": use_case, "part": missing[0]},
            hint=_("Overriding a use case means giving all four parts: %(parts)s.")
            % {"parts": ", ".join(PARTS)},
        )
    negative = [part for part in PARTS if override[part] < 0]
    if negative:
        raise ConfigError(
            _("the weight for %(part)s in %(use_case)s is below zero")
            % {"part": negative[0], "use_case": use_case},
            hint=_("A weight says how much a part counts; it cannot count less than none."),
        )
    total = sum(float(override[part]) for part in PARTS)
    if total <= 0:
        raise ConfigError(
            _("the weights for %(use_case)s add up to nothing") % {"use_case": use_case},
            hint=_("At least one part has to count for something."),
        )
    return {part: float(override[part]) / total for part in PARTS}


def shift_preference(weights: Mapping[str, float], prefer: str) -> dict[str, float]:
    """Lean the weights towards quality or speed, by section 12.1's tenth.

    Args:
        weights: The use case's weights, already resolved.
        prefer: ``balanced``, ``quality`` or ``speed``.

    Returns:
        A new set of weights that still sums to one: the shift moves weight from one part
        to the other rather than adding any, so two requests stay comparable and the total
        stays on the 0 to 100 scale.

    Raises:
        ConfigError: If ``prefer`` is not one of the three.

    The shift is capped by what the other part has to give. Reasoning already weights
    quality at 0.50 and speed at 0.15, and a preference for quality there moves a tenth;
    a use case weighting speed at 0.05 would have only 0.05 to move, and moving it into a
    negative weight would score a part as though it counted against the model.
    """
    if prefer not in PREFERENCES:
        raise ConfigError(
            _("unknown preference %(prefer)s") % {"prefer": prefer},
            hint=_("Preferences: %(known)s.") % {"known": ", ".join(PREFERENCES)},
        )
    shifted = {part: float(weights[part]) for part in PARTS}
    if prefer == "balanced":
        return shifted
    giver, taker = ("speed", "quality") if prefer == "quality" else ("quality", "speed")
    moved = min(PREFERENCE_SHIFT, shifted[giver])
    shifted[giver] -= moved
    shifted[taker] += moved
    return shifted
