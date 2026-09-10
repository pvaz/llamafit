# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What matching the request is worth, and what not matching it costs.

Two models with the same baseline are not equally good answers to the same question. A
coding model asked to write code is being used for what it was built and measured on; the
same model asked to describe an image is not. Section 11.1 of the design specification
draws the line in three places, and this module is all three:

- a model whose ``use_cases`` do not include the one the request names is excluded, the
  same way a missing required capability excludes one. A request for coding must not come
  back with a model whose own entry says it is for general chat and reasoning.
- a model missing a required capability is excluded, and :func:`missing_capabilities` says
  which one, so the exclusion can be explained rather than merely applied.
- among the models that are left, ``+5`` when the request names the model's *primary* use
  case, and nothing when it merely lists it.

The *primary* use case is the first one the catalog entry lists. An entry writes them in
the curator's order of merit — ``use_cases: [coding, reasoning, multimodal]`` says this is
a coding model that can also reason and see — so the head of the list is the job the
baseline was set against. This is the whole of the bonus, and it varies between candidates,
which is the point: it separates a model built for the job from one that also does it.

An earlier reading of this section also gave ``+3`` per required capability. That term
could not discriminate: a missing required capability excludes the candidate, so every
candidate still standing has every required capability, and the bonus was the same number
added to all of them. It looked like a judgement and was arithmetic on the length of the
request. It is gone.

Both exclusions have the same cheap fix when they are wrong. The catalog is curated by
hand, so a model that really is good at something its entry does not claim is a one-line
change to that entry, reviewed like any other.
"""

from __future__ import annotations

from llamafit.models.catalog import CatalogModel, UseCase
from llamafit.models.plan import Needs

PRIMARY_USE_CASE_BONUS = 5.0
"""What matching the model's primary use case is worth, and the whole of the bonus."""


def primary_use_case(model: CatalogModel) -> UseCase:
    """Return the job a model was built for: the first use case its entry lists.

    Args:
        model: The catalog entry. Its ``use_cases`` is never empty; the catalog model
            rejects an entry that has none.

    Returns:
        The first listed use case.
    """
    return model.use_cases[0]


def declares_use_case(model: CatalogModel, use_case: str) -> bool:
    """Say whether this model's entry offers itself for this job at all.

    Args:
        model: The catalog entry.
        use_case: What the request asked for.

    Returns:
        True when ``use_case`` is among the entry's ``use_cases``, in any position. A
        model that merely lists it competes; a model that does not is excluded rather
        than ranked low, because a curated entry's declared purpose is meant to mean
        something.
    """
    return use_case in model.use_cases


def missing_capabilities(model: CatalogModel, needs: Needs) -> tuple[str, ...]:
    """Return the required capabilities this model does not have, in the order asked for.

    Args:
        model: The catalog entry.
        needs: What the user asked for.

    Returns:
        Every entry of ``needs.capabilities`` the model lacks. Empty means the model
        clears the filter; anything else is grounds for excluding it, and naming the
        capability is what makes that exclusion something a person can act on.
    """
    have = set(model.capabilities)
    return tuple(capability for capability in needs.capabilities if capability not in have)


def alignment_bonus(model: CatalogModel, needs: Needs) -> float:
    """Return what matching this request is worth to this model: five points, or none.

    Args:
        model: The catalog entry.
        needs: What the user asked for.

    Returns:
        :data:`PRIMARY_USE_CASE_BONUS` when the request names the job this model was
        built for, and zero otherwise — including for a model that lists the use case
        second or third, which is scored on its baseline alone.
    """
    if primary_use_case(model) == needs.use_case:
        return PRIMARY_USE_CASE_BONUS
    return 0.0
