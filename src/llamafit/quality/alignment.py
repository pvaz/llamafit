# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What matching the request is worth, and what not matching it costs.

Two models with the same baseline are not equally good answers to the same question. A
coding model asked to write code is being used for what it was built and measured on; the
same model asked to describe an image is not. Section 11.1 of the design specification
draws the line in two places, and this module is both:

- a model missing a capability the request needs is excluded. The request can name one
  outright, and asking for a job names one too, through
  :data:`~llamafit.models.catalog.CAPABILITY_FOR_USE_CASE`:
  :func:`required_capability` is what a use case turns into, and
  :func:`missing_capabilities` is what the request asked for by name. Either way the
  exclusion can say which ability is missing, which is what makes it something a person
  can act on.
- among the models that are left, ``+5`` when the request names the model's *primary* use
  case, and nothing when it merely lists it.

**The gate is on what a model can do, never on what it is offered for.** An earlier reading
excluded a model whose ``use_cases`` did not contain the requested one, and it threw away
good answers: ask this catalog for a general model and Qwen3-Coder-Next, the fastest thing
that fits the reference machine, was not ranked at all — because a coding model is a
perfectly reasonable thing to hold a general conversation with, and nothing in the entry
said otherwise. It also turned every curator's judgement call into a hard filter they did
not know they were setting. Capabilities are facts about the weights and can carry a gate;
emphasis cannot. The case the old rule was written for survives the change unharmed: Llama
3.1 8B lacks the coding *capability* as well as the coding use case, so a coding request
still excludes it, and now for the reason that was always the real one.

``use_cases`` keeps a job. The *primary* use case is the first one the catalog entry lists,
and an entry writes them in the curator's order of merit — ``use_cases: [coding, reasoning,
multimodal]`` says this is a coding model that can also reason and see — so the head of the
list is the job the baseline was set against. It is worth :data:`PRIMARY_USE_CASE_BONUS`,
which is how a model built for the task still outranks one that merely can do it. That is
the right weight for an editorial opinion: it moves a close board and it hides nothing.

An earlier reading of this section also gave ``+3`` per required capability. That term
could not discriminate: a missing required capability excludes the candidate, so every
candidate still standing has every required capability, and the bonus was the same number
added to all of them. It looked like a judgement and was arithmetic on the length of the
request. It is gone.

The exclusion has a cheap fix when it is wrong. The catalog is curated by hand, so a model
that really can do something its entry does not claim is a one-line change to that entry,
reviewed like any other.
"""

from __future__ import annotations

from llamafit.models.catalog import CAPABILITY_FOR_USE_CASE, Capability, CatalogModel, UseCase
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


def required_capability(use_case: str) -> Capability | None:
    """Return the ability asking for this job asks for, when asking for it asks for one.

    Args:
        use_case: What the request asked for. An unrecognised one requires nothing: this
            is not where a typo is caught — :func:`~llamafit.scoring.weights.check_use_case`
            names the six and refuses the rest — and inventing a requirement for a job
            nobody has defined would empty the board for a reason no reader could act on.

    Returns:
        The entry in :data:`~llamafit.models.catalog.CAPABILITY_FOR_USE_CASE`, which is
        ``None`` for ``general`` and ``chat``, since neither is a specialisation and
        neither needs one.
    """
    return CAPABILITY_FOR_USE_CASE.get(use_case)


def missing_capabilities(model: CatalogModel, needs: Needs) -> tuple[str, ...]:
    """Return the required capabilities this model does not have, in the order asked for.

    Args:
        model: The catalog entry.
        needs: What the user asked for.

    Returns:
        Every entry of ``needs.capabilities`` the model lacks. Empty means the model
        clears the filter; anything else is grounds for excluding it, and naming the
        capability is what makes that exclusion something a person can act on.

        Only what the request named itself. The capability the *use case* asks for is
        :func:`required_capability`, and it is kept apart because the two have different
        fixes: one is dropped from the request, the other is answered by asking for a
        different job, and a reason that offered the wrong one would be worse than no
        reason at all.
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
