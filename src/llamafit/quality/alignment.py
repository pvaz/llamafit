"""What matching the request is worth, and what missing it costs.

Two models with the same baseline are not equally good answers to the same question. A
coding model asked to write code is being used for what it was built and measured on; the
same model asked to describe an image is not. The alignment bonus is the small correction
that says so, from section 11.1 of the design specification:

- ``+5`` when the model's primary use case is the one the request names.
- ``+3`` for each required capability the model has beyond the one its use case already
  implies.
- the whole bonus capped at ``+10``.
- a missing required capability is not a penalty at all: the candidate is excluded, and
  :func:`missing_capabilities` is what says which one is missing so the exclusion can be
  explained rather than merely applied.

Two readings of the specification are settled here, and both are worth stating because
neither is forced by the words alone.

The *primary* use case is the first one the catalog entry lists. An entry writes them in
the curator's order of merit — ``use_cases: [coding, reasoning, multimodal]`` says this is
a coding model that can also reason and see — so the head of the list is the job the
baseline was set against. A model that lists a use case second gets no bonus for it, which
is the intended sharpness: the bonus exists to separate a model built for the job from one
that merely tolerates it.

*Beyond the request's use case* means the capability the use case itself implies does not
also earn three points. A coding request that requires the ``coding`` capability would
otherwise pay for the same match twice, once through the ``+5`` and once through the
``+3``. :data:`IMPLIED_CAPABILITY` is that mapping, and it is deliberately small: only
four of the six use cases name a capability at all, because *general* and *chat* imply
nothing a catalog entry records.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from llamafit.models.catalog import CatalogModel, UseCase
from llamafit.models.plan import Needs

PRIMARY_USE_CASE_BONUS = 5.0
"""What matching the model's primary use case is worth."""

CAPABILITY_BONUS = 3.0
"""What each required capability beyond the use case's own is worth."""

MAX_ALIGNMENT_BONUS = 10.0
"""The ceiling on the whole bonus.

Without it a request naming five capabilities would hand fifteen points to every model
that survived the capability filter, which is a fifth of the quality scale awarded for
asking a longer question.
"""

IMPLIED_CAPABILITY: Mapping[str, str] = MappingProxyType(
    {
        "coding": "coding",
        "reasoning": "thinking",
        "multimodal": "vision",
        "embedding": "embeddings",
    }
)
"""The capability a use case already asks for, and so does not pay twice for.

*general* and *chat* are absent on purpose. Neither corresponds to anything in the
catalog's capability list, so for those requests every required capability counts.
"""


def primary_use_case(model: CatalogModel) -> UseCase:
    """Return the job a model was built for: the first use case its entry lists.

    Args:
        model: The catalog entry. Its ``use_cases`` is never empty; the catalog model
            rejects an entry that has none.

    Returns:
        The first listed use case.
    """
    return model.use_cases[0]


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
    """Return what matching this request is worth to this model, from 0 to 10.

    A model missing a required capability is excluded elsewhere rather than scored, so
    this function assumes nothing about whether it clears that filter: every required
    capability it does have counts, and one it does not simply earns nothing.

    Args:
        model: The catalog entry.
        needs: What the user asked for.

    Returns:
        The bonus in points, from zero to :data:`MAX_ALIGNMENT_BONUS`.
    """
    bonus = 0.0
    if primary_use_case(model) == needs.use_case:
        bonus += PRIMARY_USE_CASE_BONUS
    implied = IMPLIED_CAPABILITY.get(needs.use_case)
    have = set(model.capabilities)
    extra = [c for c in needs.capabilities if c != implied and c in have]
    bonus += CAPABILITY_BONUS * len(extra)
    return min(MAX_ALIGNMENT_BONUS, bonus)
