# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The context ladder: what each context costs, so a launch script can choose one live.

This is the most useful thing the planner produces. A recommendation sized for the free
VRAM of one moment stops being true the moment a browser opens; a table of contexts with
the memory each one needs stays true, because the launch script reads the free VRAM it
actually sees and picks the largest rung that fits (section 15.3).

Two rules keep the table safe to act on. Only contexts the model itself supports appear,
because a launch script that compares free memory against a rung must never be offered a
rung the model would refuse. And ``fits`` describes the machine as it was scanned, which
is a different question from the one the launch script asks later — the number to compare
against live free VRAM is ``vram_required``.
"""

from __future__ import annotations

from llamafit.constants import CONTEXT_TIERS, MAX_CONTEXT_SEARCH_STEP
from llamafit.logging import get_logger
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.plan import ContextTier
from llamafit.placement.modes import BudgetFn, PlacementSettings, is_acceptable

_log = get_logger("placement.tiers")


def context_tiers(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    settings: PlacementSettings,
    *,
    budget_for: BudgetFn,
    ceiling: int | None = None,
) -> tuple[ContextTier, ...]:
    """Cost every rung of section 9.3's ladder for one placement.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        settings: The chosen placement; everything except the context is held fixed, so
            each rung differs from the chosen configuration in exactly one thing.
        budget_for: The injected budget function.
        ceiling: The longest context to offer, from
            :func:`~llamafit.placement.modes.context_ceiling`; the model's native length
            when the caller states none. A launch script reads this table and picks a
            rung, so a rung above a ceiling the request set is a rung the request said
            it did not want launched.

    Returns:
        One :class:`~llamafit.models.plan.ContextTier` per rung of
        :data:`~llamafit.constants.CONTEXT_TIERS` the model supports, in increasing
        order of context.
    """
    ceiling = model.context.native if ceiling is None else min(ceiling, model.context.native)
    tiers: list[ContextTier] = []
    for tokens in CONTEXT_TIERS:
        if tokens > ceiling:
            break
        budget = budget_for(model, quant, host, settings.with_context(tokens))
        tiers.append(
            ContextTier(
                tokens=tokens,
                vram_required=budget.vram_required,
                fits=is_acceptable(budget.verdict),
                verdict=budget.verdict,
            )
        )
    return tuple(tiers)


def max_context_fit(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    settings: PlacementSettings,
    *,
    budget_for: BudgetFn,
    ceiling: int | None = None,
) -> int:
    """The largest context this placement holds, to the nearest thousand tokens.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        settings: The chosen placement, whose context is taken to fit already.
        budget_for: The injected budget function.
        ceiling: The longest context to search to, from
            :func:`~llamafit.placement.modes.context_ceiling`; the model's native length
            when the caller states none.

    Returns:
        A context at or above ``settings.context`` and at or below the ceiling, on a
        :data:`~llamafit.constants.MAX_CONTEXT_SEARCH_STEP` grid.

    Found by bisection, which assumes what section 8.1 makes true: every component that
    depends on the context — the KV cache, the compute buffer's context term — grows with
    it and none shrinks, so a configuration that fits at some length fits at every shorter
    one. Reporting the figure to the nearest token would be false precision anyway: the
    compute buffer is a fitted formula, and the reference model's cache costs 33 MiB per
    thousand tokens.
    """
    step = MAX_CONTEXT_SEARCH_STEP
    low = settings.context
    high = model.context.native if ceiling is None else min(ceiling, model.context.native)
    if high <= low:
        return low

    def fits(context: int) -> bool:
        budget = budget_for(model, quant, host, settings.with_context(context))
        return is_acceptable(budget.verdict)

    steps = (high - low) // step
    best = 0
    lower, upper = 1, steps
    while lower <= upper:
        middle = (lower + upper) // 2
        if fits(low + middle * step):
            best = middle
            lower = middle + 1
        else:
            upper = middle - 1
    found = low + best * step
    # The grid rarely lands on the model's own limit, and that limit is the one context a
    # user is most likely to ask for by name, so it is tried once on its own account.
    if best == steps and found < high and fits(high):
        found = high
    _log.debug("max context fit for %s at %s: %d", model.id, settings.mode, found)
    return found
