# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""How well a candidate uses the machine, punishing both ends of the range.

The curve, from section 11.3 of the design specification, over the worst pool's
utilisation ``u``:

===================  ===========================================================
``u < 0.20``         70, the floor for a model that barely touches the machine
``0.20 <= u < 0.50`` rising from 70 to 100
``0.50 <= u <= 0.80`` 100
``0.80 < u <= 0.98`` falling from 100 to 40
``u > 0.98``         0
===================  ===========================================================

The right-hand slope is the one that reads as obvious: a configuration filling 95 percent
of a card is one browser window away from paging to system memory, where the speed
collapses without any error being raised.

**The left-hand slope is not a bug, and this is why it exists.** A tool that scored purely
on safety would rank the smallest model that runs above every other, every time, and would
therefore tell a person with 128 GB of memory and an eight-gigabyte card to run a
0.6B model — a recommendation that is perfectly safe, perfectly fast and almost useless.
The user did not buy the machine to leave it idle. Utilisation below half is evidence that
a bigger model, a wider quant or a longer context was available and was not taken, and the
score says so by holding back thirty points. It is the one place in this program where
using more of the machine is treated as better rather than riskier, and it is deliberate.

Below ``u = 0.20`` the curve flattens rather than continuing down. The specification names
70 as the value at 0.20 and says nothing about what lies to its left, and there is no
honest slope to invent there: the difference between using two percent of a machine and
twenty percent is not a difference in what the user gets. The floor keeps a tiny model
scoreable — it is still the right answer for a smoke test, or for a machine where nothing
else fits — while making sure it can never win a comparison on fit alone.
"""

from __future__ import annotations

from llamafit.models.plan import Budget

SPARSE = 0.20
"""Below this, a candidate is simply not using the machine, and the curve flattens."""

SPARSE_SCORE = 70.0
"""What a candidate that barely touches the machine scores."""

IDEAL_LOW = 0.50
"""Where the ideal band begins."""

IDEAL_HIGH = 0.80
"""Where the ideal band ends and the risk of paging starts to count."""

CROWDED = 0.98
"""The last utilisation that scores at all."""

CROWDED_SCORE = 40.0
"""What a candidate scores at :data:`CROWDED`, immediately before the cliff."""


def worst_pool_utilisation(budget: Budget) -> float:
    """Return the utilisation of whichever pool this configuration strains most.

    A placement that leaves system memory half empty while filling the card to the brim
    is as fragile as its worst pool, and no average of the two would say so.

    Args:
        budget: What the configuration needs, per pool.

    Returns:
        The larger of the VRAM and system-memory utilisations, or the system-memory one
        alone on a machine with no card.
    """
    if budget.vram_utilisation is None:
        return budget.ram_utilisation
    return max(budget.vram_utilisation, budget.ram_utilisation)


def fit_score(utilisation: float) -> float:
    """Score how well a configuration uses the machine, from 0 to 100.

    Args:
        utilisation: The worst pool's required-over-available fraction. Negative input
            is read as zero; a pool cannot need less than nothing.

    Returns:
        The curve described in this module's documentation.
    """
    u = max(0.0, utilisation)
    if u < SPARSE:
        return SPARSE_SCORE
    if u < IDEAL_LOW:
        # Rising from 70 at 0.20 to 100 at 0.50: the machine is being put to use.
        return SPARSE_SCORE + (100.0 - SPARSE_SCORE) * (u - SPARSE) / (IDEAL_LOW - SPARSE)
    if u <= IDEAL_HIGH:
        return 100.0
    if u <= CROWDED:
        # Falling from 100 at 0.80 to 40 at 0.98: still fits, with less and less room for
        # anything else the machine is doing.
        return 100.0 - (100.0 - CROWDED_SCORE) * (u - IDEAL_HIGH) / (CROWDED - IDEAL_HIGH)
    return 0.0


def fit_score_for(budget: Budget) -> float:
    """Score a budget's fit, from its worst pool.

    Args:
        budget: What the configuration needs, per pool.

    Returns:
        :func:`fit_score` of :func:`worst_pool_utilisation`.
    """
    return fit_score(worst_pool_utilisation(budget))
