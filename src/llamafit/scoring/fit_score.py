# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""How well a candidate uses the machine, punishing both ends of the range.

Section 11.3 asks two questions, and they are not answerable from one number.

**Is it crowded?** A configuration that fills 95 percent of a card is one browser window
away from paging into system memory, where the speed collapses without any error being
raised. That is a property of the *whole* budget — weights, cache, buffers, the backend's
own overhead — read on whichever pool is worst, because a placement is as fragile as its
tightest pool and no average of the two would say so.

**Does it waste the machine?** A tool that scored purely on safety would rank the smallest
model that runs above every other, every time, and would therefore tell a person with
128 GB of memory and an eight-gigabyte card to run a 0.6B model — perfectly safe,
perfectly fast and almost useless. The user did not buy the machine to leave it idle.

**The second question cannot be asked of the first question's number, and that was the
defect.** Worst-pool utilisation on a machine with a small card is mostly not the model:
on the reference machine a 0.6B at Q8 puts 0.6 GB of weights on the card and then 6.3 GB
of key-value cache, compute buffer and backend overhead on top, so the card reads as
90 percent full while the machine as a whole holds one two-hundredth of what it could
hold. A 27B model shows the same 90 percent. Worse, the cache is *elastic* — the planner
grows the context until the card is full — so scoring waste on the total budget scores the
planner's own appetite rather than the model, and every candidate comes out equally well
fitted. Fifty times the model, the same number.

So waste is asked of the one part of the budget that is a property of the model choice and
cannot be turned down: **the weight tensors that have to be resident**, against **every
byte of memory the machine has to hold them in**, both pools added. Not the cache the
planner sized, not the buffers that cost the same whatever runs inside them. Tensors the
catalog marks streamable and the planner reads from disk are not resident and do not
count; they are not occupying the machine.

The two questions are the two arms of one curve, and the score is the worse of them,
because neither complaint excuses the other:

=======================  ======================================================
:func:`capacity_score`   the weights against the machine's memory, ``share``
:func:`crowding_score`   the whole budget against its tightest pool, ``u``
=======================  ======================================================

``share >= 0.50`` earns full marks: a model occupying half the memory a machine has is
using that machine. Below it the score falls by :data:`WASTE_PER_HALVING` for every
*halving* of the model, reaching zero a little past a fortieth of the machine.

**The rate is the specification's own and is not tuned here.** Section 11.3 already priced
waste: 100 at 0.50 and 70 at 0.20, thirty points for a shortfall of two and a half times.
The curve below passes through both of those points exactly. What changes is the axis it
is linear in — halvings rather than raw share, because a fiftieth of a machine and a
hundredth of a machine are a ratio apart and not a difference apart — and the quantity it
reads. A constant fitted until the five seeded models came out in the wanted order would
be wrong on the sixth; this one was fitted to nothing.

**There is no floor any more, and the reason the old one existed is gone.** It read: a
tiny model is still the right answer on a machine where nothing else fits, so never score
it zero. But ``share`` is relative to the machine by construction — on a small machine a
small model *is* half the memory and scores 100, and only on a machine that could hold
forty times more does it score nothing. The floor was compensating for a quantity that did
not know how big the machine was. And a zero here costs a candidate a fifth to a quarter
of its composite, never its place on the board: nothing is dropped for fitting badly.

The right-hand arm is untouched — same shape, same constants, same quantity. Penalising a
configuration that barely fits is correct and well calibrated, and the case where it does
not merely barely fit but pages is named separately, by the budget's own verdict.
"""

from __future__ import annotations

from math import inf, log2

from llamafit.models.plan import Budget

MODEL_COMPONENTS = frozenset(
    {
        "dense-weights",
        "shared-expert-weights",
        "expert-weights",
        "token-embedding",
        "output-head",
        "global-weights",
        "lazy-tables",
        "vision-projector",
    }
)
"""The budget lines that are the model itself rather than the cost of running it.

What is left out is either elastic or fixed, and neither says anything about how big the
model is: the key-value cache and the recurrent state grow with the context the planner
chose, the compute and output buffers are set by the micro-batch and the vocabulary, and
the backend context and the process overhead are the same bytes whatever is loaded.
``vision-projector`` is in because it is weights; ``vision-projector-compute`` is out
because it is a buffer.
"""

SPARSE = 0.20
"""The share of the machine's memory at which a model scores :data:`SPARSE_SCORE`."""

SPARSE_SCORE = 70.0
"""What a model occupying :data:`SPARSE` of the machine's memory scores."""

IDEAL_LOW = 0.50
"""At or above this share of the machine's memory, the weights earn full marks."""

IDEAL_HIGH = 0.80
"""Where the tightest pool starts to be crowded and the risk of paging counts."""

CROWDED = 0.98
"""The last utilisation that scores at all."""

CROWDED_SCORE = 40.0
"""What a candidate scores at :data:`CROWDED`, immediately before the cliff."""

WASTE_PER_HALVING = (100.0 - SPARSE_SCORE) / log2(IDEAL_LOW / SPARSE)
"""What each halving of the model below :data:`IDEAL_LOW` costs, about 22.7 points.

Derived, not chosen: it is the price section 11.3 already put on the shortfall between
:data:`IDEAL_LOW` and :data:`SPARSE`, read as a rate rather than as a line segment. That
the score reaches zero at roughly a fortieth of the machine follows from the rate; it is
not a second opinion about where waste becomes total.
"""


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


def resident_model_bytes(budget: Budget) -> int:
    """How many bytes of the model itself this placement holds in memory.

    Weight tensors only, and only the resident ones: a line the planner put in the
    ``disk`` pool is read as it is needed and is not occupying the machine.

    Args:
        budget: What the configuration needs, line by line.

    Returns:
        The sum of the :data:`MODEL_COMPONENTS` lines across the two memory pools.
    """
    return sum(
        line.bytes_
        for line in budget.lines
        if line.component in MODEL_COMPONENTS and line.pool in ("vram", "ram")
    )


def model_share(budget: Budget) -> float:
    """The share of the machine's memory the model's own weights occupy.

    Both pools count, added: the card and system memory are both places a weight tensor
    can live, and on a machine with a small card most of them live in the second. A
    machine with unified memory charges every line to system memory and reports no card
    at all, so nothing there is counted twice.

    Args:
        budget: What the configuration needs, per line and per pool.

    Returns:
        Resident weight bytes over the memory available to hold them. Zero for a
        placement holding no weights, and infinite on a machine reporting no free memory
        at all — which the crowding arm scores at zero anyway.
    """
    resident = resident_model_bytes(budget)
    if resident <= 0:
        return 0.0
    capacity = budget.vram_available + budget.ram_available
    if capacity <= 0:
        return inf
    return resident / capacity


def crowding_score(utilisation: float) -> float:
    """Score the tightest pool's fullness, from 0 to 100: the right-hand arm.

    Args:
        utilisation: The worst pool's required-over-available fraction.

    Returns:
        100 up to :data:`IDEAL_HIGH`, falling to :data:`CROWDED_SCORE` at
        :data:`CROWDED`, and 0 above it.
    """
    if utilisation <= IDEAL_HIGH:
        return 100.0
    if utilisation <= CROWDED:
        # Falling from 100 at 0.80 to 40 at 0.98: still fits, with less and less room for
        # anything else the machine is doing.
        span = (utilisation - IDEAL_HIGH) / (CROWDED - IDEAL_HIGH)
        return 100.0 - (100.0 - CROWDED_SCORE) * span
    return 0.0


def capacity_score(share: float) -> float:
    """Score how much of the machine the model claims, from 0 to 100: the left-hand arm.

    Args:
        share: Resident weight bytes over the machine's memory.

    Returns:
        100 at or above :data:`IDEAL_LOW`, falling by :data:`WASTE_PER_HALVING` for every
        halving below it, and 0 once there is nothing left to take off.
    """
    if share >= IDEAL_LOW:
        return 100.0
    if share <= 0.0:
        return 0.0
    return max(0.0, 100.0 - WASTE_PER_HALVING * log2(IDEAL_LOW / share))


def fit_score(share: float, utilisation: float) -> float:
    """Score how well a configuration uses the machine, from 0 to 100.

    Args:
        share: The share of the machine's memory the model's own weights occupy.
            Negative input is read as zero; a model cannot occupy less than nothing.
        utilisation: The worst pool's required-over-available fraction. Negative input is
            read as zero; a pool cannot need less than nothing.

    Returns:
        The worse of :func:`capacity_score` and :func:`crowding_score`. A configuration
        that both wastes the machine and crowds its card is excused neither complaint by
        the other, and the one that reads worse is the one worth telling a reader.
    """
    return min(capacity_score(max(0.0, share)), crowding_score(max(0.0, utilisation)))


def fit_score_for(budget: Budget) -> float:
    """Score a budget's fit, from its weights and from its worst pool.

    Args:
        budget: What the configuration needs, per line and per pool.

    Returns:
        :func:`fit_score` of :func:`model_share` and :func:`worst_pool_utilisation`.
    """
    return fit_score(model_share(budget), worst_pool_utilisation(budget))
