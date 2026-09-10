"""A model's quality for one request: the curator's score, less the quant, plus the match.

Quality is the one part of the composite score that says nothing about the machine. It is
the curator's editorial baseline (:mod:`llamafit.quality.baseline`), less what this
quantisation costs (:mod:`llamafit.quality.quant_penalty`), plus what matching the request
is worth (:mod:`llamafit.quality.alignment`), clamped to the common 0 to 100 scale::

    quality = clamp(baseline - quant_penalty + alignment_bonus, 0, 100)

Every part is kept in the :class:`~llamafit.models.plan.QualityBreakdown` that comes back,
because a quality of 90 that cannot be expanded into "85 from the curator, minus 3 for a
dynamic four-bit quant, plus 5 because coding is the job it was built for" is a number the
reader has to take on faith.
"""

from __future__ import annotations

from llamafit.models.catalog import CatalogModel
from llamafit.models.plan import Needs, QualityBreakdown
from llamafit.quality.alignment import (
    alignment_bonus,
    declares_use_case,
    missing_capabilities,
    primary_use_case,
)
from llamafit.quality.baseline import baseline_for
from llamafit.quality.quant_penalty import penalty_for

__all__ = [
    "alignment_bonus",
    "baseline_for",
    "declares_use_case",
    "missing_capabilities",
    "penalty_for",
    "primary_use_case",
    "score_quality",
]


def score_quality(model: CatalogModel, quant: str, needs: Needs) -> QualityBreakdown:
    """Score one model at one quantisation for one request, keeping the three parts.

    Args:
        model: The catalog entry.
        quant: The quant's name, as the catalog spells it.
        needs: What the user asked for.

    Returns:
        The baseline, the penalty, the bonus and the quality they add up to.

    Raises:
        ValueError: If the quantisation is not one :func:`~llamafit.quality.penalty_for`
            recognises. Scoring it anyway would mean inventing the one number in this
            calculation that nobody could defend, so the caller is told instead and
            reports it as an exclusion the user can read.
    """
    penalty = penalty_for(quant)
    if penalty is None:
        raise ValueError(f"unrecognised quantisation: {quant!r}")
    baseline = baseline_for(model)
    bonus = alignment_bonus(model, needs)
    return QualityBreakdown(
        baseline=baseline,
        quant_penalty=penalty,
        alignment_bonus=bonus,
        # Clamped at both ends: a heavily quantised experimental model can fall below
        # zero, and a frontier model with the full bonus can pass one hundred.
        quality=max(0.0, min(100.0, baseline - penalty + bonus)),
    )
