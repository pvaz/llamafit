# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The curator's editorial score for a model, before anything is taken off or added on.

The baseline is the only number in the composite score that no program produced. It is a
person's judgement of how good a model is at the job it was built for, written into the
catalog with the published benchmarks that justify it, and it is deliberately not derived
from those benchmarks by arithmetic: benchmarks measure different things on different
scales and a weighted average of them would look objective while hiding every choice that
went into the weights.

The rubric curators are held to, from section 11.1 of the design specification:

============  ==========================================================
90 and above  frontier open weights on their primary task
80 to 89      strong current generation
70 to 79      solid previous generation
50 to 69      small or dated
below 50      experimental
============  ==========================================================

An entry cites the benchmarks behind its figure, which is what makes the number arguable
rather than arbitrary: a reader who disagrees can open the entry, read the scores and the
sources, and send a pull request against the number.
"""

from __future__ import annotations

from llamafit.models.catalog import CatalogModel


def baseline_for(model: CatalogModel) -> float:
    """Return the curator's quality score for a model, on the common 0 to 100 scale.

    Args:
        model: The catalog entry.

    Returns:
        The entry's ``quality.baseline``, as a float so the arithmetic that follows it
        stays in one type.
    """
    return float(model.quality.baseline)
