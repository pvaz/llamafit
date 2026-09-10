"""What a quantisation costs a model in quality.

Quantisation trades accuracy for memory, and the trade is not linear: the first few bits
are nearly free and the last few are ruinous. The penalties below, from section 11.1 of
the design specification, are points off the curator's baseline on the common 0 to 100
scale, and they widen sharply below four bits because that is where published perplexity
and benchmark comparisons start to diverge from the unquantised model by more than the
gap between two neighbouring models.

============  =======  ==========================================================
Level         Penalty  What it means for a reader
============  =======  ==========================================================
``Q8``              0  indistinguishable from the original in practice
``Q6``              1  a rounding error away
``Q5``              2  a fair trade for the memory
``Q4``              4  the usual choice, and the point where a reader notices
``IQ4``             6  smaller than ``Q4`` for the same nominal width
``Q3``             10  visible degradation, worth it only to make a model fit
``IQ3``            12  the same, a little worse
``Q2``             20  a different model in all but name
``IQ2``            24  the same, a little worse
``IQ1``            35  a curiosity
============  =======  ==========================================================

Unsloth's dynamic quants, published with a ``UD-`` prefix, quantise different tensors to
different widths and keep the ones that matter wide. They are worth one point back
against the level they are named for, which is the specification's rule and not a
measurement: ``UD-Q4_K_XL`` is treated as a four-bit quant that costs three points rather
than four.

A name this module does not recognise returns ``None`` rather than a guess. LlamaFit's
standing rule is that it never assumes a number silently, and a quantisation whose cost
nobody can name is exactly the kind of thing that would otherwise be scored as free.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

FAMILY_PENALTIES: Mapping[str, float] = MappingProxyType(
    {
        "Q8": 0.0,
        "Q6": 1.0,
        "Q5": 2.0,
        "Q4": 4.0,
        "IQ4": 6.0,
        "Q3": 10.0,
        "IQ3": 12.0,
        "Q2": 20.0,
        "IQ2": 24.0,
        "IQ1": 35.0,
    }
)
"""Points off the baseline, by quantisation family.

The family is the letters and the digit: ``Q4_K_M``, ``Q4_K_XL`` and ``Q4_0`` are all
``Q4`` and all cost four points, which is what the specification says about the two it
names by hand. The table is keyed by family rather than by full name so that a quant
variant nobody has met yet is still scored, instead of falling off the end as unknown.
"""

UNQUANTISED = frozenset({"F32", "F16", "BF16"})
"""Widths that are not quantisations at all, and so cost nothing."""

DYNAMIC_PREFIX = "UD-"
"""The prefix Unsloth's dynamic quants carry."""

DYNAMIC_DISCOUNT = 1.0
"""What a dynamic quant is worth back against the level it is named for."""

_FAMILY_RE = re.compile(r"^(I?Q\d)")


def penalty_for(quant: str) -> float | None:
    """Return the points a quantisation costs, or ``None`` when it is not recognised.

    Args:
        quant: The quant's name as the catalog spells it, for example ``UD-Q4_K_XL``,
            ``Q4_K_M`` or ``Q8_0``. Case and surrounding whitespace do not matter.

    Returns:
        The penalty in points, never below zero, or ``None`` when the name matches
        neither an unquantised width nor a known family. A caller that gets ``None``
        has to say so to the user rather than score the candidate anyway.
    """
    name = quant.strip().upper()
    dynamic = name.startswith(DYNAMIC_PREFIX)
    if dynamic:
        name = name[len(DYNAMIC_PREFIX) :]
    if name in UNQUANTISED:
        return 0.0
    match = _FAMILY_RE.match(name)
    if match is None:
        return None
    penalty = FAMILY_PENALTIES.get(match.group(1))
    if penalty is None:
        return None
    if dynamic:
        # Never below zero: Q8 already costs nothing, and a dynamic Q8 cannot be better
        # than the original weights it was made from.
        return max(0.0, penalty - DYNAMIC_DISCOUNT)
    return penalty
