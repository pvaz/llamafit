"""The vision projector: the most expensive thing a small card can be asked to hold.

A projector is a separate file, so its weights are an exact figure from the catalog. What
it costs on the graphics card is not only those weights: the calibration record measured
862 MiB of weights, a 248 MiB compute buffer of its own, and a rise of 780 MiB in the main
compute buffer, together about 1.9 GB, or the equivalent of 52,000 tokens of context on an
8 GB card. On the processor it costs its weights and nothing measurable for text.

The two halves are kept as separate lines rather than added together, because they are
different kinds of claim: one is a file size and one is a fitted figure. The third part,
the floor the projector puts under the main compute buffer, belongs to that buffer and is
applied there.
"""

from __future__ import annotations

from llamafit.constants import PROJECTOR_COMPUTE_BYTES
from llamafit.i18n import _
from llamafit.models.catalog import Extra
from llamafit.models.plan import BudgetLine, Pool


def projector_lines(projector: Extra | None, *, pool: Pool | None) -> tuple[BudgetLine, ...]:
    """What a vision projector costs, in the pool it was placed in.

    Args:
        projector: The catalog's ``mmproj`` entry, or ``None`` when the model has no
            projector or vision was turned off to make the rest fit.
        pool: Where the projector was placed. ``None`` means it was left out, which is the
            same as having none: it costs nothing and the placement says vision is off.

    Returns:
        The projector's weights, and its own compute space when it is on the card. Nothing
        at all when there is no projector, when it was left out, or when the catalog does
        not yet know how large the file is.
    """
    if projector is None or pool is None or pool == "disk" or not projector.bytes_:
        return ()
    lines = [
        BudgetLine(
            component="vision-projector",
            pool=pool,
            bytes=projector.bytes_,
            exact=True,
            note=_("the projector file itself"),
        )
    ]
    if pool == "vram":
        lines.append(
            BudgetLine(
                component="vision-projector-compute",
                pool="vram",
                bytes=PROJECTOR_COMPUTE_BYTES,
                exact=False,
                note=_("the projector's own compute space, measured at 248 MiB"),
            )
        )
    return tuple(lines)
