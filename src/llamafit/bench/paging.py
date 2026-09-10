# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Catching the failure that does not look like one.

An NVIDIA driver on Windows or Linux does not refuse a VRAM allocation larger than the
card. It pages the excess to system memory and carries on. The server starts, the log is
clean, ``/health`` says ``ok``, and the model runs at a fraction of its speed for as long
as anybody leaves it running. Section 8.4 is the budget's half of this problem, predicting
which configurations would do it. This module is the other half: noticing that one just
did.

Section 16.4 gives the signature, and it is two things and not one:

* peak VRAM within three percent of the card's total, and
* generation below sixty percent of what the estimate said.

Both, because either alone is ordinary. A well-planned configuration is *supposed* to fill
the card -- the fit score rewards it -- so a full card on its own says the plan worked. And
an estimate can be sixty percent out for reasons that have nothing to do with the driver,
starting with the estimate being wrong. It is the conjunction that has no innocent reading:
the card is full and the work is crawling.

The detector answers in three states rather than two. ``True`` and ``False`` are the
verdict; ``None`` says the question could not be asked -- no vendor tool, no card total, no
estimate to compare against. A configuration nobody could check has not been cleared, and a
detector that reported "not paging" whenever it failed to look would be worse than no
detector, because somebody would believe it.
"""

from __future__ import annotations

from collections.abc import Sequence

from llamafit.bench.types import PagingCheck, PagingReason
from llamafit.i18n import _
from llamafit.models.plan import ContextTier

PAGING_VRAM_FRACTION = 0.97
"""How close to the card's total counts as full, from section 16.4's three percent.

Three percent of an eight-gigabyte card is 240 MiB, which is about what a driver keeps
back for itself and rather less than a desktop uses. Tighter and a run that filled the card
honestly would slip below the line; looser and a configuration with half a gigabyte spare
would qualify as full, which most well-planned ones are.
"""

PAGING_SPEED_FRACTION = 0.60
"""How far below the estimate counts as collapsed, from section 16.4's sixty percent.

The calibration record is where the figure comes from: Qwen3.8-Flash-Next runs at 13 to 14
tokens per second in every configuration that fits and 6 when six gigabytes page, which is
a factor of two, and 10 to 11 when two gigabytes page, which is not far off one. Estimates
before calibration are routinely twenty-five percent out in either direction, so a
threshold above about 0.75 would fire on ordinary error. Sixty percent sits below anything
a merely wrong estimate produces and above the mildest paging worth reporting.
"""


def detect_paging(
    *,
    peak_vram_bytes: int | None,
    vram_total_bytes: int | None,
    measured_gen_tps: float | None,
    estimated_gen_tps: float | None,
    tiers: Sequence[ContextTier] = (),
) -> PagingCheck:
    """Decide whether the driver was paging during a run.

    Args:
        peak_vram_bytes: The highest VRAM reading taken while the run was going, or
            ``None`` when no vendor tool answered.
        vram_total_bytes: The card's total, or ``None`` when it is not known.
        measured_gen_tps: What the run generated at.
        estimated_gen_tps: What the estimate said before the run. It has to be the figure
            from *before*, because an estimate corrected by this very measurement would
            agree with it by construction and the ratio would be one however badly the
            configuration was paging.
        tiers: The plan's context ladder, so a run that paged can be told what would not.

    Returns:
        The verdict, with both ratios and the reason it was reached.

    A card comfortably below the threshold is cleared without needing the speed half at
    all: the driver pages only what does not fit, so an allocation that fits has nothing to
    page. That is the one negative this detector can give on a single signal, and it is
    given because it is sound rather than because it is convenient.
    """
    if peak_vram_bytes is None:
        return PagingCheck(paged=None, reason="no-vram-reading")
    if not vram_total_bytes:
        return PagingCheck(paged=None, reason="no-card-total")

    vram_ratio = peak_vram_bytes / vram_total_bytes
    if vram_ratio < PAGING_VRAM_FRACTION:
        return PagingCheck(paged=False, vram_ratio=vram_ratio, reason="card-not-full")

    if not measured_gen_tps or not estimated_gen_tps:
        return PagingCheck(paged=None, vram_ratio=vram_ratio, reason="no-estimate")

    speed_ratio = measured_gen_tps / estimated_gen_tps
    if speed_ratio >= PAGING_SPEED_FRACTION:
        return PagingCheck(
            paged=False,
            vram_ratio=vram_ratio,
            speed_ratio=speed_ratio,
            reason="speed-as-expected",
        )
    return PagingCheck(
        paged=True,
        vram_ratio=vram_ratio,
        speed_ratio=speed_ratio,
        reason="paging",
        suggested_context=largest_safe_context(tiers),
    )


def largest_safe_context(tiers: Sequence[ContextTier]) -> int | None:
    """The biggest rung of a context ladder that would not page, or ``None`` if none would.

    Args:
        tiers: The ladder the planner built for this configuration.

    Returns:
        The largest context whose budget fits the card outright. A rung the budget already
        called ``too-tight`` is exactly a rung that pages, so it is never suggested as the
        cure for paging.
    """
    safe = [tier.tokens for tier in tiers if tier.fits and tier.verdict != "too-tight"]
    return max(safe) if safe else None


def reason_text(reason: PagingReason) -> str:
    """The verdict as a sentence, in the reader's language.

    Args:
        reason: The stored key.

    Returns:
        One sentence. The key is what is stored and this is what is shown, because a
        result written by somebody working in Portuguese is read later by somebody working
        in Arabic and the row in the database has to outlive both.
    """
    if reason == "paging":
        return _(
            "The card was full and generation came in below three fifths of the estimate:"
            " the driver was paging to system memory, which does not fail and does not"
            " appear in the log."
        )
    if reason == "card-not-full":
        return _(
            "Peak VRAM stayed clear of the card's total, so there was nothing for the"
            " driver to page."
        )
    if reason == "speed-as-expected":
        return _(
            "The card was full, which a good plan is meant to be, and the speed was close"
            " enough to the estimate that nothing was paging."
        )
    if reason == "no-vram-reading":
        return _(
            "No VRAM reading was taken during the run, so this configuration has not been"
            " cleared of paging; it has only not been checked."
        )
    if reason == "no-card-total":
        return _(
            "The card's total memory is unknown, so a reading of what was in use says"
            " nothing about how close to full it was."
        )
    return _(
        "There was no estimate to compare the measured speed against, so the second half"
        " of the signature could not be tested."
    )
