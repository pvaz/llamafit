"""The paging detector: a run that pages, a run that does not, and a run nobody checked.

Section 16.4's signature is a conjunction, and each of these tests is about one half of it
firing without the other. The reference numbers come from the calibration record: the same
model runs at 13 to 14 tokens per second in every configuration that fits and at 6 when six
gigabytes page, with a clean log either way.
"""

from __future__ import annotations

import pytest

from llamafit.bench.paging import (
    PAGING_SPEED_FRACTION,
    PAGING_VRAM_FRACTION,
    detect_paging,
    largest_safe_context,
    reason_text,
)
from llamafit.models.plan import ContextTier

GIB = 1024**3
CARD = 8188 * 1024**2


def tiers() -> tuple[ContextTier, ...]:
    """A ladder where the top two rungs page and the rest fit."""
    return (
        ContextTier(tokens=16384, vram_required=6 * GIB, fits=True, verdict="fits"),
        ContextTier(tokens=32768, vram_required=7 * GIB, fits=True, verdict="tight"),
        ContextTier(tokens=65536, vram_required=9 * GIB, fits=True, verdict="too-tight"),
        ContextTier(tokens=131072, vram_required=13 * GIB, fits=False, verdict="does-not-fit"),
    )


def test_a_full_card_and_a_collapsed_speed_is_paging() -> None:
    check = detect_paging(
        peak_vram_bytes=int(CARD * 0.99),
        vram_total_bytes=CARD,
        measured_gen_tps=6.2,
        estimated_gen_tps=13.4,
        tiers=tiers(),
    )
    assert check.paged is True
    assert check.reason == "paging"
    assert check.vram_ratio == pytest.approx(0.99)
    assert check.speed_ratio == pytest.approx(6.2 / 13.4)


def test_a_paging_run_is_told_what_context_would_not_page() -> None:
    check = detect_paging(
        peak_vram_bytes=CARD,
        vram_total_bytes=CARD,
        measured_gen_tps=6.0,
        estimated_gen_tps=14.0,
        tiers=tiers(),
    )
    # The 64K rung is where the budget already said the card would overflow, so it is
    # never offered as the cure for an overflowing card.
    assert check.suggested_context == 32768


def test_a_full_card_running_at_the_estimated_speed_is_a_good_plan_not_a_bad_one() -> None:
    """Filling the card is what the fit score rewards; on its own it says nothing."""
    check = detect_paging(
        peak_vram_bytes=int(CARD * 0.99),
        vram_total_bytes=CARD,
        measured_gen_tps=13.9,
        estimated_gen_tps=13.4,
        tiers=tiers(),
    )
    assert check.paged is False
    assert check.reason == "speed-as-expected"


def test_a_slow_run_on_a_card_with_room_to_spare_is_not_paging() -> None:
    """The estimate being wrong is not the driver paging, and they must not be confused."""
    check = detect_paging(
        peak_vram_bytes=4 * GIB,
        vram_total_bytes=CARD,
        measured_gen_tps=5.0,
        estimated_gen_tps=20.0,
    )
    assert check.paged is False
    assert check.reason == "card-not-full"


def test_a_card_with_room_is_cleared_without_needing_an_estimate_at_all() -> None:
    """The one sound negative on a single signal: the driver pages only what does not fit."""
    check = detect_paging(
        peak_vram_bytes=4 * GIB,
        vram_total_bytes=CARD,
        measured_gen_tps=None,
        estimated_gen_tps=None,
    )
    assert check.paged is False


def test_no_vram_reading_means_not_checked_and_never_means_not_paging() -> None:
    check = detect_paging(
        peak_vram_bytes=None,
        vram_total_bytes=CARD,
        measured_gen_tps=6.0,
        estimated_gen_tps=14.0,
    )
    assert check.paged is None
    assert check.reason == "no-vram-reading"


def test_a_card_whose_total_is_unknown_cannot_be_called_full_or_empty() -> None:
    check = detect_paging(
        peak_vram_bytes=7 * GIB,
        vram_total_bytes=None,
        measured_gen_tps=6.0,
        estimated_gen_tps=14.0,
    )
    assert check.paged is None
    assert check.reason == "no-card-total"


def test_a_full_card_with_nothing_to_compare_the_speed_against_is_undecided() -> None:
    check = detect_paging(
        peak_vram_bytes=CARD,
        vram_total_bytes=CARD,
        measured_gen_tps=6.0,
        estimated_gen_tps=None,
    )
    assert check.paged is None
    assert check.reason == "no-estimate"


def test_the_thresholds_are_the_ones_section_sixteen_states() -> None:
    assert PAGING_VRAM_FRACTION == 0.97
    assert PAGING_SPEED_FRACTION == 0.60


def test_a_ladder_with_no_safe_rung_suggests_nothing_rather_than_the_least_bad_one() -> None:
    paging_only = (
        ContextTier(tokens=32768, vram_required=9 * GIB, fits=True, verdict="too-tight"),
        ContextTier(tokens=65536, vram_required=13 * GIB, fits=False, verdict="does-not-fit"),
    )
    assert largest_safe_context(paging_only) is None
    assert largest_safe_context(()) is None


def test_every_reason_reads_as_a_sentence() -> None:
    for reason in (
        "paging",
        "card-not-full",
        "speed-as-expected",
        "no-vram-reading",
        "no-card-total",
        "no-estimate",
    ):
        assert reason_text(reason).strip().endswith(".")  # type: ignore[arg-type]
