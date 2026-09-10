"""The three machine-facing scores: fit, speed and context, tested at their corners.

A curve is wrong at its thresholds long before it is wrong in the middle, so every test
here sits on a boundary or immediately either side of one.
"""

from __future__ import annotations

from math import inf

import pytest

from llamafit.errors import ConfigError
from llamafit.models.plan import Budget, BudgetLine, Needs
from llamafit.scoring.context_score import DEFAULT_CONTEXT, context_score, requested_context
from llamafit.scoring.fit_score import (
    CROWDED,
    CROWDED_SCORE,
    IDEAL_HIGH,
    IDEAL_LOW,
    SPARSE,
    SPARSE_SCORE,
    WASTE_PER_HALVING,
    capacity_score,
    crowding_score,
    fit_score,
    fit_score_for,
    model_share,
    resident_model_bytes,
    worst_pool_utilisation,
)
from llamafit.scoring.speed_score import (
    SLOW_PROMPT_PENALTY,
    TARGET_TPS,
    VERY_SLOW_PROMPT_PENALTY,
    prompt_penalty,
    speed_score,
    target_tps,
)
from tests.fixtures.scoring import RAM_AVAILABLE, VRAM_AVAILABLE, budget, speed

# --- fit: the crowding arm, which this change leaves exactly where it was ----------


@pytest.mark.parametrize(
    ("utilisation", "expected"),
    [
        (0.0, 100.0),
        (SPARSE, 100.0),
        (IDEAL_LOW, 100.0),
        (0.65, 100.0),
        (IDEAL_HIGH, 100.0),
        (0.801, 99.667),
        (0.89, 70.0),
        (0.979, 40.333),
        (CROWDED, CROWDED_SCORE),
        (0.981, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ],
)
def test_the_crowding_arm_at_every_threshold_and_either_side(
    utilisation: float, expected: float
) -> None:
    assert crowding_score(utilisation) == pytest.approx(expected, abs=0.01)


def test_the_cliff_above_the_crowded_threshold_is_a_cliff() -> None:
    # A driver does not refuse an allocation larger than the card; it pages, silently.
    assert crowding_score(CROWDED) == pytest.approx(CROWDED_SCORE)
    assert crowding_score(CROWDED + 0.001) == 0.0


# --- fit: the capacity arm, which is the one that changed --------------------------


def test_the_capacity_arm_passes_through_the_two_points_the_specification_names() -> None:
    # Section 11.3 priced waste at 100 for half the machine and 70 for a fifth of it. The
    # curve still goes through both of those points; what changed is what it is a curve of.
    assert capacity_score(IDEAL_LOW) == pytest.approx(100.0)
    assert capacity_score(SPARSE) == pytest.approx(SPARSE_SCORE)


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (-0.5, 0.0),
        (0.0, 0.0),
        (1.0, 100.0),
        (0.80, 100.0),
        (IDEAL_LOW, 100.0),
        (IDEAL_LOW / 2, 100.0 - WASTE_PER_HALVING),
        (IDEAL_LOW / 4, 100.0 - 2 * WASTE_PER_HALVING),
        (IDEAL_LOW / 8, 100.0 - 3 * WASTE_PER_HALVING),
        (IDEAL_LOW / 16, 100.0 - 4 * WASTE_PER_HALVING),
        (IDEAL_LOW / 32, 0.0),
        (0.001, 0.0),
    ],
)
def test_every_halving_of_the_model_costs_the_same(share: float, expected: float) -> None:
    assert capacity_score(share) == pytest.approx(expected, abs=0.01)


def test_the_arm_falls_by_ratio_and_not_by_difference() -> None:
    # Two models a factor of two apart are the same distance apart wherever they sit, so a
    # tenth of a machine and a fiftieth of it are not "both about nothing".
    assert capacity_score(0.4) - capacity_score(0.2) == pytest.approx(
        capacity_score(0.2) - capacity_score(0.1), abs=0.01
    )
    assert capacity_score(0.1) > capacity_score(0.03) > 0.0


def test_the_arm_has_no_floor_because_the_measure_is_relative_to_the_machine() -> None:
    # The old curve flattened at 70 below a fifth, which is part of how a 0.6B model came
    # to score what a 30B one did. Nothing is held back now, and nothing needs to be.
    assert capacity_score(0.005) == 0.0
    assert capacity_score(0.60) == 100.0


# --- fit: the two arms together ----------------------------------------------------


def test_the_score_is_the_worse_of_the_two_complaints() -> None:
    assert fit_score(0.60, 0.60) == pytest.approx(100.0)
    # Fills the machine and fills the card: crowding takes it.
    assert fit_score(0.60, 0.95) == pytest.approx(crowding_score(0.95))
    # Room to spare on the card and nothing of the machine used: capacity takes it.
    assert fit_score(0.02, 0.40) == pytest.approx(capacity_score(0.02))
    # Neither complaint excuses the other.
    assert fit_score(0.02, 0.95) == pytest.approx(min(capacity_score(0.02), crowding_score(0.95)))


def test_negative_input_at_either_end_is_read_as_nothing() -> None:
    assert fit_score(-0.5, -0.5) == 0.0


# --- fit: what a budget is read for ------------------------------------------------


def test_only_the_weight_lines_count_as_the_model() -> None:
    mixed = budget(
        vram_required=6 * 1024**3,
        ram_required=2 * 1024**3,
        weight_vram=4 * 1024**3,
        weight_ram=1 * 1024**3,
    )
    assert resident_model_bytes(mixed) == 5 * 1024**3


def test_a_tensor_streamed_from_disk_is_not_occupying_the_machine() -> None:
    held = budget(vram_required=4 * 1024**3, ram_required=0, weight_vram=4 * 1024**3)
    streamed = held.model_copy(
        update={
            "lines": (
                *held.lines,
                BudgetLine(component="lazy-tables", pool="disk", bytes=60 * 1024**3, exact=True),
            )
        }
    )
    assert resident_model_bytes(streamed) == resident_model_bytes(held)


def test_the_share_is_of_both_pools_added_because_weights_live_in_both() -> None:
    split = budget(
        vram_required=6 * 1024**3,
        ram_required=30 * 1024**3,
        weight_vram=5 * 1024**3,
        weight_ram=25 * 1024**3,
    )
    assert model_share(split) == pytest.approx(30 * 1024**3 / (VRAM_AVAILABLE + RAM_AVAILABLE))


def test_a_placement_holding_no_weights_at_all_claims_nothing() -> None:
    empty = budget(vram_required=2 * 1024**3, ram_required=0, weight_vram=0)
    assert model_share(empty) == 0.0
    assert fit_score_for(empty) == 0.0


def test_a_machine_reporting_no_free_memory_is_not_a_division_by_zero() -> None:
    # Built by hand rather than through the fixture: a machine with nothing free is what
    # a failed probe looks like, and the budget it produces has infinities in it already.
    nothing_free = Budget(
        lines=(BudgetLine(component="dense-weights", pool="ram", bytes=1024**3, exact=True),),
        vram_required=0,
        ram_required=1024**3,
        vram_available=0,
        ram_available=0,
        vram_utilisation=None,
        ram_utilisation=inf,
        verdict="does-not-fit",
    )
    assert model_share(nothing_free) == inf
    assert fit_score_for(nothing_free) == 0.0


def test_fit_is_decided_on_whichever_pool_is_worst() -> None:
    strained_card = budget(vram_required=int(7.9e9), ram_required=int(10e9))
    assert worst_pool_utilisation(strained_card) == pytest.approx(strained_card.vram_utilisation)
    strained_memory = budget(
        vram_required=int(1e9), ram_required=99 * 1024**3, ram_available=100 * 1024**3
    )
    assert worst_pool_utilisation(strained_memory) == pytest.approx(0.99)
    assert fit_score_for(strained_memory) == 0.0


def test_a_machine_with_no_card_is_scored_on_system_memory_alone() -> None:
    cpu_only = budget(vram_required=0, ram_required=60 * 1024**3, vram_available=0)
    assert cpu_only.vram_utilisation is None
    assert worst_pool_utilisation(cpu_only) == pytest.approx(0.6)
    # 60 GiB of weights out of the 100 GiB free is more than half the machine, and the
    # pool is nowhere near crowded, so both arms are content.
    assert fit_score_for(cpu_only) == 100.0


def test_a_full_card_no_longer_makes_a_tiny_model_look_well_fitted() -> None:
    """The defect section 11.3 was written to prevent and could not.

    On the reference machine a 0.6B at Q8 puts 0.6 GB of weights on the card and six times
    that in cache and buffers on top, so the card reads ninety percent full — the same as
    a model fifty times its size. Read on the pool the two score alike; read on the
    weights they do not.
    """
    card = int(0.9 * VRAM_AVAILABLE)
    tiny = budget(vram_required=card, ram_required=0, weight_vram=int(0.639e9))
    large = budget(
        vram_required=card,
        ram_required=46 * 1000**3,
        weight_vram=int(3.3e9),
        weight_ram=int(46e9),
    )
    assert worst_pool_utilisation(tiny) == pytest.approx(worst_pool_utilisation(large))
    assert fit_score_for(tiny) == 0.0
    # All that holds the larger one back is the card it fills, which is the complaint
    # the right-hand arm exists to make and the one this change does not touch.
    assert fit_score_for(large) == pytest.approx(crowding_score(0.9))


def test_the_same_model_fits_a_small_machine_it_could_never_fit_a_large_one() -> None:
    # Nothing about the model changed here; the machine did. That is the property the old
    # measure's floor was standing in for, and this one has by construction.
    weights = int(0.639e9)
    on_a_workstation = budget(vram_required=2 * 1024**3, ram_required=0, weight_vram=weights)
    on_a_laptop = budget(
        vram_required=0,
        ram_required=int(0.95e9),
        weight_ram=weights,
        vram_available=0,
        ram_available=int(1.2e9),
    )
    assert fit_score_for(on_a_workstation) == 0.0
    assert fit_score_for(on_a_laptop) == 100.0


# --- speed -------------------------------------------------------------------------


def test_the_targets_are_the_ones_the_specification_names() -> None:
    assert dict(TARGET_TPS) == {
        "chat": 30.0,
        "general": 25.0,
        "coding": 20.0,
        "reasoning": 15.0,
        "multimodal": 15.0,
        "embedding": 200.0,
    }


@pytest.mark.parametrize(
    ("gen_tps", "expected"),
    [(0.0, 0.0), (10.0, 50.0), (19.9, 99.5), (20.0, 100.0), (20.1, 100.0), (200.0, 100.0)],
)
def test_speed_is_scored_up_to_the_target_and_no_further(gen_tps: float, expected: float) -> None:
    assert speed_score(speed(gen_tps, 500.0), "coding") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pp_tps", "expected"),
    [
        (39.9, VERY_SLOW_PROMPT_PENALTY),
        (40.0, SLOW_PROMPT_PENALTY),
        (99.9, SLOW_PROMPT_PENALTY),
        (100.0, 0.0),
        (100.1, 0.0),
    ],
)
def test_the_prompt_modifier_at_its_two_thresholds(pp_tps: float, expected: float) -> None:
    assert prompt_penalty(pp_tps, "coding") == expected
    assert prompt_penalty(pp_tps, "reasoning") == expected


@pytest.mark.parametrize("use_case", ["chat", "general", "multimodal", "embedding"])
def test_a_use_case_that_does_not_start_with_a_long_prompt_pays_no_modifier(
    use_case: str,
) -> None:
    assert prompt_penalty(1.0, use_case) == 0.0


def test_the_modifier_replaces_itself_rather_than_stacking() -> None:
    assert prompt_penalty(10.0, "coding") == VERY_SLOW_PROMPT_PENALTY


def test_a_slow_prompt_comes_off_the_generation_score() -> None:
    assert speed_score(speed(20.0, 50.0), "coding") == pytest.approx(90.0)
    assert speed_score(speed(20.0, 30.0), "coding") == pytest.approx(80.0)


def test_the_speed_score_never_goes_below_zero() -> None:
    assert speed_score(speed(1.0, 5.0), "coding") == 0.0


def test_an_embedding_run_is_scored_on_how_fast_it_reads() -> None:
    # It generates nothing, so a generation figure would score every model at zero.
    assert speed_score(speed(0.0, 200.0), "embedding") == 100.0
    assert speed_score(speed(0.0, 100.0), "embedding") == pytest.approx(50.0)


def test_an_unknown_use_case_is_named_rather_than_raising_a_key_error() -> None:
    with pytest.raises(ConfigError, match="unknown use case"):
        target_tps("vibes")


# --- context -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fits", "asked", "expected"),
    [
        (0, 32768, 0.0),
        (8192, 32768, 25.0),
        (16384, 32768, 50.0),
        (32767, 32768, 99.997),
        (32768, 32768, 100.0),
        (32769, 32768, 100.0),
        (262144, 32768, 100.0),
    ],
)
def test_context_is_scored_in_proportion_up_to_what_was_asked_for(
    fits: int, asked: int, expected: float
) -> None:
    assert context_score(fits, asked) == pytest.approx(expected, abs=0.01)


def test_holding_more_context_than_asked_for_is_not_punished() -> None:
    # Unlike fit, this score is one-sided: unused context costs nothing at run time, and
    # what it costs in memory is already charged to the fit score.
    assert context_score(1_048_576, 8192) == context_score(8192, 8192)


def test_a_request_for_no_context_is_not_a_request() -> None:
    with pytest.raises(ValueError, match="above zero"):
        context_score(32768, 0)


def test_each_use_case_has_the_default_context_the_specification_gives() -> None:
    assert dict(DEFAULT_CONTEXT) == {
        "general": 8192,
        "chat": 8192,
        "coding": 32768,
        "reasoning": 32768,
        "multimodal": 16384,
        "embedding": 8192,
    }


def test_an_explicit_context_beats_the_use_cases_default() -> None:
    assert requested_context(Needs(use_case="coding")) == 32768
    assert requested_context(Needs(use_case="coding", requested_context=131072)) == 131072
    assert requested_context(Needs(use_case="chat")) == 8192


def test_an_unknown_use_case_has_no_default_context() -> None:
    with pytest.raises(ConfigError, match="unknown use case"):
        requested_context(Needs(use_case="vibes"))
