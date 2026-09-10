"""The three machine-facing scores: fit, speed and context, tested at their corners.

A curve is wrong at its thresholds long before it is wrong in the middle, so every test
here sits on a boundary or immediately either side of one.
"""

from __future__ import annotations

import pytest

from llamafit.errors import ConfigError
from llamafit.models.plan import Needs
from llamafit.scoring.context_score import DEFAULT_CONTEXT, context_score, requested_context
from llamafit.scoring.fit_score import (
    CROWDED,
    CROWDED_SCORE,
    IDEAL_HIGH,
    IDEAL_LOW,
    SPARSE,
    SPARSE_SCORE,
    fit_score,
    fit_score_for,
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
from tests.fixtures.scoring import budget, speed

# --- fit ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("utilisation", "expected"),
    [
        (-0.5, SPARSE_SCORE),
        (0.0, SPARSE_SCORE),
        (0.199, SPARSE_SCORE),
        (SPARSE, SPARSE_SCORE),
        (0.201, 70.1),
        (0.35, 85.0),
        (0.499, 99.9),
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
def test_the_fit_curve_at_every_threshold_and_either_side(
    utilisation: float, expected: float
) -> None:
    assert fit_score(utilisation) == pytest.approx(expected, abs=0.01)


def test_the_curve_punishes_both_ends_and_not_the_middle() -> None:
    # The left-hand slope is the one that reads like a bug: a model far smaller than the
    # machine scores below one that uses it. That is the whole point of it.
    assert fit_score(0.10) < fit_score(0.60)
    assert fit_score(0.95) < fit_score(0.60)
    assert fit_score(0.10) == SPARSE_SCORE


def test_the_cliff_above_the_crowded_threshold_is_a_cliff() -> None:
    # A driver does not refuse an allocation larger than the card; it pages, silently.
    assert fit_score(CROWDED) == pytest.approx(CROWDED_SCORE)
    assert fit_score(CROWDED + 0.001) == 0.0


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
    assert fit_score_for(cpu_only) == 100.0


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
