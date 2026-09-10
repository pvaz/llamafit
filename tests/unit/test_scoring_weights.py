"""The weights: the specification's table, and what a user may put in its place."""

from __future__ import annotations

import pytest

from llamafit.errors import ConfigError
from llamafit.scoring.weights import DEFAULT_WEIGHTS, PARTS, USE_CASES, check_use_case, weights_for

SPECIFICATION_TABLE = {
    "general": {"quality": 0.35, "speed": 0.25, "fit": 0.25, "context": 0.15},
    "coding": {"quality": 0.40, "speed": 0.20, "fit": 0.20, "context": 0.20},
    "reasoning": {"quality": 0.50, "speed": 0.15, "fit": 0.20, "context": 0.15},
    "chat": {"quality": 0.25, "speed": 0.40, "fit": 0.25, "context": 0.10},
    "multimodal": {"quality": 0.40, "speed": 0.20, "fit": 0.25, "context": 0.15},
    "embedding": {"quality": 0.30, "speed": 0.45, "fit": 0.20, "context": 0.05},
}


def test_the_table_is_the_one_in_the_specification() -> None:
    assert {name: dict(row) for name, row in DEFAULT_WEIGHTS.items()} == SPECIFICATION_TABLE


def test_every_use_case_in_the_catalog_has_weights() -> None:
    assert set(DEFAULT_WEIGHTS) == set(USE_CASES)


@pytest.mark.parametrize("use_case", sorted(SPECIFICATION_TABLE))
def test_every_row_sums_to_one_so_totals_stay_on_the_scale(use_case: str) -> None:
    assert sum(weights_for(use_case).values()) == pytest.approx(1.0)


@pytest.mark.parametrize("use_case", sorted(SPECIFICATION_TABLE))
def test_every_row_covers_all_four_parts(use_case: str) -> None:
    assert set(weights_for(use_case)) == set(PARTS)


def test_the_opinions_in_the_table_are_the_ones_the_reasoning_claims() -> None:
    # Reasoning is dominated by quality; chat by speed; coding wants context as much as
    # it wants speed. If a future edit moves one of these, the docstring is now a lie.
    assert weights_for("reasoning")["quality"] == max(weights_for("reasoning").values())
    assert weights_for("chat")["speed"] == max(weights_for("chat").values())
    assert weights_for("coding")["context"] == weights_for("coding")["speed"]
    assert weights_for("embedding")["speed"] > weights_for("embedding")["quality"]


def test_an_unknown_use_case_is_told_which_six_exist() -> None:
    with pytest.raises(ConfigError) as caught:
        check_use_case("vibes")
    assert "vibes" in caught.value.message
    assert "coding" in (caught.value.hint or "")


def test_a_config_override_replaces_the_row_it_names() -> None:
    overrides = {"coding": {"quality": 0.7, "speed": 0.1, "fit": 0.1, "context": 0.1}}
    assert weights_for("coding", overrides) == pytest.approx(
        {"quality": 0.7, "speed": 0.1, "fit": 0.1, "context": 0.1}
    )
    # A use case the override does not mention keeps the default.
    assert weights_for("chat", overrides) == dict(DEFAULT_WEIGHTS["chat"])


def test_an_override_is_normalised_so_a_total_stays_on_the_scale() -> None:
    overrides = {"chat": {"quality": 2.0, "speed": 4.0, "fit": 2.0, "context": 2.0}}
    applied = weights_for("chat", overrides)
    assert sum(applied.values()) == pytest.approx(1.0)
    assert applied["speed"] == pytest.approx(0.4)


def test_an_override_naming_a_use_case_that_does_not_exist_is_reported() -> None:
    # Silently ignoring it would leave the user watching a setting that never applies.
    with pytest.raises(ConfigError, match="unknown use case"):
        weights_for("coding", {"vibes": {"quality": 1.0}})


def test_an_override_naming_a_part_that_does_not_exist_is_reported() -> None:
    overrides = {"coding": {"quality": 0.4, "speed": 0.2, "fit": 0.2, "vibes": 0.2}}
    with pytest.raises(ConfigError, match="unknown score part"):
        weights_for("coding", overrides)


def test_an_override_that_leaves_a_part_out_is_reported() -> None:
    with pytest.raises(ConfigError, match="leave out"):
        weights_for("coding", {"coding": {"quality": 0.5, "speed": 0.5, "fit": 0.5}})


def test_a_negative_weight_is_reported() -> None:
    overrides = {"coding": {"quality": 1.0, "speed": -0.5, "fit": 0.2, "context": 0.3}}
    with pytest.raises(ConfigError, match="below zero"):
        weights_for("coding", overrides)


def test_weights_that_add_up_to_nothing_are_reported() -> None:
    overrides = {"coding": {"quality": 0.0, "speed": 0.0, "fit": 0.0, "context": 0.0}}
    with pytest.raises(ConfigError, match="add up to nothing"):
        weights_for("coding", overrides)


def test_the_defaults_cannot_be_mutated_through_the_table() -> None:
    weights = weights_for("coding")
    weights["quality"] = 0.99
    assert weights_for("coding")["quality"] == 0.40
