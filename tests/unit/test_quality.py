"""Quality: the curator's baseline, what the quant takes off and what the match adds on."""

from __future__ import annotations

import pytest

from llamafit.catalog.loader import load_catalog
from llamafit.models.catalog import Catalog, CatalogModel
from llamafit.models.plan import Needs
from llamafit.quality import (
    alignment_bonus,
    baseline_for,
    missing_capabilities,
    penalty_for,
    primary_use_case,
    score_quality,
)
from llamafit.quality.alignment import (
    CAPABILITY_BONUS,
    MAX_ALIGNMENT_BONUS,
    PRIMARY_USE_CASE_BONUS,
)
from llamafit.quality.quant_penalty import FAMILY_PENALTIES


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    loaded, problems = load_catalog(custom_path=None)
    assert problems == []
    return loaded


def model_of(catalog: Catalog, model_id: str) -> CatalogModel:
    return catalog.by_id[model_id]


def test_the_baseline_is_the_curators_own_figure(catalog: Catalog) -> None:
    assert baseline_for(model_of(catalog, "qwen3-coder-next")) == 85.0
    assert baseline_for(model_of(catalog, "qwen3-0.6b")) == 35.0


@pytest.mark.parametrize(
    ("quant", "expected"),
    [
        ("Q8_0", 0.0),
        ("Q6_K", 1.0),
        ("Q5_K_M", 2.0),
        ("Q4_K_M", 4.0),
        ("Q4_K_XL", 4.0),
        ("Q4_0", 4.0),
        ("IQ4_XS", 6.0),
        ("Q3_K_M", 10.0),
        ("IQ3_XXS", 12.0),
        ("Q2_K", 20.0),
        ("IQ2_M", 24.0),
        ("IQ1_S", 35.0),
        ("F16", 0.0),
        ("BF16", 0.0),
    ],
)
def test_every_level_in_the_specification_costs_what_it_says(quant: str, expected: float) -> None:
    assert penalty_for(quant) == expected


@pytest.mark.parametrize(
    ("quant", "expected"),
    [
        ("UD-Q4_K_XL", 3.0),
        ("UD-Q2_K_XL", 19.0),
        ("UD-IQ1_S", 34.0),
        # Q8 already costs nothing, and a dynamic quant cannot be better than the weights
        # it was made from, so the discount stops at zero rather than going negative.
        ("UD-Q8_0", 0.0),
    ],
)
def test_a_dynamic_quant_is_worth_one_point_back(quant: str, expected: float) -> None:
    assert penalty_for(quant) == expected


@pytest.mark.parametrize("quant", ["  ud-q4_k_xl  ", "q8_0", "Ud-Q4_K_Xl"])
def test_case_and_whitespace_do_not_change_a_quants_cost(quant: str) -> None:
    assert penalty_for(quant) == penalty_for(quant.strip().upper())


@pytest.mark.parametrize(
    "quant",
    [
        "banana",
        "",
        "K_M",
        "QQ4",
        "UD-",
        "Q",
        # Shaped like a quant name, but no such level exists and no penalty is defensible.
        "Q7_K",
        "IQ8_0",
        "Q0",
    ],
)
def test_an_unrecognised_quantisation_is_none_rather_than_a_guess(quant: str) -> None:
    assert penalty_for(quant) is None


def test_the_penalty_table_widens_as_the_quant_narrows() -> None:
    order = ["Q8", "Q6", "Q5", "Q4", "IQ4", "Q3", "IQ3", "Q2", "IQ2", "IQ1"]
    penalties = [FAMILY_PENALTIES[level] for level in order]
    assert penalties == sorted(penalties)


def test_the_primary_use_case_is_the_first_one_the_entry_lists(catalog: Catalog) -> None:
    assert primary_use_case(model_of(catalog, "qwen3.8-flash-next")) == "coding"
    assert primary_use_case(model_of(catalog, "gemma-3-27b-it")) == "general"


def test_a_missing_capability_is_named_so_the_exclusion_can_be_explained(
    catalog: Catalog,
) -> None:
    gemma = model_of(catalog, "gemma-3-27b-it")
    assert missing_capabilities(gemma, Needs(capabilities=("coding", "vision"))) == ("coding",)
    assert missing_capabilities(gemma, Needs(capabilities=("vision",))) == ()


def test_matching_the_primary_use_case_is_worth_five(catalog: Catalog) -> None:
    coder = model_of(catalog, "qwen3-coder-next")
    assert alignment_bonus(coder, Needs(use_case="coding")) == PRIMARY_USE_CASE_BONUS
    assert alignment_bonus(coder, Needs(use_case="chat")) == 0.0


def test_a_use_case_a_model_merely_also_serves_earns_nothing(catalog: Catalog) -> None:
    # Gemma lists chat third; the bonus is for the job a model was built and measured on.
    assert alignment_bonus(model_of(catalog, "gemma-3-27b-it"), Needs(use_case="chat")) == 0.0


def test_the_capability_a_use_case_implies_is_not_paid_for_twice(catalog: Catalog) -> None:
    coder = model_of(catalog, "qwen3-coder-next")
    only_implied = Needs(use_case="coding", capabilities=("coding",))
    assert alignment_bonus(coder, only_implied) == PRIMARY_USE_CASE_BONUS
    with_tools = Needs(use_case="coding", capabilities=("coding", "tools"))
    assert alignment_bonus(coder, with_tools) == PRIMARY_USE_CASE_BONUS + CAPABILITY_BONUS


def test_a_capability_the_model_lacks_earns_nothing(catalog: Catalog) -> None:
    coder = model_of(catalog, "qwen3-coder-next")
    assert alignment_bonus(coder, Needs(use_case="chat", capabilities=("vision",))) == 0.0


def test_the_bonus_stops_at_ten_however_long_the_request(catalog: Catalog) -> None:
    flash = model_of(catalog, "qwen3.8-flash-next")
    long_request = Needs(
        use_case="coding",
        capabilities=("coding", "tools", "thinking", "vision", "multilingual"),
    )
    assert alignment_bonus(flash, long_request) == MAX_ALIGNMENT_BONUS


def test_the_breakdown_keeps_all_three_parts_and_their_sum(catalog: Catalog) -> None:
    needs = Needs(use_case="coding", capabilities=("coding", "tools"))
    quality = score_quality(model_of(catalog, "qwen3-coder-next"), "UD-Q4_K_XL", needs)
    assert quality.baseline == 85.0
    assert quality.quant_penalty == 3.0
    assert quality.alignment_bonus == 8.0
    assert quality.quality == 90.0


def test_quality_is_clamped_to_the_scale_at_the_top(catalog: Catalog) -> None:
    flash = model_of(catalog, "qwen3.8-flash-next").model_copy(deep=True)
    flash.quality.baseline = 98
    needs = Needs(use_case="coding", capabilities=("coding", "tools", "thinking", "vision"))
    quality = score_quality(flash, "Q8_0", needs)
    assert quality.alignment_bonus == MAX_ALIGNMENT_BONUS
    assert quality.quality == 100.0


def test_quality_is_clamped_to_the_scale_at_the_bottom(catalog: Catalog) -> None:
    tiny = model_of(catalog, "qwen3-0.6b").model_copy(deep=True)
    tiny.quality.baseline = 20
    quality = score_quality(tiny, "IQ1_S", Needs(use_case="chat"))
    assert quality.quant_penalty == 35.0
    assert quality.quality == 0.0


def test_an_unrecognised_quantisation_refuses_to_be_scored(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="unrecognised quantisation"):
        score_quality(model_of(catalog, "qwen3-0.6b"), "banana", Needs())
