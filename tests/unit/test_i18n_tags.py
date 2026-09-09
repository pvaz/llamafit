"""Putting locale tags into one shape, and matching a request to a catalog."""

import pytest

from llamafit.i18n.tags import SOURCE_LANGUAGE, match, normalise


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("pt", "pt"),
        ("pt_PT", "pt_PT"),
        ("pt-PT", "pt_PT"),
        ("PT_pt", "pt_PT"),
        ("pt_PT.UTF-8", "pt_PT"),
        ("de_DE@euro", "de_DE"),
        ("  en_GB  ", "en_GB"),
        ("es_419", "es_419"),
        ("C", SOURCE_LANGUAGE),
        ("POSIX", SOURCE_LANGUAGE),
        ("C.UTF-8", SOURCE_LANGUAGE),
        ("German_Germany", "de_DE"),
        ("portuguese_brazil", "pt_BR"),
    ],
)
def test_a_tag_comes_back_in_one_shape(tag: str, expected: str) -> None:
    assert normalise(tag) == expected


@pytest.mark.parametrize("tag", ["", "   ", "not a locale", "12", "Portuguese_Portugal"])
def test_text_that_names_no_language_comes_back_as_none(tag: str) -> None:
    assert normalise(tag) is None


def test_an_exact_match_wins() -> None:
    assert match("pt_PT", ["en", "pt", "pt_PT"]) == "pt_PT"
    assert match("pt", ["en", "pt", "pt_PT"]) == "pt"


def test_a_request_without_a_region_takes_the_specific_catalog() -> None:
    assert match("pt", ["en", "pt_PT"]) == "pt_PT"
    assert match("PT-pt.UTF-8", ["en", "pt_PT"]) == "pt_PT"


def test_a_request_with_a_region_takes_the_region_less_catalog() -> None:
    assert match("pt_PT", ["en", "pt"]) == "pt"


def test_one_region_is_never_served_by_another() -> None:
    # European and Brazilian Portuguese are different translations, so pt_BR gets
    # English rather than a catalog written for Portugal.
    assert match("pt_BR", ["en", "pt_PT"]) is None


def test_a_language_with_no_catalog_matches_nothing() -> None:
    assert match("de", ["en", "pt_PT"]) is None


def test_text_that_names_no_language_matches_nothing() -> None:
    assert match("", ["en", "pt_PT"]) is None


def test_the_available_tags_are_normalised_before_they_are_compared() -> None:
    assert match("pt_PT", ["pt-pt"]) == "pt-pt"
