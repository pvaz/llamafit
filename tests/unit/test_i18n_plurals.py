"""The Plural-Forms header, and the standard library parser it is handed to."""

import pytest

from llamafit.i18n.plurals import (
    DEFAULT_PLURAL_FORMS,
    PluralFormsError,
    PluralRule,
    parse_plural_forms,
)

POLISH = "nplurals=3; plural=(n==1 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);"


def test_the_rule_the_template_ships_is_english() -> None:
    # It is what a translator starts from, not a fallback: a catalog declaring no
    # Plural-Forms is refused by the reader rather than quietly given this one.
    rule = parse_plural_forms(DEFAULT_PLURAL_FORMS)
    assert rule.nplurals == 2
    assert [rule.index(n) for n in (0, 1, 2, 100)] == [1, 0, 1, 1]
    assert rule.expression == "(n != 1)"


def test_a_language_with_three_forms_is_parsed_and_applied() -> None:
    rule = parse_plural_forms(POLISH)
    assert rule.nplurals == 3
    assert [rule.index(n) for n in (0, 1, 2, 4, 5, 22, 25)] == [2, 0, 1, 1, 2, 1, 2]


def test_a_rule_that_differs_from_english_at_zero() -> None:
    rule = parse_plural_forms("nplurals=2; plural=(n > 1);")
    assert rule.index(0) == 0
    assert rule.index(1) == 0
    assert rule.index(2) == 1


def test_a_language_with_one_form() -> None:
    rule = parse_plural_forms("nplurals=1; plural=0;")
    assert [rule.index(n) for n in (0, 1, 5)] == [0, 0, 0]


def test_the_trailing_semicolon_is_optional_and_whitespace_is_forgiven() -> None:
    assert parse_plural_forms("  nplurals = 2 ;  plural = n != 1  ").nplurals == 2


def test_a_rule_that_names_a_form_the_header_did_not_declare_is_clamped() -> None:
    rule = parse_plural_forms("nplurals=2; plural=n;")
    assert rule.index(7) == 1
    assert rule.index(-3) == 0


@pytest.mark.parametrize(
    "header",
    [
        "",
        "nonsense",
        "plural=(n != 1);",
        "nplurals=2;",
        "nplurals=2; plural=;",
        "nplurals=2; plural=__import__('os').system('echo');",
        "nplurals=2; plural=n if n else 0;",
    ],
)
def test_a_header_that_is_not_a_rule_is_refused(header: str) -> None:
    with pytest.raises(PluralFormsError):
        parse_plural_forms(header)


@pytest.mark.parametrize("nplurals", [0, -1, 11])
def test_an_impossible_form_count_is_refused(nplurals: int) -> None:
    with pytest.raises(PluralFormsError):
        PluralRule(nplurals, "0")


def test_a_rule_that_divides_by_zero_falls_back_to_the_first_form() -> None:
    rule = parse_plural_forms("nplurals=2; plural=n/0;")
    assert rule.index(3) == 0


def test_the_rule_shows_itself_as_a_header_would_write_it() -> None:
    assert repr(parse_plural_forms(DEFAULT_PLURAL_FORMS)) == (
        "PluralRule(nplurals=2, expression='(n != 1)')"
    )
