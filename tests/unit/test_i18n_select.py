"""The order languages are chosen in, and what happens to a request that cannot be met."""

import pytest

from llamafit.i18n.detect import FixedLocale
from llamafit.i18n.select import LANGUAGE_ENV_VAR, resolve_language

AVAILABLE = ("en", "pt_PT")


def test_the_option_beats_everything() -> None:
    choice = resolve_language(
        "pt_PT",
        env={LANGUAGE_ENV_VAR: "en"},
        locale_provider=FixedLocale(["en_GB"]),
        available=AVAILABLE,
    )
    assert (choice.language, choice.source) == ("pt_PT", "option")
    assert choice.honoured
    assert choice.notice is None


def test_the_environment_variable_beats_the_operating_system() -> None:
    choice = resolve_language(
        env={LANGUAGE_ENV_VAR: "pt"},
        locale_provider=FixedLocale(["en_GB"]),
        available=AVAILABLE,
    )
    assert (choice.language, choice.source) == ("pt_PT", "environment")


def test_the_operating_system_decides_when_nobody_asked() -> None:
    choice = resolve_language(
        env={}, locale_provider=FixedLocale(["pt_PT.UTF-8"]), available=AVAILABLE
    )
    assert (choice.language, choice.source) == ("pt_PT", "system")


def test_the_first_locale_with_a_catalog_wins() -> None:
    choice = resolve_language(
        env={}, locale_provider=FixedLocale(["de_DE", "fr_FR", "pt_PT"]), available=AVAILABLE
    )
    assert choice.language == "pt_PT"


def test_english_is_the_last_resort() -> None:
    choice = resolve_language(env={}, locale_provider=FixedLocale([]), available=AVAILABLE)
    assert (choice.language, choice.source) == ("en", "default")
    assert choice.notice is None


def test_an_option_that_cannot_be_met_falls_back_to_english_and_says_so() -> None:
    choice = resolve_language("de", env={}, locale_provider=FixedLocale([]), available=AVAILABLE)
    assert choice.language == "en"
    assert choice.requested == "de"
    assert not choice.honoured
    assert choice.notice == "LlamaFit does not speak de, so it is using English."
    assert choice.hint == "Pass --language with one of: en, pt_PT."


def test_an_environment_variable_that_cannot_be_met_names_itself_in_the_hint() -> None:
    choice = resolve_language(
        env={LANGUAGE_ENV_VAR: "ja_JP"}, locale_provider=FixedLocale([]), available=AVAILABLE
    )
    assert choice.notice == "LlamaFit does not speak ja_JP, so it is using English."
    assert choice.hint == f"Pass {LANGUAGE_ENV_VAR} with one of: en, pt_PT."


def test_an_option_that_is_not_a_language_at_all_is_reported_as_written() -> None:
    choice = resolve_language(
        "klingon!", env={}, locale_provider=FixedLocale([]), available=AVAILABLE
    )
    assert choice.notice == "LlamaFit does not speak klingon!, so it is using English."


def test_an_unspoken_system_locale_is_not_worth_a_remark_on_every_run() -> None:
    choice = resolve_language(env={}, locale_provider=FixedLocale(["de_DE"]), available=AVAILABLE)
    assert (choice.language, choice.source) == ("en", "default")
    assert choice.notice is None


def test_a_blank_option_or_variable_is_not_a_request() -> None:
    choice = resolve_language(
        "  ",
        env={LANGUAGE_ENV_VAR: ""},
        locale_provider=FixedLocale(["pt_PT"]),
        available=AVAILABLE,
    )
    assert (choice.language, choice.source) == ("pt_PT", "system")


def test_asking_for_english_is_honoured_without_a_catalog() -> None:
    choice = resolve_language("en", env={}, locale_provider=FixedLocale([]), available=AVAILABLE)
    assert (choice.language, choice.source) == ("en", "option")
    assert choice.honoured


def test_the_packaged_languages_are_used_when_none_are_given() -> None:
    choice = resolve_language("pt", env={}, locale_provider=FixedLocale([]))
    assert choice.language == "pt_PT"
    assert "pt_PT" in choice.available


def test_the_real_environment_is_read_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LANGUAGE_ENV_VAR, "pt_PT")
    choice = resolve_language(locale_provider=FixedLocale([]), available=AVAILABLE)
    assert (choice.language, choice.source) == ("pt_PT", "environment")


def test_the_system_locale_is_read_from_the_machine_when_no_provider_is_given() -> None:
    choice = resolve_language(env={}, available=AVAILABLE)
    assert choice.language in AVAILABLE
