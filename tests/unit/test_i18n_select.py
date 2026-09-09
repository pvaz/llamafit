"""The order languages are chosen in, and what happens to a request that cannot be met."""

import pytest

from llamafit.i18n.catalogs import available_languages
from llamafit.i18n.detect import FixedLocale
from llamafit.i18n.select import LANGUAGE_ENV_VAR, resolve_language
from llamafit.i18n.tags import SOURCE_LANGUAGE

AVAILABLE = ("en", "pt_PT")
"""The set every test here injects.

Made up on purpose, and not read from the packaged catalogs: what this file is about is
the order a language is chosen in, which must not change when somebody contributes a
translation. Only the one test that is about the packaged set asks what actually ships.
"""


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
    # What this guards is that the packaged catalogs are consulted when a caller names
    # none, not which of them a particular request lands on. That second thing changes
    # every time somebody contributes a translation -- ask for "pt" once pt_BR ships and
    # the answer is no longer pt_PT -- and a test that pinned it would fail on their work.
    packaged = available_languages()
    assert resolve_language(env={}, locale_provider=FixedLocale([])).available == packaged
    asked = next(tag for tag in packaged if tag != SOURCE_LANGUAGE)
    choice = resolve_language(asked, env={}, locale_provider=FixedLocale([]))
    assert choice.language == asked
    assert choice.honoured
    assert choice.available == packaged


def test_the_real_environment_is_read_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LANGUAGE_ENV_VAR, "pt_PT")
    choice = resolve_language(locale_provider=FixedLocale([]), available=AVAILABLE)
    assert (choice.language, choice.source) == ("pt_PT", "environment")


def test_the_system_locale_is_read_from_the_machine_when_no_provider_is_given() -> None:
    choice = resolve_language(env={}, available=AVAILABLE)
    assert choice.language in AVAILABLE


def test_a_region_with_no_catalog_gets_another_region_and_is_told() -> None:
    choice = resolve_language("pt_BR", env={}, locale_provider=FixedLocale([]), available=AVAILABLE)
    assert choice.language == "pt_PT"
    assert choice.requested == "pt_BR"
    assert not choice.honoured
    assert choice.notice == "LlamaFit has no pt_BR translation, so it is using the pt_PT one."
    assert choice.hint == "Contribute a pt_BR catalog: docs/translations.md says how."


def test_a_substitution_is_announced_even_when_the_system_locale_asked() -> None:
    # A Brazilian machine is the common case, and the reader deserves to know why some
    # of the wording looks foreign. This is the one thing the system locale does report.
    choice = resolve_language(
        env={}, locale_provider=FixedLocale(["pt_BR.UTF-8"]), available=AVAILABLE
    )
    assert (choice.language, choice.source) == ("pt_PT", "system")
    assert choice.notice == "LlamaFit has no pt_BR translation, so it is using the pt_PT one."


def test_a_region_less_match_is_not_a_substitution_and_says_nothing() -> None:
    for asked in ("pt", "pt_PT"):
        choice = resolve_language(
            asked, env={}, locale_provider=FixedLocale([]), available=AVAILABLE
        )
        assert (choice.language, choice.notice, choice.honoured) == ("pt_PT", None, True)
