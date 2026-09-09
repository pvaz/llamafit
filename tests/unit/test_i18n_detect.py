"""Where the operating system's locale comes from, on each platform."""

import locale
import sys

import pytest

from llamafit.i18n.detect import (
    FixedLocale,
    LocaleProvider,
    SystemLocale,
    current_locale,
    windows_ui_language,
)


def _provider(**env: str) -> SystemLocale:
    return SystemLocale(env=env, ui_language=lambda: None, process_locale=lambda: None)


def test_a_fixed_provider_answers_with_what_a_test_gave_it() -> None:
    provider: LocaleProvider = FixedLocale(["pt_PT", "en_GB"])
    assert provider.locale_tags() == ("pt_PT", "en_GB")
    assert FixedLocale().locale_tags() == ()


def test_the_posix_variables_are_read_most_specific_first() -> None:
    provider = _provider(LC_ALL="pt_PT.UTF-8", LC_MESSAGES="es_ES", LANG="en_GB")
    assert provider.locale_tags() == ("pt_PT.UTF-8", "es_ES", "en_GB")


def test_language_is_a_list_of_preferences_and_beats_the_rest() -> None:
    provider = _provider(LANGUAGE="pt_PT:pt:en", LANG="de_DE")
    assert provider.locale_tags() == ("pt_PT", "pt", "en", "de_DE")


def test_empty_and_blank_variables_are_skipped() -> None:
    provider = _provider(LANGUAGE="", LC_ALL="", LANG="   ", LC_MESSAGES="pt_PT")
    assert provider.locale_tags() == ("pt_PT",)


def test_the_windows_interface_language_is_used_when_no_variable_is_set() -> None:
    provider = SystemLocale(
        env={}, ui_language=lambda: "pt_PT", process_locale=lambda: "Portuguese_Portugal"
    )
    assert provider.locale_tags() == ("pt_PT", "Portuguese_Portugal")


def test_an_environment_variable_beats_the_operating_system() -> None:
    provider = SystemLocale(
        env={"LANG": "de_DE"}, ui_language=lambda: "pt_PT", process_locale=lambda: None
    )
    assert provider.locale_tags() == ("de_DE", "pt_PT")


def test_a_provider_that_can_answer_nothing_returns_nothing() -> None:
    assert _provider().locale_tags() == ()


def test_the_real_environment_is_read_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGUAGE", "pt_PT")
    provider = SystemLocale(ui_language=lambda: None, process_locale=lambda: None)
    assert provider.locale_tags()[0] == "pt_PT"


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows interface language")
def test_windows_reports_its_interface_language_as_a_locale_tag() -> None:
    # This is the path that actually runs on Windows: GetUserDefaultUILanguage gives a
    # language identifier, which locale.windows_locale turns into a tag.
    tag = windows_ui_language()
    assert tag is not None
    assert tag in set(locale.windows_locale.values())
    assert 2 <= len(tag.split("_")[0]) <= 3


@pytest.mark.skipif(sys.platform == "win32", reason="there is no kernel32 off Windows")
def test_there_is_no_interface_language_off_windows() -> None:
    assert windows_ui_language() is None


def test_the_process_locale_is_read_without_raising() -> None:
    value = current_locale()
    assert value is None or isinstance(value, str)


@pytest.mark.skipif(sys.platform != "win32", reason="there is no kernel32 off Windows")
def test_a_kernel32_call_that_fails_is_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    class _Broken:
        def __getattr__(self, name: str) -> object:
            raise OSError("no kernel32 here")

    monkeypatch.setattr(ctypes, "windll", _Broken(), raising=False)
    assert windows_ui_language() is None


def test_an_unparsable_process_locale_is_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> tuple[str | None, str | None]:
        raise ValueError("unknown locale")

    monkeypatch.setattr(locale, "getlocale", _raise)
    assert current_locale() is None
