"""The active translator: choosing a language, and what the two functions then return."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from llamafit.i18n import translator
from llamafit.i18n.detect import FixedLocale
from llamafit.i18n.po import parse_po
from llamafit.i18n.translator import (
    CatalogTranslator,
    EnglishTranslator,
    _,
    current_language,
    get_translator,
    gettext,
    ngettext,
    set_language,
    set_translator,
)
from tests.fixtures import messages

CATALOG = """
msgid ""
msgstr ""
"Language: pt_PT\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "No GPU detected"
msgstr "Nenhuma GPU detetada"

msgid "Skip the RAM bandwidth measurement."
msgstr ""

msgid "%(count)d module"
msgid_plural "%(count)d modules"
msgstr[0] "%(count)d modulo"
msgstr[1] "%(count)d modulos"
"""


@pytest.fixture(autouse=True)
def _english_again() -> Iterator[None]:
    translator.reset()
    yield
    translator.reset()


def _catalog_dir(tmp_path: Path, text: str = CATALOG, name: str = "pt_PT.po") -> Path:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_english_is_installed_before_anything_is_chosen() -> None:
    assert current_language() == "en"
    assert isinstance(get_translator(), EnglishTranslator)
    assert _("No GPU detected") == "No GPU detected"
    assert ngettext("%(count)d module", "%(count)d modules", 1) == "%(count)d module"
    assert ngettext("%(count)d module", "%(count)d modules", 3) == "%(count)d modules"


def test_the_short_name_is_the_same_function_as_gettext() -> None:
    assert _ is gettext


def test_choosing_a_language_installs_its_catalog(tmp_path: Path) -> None:
    choice = set_language(
        "pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path)
    )
    assert choice.language == "pt_PT"
    assert current_language() == "pt_PT"
    assert _("No GPU detected") == "Nenhuma GPU detetada"
    assert ngettext("%(count)d module", "%(count)d modules", 4) == "%(count)d modulos"


def test_an_empty_translation_shows_english_not_a_blank_line(tmp_path: Path) -> None:
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert _("Skip the RAM bandwidth measurement.") == "Skip the RAM bandwidth measurement."


def test_a_message_the_catalog_lacks_shows_english(tmp_path: Path) -> None:
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert _("llama.cpp not found") == "llama.cpp not found"


def test_the_operating_system_locale_is_used_when_nobody_asked(tmp_path: Path) -> None:
    choice = set_language(
        env={},
        locale_provider=FixedLocale(["pt_PT.UTF-8"]),
        available=("en", "pt_PT"),
        directory=_catalog_dir(tmp_path),
    )
    assert (choice.language, choice.source) == ("pt_PT", "system")
    assert _("No GPU detected") == "Nenhuma GPU detetada"


def test_a_language_that_is_not_available_is_reported_once(tmp_path: Path) -> None:
    first = set_language("de", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert first.language == "en"
    assert first.notice == "LlamaFit does not speak de, so it is using English."
    assert first.hint is not None

    again = set_language("de", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert again.language == "en"
    assert again.requested == "de"
    assert again.notice is None
    assert again.hint is None


def test_a_different_unavailable_language_is_still_worth_reporting(tmp_path: Path) -> None:
    set_language("de", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    other = set_language("ja", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert other.notice == "LlamaFit does not speak ja, so it is using English."


def test_a_missing_catalog_falls_back_to_english_and_says_so(tmp_path: Path) -> None:
    choice = set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=tmp_path)
    assert choice.language == "en"
    assert current_language() == "en"
    assert choice.notice == "LlamaFit could not read its pt_PT translation, so it is using English."
    assert _("No GPU detected") == "No GPU detected"


def test_a_malformed_catalog_falls_back_to_english_and_says_so(tmp_path: Path) -> None:
    directory = _catalog_dir(tmp_path, text='msgctxt "menu"\nmsgid "a"\nmsgstr "b"\n')
    choice = set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=directory)
    assert choice.language == "en"
    assert choice.notice is not None
    assert "could not read" in choice.notice


def test_choosing_english_needs_no_catalog(tmp_path: Path) -> None:
    choice = set_language("en", env={}, available=("en", "pt_PT"), directory=tmp_path)
    assert (choice.language, choice.notice) == ("en", None)
    assert isinstance(get_translator(), EnglishTranslator)


def test_a_translator_can_be_installed_directly() -> None:
    set_translator(CatalogTranslator("pt_PT", parse_po(CATALOG)))
    assert current_language() == "pt_PT"
    assert gettext("No GPU detected") == "Nenhuma GPU detetada"


def test_the_wrapped_messages_go_through_the_active_translator(tmp_path: Path) -> None:
    assert messages.no_gpu_detected() == "No GPU detected"
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path))
    assert messages.no_gpu_detected() == "Nenhuma GPU detetada"
    assert messages.memory_modules(1) == "1 modulo"
    assert messages.memory_modules(2) == "2 modulos"
    assert messages.skip_measurement() == "Skip the RAM bandwidth measurement."


def test_the_packaged_portuguese_catalog_is_what_ships() -> None:
    choice = set_language("pt_PT", env={})
    assert choice.language == "pt_PT"
    assert messages.no_gpu_detected() == "Nenhuma GPU detetada"
    assert messages.memory_modules(1) == "1 módulo"
    assert messages.memory_modules(3) == "3 módulos"
    assert messages.catalog_problems(1) == "o catálogo tem 1 problema"
    assert messages.catalog_problems(2) == "o catálogo tem 2 problemas"
    assert messages.a_message_split_over_lines().startswith("O LlamaFit não conseguiu")
    assert messages.skip_measurement() == "Skip the RAM bandwidth measurement."


def test_a_substitution_notice_calls_the_catalog_what_the_catalog_calls_itself() -> None:
    choice = set_language("pt_BR", env={})
    assert choice.language == "pt_PT"
    assert choice.notice == (
        "LlamaFit has no pt_BR translation, so it is using the Portuguese (Portugal) one."
    )
    assert choice.hint == "Contribute a pt_BR catalog: docs/translations.md says how."
    assert messages.no_gpu_detected() == "Nenhuma GPU detetada"


def test_a_substitution_is_announced_once_like_any_other_notice() -> None:
    first = set_language("pt_BR", env={})
    again = set_language("pt_BR", env={})
    assert first.notice is not None
    assert again.language == "pt_PT"
    assert again.notice is None
    assert again.hint is None


def test_a_catalog_with_no_language_team_keeps_the_tag_in_the_notice(tmp_path: Path) -> None:
    # CATALOG declares Language but no Language-Team, so there is no name to fall back on.
    assert "Language-Team" not in CATALOG
    choice = set_language(
        "pt_BR", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path)
    )
    assert choice.language == "pt_PT"
    assert choice.notice == "LlamaFit has no pt_BR translation, so it is using the pt_PT one."
