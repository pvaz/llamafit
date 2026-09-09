"""The active translator: choosing a language, and what the two functions then return."""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from rich.errors import MarkupError
from rich.text import Text

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
    npgettext,
    pgettext,
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
    # msgctxt used to be the malformed example here; it is a supported keyword now, so
    # this asks for something the reader will always refuse.
    directory = _catalog_dir(tmp_path, text='msgid "a"\nmsgstr "b\\z"\n')
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


def test_a_context_gives_one_english_word_two_portuguese_genders() -> None:
    set_language("pt_PT", env={})
    assert messages.no_gpu_row() == "nenhuma detetada"
    assert messages.no_backends_row() == "nenhum detetado"
    assert messages.bandwidth_unknown() == "desconhecida"
    assert messages.bits_per_weight_unknown() == "desconhecido"


def test_english_answers_a_contextual_lookup_with_the_message_itself() -> None:
    assert isinstance(get_translator(), EnglishTranslator)
    assert pgettext("GPU", "none detected") == "none detected"
    assert npgettext("GPU", "%(count)d device", "%(count)d devices", 1) == "%(count)d device"
    assert npgettext("GPU", "%(count)d device", "%(count)d devices", 3) == "%(count)d devices"


def test_a_contextual_lookup_goes_through_the_installed_catalog(tmp_path: Path) -> None:
    text = CATALOG + '\nmsgctxt "GPU"\nmsgid "none detected"\nmsgstr "nenhuma detetada"\n'
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path, text))
    assert pgettext("GPU", "none detected") == "nenhuma detetada"
    assert pgettext("backends", "none detected") == "none detected"


def test_a_contextual_counting_message_goes_through_the_installed_catalog(tmp_path: Path) -> None:
    text = CATALOG + (
        '\nmsgctxt "GPU"\nmsgid "%(count)d device"\nmsgid_plural "%(count)d devices"\n'
        'msgstr[0] "%(count)d placa"\nmsgstr[1] "%(count)d placas"\n'
    )
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=_catalog_dir(tmp_path, text))
    assert npgettext("GPU", "%(count)d device", "%(count)d devices", 1) == "%(count)d placa"
    assert npgettext("GPU", "%(count)d device", "%(count)d devices", 9) == "%(count)d placas"


def _with_team(team: str) -> str:
    """CATALOG, plus a Language-Team header saying whatever a catalog's author wrote."""
    return CATALOG.replace(
        '"Language: pt_PT\\n"', '"Language: pt_PT\\n"\n"Language-Team: ' + team + '\\n"'
    )


@pytest.mark.parametrize("team", ["Portuguese [/PT]", "Portuguese [bold] (Portugal)"])
def test_a_team_name_with_a_bracket_tag_survives_being_printed(tmp_path: Path, team: str) -> None:
    # The notice quotes the catalog's own Language-Team header, which is data from a file.
    # console.print reads square brackets as markup: the first of these raises MarkupError
    # at start-up and the second silently eats the tag, so the notice loses words. Text()
    # is what the documented example passes, and this is what says so.
    directory = _catalog_dir(tmp_path, _with_team(team))
    choice = set_language("pt_BR", env={}, available=("en", "pt_PT"), directory=directory)
    assert choice.notice is not None
    assert team in choice.notice

    console = Console(file=io.StringIO(), no_color=True, width=200)
    console.print(Text(choice.notice))
    assert team in console.file.getvalue()  # type: ignore[union-attr]


def test_printing_the_notice_as_markup_is_what_the_example_avoids(tmp_path: Path) -> None:
    # Asserted rather than assumed: if Rich ever stopped treating a bracket as markup the
    # advice above would be obsolete, and this test is what would say so.
    directory = _catalog_dir(tmp_path, _with_team("Portuguese [/PT]"))
    choice = set_language("pt_BR", env={}, available=("en", "pt_PT"), directory=directory)
    assert choice.notice is not None
    console = Console(file=io.StringIO(), no_color=True, width=200)
    with pytest.raises(MarkupError):
        console.print(choice.notice)


def test_a_team_name_with_no_brackets_is_untouched_either_way(tmp_path: Path) -> None:
    directory = _catalog_dir(tmp_path, _with_team("Portuguese (Portugal)"))
    choice = set_language("pt_BR", env={}, available=("en", "pt_PT"), directory=directory)
    assert choice.notice == (
        "LlamaFit has no pt_BR translation, so it is using the Portuguese (Portugal) one."
    )


def test_a_catalog_that_holds_another_language_falls_back_to_english(tmp_path: Path) -> None:
    # The failure this branch exists to remove, in its last hiding place: before this,
    # current_language() said pt_PT and every message came out in German.
    directory = _catalog_dir(
        tmp_path,
        text=CATALOG.replace('"Language: pt_PT\\n"', '"Language: de\\n"'),
    )
    choice = set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=directory)
    assert choice.language == "en"
    assert current_language() == "en"
    assert choice.notice == "LlamaFit could not read its pt_PT translation, so it is using English."
    assert _("No GPU detected") == "No GPU detected"
