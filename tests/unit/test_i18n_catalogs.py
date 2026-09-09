"""The completeness check every shipped catalog has to pass.

A catalog must parse, must declare its plural rule, and must not carry a message the
template does not have: such a message is dead weight a translator wasted time on. A
message the catalog is *missing* is only a warning, because a translation in progress
must not break the build.
"""

import warnings
from pathlib import Path

import pytest
from scripts.gen_messages import extract, render_template

from llamafit import __version__
from llamafit.i18n import SOURCE_LANGUAGE, TEMPLATE_NAME
from llamafit.i18n.catalogs import available_languages, catalog_dir, catalog_path, load_language
from llamafit.i18n.po import parse_po, read_po
from llamafit.i18n.tags import normalise

LANGUAGES = [tag for tag in available_languages() if tag != SOURCE_LANGUAGE]


def _template_messages() -> dict[str, str | None]:
    found = extract()
    assert found.problems == [], found.problems
    return {entry.msgid: entry.plural for entry in found.entries}


def test_at_least_one_language_ships() -> None:
    assert LANGUAGES, "no .po catalogs are packaged"
    assert "pt_PT" in LANGUAGES


def test_every_catalog_file_is_named_for_its_language() -> None:
    for path in sorted(catalog_dir().glob("*.po")):
        assert normalise(path.stem) == path.stem, f"{path.name} is not a normalised language tag"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_catalog_parses(language: str) -> None:
    catalog = load_language(language)
    assert catalog.messages, f"{language} translates nothing"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_catalog_declares_its_language_and_plural_rule(language: str) -> None:
    catalog = load_language(language)
    assert catalog.plural_forms is not None, f"{language} declares no Plural-Forms"
    assert catalog.plural_rule.nplurals >= 1
    assert catalog.language == language, f"{language}.po says Language: {catalog.language!r}"
    assert catalog.headers.get("Content-Type") == "text/plain; charset=UTF-8"


@pytest.mark.parametrize("language", LANGUAGES)
def test_no_catalog_carries_a_message_the_template_does_not(language: str) -> None:
    template = _template_messages()
    catalog = load_language(language)
    extra = sorted(set(catalog.messages) - set(template))
    assert extra == [], f"{language} translates messages nothing asks for: {extra}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_plural_entry_matches_the_template_and_the_declared_form_count(
    language: str,
) -> None:
    template = _template_messages()
    catalog = load_language(language)
    for msgid, message in catalog.messages.items():
        assert message.plural == template[msgid], f"{language}: {msgid!r} has the wrong plural"
        if message.plural is not None and message.translated:
            assert len(message.translations) == catalog.plural_rule.nplurals, (
                f"{language}: {msgid!r} does not have {catalog.plural_rule.nplurals} forms"
            )


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_message_missing_from_a_catalog_is_a_warning_not_a_failure(language: str) -> None:
    template = _template_messages()
    catalog = load_language(language)
    missing = sorted(set(template) - set(catalog.messages)) + list(catalog.untranslated())
    if missing:
        warnings.warn(
            f"{language} still needs {len(missing)} message(s): {missing[0]!r}"
            + ("..." if len(missing) > 1 else ""),
            stacklevel=1,
        )
    assert True


def test_the_committed_template_is_up_to_date() -> None:
    path = catalog_dir() / TEMPLATE_NAME
    expected = render_template(extract(), version=__version__)
    assert path.read_text(encoding="utf-8") == expected, (
        "regenerate with: python scripts/gen_messages.py"
    )


def test_the_template_parses_as_a_catalog_that_translates_nothing() -> None:
    catalog = read_po(catalog_dir() / TEMPLATE_NAME)
    assert catalog.messages
    assert catalog.untranslated() == tuple(catalog.messages)
    assert catalog.plural_forms is not None


def test_the_template_falls_back_to_english_for_every_message() -> None:
    catalog = read_po(catalog_dir() / TEMPLATE_NAME)
    for msgid in catalog.messages:
        assert catalog.gettext(msgid) == msgid


def test_the_portuguese_catalog_is_european_not_brazilian() -> None:
    catalog = load_language("pt_PT")
    assert catalog.headers.get("Language-Team", "").startswith("Portuguese (Portugal)")
    # "detetada" is the European spelling; Brazilian Portuguese writes "detectada".
    assert catalog.gettext("No GPU detected") == "Nenhuma GPU detetada"


def test_the_packaged_directory_is_addressed_the_way_the_other_data_is() -> None:
    folder = catalog_dir()
    assert folder.is_dir()
    assert folder.name == "locale"
    assert (folder / "__init__.py").is_file()
    assert catalog_path("pt_PT") == folder / "pt_PT.po"


def test_english_is_always_available_even_with_no_catalogs(tmp_path: Path) -> None:
    assert available_languages(directory=tmp_path) == ("en",)
    assert available_languages(directory=tmp_path / "does-not-exist") == ("en",)


def test_a_file_that_is_not_a_language_tag_is_not_offered(tmp_path: Path) -> None:
    (tmp_path / "pt_PT.po").write_text('msgid "a"\nmsgstr "b"\n', encoding="utf-8")
    (tmp_path / "notes.po").write_text("", encoding="utf-8")
    assert available_languages(directory=tmp_path) == ("en", "pt_PT")


def test_loading_from_a_directory_a_test_chose(tmp_path: Path) -> None:
    (tmp_path / "pt_PT.po").write_text('msgid "a"\nmsgstr "A"\n', encoding="utf-8")
    assert load_language("pt_PT", directory=tmp_path).gettext("a") == "A"


def test_every_shipped_catalog_is_read_as_utf8_whatever_the_platform_default_is() -> None:
    # Decoded explicitly rather than through read_text, so a machine whose default
    # encoding is cp1252 cannot make this pass by accident.
    for path in [*sorted(catalog_dir().glob("*.po")), catalog_dir() / TEMPLATE_NAME]:
        text = path.read_bytes().decode("utf-8")
        assert parse_po(text, source=str(path)).messages is not None
