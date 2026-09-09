"""The completeness check every shipped catalog has to pass.

A catalog must parse, must declare its plural rule, must not carry a message the template
does not have — such a message is dead weight a translator wasted time on — and must fill
every placeholder its message does. A message the catalog is *missing* is only a warning,
because a translation in progress must not break the build.

The placeholder check itself lives in ``llamafit.i18n.po``, because a catalog somebody
writes themselves never comes near this file and still has to be safe to format. What is
here is the stricter half of the same rule: the reader drops a translation it cannot use
and carries on in English, and a catalog LlamaFit *ships* has to fail the build instead.
"""

import warnings
from pathlib import Path

import pytest
from scripts.gen_messages import extract, render_template

from llamafit import __version__
from llamafit.errors import ConfigError
from llamafit.i18n import SOURCE_LANGUAGE, TEMPLATE_NAME
from llamafit.i18n.catalogs import available_languages, catalog_dir, catalog_path, load_language
from llamafit.i18n.po import MessageKey, PoCatalog, parse_po, placeholders, read_po
from llamafit.i18n.tags import normalise

LANGUAGES = [tag for tag in available_languages() if tag != SOURCE_LANGUAGE]

# Words and spellings only Brazilian Portuguese uses, each with what European Portuguese
# writes instead. Enough of them to catch a catalog drifting, few enough that every one is
# unambiguous: none of these is also a European word, and none is a substring of one.
_BRAZILIAN_ONLY = {
    "detecta": "deteta",
    "detecção": "deteção",
    "arquivo": "ficheiro",
    "usuário": "utilizador",
    "gerenciar": "gerir",
    "gerenciamento": "gestão",
    "aplicativo": "aplicação",
    "cadastro": "registo",
    "registro": "registo",
    "baixar": "transferir",
    "econômic": "económic",
    "eletrônic": "eletrónic",
}


def brazilian_forms(catalog: PoCatalog) -> list[str]:
    """Report every Brazilian-only word or spelling a catalog's translations use.

    The Portuguese catalog says at the top that it wants a native speaker to go through
    it line by line, so no test may freeze one of its sentences. What has to hold whatever
    the words become is that they are European.
    """
    found: list[str] = []
    for message in catalog.messages.values():
        for translation in message.translations:
            lowered = translation.lower()
            for brazilian, european in _BRAZILIAN_ONLY.items():
                note = f"{brazilian!r} is Brazilian; European Portuguese writes {european!r}"
                if brazilian in lowered and note not in found:
                    found.append(note)
    return found


def _template_messages() -> dict[MessageKey, str | None]:
    found = extract()
    assert found.problems == [], found.problems
    return {(entry.context, entry.msgid): entry.plural for entry in found.entries}


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
    for key, message in catalog.messages.items():
        assert message.plural == template[key], f"{language}: {key} has the wrong plural"
        if message.plural is not None and message.translated:
            assert len(message.translations) == catalog.plural_rule.nplurals, (
                f"{language}: {key} does not have {catalog.plural_rule.nplurals} forms"
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
    for context, msgid in catalog.messages:
        rendered = catalog.gettext(msgid) if context is None else catalog.pgettext(context, msgid)
        assert rendered == msgid


def test_the_portuguese_catalog_is_european_not_brazilian() -> None:
    # This used to freeze one sentence word for word, which made the invitation at the
    # top of pt_PT.po — read it line by line and correct it — a way to break the build.
    # What the test is really guarding is the variety, so that is what it asserts, and
    # every sentence in the file is free to change.
    catalog = load_language("pt_PT")
    assert catalog.headers.get("Language-Team", "").startswith("Portuguese (Portugal)")
    assert catalog.language == "pt_PT"
    assert brazilian_forms(catalog) == []


def test_the_variety_check_would_catch_a_brazilian_sentence() -> None:
    # Otherwise the check above passes on any catalog at all, including an empty one.
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "No GPU detected"\nmsgstr "Nenhuma GPU detectada"\n\n'
        'msgid "file"\nmsgstr "arquivo"\n'
    )
    assert brazilian_forms(catalog) == [
        "'detecta' is Brazilian; European Portuguese writes 'deteta'",
        "'arquivo' is Brazilian; European Portuguese writes 'ficheiro'",
    ]


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
    (tmp_path / "pt_PT.po").write_text(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "a"\nmsgstr "A"\n',
        encoding="utf-8",
    )
    assert load_language("pt_PT", directory=tmp_path).gettext("a") == "A"


def test_every_shipped_catalog_is_read_as_utf8_whatever_the_platform_default_is() -> None:
    # Decoded explicitly rather than through read_text, so a machine whose default
    # encoding is cp1252 cannot make this pass by accident.
    for path in [*sorted(catalog_dir().glob("*.po")), catalog_dir() / TEMPLATE_NAME]:
        text = path.read_bytes().decode("utf-8")
        assert parse_po(text, source=str(path)).messages is not None


def test_the_two_contextual_entries_read_differently_in_portuguese() -> None:
    catalog = load_language("pt_PT")
    assert catalog.pgettext("GPU", "none detected") == "nenhuma detetada"
    assert catalog.pgettext("backends", "none detected") == "nenhum detetado"
    assert catalog.pgettext("memory bandwidth", "unknown") == "desconhecida"
    assert catalog.pgettext("bits per weight", "unknown") == "desconhecido"


def test_no_shipped_catalog_translates_a_word_two_rows_share_without_a_context() -> None:
    # These are the two the reviewer found: one msgid, two genders, and whichever
    # gender the catalog picked was wrong in the other row.
    for language in LANGUAGES:
        catalog = load_language(language)
        for msgid in ("none detected", "unknown"):
            assert (None, msgid) not in catalog.messages, (
                f"{language} translates {msgid!r} without saying which row it is for"
            )


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_shipped_catalog_has_nothing_the_reader_had_to_drop(language: str) -> None:
    # The reader drops a translation whose placeholders would not fill and carries on in
    # English, which is the right answer for a catalog a user wrote. A catalog we ship
    # has to fail the build instead: degrading quietly is what shipping it would be.
    catalog = load_language(language)
    assert list(catalog.problems) == []


def test_the_english_messages_agree_with_themselves_about_placeholders() -> None:
    # The template translates nothing, so this checks the messages, not a translation:
    # a plural pair naming the count in only one form is caught before anyone translates
    # it, and so is a literal percent sign somebody forgot to double.
    assert list(read_po(catalog_dir() / TEMPLATE_NAME).problems) == []


def _one_message(msgstr: str, msgid: str = '"%(count)d module"') -> PoCatalog:
    return parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        "msgid " + msgid + "\nmsgstr " + msgstr + "\n"
    )


@pytest.mark.parametrize(
    ("msgstr", "fragment"),
    [
        ('"%(total)d modulos"', "it fills ['total'] but the message names ['count']"),
        ('"%(count)d de %(all)d"', "it fills ['all', 'count'] but the message names"),
        ('"alguns modulos"', "it fills [] but the message names ['count']"),
        ('"%(count)d modulos a 100% de uso"', "is not a named placeholder"),
        ('"%s modulos"', "is not a named placeholder"),
    ],
)
def test_a_translation_that_breaks_a_placeholder_is_dropped_and_reported(
    msgstr: str, fragment: str
) -> None:
    catalog = _one_message(msgstr)
    assert len(catalog.problems) == 1, catalog.problems
    assert fragment in catalog.problems[0]
    assert "msgstr[0] was not used" in catalog.problems[0]
    # Dropped, so the English shows and the message counts as one the language needs.
    assert catalog.gettext("%(count)d module") == "%(count)d module"
    assert catalog.untranslated() == ((None, "%(count)d module"),)


def test_a_dropped_translation_never_reaches_a_formatted_sentence() -> None:
    # The positional case is the one that raises nothing at all: filled from a dict it
    # substitutes the dict's repr, so before the reader dropped it a user read
    # "{'count': 7} modulos" in the middle of a sentence.
    catalog = _one_message('"%s modulos"')
    assert catalog.gettext("%(count)d module") % {"count": 7} == "7 module"


def test_a_placeholder_broken_in_one_plural_form_only_costs_that_form_alone() -> None:
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "%(count)d module"\nmsgid_plural "%(count)d modules"\n'
        'msgstr[0] "%(count)d modulo"\nmsgstr[1] "varios modulos"\n'
    )
    assert len(catalog.problems) == 1, catalog.problems
    assert "msgstr[1] was not used" in catalog.problems[0]
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 1) == "%(count)d modulo"
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 5) == "%(count)d modules"


def test_an_english_plural_naming_the_count_in_one_form_only_takes_its_translations() -> None:
    # The English cannot be dropped, since it is what the caller passes in, so it is
    # reported and the build fails before any language is asked to translate it. The
    # translations go with it: which English form msgstr[0] stands in for is exactly what
    # a self-contradictory pair leaves unanswerable, and guessing is what this refuses.
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "one module"\nmsgid_plural "%(count)d modules"\nmsgstr[0] "um modulo"\n'
    )
    assert "both forms must name the same placeholders" in catalog.problems[0]
    assert "msgstr[0] was not used" in catalog.problems[1]
    assert catalog.gettext("one module") == "one module"


def test_a_doubled_percent_and_a_reordered_placeholder_are_both_fine() -> None:
    catalog = _one_message('"%(total)d, dos quais %(used)d%%"', msgid='"%(used)d%% of %(total)d"')
    assert catalog.problems == ()
    assert catalog.gettext("%(used)d%% of %(total)d") == "%(total)d, dos quais %(used)d%%"


def test_an_untranslated_form_is_not_asked_about_its_placeholders() -> None:
    # It falls back to English, so it has nothing to keep.
    assert _one_message('"   "').problems == ()


def test_the_placeholder_reader_names_what_it_found() -> None:
    assert placeholders("%(count)d of %(total)d") == (frozenset({"count", "total"}), ())
    assert placeholders("100%% sure") == (frozenset(), ())
    assert placeholders("%(pct).1f%% of %(n)05d") == (frozenset({"pct", "n"}), ())
    names, problems = placeholders("%s and 50% more")
    assert names == frozenset()
    assert len(problems) == 2


def test_every_shipped_translation_actually_formats() -> None:
    # The check above compares names; this one runs the formatting it is standing in for,
    # so a rule that passed the comparison but still raised would be caught here.
    for language in LANGUAGES:
        catalog = load_language(language)
        for key, message in catalog.messages.items():
            names, _ = placeholders(message.msgid)
            values = dict.fromkeys(names, 1)
            for translation in message.translations:
                if translation.strip():
                    assert isinstance(translation % values, str), key


def _catalog(tmp_path: Path, header: str, name: str = "pt_PT.po") -> Path:
    (tmp_path / name).write_text(
        'msgid ""\nmsgstr ""\n' + header + '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "No GPU detected"\nmsgstr "Keine GPU erkannt"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_a_catalog_whose_header_names_another_language_is_refused(tmp_path: Path) -> None:
    # Nothing downstream would notice: the tag comes from the file name, so this catalog
    # would be installed as pt_PT and would answer in German, and nothing would say so.
    directory = _catalog(tmp_path, '"Language: de\\n"\n')
    with pytest.raises(ConfigError) as caught:
        load_language("pt_PT", directory=directory)
    rendered = caught.value.render()
    assert "read as pt_PT" in rendered
    assert "'de'" in rendered
    assert "Rename the file to de.po" in rendered
    assert "set the header to pt_PT" in rendered


def test_a_catalog_that_declares_no_language_at_all_is_still_read(tmp_path: Path) -> None:
    # A translation in progress may not have filled the header in; saying nothing is not
    # the same as saying something false.
    directory = _catalog(tmp_path, "")
    assert load_language("pt_PT", directory=directory).gettext("No GPU detected") == (
        "Keine GPU erkannt"
    )


@pytest.mark.parametrize("header", ["pt-PT", "PT_pt", "pt_PT.UTF-8"])
def test_the_same_language_spelt_another_way_is_the_same_language(
    tmp_path: Path, header: str
) -> None:
    directory = _catalog(tmp_path, '"Language: ' + header + '\\n"\n')
    assert load_language("pt_PT", directory=directory).messages


def test_a_language_header_that_names_no_language_is_refused(tmp_path: Path) -> None:
    # Portuguese_Portugal is the name Windows reports and the alias table cannot resolve;
    # a header holding it declares nothing this can check against, so it is refused too.
    directory = _catalog(tmp_path, '"Language: Portuguese_Portugal\\n"\n')
    with pytest.raises(ConfigError, match="Portuguese_Portugal"):
        load_language("pt_PT", directory=directory)
