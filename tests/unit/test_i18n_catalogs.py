"""The completeness check every shipped catalog has to pass.

A catalog must parse, must declare its plural rule, must not carry a message the template
does not have — such a message is dead weight a translator wasted time on — and must fill
every placeholder its message does. A message the catalog is *missing* is only a warning,
because a translation in progress must not break the build.

The placeholder check is the one that stops a red build being a traceback in front of a
user instead. Every counted message goes through ``%`` formatting with a dictionary the
English message decided the shape of, so a renamed or an added placeholder is a
``KeyError``, an undoubled literal percent is a ``ValueError`` or a ``TypeError``, and a
dropped one raises nothing at all and silently loses the number from the sentence.
"""

import re
import warnings
from pathlib import Path

import pytest
from scripts.gen_messages import extract, render_template

from llamafit import __version__
from llamafit.i18n import SOURCE_LANGUAGE, TEMPLATE_NAME
from llamafit.i18n.catalogs import available_languages, catalog_dir, catalog_path, load_language
from llamafit.i18n.po import MessageKey, PoCatalog, parse_po, read_po
from llamafit.i18n.tags import normalise

LANGUAGES = [tag for tag in available_languages() if tag != SOURCE_LANGUAGE]

# A named %-placeholder: the name in brackets, then the flags, width, precision and
# length modifier %-formatting allows, then the conversion character.
_PLACEHOLDER_RE = re.compile(r"\((?P<name>[^)]*)\)[#0\- +]*(?:\d+)?(?:\.\d+)?[hlL]?(?P<kind>.)")
_CONVERSIONS = "diouxXeEfFgGcrsa%"


def _placeholders(text: str) -> tuple[set[str], list[str]]:
    """Return the placeholder names in ``text``, and every ``%`` that is not one.

    Only the named form is a placeholder here. A positional ``%s`` cannot be moved by a
    translator, and a lone ``%`` is a literal percent sign somebody forgot to double;
    both raise when the message is formatted, so both are reported rather than counted.
    """
    names: set[str] = set()
    problems: list[str] = []
    position = 0
    while (start := text.find("%", position)) != -1:
        if text.startswith("%%", start):
            position = start + 2
            continue
        found = _PLACEHOLDER_RE.match(text, start + 1)
        if found is None or found.group("kind") not in _CONVERSIONS:
            problems.append(
                f"{text[start : start + 12]!r} is not a named placeholder; "
                "write %% for a literal percent sign"
            )
            position = start + 1
            continue
        names.add(found.group("name"))
        position = found.end()
    return names, problems


def placeholder_problems(catalog: PoCatalog) -> list[str]:
    """Report every way a catalog's translations disagree with their messages.

    A plural entry's two English forms must name the same placeholders, so that what a
    translation has to carry does not depend on which form a count selects: a language
    draws those boundaries somewhere else, and one that carried the count in only one
    form could not be translated into it at all.
    """
    problems: list[str] = []
    for key, message in catalog.messages.items():
        english, bad = _placeholders(message.msgid)
        problems += [f"{key}: msgid: {problem}" for problem in bad]
        if message.plural is not None:
            plural, bad = _placeholders(message.plural)
            problems += [f"{key}: msgid_plural: {problem}" for problem in bad]
            if plural != english:
                problems.append(
                    f"{key}: the English singular has {sorted(english)} but the plural "
                    f"has {sorted(plural)}; both forms must name the same placeholders"
                )
                english |= plural
        for index, translation in enumerate(message.translations):
            if not translation.strip():
                continue  # untranslated, so English is what the user sees
            names, bad = _placeholders(translation)
            problems += [f"{key}: msgstr[{index}]: {problem}" for problem in bad]
            if names != english:
                problems.append(
                    f"{key}: msgstr[{index}] has {sorted(names)} but the message has "
                    f"{sorted(english)}"
                )
    return problems


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
def test_every_translation_fills_the_placeholders_its_message_does(language: str) -> None:
    problems = placeholder_problems(load_language(language))
    assert problems == [], f"{language}: {problems}"


def test_the_english_messages_agree_with_themselves_about_placeholders() -> None:
    # The template translates nothing, so this checks the messages, not a translation:
    # a plural pair naming the count in only one form is caught before anyone translates
    # it, and so is a literal percent sign somebody forgot to double.
    assert placeholder_problems(read_po(catalog_dir() / TEMPLATE_NAME)) == []


@pytest.mark.parametrize(
    ("msgstr", "fragment"),
    [
        ('"%(total)d modulos"', "['total'] but the message has ['count']"),  # renamed
        ('"%(count)d de %(all)d"', "['all', 'count'] but the message"),  # an extra one
        ('"alguns modulos"', "[] but the message has ['count']"),  # dropped: silent
        ('"%(count)d modulos a 100% de uso"', "is not a named placeholder"),  # undoubled
    ],
)
def test_a_translation_that_breaks_a_placeholder_is_reported(msgstr: str, fragment: str) -> None:
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "%(count)d module"\nmsgstr ' + msgstr + "\n"
    )
    problems = placeholder_problems(catalog)
    assert len(problems) == 1, problems
    assert fragment in problems[0]


def test_a_placeholder_broken_in_one_plural_form_only_is_still_reported() -> None:
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "%(count)d module"\nmsgid_plural "%(count)d modules"\n'
        'msgstr[0] "%(count)d modulo"\nmsgstr[1] "varios modulos"\n'
    )
    problems = placeholder_problems(catalog)
    assert len(problems) == 1, problems
    assert "msgstr[1] has [] but the message has ['count']" in problems[0]


def test_an_english_plural_that_names_the_count_in_one_form_only_is_reported() -> None:
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "one module"\nmsgid_plural "%(count)d modules"\nmsgstr[0] "um modulo"\n'
    )
    assert "both forms must name the same placeholders" in placeholder_problems(catalog)[0]


def test_a_doubled_percent_and_a_reordered_placeholder_are_both_fine() -> None:
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "%(used)d%% of %(total)d"\nmsgstr "%(total)d, dos quais %(used)d%%"\n'
    )
    assert placeholder_problems(catalog) == []


def test_an_untranslated_form_is_not_asked_about_its_placeholders() -> None:
    # It falls back to English, so it has nothing to keep.
    catalog = parse_po(
        'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "%(count)d module"\nmsgstr "   "\n'
    )
    assert placeholder_problems(catalog) == []


def test_every_shipped_translation_actually_formats() -> None:
    # The check above compares names; this one runs the formatting it is standing in for,
    # so a rule that passed the comparison but still raised would be caught here.
    for language in LANGUAGES:
        catalog = load_language(language)
        for key, message in catalog.messages.items():
            names, _ = _placeholders(message.msgid)
            values = dict.fromkeys(names, 1)
            for translation in message.translations:
                if translation.strip():
                    assert isinstance(translation % values, str), key
