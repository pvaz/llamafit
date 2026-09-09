"""The .po reader, exercised on the awkward files rather than the easy one."""

from pathlib import Path

import pytest

from llamafit.errors import ConfigError
from llamafit.i18n.po import PoSyntaxError, parse_po, read_po

HEADER = """
msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"
"Language: pt_PT\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"
"""

HEADER_BYTES = b"\xef\xbb\xbf" + HEADER.encode()
"""HEADER as a Windows editor saves it: a byte order mark, then the same text."""

POLISH_HEADER = """
msgid ""
msgstr ""
"Language: pl\\n"
"Plural-Forms: nplurals=3; plural=(n==1 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 "
"|| n%100>=20) ? 1 : 2);\\n"
"""


def test_header_is_parsed_and_is_not_a_message() -> None:
    catalog = parse_po(HEADER)
    assert catalog.language == "pt_PT"
    assert catalog.headers["Content-Type"] == "text/plain; charset=UTF-8"
    assert catalog.messages == {}
    assert catalog.plural_forms == "nplurals=2; plural=(n != 1);"


def test_a_string_split_over_several_lines_is_one_message() -> None:
    catalog = parse_po(
        HEADER
        + """
msgid ""
"LlamaFit could not read this machine's memory totals, "
"so it cannot work out what will fit."
msgstr ""
"O LlamaFit nao conseguiu ler os totais de memoria desta maquina, "
"por isso nao consegue calcular o que cabe."
"""
    )
    key = (
        "LlamaFit could not read this machine's memory totals, so it cannot work out what will fit."
    )
    assert catalog.gettext(key).startswith("O LlamaFit nao conseguiu")
    assert catalog.gettext(key).endswith("o que cabe.")


def test_embedded_quotes_tabs_and_newlines_survive() -> None:
    catalog = parse_po(HEADER + '\nmsgid "a\\"b\\tc\\nd"\nmsgstr "w\\"x\\ty\\nz"\n')
    assert catalog.gettext('a"b\tc\nd') == 'w"x\ty\nz'


def test_an_empty_translation_falls_back_to_english_not_to_a_blank_line() -> None:
    catalog = parse_po(HEADER + '\nmsgid "Skip the measurement."\nmsgstr ""\n')
    assert catalog.gettext("Skip the measurement.") == "Skip the measurement."
    assert catalog.untranslated() == ((None, "Skip the measurement."),)


def test_a_message_the_catalog_never_heard_of_comes_back_unchanged() -> None:
    catalog = parse_po(HEADER)
    assert catalog.gettext("No GPU detected") == "No GPU detected"


def test_comments_and_obsolete_entries_are_ignored() -> None:
    catalog = parse_po(
        HEADER
        + """
# a translator's note
#. an extracted comment
#: llamafit/cli/render.py:41
#, python-format
msgid "unknown"
msgstr "desconhecido"

#~ msgid "a message nobody uses any more"
#~ msgstr "uma mensagem que ja ninguem usa"
"""
    )
    assert catalog.gettext("unknown") == "desconhecido"
    assert "a message nobody uses any more" not in catalog.messages


def test_three_plural_forms_pick_the_right_one() -> None:
    catalog = parse_po(
        POLISH_HEADER
        + """
msgid "%(count)d file"
msgid_plural "%(count)d files"
msgstr[0] "%(count)d plik"
msgstr[1] "%(count)d pliki"
msgstr[2] "%(count)d plikow"
"""
    )
    assert catalog.plural_rule.nplurals == 3
    assert catalog.ngettext("%(count)d file", "%(count)d files", 1) == "%(count)d plik"
    assert catalog.ngettext("%(count)d file", "%(count)d files", 3) == "%(count)d pliki"
    assert catalog.ngettext("%(count)d file", "%(count)d files", 5) == "%(count)d plikow"


def test_a_plural_rule_that_differs_from_english_is_obeyed() -> None:
    # French puts zero in the singular form, which English does not.
    french = """
msgid ""
msgstr ""
"Language: fr\\n"
"Plural-Forms: nplurals=2; plural=(n > 1);\\n"

msgid "%(count)d model"
msgid_plural "%(count)d models"
msgstr[0] "%(count)d modele"
msgstr[1] "%(count)d modeles"
"""
    catalog = parse_po(french)
    assert catalog.ngettext("%(count)d model", "%(count)d models", 0) == "%(count)d modele"
    assert catalog.ngettext("%(count)d model", "%(count)d models", 1) == "%(count)d modele"
    assert catalog.ngettext("%(count)d model", "%(count)d models", 2) == "%(count)d modeles"


def test_an_untranslated_plural_falls_back_to_the_english_form() -> None:
    catalog = parse_po(
        HEADER
        + """
msgid "%(count)d module"
msgid_plural "%(count)d modules"
msgstr[0] ""
msgstr[1] ""
"""
    )
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 1) == "%(count)d module"
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 7) == "%(count)d modules"


def test_a_plural_form_the_catalog_omits_falls_back_to_english() -> None:
    catalog = parse_po(
        POLISH_HEADER
        + """
msgid "%(count)d file"
msgid_plural "%(count)d files"
msgstr[0] "%(count)d plik"
"""
    )
    assert catalog.ngettext("%(count)d file", "%(count)d files", 1) == "%(count)d plik"
    assert catalog.ngettext("%(count)d file", "%(count)d files", 5) == "%(count)d files"


def test_a_catalog_without_plural_forms_still_works_on_english_rules() -> None:
    catalog = parse_po('msgid ""\nmsgstr ""\n"Language: pt_PT\\n"\n')
    assert catalog.plural_forms is None
    assert catalog.plural_rule.nplurals == 2
    assert catalog.plural_rule.index(1) == 0
    assert catalog.plural_rule.index(2) == 1


def test_entries_do_not_need_a_blank_line_between_them() -> None:
    catalog = parse_po('msgid "a"\nmsgstr "A"\nmsgid "b"\nmsgstr "B"\n')
    assert (catalog.gettext("a"), catalog.gettext("b")) == ("A", "B")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('msgid "a"\nmsgstr "b\n', "expected a double-quoted string"),
        ('"orphan"\n', "a string continues nothing"),
        ('msgctxt "menu"\nmsgctxt "file"\nmsgid "Open"\n', "already has a msgctxt"),
        (
            'msgid "Open"\nmsgctxt "menu"\nmsgstr "Abrir"\n',
            "a msgctxt must come before its msgid",
        ),
        (
            'msgctxt "menu"\nmsgid "Open"\nmsgstr "A"\n\n'
            'msgctxt "menu"\nmsgid "Open"\nmsgstr "B"\n',
            "duplicate message 'Open' in context 'menu'",
        ),
        ('msgid\nmsgstr "b"\n', "expected a string after msgid"),
        ('msgid "a"\nmsgid "b"\nmsgstr "c"\n', "already has a msgid"),
        ('msgid "a"\nmsgstr "A"\n\nmsgid "a"\nmsgstr "B"\n', "duplicate message 'a'"),
        ('msgstr "A"\n', "an entry has a translation but no msgid"),
        ('msgid "a"\nmsgstr "b\\q"\n', "unknown escape \\q"),
        ('msgid "a"\nmsgstr "b\\"\n', "ends with a lone backslash"),
        ('msgid "a"\nmsgstr "b" "c"\n', "an unescaped quote ends the string early"),
        ('msgid "a"\nmsgstr[12] "b"\n', "plural form 12 is out of range"),
        ('msgid "a"\nmsgstr[0] "b"\nmsgstr[0] "c"\n', "already has msgstr[0]"),
        (
            'msgid ""\nmsgstr ""\n"Plural-Forms: nonsense\\n"\n',
            "not a Plural-Forms header",
        ),
        (
            'msgid ""\nmsgstr ""\n"Plural-Forms: nplurals=2; plural=__import__(\'os\');\\n"\n',
            "invalid plural expression",
        ),
        (
            'msgid ""\nmsgstr ""\n"Language: a\\n"\n\nmsgid ""\nmsgstr ""\n"Language: b\\n"\n',
            "two header entries",
        ),
    ],
)
def test_a_malformed_file_names_the_line_and_the_problem(text: str, fragment: str) -> None:
    with pytest.raises(PoSyntaxError) as caught:
        parse_po(text, source="broken.po")
    assert fragment in str(caught.value)
    assert caught.value.source == "broken.po"
    assert caught.value.line >= 1
    assert "broken.po: line" in caught.value.render()


def test_a_syntax_error_is_a_llamafit_error_the_cli_can_render() -> None:
    with pytest.raises(ConfigError) as caught:
        parse_po('"orphan"\n')
    assert "Hint:" in caught.value.render()


def test_reading_a_file_uses_utf8_whatever_the_platform_default_is(tmp_path: Path) -> None:
    path = tmp_path / "pt_PT.po"
    path.write_bytes(
        b'msgid ""\nmsgstr ""\n"Language: pt_PT\\n"\n\n'
        b'msgid "No GPU detected"\nmsgstr "Nenhuma GPU detetada"\n'
    )
    assert read_po(path).gettext("No GPU detected") == "Nenhuma GPU detetada"


def test_a_file_that_is_not_utf8_is_reported_not_mangled(tmp_path: Path) -> None:
    path = tmp_path / "broken.po"
    path.write_bytes(b'msgid "a"\nmsgstr "\xe9 latin-1"\n')
    with pytest.raises(PoSyntaxError, match="not valid UTF-8"):
        read_po(path)


def test_the_line_number_points_at_the_offending_line() -> None:
    with pytest.raises(PoSyntaxError) as caught:
        parse_po('msgid "a"\nmsgstr "A"\n\nmsgid "b"\nmsgstr "b\\z"\n')
    assert caught.value.line == 5


def test_a_second_msgid_plural_in_one_entry_is_refused() -> None:
    text = 'msgid "a"\nmsgid_plural "as"\nmsgid_plural "aes"\nmsgstr[0] "A"\n'
    with pytest.raises(PoSyntaxError, match="already has a msgid_plural"):
        parse_po(text)


def test_an_entry_with_no_translation_at_all_is_untranslated() -> None:
    catalog = parse_po(HEADER + '\nmsgid "%(count)d file"\nmsgid_plural "%(count)d files"\n')
    entry = catalog.messages[None, "%(count)d file"]
    assert entry.translations == ()
    assert not entry.translated
    assert catalog.gettext("%(count)d file") == "%(count)d file"
    assert catalog.ngettext("%(count)d file", "%(count)d files", 3) == "%(count)d files"


def test_a_header_line_without_a_colon_is_ignored() -> None:
    catalog = parse_po('msgid ""\nmsgstr ""\n"not a header line\\n"\n"Language: pt_PT\\n"\n')
    assert catalog.language == "pt_PT"
    assert "not a header line" not in catalog.headers


def test_a_context_and_no_context_are_two_entries_translated_differently() -> None:
    catalog = parse_po(
        HEADER
        + """
msgid "none detected"
msgstr "nenhum"

msgctxt "GPU"
msgid "none detected"
msgstr "nenhuma detetada"

msgctxt "backends"
msgid "none detected"
msgstr "nenhum detetado"
"""
    )
    assert catalog.pgettext("GPU", "none detected") == "nenhuma detetada"
    assert catalog.pgettext("backends", "none detected") == "nenhum detetado"
    assert catalog.gettext("none detected") == "nenhum"
    assert len(catalog.messages) == 3


def test_a_context_nobody_wrote_falls_back_to_english_rather_than_the_plain_entry() -> None:
    # The plain entry is a different sentence, not a looser version of this one.
    catalog = parse_po(HEADER + '\nmsgid "unknown"\nmsgstr "desconhecido"\n')
    assert catalog.gettext("unknown") == "desconhecido"
    assert catalog.pgettext("memory bandwidth", "unknown") == "unknown"


def test_a_message_with_no_context_is_not_found_under_an_empty_one() -> None:
    catalog = parse_po(HEADER + '\nmsgid "unknown"\nmsgstr "desconhecido"\n')
    assert catalog.messages[None, "unknown"].context is None
    assert catalog.pgettext("", "unknown") == "unknown"


def test_an_empty_msgctxt_is_a_context_of_its_own_not_the_absence_of_one() -> None:
    catalog = parse_po(
        HEADER
        + """
msgid "unknown"
msgstr "sem contexto"

msgctxt ""
msgid "unknown"
msgstr "contexto vazio"
"""
    )
    assert catalog.gettext("unknown") == "sem contexto"
    assert catalog.pgettext("", "unknown") == "contexto vazio"


def test_a_contextual_plural_picks_the_form_its_count_selects() -> None:
    catalog = parse_po(
        HEADER
        + """
msgctxt "GPU"
msgid "%(count)d device"
msgid_plural "%(count)d devices"
msgstr[0] "%(count)d placa"
msgstr[1] "%(count)d placas"
"""
    )
    assert catalog.npgettext("GPU", "%(count)d device", "%(count)d devices", 1) == "%(count)d placa"
    assert (
        catalog.npgettext("GPU", "%(count)d device", "%(count)d devices", 4) == "%(count)d placas"
    )
    assert catalog.ngettext("%(count)d device", "%(count)d devices", 4) == "%(count)d devices"
    assert (
        catalog.npgettext("CPU", "%(count)d device", "%(count)d devices", 1) == "%(count)d device"
    )


def test_a_msgctxt_starts_a_new_entry_with_no_blank_line_before_it() -> None:
    catalog = parse_po(
        HEADER + '\nmsgctxt "a"\nmsgid "x"\nmsgstr "A"\nmsgctxt "b"\nmsgid "x"\nmsgstr "B"\n'
    )
    assert (catalog.pgettext("a", "x"), catalog.pgettext("b", "x")) == ("A", "B")


def test_a_context_split_over_adjacent_lines_is_one_context() -> None:
    catalog = parse_po(
        HEADER + '\nmsgctxt ""\n"memory "\n"bandwidth"\nmsgid "unknown"\nmsgstr "desconhecida"\n'
    )
    assert catalog.pgettext("memory bandwidth", "unknown") == "desconhecida"


def test_an_entry_carries_the_context_it_was_filed_under() -> None:
    catalog = parse_po(HEADER + '\nmsgctxt "GPU"\nmsgid "none detected"\nmsgstr "nenhuma"\n')
    entry = catalog.messages["GPU", "none detected"]
    assert (entry.context, entry.msgid) == ("GPU", "none detected")
    assert entry.key == ("GPU", "none detected")


def test_an_empty_msgid_with_a_context_is_a_message_not_the_header() -> None:
    # The header is the entry with an empty msgid and nothing else; a msgctxt makes it
    # an ordinary entry, so a catalog whose header grew one has no header at all.
    catalog = parse_po(HEADER + '\nmsgctxt "odd"\nmsgid ""\nmsgstr "x"\n')
    assert catalog.language == "pt_PT"
    assert catalog.pgettext("odd", "") == "x"


def test_a_translation_that_is_only_whitespace_counts_as_no_translation() -> None:
    # A space, a tab, a newline and a non-breaking space: all of them reach a screen as
    # a blank line, and all of them look like a finished translation in the file.
    catalog = parse_po(HEADER + '\nmsgid "unknown"\nmsgstr " \\t\\n\u00a0"\n')
    assert catalog.gettext("unknown") == "unknown"
    assert catalog.untranslated() == ((None, "unknown"),)
    assert not catalog.messages[None, "unknown"].translated


def test_a_blank_plural_form_falls_back_like_an_empty_one() -> None:
    catalog = parse_po(
        HEADER
        + """
msgid "%(count)d module"
msgid_plural "%(count)d modules"
msgstr[0] "%(count)d modulo"
msgstr[1] "   "
"""
    )
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 1) == "%(count)d modulo"
    assert catalog.ngettext("%(count)d module", "%(count)d modules", 7) == "%(count)d modules"


def test_a_translation_that_is_only_partly_whitespace_is_kept_exactly() -> None:
    # Only a translation that is nothing but whitespace is absent. One that ends in a
    # space is a translation, and the space is the translator's, so it survives.
    catalog = parse_po(HEADER + '\nmsgid "Language: "\nmsgstr "Idioma: "\n')
    assert catalog.gettext("Language: ") == "Idioma: "
    assert catalog.untranslated() == ()


def test_a_catalog_saved_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    # What Notepad and a good few other Windows editors write by default. Refusing it
    # reported an unknown keyword whose name held an invisible character.
    path = tmp_path / "pt_PT.po"
    path.write_bytes(
        b"" + HEADER_BYTES + b'\nmsgid "No GPU detected"\nmsgstr "Nenhuma GPU detetada"\n'
    )
    assert read_po(path).gettext("No GPU detected") == "Nenhuma GPU detetada"


def test_a_catalog_saved_without_one_reads_exactly_the_same(tmp_path: Path) -> None:
    marked, plain = tmp_path / "a.po", tmp_path / "b.po"
    body = HEADER_BYTES[3:] + b'\nmsgid "No GPU detected"\nmsgstr "Nenhuma GPU detetada"\n'
    marked.write_bytes(b"\xef\xbb\xbf" + body)
    plain.write_bytes(body)
    assert read_po(marked).messages.keys() == read_po(plain).messages.keys()
    assert read_po(marked).headers == read_po(plain).headers


def test_parsed_text_that_still_carries_a_mark_is_read_too() -> None:
    # read_po strips it through the encoding; a caller that decoded the bytes itself, as
    # the packaging test does, hands the mark straight to the parser.
    catalog = parse_po("\ufeff" + HEADER + '\nmsgid "unknown"\nmsgstr "desconhecido"\n')
    assert catalog.gettext("unknown") == "desconhecido"
    assert catalog.language == "pt_PT"
