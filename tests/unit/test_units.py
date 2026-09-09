import pytest

from llamafit.units import format_bytes, gib, parse_size


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8G", 8_000_000_000),
        ("8GB", 8_000_000_000),
        ("8GiB", 8 * 1024**3),
        ("7.5GiB", int(7.5 * 1024**3)),
        ("512M", 512_000_000),
        ("512MiB", 512 * 1024**2),
        ("1T", 1_000_000_000_000),
        ("1024", 1024),
        (" 16 gb ", 16_000_000_000),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "8X", "-1G", "1.2.3G"])
def test_parse_size_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_size(bad)


def test_format_bytes_binary_and_decimal() -> None:
    assert format_bytes(8 * 1024**3) == "8.0 GiB"
    assert format_bytes(8 * 1024**3, binary=False) == "8.6 GB"
    assert format_bytes(1536 * 1024**2) == "1.5 GiB"
    assert format_bytes(900 * 1024) == "900.0 KiB"
    assert format_bytes(12) == "12 B"
    assert format_bytes(None) == "unknown"


def test_gib() -> None:
    assert gib(1024**3) == 1.0
    assert gib(None) is None


# --- numbers written the way a language writes them ---------------------------------


def _speaking(group: str, decimal: str, billions: str = "B") -> None:
    from llamafit.i18n import translator
    from llamafit.i18n.po import parse_po

    lines = [
        'msgid ""',
        'msgstr ""',
        '"Plural-Forms: nplurals=2; plural=(n != 1);' + chr(92) + 'n"',
        "",
        'msgctxt "thousands separator"',
        'msgid ","',
        'msgstr "' + group + '"',
        "",
        'msgctxt "decimal separator"',
        'msgid "."',
        'msgstr "' + decimal + '"',
        "",
        'msgctxt "parameter count"',
        'msgid "B"',
        'msgstr "' + billions + '"',
    ]
    catalog = parse_po("\n".join(lines) + "\n")
    assert catalog.problems == (), catalog.problems
    translator.set_translator(translator.CatalogTranslator("xx", catalog))


def test_english_numbers_are_left_exactly_as_they_were() -> None:
    from llamafit.units import format_bytes, format_grouped, localise_number

    assert localise_number("32,768") == "32,768"
    assert format_grouped(32768) == "32,768"
    assert format_bytes(137438953472) == "128.0 GiB"


def test_a_language_that_swaps_both_separators_gets_both_swapped() -> None:
    # The bug this exists for: 32,768 read by a Portuguese or German speaker is not a
    # foreign-looking number, it is thirty-two tokens and a bit.
    from llamafit.units import format_bytes, format_grouped, localise_number

    _speaking(group=".", decimal=",")
    assert format_grouped(32768) == "32.768"
    assert format_bytes(137438953472) == "128,0 GiB"
    assert localise_number("1,234.56") == "1.234,56"


def test_the_two_separators_are_swapped_in_one_pass_not_one_after_the_other() -> None:
    # Swapping "," then "." would turn 1,234.56 into 1.234.56: the group separator it had
    # just written would be read again as a decimal point.
    from llamafit.units import localise_number

    _speaking(group=".", decimal=",")
    assert localise_number("1,234.56").count(",") == 1


def test_a_language_that_fills_in_nothing_falls_back_to_english_not_to_blank() -> None:
    from llamafit.units import format_grouped

    _speaking(group="", decimal="")
    assert format_grouped(32768) == "32,768"


def test_a_language_that_groups_with_a_space_cannot_say_so_yet() -> None:
    # Recorded, not accepted. A msgstr holding only whitespace counts as untranslated, by
    # a rule that is right for prose and wrong for punctuation, so French, Russian,
    # Swedish, Polish and every other language that groups digits with a space silently
    # gets the English comma. The fix belongs in llamafit.i18n.po, which is what decides a
    # translation is blank; this test is here so the limitation is in the suite rather
    # than only in a report, and it should be inverted when that lands.
    from llamafit.units import format_grouped

    _speaking(group=chr(160), decimal=",")
    assert format_grouped(32768) == "32,768", "if this now says 32 768, invert the test"


def test_the_billions_suffix_is_taken_from_the_catalog() -> None:
    from llamafit.cli.render import _fmt_params

    assert _fmt_params(27, 27) == "27B"
    _speaking(group=".", decimal=",", billions="MM")
    assert _fmt_params(27, 27) == "27MM"
    assert _fmt_params(80, 3) == "80/3MM"
