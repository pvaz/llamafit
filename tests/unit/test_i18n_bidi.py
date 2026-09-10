"""The directional marks the renderer emits, asserted character by character.

Nobody here can look at a terminal, so nothing in this file claims anything about how a
line is drawn. What it asserts is what the standard prescribes and where: which codepoint,
in which position, around which run, and — just as important — that a left-to-right
language gets a string that is byte for byte the one it got before any of this existed.

The marks are spelled out as escapes rather than written as themselves. An invisible
character in a test is a test nobody can review.
"""

from collections.abc import Iterator

import pytest

from llamafit.i18n import bidi, translator
from llamafit.i18n.bidi import (
    FIRST_STRONG_ISOLATE as FSI,
)
from llamafit.i18n.bidi import (
    POP_DIRECTIONAL_ISOLATE as PDI,
)
from llamafit.i18n.bidi import (
    RIGHT_TO_LEFT_MARK as RLM,
)
from llamafit.i18n.bidi import (
    for_display,
    is_rtl,
    isolate,
    isolate_identifiers,
    mirror_justify,
    reading_order,
)

# Short Arabic words, letters only, so nothing in this file depends on Arabic punctuation:
# "test" and "the file".
ARABIC = "\u0627\u062e\u062a\u0628\u0627\u0631"
ARABIC_TWO = "\u0627\u0644\u0645\u0644\u0641"


@pytest.fixture
def arabic() -> Iterator[None]:
    """Speak Arabic for the duration of one test."""
    translator.set_translator(_FakeArabic())
    yield
    translator.reset()


class _FakeArabic:
    """A translator that speaks Arabic and translates nothing.

    The marks are decided by the language tag alone, so a real catalog would only make
    these tests depend on wording somebody is still free to change.
    """

    @property
    def language(self) -> str:
        return "ar"

    def gettext(self, message: str) -> str:
        return message

    def pgettext(self, context: str, message: str) -> str:
        return message

    def pgettext_literal(self, context: str, message: str) -> str:
        return message

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        return singular if n == 1 else plural

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        return singular if n == 1 else plural


def test_the_characters_are_the_ones_the_standard_names() -> None:
    assert FSI == "\u2068"
    assert PDI == "\u2069"
    assert RLM == "\u200f"


def test_the_three_shipped_right_to_left_catalogs_are_recognised() -> None:
    assert is_rtl("ar")
    assert is_rtl("he")
    assert is_rtl("ur")


def test_a_region_or_a_codeset_does_not_hide_the_language() -> None:
    assert is_rtl("ar_EG")
    assert is_rtl("he-IL.UTF-8")
    assert is_rtl("AR")


def test_a_left_to_right_language_is_not_right_to_left() -> None:
    for tag in ("en", "pt_PT", "ja", "zh_CN", "hi", "th", ""):
        assert not is_rtl(tag), tag


def test_is_rtl_asks_the_installed_language_when_it_is_given_none() -> None:
    assert not is_rtl()  # the suite speaks English unless a test says otherwise


def test_is_rtl_reads_the_installed_language(arabic: None) -> None:
    assert is_rtl()


# --- isolate: one value, one island ----------------------------------------------


def test_isolate_changes_nothing_for_a_left_to_right_language() -> None:
    for value in ("--verbose", "nvidia-smi", "D:\\models", "7.6 GiB", "", 27):
        assert isolate(value) == str(value)


def test_isolate_wraps_a_value_in_the_two_isolate_characters(arabic: None) -> None:
    assert isolate("--verbose") == f"{FSI}--verbose{PDI}"
    assert isolate("llama-server") == f"{FSI}llama-server{PDI}"
    assert isolate("7.6 GiB") == f"{FSI}7.6 GiB{PDI}"


def test_isolate_leaves_an_empty_value_empty(arabic: None) -> None:
    assert isolate("") == ""


def test_isolate_accepts_anything_that_can_be_a_string(arabic: None) -> None:
    assert isolate(27) == f"{FSI}27{PDI}"


# --- isolate_identifiers: what a translator had to keep verbatim ------------------


def test_no_identifier_is_marked_for_a_left_to_right_language() -> None:
    sentence = "Run again with --verbose for the full traceback."
    assert isolate_identifiers(sentence) == sentence
    assert for_display(sentence) == sentence


def test_a_long_option_inside_a_sentence_becomes_an_island(arabic: None) -> None:
    assert isolate_identifiers(f"{ARABIC} --verbose") == f"{ARABIC} {FSI}--verbose{PDI}"


def test_a_short_option_is_an_island_too(arabic: None) -> None:
    assert isolate_identifiers(f"{ARABIC} -v") == f"{ARABIC} {FSI}-v{PDI}"


def test_a_backticked_command_is_isolated_whole_backticks_included(arabic: None) -> None:
    text = f"{ARABIC} `pip install --force-reinstall psutil`."
    assert isolate_identifiers(text) == (
        f"{ARABIC} {FSI}`pip install --force-reinstall psutil`{PDI}."
    )


def test_a_url_is_isolated_without_the_full_stop_that_ends_the_sentence(arabic: None) -> None:
    text = f"{ARABIC} https://github.com/pvaz/llamafit."
    assert isolate_identifiers(text) == f"{ARABIC} {FSI}https://github.com/pvaz/llamafit{PDI}."


def test_a_hyphen_between_two_latin_letters_is_left_alone(arabic: None) -> None:
    # `nvidia-smi` resolves as Latin on both sides of its hyphen, so it is not at risk
    # and marking it would only put characters on a screen for nothing.
    assert isolate_identifiers(f"{ARABIC} nvidia-smi") == f"{ARABIC} nvidia-smi"


def test_a_value_already_isolated_is_not_isolated_a_second_time(arabic: None) -> None:
    text = f"{ARABIC} {FSI}llama-server --version{PDI} {ARABIC_TWO}"
    assert isolate_identifiers(text) == text


def test_an_identifier_after_an_isolate_is_still_found(arabic: None) -> None:
    text = f"{FSI}nvidia-smi{PDI} {ARABIC} --verbose"
    assert isolate_identifiers(text) == f"{FSI}nvidia-smi{PDI} {ARABIC} {FSI}--verbose{PDI}"


def test_an_unclosed_isolate_swallows_the_rest_rather_than_being_marked_twice(
    arabic: None,
) -> None:
    text = f"{ARABIC} {FSI}--verbose"
    assert isolate_identifiers(text) == text


def test_isolates_inside_isolates_are_counted_rather_than_closed_early(arabic: None) -> None:
    """A nested island is legal, and its inner PDI must not reopen the outer text."""
    text = f"{ARABIC} {FSI}a {FSI}--inner{PDI} b{PDI} --verbose"
    assert isolate_identifiers(text) == (
        f"{ARABIC} {FSI}a {FSI}--inner{PDI} b{PDI} {FSI}--verbose{PDI}"
    )


def test_a_stray_pop_outside_any_isolate_is_left_where_it_is(arabic: None) -> None:
    text = f"{ARABIC}{PDI} --verbose"
    assert isolate_identifiers(text) == f"{ARABIC}{PDI} {FSI}--verbose{PDI}"


def test_isolate_identifiers_leaves_empty_text_alone(arabic: None) -> None:
    assert isolate_identifiers("") == ""


# --- for_display: the base direction of a finished line ---------------------------


def test_a_line_with_arabic_in_it_is_opened_with_a_right_to_left_mark(arabic: None) -> None:
    assert for_display(f"llama-server {ARABIC}") == f"{RLM}llama-server {ARABIC}"


def test_a_line_of_nothing_but_latin_gets_no_mark(arabic: None) -> None:
    assert for_display("qwen3-coder-next") == "qwen3-coder-next"
    assert for_display("Q4_K_M") == "Q4_K_M"


def test_the_mark_is_not_added_twice(arabic: None) -> None:
    once = for_display(ARABIC)
    assert once == f"{RLM}{ARABIC}"
    assert for_display(once) == once


def test_every_line_of_a_multi_line_cell_is_directed_on_its_own(arabic: None) -> None:
    text = f"{ARABIC}\nQ4_K_M\n{ARABIC_TWO} --verbose"
    assert for_display(text) == (f"{RLM}{ARABIC}\nQ4_K_M\n{RLM}{ARABIC_TWO} {FSI}--verbose{PDI}")


def test_for_display_leaves_empty_text_alone(arabic: None) -> None:
    assert for_display("") == ""


def test_the_marks_are_the_only_thing_for_display_adds(arabic: None) -> None:
    """Stripping every mark must give back exactly what went in."""
    text = f"{ARABIC} `pip install --force-reinstall psutil` https://example.com/a."
    marked = for_display(text)
    assert marked.replace(RLM, "").replace(FSI, "").replace(PDI, "") == text


# --- the table's own direction -----------------------------------------------------


def test_columns_keep_their_order_for_a_left_to_right_language() -> None:
    assert reading_order(["id", "quality", "params"]) == ["id", "quality", "params"]
    assert mirror_justify("left") == "left"
    assert mirror_justify("right") == "right"


def test_columns_are_reversed_and_alignment_mirrored_for_arabic(arabic: None) -> None:
    assert reading_order(["id", "quality", "params"]) == ["params", "quality", "id"]
    assert mirror_justify("left") == "right"
    assert mirror_justify("right") == "left"
    assert mirror_justify("center") == "center"
    assert mirror_justify("full") == "full"


def test_reading_order_copies_rather_than_reordering_the_caller_s_list() -> None:
    original = ["a", "b"]
    assert reading_order(original) is not original


def test_the_set_of_right_to_left_languages_holds_the_three_that_ship() -> None:
    assert {"ar", "he", "ur"} <= bidi.RTL_LANGUAGES
