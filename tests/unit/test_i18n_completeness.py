"""A language that is only part translated has to say so, and say it once.

All 37 shipped catalogs are complete today, and this is still here for the next one that
is not: a catalog is written from the top of the template down, so a part-written one
carries the fragments -- a column heading, a verdict, a unit -- and not the prose. The
screen a reader gets is labelled in their own language and explained in English, and
nothing about that is visible from inside it: a reader has no way to tell a translation
in progress from a translation that went wrong.

So what is pinned here is the sentence that tells them, the figure in it, and the two
ways it could be got wrong -- said on a run that had nothing to report, and swallowed on
a run that already had something else to say.

The part-written catalogs the tests below need are written into a temporary directory
rather than borrowed from the ones that ship. They used to be borrowed, which worked
until the day every shipped catalog was finished and the two tests that did so had
nothing left to point at. A fixture cannot be completed out from under a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.i18n import catalogs, translator
from llamafit.i18n.catalogs import available_languages, load_language
from llamafit.i18n.completeness import (
    NEARLY_COMPLETE,
    SHORTFALL_HINT,
    Completeness,
    _count,
    completeness,
    shortfall_after,
    template_message_count,
)
from llamafit.i18n.select import LanguageChoice
from llamafit.i18n.tags import SOURCE_LANGUAGE
from llamafit.i18n.translator import set_language

runner = CliRunner()

LANGUAGES = [tag for tag in available_languages() if tag != SOURCE_LANGUAGE]

TEMPLATE = """
msgid ""
msgstr ""
"Language: \\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "one"
msgstr ""

msgid "two"
msgstr ""

msgid "three"
msgstr ""

msgid "four"
msgstr ""
"""

PART_WRITTEN = """
msgid ""
msgstr ""
"Language: pt_PT\\n"
"Language-Team: Portuguese (Portugal)\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "one"
msgstr "um"
"""

FINISHED = """
msgid ""
msgstr ""
"Language: pt_PT\\n"
"Language-Team: Portuguese (Portugal)\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "one"
msgstr "um"

msgid "two"
msgstr "dois"

msgid "three"
msgstr "tres"

msgid "four"
msgstr "quatro"
"""


def a_locale_dir(tmp_path: Path, catalog: str, *, template: str | None = TEMPLATE) -> Path:
    """A directory holding one Portuguese catalog and, unless asked otherwise, a template.

    The template goes in first and is never rewritten: the count behind it is cached for
    the life of a process, because it is a property of the release and cannot change
    under a running one, and a test that wrote it twice would be testing the cache.
    """
    if template is not None:
        (tmp_path / "messages.pot").write_text(template, encoding="utf-8")
    (tmp_path / "pt_PT.po").write_text(catalog, encoding="utf-8")
    return tmp_path


def choose(directory: Path) -> LanguageChoice:
    """Install the Portuguese catalog in ``directory`` the way a command's start-up does."""
    return set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=directory)


# --- measuring one catalog ---------------------------------------------------------------


def test_a_catalog_is_measured_against_the_template_and_not_against_itself(
    tmp_path: Path,
) -> None:
    # The whole of the finding is here. A part-written catalog carries no entry at all for
    # a message nobody has reached yet, so counting its filled entries against its own
    # length says it is finished: one of one, a hundred per cent, and the reader is told
    # nothing on a screen that is three quarters English.
    catalog = load_language("pt_PT", directory=a_locale_dir(tmp_path, PART_WRITTEN))
    assert len(catalog.messages) == 1, "the catalog itself knows about one message"
    measured = completeness(catalog, directory=tmp_path)
    assert measured == Completeness(translated=1, total=4)
    assert measured.percent == 25


def test_an_entry_left_blank_counts_as_missing_and_not_as_present(tmp_path: Path) -> None:
    blank = PART_WRITTEN + '\nmsgid "two"\nmsgstr ""\n'
    catalog = load_language("pt_PT", directory=a_locale_dir(tmp_path, blank))
    assert len(catalog.messages) == 2
    assert completeness(catalog, directory=tmp_path).translated == 1


def test_the_share_is_rounded_down_so_the_gap_is_never_understated() -> None:
    assert Completeness(translated=239, total=1046).percent == 22
    assert Completeness(translated=1045, total=1046).percent == 99


def test_a_catalog_carrying_more_than_the_template_does_not_claim_more_than_all_of_it() -> None:
    # Dead weight, which the shipped catalogs are held to by their own test. Whatever it
    # is, it is not a hundred and fifty per cent of a translation.
    over = Completeness(translated=6, total=4)
    assert over.share == 1.0
    assert over.percent == 100
    assert over.nearly_complete


def test_with_no_template_to_measure_against_nothing_is_claimed_either_way() -> None:
    unknown = Completeness(translated=3, total=0)
    assert not unknown.known
    assert unknown.nearly_complete, "an unanswerable question is not an accusation"
    # And the share is one rather than a division by zero or a nought: whatever is being
    # asked, the answer must not be a figure that reads as "this catalog is empty".
    assert unknown.share == 1.0
    assert unknown.percent == 100


def test_a_missing_template_counts_as_nothing_known_rather_than_raising(tmp_path: Path) -> None:
    assert template_message_count(directory=tmp_path) == 0


# --- what the reader is told -------------------------------------------------------------


def test_a_part_written_catalog_is_reported_with_its_figure_and_what_to_do(
    tmp_path: Path,
) -> None:
    choice = choose(a_locale_dir(tmp_path, PART_WRITTEN))
    assert choice.notice is not None
    notice = choice.notice
    assert "25%" in notice
    # The catalog's own name for itself, not its tag: a reader who reads "pt_PT" learns
    # less than one who reads the name the translators put in the file.
    assert "Portuguese (Portugal)" in notice
    assert choice.hint == SHORTFALL_HINT


def test_a_finished_catalog_is_not_remarked_on(tmp_path: Path) -> None:
    choice = choose(a_locale_dir(tmp_path, FINISHED))
    assert choice.notice is None


def test_a_catalog_with_no_template_beside_it_is_not_accused_of_anything(
    tmp_path: Path,
) -> None:
    choice = choose(a_locale_dir(tmp_path, PART_WRITTEN, template=None))
    assert choice.notice is None


def test_the_shortfall_is_said_once_in_a_process(tmp_path: Path) -> None:
    directory = a_locale_dir(tmp_path, PART_WRITTEN)
    assert choose(directory).notice is not None
    assert choose(directory).notice is None, "the second command must not repeat it"


def test_a_substitution_onto_a_part_written_catalog_says_both_things(tmp_path: Path) -> None:
    # Two things are wrong with this run: the reader is getting another region's words,
    # and that region's catalog is a quarter written. Told only the first, they would put
    # the English down to the region they did not get.
    directory = a_locale_dir(tmp_path, PART_WRITTEN)
    choice = set_language("pt_BR", env={}, available=("en", "pt_PT"), directory=directory)
    notice = choice.notice
    assert notice is not None
    assert "no pt_BR translation" in notice
    assert "25%" in notice
    assert choice.hint is not None and "pt_BR" in choice.hint, (
        "the substitution's own hint is the more useful of the two and is kept"
    )


def test_the_follow_on_sentence_points_back_at_the_catalog_already_named() -> None:
    both = shortfall_after(
        "LlamaFit has no zz_ZZ translation, so it is using the Zed one.",
        Completeness(translated=1, total=4),
    )
    assert both.startswith("LlamaFit has no zz_ZZ translation")
    assert "That one is about 25% written" in both


# --- English, and the cost of asking the question ----------------------------------------


def test_english_never_opens_the_template(monkeypatch: pytest.MonkeyPatch) -> None:
    # The count is the only part of this that costs anything, and a reader of the source
    # language must not pay it. Nothing installs a catalog for English, so nothing here
    # should be reached -- and a stub that raises is the only way to prove it.
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("English asked how complete a catalog it never loaded is")

    monkeypatch.setattr(translator, "completeness", refuse)
    assert set_language("en", env={}, available=("en", "pt_PT")).language == "en"


# --- the catalogs that actually ship -----------------------------------------------------


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_shipped_catalog_either_is_finished_or_says_it_is_not(language: str) -> None:
    # No list of which languages are done: one lands finished and a test naming the set
    # fails on somebody else's contribution. What holds however many arrive is that a
    # catalog short of the line is announced and one past it is not.
    measured = completeness(load_language(language))
    translator.reset()
    choice = set_language(language, env={}, directory=None)
    assert measured.known, "the template ships beside the catalogs it was extracted from"
    assert (choice.notice is None) == measured.nearly_complete
    if choice.notice is not None:
        assert f"{measured.percent}%" in choice.notice


def test_a_shipped_catalog_can_reach_the_line() -> None:
    # A threshold no real catalog can clear would silence nothing and announce everything,
    # and every test of the quiet case above uses a catalog written for the occasion --
    # four messages, four translations -- which cannot catch that. This one is measured
    # against the real template by the real catalogs.
    #
    # The opposite half of this used to be here too: that some shipped catalog fell short,
    # so both branches were exercised by real data. All 37 are complete now, and it is not
    # coming back as `all(...)`. docs/translations.md welcomes a partly finished catalog
    # that a person has read; a suite that failed on one would enforce the reverse of the
    # rule the project actually has. The short branch is exercised above, by a catalog
    # written for it.
    shares = [completeness(load_language(tag)).share for tag in LANGUAGES]
    assert any(share >= NEARLY_COMPLETE for share in shares)


def test_the_notice_reaches_stderr_on_a_real_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The route is the one the substitution notice already takes, and the point of this
    # is that it is the same route: a sentence that never leaves the library is not a
    # notice. `--version` is enough, because the language is chosen before it prints.
    #
    # The command reads the packaged directory and takes no argument for one, so the
    # packaged directory is what moves. What moves is `packaged_dir`, one level below
    # `catalog_dir`, and not `catalog_dir` itself: two modules imported that name at
    # import time, so patching it reaches one of them and leaves the other pointing at
    # the real catalogs. Patching what it calls reaches both, because both are holding
    # the same function object.
    #
    # The template count is cached under the directory it was read from, so the cache is
    # dropped either side: a stale 1153 here would make a four-message catalog look
    # finished, and a stale 4 afterwards would make every real catalog look finished to
    # every test that ran next.
    folder = a_locale_dir(tmp_path, PART_WRITTEN)
    monkeypatch.setattr(catalogs, "packaged_dir", lambda *args, **kwargs: folder)
    _count.cache_clear()
    translator.reset()
    try:
        result = runner.invoke(app, ["--language", "pt_PT", "--version"])
        assert result.exit_code == 0
        remarked = " ".join(result.stderr.split())
        assert "written, so much of what it says will be in English" in remarked
        assert SHORTFALL_HINT in remarked
        # And not on stdout, which is where a `--json` document would have been.
        assert "written, so much of" not in result.stdout
    finally:
        _count.cache_clear()
