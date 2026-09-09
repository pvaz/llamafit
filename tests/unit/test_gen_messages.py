"""The message extractor: what it finds, what it refuses, and what it writes.

``tests/unit/test_i18n_catalogs.py`` checks that the committed template matches the real
sources; these tests exercise the rules on synthetic files, including the ones the real
sources do not happen to break.
"""

from pathlib import Path

import pytest
from scripts.gen_messages import (
    DEST,
    EXCLUDED,
    SOURCE_ROOTS,
    extract,
    main,
    python_files,
    render_template,
)

from llamafit.i18n.po import parse_po


def _write(tmp_path: Path, text: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_simple_call_is_found(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, 'from llamafit.i18n import _\n\n_("No GPU detected")\n')])
    assert found.problems == []
    assert [(e.msgid, e.plural) for e in found.entries] == [("No GPU detected", None)]


def test_the_full_name_and_a_dotted_call_are_found_too(tmp_path: Path) -> None:
    source = 'gettext("a")\ni18n._("b")\nself.i18n.ngettext("c", "cs", 2)\n'
    found = extract([_write(tmp_path, source)])
    assert [e.msgid for e in found.entries] == ["a", "b", "c"]
    assert found.entries[2].plural == "cs"


def test_a_message_split_over_lines_extracts_as_one_sentence(tmp_path: Path) -> None:
    source = (
        '_(\n    "LlamaFit could not read the totals, "\n    "so it cannot say what fits."\n)\n'
    )
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].msgid == (
        "LlamaFit could not read the totals, so it cannot say what fits."
    )


def test_a_call_inside_a_comment_or_a_string_is_not_a_call(tmp_path: Path) -> None:
    source = '# _("in a comment")\nHELP = \'_("in a string")\'\n_("real")\n'
    found = extract([_write(tmp_path, source)])
    assert [e.msgid for e in found.entries] == ["real"]


def test_the_same_message_twice_is_one_entry_with_two_references(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, '_("twice")\n_("twice")\n')])
    assert len(found.entries) == 1
    assert len(found.entries[0].references) == 2


@pytest.mark.parametrize(
    "source",
    [
        "_(name)\n",
        '_(f"hello {name}")\n',
        '_("a" + name)\n',
        '_("a".upper())\n',
        'ngettext("one", plural, 2)\n',
        "_(*args)\n",
    ],
)
def test_a_call_that_is_not_a_literal_is_reported_with_its_file_and_line(
    tmp_path: Path, source: str
) -> None:
    found = extract([_write(tmp_path, source)])
    assert found.entries == []
    assert len(found.problems) == 1
    assert "sample.py:1" in found.problems[0]


def test_a_call_with_no_arguments_at_all_is_reported(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, "_()\n")])
    assert "missing its message argument" in found.problems[0]


def test_a_file_that_does_not_parse_is_reported_not_skipped(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, "def broken(:\n")])
    assert found.entries == []
    assert "cannot parse" in found.problems[0]


def test_the_same_message_with_two_different_plurals_is_reported(tmp_path: Path) -> None:
    source = (
        'ngettext("%(count)d file", "%(count)d files", n)\n'
        'ngettext("%(count)d file", "%(count)d FILES", n)\n'
    )
    found = extract([_write(tmp_path, source)])
    assert "already has a different plural form" in found.problems[0]


def test_the_translation_machinery_is_not_extracted_from_itself() -> None:
    files = list(python_files(SOURCE_ROOTS))
    assert files
    for excluded in EXCLUDED:
        assert not any(excluded in path.parents for path in files)


def test_caches_are_skipped(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "sample.py").write_text('_("cached")\n', encoding="utf-8")
    _write(tmp_path, '_("real")\n')
    assert [e.msgid for e in extract([tmp_path]).entries] == ["real"]


def test_entries_come_out_in_the_order_they_appear(tmp_path: Path) -> None:
    source = '_("first")\nngettext("second", "seconds", n)\n_("third")\n'
    found = extract([_write(tmp_path, source)])
    assert [e.msgid for e in found.entries] == ["first", "second", "third"]


def test_the_rendered_template_parses_and_translates_nothing(tmp_path: Path) -> None:
    source = (
        '_("plain")\n'
        '_("two\\nlines")\n'
        "_('a \"quoted\" word')\n"
        'ngettext("%(count)d x", "%(count)d xs", n)\n'
    )
    template = render_template(extract([_write(tmp_path, source)]), version="9.9.9")
    catalog = parse_po(template)
    assert catalog.plural_forms == "nplurals=2; plural=(n != 1);"
    assert catalog.headers["Project-Id-Version"] == "llamafit 9.9.9"
    assert set(catalog.messages) == {
        (None, "plain"),
        (None, "two\nlines"),
        (None, 'a "quoted" word'),
        (None, "%(count)d x"),
    }
    assert catalog.untranslated() == tuple(catalog.messages)
    assert catalog.messages[None, "%(count)d x"].translations == ("", "")


def test_a_multi_line_message_is_written_the_way_msgfmt_writes_it(tmp_path: Path) -> None:
    template = render_template(extract([_write(tmp_path, '_("one\\ntwo")\n')]), version="1")
    assert 'msgid ""\n"one\\n"\n"two"\n' in template


def test_the_template_ends_with_a_newline_and_has_a_reference_per_entry(tmp_path: Path) -> None:
    template = render_template(extract([_write(tmp_path, '_("a")\n')]), version="1")
    assert template.endswith("\n")
    assert "#: " in template


def test_a_run_over_clean_sources_writes_the_template(tmp_path: Path) -> None:
    before = DEST.read_text(encoding="utf-8")
    try:
        assert main([str(_write(tmp_path, '_("only this")\n'))]) == 0
        assert 'msgid "only this"' in DEST.read_text(encoding="utf-8")
    finally:
        DEST.write_text(before, encoding="utf-8", newline="\n")


def test_a_run_that_finds_a_problem_writes_nothing_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = DEST.read_text(encoding="utf-8")
    assert main([str(_write(tmp_path, "_(name)\n"))]) == 1
    assert DEST.read_text(encoding="utf-8") == before
    captured = capsys.readouterr()
    assert "not a literal string" in captured.err
    assert "refusing to write messages.pot" in captured.err


def test_a_run_with_no_arguments_uses_the_source_roots() -> None:
    before = DEST.read_text(encoding="utf-8")
    assert main([]) == 0
    assert DEST.read_text(encoding="utf-8") == before


def test_a_contextual_call_is_extracted_with_its_context(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, 'pgettext("GPU", "none detected")\n')])
    assert found.problems == []
    assert [(e.context, e.msgid) for e in found.entries] == [("GPU", "none detected")]


def test_one_message_under_two_contexts_is_two_entries(tmp_path: Path) -> None:
    source = (
        'pgettext("GPU", "none detected")\n'
        'pgettext("backends", "none detected")\n'
        '_("none detected")\n'
    )
    found = extract([_write(tmp_path, source)])
    assert [(e.context, e.msgid) for e in found.entries] == [
        ("GPU", "none detected"),
        ("backends", "none detected"),
        (None, "none detected"),
    ]


def test_the_same_contextual_call_twice_is_one_entry_with_two_references(tmp_path: Path) -> None:
    source = 'pgettext("GPU", "none detected")\nlazy_pgettext("GPU", "none detected")\n'
    found = extract([_write(tmp_path, source)])
    assert len(found.entries) == 1
    assert len(found.entries[0].references) == 2


def test_a_contextual_counting_call_is_extracted_with_both_forms(tmp_path: Path) -> None:
    source = 'npgettext("GPU", "%(count)d device", "%(count)d devices", n)\n'
    found = extract([_write(tmp_path, source)])
    assert [(e.context, e.msgid, e.plural) for e in found.entries] == [
        ("GPU", "%(count)d device", "%(count)d devices")
    ]


def test_a_deferred_contextual_counting_call_is_extracted_too(tmp_path: Path) -> None:
    source = 'lazy_npgettext("GPU", "%(count)d device", "%(count)d devices", 3)\n'
    found = extract([_write(tmp_path, source)])
    assert [(e.context, e.msgid, e.plural) for e in found.entries] == [
        ("GPU", "%(count)d device", "%(count)d devices")
    ]


@pytest.mark.parametrize(
    "source",
    [
        'pgettext(where, "none detected")\n',
        'pgettext("GPU", name)\n',
        'npgettext("GPU", "%(count)d device", plural, n)\n',
    ],
)
def test_a_contextual_call_that_is_not_a_literal_is_reported(tmp_path: Path, source: str) -> None:
    found = extract([_write(tmp_path, source)])
    assert found.entries == []
    assert "not a literal string" in found.problems[0]


def test_a_contextual_call_missing_its_message_is_reported(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, 'pgettext("GPU")\n')])
    assert "missing its message argument" in found.problems[0]


def test_an_empty_context_is_refused_rather_than_filed_as_a_third_thing(tmp_path: Path) -> None:
    found = extract([_write(tmp_path, 'pgettext("", "none detected")\n')])
    assert found.entries == []
    assert "the context is empty" in found.problems[0]


def test_the_template_writes_a_msgctxt_line_above_the_msgid(tmp_path: Path) -> None:
    source = 'pgettext("GPU", "none detected")\n_("none detected")\n'
    template = render_template(extract([_write(tmp_path, source)]), version="1")
    assert 'msgctxt "GPU"\nmsgid "none detected"' in template
    catalog = parse_po(template)
    assert set(catalog.messages) == {("GPU", "none detected"), (None, "none detected")}


# --- the note a call site leaves for the translator ---------------------------------


def test_a_marked_comment_above_a_call_reaches_the_template(tmp_path: Path) -> None:
    source = (
        "# Translators: this is punctuation, not prose.\n"
        "# Write what your own language uses.\n"
        'x = pgettext("thousands separator", ",")\n'
    )
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].comments == [
        "Translators: this is punctuation, not prose.",
        "Write what your own language uses.",
    ]
    rendered = render_template(found, version="0.0.0")
    assert "#. Translators: this is punctuation, not prose." in rendered
    assert "#. Write what your own language uses." in rendered


def test_an_unmarked_comment_stays_in_the_code(tmp_path: Path) -> None:
    # A note about the code is written for whoever maintains it, and a translator could
    # not act on it, so only a block opening with the marker is copied.
    source = '# this is why the call is here\n_("No GPU detected")\n'
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].comments == []
    assert "#." not in render_template(found, version="0.0.0")


def test_a_blank_line_ends_the_block(tmp_path: Path) -> None:
    # The note has to touch the call, or a comment further up the file would attach
    # itself to whatever message happened to come next.
    source = '# Translators: a note.\n\n_("No GPU detected")\n'
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].comments == []


def test_the_same_note_on_two_call_sites_is_not_recorded_twice(tmp_path: Path) -> None:
    source = (
        "# Translators: a note.\n"
        '_("No GPU detected")\n'
        "# Translators: a note.\n"
        '_("No GPU detected")\n'
    )
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].comments == ["Translators: a note."]
    assert len(found.entries[0].references) == 2


def test_two_different_notes_on_one_message_are_both_kept(tmp_path: Path) -> None:
    source = (
        "# Translators: the GPU row.\n"
        '_("none detected")\n'
        "# Translators: the backends row.\n"
        '_("none detected")\n'
    )
    found = extract([_write(tmp_path, source)])
    assert found.entries[0].comments == [
        "Translators: the GPU row.",
        "Translators: the backends row.",
    ]
