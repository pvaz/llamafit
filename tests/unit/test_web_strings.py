"""The page's words, and the two places they have to agree with something else.

There is no JavaScript runtime in this project -- goal one forbids Node -- so ``app.js``
cannot be unit-tested the way a Python module can. What can be tested is every promise it
makes to Python: that each key it asks for exists, that each label table it reads is
served, and that the one piece of formatting it repeats still matches ``llamafit.units``.
Those are the ways the page could quietly start drawing the wrong word or the wrong size,
and each of them fails here instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from llamafit.cli.render import _capability_label
from llamafit.i18n import set_language, translator
from llamafit.models.catalog import Capability
from llamafit.models.plan import Confidence, Pool, RunMode, Verdict
from llamafit.units import _BINARY_UNITS, format_bytes
from llamafit.web import strings

STATIC = Path(strings.__file__).parent / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


def test_the_capability_words_are_the_command_lines_own() -> None:
    """Two tables, one catalog entry each: they must never name a capability differently."""
    for capability in get_args(Capability):
        assert strings.capability_label(capability) == _capability_label(capability)


def test_a_capability_the_table_has_not_met_comes_back_in_english() -> None:
    assert strings.capability_label("telepathy") == "telepathy"


@pytest.mark.parametrize(
    ("table", "values"),
    [
        ("verdict", get_args(Verdict)),
        ("verdict_sentence", get_args(Verdict)),
        ("mode", get_args(RunMode)),
        ("confidence", get_args(Confidence)),
        ("pool", get_args(Pool)),
        ("capability", get_args(Capability)),
    ],
)
def test_every_value_the_documents_carry_has_a_word(table: str, values: tuple[str, ...]) -> None:
    """A value with no entry would reach the reader as an identifier."""
    served = strings.value_labels()[table]
    assert set(served) == set(values)
    assert all(served.values())


def test_the_budget_components_are_the_ones_the_terminal_names() -> None:
    from llamafit.cli.render_board import component_label

    labels = strings.value_labels()["component"]
    assert set(labels) == set(strings.BUDGET_COMPONENTS)
    for component, word in labels.items():
        assert word == component_label(component)


def test_every_string_the_page_asks_for_is_served() -> None:
    """The keys in app.js and index.html, checked against the table Python builds."""
    payload = strings.ui_payload()
    served = dict(payload["strings"])  # type: ignore[arg-type]
    columns = dict(payload["columns"])  # type: ignore[arg-type]
    asked = set(re.findall(r'\bt\("([a-z_]+\.[a-z_]+)"\)', APP_JS))
    asked |= set(re.findall(r'data-t="([a-z_]+\.[a-z_]+)"', INDEX))
    for key in sorted(asked):
        table, name = key.split(".", 1)
        if table == "columns":
            assert name in columns, f"app.js asks for a column heading {name!r} nobody serves"
        else:
            assert key in served, f"the page asks for {key!r} and strings.py does not serve it"


def test_every_column_heading_the_page_asks_for_is_served() -> None:
    columns = strings.column_headings()
    for name in sorted(set(re.findall(r'\bcol\("([a-z_]+)"\)', APP_JS))):
        assert name in columns, f"app.js asks for the {name!r} heading and nobody serves it"


def test_every_label_table_the_page_reads_is_served() -> None:
    tables = strings.value_labels()
    for name in sorted(set(re.findall(r'\blabel\("([a-z_]+)",', APP_JS))):
        assert name in tables, f"app.js reads a {name!r} label table nobody serves"


def test_no_string_is_served_empty() -> None:
    assert all(strings.page_strings().values())
    assert all(strings.column_headings().values())


def test_the_byte_units_the_page_writes_are_the_ones_python_writes() -> None:
    """``app.js`` repeats this ladder; a change here has to be made there too."""
    assert _BINARY_UNITS == ["B", "KiB", "MiB", "GiB", "TiB"]
    assert 'const BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"];' in APP_JS


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"), (8 * 1024**3, "8.0 GiB")],
)
def test_the_sizes_the_page_will_draw_are_pinned(size: int, expected: str) -> None:
    """The four shapes app.js has to reproduce: bytes, and each rung of the ladder."""
    assert format_bytes(size) == expected


def test_the_payload_follows_the_language_the_process_speaks() -> None:
    """A reader who started LlamaFit in Portuguese gets a Portuguese page.

    The two entries checked are ones the terminal already asks for, which is the whole
    point of writing the page's messages as the same literals: every word a translator
    has filled in for the command line is a word the dashboard already speaks. A message
    only the page asks for falls back to English until somebody translates it, exactly as
    a new message anywhere else in LlamaFit does.
    """
    try:
        set_language("pt_PT")
        payload = strings.ui_payload()
        assert payload["language"] == "pt_PT"
        assert payload["direction"] == "ltr"
        assert payload["strings"]["host.memory"] == "Memória"  # type: ignore[index]
        assert payload["labels"]["capability"]["coding"] != "coding"  # type: ignore[index]
    finally:
        translator.reset()


def test_a_right_to_left_language_asks_the_page_to_turn_round() -> None:
    try:
        set_language("ar")
        assert strings.ui_payload()["direction"] == "rtl"
    finally:
        translator.reset()


def test_the_footer_link_points_at_the_source() -> None:
    """Section 13 of the AGPL: whoever reaches this over a network is owed the source."""
    assert strings.SOURCE_URL in INDEX
    assert strings.ui_payload()["source_url"] == strings.SOURCE_URL
