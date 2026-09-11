"""One vocabulary for the board, read by the command line, the dashboard and the page.

Three interfaces draw the same board and used to decide separately which columns it had,
in which order, at which width; two of them had a chooser each and the page had a list of
its own. What is pinned here is that there is one answer: the same headings in the same
order at the same width from all three, one set of sort words accepted by all three, and
the unranked candidates on the table beside the ranked ones with the reason said once.

The column tests draw the real thing -- a Rich console of the given width, a Textual
pilot of the given size, the page's own ``app.js`` -- rather than asking the chooser
what it would choose, because the finding this file closes was three drawings drifting
apart while every one of them was internally consistent.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rich.cells import cell_len
from rich.console import Console
from textual.widgets import DataTable
from typer.testing import CliRunner

from llamafit.cli import render_board as rb
from llamafit.cli.app import app
from llamafit.i18n import set_language
from llamafit.i18n import translator as translator_module
from llamafit.models.plan import Candidate, Needs
from llamafit.services.recommend import BoardRow, build_board, build_fit_board
from llamafit.tui import entry
from llamafit.tui.app import LlamaFitApp
from llamafit.tui.screens.board import BoardPane
from llamafit.web import strings
from llamafit.web.api import Dashboard as WebDashboard
from llamafit.web.api import create_app
from tests.fixtures.board import catalog, report
from tests.fixtures.budget_hosts import reference_host
from tests.fixtures.dashboard import ready

WIDTHS = (80, 100, 120, 140, 160, 200)
"""The widths the brief named; the chooser is checked at every one of them."""

STATIC = Path(strings.__file__).parent / "static"
runner = CliRunner()


@pytest.fixture(autouse=True)
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command sees the reference machine, never the one running the tests."""
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())


@pytest.fixture
def english() -> Iterator[None]:
    """Whatever a test switched the language to, the next one starts in English."""
    yield
    translator_module.reset()
    set_language("en", env={})


def a_board(**needs: Any) -> Any:
    """The bundled catalog ranked on the reference machine, every row kept."""
    return build_board(catalog(), reference_host(), Needs(**needs))


def drawn(renderable: object, width: int) -> str:
    console = Console(width=width, no_color=True, highlight=False, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def cli_headings(board: Any, width: int) -> list[str]:
    """The headings the command line actually draws at this width, left to right."""
    text = drawn(
        rb.render_board(board, console_width=width, budget=rb.column_budget(catalog())), width
    )
    header = next(
        line for line in text.splitlines() if "│" in line and rb.column_heading("model") in line
    )
    return [cell.strip() for cell in header.strip().strip("│").split("│")]


def page_order() -> list[str]:
    """The page's columns, in the order ``app.js`` draws them, read off the file."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    start = js.index("const BOARD_COLUMNS")
    return re.findall(r'name: "(\w+)"', js[start : js.index("];", start)])


# --- which columns, in which order, at which width ------------------------------------


def test_the_page_draws_the_vocabulary_s_columns_in_the_vocabulary_s_order() -> None:
    # ``have`` is the one column the page keeps inside the row; everything else is drawn
    # in exactly the order BOARD_ORDER writes down, which is what the terminal follows.
    assert page_order() == [column for column in rb.BOARD_ORDER if column != "have"]


@pytest.mark.asyncio
@pytest.mark.parametrize("width", WIDTHS)
async def test_the_three_interfaces_draw_the_same_headings_in_the_same_order(width: int) -> None:
    board = a_board()
    expected = [
        rb.column_heading(column)
        for column in rb.board_columns(
            width, mixed_confidence=False, budget=rb.column_budget(catalog())
        )
    ]
    # The terminal's headings are the page's, restricted to what the width admits; ``have``
    # is the one column the page keeps inside the row instead.
    on_page = [rb.column_heading(name) for name in page_order()]
    have = rb.column_heading("have")
    assert [h for h in expected if h != have] == [h for h in on_page if h in expected]
    assert cli_headings(board, width) == expected
    dashboard = LlamaFitApp(ready())
    async with dashboard.run_test(size=(width, 60)):
        table = dashboard.query_one("#board-table", DataTable)
        assert [str(column.label) for column in table.columns.values()] == expected


@pytest.mark.parametrize("width", range(20, 240, 3))
def test_the_four_columns_that_identify_a_row_are_never_dropped(width: int) -> None:
    for mixed in (True, False):
        chosen = rb.board_columns(width, mixed_confidence=mixed)
        assert set(rb.BOARD_REQUIRED) <= set(chosen)


@pytest.mark.parametrize("width", range(20, 240, 3))
def test_a_wider_terminal_never_shows_fewer_columns(width: int) -> None:
    narrow = rb.board_columns(width, mixed_confidence=False)
    wide = rb.board_columns(width + 1, mixed_confidence=False)
    assert set(narrow) <= set(wide)


@pytest.mark.parametrize("width", range(20, 240, 3))
def test_a_speed_is_never_shown_without_its_label_when_the_rows_disagree(width: int) -> None:
    chosen = rb.board_columns(width, mixed_confidence=True)
    assert ("gen" in chosen) == ("confidence" in chosen)


def test_when_every_row_agrees_the_verdict_is_worth_more_than_a_repeated_label() -> None:
    at_eighty = rb.board_columns(80, mixed_confidence=False)
    assert "gen" in at_eighty and "verdict" in at_eighty
    assert "confidence" not in at_eighty
    assert "confidence" in rb.board_columns(200, mixed_confidence=False)


def test_a_column_keeps_its_place_whatever_the_width() -> None:
    """Widening a terminal adds a column where the page has it, never on the far right."""
    for width in WIDTHS:
        chosen = rb.board_columns(width, mixed_confidence=False)
        assert list(chosen) == [column for column in rb.BOARD_ORDER if column in chosen]


def test_the_column_set_does_not_depend_on_how_many_rows_a_limit_left() -> None:
    """The finding: nine columns at 100 under ``--limit 2``, seven under the default."""
    for limit in (1, 2, 10, 500):
        board = build_board(catalog(), reference_host(), Needs(), limit=limit)
        assert cli_headings(board, 100) == cli_headings(a_board(), 100)


def test_the_budget_is_measured_from_the_catalog_and_capped() -> None:
    budget = rb.column_budget(catalog())
    assert budget.model == rb.ID_COLUMN_MAX_WIDTH
    assert budget.quant == max(
        cell_len(q) for m in catalog().models for q in [x.name for x in m.sources[0].quants]
    )
    assert rb.column_budget(None) == rb.ColumnBudget()


@pytest.mark.asyncio
async def test_a_translated_board_never_overflows_the_terminal(english: None) -> None:
    """The trap: a Portuguese label is wider than its English, and a budget in English
    admitted a column the Portuguese then overflowed. The budget is measured from the
    labels in force, so a narrower set is admitted instead."""
    set_language("pt_PT", env={})
    widths = rb.column_widths()
    assert widths["verdict"] > 7, "the Portuguese verdicts are the wider ones this test needs"
    dashboard = LlamaFitApp(ready())
    async with dashboard.run_test(size=(80, 60)):
        table = dashboard.query_one("#board-table", DataTable)
        assert sum(column.get_render_width(table) for column in table.columns.values()) <= 80
    text = drawn(rb.render_board(a_board(), console_width=80), 80)
    assert all(cell_len(line) <= 80 for line in text.splitlines())


def test_the_fit_listing_chooses_its_columns_the_same_way() -> None:
    at_eighty = rb.fit_columns(80)
    assert set(rb.FIT_REQUIRED) <= set(at_eighty)
    assert list(at_eighty) == [column for column in rb.FIT_ORDER if column in at_eighty]
    assert set(rb.fit_columns(200)) == set(rb.FIT_ORDER)


# --- the unranked candidates on the same table ----------------------------------------


def test_an_unranked_candidate_is_a_dimmed_row_with_every_figure_and_the_word_for_why() -> None:
    board = a_board()
    slow = next(row for row in board.excluded if row.candidate.excluded_tag == "too slow")
    text = drawn(rb.render_board(board, console_width=200), 200)
    table = text[: text.index("Speeds are for")]
    line = next(line for line in table.splitlines() if slow.model_id in line)
    assert "too slow" in line
    # Placed, sized and estimated before it was excluded; those figures are on the row.
    assert slow.candidate.speed is not None
    assert f"{slow.candidate.speed.gen_tps:.1f}" in line


def test_hiding_the_unranked_rows_keeps_the_reasons() -> None:
    board = a_board()
    text = drawn(rb.render_board(board, console_width=200, view=rb.View(excluded=False)), 200)
    assert "too slow" not in text
    assert rb.render_reasons(board) is not None


def test_a_reason_with_no_tag_gets_a_word_rather_than_a_blank() -> None:
    unsized = BoardRow(
        rank=None,
        model_id="x",
        name="X",
        quant="Q4_K_M",
        candidate=Candidate(model_id="x", quant="Q4_K_M", excluded_because="no header was read"),
    )
    facts = rb.facts_of(unsized)
    assert rb.exclusion_tag(facts) == "not ranked"
    assert str(rb.board_cell(facts, "score")) == "not ranked"
    # And a figure that was never computed is a mark, not a blank a reader takes for zero.
    for column in ("gen", "verdict", "mode", "vram", "ram", "context"):
        assert str(rb.board_cell(facts, column)).strip()
        assert str(rb.board_cell(facts, column)) != "0"


def test_an_unsupported_placement_shows_no_figures_it_never_had() -> None:
    board = a_board()
    row = next(r for r in board.excluded if r.candidate.excluded_tag == "unsupported")
    facts = rb.facts_of(row)
    assert str(rb.board_cell(facts, "gen")) != "0.0"
    assert str(rb.board_cell(facts, "vram")) != "0 B"
    assert str(rb.board_cell(facts, "mode")) == rb.mode_label("unsupported")


def test_the_fit_listing_puts_the_unplaced_models_on_the_table_with_their_tag() -> None:
    board = build_fit_board(catalog(), reference_host())
    text = drawn(rb.render_fit(board, console_width=200), 200)
    table = text[: text.index("Sized for")]
    assert board.excluded
    assert board.excluded[0].model_id in table
    assert "no room" in table


def test_an_unranked_row_explains_itself_without_a_rank_to_lead_with() -> None:
    board = a_board()
    row = board.excluded[0]
    text = " ".join(drawn(rb.render_explanation(row, board), 200).split())
    assert "None." not in text
    assert row.name in text
    assert (row.candidate.excluded_because or "")[:30] in text


# --- one sort vocabulary ---------------------------------------------------------------


def test_every_sort_word_is_a_flag_a_query_parameter_and_a_dashboard_state() -> None:
    from llamafit.web import api

    assert api.SORT_KEYS is rb.SORT_KEYS
    client = TestClient(
        create_app(WebDashboard(scan=report, catalog_loader=lambda: (catalog(), []))),
        base_url="http://127.0.0.1",
    )
    for key in rb.SORT_KEYS:
        result = runner.invoke(app, ["--json", "recommend", "--sort", key, "--limit", "3"])
        assert result.exit_code == 0, (key, result.output)
        assert client.get("/api/v1/models/top", params={"sort": key, "limit": 3}).status_code == 200


@pytest.mark.asyncio
async def test_the_dashboard_s_sort_key_walks_every_word_in_order() -> None:
    dashboard = LlamaFitApp(ready())
    async with dashboard.run_test(size=(120, 44)) as pilot:
        seen = []
        for _ in rb.SORT_KEYS:
            await pilot.press("s")
            seen.append(dashboard.query_one("#board", BoardPane).view.sort)
        assert seen == [*rb.SORT_KEYS[1:], rb.SORT_KEYS[0]]


def test_a_figure_sorts_largest_first_a_word_a_to_z_and_a_missing_value_last() -> None:
    board = a_board()
    rows = [*board.rows, *board.excluded]
    by_size = rb.sorted_rows(rows, "size")
    sizes = [r.download_bytes for r in by_size if r.download_bytes is not None]
    assert sizes == sorted(sizes, reverse=True)
    smallest_first = rb.sorted_rows(rows, "size", descending=False)
    assert smallest_first[0].download_bytes == min(sizes)
    by_model = rb.sorted_rows(rows, "model")
    assert [r.model_id for r in by_model] == sorted((r.model_id for r in rows), key=str.casefold)
    by_speed = rb.sorted_rows(rows, "speed")
    # The rows with no speed at all -- unsupported, unsized -- come after every figure.
    tail = [r for r in by_speed if rb.sort_value(rb.facts_of(r), "speed") is None]
    assert by_speed[-len(tail) :] == tail
    by_fit = rb.sorted_rows(rows, "fit")
    first = by_fit[0].candidate.placement
    assert first is not None and first.budget.verdict in ("comfortable", "fits")


def test_the_rank_travels_with_the_row_under_every_sort() -> None:
    board = a_board()
    for key in rb.SORT_KEYS:
        for row in rb.sorted_rows(board.rows, key):
            original = next(r for r in board.rows if r.model_id == row.model_id)
            assert row.rank == original.rank


@pytest.mark.parametrize("text", ["speed:sideways", "vibes", "speed:asc:desc", ""])
def test_a_sort_that_is_not_one_is_refused_with_its_text(text: str) -> None:
    with pytest.raises(ValueError, match=re.escape(text) if text else "^$"):
        rb.parse_sort(text)


def test_the_fit_listing_refuses_a_sort_it_has_no_figure_for() -> None:
    assert "speed" not in rb.FIT_SORT_KEYS
    result = runner.invoke(app, ["--language", "en", "fit", "--sort", "speed"])
    assert result.exit_code == 1
    assert "context" in result.exception.render()  # type: ignore[union-attr]


# --- the filter terms ------------------------------------------------------------------


def test_the_box_reads_every_term_the_page_has_a_box_for() -> None:
    filters = rb.parse_filters(
        "qwen fit>=fits speed>=20 size<=30G card<=6G ram<=32G ctx>=32K quality>=70 runs=gpu have"
    )
    assert filters == rb.Filters(
        search="qwen",
        min_fit="fits",
        installed=True,
        mode="gpu",
        min_speed=20.0,
        max_size=30 * 1000**3,
        max_card=6 * 1000**3,
        max_ram=32 * 1000**3,
        min_context=32 * 1024,
        min_quality=70.0,
    )
    assert rb.parse_filters("") == rb.Filters()
    assert not rb.Filters().active and filters.active


def test_the_terms_read_back_to_the_filters_they_came_from() -> None:
    text = "qwen coder fit>=tight speed>=20 size<=30G ctx>=32768 runs=cpu have"
    filters = rb.parse_filters(text)
    assert rb.parse_filters(rb.filter_terms(filters)) == filters


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("fit>=snug", "comfortable"),
        ("runs=walk", "gpu"),
        ("speed>=fast", "speed>=fast"),
        ("size<=lots", "8G"),
        ("colour=red", "fit>=VERDICT"),
        ("speed<=20", "speed<=20"),
    ],
)
def test_a_term_that_cannot_be_read_is_refused_by_name_and_nothing_is_applied(
    text: str, named: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(named)):
        rb.parse_filters(f"qwen {text}")


def test_a_row_with_no_figure_is_not_hidden_by_a_threshold_on_it() -> None:
    unplaced = rb.facts_of(
        BoardRow(
            rank=None,
            model_id="x",
            name="X",
            quant="Q4",
            candidate=Candidate(model_id="x", quant="Q4", excluded_because="no header"),
        )
    )
    assert rb.passes(unplaced, rb.Filters(min_speed=20, max_size=1, min_context=10**6))
    # "Does it run" is a question about a placement, and a row with none has no answer.
    assert not rb.passes(unplaced, rb.Filters(min_fit="tight"))
    assert not rb.passes(unplaced, rb.Filters(installed=True))


def test_the_state_line_names_every_term_that_is_hiding_something() -> None:
    view = rb.View(
        sort="speed",
        descending=False,
        filters=rb.Filters(
            search="qwen", min_fit="tight", installed=True, mode="gpu", min_speed=20
        ),
    )
    line = rb.state_line(2, 40, view)
    assert "2" in line and "40" in line
    assert rb.sort_label("speed") in line and "reverse" in line
    assert rb.filter_label("tight") in line
    assert "qwen" in line and rb.mode_label("gpu") in line and "20" in line
    assert "already on this machine" in line
    plain = rb.state_line(2, 40, rb.View(filters=rb.Filters(installed=True)))
    assert plain.endswith("already on this machine.")


# --- the command line's view flags -----------------------------------------------------


def test_sort_reorders_the_json_rows_and_leaves_every_rank_alone() -> None:
    plain = json.loads(runner.invoke(app, ["--json", "recommend"]).output)
    fast = json.loads(runner.invoke(app, ["--json", "recommend", "--sort", "speed"]).output)
    speeds = [row["candidate"]["speed"]["gen_tps"] for row in fast["rows"]]
    assert speeds == sorted(speeds, reverse=True)
    assert sorted(row["rank"] for row in fast["rows"]) == [row["rank"] for row in plain["rows"]]
    assert fast["ranked_total"] == plain["ranked_total"]
    assert fast["excluded"] == plain["excluded"]


def _document(arguments: list[str]) -> dict[str, Any]:
    """A ``--json`` document with the one field that moves between runs taken out."""
    document: dict[str, Any] = json.loads(runner.invoke(app, arguments).output)
    document["machine"].pop("scanned_at")
    return document


def test_a_view_filter_never_touches_the_json() -> None:
    plain = _document(["--json", "recommend"])
    for flags in (
        ["--search", "qwen"],
        ["--installed"],
        ["--runs", "gpu"],
        ["--min-fit", "comfortable"],
        ["--hide-excluded"],
    ):
        assert _document(["--json", "recommend", *flags]) == plain, flags


def test_the_view_flags_draw_less_and_say_so() -> None:
    result = runner.invoke(
        app,
        ["--language", "en", "recommend", "--search", "qwen", "--runs", "gpu", "--min-fit", "fits"],
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "matching qwen" in text
    assert rb.mode_label("gpu") in text
    assert rb.filter_label("fits") in text
    # The request was not changed: the count that qualified is the same sentence as before.
    assert "that qualified" in text


def test_columns_draws_exactly_those_in_that_order_and_wide_draws_everything() -> None:
    board = a_board()
    chosen = cli_headings_of(
        drawn(
            rb.render_board(
                board, console_width=100, view=rb.View(columns=("model", "gen", "size"))
            ),
            100,
        )
    )
    assert chosen == [
        rb.column_heading("model"),
        rb.column_heading("gen"),
        rb.column_heading("size"),
    ]
    assert rb.column_heading("ram") not in cli_headings(board, 176)
    every = cli_headings_of(
        drawn(rb.render_board(board, console_width=176, view=rb.View(wide=True)), 176)
    )
    assert rb.column_heading("ram") in every and rb.column_heading("prompt") in every


def cli_headings_of(text: str) -> list[str]:
    header = next(
        line for line in text.splitlines() if "│" in line and rb.column_heading("model") in line
    )
    return [cell.strip() for cell in header.strip().strip("│").split("│")]


@pytest.mark.parametrize(
    ("flags", "listed"),
    [
        (["--sort", "sideways"], "quality"),
        (["--runs", "walk"], "moe-offload"),
        (["--columns", "model,nope"], "verdict"),
        (["--min-fit", "snug"], "comfortable"),
    ],
)
def test_a_view_flag_that_names_nothing_lists_what_it_could_have(
    flags: list[str], listed: str
) -> None:
    result = runner.invoke(app, ["recommend", *flags])
    assert result.exit_code == 1
    assert listed in result.exception.render()  # type: ignore[union-attr]


def test_an_installed_file_shows_as_have_and_survives_the_installed_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llamafit.models.llamacpp import LocalModel

    model = catalog().by_id["qwen3-coder-next"]
    name = model.sources[0].quants[0].files[0].rsplit("/", 1)[-1]
    local = LocalModel.model_validate({"path": f"/models/{name}", "bytes": 1})
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kw: report(local_models=[local]))
    result = runner.invoke(
        app, ["--language", "en", "recommend", "--installed", "--columns", "model,have"]
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "qwen3-coder-next" in text and "yes" in text
    assert "already on this machine" in text


# --- the bare command ------------------------------------------------------------------


def test_json_with_no_command_never_opens_a_full_screen_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entry, "interactive", lambda *a, **k: True)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a dashboard was opened over a --json document")

    monkeypatch.setattr(entry, "run_dashboard", refuse)
    result = runner.invoke(app, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["rows"]


def test_the_help_screen_says_what_the_bare_command_does() -> None:
    result = runner.invoke(app, ["--language", "en", "--help"])
    assert result.exit_code == 0
    assert "opens the terminal dashboard" in " ".join(result.output.split())


def test_the_fallback_board_is_the_dashboard_s_board() -> None:
    """Both go through one chooser; at one width they draw one set of headings."""
    result = runner.invoke(app, ["--language", "en", "--no-color"], env={"COLUMNS": "100"})
    assert result.exit_code == 0, result.output
    assert cli_headings_of(result.output) == cli_headings(a_board(), 100)


# --- what the web page is told ---------------------------------------------------------


def test_the_page_is_served_every_heading_in_the_vocabulary_s_order() -> None:
    headings = strings.column_headings()
    assert list(headings)[: len(rb.BOARD_ORDER)] == list(rb.BOARD_ORDER)
    for column in rb.BOARD_ORDER:
        assert headings[column] == rb.column_heading(column)
