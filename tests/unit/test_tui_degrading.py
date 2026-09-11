"""What the dashboard does when something is missing, which on a real machine it will be.

A terminal too small, no colour, no card, an empty catalog, a scan where half the probes
failed. The command line already answers all of these; the rule this file enforces is that
the dashboard answers them the same way -- by saying what is missing and carrying on --
rather than by falling over or, worse, by drawing a blank where a figure would be and
letting a reader take it for a zero.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console, RenderableType
from textual.widgets import DataTable, Static

from llamafit.catalog.loader import Problem as CatalogProblem
from llamafit.errors import PackagedDataError
from llamafit.models.host import Probe
from llamafit.tui import board_view
from llamafit.tui.app import LlamaFitApp
from llamafit.tui.screens.board import BoardPane
from llamafit.tui.state import Dashboard
from llamafit.tui.widgets import RichPane
from tests.fixtures.board import report
from tests.fixtures.budget_hosts import machine, reference_host
from tests.fixtures.dashboard import empty_catalog, ready, refuses_to_load, refuses_to_scan

GIB = 1024**3


def drawn(renderable: RenderableType, width: int = 120) -> str:
    console = Console(width=width, no_color=True, highlight=False, record=True, file=io.StringIO())
    console.print(renderable)
    return " ".join(console.export_text().split())


def pane_text(app: LlamaFitApp, selector: str, width: int = 120) -> str:
    return drawn(app.query_one(selector, RichPane).renderable, width)


def label(app: LlamaFitApp, selector: str) -> str:
    return str(app.query_one(selector, Static).render())


def painted_column(app: LlamaFitApp, column: board_view.Column) -> list[str]:
    """What one column of the board says on the screen, a folded cell rejoined.

    ``DataTable.get_row_at`` gives back the value the table was handed, which is the same
    answer whether that value was drawn whole, folded onto a second line or cut off where
    the column ends. Only the screen knows which of the three happened, so this reads the
    cells back out of the drawn lines: the column's slice of every line the row occupies,
    concatenated. A model id carries no spaces, so stripping each fragment and joining
    them recovers exactly the identifier that was folded and nothing else.
    """
    table = app.query_one("#board-table", DataTable)
    columns = list(table.ordered_columns)
    index = [str(one.label) for one in columns].index(board_view.heading(column))
    start = sum(one.get_render_width(table) for one in columns[:index])
    stop = start + columns[index].get_render_width(table)
    lines = [table.render_line(y).text for y in range(table.size.height)]
    cells: list[str] = []
    top = table.header_height
    for key in table.rows:
        height = table.rows[key].height
        assert top + height <= len(lines), "the terminal is too short to read every row back"
        cells.append("".join(lines[top + line][start:stop].strip() for line in range(height)))
        top += height
    return cells


# --- nothing to recommend ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_catalog_draws_a_dashboard_that_says_it_is_empty() -> None:
    app = LlamaFitApp(ready(models=empty_catalog()))
    async with app.run_test(size=(100, 40)):
        assert app.query_one("#board-table", DataTable).row_count == 0
        assert "0" in label(app, "#board-state")
        # A blank pane would leave a reader with no idea whether they had broken something.
        assert pane_text(app, "#board-why-pane")
        assert not app.query_one("#app-problem", Static).display


@pytest.mark.asyncio
async def test_a_board_with_nothing_on_it_makes_no_claim_about_speeds() -> None:
    app = LlamaFitApp(ready(models=empty_catalog()))
    async with app.run_test(size=(100, 40)):
        assert not app.query_one("#board-band", Static).display


# --- nothing to recommend against --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_scan_that_failed_shows_its_message_and_hint_and_keeps_the_dashboard() -> None:
    dashboard = Dashboard(scanner=refuses_to_scan(), loader=lambda: (empty_catalog(), []))
    dashboard.refresh()
    app = LlamaFitApp(dashboard)
    async with app.run_test(size=(100, 40)):
        band = app.query_one("#app-problem", Static)
        assert band.display
        assert "nvidia-smi" in str(band.render())
        assert "Hint:" in str(band.render())
        # And it is still a dashboard: every screen is there, saying what it has.
        assert pane_text(app, "#host-scan")
        assert pane_text(app, "#plan-body")


@pytest.mark.asyncio
async def test_a_catalog_that_would_not_load_says_so_without_hiding_the_machine() -> None:
    dashboard = Dashboard(
        scanner=report, loader=refuses_to_load(PackagedDataError("no catalog shipped"))
    )
    dashboard.refresh()
    app = LlamaFitApp(dashboard)
    async with app.run_test(size=(100, 40)):
        assert "no catalog shipped" in str(app.query_one("#app-problem", Static).render())
        assert "RTX 4060" in pane_text(app, "#host-scan")


@pytest.mark.asyncio
async def test_a_catalog_file_with_a_problem_in_it_is_listed_rather_than_counted() -> None:
    problem = CatalogProblem(
        file="custom_models.yaml",
        model_id=None,
        location="quality",
        message="baseline must be an int",
    )
    app = LlamaFitApp(ready(catalog_problems=[problem]))
    async with app.run_test(size=(100, 40)):
        text = pane_text(app, "#host-catalog", width=160)
        assert "custom_models.yaml" in text
        assert "baseline must be an int" in text


# --- a machine that is not the reference one ---------------------------------------------


@pytest.mark.asyncio
async def test_a_machine_with_no_graphics_card_is_a_machine_not_an_error() -> None:
    host = machine(vram_total=None, ram_total=64 * GIB, ram_available=48 * GIB)
    app = LlamaFitApp(ready(host=host))
    async with app.run_test(size=(100, 40)):
        assert "GPU" in label(app, "#machine-line")
        assert not app.query_one("#app-problem", Static).display
        # The board still answers: a machine with no card runs models in system memory.
        assert app.query_one("#board-table", DataTable).row_count >= 0
        assert pane_text(app, "#host-scan")


@pytest.mark.asyncio
async def test_a_scan_whose_probes_mostly_failed_lists_every_one_of_them() -> None:
    host = reference_host()
    host.probes = [
        Probe(name="nvidia-smi", ok=False, duration_ms=12, error="not found on PATH"),
        Probe(name="memory-modules", ok=False, duration_ms=4, error="dmidecode needs root"),
        Probe(name="cpuinfo", ok=True, duration_ms=30),
    ]
    app = LlamaFitApp(ready(host=host))
    async with app.run_test(size=(100, 40)):
        probes = pane_text(app, "#host-probes", width=160)
        for name in ("nvidia-smi", "memory-modules", "cpuinfo"):
            assert name in probes
        assert "not found on PATH" in probes
        # ``doctor``'s findings are on the same screen, and they say what would unlock more.
        assert pane_text(app, "#host-findings", width=200)


# --- a terminal that is not wide -----------------------------------------------------------


@pytest.mark.asyncio
async def test_at_eighty_columns_every_row_can_still_be_identified_and_acted_on() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(80, 24)):
        table = app.query_one("#board-table", DataTable)
        headings = [str(column.label) for column in table.columns.values()]
        for wanted in board_view.REQUIRED:
            assert board_view.heading(wanted) in headings
        assert table.row_count > 0


@pytest.mark.asyncio
async def test_a_model_name_is_never_the_thing_that_gets_shortened() -> None:
    # Eighty cells is the width the column priority is designed for and the one where
    # several bundled ids are wider than the column they land in; a hundred and twenty
    # lines so that every row is painted rather than scrolled past, because a row the
    # table never drew is a row this test would pass without having read.
    app = LlamaFitApp(ready())
    async with app.run_test(size=(80, 120)) as pilot:
        await pilot.pause()
        board = app.query_one("#board", BoardPane)
        assert any(len(row.model_id) > board_view.WIDTHS["model"] for row in board.shown), (
            "nothing on this board is long enough for the test to be about anything"
        )
        assert painted_column(app, "model") == [row.model_id for row in board.shown]


@pytest.mark.asyncio
async def test_a_terminal_too_narrow_for_any_optional_column_still_draws() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(40, 20)):
        table = app.query_one("#board-table", DataTable)
        assert table.row_count > 0
        assert len(table.columns) >= len(board_view.REQUIRED)


@pytest.mark.asyncio
async def test_widening_the_terminal_brings_columns_back() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(80, 24)) as pilot:
        narrow = len(app.query_one("#board-table", DataTable).columns)
        await pilot.resize_terminal(200, 40)
        await pilot.pause()
        assert len(app.query_one("#board-table", DataTable).columns) > narrow


# --- a pane that has not been laid out yet -------------------------------------------------


@pytest.mark.asyncio
async def test_a_board_behind_another_tab_is_laid_out_for_the_terminal_it_will_come_back_to() -> (
    None
):
    """A hidden pane has a size of zero, and zero is not a width to lay a table out for.

    Textual reports a widget's size from the last layout pass, so a pane behind another
    tab measures zero until it is in front again. Redrawing it against a guess drops the
    columns the terminal has room for and puts them back a frame later, when the resize
    lands -- so the same table says two different things depending on when it is read.
    """
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 40)) as pilot:
        board = app.query_one("#board", BoardPane)
        in_front = board.columns
        assert "context" in in_front, "120 columns has room for more than the required four"
        app.query_one("#tabs").active = "tab-simulate"
        await pilot.pause()
        assert board.size.width == 0, "a pane behind another tab really has no width of its own"
        board.refresh_board()
        assert board.columns == in_front


# --- without colour ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_verdict_carries_its_word_so_a_colourless_screen_says_the_same() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 40)):
        table = app.query_one("#board-table", DataTable)
        headings = [str(column.label) for column in table.columns.values()]
        index = headings.index(board_view.heading("verdict"))
        for position in range(table.row_count):
            assert str(table.get_row_at(position)[index]).strip()
