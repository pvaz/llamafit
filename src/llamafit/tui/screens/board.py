# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The Board: what this machine can run, why, and how to get a command line for it.

This is the screen ``llamafit`` opens on, and the first thing anybody sees. Everything
about it is arranged around one sentence: somebody who has never used llama.cpp should be
able to open it, see what their machine can run, pick something, and get a command line,
without reading a flag or a manual.

So three lines come before the table, and none of them is a footnote.

The **first says what to do**: move with the arrow keys, press Enter for why a row is where
it is, press P for the command line that runs it. It is at the top rather than behind a
question mark, because a person who does not know a program can do something does not know
to ask it whether it can.

The **second says what a speed is**. Nothing has been benchmarked on any machine yet, so
every figure in the ``Tok/s`` column came out of a formula on default constants, and that
sentence stays on the screen instead of sitting under the table where the eye does not go.

The **third says what is on the screen**: how many of how many, in what order, under which
filter. Four of this screen's keys can hide a row, and a reader who cannot find the model
they came for is owed the reason.

Below the table is the explanation, open by default. Section 12.3 says a recommendation
expands into the inputs that produced it; on a command line that costs a page per row and
is behind a flag, and here it costs nothing, so here it is not behind anything.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Input, Static

from llamafit.i18n import _
from llamafit.services.recommend import BoardRow
from llamafit.tui import board_view, explain
from llamafit.tui.keys import BOARD_KEYS, bindable
from llamafit.tui.state import Dashboard, Request
from llamafit.tui.summary import speed_band
from llamafit.tui.widgets import RichPane


class BoardPane(Vertical):
    """The ranked table, the line that says what to do with it, and the explanation."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in bindable(BOARD_KEYS)
    ]

    DEFAULT_CSS = """
    BoardPane { height: 1fr; }
    BoardPane > #board-hint { height: auto; padding: 0 1; text-style: bold; }
    BoardPane > #board-band { height: auto; padding: 0 1; color: $warning; }
    BoardPane > #board-state { height: auto; padding: 0 1; color: $text-muted; }
    BoardPane > #board-search { display: none; }
    BoardPane > #board-search.searching { display: block; }
    BoardPane > #board-table { height: 2fr; min-height: 4; }
    BoardPane > #board-why { height: 3fr; min-height: 3; border-top: solid $panel; }
    BoardPane > #board-why.hidden { display: none; }
    """

    class PlanRequested(Message):
        """The reader asked for the command line that runs the row under the cursor."""

        def __init__(self, row: BoardRow) -> None:
            super().__init__()
            self.row = row

    def __init__(self, dashboard: Dashboard, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id)
        self.dashboard = dashboard
        self.sort: board_view.Sort = "score"
        self.fit: board_view.FitFilter = "all"
        self.installed_only = False
        self.search = ""
        self.columns: tuple[board_view.Column, ...] = ()
        self.shown: list[BoardRow] = []

    def compose(self) -> ComposeResult:
        """The three lines that say what to do, the table, and the explanation under it."""
        yield Static(
            _(
                "Move with the up and down arrows. Enter says why a row is where it is; "
                "p gives the command line that runs it."
            ),
            id="board-hint",
            markup=False,
        )
        yield Static("", id="board-band", markup=False)
        yield Static("", id="board-state", markup=False)
        yield Input(placeholder=_("Part of a model name, then Enter"), id="board-search")
        yield DataTable(id="board-table", cursor_type="row", zebra_stripes=True)
        with VerticalScroll(id="board-why"):
            yield RichPane(explain.nothing_selected(), id="board-why-pane")

    # --- drawing -----------------------------------------------------------------------

    def refresh_board(self) -> None:
        """Redraw everything from whatever the dashboard currently holds.

        Called after a scan, after the request changes and after the machine is
        substituted. It asks no service anything; it reads the last answer one gave.
        """
        self._draw_band()
        self._draw_table()
        self._draw_why()

    def _width(self) -> int:
        """How many cells the table has to lay itself out in.

        ``self.size`` reports the last layout pass, and it is zero twice over: before the
        first pass, and for as long as this pane sits behind another tab. Neither zero is
        a width. Standing a constant in for one lays the table out for a terminal nobody
        is looking at and then jumps to the real one a frame later, when the resize
        arrives -- which is a column appearing and disappearing under a reader who did
        nothing, and a redraw that says something different depending on when it is read.

        The terminal's width was never the unknown. The application is told it before a
        widget is mounted and again before any widget hears it has been resized, and
        ``TabPane { padding: 0 }`` in ``styles.tcss`` hands this pane the whole of it. So
        the measured width answers when there is one and the terminal's answers when
        there is not, and both give the same number -- which is what makes a redraw from
        behind another tab draw the same table as a redraw in front of it.
        """
        return self.size.width or self.app.size.width

    def _wanted_columns(self) -> tuple[board_view.Column, ...]:
        """Which columns this width and this board's speed labels ask for."""
        return board_view.columns_for_width(
            self._width(), uniform_confidence=board_view.one_confidence(self.dashboard.rows)
        )

    def _draw_band(self) -> None:
        rows = self.dashboard.rows
        band = self.query_one("#board-band", Static)
        band.update(speed_band(rows))
        band.display = bool(rows)

    def _draw_table(self) -> None:
        table = self.query_one("#board-table", DataTable)
        rows = self.dashboard.rows
        self.shown = board_view.visible_rows(
            rows,
            search=self.search,
            fit=self.fit,
            installed_only=self.installed_only,
            sort=self.sort,
        )
        self.columns = self._wanted_columns()
        table.clear(columns=True)
        for column in self.columns:
            table.add_column(board_view.heading(column), width=board_view.WIDTHS[column])
        for row in self.shown:
            # ``height=None`` is Textual's auto-height, and it is the whole of how this
            # table keeps its promise never to truncate a model's name. A row of a fixed
            # height is drawn with wrapping turned off, so a cell wider than its column is
            # cut where the column ends with nothing on the screen to say it was, and
            # ``nemotron-3.5-lightning-30b-a3b`` arrives as ``nemotron-3.5-lightning-3``,
            # which is not that model or any other. Auto-height folds the cell onto a
            # second line instead, the way the command line's board already folds it, and
            # a row grows only when something in it actually needed the room.
            table.add_row(*[board_view.cell(row, column) for column in self.columns], height=None)
        self.query_one("#board-state", Static).update(
            board_view.state_line(
                len(self.shown),
                len(rows),
                sort=self.sort,
                fit=self.fit,
                installed_only=self.installed_only,
            )
        )

    def _draw_why(self) -> None:
        pane = self.query_one("#board-why-pane", RichPane)
        board = self.dashboard.board
        row = self.selected_row
        if row is not None and board is not None:
            pane.show(explain.explanation(row, board, self.dashboard.catalog))
        elif self.dashboard.rows:
            pane.show(explain.nothing_selected())
        else:
            # Nothing is ranked. The answer to "why is there nothing here" is the whole of
            # the not-ranked table, so it is what the pane shows rather than a blank.
            pane.show(explain.not_ranked(board))

    @property
    def selected_row(self) -> BoardRow | None:
        """The row under the cursor, or ``None`` when there is nothing to put one on."""
        index = self.query_one("#board-table", DataTable).cursor_row
        if not self.shown or not 0 <= index < len(self.shown):
            return None
        return self.shown[index]

    # --- what the reader does ----------------------------------------------------------

    def on_resize(self) -> None:
        """Admit or drop columns as the terminal changes width."""
        if self._wanted_columns() != self.columns:
            self._draw_table()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Follow the cursor: the explanation is always about the row it is on."""
        event.stop()
        self._draw_why()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter shows the explanation, opening the pane when it had been closed."""
        event.stop()
        self.query_one("#board-why").set_class(False, "hidden")
        self._draw_why()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply the typed search and put the cursor back on the table."""
        event.stop()
        self.search = event.value
        self.query_one("#board-search").set_class(False, "searching")
        self._draw_table()
        self._draw_why()
        self.query_one("#board-table", DataTable).focus()

    def action_search(self) -> None:
        """Open the search box."""
        box = self.query_one("#board-search", Input)
        box.set_class(True, "searching")
        box.focus()

    def action_cycle_fit(self) -> None:
        """Show everything, then only what runs, then only what has room to spare."""
        filters = board_view.FILTERS
        self.fit = filters[(filters.index(self.fit) + 1) % len(filters)]
        self._draw_table()
        self._draw_why()

    def action_cycle_sort(self) -> None:
        """Reorder the rows; the rank column keeps saying where the ranking put each one."""
        sorts = board_view.SORTS
        self.sort = sorts[(sorts.index(self.sort) + 1) % len(sorts)]
        self._draw_table()
        self._draw_why()

    def action_toggle_installed(self) -> None:
        """Show only the quantisations whose file llama.cpp already has."""
        self.installed_only = not self.installed_only
        self._draw_table()
        self._draw_why()

    def action_toggle_quants(self) -> None:
        """Show every quantisation of a model rather than the one that scores best here."""
        request = self.dashboard.request
        self.dashboard.ask(
            Request(
                needs=request.needs,
                prefer=request.prefer,
                licenses=request.licenses,
                vision=request.vision,
                all_quants=not request.all_quants,
            )
        )
        self.refresh_board()

    def action_toggle_why(self) -> None:
        """Close the explanation to give the table the whole screen, or open it again."""
        self.query_one("#board-why").toggle_class("hidden")

    def action_not_ranked(self) -> None:
        """Show the candidates that did not qualify, each with the reason it did not."""
        self.query_one("#board-why").set_class(False, "hidden")
        self.query_one("#board-why-pane", RichPane).show(explain.not_ranked(self.dashboard.board))

    def action_plan(self) -> None:
        """Ask for the placement, the budget and the command line for the selected row."""
        row = self.selected_row
        if row is not None:
            self.post_message(self.PlanRequested(row))
