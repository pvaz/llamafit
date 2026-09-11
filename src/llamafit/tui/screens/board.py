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
filter. Half of this screen's keys can hide a row or move it, and a reader who cannot find
the model they came for is owed the reason.

Below the table is the explanation, open by default. Section 12.3 says a recommendation
expands into the inputs that produced it; on a command line that costs a page per row and
is behind a flag, and here it costs nothing, so here it is not behind anything.

The table is the command line's board, drawn by Textual instead of Rich: the same
columns at the same width, chosen by :func:`llamafit.cli.render_board.board_columns`;
the same cells from :func:`~llamafit.cli.render_board.board_cell`; the same order, filters
and state line from the same :class:`~llamafit.cli.render_board.View`. Nothing here
decides what a column is called or which one a narrow terminal loses first, and that is
what keeps this screen from drifting away from the board a pipe prints.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Input, Static

from llamafit.cli.render_board import (
    BOARD_ORDER,
    FILTER_TERMS,
    SORT_KEYS,
    Column,
    View,
    board_cell,
    board_columns,
    column_budget,
    column_heading,
    column_widths,
    default_descending,
    exclusion_tag,
    facts_of,
    filter_terms,
    mixed_confidence,
    parse_filters,
    state_line,
    visible_rows,
)
from llamafit.i18n import _
from llamafit.models.plan import Verdict
from llamafit.services.recommend import BoardRow
from llamafit.tui import explain
from llamafit.tui.keys import BOARD_KEYS, bindable
from llamafit.tui.state import Dashboard, Request
from llamafit.tui.summary import speed_band
from llamafit.tui.widgets import RichPane

_FIT_CYCLE: tuple[Verdict | None, ...] = (None, "tight", "fits", "comfortable")
"""What ``f`` cycles through: every candidate, then each floor a verdict can be, best last.

The ``/`` box can set a floor this cycle does not visit -- ``fit>=too-tight`` is a legal
term -- so the key steps from wherever the filter is to the cycle's start rather than
insisting the floor was one of its own.
"""

_NARROWEST_MODEL = 16
"""How far the model column gives way to a wide tag before the names fold too hard to read."""


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
    BoardPane > #board-error { height: auto; padding: 0 1; color: $error; }
    BoardPane > #board-terms { display: none; height: auto; padding: 0 1; color: $text-muted; }
    BoardPane > #board-terms.searching { display: block; }
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
        self.view = View()
        self.columns: tuple[Column, ...] = ()
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
        yield Static("", id="board-error", markup=False)
        yield Static(
            _(
                "Terms: %(terms)s; anything else is text to look for. Enter with the box "
                "empty clears them."
            )
            % {"terms": ", ".join(FILTER_TERMS)},
            id="board-terms",
            markup=False,
        )
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

    def _wanted_columns(self) -> tuple[Column, ...]:
        """Which columns this width and this board's speed labels ask for.

        The command line's own chooser with the command line's own budget, so that the
        board a pipe prints and the board this screen draws have the same columns at the
        same width. ``c`` asks for every column instead, and the table scrolls sideways.
        """
        if self.view.wide:
            return BOARD_ORDER
        return board_columns(
            self._width(),
            mixed_confidence=mixed_confidence(self.dashboard.rows),
            budget=column_budget(self.dashboard.catalog),
        )

    def _carried(self) -> list[BoardRow]:
        """Every row the table may draw: the ranked ones, then the unranked ones if shown."""
        board = self.dashboard.board
        if board is None:
            return []
        return [*board.rows, *(board.excluded if self.view.excluded else [])]

    def _draw_band(self) -> None:
        rows = self.dashboard.rows
        band = self.query_one("#board-band", Static)
        band.update(speed_band(rows))
        band.display = bool(rows)

    def _draw_table(self) -> None:
        table = self.query_one("#board-table", DataTable)
        carried = self._carried()
        self.shown = visible_rows(carried, self.view)
        self.columns = self._wanted_columns()
        widths = column_widths(column_budget(self.dashboard.catalog))
        # A tag wider than the score column widens it, and the model column gives up
        # the difference, which is what Rich does on the command line's board: a tag
        # costs a few more folded names and never a column.
        facts = [facts_of(row) for row in self.shown]
        tags = [exclusion_tag(one) or "" for one in facts if not one.ranked]
        extra = max([cell_len(tag) for tag in tags] + [widths["score"]]) - widths["score"]
        widths["score"] += extra
        widths["model"] = max(_NARROWEST_MODEL, widths["model"] - extra)
        table.clear(columns=True)
        for column in self.columns:
            table.add_column(column_heading(column), width=widths[column], key=column)
        for one in facts:
            # ``height=None`` is Textual's auto-height, and it is the whole of how this
            # table keeps its promise never to truncate a model's name. A row of a fixed
            # height is drawn with wrapping turned off, so a cell wider than its column is
            # cut where the column ends with nothing on the screen to say it was, and
            # ``nemotron-3.5-lightning-30b-a3b`` arrives as ``nemotron-3.5-lightning-3``,
            # which is not that model or any other. Auto-height folds the cell onto a
            # second line instead, the way the command line's board already folds it, and
            # a row grows only when something in it actually needed the room.
            table.add_row(*[board_cell(one, column) for column in self.columns], height=None)
        self.query_one("#board-state", Static).update(
            state_line(len(self.shown), len(carried), self.view)
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
            # the reasons list, so it is what the pane shows rather than a blank.
            pane.show(explain.not_ranked(board))

    @property
    def selected_row(self) -> BoardRow | None:
        """The row under the cursor, or ``None`` when there is nothing to put one on."""
        index = self.query_one("#board-table", DataTable).cursor_row
        if not self.shown or not 0 <= index < len(self.shown):
            return None
        return self.shown[index]

    def _redraw(self) -> None:
        """The table and the explanation, after the view changed."""
        self._draw_table()
        self._draw_why()

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
        """Apply the typed terms and put the cursor back on the table.

        A term that cannot be read is said under the box and nothing is applied: a
        filter half applied hides rows for a reason the state line could not name.
        """
        event.stop()
        error = self.query_one("#board-error", Static)
        try:
            filters = parse_filters(event.value)
        except ValueError as exc:
            error.update(str(exc))
            return
        error.update("")
        self.view = replace(self.view, filters=filters)
        self.query_one("#board-search").set_class(False, "searching")
        self.query_one("#board-terms").set_class(False, "searching")
        self._redraw()
        self.query_one("#board-table", DataTable).focus()

    def action_search(self) -> None:
        """Open the filter box, showing the terms already in force so they can be edited."""
        box = self.query_one("#board-search", Input)
        box.value = filter_terms(self.view.filters)
        box.set_class(True, "searching")
        self.query_one("#board-terms").set_class(True, "searching")
        box.focus()

    def action_cycle_fit(self) -> None:
        """Show everything, then only what runs, then only what has room to spare."""
        current = self.view.filters.min_fit
        here = _FIT_CYCLE.index(current) if current in _FIT_CYCLE else -1
        wanted = _FIT_CYCLE[(here + 1) % len(_FIT_CYCLE)]
        self.view = replace(self.view, filters=replace(self.view.filters, min_fit=wanted))
        self._redraw()

    def _sort_by(self, step: int) -> None:
        """Move along the sort keys; the direction goes back to the new key's own."""
        here = SORT_KEYS.index(self.view.sort)
        self.view = replace(
            self.view, sort=SORT_KEYS[(here + step) % len(SORT_KEYS)], descending=None
        )
        self._redraw()

    def action_cycle_sort(self) -> None:
        """Reorder the rows; the rank column keeps saying where the ranking put each one."""
        self._sort_by(1)

    def action_previous_sort(self) -> None:
        """The sort key before this one, for a reader who went one too far."""
        self._sort_by(-1)

    def action_reverse_sort(self) -> None:
        """Turn the current order round: smallest first, or largest again."""
        descending = self.view.descending
        if descending is None:
            descending = default_descending(self.view.sort)
        self.view = replace(self.view, descending=not descending)
        self._redraw()

    def action_toggle_installed(self) -> None:
        """Show only the quantisations whose file llama.cpp already has."""
        filters = self.view.filters
        self.view = replace(self.view, filters=replace(filters, installed=not filters.installed))
        self._redraw()

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

    def action_toggle_excluded(self) -> None:
        """Take the candidates that were not ranked off the table, or put them back."""
        self.view = replace(self.view, excluded=not self.view.excluded)
        self._redraw()

    def action_toggle_columns(self) -> None:
        """Every column whatever the width, or the ones the width admits again.

        With every column the table is wider than the terminal and scrolls sideways
        under the left and right arrows, which Textual binds for a row cursor.
        """
        self.view = replace(self.view, wide=not self.view.wide)
        self._redraw()

    def action_toggle_why(self) -> None:
        """Close the explanation to give the table the whole screen, or open it again."""
        self.query_one("#board-why").toggle_class("hidden")

    def action_not_ranked(self) -> None:
        """Go to the first candidate that did not qualify, showing them if they were hidden.

        The explanation pane follows the cursor, so landing on an unranked row reads its
        reason with no further key. When there is no such row the pane shows the reasons
        list itself, which for an empty board is the whole answer.
        """
        if not self.view.excluded:
            self.view = replace(self.view, excluded=True)
            self._draw_table()
        first = next((i for i, row in enumerate(self.shown) if row.rank is None), None)
        self.query_one("#board-why").set_class(False, "hidden")
        if first is None:
            self.query_one("#board-why-pane", RichPane).show(
                explain.not_ranked(self.dashboard.board)
            )
            return
        self.query_one("#board-table", DataTable).move_cursor(row=first, scroll=True)
        self._draw_why()

    def action_plan(self) -> None:
        """Ask for the placement, the budget and the command line for the selected row."""
        row = self.selected_row
        if row is not None:
            self.post_message(self.PlanRequested(row))
