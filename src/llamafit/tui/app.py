# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The dashboard: a header, a tab bar, five screens and one key bar.

Section 13.2's shape. What the code here actually does is small, and deliberately so --
every screen reads a :class:`~llamafit.tui.state.Dashboard`, the dashboard is the only
thing that calls a service, and this module is what wires the two together and redraws
everything when the answer changes.

Three decisions are worth reading.

**The scan runs on a thread.** Probing a machine takes seconds -- a memory bandwidth
measurement, a vendor tool, an HTTP request to a port that may not answer -- and a screen
frozen for those seconds is a screen a newcomer closes. So the first paint happens against
whatever the dashboard already holds, the scan runs in a worker, and the screens are
redrawn when it lands. A dashboard handed a scan that has already been taken does no work
at start-up at all, which is what makes the whole application testable without a probe.

**The key bar is drawn, not inherited.** Textual's own footer would show each binding's
description, and a binding is built while this module is imported -- before ``--language``
has been read. A bar built at draw time can be in the reader's language; a class attribute
cannot, and it would have been English for the life of the process with nothing failing.

**A failure is drawn too.** The scan can fail, the packaged catalog can be missing, a
substitution can be refused. Each of those is a band under the header carrying the message
and the hint the error was raised with, and the rest of the screen keeps whatever it still
has. Section 17 asks the command line to render an error without a traceback; a dashboard
has the same duty and one more, which is to remain a dashboard afterwards.
"""

from __future__ import annotations

from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Header, Static, TabbedContent, TabPane

from llamafit.i18n import _
from llamafit.tui import keys
from llamafit.tui.screens.board import BoardPane
from llamafit.tui.screens.help import KeyMapScreen
from llamafit.tui.screens.host import HostPane
from llamafit.tui.screens.needs import NeedsPane
from llamafit.tui.screens.plan import PlanPane
from llamafit.tui.screens.simulate import SimulatePane
from llamafit.tui.state import Dashboard
from llamafit.tui.widgets import MachineBar

_TAB_KEYS: dict[str, tuple[keys.Key, ...]] = {
    "tab-board": keys.BOARD_KEYS,
    "tab-needs": keys.NEEDS_KEYS,
    "tab-host": keys.HOST_KEYS,
    "tab-plan": keys.PLAN_KEYS,
    "tab-simulate": keys.SIMULATE_KEYS,
}

_TAB_FOCUS: dict[str, str] = {
    "tab-board": "#board-table",
    "tab-needs": "#needs",
    "tab-host": "#host",
    "tab-plan": "#plan",
    "tab-simulate": "#simulate",
}
"""What takes the cursor when a screen comes to the front.

Textual keeps the tab bar and the focus in step: a widget focused inside a hidden pane
brings that pane forward, and a pane brought forward while the cursor is still on the last
one is pushed straight back. Moving the focus with the tab is therefore not decoration --
without it a tab switch does not stick, and the keys of the screen in front of the reader
would belong to the screen behind it.
"""


class LlamaFitApp(App[None]):
    """The terminal dashboard ``llamafit`` opens with no arguments."""

    CSS_PATH = "styles.tcss"
    TITLE = "LlamaFit"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in keys.bindable(keys.GLOBAL_KEYS)
    ]

    def __init__(self, dashboard: Dashboard | None = None) -> None:
        super().__init__()
        self.dashboard = dashboard or Dashboard()

    # --- the screen --------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """The header, the machine band, the five screens and the key bar."""
        yield Header()
        yield MachineBar(id="machine-bar")
        yield Static("", id="app-problem", markup=False)
        with TabbedContent(id="tabs"):
            with TabPane(_("Board"), id="tab-board"):
                yield BoardPane(self.dashboard, id="board")
            with TabPane(_("Needs"), id="tab-needs"):
                yield NeedsPane(self.dashboard, id="needs")
            with TabPane(_("Host"), id="tab-host"):
                yield HostPane(self.dashboard, id="host")
            with TabPane(_("Plan"), id="tab-plan"):
                yield PlanPane(self.dashboard, id="plan")
            with TabPane(_("Simulate"), id="tab-simulate"):
                yield SimulatePane(self.dashboard, id="simulate")
        yield Static("", id="key-bar", markup=False)

    def on_mount(self) -> None:
        """Draw whatever is already known, then go and find out the rest."""
        self.repaint()
        self.focus_active_screen()
        if self.dashboard.report is None and self.dashboard.catalog is None:
            self.rescan()

    # --- redrawing ---------------------------------------------------------------------

    def repaint(self) -> None:
        """Redraw every screen from what the dashboard currently holds."""
        report = self.dashboard.report
        self.query_one("#machine-bar", MachineBar).show(
            self.dashboard.host, None if report is None else report.llamacpp
        )
        self._draw_problem()
        self.query_one("#board", BoardPane).refresh_board()
        self.query_one("#needs", NeedsPane).refresh_needs()
        self.query_one("#host", HostPane).refresh_host()
        self.query_one("#plan", PlanPane).refresh_plan()
        self.query_one("#simulate", SimulatePane).refresh_profiles()
        self._draw_key_bar()

    def _draw_problem(self) -> None:
        """The band that carries whatever went wrong, and nothing when nothing has."""
        problem = self.dashboard.problem
        band = self.query_one("#app-problem", Static)
        band.update("" if problem is None else problem.render())
        band.display = problem is not None

    def _draw_key_bar(self) -> None:
        """The keys that work here, in the reader's language, on the screen they work on."""
        active = self.query_one("#tabs", TabbedContent).active
        here = _TAB_KEYS.get(active, ())
        self.query_one("#key-bar", Static).update(keys.bar((*here, *keys.GLOBAL_KEYS)))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Follow the tab bar: the key bar and the cursor are about the screen in front of you."""
        event.stop()
        self._draw_key_bar()
        self.focus_active_screen()

    def focus_active_screen(self) -> None:
        """Put the cursor on the screen in front of the reader, if it is not there already."""
        active = self.query_one("#tabs", TabbedContent).active
        selector = _TAB_FOCUS.get(active)
        if selector is None:  # pragma: no cover - every tab is in the table
            return
        wanted = self.query_one(selector)
        focused = self.focused
        if focused is not None and wanted in focused.ancestors_with_self:
            return
        self.set_focus(wanted)

    def go_to(self, tab: str) -> None:
        """Bring one screen to the front and move the cursor onto it."""
        self.query_one("#tabs", TabbedContent).active = tab
        self.focus_active_screen()

    # --- the scan ----------------------------------------------------------------------

    @work(thread=True, exclusive=True, group="scan")
    def rescan(self) -> None:
        """Scan the machine and rank the catalog against it, off the interface thread."""
        self.dashboard.refresh()
        self.call_from_thread(self.repaint)

    # --- what the screens ask for ------------------------------------------------------

    def on_board_pane_plan_requested(self, event: BoardPane.PlanRequested) -> None:
        """Show the plan for the row the Board was on, and go to the screen showing it."""
        event.stop()
        self.query_one("#plan", PlanPane).show_row(event.row)
        self.go_to("tab-plan")

    def on_needs_pane_applied(self, event: NeedsPane.Applied) -> None:
        """The request changed: redraw, and go back to the answer it changed."""
        event.stop()
        self.repaint()
        self.go_to("tab-board")

    def on_simulate_pane_applied(self, event: SimulatePane.Applied) -> None:
        """The machine changed: redraw everything, since everything was about the old one."""
        event.stop()
        self.repaint()
        self.go_to("tab-board")

    def on_host_pane_rescan_requested(self, event: HostPane.RescanRequested) -> None:
        """Look at the machine again."""
        event.stop()
        self.rescan()

    # --- global keys -------------------------------------------------------------------

    def action_help(self) -> None:
        """Show every key, under the screen it belongs to."""
        self.push_screen(KeyMapScreen())

    def action_next_theme(self) -> None:
        """Cycle through the themes Textual offers, which is what ``t`` does everywhere."""
        names = sorted(self.available_themes)
        if not names:  # pragma: no cover - Textual always registers its own
            return
        here = names.index(self.theme) if self.theme in names else -1
        self.theme = names[(here + 1) % len(names)]
