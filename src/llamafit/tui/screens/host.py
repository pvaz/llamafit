# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The Host screen: the scan, probe by probe, and everything that limited it.

``llamafit system`` and ``llamafit doctor`` are two commands because a terminal prints one
thing at a time. A screen does not have that problem, so this is both of them: what was
found, what each probe did, what llama.cpp is, and what would unlock more.

Every table here is the command line's own renderable. That is not laziness; it is the
only way the two interfaces can be guaranteed to describe the same machine in the same
words, and the words are the point -- a bandwidth figure carries whether it was measured,
estimated or assumed, and a screen that redrew the table without that would be showing a
different, more confident number.

A probe that failed does not stop this screen: section 17 says a failed probe becomes a
``problems[]`` entry with a hint, never an abort, and what that buys is exactly this --
a machine where half the probes failed still has a Host screen, and it says which half.
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.message import Message

from llamafit.catalog.loader import Problem as CatalogProblem
from llamafit.cli.render import render_findings, render_host, render_llamacpp, render_probes
from llamafit.i18n import _, for_display, isolate, pgettext
from llamafit.services.doctor import diagnose
from llamafit.tui.keys import HOST_KEYS, bindable
from llamafit.tui.state import Dashboard
from llamafit.tui.widgets import RichPane


def catalog_problems_table(problems: list[CatalogProblem]) -> RenderableType | None:
    """Every catalog file the loader could not read, or ``None`` when it read them all.

    The command line writes one warning line to stderr and sends the reader to
    ``llamafit catalog validate`` for the detail, which is right for a command that has
    somewhere else to put a table and a stderr for the warning to go to. A dashboard has
    neither, and a model missing from the board because its file has a typo in it is
    exactly the kind of absence a reader would otherwise blame on their machine.
    """
    if not problems:
        return None
    table = Table(title=for_display(_("Catalog files with a problem")))
    table.add_column(for_display(pgettext("column heading", "File")))
    table.add_column(for_display(pgettext("column heading", "Where")))
    table.add_column(for_display(pgettext("column heading", "What")))
    for problem in problems:
        table.add_row(
            Text(for_display(isolate(problem.file))),
            Text(for_display(isolate(problem.location))),
            Text(for_display(problem.message)),
        )
    return table


class HostPane(VerticalScroll):
    """The scan, llama.cpp, every probe, every finding, and the catalog's own problems."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in bindable(HOST_KEYS)
    ]

    DEFAULT_CSS = """
    HostPane { height: 1fr; padding: 0 1; }
    HostPane > RichPane { margin-bottom: 1; }
    """

    class RescanRequested(Message):
        """The reader asked for the machine to be looked at again."""

    def __init__(self, dashboard: Dashboard, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id)
        self.dashboard = dashboard

    def compose(self) -> ComposeResult:
        """One pane per table: the scan, llama.cpp, the probes, the findings, the catalog."""
        yield RichPane(id="host-scan")
        yield RichPane(id="host-llamacpp")
        yield RichPane(id="host-probes")
        yield RichPane(id="host-findings")
        yield RichPane(id="host-catalog")

    def refresh_host(self) -> None:
        """Redraw from the last scan, saying so plainly when there has not been one."""
        report = self.dashboard.report
        host = self.dashboard.host
        if report is None or host is None:
            self._blank()
            return
        self.query_one("#host-scan", RichPane).show(render_host(host))
        self.query_one("#host-llamacpp", RichPane).show(render_llamacpp(report.llamacpp))
        self.query_one("#host-probes", RichPane).show(
            render_probes([*report.host.probes, *report.llamacpp.probes])
        )
        self.query_one("#host-findings", RichPane).show(render_findings(diagnose(report).findings))
        self._show_catalog_problems()

    def _blank(self) -> None:
        """What the screen says before a scan has come back, or after one could not be taken."""
        problem = self.dashboard.problem
        message = (
            problem.render() if problem is not None else _("This machine has not been scanned yet.")
        )
        style = "" if problem is None else "red"
        self.query_one("#host-scan", RichPane).show_text(message, style=style)
        for pane in ("#host-llamacpp", "#host-probes", "#host-findings"):
            self.query_one(pane, RichPane).show("")
        self._show_catalog_problems()

    def _show_catalog_problems(self) -> None:
        table = catalog_problems_table(self.dashboard.catalog_problems)
        self.query_one("#host-catalog", RichPane).show(table if table is not None else "")

    def action_rescan(self) -> None:
        """Look at the machine again, which the app does on a thread."""
        self.post_message(self.RescanRequested())
