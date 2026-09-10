# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The Plan: one row placed, costed, and turned into a command line somebody can paste.

This is where a newcomer's five minutes end, and it is the screen where a mistake costs
the most: everything before it can be argued with, and the last line of this one is copied
into a terminal by somebody who has already decided to trust it.

So it prints ``llamafit plan``'s own output, whole. The budget line by line with the source
of every line, the context ladder with what each rung costs the card and what happens
there, where a token's time goes, the runs the catalog records beside the estimate rather
than in place of it, whether the file is on this machine, and then the command.

Three keys change the plan rather than the view. ``+`` and ``-`` move along the context
ladder the planner produced -- not along a step this screen invented, because a context
this machine cannot stand on is not a choice -- and each move replans from scratch, since
context moves the cache, the compute buffer and with them the verdict. ``v`` puts the
vision projector back or takes it away, which is memory a reader may want for context
instead.
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import Group, RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll

from llamafit.cli.render_board import render_plan
from llamafit.errors import LlamaFitError
from llamafit.i18n import _, isolate
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.services.plan import plan_report
from llamafit.services.recommend import BoardRow, quant_entries
from llamafit.tui.format import context as format_context
from llamafit.tui.keys import PLAN_KEYS, bindable
from llamafit.tui.state import Dashboard
from llamafit.tui.widgets import RichPane


class PlanPane(VerticalScroll):
    """One model, one quantisation, and the command line that runs it here."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in bindable(PLAN_KEYS)
    ]

    DEFAULT_CSS = """
    PlanPane { height: 1fr; padding: 0 1; }
    PlanPane > #plan-notice { height: auto; color: $warning; }
    """

    def __init__(self, dashboard: Dashboard, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id)
        self.dashboard = dashboard
        self.row: BoardRow | None = None
        self.context: int | None = None
        self.vision: bool | None = None
        self.command: list[str] = []
        self.tiers: tuple[int, ...] = ()

    def compose(self) -> ComposeResult:
        """A notice line for what a key could not do, and the plan itself under it."""
        yield RichPane("", id="plan-notice")
        yield RichPane(self._nothing_chosen(), id="plan-body")

    def _nothing_chosen(self) -> RenderableType:
        return Text(
            _("Choose a row on the Board and press p, and its plan and command line appear here.")
        )

    # --- drawing -----------------------------------------------------------------------

    def show_row(self, row: BoardRow) -> None:
        """Plan this row from its own defaults, forgetting whatever the last one was set to."""
        self.row = row
        self.context = None
        self.vision = None
        self.refresh_plan()

    def refresh_plan(self) -> None:
        """Replan and redraw, saying plainly why there is nothing to draw when there is not."""
        body = self.query_one("#plan-body", RichPane)
        notice = self.query_one("#plan-notice", RichPane)
        notice.show("")
        row, host, catalog = self.row, self.dashboard.host, self.dashboard.catalog
        if row is None or host is None or catalog is None:
            body.show(self._nothing_chosen())
            return
        model = catalog.by_id.get(row.model_id)
        quant = None if model is None else self._quant(model, row.quant)
        if model is None or quant is None:
            body.show_text(
                _("%(model)s is no longer in the catalog this board was built from.")
                % {"model": isolate(row.model_id)}
            )
            return
        needs = self.dashboard.request.needs.model_copy(
            update={"requested_context": self.context} if self.context is not None else {}
        )
        vision = self.dashboard.request.vision if self.vision is None else self.vision
        try:
            report = plan_report(
                model,
                quant,
                host,
                needs=needs,
                vision=vision,
                local_files=[local.path for local in self.dashboard.local_models],
            )
        except LlamaFitError as exc:
            body.show_text(exc.render(), style="red")
            self.command = []
            self.tiers = ()
            return
        self.command = list(report.command)
        self.tiers = tuple(tier.tokens for tier in report.placement.tiers)
        body.show(Group(Text(self._heading(vision)), Text(""), render_plan(report)))

    def _heading(self, vision: bool) -> str:
        """One line saying what the two keys that change the plan are currently set to."""
        if vision:
            return _("Planned with the vision projector; press v to leave it out.")
        return _("Planned without the vision projector; press v to put it back.")

    @staticmethod
    def _quant(model: CatalogModel, name: str) -> Quant | None:
        """The quantisation the board row names, or ``None`` if the catalog no longer has it."""
        return next((quant for quant in quant_entries(model) if quant.name == name), None)

    # --- what the reader does ----------------------------------------------------------

    def action_longer(self) -> None:
        """Move up the context ladder the planner produced, if there is a rung above."""
        self._step(up=True)

    def action_shorter(self) -> None:
        """Move down the context ladder the planner produced, if there is a rung below."""
        self._step(up=False)

    def _step(self, *, up: bool) -> None:
        """Take the next rung in one direction, or say there is not one.

        The rungs are section 9.3's own ladder, carried on the placement that was just
        drawn. A screen that stepped by doubling would offer contexts the planner never
        costed, and the first thing a reader would do with one is paste it.
        """
        current = self.context if self.context is not None else self._planned_context()
        if current is None:
            return
        above = [tokens for tokens in self.tiers if tokens > current]
        below = [tokens for tokens in self.tiers if tokens < current]
        wanted = (min(above) if above else None) if up else (max(below) if below else None)
        if wanted is None:
            self.query_one("#plan-notice", RichPane).show_text(
                _("The ladder has no rung above %(context)s.")
                % {"context": format_context(current)}
                if up
                else _("The ladder has no rung below %(context)s.")
                % {"context": format_context(current)}
            )
            return
        self.context = wanted
        self.refresh_plan()

    def _planned_context(self) -> int | None:
        """The context the last plan actually settled on, which is where a step starts from."""
        row = self.row
        if row is None or row.candidate.placement is None:
            return None
        return row.candidate.placement.context

    def action_toggle_vision(self) -> None:
        """Plan with or without the vision projector, and replan around the difference."""
        current = self.dashboard.request.vision if self.vision is None else self.vision
        self.vision = not current
        self.refresh_plan()

    def action_copy(self) -> None:
        """Put the command line on the clipboard, and say whether there was one to put."""
        if not self.command:
            self.notify(_("There is no command line to copy yet."), severity="warning")
            return
        self.app.copy_to_clipboard(" ".join(self.command))
        self.notify(_("The command line is on the clipboard."))
