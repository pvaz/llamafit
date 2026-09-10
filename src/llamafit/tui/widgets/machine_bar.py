# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The band under the title: which machine, and whether it is this one.

Two lines that never scroll. The first says what the figures on every screen are about;
the second is a badge that appears only when they are about somebody else's machine, and
it carries the word rather than only a colour, because a screen read without colour has to
say the same thing as one read with it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from llamafit.models.host import Host
from llamafit.models.llamacpp import LlamaCpp
from llamafit.tui.summary import machine_line, simulated_badge


class MachineBar(Horizontal):
    """The machine line, with the ``SIMULATED`` badge beside it when one is earned."""

    DEFAULT_CSS = """
    MachineBar {
        height: 1;
        width: 100%;
        background: $panel;
    }
    MachineBar > #machine-badge {
        width: auto;
        padding: 0 1;
        color: $text;
        background: $error;
        text-style: bold;
    }
    MachineBar > #machine-line {
        width: 1fr;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """The badge, which is usually hidden, and the machine line beside it."""
        yield Static("", id="machine-badge", markup=False)
        yield Static("", id="machine-line", markup=False)

    def show(self, host: Host | None, llamacpp: LlamaCpp | None) -> None:
        """Redraw both lines for the machine currently being answered against."""
        badge = simulated_badge(host)
        badge_widget = self.query_one("#machine-badge", Static)
        badge_widget.update(badge)
        badge_widget.display = bool(badge)
        self.query_one("#machine-line", Static).update(machine_line(host, llamacpp))
