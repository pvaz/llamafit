# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The key map behind ``?``, built from the same table the key bar is built from.

A help screen listing a key nothing is bound to is the reason help screens stop being
trusted, so this one cannot: it reads :mod:`llamafit.tui.keys`, and so does every binding
and the bar along the bottom of each screen.

It is a second way to find the keys, not the only way. The keys that matter on the screen
a reader is looking at are already on that screen, because a newcomer who does not know a
program can do something does not know to press ``?`` to ask whether it can.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import ModalScreen

from llamafit.i18n import _, for_display, isolate, pgettext
from llamafit.tui import keys
from llamafit.tui.widgets import RichPane


def key_map() -> RenderableType:
    """Every key, under the screen it belongs to, in the reader's language."""
    table = Table(title=for_display(_("Keys")))
    table.add_column(for_display(pgettext("column heading", "Screen")))
    table.add_column(for_display(pgettext("column heading", "Key")), no_wrap=True)
    table.add_column(for_display(pgettext("column heading", "What it does")))
    sections: Sequence[tuple[str, Sequence[keys.Key]]] = (
        (pgettext("dashboard screen", "Everywhere"), keys.GLOBAL_KEYS),
        (pgettext("dashboard screen", "Board"), keys.BOARD_KEYS),
        (pgettext("dashboard screen", "Needs"), keys.NEEDS_KEYS),
        (pgettext("dashboard screen", "Host"), keys.HOST_KEYS),
        (pgettext("dashboard screen", "Plan"), keys.PLAN_KEYS),
        (pgettext("dashboard screen", "Simulate"), keys.SIMULATE_KEYS),
    )
    for screen, section in sections:
        for index, key in enumerate(section):
            table.add_row(
                Text(for_display(screen if index == 0 else "")),
                Text(for_display(isolate(key.shown))),
                Text(for_display(str(key.description))),
            )
    return table


class KeyMapScreen(ModalScreen[None]):
    """The key map, over whatever was on screen, until any of three keys closes it."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", show=False),
        Binding("question_mark", "dismiss", show=False),
        Binding("q", "dismiss", show=False),
    ]

    DEFAULT_CSS = """
    KeyMapScreen { align: center middle; }
    KeyMapScreen > VerticalScroll {
        width: 80%;
        max-width: 78;
        height: 80%;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        """The key map, and the line saying how to close it."""
        with VerticalScroll():
            yield RichPane(key_map(), id="key-map")
            yield RichPane(Text(_("Press Escape to go back.")), id="key-map-close")
