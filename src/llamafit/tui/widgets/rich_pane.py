# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""A pane that shows one Rich renderable, and keeps hold of it.

Almost everything this dashboard has to draw has already been drawn once. The budget line
by line with the source of each, the context ladder with what every rung costs, where a
token's time goes, the four scores and their weights: :mod:`llamafit.cli.render_board`
builds all of them as Rich renderables, and Textual will put a Rich renderable on the
screen unchanged. So the dashboard shows the *same objects* the command line prints rather
than a second rendering of the same facts, and section 12.3's promise -- that the two
interfaces show the same explanation -- is kept by construction instead of by discipline.

The one thing added here is :attr:`RichPane.renderable`. A test that wants to know what a
pane says draws that through a Rich console at a fixed width, exactly as the command
line's own renderer tests do, and gets text it can assert on. A picture of a terminal
would be a screenshot nobody can check.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.widgets import Static


class RichPane(Static):
    """A read-only pane holding one Rich renderable.

    Attributes:
        renderable: What is currently on it. Public because it is what a test reads, and
            reading it is the only way to check a pane's contents without a screenshot.
    """

    DEFAULT_CSS = """
    RichPane {
        height: auto;
        width: 100%;
    }
    """

    def __init__(self, renderable: RenderableType = "", **kwargs: object) -> None:
        # ``markup=False``: a model id, a path, a flag or a licence identifier may hold
        # square brackets, and Textual would otherwise read them as a style tag. The
        # command line takes the same precaution for the same reason.
        super().__init__(renderable, markup=False, **kwargs)  # type: ignore[arg-type]
        self.renderable: RenderableType = renderable

    def show(self, renderable: RenderableType) -> None:
        """Replace what the pane shows."""
        self.renderable = renderable
        self.update(renderable)

    def show_text(self, text: str, *, style: str = "") -> None:
        """Replace what the pane shows with one run of plain, unparsed text."""
        self.show(Text(text, style=style))
