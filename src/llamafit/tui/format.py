# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Turning one figure into one cell, the way :mod:`llamafit.cli.render_board` turns it.

Four functions, each three lines, each doing exactly what its namesake in the command
line's board renderer does: a byte count with its unit, a figure with this language's
separators, a share as a percentage, a context length compactly. They are written again
here rather than reached for across the interface boundary because the originals are that
module's private helpers, and an interface that imports another interface's underscore
names is one refactor away from breaking without a test to say so.

Nothing here computes anything. Every number these are handed was produced by a service;
these decide only how it is spelled. The words they fall back to when a figure is missing
are the command line's own messages, context and all, so the catalog gains no entry from
this file and a reader meets the same word on both screens.
"""

from __future__ import annotations

from llamafit.i18n import isolate, pgettext
from llamafit.units import format_bytes, format_grouped, localise_number


def size(n: int | None) -> str:
    """A byte size as one directional island, or the word for a size nobody could read."""
    return isolate(format_bytes(n)) if n is not None else format_bytes(n)


def number(value: float, digits: int = 1) -> str:
    """A figure with its separators in this language's punctuation, as one island."""
    return isolate(localise_number(f"{value:,.{digits}f}"))


def percent(share: float | None) -> str:
    """A utilisation as a percentage, or the word for one nobody could compute."""
    if share is None:
        return pgettext("utilisation", "n/a")
    return isolate(localise_number(f"{share * 100:.0f}%"))


def context(tokens: int) -> str:
    """A context length compactly, for example ``40K``, and plainly when it is not round."""
    if tokens % 1024 == 0:
        return isolate(f"{tokens // 1024}K")
    return isolate(format_grouped(tokens))
