# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Telling somebody what is happening to their disk and their line, at any window width.

The engine knows nothing about Rich. It calls five methods on a :class:`ProgressReporter`,
and what draws is decided here, once, from the console it was handed.

Three shapes, chosen by what the console actually is:

* **Wide terminal.** Name, bar, percentage, bytes of bytes, rate, time left.
* **Narrow terminal.** The columns come off from the right, and below sixty cells the bar
  goes too. A progress display that wraps is worse than none: every refresh leaves another
  torn copy of itself on the scrollback, and the number a reader wanted is the one that
  fell off the end.
* **Not a terminal at all** — a log file, a CI job, a pipe. No live display, because
  nothing can redraw a line that has already been written; one line when a file starts and
  one when it is done, which is what somebody reading a log afterwards wanted anyway.

Every byte figure goes through :func:`llamafit.units.format_bytes`, so a Portuguese reader
is told ``4,7 GiB`` and not ``4.7 GiB``, and every sentence goes through the translator.
Identifiers inside a sentence — a file name, a rate — are isolated, so the display is
readable in Arabic and Hebrew too, the same way the tables in :mod:`llamafit.cli.render`
are.
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

from llamafit.i18n import _, for_display, isolate
from llamafit.units import format_bytes

NARROW_CONSOLE_CELLS = 60
"""Below this width the bar is dropped; there is no room for it and the numbers both."""

MEDIUM_CONSOLE_CELLS = 90
"""Below this width the file name and the time remaining are dropped."""


class ProgressReporter(Protocol):
    """What the engine tells whoever is watching.

    Deliberately five methods and no state: the engine reports events, and every decision
    about how to draw one belongs to the implementation.
    """

    def file_started(self, name: str, total: int, already: int) -> None:
        """A file has begun, at ``already`` of ``total`` bytes (resume starts above zero)."""
        ...

    def bytes_received(self, name: str, count: int) -> None:
        """``count`` more bytes are on disk; negative when a failed attempt is unwound."""
        ...

    def file_verifying(self, name: str, total: int) -> None:
        """The bytes are all there and the checksum is being computed."""
        ...

    def file_finished(self, name: str) -> None:
        """The file is verified and in place."""
        ...

    def note(self, message: str) -> None:
        """Something the reader should know that is not a failure."""
        ...


class NullProgress:
    """A reporter that draws nothing, for ``--json`` and for tests that do not care."""

    def file_started(self, name: str, total: int, already: int) -> None:
        """Ignore it."""

    def bytes_received(self, name: str, count: int) -> None:
        """Ignore it."""

    def file_verifying(self, name: str, total: int) -> None:
        """Ignore it."""

    def file_finished(self, name: str) -> None:
        """Ignore it."""

    def note(self, message: str) -> None:
        """Ignore it."""


class _BytesColumn(ProgressColumn):
    """``4.7 GiB of 61.2 GiB``, in this language's punctuation."""

    def render(self, task: Task) -> Text:
        """Draw the cell for one task."""
        completed = int(task.completed or 0)
        total = int(task.total or 0)
        return Text(
            for_display(
                _("%(done)s of %(total)s")
                % {
                    "done": isolate(format_bytes(completed)),
                    "total": isolate(format_bytes(total)),
                }
            )
        )


class _RateColumn(ProgressColumn):
    """``12.4 MiB/s``, or a dash before there is anything to divide by."""

    def render(self, task: Task) -> Text:
        """Draw the cell for one task."""
        speed = task.speed
        if not speed:
            return Text("--")
        return Text(
            for_display(
                _("%(rate)s/s") % {"rate": isolate(format_bytes(int(speed)))},
            )
        )


class RichProgress:
    """The live display, sized to the console it was handed.

    Used as a context manager. Nothing is drawn before the first ``file_started``, so the
    summary printed before a download is not shoved up the screen by an empty bar.
    """

    def __init__(self, console: Console) -> None:
        """Build a display suited to ``console``, live or line-by-line."""
        self.console = console
        self.live = console.is_terminal
        self._progress = Progress(*self._columns(console), console=console, transient=False)
        self._tasks: dict[str, TaskID] = {}
        self._totals: dict[str, int] = {}
        self._started = False

    def _columns(self, console: Console) -> list[ProgressColumn]:
        """The columns that fit, widest set first."""
        width = console.width
        if width >= MEDIUM_CONSOLE_CELLS:
            return [
                TextColumn("{task.description}", table_column=Column(no_wrap=True)),
                BarColumn(bar_width=None),
                TextColumn("{task.percentage:>3.0f}%"),
                _BytesColumn(),
                _RateColumn(),
                TimeRemainingColumn(),
            ]
        if width >= NARROW_CONSOLE_CELLS:
            return [
                BarColumn(bar_width=None),
                TextColumn("{task.percentage:>3.0f}%"),
                _BytesColumn(),
            ]
        return [TextColumn("{task.percentage:>3.0f}%"), _BytesColumn()]

    def __enter__(self) -> RichProgress:
        """Start the live display, when this console has one."""
        if self.live:
            self._progress.start()
            self._started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the live display, leaving the finished bars on the screen."""
        if self._started:
            self._progress.stop()
            self._started = False

    def file_started(self, name: str, total: int, already: int) -> None:
        """Add a bar for ``name``, already filled to ``already`` on a resumed transfer."""
        if not self.live:
            self.console.print(
                Text(
                    for_display(
                        _("Fetching %(file)s (%(size)s)")
                        % {"file": isolate(name), "size": isolate(format_bytes(total))}
                    )
                )
            )
            return
        task = self._progress.add_task(name, total=max(total, 1), completed=already)
        self._tasks[name] = task
        self._totals[name] = max(total, 1)

    def bytes_received(self, name: str, count: int) -> None:
        """Move ``name``'s bar by ``count``, which is negative when an attempt is unwound."""
        task = self._tasks.get(name)
        if task is not None:
            self._progress.advance(task, count)

    def file_verifying(self, name: str, total: int) -> None:
        """Say that the checksum is being computed, which on a large file takes minutes."""
        message = Text(
            for_display(_("Checking %(file)s against its checksum") % {"file": isolate(name)})
        )
        if not self.live:
            self.console.print(message)
            return
        task = self._tasks.get(name)
        if task is not None:
            self._progress.reset(task, total=max(total, 1), description=str(message))

    def file_finished(self, name: str) -> None:
        """Fill ``name``'s bar and leave it on the screen."""
        if not self.live:
            self.console.print(Text(for_display(_("Done: %(file)s") % {"file": isolate(name)})))
            return
        task = self._tasks.get(name)
        if task is not None:
            self._progress.update(task, description=name, completed=self._totals.get(name, 1))

    def note(self, message: str) -> None:
        """Print a remark above the bars without disturbing them."""
        self._progress.console.print(Text(for_display(message), style="yellow"))
