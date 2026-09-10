# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What ``llamafit`` with no arguments does.

Section 13.1 gives it one row of its own: it opens the dashboard, and falls back to
``recommend`` when there is no terminal to open one in. Both halves matter. The first is
why the dashboard exists -- somebody who has just installed this types the name of the
program, not a subcommand of it. The second is why a pipe, a CI job or a redirect still
gets an answer instead of a full-screen application fighting with a file handle.

The fallback runs the registered ``recommend`` command with its own defaults rather than
reproducing what it does. Two boards that could disagree about the same machine would be
worse than no fallback at all, and the flag a reader would then be told to pass -- there is
none -- could not be the fix.

Textual is imported inside the function, not at the top. This module is imported while
``llamafit.cli.app`` is, which is on the way to every command including ``--version``, and
a full user-interface framework loaded to print a version string is a cost every command
pays for one of them. It also means the screens are imported after ``--language`` has been
read, which is the only moment at which a deferred message could still have gone wrong.
"""

from __future__ import annotations

import sys
from typing import Any, cast

import typer
from rich.console import Console
from rich.text import Text

from llamafit.i18n import _


def interactive(stdin: object | None = None, stdout: object | None = None) -> bool:
    """Whether there is a terminal on both ends to run a full-screen application in.

    Both ends, because either one being a file is enough to make a dashboard the wrong
    answer: output redirected to a file wants text, and input redirected from one has no
    keystrokes to give a screen that exists to be navigated.
    """
    streams = (
        stdin if stdin is not None else sys.stdin,
        stdout if stdout is not None else sys.stdout,
    )
    return all(getattr(stream, "isatty", lambda: False)() for stream in streams)


def run_dashboard() -> None:
    """Open the dashboard and block until the reader quits it."""
    from llamafit.tui.app import LlamaFitApp

    LlamaFitApp().run()


def open_dashboard(ctx: typer.Context) -> None:
    """``llamafit`` with no arguments: the dashboard, or the board it would have shown.

    Args:
        ctx: The root command's context, which carries the global options and is what the
            fallback is invoked through.
    """
    if interactive():
        run_dashboard()
        return
    console = Console(stderr=True, no_color=True, highlight=False)
    console.print(
        Text(
            _(
                "There is no terminal here to draw the dashboard in, so this is what "
                "`llamafit recommend` would print."
            ),
            style="dim",
        )
    )
    fall_back_to_recommend(ctx)


def recommend_defaults() -> dict[str, Any]:
    """Every value ``llamafit recommend`` would use when given no flags at all.

    Read off the registered command rather than written down again, so a default that
    changes on the flag changes here too. A fallback that quietly used a different limit,
    or a different use case, from the command it claims to be standing in for would be a
    second board nobody knew they were looking at.
    """
    from typer.main import get_command

    from llamafit.cli.app import app

    group = cast("Any", get_command(app))
    command = group.commands["recommend"]
    return {
        parameter.name: parameter.default
        for parameter in command.params
        if parameter.expose_value and parameter.name is not None
    }


def fall_back_to_recommend(ctx: typer.Context) -> None:
    """Run the registered ``recommend`` command with every default it declares."""
    from llamafit.cli.board_cmd import recommend_command

    recommend_command(ctx, **recommend_defaults())
