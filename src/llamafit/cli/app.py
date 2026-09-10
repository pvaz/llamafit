# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Typer application: global options, error rendering, command registration.

The language is chosen here and nowhere else. ``--language`` is eager, so it is read
before Click renders anything: a help screen asked for in Portuguese comes out in
Portuguese, which it could not if the language were chosen in the callback body that
``--help`` never reaches.

Every ``help=`` string on this page is deferred. A decorator runs while the module is
imported, long before any language has been chosen, so an eager ``_()`` here would freeze
the whole interface in English and nothing would fail. ``cast(str, ...)`` is what says out
loud that Typer's ``str`` annotation is being taken at its word: Typer only ever renders
these, and rendering is exactly what a ``LazyString`` does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import cast

import typer
from rich.console import Console
from rich.text import Text

from llamafit import __version__
from llamafit.errors import LlamaFitError, NotInstalledError, PackagedDataError, ProbeError
from llamafit.i18n import _, lazy_gettext, set_language
from llamafit.logging import setup_logging
from llamafit.paths import get_paths

app = typer.Typer(
    name="llamafit",
    help=cast(
        str,
        lazy_gettext(
            "Find, size, install and verify open-weight LLMs for llama.cpp on your own machine."
        ),
    ),
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)


@dataclass
class CliState:
    """Options shared by every command, stored in ``ctx.obj``."""

    json_output: bool = False
    verbose: bool = False
    no_color: bool = False

    @property
    def console(self) -> Console:
        """A console for normal output, respecting ``--no-color``."""
        return Console(no_color=self.no_color, highlight=False)


def _colours_off() -> bool:
    """Whether colour is unwanted, read from the raw arguments and the environment.

    The language notice and the error report are both written before, or instead of, the
    parsed options, so neither can ask a :class:`CliState` that does not exist yet.
    """
    return "--no-color" in sys.argv or bool(os.environ.get("NO_COLOR"))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llamafit {__version__}")
        raise typer.Exit()


def _language_callback(value: str | None) -> str | None:
    """Install the chosen language and say out loud when the request was not met exactly.

    A notice arrives on a request served by another region's catalog as well as on one
    LlamaFit cannot honour at all, so it does not mean the request was refused. It is the
    only place a reader is told why some of the wording looks foreign, and losing it is
    the failure this whole layer exists to prevent.

    It goes to stderr as ``Text``: as ``Text`` because it quotes a catalog's own
    ``Language-Team`` header, which is text read from a file and would otherwise be
    parsed as Rich markup, and to stderr because ``--json`` writes machine-readable
    output to stdout that a remark must not join.
    """
    choice = set_language(value)
    if choice.notice:
        console = Console(stderr=True, no_color=_colours_off(), highlight=False)
        console.print(Text(choice.notice, style="yellow"))
        if choice.hint:
            console.print(Text(choice.hint, style="dim"))
    return value


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help=cast(str, lazy_gettext("Print machine-readable JSON instead of tables.")),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=cast(
            str,
            lazy_gettext("Write details to llamafit.log in the log directory and show tracebacks."),
        ),
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help=cast(str, lazy_gettext("Disable colours."))
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        metavar="TAG",
        callback=_language_callback,
        is_eager=True,
        help=cast(
            str,
            lazy_gettext(
                "Speak this language, for example pt_PT. A language LlamaFit does not have "
                "falls back to English, and it says which ones it does have."
            ),
        ),
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help=cast(str, lazy_gettext("Print the version and exit.")),
    ),
) -> None:
    """LlamaFit command-line interface."""
    ctx.obj = CliState(json_output=json_output, verbose=verbose, no_color=no_color)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    """Run the app, turning known errors into messages and unexpected ones into a short report.

    Error text can contain anything, including square brackets from a path, so it is printed
    as ``Text`` and never parsed as Rich markup.
    """
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        setup_logging(get_paths().log_dir, verbose=True)
    console = Console(stderr=True, no_color=_colours_off(), highlight=False)
    try:
        app(standalone_mode=True)
    except LlamaFitError as exc:
        console.print(Text(exc.render(), style="red"))
        # 1 is a user or configuration error, 2 an environment problem: llama.cpp or a
        # tool missing, a probe that could not run, or an installation missing the data
        # that shipped inside it. A script has to be able to tell "fix your file" from
        # "fix your machine"; docs/cli.md documents both.
        environment = (NotInstalledError, PackagedDataError, ProbeError)
        sys.exit(2 if isinstance(exc, environment) else 1)
    except Exception as exc:  # an unexpected failure is a bug, not a user error
        if verbose:
            raise
        console.print(Text(_("Unexpected error: %(error)s") % {"error": exc}, style="red"))
        console.print(Text(_("Run again with --verbose for the full traceback."), style="dim"))
        sys.exit(1)


from llamafit.cli import (  # noqa: E402  (registers commands on import)
    catalog_cmd,
    doctor_cmd,
    hardware_cmd,
    system_cmd,
)

__all__ = [
    "CliState",
    "app",
    "catalog_cmd",
    "doctor_cmd",
    "hardware_cmd",
    "main",
    "system_cmd",
]
