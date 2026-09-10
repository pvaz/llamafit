# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Typer application: global options, error rendering, command registration.

The language is chosen here and nowhere else. ``--language`` is eager, so it is read
before Click renders anything: a help screen asked for in Portuguese comes out in
Portuguese, which it could not if the language were chosen in the callback body that
``--help`` never reaches.

Section 13.1's five substitution flags -- ``--profile``, ``--memory``, ``--ram``,
``--cpu-cores`` and ``--max-context`` -- are global options here because they are
questions about the *subject* of every answer rather than about any one command. The
first four replace the machine; the fifth caps the context. Nothing in this module acts
on them: they are stored on :class:`CliState` as typed and
:mod:`llamafit.cli.common` turns them into a :class:`~llamafit.models.host.Host` and a
:class:`~llamafit.models.plan.Needs`, so a command that never reads a machine cannot
half-honour them.

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
    """Options shared by every command, stored in ``ctx.obj``.

    The five after ``no_color`` are section 13.1's substitution flags. They are kept here
    exactly as typed -- strings for the two sizes, which nothing in this module is allowed
    to parse -- and :mod:`llamafit.cli.common` turns them into a machine and a request.
    ``app.py`` cannot import ``common``, which imports ``CliState`` from here, and putting
    the arithmetic behind the import would be a cycle for the sake of five fields.
    """

    json_output: bool = False
    verbose: bool = False
    no_color: bool = False
    profile: str | None = None
    memory: str | None = None
    ram: str | None = None
    cpu_cores: int | None = None
    max_context: int | None = None

    @property
    def console(self) -> Console:
        """A console for normal output, respecting ``--no-color``."""
        return Console(no_color=self.no_color, highlight=False)

    @property
    def substituting(self) -> bool:
        """Whether any machine other than this one was asked for.

        ``--max-context`` is not one of these. It changes the question, not the machine,
        so a command that reads this to refuse a stand-in machine must not refuse it too.
        """
        return any(
            value is not None for value in (self.profile, self.memory, self.ram, self.cpu_cores)
        )


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
    profile: str | None = typer.Option(
        None,
        "--profile",
        metavar="NAME|FILE",
        help=cast(
            str,
            lazy_gettext(
                "Answer for the machine this hardware profile describes instead of this one. "
                "Nothing about this machine is probed, and every answer is marked as "
                "simulated. `llamafit hardware list` names the profiles you have."
            ),
        ),
    ),
    memory: str | None = typer.Option(
        None,
        "--memory",
        metavar="SIZE",
        help=cast(
            str,
            lazy_gettext(
                "Pretend the graphics card has this much memory, for example 24G. Everything "
                "else stays as scanned, and every answer is marked as simulated."
            ),
        ),
    ),
    ram: str | None = typer.Option(
        None,
        "--ram",
        metavar="SIZE",
        help=cast(
            str,
            lazy_gettext(
                "Pretend the machine has this much system memory, for example 128GiB. "
                "Every answer is marked as simulated."
            ),
        ),
    ),
    cpu_cores: int | None = typer.Option(
        None,
        "--cpu-cores",
        metavar="N",
        min=1,
        help=cast(
            str,
            lazy_gettext(
                "Pretend the processor has this many physical cores. Every answer is "
                "marked as simulated."
            ),
        ),
    ),
    max_context: int | None = typer.Option(
        None,
        "--max-context",
        metavar="N",
        min=1,
        help=cast(
            str,
            lazy_gettext(
                "Plan, report and score no context longer than this. It caps the model's "
                "own length; it does not describe a machine."
            ),
        ),
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
    ctx.obj = CliState(
        json_output=json_output,
        verbose=verbose,
        no_color=no_color,
        profile=profile,
        memory=memory,
        ram=ram,
        cpu_cores=cpu_cores,
        max_context=max_context,
    )
    if ctx.invoked_subcommand is None:
        open_dashboard(ctx)


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
    bench_cmd,
    board_cmd,
    catalog_cmd,
    doctor_cmd,
    hardware_cmd,
    install_cmd,
    llamacpp_cmd,
    plan_cmd,
    preset_cmd,
    serve_cmd,
    system_cmd,
)
from llamafit.tui.entry import open_dashboard  # noqa: E402  (`llamafit` with no arguments)

__all__ = [
    "CliState",
    "app",
    "bench_cmd",
    "board_cmd",
    "catalog_cmd",
    "doctor_cmd",
    "hardware_cmd",
    "install_cmd",
    "llamacpp_cmd",
    "main",
    "open_dashboard",
    "plan_cmd",
    "preset_cmd",
    "serve_cmd",
    "system_cmd",
]
