"""Typer application: global options, error rendering, command registration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.text import Text

from llamafit import __version__
from llamafit.errors import LlamaFitError, NotInstalledError, ProbeError
from llamafit.logging import setup_logging
from llamafit.paths import get_paths

app = typer.Typer(
    name="llamafit",
    help="Find, size, install and verify open-weight LLMs for llama.cpp on your own machine.",
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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llamafit {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON instead of tables."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Write details to llamafit.log in the log directory and show tracebacks.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colours."),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
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
    no_color = "--no-color" in sys.argv or bool(os.environ.get("NO_COLOR"))
    if verbose:
        setup_logging(get_paths().log_dir, verbose=True)
    console = Console(stderr=True, no_color=no_color, highlight=False)
    try:
        app(standalone_mode=True)
    except LlamaFitError as exc:
        console.print(Text(exc.render(), style="red"))
        sys.exit(2 if isinstance(exc, (NotInstalledError, ProbeError)) else 1)
    except Exception as exc:  # an unexpected failure is a bug, not a user error
        if verbose:
            raise
        console.print(Text(f"Unexpected error: {exc}", style="red"))
        console.print(Text("Run again with --verbose for the full traceback.", style="dim"))
        sys.exit(1)


from llamafit.cli import (  # noqa: E402  (registers commands on import)
    catalog_cmd,
    doctor_cmd,
    system_cmd,
)

__all__ = ["CliState", "app", "catalog_cmd", "doctor_cmd", "main", "system_cmd"]
