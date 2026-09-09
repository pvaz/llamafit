"""Typer application: global options, error rendering, command registration."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import typer
from rich.console import Console

from llamafit import __version__
from llamafit.errors import LlamaFitError, NotInstalledError, ProbeError

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

    @property
    def err_console(self) -> Console:
        """A console for errors, on stderr."""
        return Console(stderr=True, no_color=self.no_color, highlight=False)


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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log details and show tracebacks."),
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
    """Entry point: run the app and turn LlamaFit errors into messages and exit codes."""
    try:
        app(standalone_mode=True)
    except LlamaFitError as exc:  # raised inside commands before Typer's own handling
        Console(stderr=True).print(f"[red]{exc.render()}[/red]")
        code = 2 if isinstance(exc, (NotInstalledError, ProbeError)) else 1
        sys.exit(code)


from llamafit.cli import doctor_cmd, system_cmd  # noqa: E402  (registers commands on import)

__all__ = ["CliState", "app", "doctor_cmd", "main", "system_cmd"]
