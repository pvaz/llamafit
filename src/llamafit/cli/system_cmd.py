"""``llamafit system``: print the host scan and llama.cpp status."""

from __future__ import annotations

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_host, render_llamacpp
from llamafit.services.scan import scan_system


@app.command("system")
def system_command(
    ctx: typer.Context,
    no_measure: bool = typer.Option(
        False, "--no-measure", help="Skip the RAM bandwidth measurement."
    ),
) -> None:
    """Show what this machine has: CPU, memory, GPUs, disks and llama.cpp."""
    state: CliState = ctx.obj
    report = scan_system(measure_bandwidth=not no_measure)
    if state.json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    console = state.console
    console.print(render_host(report.host))
    console.print()
    console.print(render_llamacpp(report.llamacpp))
