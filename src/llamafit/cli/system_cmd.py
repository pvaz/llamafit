# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit system``: print the host scan and llama.cpp status."""

from __future__ import annotations

from typing import cast

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_host, render_llamacpp
from llamafit.i18n import lazy_gettext
from llamafit.services.scan import scan_system


@app.command(
    "system",
    # Typer takes a command's help from its docstring, and a docstring is a literal, not
    # a call, so no wrapper can reach it: a command whose help has to be translatable
    # needs an explicit help= here. Deferred, because the decorator runs at import time,
    # before any language has been chosen.
    help=cast(
        str,
        lazy_gettext("Show what this machine has: CPU, memory, GPUs, disks and llama.cpp."),
    ),
)
def system_command(
    ctx: typer.Context,
    no_measure: bool = typer.Option(
        False,
        "--no-measure",
        help=cast(str, lazy_gettext("Skip the RAM bandwidth measurement.")),
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
