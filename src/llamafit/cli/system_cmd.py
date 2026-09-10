# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit system``: print the host scan and llama.cpp status.

Section 13.1's substitution flags apply here, and this is the command where a reader can
see what they did: ``llamafit --profile NAME system`` prints the machine that profile
stands in for, under the same red ``SIMULATED`` line ``hardware show NAME --as-host``
prints it under, with this machine's llama.cpp beside it. The llama.cpp half is never
substituted, because the binary and the GGUF files on this disk are real whichever
machine the numbers describe.
"""

from __future__ import annotations

from typing import cast

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.common import machine
from llamafit.cli.render import render_host, render_llamacpp
from llamafit.i18n import lazy_gettext


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
    report = machine(state, measure_bandwidth=not no_measure)
    if state.json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    console = state.console
    console.print(render_host(report.host))
    console.print()
    console.print(render_llamacpp(report.llamacpp))
