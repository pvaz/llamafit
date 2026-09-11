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

This is also where the memory bandwidth is re-timed. The figure is measured once per
machine and kept, because re-timing it on every scan is what made the same board report a
different tokens-per-second from one minute to the next; the host table says *cached* when
it was read back, and ``--refresh-bandwidth`` here is how somebody who has changed
something, or does not believe the number, makes it take the measurement again. Every
other command then reads what this one leaves behind.
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
    refresh_bandwidth: bool = typer.Option(
        False,
        "--refresh-bandwidth",
        help=cast(
            str,
            lazy_gettext(
                "Measure RAM bandwidth again instead of reading back the figure kept for "
                "this machine, and keep the new one."
            ),
        ),
    ),
) -> None:
    """Show what this machine has: CPU, memory, GPUs, disks and llama.cpp."""
    state: CliState = ctx.obj
    report = machine(state, measure_bandwidth=not no_measure, refresh_bandwidth=refresh_bandwidth)
    if state.json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    console = state.console
    console.print(render_host(report.host))
    console.print()
    console.print(render_llamacpp(report.llamacpp))
