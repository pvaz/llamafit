# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit doctor``: probe-by-probe report and actionable findings."""

from __future__ import annotations

from typing import cast

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_findings, render_probes
from llamafit.i18n import lazy_gettext
from llamafit.services.doctor import diagnose
from llamafit.services.scan import scan_system


@app.command(
    "doctor",
    # An explicit help=, deferred: Typer would otherwise take this command's help from
    # its docstring, which is a literal no wrapper can reach.
    help=cast(
        str,
        lazy_gettext("Explain what was detected, what failed, and what would unlock more."),
    ),
)
def doctor_command(ctx: typer.Context) -> None:
    """Explain what was detected, what failed, and what would unlock more."""
    state: CliState = ctx.obj
    report = scan_system()
    diagnosis = diagnose(report)
    if state.json_output:
        typer.echo(diagnosis.model_dump_json(indent=2))
    else:
        console = state.console
        console.print(render_probes([*report.host.probes, *report.llamacpp.probes]))
        console.print()
        console.print(render_findings(diagnosis.findings))
    if diagnosis.worst_level == "error":
        raise typer.Exit(code=2)
