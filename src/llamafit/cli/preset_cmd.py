# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit preset`` and ``llamafit launch``: write the scripts, and run one.

Two commands in one module because they are two halves of the same thing. ``preset`` turns
a plan into files; ``launch`` runs one of those files. ``launch`` deliberately does not
rebuild the command line for itself -- it starts the script -- because the script is where
the context ladder lives and where the user's own edits live, and a launcher that bypassed
both would be launching something nobody chose.

Everything this module prints goes through the translation layer, as every interface in
this project does. What it *writes* does not: the generated scripts are English, for the
reasons set out in :mod:`llamafit.presets`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import cast

import typer
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from llamafit import __version__
from llamafit.cli.app import CliState, app
from llamafit.cli.common import checked_max_context, find_model, load_catalog_or_warn, machine
from llamafit.cli.plan_cmd import choose_quant
from llamafit.constants import DEFAULT_SERVER_PORT
from llamafit.errors import BudgetError, NotInstalledError
from llamafit.i18n import _, for_display, isolate, lazy_gettext, pgettext
from llamafit.llamacpp.detect import exe_name
from llamafit.llamacpp.server import HttpClient, HttpxClient
from llamafit.models.plan import Needs
from llamafit.models.report import SystemReport
from llamafit.paths import get_paths
from llamafit.placement import LaunchOptions, render_flags
from llamafit.presets import (
    Launcher,
    LaunchOutcome,
    PresetSpec,
    SubprocessLauncher,
    build_spec,
    launch_preset,
    port_of_script,
    render_files,
    stop_preset,
    write_files,
)
from llamafit.presets.render import PresetReport, report_of
from llamafit.services.plan import plan_report
from llamafit.units import format_grouped

_MODEL_ARGUMENT: str = typer.Argument(
    ..., metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
)
_DIR_OPTION: Path | None = typer.Option(
    None,
    "--dir",
    help=cast(
        str,
        lazy_gettext("Write the files here instead of in LlamaFit's own presets directory."),
    ),
)
_PORT_OPTION: int | None = typer.Option(
    None,
    "--port",
    min=1,
    max=65535,
    help=cast(str, lazy_gettext("Bind this port instead of llama.cpp's default 8080.")),
)
_QUANT_OPTION: str | None = typer.Option(
    None,
    "--quant",
    help=cast(
        str,
        lazy_gettext("Plan this quantisation instead of the one that scores best on this machine."),
    ),
)
_CONTEXT_OPTION: int | None = typer.Option(
    None,
    "--context",
    min=1,
    help=cast(
        str,
        lazy_gettext(
            "Size for this many tokens. It becomes the top of the ladder the script "
            "chooses from; the script never goes above it."
        ),
    ),
)


def preset_directory(directory: Path | None) -> Path:
    """Where the files go: what was asked for, or LlamaFit's own presets directory."""
    return directory if directory is not None else get_paths().data_dir / "presets"


def server_binary(report: SystemReport) -> str | None:
    """The ``llama-server`` on this machine, or ``None`` when there is not one yet.

    ``None`` is not a failure. A preset can be written before llama.cpp is installed --
    the plan is about the model and the machine, not about which directory a binary is in
    -- and the script falls back to whatever is on ``PATH``, which is what an installation
    that happens afterwards will have put there.
    """
    if not report.llamacpp.installed or report.llamacpp.path is None:
        return None
    return str(Path(report.llamacpp.path) / exe_name("llama-server", report.host.os))


def spec_for(
    model_id: str,
    state: CliState,
    *,
    quant: str | None = None,
    context: int | None = None,
    port: int | None = None,
    today: date | None = None,
) -> PresetSpec:
    """Plan one model on this machine and freeze the plan into a preset specification.

    Args:
        model_id: The catalog id.
        state: The shared command state, for the catalog warning.
        quant: The quantisation to plan, or the best one for this machine.
        context: What to size for, or the project's default.
        port: The port to bind, or llama.cpp's default.
        today: The date to stamp the files with; today's when unset.

    Returns:
        The specification both renderers read.

    Raises:
        CatalogError: If the model or the quantisation is not one the catalog has.
        BudgetError: If nothing about this model fits this machine, so there is no
            configuration for a script to launch.

    The flags are rendered here rather than taken from the plan report, because the port is
    an option of this command and a preset whose header said one port and whose command
    line said another would be a preset that works and lies.
    """
    catalog = load_catalog_or_warn(state)
    model = find_model(catalog, model_id)
    report = machine(state)
    chosen = choose_quant(model, report.host, quant)
    planned = plan_report(
        model,
        chosen,
        report.host,
        needs=Needs(
            use_case=model.use_cases[0],
            requested_context=context,
            max_context=checked_max_context(state),
        ),
        local_files=[local.path for local in report.llamacpp.local_models],
    )
    if planned.placement.mode == "unsupported":
        raise BudgetError(
            _("%(model)s does not fit this machine in any configuration") % {"model": model.id},
            hint=_("Run `llamafit plan %(model)s` to see what it would need.")
            % {"model": model.id},
        )
    options = LaunchOptions(
        model_path=planned.model_path,
        projector_path=planned.projector_path,
        port=port or DEFAULT_SERVER_PORT,
    )
    return build_spec(
        model=model,
        quant=chosen.name,
        placement=planned.placement,
        flags=render_flags(planned.placement, model, options),
        model_path=planned.model_path,
        projector_path=planned.projector_path,
        host=report.host,
        server_path=server_binary(report),
        llamacpp_build=report.llamacpp.build,
        port=options.port,
        generated=today or date.today(),
        version=__version__,
    )


@app.command(
    "preset",
    help=cast(
        str,
        lazy_gettext(
            "Write launch scripts for one model: a script that picks its context from "
            "free card memory at start time, a router section, and a README.\n\n"
            "A file you have edited is never overwritten without --force."
        ),
    ),
)
def preset_command(
    ctx: typer.Context,
    model_id: str = _MODEL_ARGUMENT,
    directory: Path | None = _DIR_OPTION,
    port: int | None = _PORT_OPTION,
    quant: str | None = _QUANT_OPTION,
    context: int | None = _CONTEXT_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        help=cast(
            str,
            lazy_gettext("Overwrite files you have edited. It says which ones it discarded."),
        ),
    ),
) -> None:
    """Write the launch scripts for one model."""
    state: CliState = ctx.obj
    spec = spec_for(model_id, state, quant=quant, context=context, port=port)
    target = preset_directory(directory)
    results = write_files(render_files(spec), target, force=force)
    report = report_of(spec, target, results)
    if state.json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    state.console.print(render_preset(report))


@app.command(
    "launch",
    help=cast(
        str,
        lazy_gettext(
            "Run a model's preset, wait for the server to answer, and print its "
            "endpoints. --stop stops the server LlamaFit started."
        ),
    ),
)
def launch_command(
    ctx: typer.Context,
    model_id: str = _MODEL_ARGUMENT,
    directory: Path | None = _DIR_OPTION,
    port: int | None = _PORT_OPTION,
    stop: bool = typer.Option(
        False,
        "--stop",
        help=cast(str, lazy_gettext("Stop the server LlamaFit started for this model.")),
    ),
    timeout: float = typer.Option(
        180.0,
        "--timeout",
        min=1.0,
        help=cast(
            str,
            lazy_gettext("Seconds to wait for /health before reporting that it has not answered."),
        ),
    ),
) -> None:
    """Run a model's preset and wait for the server it starts."""
    state: CliState = ctx.obj
    if stop:
        _stop(state, model_id)
        return
    target = preset_directory(directory)
    script = _script_in(target, model_id)
    if script is None:
        # Writing it first is the friendly half of "from nothing to a running server":
        # a person who has just been told to run this should not be sent away to run a
        # different command and come back.
        spec = spec_for(model_id, state, port=port)
        write_files(render_files(spec), target)
        script = _script_in(target, model_id)
        if script is None:  # pragma: no cover - write_files creates it or raises
            raise NotInstalledError(_("the preset could not be written"))
        state.console.print(
            Text(
                for_display(_("Wrote a preset first: %(path)s") % {"path": isolate(str(script))}),
                style="dim",
            )
        )
    url = _endpoint_of(script, port)
    outcome = launch_preset(
        model_id,
        script,
        url,
        launcher=launcher(),
        http=http_client(),
        timeout=timeout,
    )
    if state.json_output:
        typer.echo(_launch_json(outcome, script))
        return
    state.console.print(render_launch(outcome, model_id))


def launcher() -> Launcher:
    """How processes are started; replaced in tests, the way every probe's runner is."""
    return SubprocessLauncher()


def http_client() -> HttpClient:
    """How ``/health`` is asked; replaced in tests by the recorded client."""
    return HttpxClient()


def _stop(state: CliState, model_id: str) -> None:
    """Stop a running preset and say what was stopped, or that there was nothing."""
    record = stop_preset(model_id, launcher=launcher())
    if state.json_output:
        typer.echo(record.model_dump_json(indent=2) if record is not None else '{"stopped": false}')
        return
    if record is None:
        state.console.print(
            Text(
                for_display(
                    _("Nothing to stop: LlamaFit has no record of starting %(model)s.")
                    % {"model": isolate(model_id)}
                )
            )
        )
        return
    state.console.print(
        Text(
            for_display(
                _("Stopped %(model)s, which was serving on %(url)s.")
                % {"model": isolate(model_id), "url": isolate(record.url)}
            )
        )
    )


def _script_in(directory: Path, model_id: str) -> Path | None:
    """The preset script for this model in this directory, whichever dialect it is.

    Both are looked for rather than only the one this platform writes, because a
    directory can be shared -- a models folder on a second disk, a dotfiles repository --
    and a person on the machine that owns the script is better served by being told it is
    the wrong dialect than by being told there is none.
    """
    for suffix in (".cmd", ".sh"):
        candidate = directory / f"start-{model_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _endpoint_of(script: Path, port: int | None) -> str:
    """Where the server will answer: the port asked for, else the script's own.

    The script's own comes first among the two that are not asked for, because the file may
    have been edited and the edit is the more recent decision.
    """
    if port is not None:
        return f"http://127.0.0.1:{port}"
    found = port_of_script(script.read_text(encoding="utf-8", errors="replace"))
    return f"http://127.0.0.1:{found or DEFAULT_SERVER_PORT}"


def _launch_json(outcome: LaunchOutcome, script: Path) -> str:
    """The launch result as one JSON object, with the note when there is one."""
    return json.dumps(
        {
            "result": outcome.result,
            "url": outcome.url,
            "script": str(script),
            "error": outcome.error,
            "pid": None if outcome.record is None else outcome.record.pid,
        },
        indent=2,
    )


def render_preset(report: PresetReport) -> Group:
    """What was written, what it will choose at start time, and where the server will be."""
    pieces: list[RenderableType] = [
        Text(
            for_display(
                _("Preset for %(name)s (%(quant)s) in %(dir)s")
                % {
                    "name": isolate(report.name),
                    "quant": isolate(report.quant),
                    "dir": isolate(report.directory),
                }
            ),
            style="bold",
        ),
        _files_table(report),
        _rungs_table(report),
        Text(for_display(_ladder_sentence(report))),
        Text(
            for_display(_("The server will answer on %(url)s.") % {"url": isolate(report.endpoint)})
        ),
    ]
    kept = [file for file in report.files if file.outcome == "kept-your-edits"]
    if kept:
        pieces.append(
            Text(
                for_display(
                    _(
                        "Your edits were kept. Nothing above was written to those files; "
                        "run the command again with --force to replace them."
                    )
                ),
                style="yellow",
            )
        )
    return Group(*pieces)


def _ladder_sentence(report: PresetReport) -> str:
    """The one sentence that says what the script will do with the table above it."""
    if report.probe_free_memory:
        return _(
            "At start time the script reads how much card memory is free, keeps "
            "%(reserve)s MiB back, and takes the largest context that still fits."
        ) % {"reserve": format_grouped(report.reserve_mib)}
    return _(
        "This machine has nothing the script can ask for free card memory, so it will "
        "start at %(context)s tokens; the script's header says why."
    ) % {"context": format_grouped(report.planned_context)}


def _files_table(report: PresetReport) -> Table:
    """One row per file, with what became of it."""
    table = Table(title=for_display(_("Files")))
    # The path folds and the outcome does not: a temporary directory is long enough to
    # push the only column that matters off the edge of a terminal, and "kept: you edited
    # it" is the one thing on this page a reader has to act on.
    table.add_column(for_display(pgettext("column heading", "File")), overflow="fold")
    table.add_column(for_display(pgettext("column heading", "What happened")), no_wrap=True)
    for file in report.files:
        table.add_row(
            Text(for_display(isolate(file.path))),
            Text(for_display(_outcome_label(file.outcome)), style=_outcome_style(file.outcome)),
        )
    return table


def _rungs_table(report: PresetReport) -> Table:
    """The ladder, with the free card memory each rung asks for."""
    table = Table(title=for_display(_("Context tiers the script may choose")))
    table.add_column(
        for_display(pgettext("column heading", "Context")), justify="right", no_wrap=True
    )
    table.add_column(
        for_display(pgettext("column heading", "Card needs")), justify="right", no_wrap=True
    )
    table.add_column(
        for_display(pgettext("column heading", "Free card memory to reach it")),
        justify="right",
        no_wrap=True,
    )
    for rung in report.rungs:
        needs = rung.vram_required_mib
        table.add_row(
            Text(for_display(isolate(format_grouped(rung.tokens)))),
            Text(for_display(isolate(_mib(needs)))),
            Text(for_display(isolate(_mib(needs + report.reserve_mib)))),
        )
    return table


def _mib(value: int) -> str:
    """A mebibyte figure with the unit, grouped for the reader's own language."""
    return _("%(value)s MiB") % {"value": format_grouped(value)}


def _outcome_label(outcome: str) -> str:
    """What became of one file, in a word a reader can act on."""
    labels = {
        "created": _("written"),
        "updated": _("brought up to date"),
        "unchanged": _("already up to date"),
        "kept-your-edits": _("kept: you edited it"),
        "overwritten": _("overwritten: your edits are gone"),
    }
    return labels.get(outcome, outcome)


def _outcome_style(outcome: str) -> str:
    """Colour for one outcome; the two that lose or keep an edit are the loud ones."""
    if outcome == "kept-your-edits":
        return "yellow"
    if outcome == "overwritten":
        return "red"
    return "green"


def render_launch(outcome: LaunchOutcome, model_id: str) -> Group:
    """What happened when the preset was run, and where to point a client."""
    if outcome.result == "failed-to-start":
        return Group(
            Text(
                for_display(
                    _("The preset could not be started: %(error)s")
                    % {"error": isolate(outcome.error or "")}
                ),
                style="red",
            )
        )
    if outcome.result == "no-health":
        return Group(
            Text(
                for_display(
                    _(
                        "Started, but %(url)s has not answered yet. A large model can "
                        "spend minutes reading its weights; try the health endpoint again "
                        "in a moment, or run `llamafit launch %(model)s --stop`."
                    )
                    % {"url": isolate(outcome.url), "model": isolate(model_id)}
                ),
                style="yellow",
            )
        )
    opening = (
        _("%(model)s was already serving on %(url)s.")
        if outcome.result == "already-running"
        else _("%(model)s is serving on %(url)s.")
    )
    return Group(
        Text(
            for_display(opening % {"model": isolate(model_id), "url": isolate(outcome.url)}),
            style="bold",
        ),
        _endpoints_table(outcome.url, model_id),
        Text(
            for_display(
                _("Stop it with `llamafit launch %(model)s --stop`.") % {"model": isolate(model_id)}
            ),
            style="dim",
        ),
    )


def _endpoints_table(url: str, model_id: str) -> Table:
    """The four addresses a person actually wants after a server comes up."""
    table = Table(title=for_display(_("Endpoints")))
    table.add_column(for_display(pgettext("column heading", "What")), no_wrap=True)
    table.add_column(for_display(pgettext("column heading", "Where")), no_wrap=True)
    rows = [
        (_("Web interface"), f"{url}/"),
        (_("Chat completions"), f"{url}/v1/chat/completions"),
        (_("Models"), f"{url}/v1/models"),
        (_("Health"), f"{url}/health"),
    ]
    for label, where in rows:
        table.add_row(Text(for_display(label)), Text(for_display(isolate(where))))
    table.caption = for_display(
        _("Ask for the model by the alias %(alias)s.") % {"alias": isolate(model_id)}
    )
    return table
