# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit install model``: fetch a model's weights, and say what that will cost first.

The command is written around one sentence: nobody should start a hundred-gigabyte
download without being told it is a hundred gigabytes. So before a socket is opened, this
prints the model, the quantisation, the repository, every file it will fetch, what the set
weighs, where it is going, how much of it is already there, and what will be left on the
volume afterwards. In a terminal it then asks. Outside one — a script, a pipeline — it does
not ask, because there is nobody there to answer and whoever wrote the script asked by
running it.

``Ctrl+C`` is a normal outcome, not a crash. The event the workers watch is set, the part
files and their records stay where they are, and the last line says the run can be picked
up where it stopped. That claim is only worth making because
:mod:`llamafit.download.state` makes it true.

Every ``help=`` here is deferred for the reason :mod:`llamafit.cli.app` gives: a decorator
runs while the module is imported, long before a language has been chosen. Every figure
goes through :mod:`llamafit.units`, and every identifier inside a sentence is isolated, so
the output reads correctly in a language written right to left.
"""

from __future__ import annotations

import json
import signal
import threading
from contextlib import ExitStack, suppress
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, cast

import typer
from rich.table import Table
from rich.text import Text

from llamafit.cli.app import CliState, app
from llamafit.cli.common import check_size, find_model, load_catalog_or_warn
from llamafit.download.engine import DEFAULT_WORKERS, MAX_WORKERS, DownloadOptions
from llamafit.download.errors import DownloadCancelledError
from llamafit.download.history import history_path, read_history, recent
from llamafit.download.install import InstallOutcome, install_model, prepare
from llamafit.download.plan import DiskCheck, DownloadPlan, build_plan, check_disk_space
from llamafit.download.progress import NullProgress, ProgressReporter, RichProgress
from llamafit.download.transport import HttpRangeReader
from llamafit.i18n import _, for_display, isolate, lazy_gettext, ngettext
from llamafit.paths import get_paths
from llamafit.units import format_bytes

install_app = typer.Typer(
    name="install",
    help=cast(str, lazy_gettext("Install what it takes to run a model: the weights, for now.")),
)
app.add_typer(install_app, name="install")

# Module-level singletons rather than inline calls in a default position, for the reason
# llamafit/cli/catalog_cmd.py gives: ruff's B008 does not recognise every annotation shape
# as safe to call there, and a singleton sidesteps the warning without arguing with it.
_MODEL_ARGUMENT: str = typer.Argument(
    ..., metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
)
_QUANT_OPTION: str | None = typer.Option(
    None,
    "--quant",
    help=cast(
        str,
        lazy_gettext("Fetch this quantisation instead of the first the catalog publishes."),
    ),
)
_DIR_OPTION: Path | None = typer.Option(
    None,
    "--dir",
    help=cast(
        str,
        lazy_gettext("Put the files here instead of under the configured downloads directory."),
    ),
)
_RATE_OPTION: str | None = typer.Option(
    None,
    "--limit-rate",
    metavar="RATE",
    help=cast(
        str,
        lazy_gettext(
            "Use at most this much bandwidth, for example 5M. Shared across every worker, "
            "so the figure is the figure whatever --workers says."
        ),
    ),
)
_WORKERS_OPTION: int = typer.Option(
    DEFAULT_WORKERS,
    "--workers",
    min=1,
    max=MAX_WORKERS,
    help=cast(
        str,
        lazy_gettext(
            "How many requests to keep in flight. More is not always faster, and a home "
            "connection is shared with whoever else is using it."
        ),
    ),
)
_YES_OPTION: bool = typer.Option(
    False, "--yes", "-y", help=cast(str, lazy_gettext("Do not ask before starting."))
)
_DRY_RUN_OPTION: bool = typer.Option(
    False,
    "--dry-run",
    help=cast(str, lazy_gettext("Show what would be fetched and stop before fetching it.")),
)
_NO_EXTRAS_OPTION: bool = typer.Option(
    False,
    "--no-extras",
    help=cast(
        str,
        lazy_gettext("Fetch only the weights, not the vision projector or the draft model."),
    ),
)
_RECHECK_OPTION: bool = typer.Option(
    False,
    "--recheck",
    help=cast(
        str,
        lazy_gettext("Re-read files that are already here and check them against the catalog."),
    ),
)
_ALLOW_UNVERIFIED_OPTION: bool = typer.Option(
    False,
    "--allow-unverified",
    help=cast(
        str,
        lazy_gettext(
            "Fetch files the catalog holds no checksum for. Nothing will be able to tell "
            "you whether they arrived intact."
        ),
    ),
)
_LIMIT_OPTION: int = typer.Option(
    10, "--limit", min=1, help=cast(str, lazy_gettext("Show at most this many entries."))
)


def _status_text(*, present: bool) -> str:
    """What a file's Status cell says: already here, or still to fetch."""
    return _("already here") if present else _("to fetch")


def _plan_table(plan: DownloadPlan) -> Table:
    """The summary a reader sees before anything is fetched.

    One row per file with its size where the catalog knows it. The columns are few on
    purpose: this is read once, in a hurry, by somebody deciding whether to spend the next
    two hours on it. A shard's own size is not among the things the catalog knows, so that
    cell reads *unknown* and the honest total is on the line below the table.
    """
    table = Table(title=_("About to download"), title_justify="left", expand=False)
    table.add_column(_("File"), overflow="fold")
    table.add_column(_("Role"))
    table.add_column(_("Size"), justify="right")
    table.add_column(_("Status"))
    for file in plan.files:
        present = file.already_present()
        table.add_row(
            Text(for_display(isolate(file.name))),
            Text(for_display(isolate(file.role))),
            Text(for_display(isolate(format_bytes(file.expected_size)))),
            Text(for_display(_status_text(present=present)), style="green" if present else ""),
        )
    return table


def _print_plan(state: CliState, plan: DownloadPlan, check: DiskCheck) -> None:
    """Say what is about to happen and how big it is, before a socket is opened."""
    console = state.console
    console.print(
        Text(
            for_display(
                _("%(name)s %(quant)s from %(repo)s")
                % {
                    "name": isolate(plan.model_name),
                    "quant": isolate(plan.quant),
                    "repo": isolate(plan.repo),
                }
            ),
            style="bold",
        )
    )
    console.print(_plan_table(plan))
    console.print(
        Text(
            for_display(
                ngettext(
                    "%(count)d file, %(total)s in all, %(needed)s still to fetch.",
                    "%(count)d files, %(total)s in all, %(needed)s still to fetch.",
                    len(plan.files),
                )
                % {
                    "count": len(plan.files),
                    "total": isolate(format_bytes(plan.total_bytes)),
                    "needed": isolate(format_bytes(check.needed)),
                }
            )
        )
    )
    console.print(
        Text(
            for_display(
                _("Into %(path)s, which has %(free)s free; %(after)s would be left.")
                % {
                    "path": isolate(str(plan.directory)),
                    "free": isolate(format_bytes(check.free)),
                    "after": isolate(format_bytes(max(0, check.free - check.needed))),
                }
            )
        )
    )
    if plan.unverifiable:
        console.print(
            Text(
                for_display(
                    ngettext(
                        "The catalog holds no checksum for %(files)s.",
                        "The catalog holds no checksums for %(files)s.",
                        len(plan.unverifiable),
                    )
                    % {"files": isolate(", ".join(plan.unverifiable))}
                ),
                style="yellow",
            )
        )


def _reporter(state: CliState) -> ProgressReporter:
    """A live display, or nothing at all when the output is machine-readable."""
    if state.json_output:
        return NullProgress()
    return RichProgress(state.console)


def _outcome_payload(outcome: InstallOutcome) -> dict[str, object]:
    """The ``--json`` document for a finished install."""
    return {
        "model": outcome.plan.model_id,
        "quant": outcome.plan.quant,
        "repo": outcome.plan.repo,
        "directory": str(outcome.plan.directory),
        "manifest": str(outcome.manifest),
        "total_bytes": outcome.plan.total_bytes,
        "fetched_bytes": outcome.fetched_bytes,
        "resumed": outcome.resumed,
        "files": [
            {
                "name": file.name,
                "path": str(file.path),
                "bytes": file.size,
                "fetched_bytes": file.fetched,
                "already_present": file.already_present,
                "verified": file.verified,
                "sha256": file.sha256,
            }
            for file in outcome.files
        ],
    }


def _plan_payload(plan: DownloadPlan, check: DiskCheck) -> dict[str, object]:
    """The ``--json`` document for ``--dry-run``."""
    return {
        "model": plan.model_id,
        "quant": plan.quant,
        "repo": plan.repo,
        "directory": str(plan.directory),
        "total_bytes": plan.total_bytes,
        "needed_bytes": check.needed,
        "free_bytes": check.free,
        "headroom_bytes": check.headroom,
        "enough_space": check.ok,
        "unverifiable": list(plan.unverifiable),
        "files": [
            {
                "name": file.name,
                "repo_path": file.repo_path,
                "url": file.url,
                "target": str(file.target),
                "bytes": file.expected_size,
                "sha256": file.sha256,
                "role": file.role,
                "already_present": file.already_present(),
            }
            for file in plan.files
        ],
    }


@install_app.command(
    "model",
    help=cast(
        str,
        lazy_gettext(
            "Download a model's weights, resuming anything an earlier run left and "
            "checking every file against the catalog's checksum."
        ),
    ),
)
def install_model_command(
    ctx: typer.Context,
    model_id: str = _MODEL_ARGUMENT,
    quant: str | None = _QUANT_OPTION,
    directory: Path | None = _DIR_OPTION,
    workers: int = _WORKERS_OPTION,
    limit_rate: str | None = _RATE_OPTION,
    yes: bool = _YES_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    no_extras: bool = _NO_EXTRAS_OPTION,
    recheck: bool = _RECHECK_OPTION,
    allow_unverified: bool = _ALLOW_UNVERIFIED_OPTION,
) -> None:
    """Download a model's weights, resuming and verifying as it goes."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
    model = find_model(catalog, model_id)
    plan = build_plan(model, quant_name=quant, directory=directory, with_extras=not no_extras)
    check = check_disk_space(plan)

    if state.json_output and dry_run:
        typer.echo(json.dumps(_plan_payload(plan, check), indent=2))
        return
    if not state.json_output:
        _print_plan(state, plan, check)
    if dry_run:
        return

    # Refuse before asking, not after: a person should not be invited to confirm a
    # download that this is about to turn down anyway.
    prepare(plan, allow_unverified=allow_unverified)

    if plan.is_installed() and not recheck:
        if not state.json_output:
            state.console.print(
                Text(
                    for_display(
                        _("%(model)s %(quant)s is already here; nothing to do.")
                        % {"model": isolate(plan.model_id), "quant": isolate(plan.quant)}
                    )
                )
            )
        else:
            typer.echo(json.dumps(_plan_payload(plan, check), indent=2))
        return

    if not yes and not state.json_output and state.console.is_terminal:
        typer.confirm(_("Start the download?"), abort=True)

    options = DownloadOptions(
        workers=workers,
        limit_rate=None if limit_rate is None else check_size(limit_rate, option="--limit-rate"),
        recheck=recheck,
    )
    stop = threading.Event()
    with _StoppedByInterrupt(stop), HttpRangeReader() as reader:
        try:
            outcome = _run(plan, reader, options, _reporter(state), stop, allow_unverified)
        except DownloadCancelledError:
            _say_stopped(state, plan)
            raise typer.Exit(code=1) from None

    if state.json_output:
        typer.echo(json.dumps(_outcome_payload(outcome), indent=2))
        return
    _say_done(state, outcome)


def _run(
    plan: DownloadPlan,
    reader: HttpRangeReader,
    options: DownloadOptions,
    reporter: ProgressReporter,
    stop: threading.Event,
    allow_unverified: bool,
) -> InstallOutcome:
    """Do the download, with the live display running for as long as it lasts."""
    with ExitStack() as stack:
        if isinstance(reporter, RichProgress):
            stack.enter_context(reporter)
        return install_model(
            plan,
            reader,
            options=options,
            reporter=reporter,
            stop=stop,
            allow_unverified=allow_unverified,
            data_dir=get_paths().data_dir,
        )


def _say_stopped(state: CliState, plan: DownloadPlan) -> None:
    """Say that a stopped run kept what it had, which is the whole point of saying it."""
    state.console.print(
        Text(
            for_display(
                _(
                    "Stopped. What arrived is kept; run `llamafit install model %(model)s` "
                    "again to carry on from there."
                )
                % {"model": isolate(plan.model_id)}
            ),
            style="yellow",
        )
    )


def _say_done(state: CliState, outcome: InstallOutcome) -> None:
    """The last two lines: what is on disk now, and where."""
    console = state.console
    console.print(
        Text(
            for_display(
                ngettext(
                    "%(count)d file, %(size)s, checked against the catalog.",
                    "%(count)d files, %(size)s, checked against the catalog.",
                    len(outcome.files),
                )
                % {
                    "count": len(outcome.files),
                    "size": isolate(format_bytes(outcome.plan.total_bytes)),
                }
            ),
            style="green",
        )
    )
    console.print(
        Text(
            for_display(
                _("In %(path)s. Run `llamafit plan %(model)s` for the command line that runs it.")
                % {
                    "path": isolate(str(outcome.plan.directory)),
                    "model": isolate(outcome.plan.model_id),
                }
            )
        )
    )


class _StoppedByInterrupt:
    """Turn ``Ctrl+C`` into a flag the workers read, rather than into a traceback.

    A ``KeyboardInterrupt`` raised into whichever thread happened to be running would
    leave the pool half-torn-down and the part file's record possibly one chunk behind
    what is actually on disk. Setting an event instead lets every worker stop where it is,
    between pieces, with the record and the file agreeing — which is what makes the
    promise that the next run resumes worth printing.

    A second interrupt puts Python's own handler back and raises, because somebody
    pressing it twice means it now.
    """

    def __init__(self, stop: threading.Event) -> None:
        """Remember the event to set."""
        self.stop = stop
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> _StoppedByInterrupt:
        """Install the handler, when this thread is allowed to have one."""
        try:
            self._previous = signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        except (ValueError, OSError):
            # Not the main thread, or a platform without it: the download still works, it
            # just cannot be stopped this politely.
            self._installed = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Put the previous handler back."""
        if not self._installed:
            return
        with suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Ask the workers to stop; on a second press, let Python do its usual thing."""
        if self.stop.is_set():
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        self.stop.set()


@install_app.command(
    "history",
    help=cast(str, lazy_gettext("Show what has been downloaded, newest first.")),
)
def install_history_command(ctx: typer.Context, limit: int = _LIMIT_OPTION) -> None:
    """Show what has been downloaded, newest first."""
    state: CliState = ctx.obj
    records = recent(read_history(history_path(get_paths().data_dir)), limit)
    if state.json_output:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
        return
    console = state.console
    if not records:
        console.print(_("Nothing has been downloaded yet."))
        return
    table = Table(title=_("Downloads"), title_justify="left")
    table.add_column(_("When"))
    table.add_column(_("Model"), overflow="fold")
    table.add_column(_("Quant"))
    table.add_column(_("Size"), justify="right")
    table.add_column(_("Outcome"))
    for record in records:
        table.add_row(
            Text(record.at.strftime("%Y-%m-%d %H:%M")),
            Text(for_display(isolate(record.model_id))),
            Text(for_display(isolate(record.quant))),
            Text(for_display(isolate(format_bytes(record.total_bytes)))),
            Text(
                for_display(_("finished") if record.finished else _("stopped")),
                style="green" if record.finished else "yellow",
            ),
        )
    console.print(table)
