# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit install``: the group, the weights it fetches, and what it has already fetched.

One rule holds the whole group together: nobody should start something large and
irreversible without being told what it costs first. So every command under it prints its
whole plan -- what will be fetched, how big it is, where it will land, how much room that
leaves and what is already there -- before a byte is written, and only then asks.

``install model`` is the one that can take two hours, and the two hours are what shape it.
It plans, refuses before asking rather than after, and skips the question outside a
terminal -- a script, a pipeline -- because there is nobody there to answer and whoever
wrote the script asked by running it. ``Ctrl+C`` is a normal outcome rather than a crash:
the event the workers watch is set, the part files and their records stay where they are,
and the last line says the run can be picked up where it stopped. That claim is only worth
making because :mod:`llamafit.download.state` makes it true.

``install history`` asks nothing and writes nothing; it says what the other commands
already fetched, which is why it sits beside the one that writes those records.

``install llama.cpp`` is the group's third command and lives in
:mod:`llamafit.cli.llamacpp_cmd`. It installs a program rather than data -- a published
archive, chosen for this machine's card, unpacked over a directory that may not be ours --
and it shares nothing with these two but the group and the rule above. What a reader sees
is in :mod:`llamafit.cli.render_install` and what a script reads is in
:mod:`llamafit.cli.install_json`, for all three.

Every ``help=`` here is deferred for the reason :mod:`llamafit.cli.app` gives: a decorator
runs while the module is imported, long before a language has been chosen, so an eager
``_()`` would freeze the interface in English.
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

from llamafit.cli.app import CliState, app
from llamafit.cli.common import check_size, find_model, load_catalog_or_warn
from llamafit.cli.install_json import download_outcome_payload, download_plan_payload
from llamafit.cli.render_install import (
    render_already_here,
    render_download_done,
    render_download_plan,
    render_download_stopped,
    render_history,
)
from llamafit.download.engine import DEFAULT_WORKERS, MAX_WORKERS, DownloadOptions
from llamafit.download.errors import DownloadCancelledError
from llamafit.download.history import history_path, read_history, recent
from llamafit.download.install import InstallOutcome, install_model, prepare
from llamafit.download.plan import DownloadPlan, build_plan, check_disk_space
from llamafit.download.progress import NullProgress, ProgressReporter, RichProgress
from llamafit.download.transport import HttpRangeReader
from llamafit.i18n import _, lazy_gettext
from llamafit.paths import get_paths

install_app = typer.Typer(
    name="install",
    help=cast(
        str,
        lazy_gettext("Install what it takes to run a model: llama.cpp itself, and the weights."),
    ),
)
app.add_typer(install_app, name="install")

# --- llamafit install model ---------------------------------------------------------------

# Module-level singletons rather than inline `typer.Option(...)`/`typer.Argument(...)`
# calls, and every help= deferred: the same two reasons as in catalog_cmd.py -- ruff's B008
# does not recognise every annotation shape as safe to call in a default position, and a
# decorator runs at import time, before any language has been chosen, so an eager _() would
# freeze the interface in English.
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
_MODEL_DIR_OPTION: Path | None = typer.Option(
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


def _reporter(state: CliState) -> ProgressReporter:
    """A live display, or nothing at all when the output is machine-readable."""
    if state.json_output:
        return NullProgress()
    return RichProgress(state.console)


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
    directory: Path | None = _MODEL_DIR_OPTION,
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
        typer.echo(json.dumps(download_plan_payload(plan, check), indent=2))
        return
    if not state.json_output:
        state.console.print(render_download_plan(plan, check))
    if dry_run:
        return

    # Refuse before asking, not after: a person should not be invited to confirm a
    # download that this is about to turn down anyway.
    prepare(plan, allow_unverified=allow_unverified)

    if plan.is_installed() and not recheck:
        if not state.json_output:
            state.console.print(render_already_here(plan))
        else:
            typer.echo(json.dumps(download_plan_payload(plan, check), indent=2))
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
            state.console.print(render_download_stopped(plan))
            raise typer.Exit(code=1) from None

    if state.json_output:
        typer.echo(json.dumps(download_outcome_payload(outcome), indent=2))
        return
    state.console.print(render_download_done(outcome))


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


class _StoppedByInterrupt:
    """Turn ``Ctrl+C`` into a flag the workers read, rather than into a traceback.

    A ``KeyboardInterrupt`` raised into whichever thread happened to be running would
    leave the pool half-torn-down and the part file's record possibly one chunk behind
    what is actually on disk. Setting an event instead lets every worker stop where it is,
    between pieces, with the record and the file agreeing -- which is what makes the
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


# --- llamafit install history -------------------------------------------------------------


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
    if not records:
        state.console.print(_("Nothing has been downloaded yet."))
        return
    state.console.print(render_history(records))


__all__ = [
    "install_app",
    "install_history_command",
    "install_model_command",
]
