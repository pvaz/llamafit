# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit install``: the commands that fetch something large and write it to disk.

Three commands share this group and one rule holds all of them together: nobody should
start something large and irreversible without being told what it costs first. So each
command prints its whole plan -- what will be fetched, how big it is, where it will land,
how much room that leaves and what is already there -- before a byte is written, and only
then asks.

``install llama.cpp`` is the first command that writes outside our own directories. Its
plan is the release, the archive, its size, whether it can be checksummed, the directory
it will go into and what is already in that directory. ``--dry-run`` stops after the plan,
and so does ``--json`` without ``--yes``, because a machine-readable run should never be
the one that opens a prompt nobody is watching.

``install model`` is the one that can take two hours. Its plan is the model, the
quantisation, the repository, every file it will fetch, what the set weighs, how much of
it is already here and what would be left on the volume afterwards. In a terminal it then
asks; outside one -- a script, a pipeline -- it does not, because there is nobody there to
answer and whoever wrote the script asked by running it. ``Ctrl+C`` is a normal outcome
rather than a crash: the event the workers watch is set, the part files and their records
stay where they are, and the last line says the run can be picked up where it stopped.
That claim is only worth making because :mod:`llamafit.download.state` makes it true.

``install history`` asks nothing and writes nothing; it says what the other two already
fetched.

Every ``help=`` here is deferred for the reason :mod:`llamafit.cli.app` gives: a decorator
runs while the module is imported, long before a language has been chosen, so an eager
``_()`` would freeze the interface in English. Every figure goes through
:mod:`llamafit.units`, and every identifier inside a sentence is isolated, so the output
reads correctly in a language written right to left.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, cast, get_args

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
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
from llamafit.errors import ConfigError
from llamafit.hardware import current_arch, current_os
from llamafit.hardware.gpu import detect_gpus
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.i18n import _, for_display, isolate, lazy_gettext, ngettext
from llamafit.llamacpp.install import (
    HttpFetcher,
    InstallError,
    InstallPlan,
    InstallResult,
    PathChange,
    ProgressCallback,
    apply_path_change,
    install_release,
    managed_dir,
    plan_install,
    plan_path_change,
    uninstall_note,
)
from llamafit.llamacpp.releases import (
    HttpReleaseClient,
    Release,
    ReleaseClient,
    companion_assets,
    resolve_backend,
)
from llamafit.models.host import Arch, Backend, Gpu, OsName, Vendor
from llamafit.paths import get_paths
from llamafit.units import format_bytes

VALID_BACKENDS: tuple[str, ...] = get_args(Backend)
"""Every backend name the rest of LlamaFit uses, taken from its own type."""

install_app = typer.Typer(
    name="install",
    help=cast(
        str,
        lazy_gettext("Install what it takes to run a model: llama.cpp itself, and the weights."),
    ),
)
app.add_typer(install_app, name="install")

# --- llamafit install llama.cpp ----------------------------------------------------------

# Module-level singletons rather than inline `typer.Option(...)` calls, and every help=
# deferred: the same two reasons as in catalog_cmd.py -- ruff's B008 does not recognise
# every annotation shape as safe to call in a default position, and a decorator runs at
# import time, before any language has been chosen, so an eager _() would freeze the
# interface in English.
_BACKEND_OPTION: str | None = typer.Option(
    None,
    "--backend",
    help=cast(
        str,
        lazy_gettext(
            "Install this backend instead of the one chosen for your GPU: "
            "cuda, vulkan, hip, sycl, metal or cpu."
        ),
    ),
)
_LLAMACPP_DIR_OPTION: Path | None = typer.Option(
    None,
    "--dir",
    help=cast(
        str,
        lazy_gettext("Install here instead of ~/.llamafit/llama.cpp."),
    ),
)
_TAG_OPTION: str | None = typer.Option(
    None,
    "--tag",
    help=cast(str, lazy_gettext("Install this release tag, for example b6100, not the latest.")),
)


def _check_backend(value: str | None) -> Backend | None:
    """Return the backend, or refuse it by name and list the ones that exist."""
    if value is None:
        return None
    if value not in VALID_BACKENDS:
        raise ConfigError(
            _("invalid --backend %(value)s") % {"value": repr(value)},
            hint=_("Valid backends: %(values)s") % {"values": ", ".join(VALID_BACKENDS)},
        )
    return cast("Backend", value)


def _primary_gpu(runner: Runner, os_name: OsName) -> Gpu | None:
    """The machine's main graphics device, or ``None`` when it has none.

    Two things are read from it and nothing else: the vendor, which chooses the backend,
    and the driver version, which chooses between the CUDA archives a release publishes
    side by side. So only the cheap probes run — no bandwidth measurement, no llama.cpp
    scan, nothing that would make an install wait on work it does not use.
    """
    gpus, _probes = detect_gpus(runner, os_name)
    if not gpus:
        return None
    return max(gpus, key=lambda gpu: gpu.vram_total_bytes or -1)


def _release_for(client: ReleaseClient, tag: str | None) -> Release:
    """The release to install: the one named, or the newest published."""
    return client.by_tag(tag) if tag else client.latest()


def _build_plan(
    release: Release,
    *,
    os_name: OsName,
    arch: Arch,
    vendor: Vendor | None,
    driver: str | None,
    wanted: Backend | None,
    root: Path,
    cache_dir: Path,
) -> InstallPlan:
    """Choose the archive for this machine and describe what installing it would do.

    Raises:
        InstallError: The release publishes nothing for this machine, or nothing for
            the backend that was asked for by name.
    """
    chosen = resolve_backend(
        release.assets, os_name=os_name, arch=arch, vendor=vendor, wanted=wanted, driver=driver
    )
    if chosen is None:
        if wanted is not None:
            raise InstallError(
                _("release %(tag)s publishes no %(backend)s build for %(os)s %(arch)s.")
                % {"tag": release.tag, "backend": wanted, "os": os_name, "arch": arch},
                hint=_("Leave --backend out and LlamaFit will choose a published one."),
            )
        raise InstallError(
            _("release %(tag)s publishes nothing for %(os)s %(arch)s.")
            % {"tag": release.tag, "os": os_name, "arch": arch},
            hint=_("Build llama.cpp from source; see the project's own instructions."),
        )
    backend, asset = chosen
    return plan_install(
        release,
        asset,
        companions=companion_assets(release.assets, asset, os_name=os_name, backend=backend),
        backend=backend,
        os_name=os_name,
        arch=arch,
        root=root,
        cache_dir=cache_dir,
    )


def _release_plan_payload(plan: InstallPlan) -> dict[str, Any]:
    """The plan as JSON: identifiers stay identifiers, sizes stay numbers."""
    return {
        "tag": plan.release.tag,
        "build": plan.release.build,
        "commit": plan.release.commit,
        "release_url": plan.release.url,
        "backend": plan.backend,
        "os": plan.os_name,
        "arch": plan.arch,
        "asset": plan.asset.name,
        "asset_bytes": plan.asset.size,
        "asset_sha256": plan.asset.sha256,
        "companions": [
            {"name": extra.name, "bytes": extra.size, "sha256": extra.sha256}
            for extra in plan.companions
        ],
        "verified": plan.verified,
        "target": str(plan.target.path),
        "bin_dir": str(plan.bin_dir),
        "ownership": plan.target.ownership,
        "replaces_tag": plan.replaces.tag if plan.replaces else None,
        "archive": str(plan.archive_path),
        "resume_bytes": plan.resume_bytes,
        "download_bytes": plan.download_bytes,
        "space": [
            {
                "path": str(check.path),
                "purpose": check.purpose,
                "needed_bytes": check.needed,
                "free_bytes": check.free,
                "enough": check.enough,
            }
            for check in plan.space
        ],
    }


def _result_payload(result: InstallResult, change: PathChange | None) -> dict[str, Any]:
    """The finished install as JSON."""
    return {
        "installed": True,
        "root": str(result.root),
        "bin_dir": str(result.bin_dir),
        "tag": result.marker.tag,
        "build": result.marker.build,
        "commit": result.marker.commit,
        "backend": result.marker.backend,
        "backends": list(result.backends),
        "replaced_tag": result.replaced.tag if result.replaced else None,
        "warnings": list(result.warnings),
        "path_change": None
        if change is None
        else {"kind": change.kind, "where": change.where, "applied": change.needed},
    }


def _release_plan_table(plan: InstallPlan) -> Table:
    """The plan as the table a person reads before agreeing to it."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    build = plan.release.build
    table.add_row(
        _("Release"),
        Text(
            _("%(tag)s (build %(build)s)") % {"tag": plan.release.tag, "build": build}
            if build is not None
            else plan.release.tag
        ),
    )
    table.add_row(_("Backend"), Text(plan.backend))
    table.add_row(_("Archive"), Text(isolate(plan.asset.name)))
    for extra in plan.companions:
        table.add_row(_("Also"), Text(isolate(extra.name)))
    table.add_row(
        _("Download"),
        Text(
            _("%(size)s (%(done)s already on disk)")
            % {
                "size": format_bytes(plan.download_bytes),
                "done": format_bytes(plan.resume_bytes),
            }
            if plan.resume_bytes
            else format_bytes(plan.download_bytes)
        ),
    )
    table.add_row(
        _("Checksum"),
        Text(
            _("published, checked before anything is unpacked")
            if plan.verified
            else _("none published"),
            style="" if plan.verified else "yellow",
        ),
    )
    table.add_row(_("Install into"), Text(isolate(str(plan.bin_dir))))
    table.add_row(_("Directory"), Text(_ownership_line(plan), style=_ownership_style(plan)))
    for check in plan.space:
        table.add_row(
            _("Free space") if check.purpose == "install" else _("Free space (cache)"),
            Text(
                _("%(free)s on %(path)s, %(needed)s needed")
                % {
                    "free": format_bytes(check.free),
                    "path": isolate(str(check.path)),
                    "needed": format_bytes(check.needed),
                },
                style="" if check.enough else "red",
            ),
        )
    return table


def _ownership_line(plan: InstallPlan) -> str:
    """One sentence about what is already in the install directory."""
    target = plan.target
    if target.ownership == "absent":
        return _("will be created")
    if target.ownership == "empty":
        return _("exists and is empty")
    if target.ownership == "ours":
        marker = target.marker
        if marker is not None and marker.tag:
            return _("holds %(tag)s, installed by LlamaFit; it will be replaced") % {
                "tag": marker.tag
            }
        return _("installed by LlamaFit; it will be replaced")
    return target.detail or _("was not installed by LlamaFit")


def _ownership_style(plan: InstallPlan) -> str:
    return "red" if plan.target.ownership == "foreign" else ""


@contextmanager
def _progress_bar(console: Console) -> Iterator[ProgressCallback]:
    """A download bar, yielding the callback the installer reports bytes to."""
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console, transient=True) as bar:
        task = bar.add_task(_("Downloading"), total=None)

        def update(done: int, total: int | None) -> None:
            bar.update(task, completed=done, total=total)

        yield update


def _report(console: Console, result: InstallResult) -> None:
    """Say what was installed, what was detected in it, and how to undo it."""
    console.print(
        Text(
            _("Installed llama.cpp %(tag)s into %(path)s.")
            % {"tag": result.marker.tag, "path": isolate(str(result.bin_dir))},
            style="green",
        )
    )
    if result.replaced is not None and result.replaced.tag:
        console.print(Text(_("Replaced %(tag)s.") % {"tag": result.replaced.tag}, style="dim"))
    console.print(
        Text(
            _("Backends found in it: %(backends)s")
            % {"backends": ", ".join(result.backends) or _("none")}
        )
    )
    for warning in result.warnings:
        console.print(Text(warning, style="yellow"))
    console.print(Text(_("Check it with `llamafit doctor`."), style="dim"))
    console.print(Text(uninstall_note(result.root), style="dim"))


def _offer_path(
    console: Console,
    *,
    bin_dir: Path,
    os_name: OsName,
    runner: Runner,
    assume_yes: bool,
) -> PathChange | None:
    """Show the change and its undo together, ask, and apply only on a yes.

    The undo is printed in the same breath as the offer, never afterwards: the change
    outlives this process, and a person agreeing to it should already know how to take
    it back.
    """
    change = plan_path_change(
        bin_dir, os_name=os_name, runner=runner, home=Path.home(), env=os.environ
    )
    console.print(Text(change.description))
    if not change.needed:
        return change
    console.print(Text(change.undo, style="dim"))
    if not assume_yes and not typer.confirm(_("Change your PATH?"), default=False):
        console.print(Text(_("PATH left alone."), style="dim"))
        return None
    apply_path_change(change, runner=runner)
    console.print(
        Text(_("PATH changed. Open a new terminal for it to take effect."), style="green")
    )
    return change


@install_app.command(
    "llama.cpp",
    help=cast(
        str,
        lazy_gettext(
            "Download a published llama.cpp build and install it into ~/.llamafit/llama.cpp.\n\n"
            "The plan is printed before anything is written. The archive's checksum is "
            "verified before it is unpacked, an interrupted download continues where it "
            "stopped, and a directory LlamaFit did not create is never overwritten."
        ),
    ),
)
def llamacpp_command(
    ctx: typer.Context,
    backend: str | None = _BACKEND_OPTION,
    directory: Path | None = _LLAMACPP_DIR_OPTION,
    tag: str | None = _TAG_OPTION,
    add_to_path: bool = typer.Option(
        False,
        "--add-to-path",
        help=cast(
            str,
            lazy_gettext("Offer to add the bin directory to your PATH, showing the undo first."),
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=cast(
            str,
            lazy_gettext("Install over a directory LlamaFit did not create. Ask yourself twice."),
        ),
    ),
    allow_unverified: bool = typer.Option(
        False,
        "--allow-unverified",
        help=cast(
            str,
            lazy_gettext("Install an archive that publishes no checksum. Nothing can be checked."),
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help=cast(str, lazy_gettext("Print the plan and write nothing."))
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help=cast(str, lazy_gettext("Do not ask; assume yes."))
    ),
) -> None:
    """Download a published llama.cpp build and install it into ~/.llamafit/llama.cpp."""
    state: CliState = ctx.obj
    console = state.console
    wanted = _check_backend(backend)
    os_name = current_os()
    arch = current_arch()
    runner: Runner = SubprocessRunner()
    root = Path(directory).expanduser() if directory is not None else managed_dir()

    gpu = _primary_gpu(runner, os_name)
    with httpx.Client(follow_redirects=True) as http:
        client = HttpReleaseClient(client=http)
        release = _release_for(client, tag)
        plan = _build_plan(
            release,
            os_name=os_name,
            arch=arch,
            vendor=gpu.vendor if gpu else None,
            driver=gpu.driver if gpu else None,
            wanted=wanted,
            root=root,
            cache_dir=get_paths().cache_dir,
        )

        # --json without --yes is a dry run on purpose: a machine-readable run must not
        # be the one that stops at a prompt nobody is there to answer.
        preview_only = dry_run or (state.json_output and not yes)
        if state.json_output:
            payload = _release_plan_payload(plan)
            payload["applied"] = False
            if preview_only:
                typer.echo(json.dumps(payload, indent=2))
                return
        else:
            console.print(_release_plan_table(plan))
            if preview_only:
                return
            if not yes and not typer.confirm(_("Install this?"), default=True):
                console.print(Text(_("Nothing was written."), style="dim"))
                raise typer.Exit(code=1)

        fetcher = HttpFetcher(client=http)
        if state.json_output:
            result = install_release(
                plan, fetcher=fetcher, force=force, require_checksum=not allow_unverified
            )
        else:
            with _progress_bar(console) as progress:
                result = install_release(
                    plan,
                    fetcher=fetcher,
                    progress=progress,
                    force=force,
                    require_checksum=not allow_unverified,
                )

    change = (
        _offer_path(
            console if not state.json_output else Console(quiet=True),
            bin_dir=result.bin_dir,
            os_name=os_name,
            runner=runner,
            assume_yes=yes,
        )
        if add_to_path
        else None
    )

    if state.json_output:
        typer.echo(json.dumps(_result_payload(result, change), indent=2))
    else:
        _report(console, result)


# --- llamafit install model, and what has already been fetched ----------------------------

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


def _status_text(*, present: bool) -> str:
    """What a file's Status cell says: already here, or still to fetch."""
    return _("already here") if present else _("to fetch")


def _download_plan_table(plan: DownloadPlan) -> Table:
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
    console.print(_download_plan_table(plan))
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


def _download_plan_payload(plan: DownloadPlan, check: DiskCheck) -> dict[str, object]:
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
        typer.echo(json.dumps(_download_plan_payload(plan, check), indent=2))
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
            typer.echo(json.dumps(_download_plan_payload(plan, check), indent=2))
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


__all__ = [
    "install_app",
    "install_history_command",
    "install_model_command",
    "llamacpp_command",
]
