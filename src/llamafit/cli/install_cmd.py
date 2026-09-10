# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit install llama.cpp``: the first command that writes outside our own directories.

Everything a person needs in order to say no is printed before anything is written: the
release, the archive, its size, whether it can be checksummed, where it will go, how much
room that leaves, and what is already in the directory. Only then is the question asked.

``--dry-run`` stops after the plan. So does ``--json`` without ``--yes``, because a
machine-readable run should never be the one that opens a prompt nobody is watching.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast, get_args

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from llamafit.cli.app import CliState, app
from llamafit.errors import ConfigError
from llamafit.hardware import current_arch, current_os
from llamafit.hardware.gpu import detect_gpus
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.i18n import _, isolate, lazy_gettext
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
    help=cast(str, lazy_gettext("Install what LlamaFit needs: llama.cpp, and later models.")),
)
app.add_typer(install_app, name="install")

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
_DIR_OPTION: Path | None = typer.Option(
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


def _plan_payload(plan: InstallPlan) -> dict[str, Any]:
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


def _plan_table(plan: InstallPlan) -> Table:
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
    directory: Path | None = _DIR_OPTION,
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
            payload = _plan_payload(plan)
            payload["applied"] = False
            if preview_only:
                typer.echo(json.dumps(payload, indent=2))
                return
        else:
            console.print(_plan_table(plan))
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


__all__ = ["install_app", "llamacpp_command"]
