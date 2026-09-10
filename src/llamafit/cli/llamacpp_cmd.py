# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit install llama.cpp``: put the program that runs a model on this machine.

This is the first command that writes outside LlamaFit's own directories, and everything
about it is arranged around that. Its plan is the release, the archive chosen for this
machine, its size, whether it can be checksummed, the directory it will go into and what is
already in that directory — all printed before a byte is written. ``--dry-run`` stops after
the plan, and so does ``--json`` without ``--yes``, because a machine-readable run should
never be the one that opens a prompt nobody is watching. A directory LlamaFit did not
create is never written over without ``--force``.

Choosing the archive is the part with a judgement in it, and it is three small steps kept
apart on purpose: which backend was asked for (or none), what card this machine has, and
what the release actually publishes for that pair. Only the cheap GPU probes run — the
vendor and the driver version are all a backend choice needs — so an install never waits on
a bandwidth measurement it will not use.

``--add-to-path`` prints the change and its undo in the same breath as the offer, never
afterwards: the change outlives this process, and a person agreeing to it should already
know how to take it back.

It shares the ``install`` group, and only the group, with the commands that fetch weights
in :mod:`llamafit.cli.install_cmd`. What a reader sees is in
:mod:`llamafit.cli.render_install` and what a script reads is in
:mod:`llamafit.cli.install_json`.

Every ``help=`` here is deferred for the reason :mod:`llamafit.cli.app` gives: a decorator
runs while the module is imported, long before a language has been chosen, so an eager
``_()`` would freeze the interface in English.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast, get_args

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
from rich.text import Text

from llamafit.cli.app import CliState
from llamafit.cli.install_cmd import install_app
from llamafit.cli.install_json import release_plan_payload, release_result_payload
from llamafit.cli.render_install import render_release_plan, render_release_result
from llamafit.errors import ConfigError
from llamafit.hardware import current_arch, current_os
from llamafit.hardware.gpu import detect_gpus
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.i18n import _, lazy_gettext
from llamafit.llamacpp.install import (
    HttpFetcher,
    InstallError,
    InstallPlan,
    PathChange,
    ProgressCallback,
    apply_path_change,
    install_release,
    managed_dir,
    plan_install,
    plan_path_change,
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

VALID_BACKENDS: tuple[str, ...] = get_args(Backend)
"""Every backend name the rest of LlamaFit uses, taken from its own type."""

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
            payload = release_plan_payload(plan)
            payload["applied"] = False
            if preview_only:
                typer.echo(json.dumps(payload, indent=2))
                return
        else:
            console.print(render_release_plan(plan))
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
        typer.echo(json.dumps(release_result_payload(result, change), indent=2))
    else:
        console.print(render_release_result(result))


__all__ = ["VALID_BACKENDS", "llamacpp_command"]
