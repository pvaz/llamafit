# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Installing a model: the order the steps have to happen in, and what is written at the end.

The order is the interesting part, and it is the same order the failures argue for.

1. **Build the plan** from the catalog. No network, so a mistyped quant name or a catalog
   that has never been refreshed fails in a second rather than after a redirect.
2. **Refuse what cannot be checked**, unless the user said otherwise in so many words.
3. **Check the disk.** Before the first byte. This is the step whose absence costs hours.
4. **Fetch each file**, resuming whatever an earlier run left, verifying each one against
   the catalog's checksum before it is moved into place.
5. **Write the manifest**, and only then. Its presence is what "this model is installed"
   means, and a run that fetched three shards of four does not write one. That is how a
   split model stays one thing to the person who asked for it.
6. **Record it in the history**, finished or not.

Step five is also where a shard's real size is recorded, since the catalog holds one size
for a whole quantisation; without it, the next run could not tell a finished shard from a
file that happens to have the right name.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llamafit import __version__
from llamafit.download.engine import DownloadOptions, FileOutcome, download_plan
from llamafit.download.errors import DownloadError
from llamafit.download.history import DownloadRecord, append_record, history_path
from llamafit.download.plan import (
    DISK_HEADROOM_BYTES,
    DiskCheck,
    DownloadPlan,
    check_disk_space,
    refuse_unverifiable,
)
from llamafit.download.progress import NullProgress, ProgressReporter
from llamafit.download.transport import RangeReader
from llamafit.errors import LlamaFitError
from llamafit.i18n import _

MANIFEST_SCHEMA_VERSION = 1
"""Bumped when the manifest's shape changes."""


@dataclass(frozen=True)
class InstallOutcome:
    """What one ``install model`` run did.

    Attributes:
        plan: What it set out to do.
        files: What happened to each file, in plan order.
        manifest: Where the completion record was written.
        fetched_bytes: What this run pulled over the network.
    """

    plan: DownloadPlan
    files: tuple[FileOutcome, ...]
    manifest: Path
    fetched_bytes: int

    @property
    def verified(self) -> int:
        """How many files were checked against a checksum this run."""
        return sum(1 for file in self.files if file.verified)

    @property
    def resumed(self) -> bool:
        """Whether any file continued a transfer an earlier run had started."""
        return any(file.resumed for file in self.files)


def write_manifest(plan: DownloadPlan, files: tuple[FileOutcome, ...]) -> Path:
    """Record that every file of this model is here, checked, and what size each is.

    Written to a sibling temporary path and moved into place, so an interrupted write
    leaves no manifest rather than half of one — and no manifest is exactly the state that
    means "not installed", which is the safe way for this particular write to fail.

    Args:
        plan: The model that was installed.
        files: What happened to each file.

    Returns:
        Where the manifest was written.

    Raises:
        DownloadError: If it cannot be written, which would leave a complete download that
            nothing could recognise as one.
    """
    document = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "llamafit": __version__,
        "model": plan.model_id,
        "name": plan.model_name,
        "quant": plan.quant,
        "repo": plan.repo,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "name": file.name,
                "bytes": file.size,
                "sha256": file.sha256,
                "verified": file.verified,
            }
            for file in files
        ],
    }
    path = plan.manifest_path
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        raise DownloadError(
            _("could not write %(path)s: %(error)s") % {"path": path, "error": exc},
            hint=_("Check that the download directory is writable."),
        ) from exc
    return path


def prepare(
    plan: DownloadPlan,
    *,
    allow_unverified: bool = False,
    headroom: int = DISK_HEADROOM_BYTES,
    free_bytes: int | None = None,
) -> DiskCheck:
    """Everything that has to be true before the first byte is written.

    Args:
        plan: What is about to be downloaded.
        allow_unverified: Whether to proceed without checksums for some files.
        headroom: How much disk to leave spare.
        free_bytes: The free figure to use instead of measuring, for tests.

    Returns:
        The disk check, so the caller can show the numbers it was satisfied by.

    Raises:
        DownloadError: If a file has no checksum and ``allow_unverified`` was not given.
        DiskSpaceError: If the volume cannot hold what is left to fetch.
    """
    if not allow_unverified:
        refuse_unverifiable(plan)
    check = check_disk_space(plan, headroom=headroom, free_bytes=free_bytes)
    check.raise_if_short()
    return check


def install_model(
    plan: DownloadPlan,
    reader: RangeReader,
    *,
    options: DownloadOptions | None = None,
    reporter: ProgressReporter | None = None,
    stop: threading.Event | None = None,
    allow_unverified: bool = False,
    headroom: int = DISK_HEADROOM_BYTES,
    free_bytes: int | None = None,
    data_dir: Path | None = None,
) -> InstallOutcome:
    """Fetch, verify and record one model.

    Args:
        plan: What to install.
        reader: Where bytes come from.
        options: How many workers, how big a chunk, whether to re-check what is here.
        reporter: Who to tell about progress.
        stop: Set to cancel; everything already fetched stays on disk.
        allow_unverified: Proceed even for files the catalog holds no checksum for.
        headroom: How much disk to leave spare.
        free_bytes: The free figure to use instead of measuring, for tests.
        data_dir: Where the downloads history lives, or ``None`` to skip recording.

    Returns:
        What the run did.

    Raises:
        LlamaFitError: If anything went wrong -- a :class:`DownloadError` for a refusal
            this package made, a :class:`~llamafit.errors.NetworkError` for one the
            network made. Whatever was verified stays where it is, so the next run
            resumes rather than restarts, and no manifest is written, so nothing
            believes a half-fetched model is installed.
    """
    reporter = reporter or NullProgress()
    prepare(
        plan,
        allow_unverified=allow_unverified,
        headroom=headroom,
        free_bytes=free_bytes,
    )
    plan.directory.mkdir(parents=True, exist_ok=True)
    try:
        outcomes = tuple(download_plan(plan, reader, options=options, reporter=reporter, stop=stop))
    except LlamaFitError as exc:
        # LlamaFitError rather than DownloadError: a server that answers 500 raises the
        # project's own NetworkError, and a history that recorded only this package's own
        # failures would be missing exactly the ones a user comes back to ask about.
        _record(plan, (), data_dir, error=exc.message)
        raise
    manifest = write_manifest(plan, outcomes)
    _record(plan, outcomes, data_dir)
    return InstallOutcome(
        plan=plan,
        files=outcomes,
        manifest=manifest,
        fetched_bytes=sum(file.fetched for file in outcomes),
    )


def _record(
    plan: DownloadPlan,
    outcomes: tuple[FileOutcome, ...],
    data_dir: Path | None,
    *,
    error: str | None = None,
) -> None:
    """Add this run to the downloads history, when there is somewhere to put it."""
    if data_dir is None:
        return
    append_record(
        history_path(data_dir),
        DownloadRecord(
            model_id=plan.model_id,
            quant=plan.quant,
            repo=plan.repo,
            directory=str(plan.directory),
            files=[file.name for file in plan.files],
            total_bytes=plan.total_bytes,
            fetched_bytes=sum(file.fetched for file in outcomes),
            finished=error is None,
            error=error,
        ),
    )
