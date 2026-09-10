# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Free disk space at the paths that matter.

Volumes are identified by drive letter on Windows and by device id (``st_dev``)
everywhere else, since every absolute POSIX path shares the anchor ``/`` regardless
of which filesystem it lives on.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from llamafit.models.host import Disk


def _mount_key(path: Path) -> str:
    """Identify the volume a path lives on: the drive on Windows, the device id elsewhere."""
    if os.name == "nt":
        return path.anchor.lower()
    return str(os.stat(path).st_dev)


def detect_disks(paths: Iterable[Path]) -> list[Disk]:
    """One ``Disk`` per distinct volume among ``paths``.

    Paths are made absolute first, so a root-relative path such as
    ``/definitely/missing/xyz`` resolves against the current drive on Windows before
    walking up to the nearest existing ancestor. Volumes are deduplicated by
    ``_mount_key`` (drive letter on Windows, device id elsewhere); unreadable or
    missing paths are skipped.
    """
    seen: dict[str, Disk] = {}
    for path in paths:
        candidate = Path(path).absolute()
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        if not candidate.exists():
            continue
        try:
            key = _mount_key(candidate)
            usage = shutil.disk_usage(candidate)
        except OSError:
            continue
        if key not in seen:
            seen[key] = Disk(path=str(candidate), free_bytes=usage.free, total_bytes=usage.total)
    return list(seen.values())
