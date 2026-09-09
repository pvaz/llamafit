"""Free disk space at the paths that matter."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from llamafit.models.host import Disk


def detect_disks(paths: Iterable[Path]) -> list[Disk]:
    """One ``Disk`` per distinct mount among ``paths``.

    Paths are made absolute first, so a root-relative path such as
    ``/definitely/missing/xyz`` resolves against the current drive on Windows before
    walking up; missing paths fall back to their nearest existing parent.
    """
    seen: dict[str, Disk] = {}
    for path in paths:
        candidate = Path(path).absolute()
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        if not candidate.exists():
            continue
        anchor = str(candidate.anchor or candidate)
        try:
            usage = shutil.disk_usage(candidate)
        except OSError:
            continue
        if anchor not in seen:
            seen[anchor] = Disk(path=anchor, free_bytes=usage.free, total_bytes=usage.total)
    return list(seen.values())
