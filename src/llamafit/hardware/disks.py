"""Free disk space at the paths that matter."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from llamafit.models.host import Disk


def detect_disks(paths: Iterable[Path]) -> list[Disk]:
    """One ``Disk`` per distinct mount among ``paths``; missing paths fall back to parents."""
    seen: dict[str, Disk] = {}
    for path in paths:
        candidate = Path(path)
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
