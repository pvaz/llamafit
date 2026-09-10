# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The half-finished file on disk, and the record of which parts of it have arrived.

A file is downloaded into ``<name>.part``, created at its full final size so that any
worker can seek to its own range and write there. Beside it sits ``<name>.part.state``, a
small JSON document naming the URL, the total size, the chunk size and every chunk index
already written. That sidecar is the whole of what makes resume real: the length of the
part file says nothing, because sixteen workers fill it with holes.

A chunk is recorded only after its last byte has been written and flushed, so the record is
always behind the file and never ahead of it. The cost of being behind is re-fetching one
chunk; the cost of being ahead would be a file with a hole in it that passes every check
until the checksum, hours later.

The state is written the way :mod:`llamafit.catalog.refresh` writes the facts file: to a
sibling temporary path, then moved into place with :func:`os.replace`. A crash during the
write leaves the previous record or the new one, never half of either.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from llamafit.download.errors import DownloadError
from llamafit.i18n import _

STATE_SCHEMA_VERSION = 1
"""Bumped when the sidecar's shape changes; an older one is discarded, not guessed at."""

DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
"""How much of a file one request asks for: 32 MiB.

Large enough that the per-request overhead disappears against the transfer and that a
hundred-gigabyte model is a few thousand chunks rather than a hundred thousand; small
enough that an interruption costs at most half a minute of a home connection's work, and
that a rate-limited server is told to slow down within seconds rather than minutes.
"""

PART_SUFFIX = ".part"
STATE_SUFFIX = ".part.state"


def part_path_for(target: Path) -> Path:
    """Where the half-finished copy of ``target`` lives."""
    return target.with_name(target.name + PART_SUFFIX)


def state_path_for(target: Path) -> Path:
    """Where the record of ``target``'s finished chunks lives."""
    return target.with_name(target.name + STATE_SUFFIX)


@dataclass
class PartFile:
    """A file being downloaded, and the chunks of it already on disk.

    Attributes:
        target: Where the finished file will go.
        url: What is being fetched. A change of URL invalidates the record, because two
            URLs are two files until something proves otherwise.
        size: The file's final size, from the catalog.
        chunk_size: How many bytes one chunk covers.
        done: The indices of the chunks already written and flushed.
    """

    target: Path
    url: str
    size: int
    chunk_size: int = DEFAULT_CHUNK_BYTES
    done: set[int] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def path(self) -> Path:
        """The half-finished file itself."""
        return part_path_for(self.target)

    @property
    def state_path(self) -> Path:
        """The sidecar recording which chunks are in it."""
        return state_path_for(self.target)

    @property
    def chunk_count(self) -> int:
        """How many chunks the file is divided into; one for an empty file."""
        if self.size <= 0:
            return 1
        return -(-self.size // self.chunk_size)

    def bounds(self, index: int) -> tuple[int, int]:
        """The first and last byte of chunk ``index``, inclusive, the way HTTP counts."""
        start = index * self.chunk_size
        end = min(start + self.chunk_size, self.size) - 1
        return start, max(start, end)

    def length_of(self, index: int) -> int:
        """How many bytes chunk ``index`` holds."""
        start, end = self.bounds(index)
        return 0 if self.size <= 0 else end - start + 1

    def pending(self) -> list[int]:
        """Every chunk index still to fetch, in order."""
        with self._lock:
            done = set(self.done)
        return [index for index in range(self.chunk_count) if index not in done]

    def bytes_done(self) -> int:
        """How many bytes are already on disk, from the record rather than from the file."""
        with self._lock:
            done = set(self.done)
        return sum(self.length_of(index) for index in done)

    def is_complete(self) -> bool:
        """Whether every chunk has arrived."""
        return not self.pending()

    def mark_done(self, index: int) -> None:
        """Record chunk ``index`` as written, and flush the record to disk.

        The write happens under the same lock that adds the index, not after it. Sixteen
        workers finishing chunks at once would otherwise race on the one temporary file
        the record is written through, and on Windows the loser of that race gets a
        sharing violation rather than a merged result -- a download failing at chunk four
        hundred because two threads tidied up at the same moment. The write is a hundred
        bytes; serialising it costs nothing worth measuring.
        """
        with self._lock:
            self.done.add(index)
            self._write_state(sorted(self.done))

    def allocate(self) -> None:
        """Make sure the part file exists at its final size, so a worker can seek into it.

        Raises:
            DownloadError: If the file cannot be created or sized, which on a volume with
                no room left is what a full disk looks like at this layer.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with self.path.open("wb") as handle:
                    handle.truncate(self.size)
                return
            if self.path.stat().st_size != self.size:
                with self.path.open("r+b") as handle:
                    handle.truncate(self.size)
        except OSError as exc:
            raise DownloadError(
                _("could not create %(path)s: %(error)s") % {"path": self.path, "error": exc},
                hint=_("Check that the download directory is writable and has room."),
            ) from exc

    def discard(self) -> None:
        """Delete the part file and its record, so the next attempt starts from nothing."""
        with self._lock:
            self.done.clear()
        for path in (self.path, self.state_path):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                # A file another process is holding open is a problem for the caller's
                # next step, which will fail with a message about that step. Failing here
                # would replace it with a less useful one about tidying up.
                continue

    def finish(self) -> None:
        """Move the finished file into place and delete the record.

        Raises:
            DownloadError: If the move fails.
        """
        try:
            os.replace(self.path, self.target)
        except OSError as exc:
            raise DownloadError(
                _("could not move %(source)s into place: %(error)s")
                % {"source": self.path, "error": exc},
                hint=_("Check that the download directory is writable."),
            ) from exc
        # The bytes are in place and verified; a leftover sidecar is untidy, not wrong.
        with suppress(OSError):
            self.state_path.unlink()

    def _write_state(self, done: list[int]) -> None:
        document = {
            "schema_version": STATE_SCHEMA_VERSION,
            "url": self.url,
            "size": self.size,
            "chunk_size": self.chunk_size,
            "done": done,
        }
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(document), encoding="utf-8")
            os.replace(temporary, self.state_path)
        except OSError as exc:
            raise DownloadError(
                _("could not record progress in %(path)s: %(error)s")
                % {"path": self.state_path, "error": exc},
                hint=_("Check that the download directory is writable and has room."),
            ) from exc


def open_part(
    target: Path, url: str, size: int, *, chunk_size: int = DEFAULT_CHUNK_BYTES
) -> PartFile:
    """Open the part file for ``target``, resuming from its record when one still fits.

    The record is kept only when it agrees with everything this call was told: the same
    URL, the same total size, the same chunk size, and a part file that is actually there
    at that size. Anything else and the run starts from nothing, which costs bandwidth and
    is the only answer that cannot produce a file made of two different downloads.

    Args:
        target: Where the finished file will go.
        url: What is being fetched.
        size: The file's final size, from the catalog.
        chunk_size: How many bytes one request asks for.

    Returns:
        A :class:`PartFile` carrying whichever chunks were already recorded.
    """
    part = PartFile(target=target, url=url, size=size, chunk_size=chunk_size)
    recorded = _read_state(part.state_path)
    if recorded is None:
        return part
    if (
        recorded.get("schema_version") != STATE_SCHEMA_VERSION
        or recorded.get("url") != url
        or recorded.get("size") != size
        or recorded.get("chunk_size") != chunk_size
    ):
        return part
    if not part.path.is_file() or part.path.stat().st_size != size:
        return part
    done = recorded.get("done")
    if not isinstance(done, list):
        return part
    count = part.chunk_count
    part.done = {index for index in done if isinstance(index, int) and 0 <= index < count}
    return part


def _read_state(path: Path) -> dict[str, object] | None:
    """The recorded state at ``path``, or ``None`` when there is none worth reading.

    A sidecar that cannot be read is treated as absent rather than as a failure: the file
    it describes is about to be re-fetched, which is slow but always correct, and asking a
    user to delete a file by hand before their tool will work again is never the answer.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        document = json.loads(text)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None
