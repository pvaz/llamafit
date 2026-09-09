"""Where header bytes come from: a local file, an HTTP range request, or a test.

A GGUF file can be a hundred gigabytes; its header is a few hundred kilobytes at
the front. Every reader here fetches only the ranges the parser asks for, so a
remote header costs one or two requests rather than a download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx

from llamafit.errors import CatalogError, NetworkError


class ByteSource(Protocol):
    """A random-access source of bytes."""

    def read(self, offset: int, length: int) -> bytes:
        """Return exactly ``length`` bytes from ``offset``, or fewer at the end."""
        ...

    def size(self) -> int | None:
        """The total size when it is known."""
        ...


@dataclass
class FakeSource:
    """An in-memory source that records every range it was asked for."""

    data: bytes
    reads: list[tuple[int, int]] = field(default_factory=list)

    def read(self, offset: int, length: int) -> bytes:
        """Return the slice and record the request."""
        self.reads.append((offset, length))
        return self.data[offset : offset + length]

    def size(self) -> int | None:
        """The length of the buffer."""
        return len(self.data)


class LocalSource:
    """Reads byte ranges from a local file, opening it lazily and keeping the handle."""

    def __init__(self, path: Path) -> None:
        """Remember ``path``; nothing is opened until the first read."""
        self.path = path
        self._handle: BinaryIO | None = None

    def read(self, offset: int, length: int) -> bytes:
        """Return up to ``length`` bytes starting at ``offset``.

        Raises:
            CatalogError: If the file cannot be opened or read.
        """
        try:
            handle = self._open()
            handle.seek(offset)
            return handle.read(length)
        except OSError as exc:
            raise CatalogError(
                f"could not read {self.path}: {exc}",
                hint="Check that the file exists and is readable.",
            ) from exc

    def size(self) -> int | None:
        """The file's size in bytes.

        Raises:
            CatalogError: If the file cannot be stat'd.
        """
        try:
            return self.path.stat().st_size
        except OSError as exc:
            raise CatalogError(
                f"could not read {self.path}: {exc}",
                hint="Check that the file exists and is readable.",
            ) from exc

    def _open(self) -> BinaryIO:
        if self._handle is None:
            self._handle = self.path.open("rb")
        return self._handle


class HttpRangeSource:
    """Reads byte ranges of a remote file over HTTP, without downloading it whole.

    Keeps whatever it last fetched so a sequential parser that re-reads nearby
    offsets does not issue a request per field; each miss fetches at least
    ``chunk`` bytes.
    """

    def __init__(self, url: str, client: httpx.Client | None = None, chunk: int = 1 << 20) -> None:
        """Remember the URL, the optional client to reuse, and the fetch chunk size."""
        self.url = url
        self.client = client
        self.chunk = chunk
        self._data = b""
        self._start = 0
        self._total_size: int | None = None

    def read(self, offset: int, length: int) -> bytes:
        """Return up to ``length`` bytes starting at ``offset``, fetching if needed."""
        end = offset + length
        if not (self._start <= offset and end <= self._start + len(self._data)):
            self._fetch(offset, max(length, self.chunk))
        start = offset - self._start
        return self._data[start : start + length]

    def size(self) -> int | None:
        """The remote file's total size, known once a range has been fetched."""
        return self._total_size

    def _fetch(self, offset: int, length: int) -> None:
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        try:
            if self.client is not None:
                response = self.client.get(self.url, headers=headers)
            else:
                response = httpx.get(self.url, headers=headers)
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"could not fetch {self.url}: {exc}",
                hint="Check your network connection and that the URL is reachable.",
            ) from exc
        if response.status_code != 206:
            raise NetworkError(
                f"{self.url} returned HTTP {response.status_code} instead of 206 Partial "
                "Content; the server may not support range requests.",
                hint="Confirm the URL points at a downloadable GGUF file.",
            )
        content_range = response.headers.get("content-range", "")
        total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
        if total.isdigit():
            self._total_size = int(total)
        self._data = response.content
        self._start = offset
