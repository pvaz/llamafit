"""Where header bytes come from: a local file, an HTTP range request, or a test.

A GGUF file can be a hundred gigabytes; its header is a few hundred kilobytes at
the front. Every reader here fetches only the ranges the parser asks for, so a
remote header costs one or two requests rather than a download.

A mock transport never redirects on its own, so a bug in how redirects are
followed here can hide behind a fully green test suite; that is exactly why the
remote path is checked against the real Hugging Face API before every release.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx

from llamafit.errors import CatalogError, NetworkError
from llamafit.i18n import _


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
                _("could not read %(path)s: %(error)s") % {"path": self.path, "error": exc},
                hint=_("Check that the file exists and is readable."),
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
                _("could not read %(path)s: %(error)s") % {"path": self.path, "error": exc},
                hint=_("Check that the file exists and is readable."),
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
        self._etag: str | None = None

    @contextmanager
    def _http_client(self) -> Iterator[httpx.Client]:
        """Yield a client to send one request with.

        The injected client is used as given, but never trusted to redirect on its
        own: every caller of this passes ``follow_redirects=True`` on the request
        itself. When no client was injected, a fresh one is created here with
        ``follow_redirects=True`` and closed once the request completes.
        """
        if self.client is not None:
            yield self.client
        else:
            with httpx.Client(follow_redirects=True) as client:
                yield client

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

    @property
    def etag(self) -> str | None:
        """The remote file's ``ETag``, captured from its first response, if it sent one.

        Two different files served from the same URL at different times generally
        carry different ETags; a cache key built from this (see
        :func:`llamafit.gguf.cache.cache_key_for_url`) changes along with the file,
        instead of colliding on the URL alone.
        """
        return self._etag

    def head(self) -> tuple[str | None, int | None]:
        """Learn the ``ETag`` and ``Content-Length`` with a ``HEAD`` request, no body.

        A cache keyed by :func:`llamafit.gguf.cache.cache_key_for_url` can check for a
        hit before paying for a range request, as long as it knows the current ETag;
        this is how it learns it for a fraction of the cost of fetching a header.
        Redirects (for example to a content-delivery network) are followed, and the
        ``ETag`` is read from the final response.

        Never raises: a network failure, a non-2xx status (some servers reject
        ``HEAD`` outright), or a response with no ``ETag`` header are all reported as
        ``None`` so the caller can fall back to a range request, which is the correct
        response either way. A server that never sends an ``ETag`` at all means this
        never returns one, so a cache built on it can key only by URL and will not
        detect a file replaced at the same URL — the honest trade for a host that
        gives us nothing better.

        Returns:
            The ``ETag`` and ``Content-Length``, each ``None`` when not learned.
        """
        try:
            with self._http_client() as client:
                response = client.head(self.url, follow_redirects=True)
        except httpx.HTTPError:
            return None, None
        if response.status_code >= 400:
            return None, None
        etag = response.headers.get("etag")
        if etag and self._etag is None:
            self._etag = etag
        content_length = response.headers.get("content-length")
        size = (
            int(content_length) if content_length is not None and content_length.isdigit() else None
        )
        if size is not None and self._total_size is None:
            self._total_size = size
        return etag, size

    def _fetch(self, offset: int, length: int) -> None:
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        try:
            with self._http_client() as client:
                response = client.get(self.url, headers=headers, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise NetworkError(
                _("could not fetch %(url)s: %(error)s") % {"url": self.url, "error": exc},
                hint=_("Check your network connection and that the URL is reachable."),
            ) from exc
        if response.status_code != 206:
            # A redirect (for example to a content-delivery network) that ends in a
            # full 200 response instead of an honoured range is not usable: reading
            # it as if it were the header would silently pull in the whole file.
            raise NetworkError(
                _(
                    "%(url)s returned HTTP %(status)d instead of 206 Partial Content; "
                    "the server may not support range requests."
                )
                % {"url": self.url, "status": response.status_code},
                hint=_("Confirm the URL points at a downloadable GGUF file."),
            )
        content_range = response.headers.get("content-range", "")
        total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
        if total.isdigit():
            self._total_size = int(total)
        if self._etag is None:
            etag = response.headers.get("etag")
            if etag:
                self._etag = etag
        self._data = response.content
        self._start = offset
