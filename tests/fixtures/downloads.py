"""A fake Hugging Face that behaves badly on request, and a reporter that remembers.

Every download test drives the real :class:`~llamafit.download.transport.HttpRangeReader`
through an ``httpx`` mock transport, so the code under test is the code that ships, right
down to how it follows the redirect Hugging Face issues to its content network. The
transport is the only thing replaced, and nothing here opens a socket.

``FakeHub`` redirects by default, because that redirect is where this project has already
been bitten once and a test that skipped it would prove nothing about the real path.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import httpx

from llamafit.download.transport import HttpRangeReader

CANONICAL = "https://huggingface.co/acme/model-gguf/resolve/main/model-Q4_K_M.gguf"
CDN = "https://cdn-lfs.example.invalid/acme/model-Q4_K_M.gguf"


def blob(size: int, seed: int = 7) -> bytes:
    """Deterministic bytes that differ everywhere, so a misplaced chunk cannot pass."""
    out = bytearray(size)
    value = seed
    for index in range(size):
        value = (value * 1103515245 + 12345) & 0xFFFFFFFF
        out[index] = (value >> 16) & 0xFF
    return bytes(out)


def sha256(data: bytes) -> str:
    """The digest of some bytes, lowercase hexadecimal."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class FakeHub:
    """A server that serves byte ranges, and misbehaves in whichever way a test asked for.

    Attributes:
        data: The file's bytes.
        redirect: Whether the canonical URL answers with a 302 to the content network,
            the way Hugging Face does.
        ranges: Every ``(start, end)`` a range request asked for, in order.
        truncate_after: Send only this many bytes of each range, for a transfer that ends
            early. Applied to the requests whose index is in ``truncate_requests``, or to
            every one when that is empty.
        truncate_requests: Which range requests (counting from zero, excluding the
            one-byte probe) are truncated.
        rate_limit_requests: Which range requests answer 429 instead.
        retry_after: What the 429 puts in its ``Retry-After`` header.
        drop_ranges_after: After this many range requests, answer with the whole file and
            a 200, the way a server that stopped honouring ranges would.
        stop_after: Set this event once this many range requests have been served.
        stop: The event to set.
        fail_status: Answer every range request with this status instead.
    """

    data: bytes
    redirect: bool = True
    ranges: list[tuple[int, int]] = field(default_factory=list)
    truncate_after: int | None = None
    truncate_requests: tuple[int, ...] = ()
    rate_limit_requests: tuple[int, ...] = ()
    retry_after: str | None = None
    drop_ranges_after: int | None = None
    stop_after: int | None = None
    stop: threading.Event | None = None
    fail_status: int | None = None
    forbidden_below: int | None = None
    served: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request the way this hub was configured to."""
        if self.redirect and str(request.url) == CANONICAL:
            return httpx.Response(302, headers={"location": CDN})
        header = request.headers.get("range", "")
        match = re.match(r"bytes=(\d+)-(\d+)", header)
        assert match is not None, f"no range header: {header!r}"
        start, end = int(match.group(1)), min(int(match.group(2)), len(self.data) - 1)
        if end - start == 0 and start == 0:
            return self._probe()
        return self._serve(start, end)

    def _probe(self) -> httpx.Response:
        """The one-byte request the engine uses to learn the file's real size."""
        return httpx.Response(
            206,
            content=self.data[:1],
            headers={"content-range": f"bytes 0-0/{len(self.data)}"},
        )

    def _serve(self, start: int, end: int) -> httpx.Response:
        with self._lock:
            index = self.served
            self.served += 1
            self.ranges.append((start, end))
        if self.forbidden_below is not None and start < self.forbidden_below:
            raise AssertionError(f"asked again for bytes {start}-{end}, which were already here")
        if self.fail_status is not None:
            return httpx.Response(self.fail_status)
        if index in self.rate_limit_requests:
            headers = {"retry-after": self.retry_after} if self.retry_after else {}
            return httpx.Response(429, headers=headers)
        if self.drop_ranges_after is not None and index >= self.drop_ranges_after:
            return httpx.Response(
                200,
                content=self.data,
                headers={"content-length": str(len(self.data))},
            )
        body = self.data[start : end + 1]
        truncated = self.truncate_after is not None and (
            not self.truncate_requests or index in self.truncate_requests
        )
        if truncated:
            body = body[: self.truncate_after]
        return httpx.Response(
            206,
            content=self._stream(body, index),
            headers={"content-range": f"bytes {start}-{end}/{len(self.data)}"},
        )

    def _stream(self, body: bytes, index: int) -> Iterator[bytes]:
        """Hand the body over in pieces, stopping the run partway when asked to."""
        step = max(1, len(body) // 4 or 1)
        sent = 0
        for offset in range(0, len(body), step):
            yield body[offset : offset + step]
            sent += len(body[offset : offset + step])
        if self.stop is not None and self.stop_after is not None and index + 1 >= self.stop_after:
            self.stop.set()

    def client(self) -> httpx.Client:
        """A client wired to this hub, which does not follow redirects on its own.

        Left at the default on purpose: the reader has to ask for redirects on each
        request, which is the bug this project has already paid for once.
        """
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def reader(self) -> HttpRangeReader:
        """The real reader, wired to this hub."""
        return HttpRangeReader(self.client(), piece_bytes=256)


@dataclass
class RecordingProgress:
    """A reporter that remembers everything it was told, for tests to read back."""

    started: list[tuple[str, int, int]] = field(default_factory=list)
    deltas: list[tuple[str, int]] = field(default_factory=list)
    verifying: list[str] = field(default_factory=list)
    at_verify: dict[str, int] = field(default_factory=dict)
    finished: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def file_started(self, name: str, total: int, already: int) -> None:
        """Remember that a file began."""
        with self._lock:
            self.started.append((name, total, already))

    def bytes_received(self, name: str, count: int) -> None:
        """Remember a movement of the bar, forwards or backwards."""
        with self._lock:
            self.deltas.append((name, count))

    def file_verifying(self, name: str, total: int) -> None:
        """Remember that a checksum was computed, and where the bar stood when it began.

        Hashing reports its own progress, so the bar's final position says nothing about
        how much was transferred. The figure worth asserting on is this one.
        """
        with self._lock:
            self.verifying.append(name)
            self.at_verify.setdefault(name, self._net(name))

    def file_finished(self, name: str) -> None:
        """Remember that a file was finished."""
        with self._lock:
            self.finished.append(name)

    def note(self, message: str) -> None:
        """Remember a remark."""
        with self._lock:
            self.notes.append(message)

    def net(self, name: str) -> int:
        """The bar's position for one file, counting the rewinds."""
        with self._lock:
            return self._net(name)

    def _net(self, name: str) -> int:
        return sum(count for who, count in self.deltas if who == name)


def counting_handler(hub: FakeHub) -> Callable[[httpx.Request], httpx.Response]:
    """The hub's handler, for a test that wants to wrap it."""
    return hub.handle
