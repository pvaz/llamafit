# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The HTTP surface a download needs, behind a protocol so a test never reaches the network.

One method, ``open``, which asks for a byte range and hands back a status, the headers and
a stream of pieces. That is deliberately less than an HTTP client: the engine above it has
no business knowing about connection pools, and a test has no business standing one up.

Redirects are followed on the request itself and never left to the client's own setting,
which is the same rule :mod:`llamafit.gguf.source` follows and for the same reason: Hugging
Face answers a weights URL with a redirect to its content network, and a client injected by
a caller (or by a test) cannot be assumed to follow it. That bug has already cost this
project once, and a mock transport never redirects on its own, so nothing in a green test
suite would have said so.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

import httpx

from llamafit import __version__
from llamafit.errors import NetworkError
from llamafit.i18n import _

STREAM_PIECE_BYTES = 1 << 16
"""How much of a response body is held in memory at a time: 64 KiB.

Small enough that eight workers cost half a megabyte between them, large enough that a
gigabit link is not spending its time in the loop rather than on the wire.
"""

CONNECT_TIMEOUT_SECONDS = 15.0
READ_TIMEOUT_SECONDS = 60.0
"""A stalled read is a failure worth retrying, not something to wait out forever."""


@dataclass
class RangeResponse:
    """What a range request came back with.

    Attributes:
        status: The HTTP status of the final response, after any redirect.
        headers: That response's headers.
        body: The response body, in pieces, to be consumed inside the ``with`` block that
            produced it.
    """

    status: int
    headers: Mapping[str, str]
    body: Iterator[bytes]


class RangeReader(Protocol):
    """Something that can hand over one byte range of one URL."""

    def open(self, url: str, start: int, end: int) -> AbstractContextManager[RangeResponse]:
        """Request ``bytes=start-end`` of ``url``, inclusive at both ends."""
        ...


class HttpRangeReader:
    """A :class:`RangeReader` backed by ``httpx``.

    The client may be injected, which is how the command line reuses one connection pool
    for every file of a split model and how a test hands in a mock transport. An injected
    client is never trusted to follow a redirect on its own.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        piece_bytes: int = STREAM_PIECE_BYTES,
        token: str | None = None,
    ) -> None:
        """Remember the client to use, the streaming piece size and the bearer token.

        Args:
            client: An open client to reuse, or ``None`` to create (and own) one.
            piece_bytes: How much of the body to hold in memory at a time.
            token: A Hugging Face token for a gated repository; read from ``HF_TOKEN``
                when not given, the same way :mod:`llamafit.catalog.hf` reads it.
        """
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(
                follow_redirects=True,
                timeout=httpx.Timeout(
                    READ_TIMEOUT_SECONDS,
                    connect=CONNECT_TIMEOUT_SECONDS,
                ),
            )
        )
        self._piece_bytes = piece_bytes
        self._token = token if token is not None else os.environ.get("HF_TOKEN")

    def close(self) -> None:
        """Close the underlying client, but only when this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpRangeReader:
        """Return this reader for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client on the way out of the ``with`` block."""
        self.close()

    @contextmanager
    def open(self, url: str, start: int, end: int) -> Iterator[RangeResponse]:
        """Stream ``bytes=start-end`` of ``url``.

        Args:
            url: What to fetch.
            start: First byte wanted, counting from zero.
            end: Last byte wanted, inclusive, the way HTTP counts them.

        Yields:
            The response, whose body must be consumed before the block ends.

        Raises:
            NetworkError: The request could not be made, or the connection failed while
                the body was being read. Both are the same thing to a caller that is
                about to retry, and neither should reach a user as an ``httpx`` type.
        """
        headers = {
            "Range": f"bytes={start}-{end}",
            "User-Agent": f"llamafit/{__version__}",
            # A cache that answered from a stale copy would hand back bytes that no longer
            # belong beside the ones already on disk, and the checksum would only say so
            # at the end of a very long download.
            "Accept-Encoding": "identity",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with self._client.stream(
                "GET", url, headers=headers, follow_redirects=True
            ) as response:
                yield RangeResponse(
                    status=response.status_code,
                    headers=response.headers,
                    body=response.iter_bytes(self._piece_bytes),
                )
        except httpx.HTTPError as exc:
            raise NetworkError(
                _("could not fetch %(url)s: %(error)s") % {"url": url, "error": exc},
                hint=_("Check your network connection and run the command again to resume."),
            ) from exc
