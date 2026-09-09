"""Cache parsed GGUF headers on disk and expose the one function callers need.

Parsing a header is cheap once the bytes are in hand, but fetching those bytes from a
remote file costs a network round trip; caching the parsed header, keyed by the local
file's size and modification time or the remote file's URL and ETag, avoids repeating
that cost for a file that has not changed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import httpx

from llamafit.gguf.facts import derive_facts
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import ByteSource, HttpRangeSource, LocalSource
from llamafit.models.gguf import GgufFacts, GgufHeader


def cache_key_for_path(path: Path) -> str:
    """A key that changes whenever the local file's size or modification time does."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime).encode("utf-8"))
    return digest.hexdigest()


def cache_key_for_url(url: str, etag: str | None) -> str:
    """A key that changes whenever the remote file's URL or ETag does."""
    digest = hashlib.sha256()
    digest.update(url.encode("utf-8"))
    digest.update((etag or "").encode("utf-8"))
    return digest.hexdigest()


class HeaderCache:
    """Parsed headers stored as one JSON file per key under a directory."""

    def __init__(self, directory: Path) -> None:
        """Remember ``directory``; it is created lazily on the first ``put``."""
        self.directory = directory

    def get(self, key: str) -> GgufHeader | None:
        """Return the cached header for ``key``, or ``None`` on any miss or corruption."""
        try:
            text = (self.directory / f"{key}.json").read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return GgufHeader.model_validate_json(text)
        except ValueError:
            return None

    def put(self, key: str, header: GgufHeader) -> None:
        """Store ``header`` under ``key``, creating the cache directory if needed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(header.model_dump_json(), encoding="utf-8")


def read_header_cached(source: ByteSource, key: str, cache: HeaderCache | None) -> GgufHeader:
    """Read a header from ``cache`` when present, else parse it from ``source`` and store it."""
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    header = read_header(source)
    if cache is not None:
        cache.put(key, header)
    return header


def read_facts(
    target: Path | str,
    *,
    lazy_tensor_names: Sequence[str] = (),
    cache: HeaderCache | None = None,
    client: httpx.Client | None = None,
) -> GgufFacts:
    """Read architecture facts from a local file or a remote URL.

    A ``Path``, or a string with no ``http://`` or ``https://`` scheme, is read from the
    local filesystem; a URL is read over HTTP range requests, without downloading the
    file. This is the one function the rest of LlamaFit calls to get a GGUF file's facts.
    """
    source: ByteSource
    if isinstance(target, str) and target.startswith(("http://", "https://")):
        source = HttpRangeSource(target, client=client)
        key = cache_key_for_url(target, None)
    else:
        path = Path(target)
        source = LocalSource(path)
        key = cache_key_for_path(path)
    header = read_header_cached(source, key, cache)
    return derive_facts(header, lazy_tensor_names=lazy_tensor_names)
