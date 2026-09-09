"""Cache parsed GGUF headers on disk and expose the one function callers need.

Parsing a header is cheap once the bytes are in hand, but fetching those bytes from a
remote file costs a network round trip; caching the parsed header, keyed by the local
file's size and modification time or the remote file's URL and ETag, avoids repeating
that cost for a file that has not changed.

A split model is several files, and each one is read and cached under its own identity;
nothing is ever stored against the set as a whole. A set therefore cannot collide with
another set, not even one that shares some of its shards, and it costs no extra
requests: a set-wide key would still have to learn every shard's ETag before it could
be computed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import httpx

from llamafit.gguf.facts import derive_facts
from llamafit.gguf.reader import merge_shard_headers, read_header
from llamafit.gguf.source import ByteSource, HttpRangeSource, LocalSource
from llamafit.models.gguf import GgufFacts, GgufHeader

GgufTarget: TypeAlias = Path | str
"""One GGUF file: a local path, or an ``http(s)`` URL read over range requests."""


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


def read_one_header(
    target: GgufTarget,
    *,
    cache: HeaderCache | None = None,
    client: httpx.Client | None = None,
) -> GgufHeader:
    """Read one GGUF file's header from a local path or a remote URL, via ``cache``.

    A ``Path``, or a string with no ``http://`` or ``https://`` scheme, is read from the
    local filesystem; a URL is read over HTTP range requests, without downloading the
    file.

    A local file's cache key is known before any read (its path, size and modification
    time), so an unchanged file can skip re-parsing entirely. A remote file's identity
    is normally only known from a response header, which would force a fetch before a
    cache lookup could mean anything; a cheap ``HEAD`` request (see
    :meth:`~llamafit.gguf.source.HttpRangeSource.head`) learns the ``ETag`` first, at a
    fraction of the cost of a range request, so a cache hit needs only that one small
    request and never fetches a range at all. When ``HEAD`` fails or the server sends
    no ``ETag``, this falls back to fetching and keying by URL alone: an honest cache
    that cannot detect a file replaced at that URL, rather than one that silently
    pretends it can.

    Returns:
        The file's parsed header.
    """
    if isinstance(target, str) and target.startswith(("http://", "https://")):
        http_source = HttpRangeSource(target, client=client)
        head_etag: str | None = None
        if cache is not None:
            head_etag, _ = http_source.head()
            if head_etag is not None:
                cached = cache.get(cache_key_for_url(target, head_etag))
                if cached is not None:
                    return cached
        header = read_header(http_source)
        if cache is not None:
            etag = http_source.etag if http_source.etag is not None else head_etag
            cache.put(cache_key_for_url(target, etag), header)
        return header
    path = Path(target)
    return read_header_cached(LocalSource(path), cache_key_for_path(path), cache)


def read_facts(
    target: GgufTarget | Sequence[GgufTarget],
    *,
    lazy_tensor_names: Sequence[str] = (),
    cache: HeaderCache | None = None,
    client: httpx.Client | None = None,
) -> GgufFacts:
    """Read architecture facts from one GGUF file, or from a split model's shards.

    This is the one function the rest of LlamaFit calls to get a GGUF file's facts.
    ``target`` is either one file — a local ``Path``, or a string that is a URL when it
    has an ``http://`` or ``https://`` scheme and a local path otherwise — or a sequence
    of them, which must be every shard of one split model. Each file is read and cached
    by :func:`read_one_header`, and a set is then merged by
    :func:`~llamafit.gguf.reader.merge_shard_headers` before any facts are derived.

    Passing only the first shard of a split model is the mistake this signature exists
    to prevent: that shard carries the model's whole metadata and, commonly, none of its
    tensors, so the facts would look complete while reporting zero bytes of weights.

    Args:
        target: One GGUF file, or every shard of one split model in any order.
        lazy_tensor_names: Name prefixes of tensors that llama.cpp can stream from disk
            rather than hold in memory; their bytes go to ``bytes_lazy_tables``.
        cache: Where parsed headers are stored between runs, or ``None`` to read every
            time.
        client: An HTTP client to reuse for remote reads, or ``None`` for a new request
            each time.

    Returns:
        The facts derived from the file, or from the union of the shards.

    Raises:
        CatalogError: If no target is given, or if a set of them is not one complete
            split model.
    """
    targets = [target] if isinstance(target, (str, Path)) else list(target)
    headers = [read_one_header(one, cache=cache, client=client) for one in targets]
    return derive_facts(merge_shard_headers(headers), lazy_tensor_names=lazy_tensor_names)
