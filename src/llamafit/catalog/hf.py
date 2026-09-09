"""Read file listings from the Hugging Face Hub without downloading any weights.

The catalog refresh step needs the exact size and checksum of each GGUF file in a
model's repository so it can record them in the seed catalog. This module reads that
information from the public Hugging Face models API and matches the files that make
up one named quantization.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

import httpx

from llamafit import __version__
from llamafit.errors import NetworkError

_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_SHARD_SUFFIX = r"-\d{5}-of-\d{5}\.gguf$"


def _quant_pattern(quant_name: str) -> re.Pattern[str]:
    """Match a quant name only where a GGUF filename can actually carry one."""
    name = re.escape(quant_name)
    return re.compile(rf"(?:^|[-_./]){name}(?:\.gguf$|{_SHARD_SUFFIX})", re.IGNORECASE)


def _select_shard_set(files: Sequence[RepoFile]) -> list[RepoFile]:
    """Pick the shards of the largest shard set among ``files``, or all of them sorted.

    Args:
        files: Files already known to belong to one quant.

    Returns:
        When any file is part of a ``-NNNNN-of-MMMMM.gguf`` shard set, only the shards
        of the largest such set, sorted by shard index; otherwise every file, sorted
        by path.
    """
    shard_groups: dict[int, list[tuple[int, RepoFile]]] = {}
    for file in files:
        _, _, filename = file.path.rpartition("/")
        match = _SHARD_RE.search(filename)
        if match:
            index, total = int(match.group(1)), int(match.group(2))
            shard_groups.setdefault(total, []).append((index, file))

    if shard_groups:
        largest_total = max(shard_groups)
        return [file for _, file in sorted(shard_groups[largest_total])]

    return sorted(files, key=lambda file: file.path)


@dataclass(frozen=True)
class RepoFile:
    """One file in a Hugging Face repository, as reported by the models API.

    Args:
        path: The file's path within the repository (``rfilename``).
        size: The file's size in bytes, when the API reports one.
        sha256: The file's LFS checksum, when the file is stored via Git LFS.
    """

    path: str
    size: int | None
    sha256: str | None


class HfClient(Protocol):
    """Minimal Hugging Face metadata surface the catalog needs."""

    def list_files(self, repo: str) -> list[RepoFile]:
        """List every file in ``repo``, with size and checksum where known."""
        ...

    def file_url(self, repo: str, path: str) -> str:
        """Build the direct download URL for ``path`` inside ``repo``."""
        ...


class HttpHfClient:
    """Real Hugging Face metadata client, backed by the public models API."""

    def __init__(self, client: httpx.Client | None = None, token: str | None = None) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=True)
        self._token = token if token is not None else os.environ.get("HF_TOKEN")

    def close(self) -> None:
        """Close the underlying HTTP client, but only when this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpHfClient:
        """Return this client for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client on the way out of the ``with`` block."""
        self.close()

    def list_files(self, repo: str) -> list[RepoFile]:
        """List every file in ``repo``, with size and checksum where known.

        Raises:
            NetworkError: The request failed, the response was not 200, its body was
                not JSON, or its JSON did not include a file listing.
        """
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        headers = {"User-Agent": f"llamafit/{__version__}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            # An injected client is never assumed to redirect on its own.
            response = self._client.get(url, headers=headers, follow_redirects=True)
            if response.status_code != 200:
                raise NetworkError(
                    f"Hugging Face returned {response.status_code} while listing files for {repo}.",
                    hint="Check that the repository id is correct and public.",
                )
            data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise NetworkError(
                f"Could not read the file listing for {repo} from Hugging Face.",
                hint="Check your network connection and try again.",
            ) from exc

        siblings = data.get("siblings") if isinstance(data, dict) else None
        if not isinstance(siblings, list):
            raise NetworkError(
                f"Hugging Face's response for {repo} did not include a file listing.",
                hint="Check that the repository id is correct and public.",
            )

        files: list[RepoFile] = []
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            path = sibling.get("rfilename")
            if not isinstance(path, str):
                continue
            size = sibling.get("size")
            lfs = sibling.get("lfs")
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
            files.append(
                RepoFile(
                    path=path,
                    size=size if isinstance(size, int) else None,
                    sha256=sha256 if isinstance(sha256, str) else None,
                )
            )
        return files

    def file_url(self, repo: str, path: str) -> str:
        """Build the direct download URL for ``path`` inside ``repo``."""
        return f"https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass
class FakeHfClient:
    """Canned file listings keyed by repository id, for tests."""

    files: Mapping[str, list[RepoFile]]

    def list_files(self, repo: str) -> list[RepoFile]:
        """Return the canned listing for ``repo``, or an empty list."""
        return list(self.files.get(repo, []))

    def file_url(self, repo: str, path: str) -> str:
        """Build the direct download URL for ``path`` inside ``repo``."""
        return f"https://huggingface.co/{repo}/resolve/main/{path}"


def match_quant_files(files: Sequence[RepoFile], quant_name: str) -> list[RepoFile]:
    """Return the files that make up one quantization, sorted by name.

    A quant is either a single file named ``*<quant_name>*.gguf`` or a shard set named
    ``*<quant_name>*-NNNNN-of-MMMMM.gguf``. ``quant_name`` is matched only where a GGUF
    filename can actually carry one: preceded by the start of the basename or by one of
    ``-_./``, and followed by ``.gguf`` or a shard suffix, so a longer quant name that
    merely starts with ``quant_name`` (or a directory of one) never matches.

    Note that this alone cannot tell apart a shorter quant name that is a suffix of a
    longer one (``Q4_K_XL`` inside ``UD-Q4_K_XL``); use `assign_files_to_quants` when a
    repository's full set of quant names is known.

    Args:
        files: Every file in the repository.
        quant_name: The quantization to match, for example ``Q4_K_M`` or ``UD-Q4_K_XL``.

    Returns:
        The matching files. When any of them is part of a shard set, only the shards
        of the largest such set are returned, sorted by shard index; otherwise every
        matching file is returned, sorted by path.
    """
    pattern = _quant_pattern(quant_name)
    candidates = [
        file for file in files if file.path.endswith(".gguf") and pattern.search(file.path)
    ]
    return _select_shard_set(candidates)


def assign_files_to_quants(
    files: Sequence[RepoFile], quant_names: Sequence[str]
) -> dict[str, list[RepoFile]]:
    """Group a repository's files by quant, giving each file to the longest name that matches.

    A repository that publishes both ``Q4_K_XL`` and ``UD-Q4_K_XL`` would otherwise have the
    second one's files counted under the first, because the first is a suffix of the second.
    Trying the longest quant names first, and stopping at the first match, guarantees each
    file is assigned to at most one quant.

    Args:
        files: Every file in the repository.
        quant_names: Every quant name published by the repository.

    Returns:
        One list per name in ``quant_names`` (in that order), each reduced the same way
        `match_quant_files` reduces a single quant's files.
    """
    patterns = {name: _quant_pattern(name) for name in quant_names}
    ordered_names = sorted(quant_names, key=len, reverse=True)
    buckets: dict[str, list[RepoFile]] = {name: [] for name in quant_names}
    for file in files:
        if not file.path.endswith(".gguf"):
            continue
        for name in ordered_names:
            if patterns[name].search(file.path):
                buckets[name].append(file)
                break
    return {name: _select_shard_set(bucket) for name, bucket in buckets.items()}
