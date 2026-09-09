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
from typing import Any, Protocol

import httpx

from llamafit import __version__
from llamafit.errors import NetworkError

_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")


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
        self._client = client if client is not None else httpx.Client()
        self._token = token if token is not None else os.environ.get("HF_TOKEN")

    def list_files(self, repo: str) -> list[RepoFile]:
        """List every file in ``repo``, with size and checksum where known.

        Raises:
            NetworkError: The request failed, or Hugging Face did not answer with 200.
        """
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        headers = {"User-Agent": f"llamafit/{__version__}"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"Could not reach Hugging Face to list files for {repo}.",
                hint="Check your network connection and try again.",
            ) from exc
        if response.status_code != 200:
            raise NetworkError(
                f"Hugging Face returned {response.status_code} while listing files for {repo}.",
                hint="Check that the repository id is correct and public.",
            )
        data: Any = response.json()
        siblings = data.get("siblings") if isinstance(data, dict) else None
        if not isinstance(siblings, list):
            return []
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
    ``*<quant_name>*-00001-of-000NN.gguf``. A file inside a subdirectory is only
    considered when that subdirectory's name is ``quant_name`` itself, so files that
    live under a different quant's directory are ignored even if their own filename
    happens to contain this quant's name.

    Args:
        files: Every file in the repository.
        quant_name: The quantization to match, for example ``Q4_K_M`` or ``UD-Q4_K_XL``.

    Returns:
        The matching files. When any of them is part of a shard set, only the shards
        of the largest such set are returned, sorted by shard index; otherwise every
        matching file is returned, sorted by path.
    """
    token = re.compile(rf"\b{re.escape(quant_name)}\b")
    candidates: list[RepoFile] = []
    for file in files:
        if not file.path.endswith(".gguf"):
            continue
        directory, _, filename = file.path.rpartition("/")
        if directory and not token.search(directory):
            continue
        if not token.search(filename):
            continue
        candidates.append(file)

    shard_groups: dict[int, list[tuple[int, RepoFile]]] = {}
    for file in candidates:
        _, _, filename = file.path.rpartition("/")
        match = _SHARD_RE.search(filename)
        if match:
            index, total = int(match.group(1)), int(match.group(2))
            shard_groups.setdefault(total, []).append((index, file))

    if shard_groups:
        largest_total = max(shard_groups)
        return [file for _, file in sorted(shard_groups[largest_total])]

    return sorted(candidates, key=lambda file: file.path)
