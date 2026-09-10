# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Read file listings from the Hugging Face Hub without downloading any weights.

The catalog refresh step needs the exact size and checksum of each GGUF file in a
model's repository so it can record them in the seed catalog. This module reads that
information from the public Hugging Face models API and matches the files that make
up one named quantization.
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, get_args

import httpx

from llamafit import __version__
from llamafit.errors import NetworkError
from llamafit.i18n import _
from llamafit.models.catalog import ExtraRole

_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_SHARD_SUFFIX = r"-\d{5}-of-\d{5}\.gguf$"

_EXTRA_ROLE_RE = re.compile(rf"^(?:{'|'.join(get_args(ExtraRole))})[-_.]", re.IGNORECASE)
"""A file name that opens with an auxiliary role, for example ``mmproj-Q8_0.gguf``.

The roles are read from :data:`~llamafit.models.catalog.ExtraRole` rather than listed
again here, so a role added to the catalog model is known here on the same day.
"""


def _quant_pattern(quant_name: str) -> re.Pattern[str]:
    """Match a quant name only where a GGUF filename can actually carry one."""
    name = re.escape(quant_name)
    return re.compile(rf"(?:^|[-_./]){name}(?:\.gguf$|{_SHARD_SUFFIX})", re.IGNORECASE)


def _could_be_a_quant(file: RepoFile, extra_files: Collection[str]) -> bool:
    """Whether a repository file could belong to a quant at all, before any name is matched.

    Three kinds never can, and each would otherwise be counted into a quant's size: a
    file that is not a GGUF; one the catalog already declares as an extra, claimed
    outright by its name; and one whose name opens with an auxiliary role, since
    ``mmproj-Q8_0.gguf`` names a quantisation without being one, and a projector
    counted as a quant inflates that model by the projector's whole size.
    """
    if not file.path.endswith(".gguf"):
        return False
    # Named rather than thrown away as `_`: this module imports the translator
    # under that name, and a throwaway inside a function rebinds it to a string
    # that is not callable at the next translated call, silently until then.
    _head, _slash, name = file.path.rpartition("/")
    if file.path in extra_files or name in extra_files:
        return False
    return _EXTRA_ROLE_RE.match(name) is None


def _is_whole_set(ordered: Sequence[tuple[int, RepoFile]], total: int) -> bool:
    """Whether an ordered shard group holds exactly one file for each declared index.

    Short and doubled are both refused, for the same reason: neither is one whole set.
    A missing shard cannot be sized, and two files claiming one index are two
    publications of the quant — an ``imat`` path and a ``main`` path, say — which are
    usually genuinely different files. Summing them would record a size half again too
    large; choosing between them would be guessing which one a curator meant.
    """
    return [index for index, _file in ordered] == list(range(1, total + 1))


def _select_shard_set(files: Sequence[RepoFile]) -> list[RepoFile]:
    """Pick the largest whole shard set among ``files``, or all of them sorted.

    A split file's name declares how many shards there are, ``-00001-of-00004``, so a
    set that is short — a repository mid-upload, a partial mirror — or one published
    twice over is recognised here from the listing alone, without asking the network
    anything, and left out rather than handed on as if it were whole. The caller then
    sees no files for that quant, which it already treats as a warning, instead of
    fetching headers for a set that could never add up.

    Each candidate group is ordered before it is judged, and the sort is keyed on the
    index and then the path rather than on the pair: :class:`RepoFile` has no ordering,
    so a plain tuple sort would fall through to comparing the dataclasses and raise the
    moment two files shared an index — which is exactly the case being judged.

    Args:
        files: Files already known to belong to one quant.

    Unsplit files are held to the same standard. One is a quant; two are two
    publications of one quant name, an ``imat`` copy and a ``main`` copy, and picking
    between them would be guessing which a curator meant while adding them together
    would record about twice the real size. Two is therefore nothing, the same as a
    doubled shard set.

    Returns:
        When any file is part of a ``-NNNNN-of-MMMMM.gguf`` shard set, the largest such
        set that is whole, sorted by shard index, and nothing at all when none is;
        otherwise the single file that matched, and nothing at all when more than one
        did.
    """
    shard_groups: dict[int, list[tuple[int, RepoFile]]] = {}
    for file in files:
        _head, _slash, filename = file.path.rpartition("/")
        match = _SHARD_RE.search(filename)
        if match:
            index, total = int(match.group(1)), int(match.group(2))
            shard_groups.setdefault(total, []).append((index, file))

    if shard_groups:
        for total in sorted(shard_groups, reverse=True):
            ordered = sorted(shard_groups[total], key=lambda pair: (pair[0], pair[1].path))
            if _is_whole_set(ordered, total):
                return [file for _index, file in ordered]
        return []

    if len(files) > 1:
        return []
    return list(files)


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
                    _("Hugging Face returned %(status)d while listing files for %(repo)s.")
                    % {"status": response.status_code, "repo": repo},
                    hint=_("Check that the repository id is correct and public."),
                )
            data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise NetworkError(
                _("Could not read the file listing for %(repo)s from Hugging Face.")
                % {"repo": repo},
                hint=_("Check your network connection and try again."),
            ) from exc

        siblings = data.get("siblings") if isinstance(data, dict) else None
        if not isinstance(siblings, list):
            raise NetworkError(
                _("Hugging Face's response for %(repo)s did not include a file listing.")
                % {"repo": repo},
                hint=_("Check that the repository id is correct and public."),
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

    A file whose name opens with an auxiliary role is never a candidate, because
    ``mmproj-Q8_0.gguf`` carries a quantisation in its name without being that quant's
    weights. Pass a repository's declared extras to `assign_files_to_quants` to claim
    the ones that are named some other way.

    Note that this alone cannot tell apart a shorter quant name that is a suffix of a
    longer one (``Q4_K_XL`` inside ``UD-Q4_K_XL``); use `assign_files_to_quants` when a
    repository's full set of quant names is known.

    Args:
        files: Every file in the repository.
        quant_name: The quantization to match, for example ``Q4_K_M`` or ``UD-Q4_K_XL``.

    Returns:
        The matching files, reduced to one whole quant: the largest complete shard set,
        or the single file that matched. Nothing at all when no set is whole or when
        more than one file matched, since a quant published twice over is left out
        rather than guessed at or added up.
    """
    pattern = _quant_pattern(quant_name)
    candidates = [
        file for file in files if _could_be_a_quant(file, ()) and pattern.search(file.path)
    ]
    return _select_shard_set(candidates)


def assign_files_to_quants(
    files: Sequence[RepoFile],
    quant_names: Sequence[str],
    extra_files: Collection[str] = (),
) -> dict[str, list[RepoFile]]:
    """Group a repository's files by quant, giving each file to the longest name that matches.

    A repository that publishes both ``Q4_K_XL`` and ``UD-Q4_K_XL`` would otherwise have the
    second one's files counted under the first, because the first is a suffix of the second.
    Trying the longest quant names first, and stopping at the first match, guarantees each
    file is assigned to at most one quant.

    A file the catalog declares as an extra is claimed outright and offered to no quant,
    however it happens to be named, and so is any file whose name opens with an auxiliary
    role. Without that, a projector named for a quantisation is counted as part of that
    quant, and the size recorded for the model is the projector's larger too.

    Args:
        files: Every file in the repository.
        quant_names: Every quant name published by the repository.
        extra_files: The file names this source declares as extras, by path or basename.

    Returns:
        One list per name in ``quant_names`` (in that order), each reduced the same way
        `match_quant_files` reduces a single quant's files.
    """
    patterns = {name: _quant_pattern(name) for name in quant_names}
    ordered_names = sorted(quant_names, key=len, reverse=True)
    buckets: dict[str, list[RepoFile]] = {name: [] for name in quant_names}
    for file in files:
        if not _could_be_a_quant(file, extra_files):
            continue
        for name in ordered_names:
            if patterns[name].search(file.path):
                buckets[name].append(file)
                break
    return {name: _select_shard_set(bucket) for name, bucket in buckets.items()}
