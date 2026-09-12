# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What the llama.cpp project publishes, and which of it belongs on this machine.

The llama.cpp repository publishes one GitHub release per build, holding a couple of
dozen archives: one per operating system, architecture and compiled backend. Choosing
between them is the whole job of this module, and it is done from the asset names
alone, which is why it can be tested against a recorded listing without a network.

Names are read as *tokens*, not as substrings: ``llama-b6100-bin-win-cuda-12.4-x64.zip``
is ``llama, b6100, bin, win, cuda, 12, 4, x64, zip``. A substring match would find
``cu`` inside ``vulkan`` and ``x64`` inside a future ``x640``; a token match cannot, and
it survives the punctuation drifting from ``cu12.4`` to ``cuda-12.4``, which it has.

Two facts about the published matrix are worth knowing before reading the code:

* The plain macOS arm64 archive *is* the Metal build. Metal is compiled in and there is
  no separate asset for it, so asking for ``metal`` selects the archive with no backend
  token at all — the same rule that selects a CPU build.
* The Windows CUDA archive does not contain the CUDA runtime libraries. They are
  published beside it as ``cudart-llama-bin-win-cuda-<ver>-x64.zip``, and a CUDA install
  without them starts and immediately fails to load. :func:`companion_assets` finds it,
  and the installer unpacks it into the same directory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from llamafit import __version__
from llamafit.errors import NetworkError
from llamafit.i18n import _
from llamafit.models.host import Arch, Backend, OsName, Vendor

REPO = "ggml-org/llama.cpp"
"""The repository whose releases are installed. Not configurable: this is llama.cpp."""

API_ROOT = "https://api.github.com"

_LIST_PAGE = 30
"""How many releases to list when looking for the newest build.

llama.cpp publishes several builds a day and every one of them is a prerelease, so the
first page is always dozens of builds deep. Thirty is plenty and is one request.
"""

CUDA_DRIVER_FLOOR: dict[int, float] = {11: 450.0, 12: 525.0, 13: 580.0}
"""The lowest NVIDIA driver each CUDA major will load on, from NVIDIA's own table.

A release carries CUDA archives for two or three majors at once — 12.4 and 13.3 as this
is written — and the newest is not automatically the right one: it will not start at all
on a driver older than its floor.
"""

ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")
"""What an asset has to end with to be an archive this installer can unpack."""

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")

_OS_TOKENS: dict[OsName, tuple[str, ...]] = {
    "windows": ("win", "windows"),
    "macos": ("macos", "osx", "darwin"),
    "linux": ("ubuntu", "linux", "debian"),
}

_ARCH_TOKENS: dict[Arch, tuple[str, ...]] = {
    "x86_64": ("x64", "amd64"),
    "arm64": ("arm64", "aarch64"),
    "other": (),
}

_BACKEND_TOKENS: dict[Backend, tuple[str, ...]] = {
    "cuda": ("cuda", "cu11", "cu12", "cu13"),
    "hip": ("hip", "rocm"),
    "vulkan": ("vulkan",),
    "sycl": ("sycl",),
    "metal": (),
    "cpu": (),
}
"""Tokens that mark a build as carrying a backend.

``metal`` and ``cpu`` have none on purpose: both are selected by the *absence* of every
accelerator token, because llama.cpp compiles Metal into the plain macOS arm64 build.
"""

ACCELERATOR_TOKENS = frozenset(
    {
        "cuda",
        "cu11",
        "cu12",
        "cu13",
        "hip",
        "rocm",
        "radeon",
        "vulkan",
        "sycl",
        "musa",
        "cann",
        "opencl",
        "kompute",
        "adreno",
        "openvino",
        "webgpu",
        "zdnn",
    }
)
"""Every token that means "this build talks to an accelerator", for the CPU rule.

The list has to be complete rather than merely correct: a CPU build is chosen by the
*absence* of every one of these, so a backend missing from here is a build that gets
handed to somebody who asked for CPU. ``openvino`` is the cautionary tale — the Linux
OpenVINO archive is named ``...-bin-ubuntu-openvino-2026.3.1-x64.tar.gz``, with no other
word marking it, and while the token was missing it sorted ahead of the plain
``...-bin-ubuntu-x64.tar.gz`` on the alphabet alone.
"""

_CPU_VARIANT_RANK = {"cpu": 0, "avx2": 1, "avx": 2, "openblas": 3, "avx512": 4, "noavx": 5}
"""Preference among several CPU builds of one platform, best first.

``cpu`` is the modern archive that dispatches on the instruction sets it finds at run
time, so it is always right. Where only the older split archives exist, ``avx2`` is the
safe default: ``avx512`` is faster on the processors that have it and will not start on
the ones that do not, and choosing a build that cannot run is worse than leaving speed
on the table.
"""

_EXCLUDED_TOKENS = frozenset({"cudart", "xcframework", "framework", "source"})
"""Assets that are published beside the runtime archives but are not one."""


def tokens(name: str) -> list[str]:
    """Split an asset name into lowercase alphanumeric tokens."""
    return [token for token in _TOKEN_SPLIT.split(name.lower()) if token]


@dataclass(frozen=True)
class ReleaseAsset:
    """One downloadable file attached to a release.

    Args:
        name: The file name, which is what every selection rule reads.
        size: Size in bytes, as the API reports it.
        url: Direct download URL.
        sha256: The checksum GitHub publishes for the asset, when it publishes one.
            ``None`` means nothing about this file can be verified before it is
            unpacked, which the installer refuses to do quietly.
    """

    name: str
    size: int
    url: str
    sha256: str | None = None


@dataclass(frozen=True)
class Release:
    """One published llama.cpp build.

    Args:
        tag: The release tag, ``b6100``.
        assets: Every file attached to it.
        url: The release page, for a person who wants to read the notes.
        published_at: When it was published, as the API's ISO-8601 string.
        commit: The commit the tag points at, when it could be learned. It goes into
            ``VERSION.txt`` so that ``llamafit doctor`` reports the same commit for a
            build installed here as it does for one the user compiled themselves.
    """

    tag: str
    assets: tuple[ReleaseAsset, ...] = ()
    url: str | None = None
    published_at: str | None = None
    commit: str | None = None

    @property
    def build(self) -> int | None:
        """The build number in the tag, ``6100`` for ``b6100``, or ``None``."""
        return parse_build(self.tag)

    def asset(self, name: str) -> ReleaseAsset | None:
        """The asset with exactly this name, or ``None``."""
        for candidate in self.assets:
            if candidate.name == name:
                return candidate
        return None


def parse_build(tag: str) -> int | None:
    """Read the build number out of a release tag such as ``b10892``.

    The whole tag has to be the build, so that the repository's other tags — it also
    carries semantic-version ones such as ``v0.4.0`` — come back as ``None`` and are
    left out of :meth:`HttpReleaseClient.latest`.
    """
    match = re.fullmatch(r"b?(\d{3,})", tag.strip())
    return int(match.group(1)) if match else None


class ReleaseClient(Protocol):
    """The two questions the installer asks GitHub."""

    def latest(self) -> Release:
        """The most recent published release."""
        ...

    def by_tag(self, tag: str) -> Release:
        """The release with this tag."""
        ...


def _rate_limited(response: httpx.Response) -> bool:
    """Whether a refusal is GitHub's hourly limit rather than a missing release."""
    if response.status_code not in (403, 429):
        return False
    remaining = response.headers.get("x-ratelimit-remaining")
    return remaining == "0" or response.status_code == 429


class HttpReleaseClient:
    """Reads releases from the GitHub API.

    A ``GITHUB_TOKEN`` in the environment is sent when there is one. It is not required,
    and nothing here needs any permission beyond reading a public repository; it exists
    because the anonymous hourly limit is sixty requests per address, which a shared
    address can exhaust without the user doing anything wrong.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        repo: str = REPO,
        token: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 15.0,
    ) -> None:
        """Remember the client to reuse, the repository, and the token to send."""
        environ = os.environ if env is None else env
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=True)
        self._repo = repo
        self._token = token if token is not None else environ.get("GITHUB_TOKEN")
        self._timeout = timeout

    def close(self) -> None:
        """Close the underlying HTTP client, but only when this instance created it."""
        if self._owns_client:
            self._client.close()

    def latest(self) -> Release:
        """The newest published build.

        Deliberately *not* ``/releases/latest``. llama.cpp publishes a release per build,
        every one of them marked a prerelease, and GitHub's "latest" endpoint skips
        prereleases: it answers with the repository's semantic-version tag, whose only
        attachment is a text file naming the current nightly. An installer built on that
        endpoint downloads nothing and cannot say why. So the releases are listed and the
        highest ``bNNNN`` tag among them is taken, which is the build everyone means.

        A release carrying no assets is passed over. llama.cpp tags several builds a day
        and the archives are uploaded after the tag, so for a few minutes the newest
        release is a tag with nothing on it. Taking it produced "release b10931 publishes
        nothing for windows x86_64", which is true of that tag and reads as a statement
        about the platform; the release an hour older had all twenty-seven archives. A tag
        with no files is not a build anybody can install, and finding the build is what
        this method is for.

        Raises:
            NetworkError: The request failed, was refused, or the listing held no build.
        """
        data = self._get(f"/repos/{self._repo}/releases?per_page={_LIST_PAGE}")
        entries = data if isinstance(data, list) else []
        builds = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("tag_name"), str)
            and parse_build(entry["tag_name"]) is not None
            and entry.get("assets")
        ]
        if not builds:
            raise NetworkError(
                _("the most recent %(count)d llama.cpp releases hold no build.")
                % {"count": _LIST_PAGE},
                hint=_("Name one with `llamafit install llama.cpp --tag ...`."),
            )
        newest = max(builds, key=lambda entry: parse_build(entry["tag_name"]) or 0)
        return self._read_release(newest)

    def by_tag(self, tag: str) -> Release:
        """The release with this tag.

        Raises:
            NetworkError: The request failed, was refused, or the tag does not exist.
        """
        return self._release(f"/repos/{self._repo}/releases/tags/{tag}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"llamafit/{__version__}",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, path: str) -> Any:
        url = f"{API_ROOT}{path}"
        try:
            response = self._client.get(
                url, headers=self._headers(), timeout=self._timeout, follow_redirects=True
            )
        except httpx.HTTPError as exc:
            raise NetworkError(
                _("could not reach GitHub at %(url)s: %(error)s") % {"url": url, "error": exc},
                hint=_("Check your network connection and try again."),
            ) from exc
        if _rate_limited(response):
            raise NetworkError(
                _("GitHub refused the request for %(url)s: the rate limit is exhausted.")
                % {"url": url},
                hint=_(
                    "Wait for the limit to reset, or set GITHUB_TOKEN to a personal "
                    "access token with no scopes."
                ),
            )
        if response.status_code == 404:
            raise NetworkError(
                _("GitHub has no release at %(url)s.") % {"url": url},
                hint=_("Check the tag with `llamafit install llama.cpp --tag ...`."),
            )
        if response.status_code != 200:
            raise NetworkError(
                _("GitHub returned %(status)d for %(url)s.")
                % {"status": response.status_code, "url": url},
                hint=_("Check your network connection and try again."),
            )
        try:
            return response.json()
        except ValueError as exc:
            raise NetworkError(
                _("GitHub's answer for %(url)s was not JSON.") % {"url": url},
                hint=_("Check your network connection and try again."),
            ) from exc

    def _release(self, path: str) -> Release:
        data = self._get(path)
        if not isinstance(data, dict):
            raise NetworkError(
                _("GitHub's answer did not describe a release."),
                hint=_("Check your network connection and try again."),
            )
        return self._read_release(data)

    def _read_release(self, data: Mapping[str, Any]) -> Release:
        tag = data.get("tag_name")
        if not isinstance(tag, str) or not tag:
            raise NetworkError(
                _("GitHub's answer did not describe a release."),
                hint=_("Check your network connection and try again."),
            )
        published = data.get("published_at")
        return Release(
            tag=tag,
            assets=tuple(_read_assets(data.get("assets"))),
            url=data.get("html_url") if isinstance(data.get("html_url"), str) else None,
            published_at=published if isinstance(published, str) else None,
            commit=self._commit_for(tag),
        )

    def _commit_for(self, tag: str) -> str | None:
        """The commit a tag points at, or ``None`` when it cannot be learned.

        Never raises. The commit is a nicety — it goes into ``VERSION.txt`` so a
        detected build reports the same commit a self-compiled one does — and an
        install must not fail because one extra request did.
        """
        try:
            data = self._get(f"/repos/{self._repo}/git/ref/tags/{tag}")
        except NetworkError:
            return None
        obj = data.get("object") if isinstance(data, dict) else None
        if not isinstance(obj, dict) or obj.get("type") != "commit":
            return None
        sha = obj.get("sha")
        return sha if isinstance(sha, str) and sha else None


def _read_assets(raw: Any) -> list[ReleaseAsset]:
    """Turn the API's asset list into :class:`ReleaseAsset` values, skipping nonsense."""
    if not isinstance(raw, list):
        return []
    assets: list[ReleaseAsset] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("browser_download_url")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        assets.append(
            ReleaseAsset(
                name=name,
                size=size if isinstance(size, int) else 0,
                url=url,
                sha256=_digest(item.get("digest")),
            )
        )
    return assets


def _digest(raw: Any) -> str | None:
    """Read GitHub's ``digest`` field, which is ``sha256:<hex>`` when it is present."""
    if not isinstance(raw, str):
        return None
    algorithm, separator, value = raw.partition(":")
    if separator and algorithm.lower() == "sha256" and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    if re.fullmatch(r"[0-9a-fA-F]{64}", raw):
        return raw.lower()
    return None


@dataclass
class FakeReleaseClient:
    """Canned releases keyed by tag, for tests.

    Args:
        releases: Every release this client knows, by tag.
        latest_tag: Which of them ``latest()`` returns.
        calls: Records each lookup, so a test can assert nothing extra was asked for.
    """

    releases: Mapping[str, Release]
    latest_tag: str
    calls: list[str] = field(default_factory=list)

    def latest(self) -> Release:
        """The release named by ``latest_tag``.

        Raises:
            NetworkError: When ``latest_tag`` names no known release.
        """
        self.calls.append("latest")
        return self.by_tag(self.latest_tag)

    def by_tag(self, tag: str) -> Release:
        """The canned release with this tag.

        Raises:
            NetworkError: When the tag is not one of the canned releases.
        """
        self.calls.append(tag)
        release = self.releases.get(tag)
        if release is None:
            raise NetworkError(
                _("GitHub has no release at %(url)s.")
                % {"url": f"{API_ROOT}/repos/{REPO}/releases/tags/{tag}"},
                hint=_("Check the tag with `llamafit install llama.cpp --tag ...`."),
            )
        return release


def is_runtime_archive(name: str) -> bool:
    """Whether an asset name is one of the archives that hold llama.cpp binaries.

    The releases also carry the CUDA runtime bundle, an Apple xcframework and the
    source tarballs GitHub attaches by itself. None of them is a build of llama.cpp,
    and every one of them would otherwise match some platform's tokens.
    """
    lowered = name.lower()
    if not lowered.endswith(ARCHIVE_SUFFIXES):
        return False
    found = set(tokens(lowered))
    if found & _EXCLUDED_TOKENS:
        return False
    return "bin" in found and "llama" in found


def _matches_os(found: set[str], os_name: OsName) -> bool:
    return bool(found & set(_OS_TOKENS[os_name]))


def _matches_arch(found: set[str], arch: Arch) -> bool:
    wanted = _ARCH_TOKENS[arch]
    if not wanted:
        return False
    if found & set(wanted):
        return True
    # ``x86_64`` written with a separator arrives as two tokens.
    return arch == "x86_64" and "x86" in found and "64" in found


def _matches_backend(found: set[str], backend: Backend) -> bool:
    wanted = _BACKEND_TOKENS[backend]
    if wanted:
        return bool(found & set(wanted))
    # ``cpu`` and ``metal``: the build that carries no accelerator at all.
    return not (found & ACCELERATOR_TOKENS)


def cuda_version(name: str) -> tuple[int, ...]:
    """The CUDA version an asset name declares, as ``(12, 4)``, or ``()``.

    Both spellings the project has used are read: ``cuda-12.4`` and ``cu12.4``.
    """
    found = tokens(name)
    for index, token in enumerate(found):
        if token == "cuda":
            digits = []
            for following in found[index + 1 :]:
                if not following.isdigit():
                    break
                digits.append(int(following))
            if digits:
                return tuple(digits)
        match = re.fullmatch(r"cu(\d+)", token)
        if match:
            rest = found[index + 1] if index + 1 < len(found) else ""
            minor = (int(rest),) if rest.isdigit() else ()
            return (int(match.group(1)), *minor)
    return ()


def _cpu_rank(found: set[str]) -> int:
    """How much a CPU build is wanted, lower being better; 9 for an unlabelled one."""
    ranked = [rank for token, rank in _CPU_VARIANT_RANK.items() if token in found]
    return min(ranked) if ranked else 9


def driver_version(text: str | None) -> float | None:
    """The leading number of an NVIDIA driver string, ``610.88`` from ``"610.88"``.

    ``None`` when there is nothing to read, which is how a machine with no NVIDIA card,
    or one whose driver could not be probed, is told apart from one whose driver is
    simply old.
    """
    if not text:
        return None
    match = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def cuda_runs_on(name: str, driver: float | None) -> bool:
    """Whether a CUDA build could load at all on this driver.

    A CUDA 13 build on a driver below 580 does not run slowly; it does not start, and
    the message says a CUDA library is missing rather than that the driver is too old.
    Filtering here is what keeps that from being the outcome of an install that reported
    success.

    A CUDA major nobody has told this table about is allowed through: refusing an archive
    because a constant has not been updated would be the same silent failure the other
    way round.
    """
    version = cuda_version(name)
    if not version or driver is None:
        return True
    floor = CUDA_DRIVER_FLOOR.get(version[0])
    return floor is None or driver >= floor


def _preference(asset: ReleaseAsset, *, newest_cuda: bool) -> tuple[int, tuple[int, ...], str]:
    """Sort key that puts the best asset of a matching group first.

    ``newest_cuda`` says which way the CUDA versions run. With a known driver, everything
    it cannot load has already been filtered out and the newest of what is left is the
    fastest. With no driver to check against, the *oldest* published CUDA is taken: it is
    the one with the widest driver support, and a build that will not start is worse than
    one that leaves a little speed on the table — the same rule that prefers the AVX2
    archive to the AVX512 one.

    Ties fall back to the name, so the choice is the same on every run and every machine.
    """
    found = set(tokens(asset.name))
    version = cuda_version(asset.name)
    # Negated element-wise: tuples do not negate, and a plain reverse sort would also
    # reverse the name tie-break, which should stay alphabetical.
    ordered = tuple(-part for part in version) if newest_cuda else version
    return (_cpu_rank(found), ordered, asset.name)


def select_asset(
    assets: Iterable[ReleaseAsset],
    *,
    os_name: OsName,
    arch: Arch,
    backend: Backend,
    driver: str | None = None,
) -> ReleaseAsset | None:
    """The one archive to install for this platform and backend, or ``None``.

    Args:
        assets: Everything attached to the release.
        os_name: Which operating system the archive has to be built for.
        arch: Which architecture.
        backend: Which compiled backend. ``metal`` and ``cpu`` both mean "the build
            that carries no accelerator", because Metal is compiled into the plain
            macOS arm64 archive.
        driver: The NVIDIA driver version on the machine, when it is known. A release
            publishes CUDA archives for two or three CUDA majors at once and they need
            different minimum drivers; knowing the driver is what turns that from a
            guess into a choice. Without it the most widely compatible archive is taken.

    Returns:
        The best matching archive, or ``None`` when the release publishes none — which
        is a normal answer, not a failure: llama.cpp publishes no CUDA build for Linux
        at all, and the caller falls back to the next backend it would accept.
    """
    driver_number = driver_version(driver)
    candidates = [
        asset
        for asset in assets
        if is_runtime_archive(asset.name)
        and _matches_os(set(tokens(asset.name)), os_name)
        and _matches_arch(set(tokens(asset.name)), arch)
        and _matches_backend(set(tokens(asset.name)), backend)
        and cuda_runs_on(asset.name, driver_number)
    ]
    if not candidates:
        return None
    return sorted(
        candidates, key=lambda asset: _preference(asset, newest_cuda=driver_number is not None)
    )[0]


def companion_assets(
    assets: Iterable[ReleaseAsset],
    chosen: ReleaseAsset,
    *,
    os_name: OsName,
    backend: Backend,
) -> tuple[ReleaseAsset, ...]:
    """Archives that have to be unpacked beside ``chosen`` for it to run at all.

    Today there is exactly one: the Windows CUDA build links against the CUDA runtime
    libraries, which are published as their own ``cudart-...`` archive. Install the
    build without them and ``llama-server.exe`` exits before it prints a version, with
    a message about a missing DLL that says nothing about llama.cpp.

    The companion is matched on the same CUDA version as the build, so a release that
    publishes runtimes for two CUDA versions cannot pair the wrong one.
    """
    if os_name != "windows" or backend != "cuda":
        return ()
    version = cuda_version(chosen.name)
    found = [
        asset
        for asset in assets
        if "cudart" in tokens(asset.name)
        and asset.name.lower().endswith(ARCHIVE_SUFFIXES)
        and _matches_os(set(tokens(asset.name)), os_name)
        and (not version or cuda_version(asset.name) == version)
    ]
    return tuple(sorted(found, key=lambda asset: asset.name)[:1])


def backend_preference(
    *, os_name: OsName, arch: Arch, vendor: Vendor | None
) -> tuple[Backend, ...]:
    """Which backends to try for this machine, best first.

    The list is a preference and not a decision: what is actually installed is the first
    entry the release publishes an archive for, so a machine whose ideal backend has no
    published build still gets a working one rather than an error.

    On Windows an AMD card is sent to Vulkan even though a ROCm archive is published,
    because section 15 of the specification names ``cuda``, ``vulkan`` and ``cpu`` as
    the Windows matrix, and Vulkan works on every card the ROCm build supports and on
    several it does not. On Linux, ROCm is asked for first and falls through to Vulkan
    on its own, which is what "``rocm`` when published" means in practice — and it is
    published there today. An NVIDIA card on Linux falls the same way, because llama.cpp
    publishes no Linux CUDA archive at all.
    """
    if os_name == "macos":
        return ("metal", "cpu") if arch == "arm64" else ("cpu",)
    if vendor == "nvidia":
        return ("cuda", "vulkan", "cpu")
    if vendor == "amd":
        return ("hip", "vulkan", "cpu") if os_name == "linux" else ("vulkan", "cpu")
    if vendor == "intel":
        return ("vulkan", "cpu")
    return ("cpu",)


def resolve_backend(
    assets: Sequence[ReleaseAsset],
    *,
    os_name: OsName,
    arch: Arch,
    vendor: Vendor | None,
    wanted: Backend | None = None,
    driver: str | None = None,
) -> tuple[Backend, ReleaseAsset] | None:
    """Pick the backend and the archive together, since neither is decidable alone.

    Args:
        assets: Everything attached to the release.
        os_name: The machine's operating system.
        arch: The machine's architecture.
        vendor: The primary GPU's vendor, or ``None`` on a machine with no GPU.
        wanted: A backend the user asked for by name. It is honoured exactly: if the
            release publishes no archive for it, nothing is returned and the caller
            reports that, rather than quietly installing something else. A person who
            typed ``--backend cuda`` wants to hear that there is no CUDA build far more
            than they want a CPU one.
        driver: The NVIDIA driver version, passed through to `select_asset`.

    Returns:
        The backend and its archive, or ``None`` when nothing matches.
    """
    order = (
        (wanted,)
        if wanted is not None
        else backend_preference(os_name=os_name, arch=arch, vendor=vendor)
    )
    for backend in order:
        asset = select_asset(assets, os_name=os_name, arch=arch, backend=backend, driver=driver)
        if asset is not None:
            return backend, asset
    return None


__all__ = [
    "ACCELERATOR_TOKENS",
    "API_ROOT",
    "ARCHIVE_SUFFIXES",
    "CUDA_DRIVER_FLOOR",
    "REPO",
    "FakeReleaseClient",
    "HttpReleaseClient",
    "Release",
    "ReleaseAsset",
    "ReleaseClient",
    "backend_preference",
    "companion_assets",
    "cuda_runs_on",
    "cuda_version",
    "driver_version",
    "is_runtime_archive",
    "parse_build",
    "resolve_backend",
    "select_asset",
    "tokens",
]
