# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Put a published llama.cpp build on this machine, without destroying anything.

This is the first thing LlamaFit does that writes outside its own directories, and the
order of operations is the whole design:

1. **Look before touching.** :func:`inspect_target` decides whether the directory is
   ours, empty, or somebody else's, and a directory that is somebody else's stops the
   install rather than being overwritten. People build llama.cpp themselves, with their
   own flags; replacing that destroys work no download can restore.
2. **Say the plan.** :func:`plan_install` returns every fact a person needs before
   agreeing — release, archive, size, checksum, destination, free space, what is already
   there — and writes nothing at all.
3. **Download, resuming.** :func:`download_asset` appends to a ``.part`` file beside the
   destination, so a dropped connection costs the seconds since the last chunk and not
   the whole archive.
4. **Verify, then unpack.** The checksum is checked on the ``.part`` file, and only a
   file that matches is renamed to the archive name and opened. A corrupt archive
   unpacked over a working installation is a far worse outcome than a failed download,
   and that ordering is what makes it impossible.
5. **Swap, don't dribble.** The new ``bin`` directory is assembled whole, beside the old
   one, and moved into place in one step; the old one is only deleted once the new one
   has arrived.

What is installed has to be what :mod:`llamafit.llamacpp.detect` finds: the managed
directory is the first entry of that module's well-known list, the binaries land in its
``bin`` subdirectory, ``VERSION.txt`` is written in the form its version parser reads,
and the compiled backends are read back with its own :func:`~detect.detect_backends`
before the install is called a success.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from llamafit import __version__
from llamafit.errors import LlamaFitError, NetworkError
from llamafit.hardware.runner import Runner
from llamafit.i18n import _, ngettext
from llamafit.llamacpp.detect import detect_backends, exe_name
from llamafit.llamacpp.releases import Release, ReleaseAsset
from llamafit.logging import get_logger
from llamafit.models.host import Arch, Backend, OsName
from llamafit.units import format_bytes

_log = get_logger("llamacpp.install")

MARKER_NAME = ".llamafit-install.json"
"""The file that says a directory was created by LlamaFit and may be replaced."""

MARKER_TOOL = "llamafit"
"""The value the marker's ``tool`` field must hold for the directory to count as ours."""

VERSION_FILE = "VERSION.txt"
"""Written beside the binaries; ``detect.parse_version`` reads it when the exe cannot run."""

STAGING_NAME = ".llamafit-staging"
"""Where an install is assembled, inside the target so the final move never crosses a volume."""

DOWNLOAD_CHUNK = 1 << 20
"""How much of a response body to hold before writing it out."""

ARCHIVE_EXPANSION = 3.0
"""How much larger the unpacked tree is assumed to be than the archive.

Measured against the published Windows CUDA and Linux Vulkan archives, which expand by
about 2.4 and 2.7 times. Rounded up, because a space check that is optimistic is a
check that lets an install run out of room half way through.
"""

RETRY_ATTEMPTS = 3
"""How many times a download may resume after the connection drops before giving up."""


class InstallError(LlamaFitError):
    """An install cannot proceed or did not complete.

    A subclass of :class:`~llamafit.errors.LlamaFitError` so the CLI renders it as a
    message with a hint rather than a traceback. Every raise site says what was found
    and what the user can do about it, because most of these are recoverable by hand:
    free some space, pass ``--force``, run the command again.
    """


ProgressCallback = Callable[[int, int | None], None]
"""Called with bytes written so far and the total when it is known."""


# --------------------------------------------------------------------------------------
# Where things go
# --------------------------------------------------------------------------------------


def managed_dir(home: Path | None = None) -> Path:
    """The directory LlamaFit installs llama.cpp into.

    ``~/.llamafit/llama.cpp``, which is deliberately the first entry of
    :func:`llamafit.llamacpp.detect.well_known_dirs`: an install here is found by the
    next scan with no configuration, no ``PATH`` change and no environment variable.
    """
    return (home or Path.home()) / ".llamafit" / "llama.cpp"


def bin_dir_of(root: Path) -> Path:
    """Where the binaries live inside an install root."""
    return root / "bin"


# --------------------------------------------------------------------------------------
# Whose directory is this
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallMarker:
    """What LlamaFit wrote into a directory it created.

    Its presence, with ``tool`` reading ``llamafit``, is the only thing that makes a
    directory replaceable without an explicit instruction. Everything else in it is for
    the reader: which release, which archive, which checksum, when.
    """

    tool: str = MARKER_TOOL
    llamafit_version: str = __version__
    installed_at: str = ""
    tag: str = ""
    build: int | None = None
    commit: str | None = None
    asset: str = ""
    sha256: str | None = None
    backend: str = ""
    os_name: str = ""
    arch: str = ""
    backends: tuple[str, ...] = ()

    @property
    def ours(self) -> bool:
        """Whether this marker claims the directory for LlamaFit."""
        return self.tool == MARKER_TOOL

    def to_json(self) -> str:
        """Render the marker as the JSON text written into the directory."""
        payload: dict[str, Any] = {
            "tool": self.tool,
            "llamafit_version": self.llamafit_version,
            "installed_at": self.installed_at,
            "tag": self.tag,
            "build": self.build,
            "commit": self.commit,
            "asset": self.asset,
            "sha256": self.sha256,
            "backend": self.backend,
            "os": self.os_name,
            "arch": self.arch,
            "backends": list(self.backends),
        }
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> InstallMarker | None:
        """Read a marker, or ``None`` when the text is not one.

        Unreadable is not the same as absent and is deliberately not treated as ours:
        a directory whose marker cannot be parsed cannot be proven to be LlamaFit's,
        and the whole point of the marker is that the proof is what licenses deleting
        what is there.
        """
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("tool") != MARKER_TOOL:
            return None
        backends = data.get("backends")
        build = data.get("build")
        return cls(
            tool=MARKER_TOOL,
            llamafit_version=str(data.get("llamafit_version", "")),
            installed_at=str(data.get("installed_at", "")),
            tag=str(data.get("tag", "")),
            build=build if isinstance(build, int) else None,
            commit=data.get("commit") if isinstance(data.get("commit"), str) else None,
            asset=str(data.get("asset", "")),
            sha256=data.get("sha256") if isinstance(data.get("sha256"), str) else None,
            backend=str(data.get("backend", "")),
            os_name=str(data.get("os", "")),
            arch=str(data.get("arch", "")),
            backends=tuple(str(b) for b in backends) if isinstance(backends, list) else (),
        )


Ownership = Literal["absent", "empty", "ours", "foreign"]
"""What is at the install directory: nothing, nothing that matters, LlamaFit's, or somebody's."""


@dataclass(frozen=True)
class Target:
    """The install directory and what is already in it.

    Args:
        path: The directory itself.
        ownership: ``absent`` when it does not exist, ``empty`` when it holds nothing,
            ``ours`` when it carries LlamaFit's marker, ``foreign`` for anything else.
        marker: The marker read from it, when there was a readable one.
        detail: One translated sentence saying what was found, for ``foreign`` only.
    """

    path: Path
    ownership: Ownership
    marker: InstallMarker | None = None
    detail: str | None = None

    @property
    def replaceable(self) -> bool:
        """Whether installing here needs no further permission from the user."""
        return self.ownership in ("absent", "empty", "ours")


def read_marker(root: Path) -> InstallMarker | None:
    """The marker in ``root``, or ``None`` when there is none or it cannot be read."""
    try:
        text = (root / MARKER_NAME).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return InstallMarker.from_json(text)


def inspect_target(root: Path) -> Target:
    """Decide whether ``root`` is LlamaFit's to write into.

    A directory is LlamaFit's only when it carries a marker naming LlamaFit. Anything
    else with contents in it — a build somebody compiled, an unpacked release, a
    directory whose marker will not parse — is foreign, and the caller stops.

    A file where the directory should be is foreign too: it is not ours, and silently
    deleting it would be exactly the failure this function exists to prevent.
    """
    if not root.exists():
        return Target(path=root, ownership="absent")
    if not root.is_dir():
        return Target(
            path=root,
            ownership="foreign",
            detail=_("%(path)s is a file, not a directory.") % {"path": root},
        )
    marker = read_marker(root)
    if marker is not None and marker.ours:
        return Target(path=root, ownership="ours", marker=marker)
    try:
        entries = [entry for entry in root.iterdir() if entry.name != MARKER_NAME]
    except OSError as exc:
        return Target(
            path=root,
            ownership="foreign",
            detail=_("%(path)s cannot be read: %(error)s") % {"path": root, "error": exc},
        )
    if not entries:
        return Target(path=root, ownership="empty")
    if (root / MARKER_NAME).exists():
        return Target(
            path=root,
            ownership="foreign",
            detail=_("%(path)s has a marker file that cannot be read.")
            % {"path": root / MARKER_NAME},
        )
    detail = ngettext(
        "%(path)s already holds %(count)d file that LlamaFit did not install.",
        "%(path)s already holds %(count)d files that LlamaFit did not install.",
        len(entries),
    ) % {"path": root, "count": len(entries)}
    return Target(path=root, ownership="foreign", detail=detail)


# --------------------------------------------------------------------------------------
# Room on the disk
# --------------------------------------------------------------------------------------


def free_bytes(path: Path) -> int | None:
    """Free space on the volume ``path`` lives on, walking up to an existing ancestor.

    ``None`` when it cannot be read at all, which is reported as unknown rather than as
    zero: refusing to install because a figure could not be obtained would be worse than
    trying and failing with the operating system's own message.
    """
    candidate = Path(path).absolute()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    try:
        return shutil.disk_usage(candidate).free
    except OSError:
        return None


@dataclass(frozen=True)
class SpaceCheck:
    """How much room one step needs and how much the volume has.

    Args:
        path: A path on the volume being checked.
        purpose: ``download`` or ``install``; a machine whose cache and home are on
            different volumes has to satisfy both.
        needed: Bytes the step will use.
        free: Bytes available, or ``None`` when it could not be read.
    """

    path: Path
    purpose: Literal["download", "install"]
    needed: int
    free: int | None

    @property
    def enough(self) -> bool:
        """Whether the step fits. Unknown free space counts as enough; see `free_bytes`."""
        return self.free is None or self.free >= self.needed


def check_space(
    *,
    archive_dir: Path,
    install_dir: Path,
    download_bytes: int,
    unpacked_bytes: int,
    free: Callable[[Path], int | None] = free_bytes,
) -> tuple[SpaceCheck, ...]:
    """What the download and the unpacked tree need, per volume.

    ``free`` is injected so a test can say "this volume is full" without filling one.
    """
    checks = [
        SpaceCheck(
            path=archive_dir, purpose="download", needed=download_bytes, free=free(archive_dir)
        ),
        SpaceCheck(
            path=install_dir, purpose="install", needed=unpacked_bytes, free=free(install_dir)
        ),
    ]
    return tuple(checks)


# --------------------------------------------------------------------------------------
# Getting the bytes
# --------------------------------------------------------------------------------------


@dataclass
class Body:
    """A response being read in pieces.

    Args:
        chunks: The bytes, in the order they arrive.
        total: The size of the whole file when the server said, counting from zero even
            for a partial response.
        resumed: Whether the server honoured the requested range. A server that answers
            ``200`` to a ranged request is sending the file from the start, and appending
            that to what is already on disk would build a file that is part duplicate.
    """

    chunks: Iterator[bytes]
    total: int | None = None
    resumed: bool = False


class Fetcher(Protocol):
    """Reads a URL, optionally starting part way in.

    The one HTTP surface this module needs, behind a protocol for the same reason
    :class:`llamafit.hardware.runner.Runner` is one: the tests substitute a recorded
    body, a body that stops half way, and a server that ignores ranges, and none of
    them touches a network.
    """

    def open_range(self, url: str, *, start: int = 0) -> AbstractContextManager[Body]:
        """Open ``url`` from byte ``start``, yielding the body and closing it after."""
        ...


class HttpFetcher:
    """Streams a URL over HTTPS with a ``Range`` header when resuming."""

    def __init__(self, client: httpx.Client | None = None, *, timeout: float = 60.0) -> None:
        """Remember the client to reuse, if any, and the per-read timeout."""
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=True)
        self._timeout = timeout

    def close(self) -> None:
        """Close the underlying HTTP client, but only when this instance created it."""
        if self._owns_client:
            self._client.close()

    @contextmanager
    def open_range(self, url: str, *, start: int = 0) -> Iterator[Body]:
        """Open ``url``, asking for everything from ``start`` onwards.

        Raises:
            NetworkError: The request failed or was answered with an unusable status.
        """
        headers = {"User-Agent": f"llamafit/{__version__}", "Accept": "application/octet-stream"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with self._client.stream(
                "GET", url, headers=headers, timeout=self._timeout, follow_redirects=True
            ) as response:
                if response.status_code not in (200, 206):
                    raise NetworkError(
                        _("%(url)s returned HTTP %(status)d.")
                        % {"url": url, "status": response.status_code},
                        hint=_("Check your network connection and try again."),
                    )
                yield Body(
                    chunks=_guard(response.iter_bytes(DOWNLOAD_CHUNK), url),
                    total=_total_from(response, start),
                    resumed=response.status_code == 206,
                )
        except httpx.HTTPError as exc:
            raise NetworkError(
                _("could not download %(url)s: %(error)s") % {"url": url, "error": exc},
                hint=_("Check your network connection and try again."),
            ) from exc


def _total_from(response: httpx.Response, start: int) -> int | None:
    """The whole file's size, from ``Content-Range`` when present or ``Content-Length``."""
    content_range = response.headers.get("content-range", "")
    tail = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    if tail.isdigit():
        return int(tail)
    length = response.headers.get("content-length")
    if length is not None and length.isdigit():
        return int(length) + (start if response.status_code == 206 else 0)
    return None


def _guard(chunks: Iterator[bytes], url: str) -> Iterator[bytes]:
    """Turn a mid-stream transport failure into a :class:`NetworkError`.

    Without this the failure arrives at the caller as an ``httpx`` exception from inside
    a ``for`` loop, and the resume logic would have to know about ``httpx`` to catch it.
    """
    try:
        yield from chunks
    except httpx.HTTPError as exc:
        raise NetworkError(
            _("the download of %(url)s stopped early: %(error)s") % {"url": url, "error": exc},
            hint=_("Run the command again; it continues from where it stopped."),
        ) from exc


@dataclass
class FakeFetcher:
    """Serves canned bodies, with the failures a real download meets.

    Args:
        bodies: The whole file, by URL.
        chunk: How many bytes each yielded piece holds.
        stop_after: Stop the body after this many bytes, once, as a dropped connection
            does. Cleared after it fires, so the next attempt succeeds — which is what
            makes a resume testable.
        honours_range: When ``False``, every response starts at byte zero whatever was
            asked for, like a server or proxy that does not do ranges.
        calls: ``(url, start)`` for every request, so a test can prove a resume asked
            for the right offset.
    """

    bodies: Mapping[str, bytes]
    chunk: int = 8
    stop_after: int | None = None
    honours_range: bool = True
    calls: list[tuple[str, int]] = field(default_factory=list)

    @contextmanager
    def open_range(self, url: str, *, start: int = 0) -> Iterator[Body]:
        """Yield the canned body from ``start``, or from zero when ranges are refused.

        Raises:
            NetworkError: When no body is registered for ``url``.
        """
        self.calls.append((url, start))
        data = self.bodies.get(url)
        if data is None:
            raise NetworkError(
                _("%(url)s returned HTTP %(status)d.") % {"url": url, "status": 404},
                hint=_("Check your network connection and try again."),
            )
        offset = start if self.honours_range else 0
        yield Body(
            chunks=self._chunks(url, data[offset:]),
            total=len(data),
            resumed=bool(offset) and self.honours_range,
        )

    def _chunks(self, url: str, data: bytes) -> Iterator[bytes]:
        sent = 0
        for index in range(0, len(data), self.chunk):
            piece = data[index : index + self.chunk]
            if self.stop_after is not None and sent + len(piece) > self.stop_after:
                piece = piece[: max(self.stop_after - sent, 0)]
                if piece:
                    yield piece
                self.stop_after = None
                raise NetworkError(
                    _("the download of %(url)s stopped early: %(error)s")
                    % {"url": url, "error": _("the connection was closed")},
                    hint=_("Run the command again; it continues from where it stopped."),
                )
            sent += len(piece)
            yield piece


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, read in chunks so a large archive never lands in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def part_path(destination: Path) -> Path:
    """Where a partial download is kept: the destination with ``.part`` appended."""
    return destination.with_name(destination.name + ".part")


def download_asset(
    asset: ReleaseAsset,
    destination: Path,
    *,
    fetcher: Fetcher,
    progress: ProgressCallback | None = None,
    attempts: int = RETRY_ATTEMPTS,
    require_checksum: bool = True,
) -> Path:
    """Download ``asset`` to ``destination``, resuming and verifying before it counts.

    The bytes go to ``destination.part`` and are only renamed to ``destination`` once
    the checksum matches, so a file under the archive's real name is always a file that
    has been verified. Every caller downstream may therefore assume it.

    A connection that drops is retried up to ``attempts`` times, each attempt continuing
    from what is already on disk. A server that answers a ranged request with the whole
    file is detected and the partial file is discarded rather than appended to.

    Args:
        asset: What to fetch, including the published checksum when there is one.
        destination: Where the verified archive should end up.
        fetcher: How to read the URL.
        progress: Called with bytes on disk and the total, as often as chunks arrive.
        attempts: How many times to resume before giving up.
        require_checksum: Refuse an asset with no published checksum. Passing ``False``
            is what ``--allow-unverified`` does, and the caller has to say so out loud.

    Returns:
        The path of the verified archive.

    Raises:
        InstallError: The checksum does not match, no checksum was published and one is
            required, or the file could not be written.
        NetworkError: Every attempt failed.
    """
    if require_checksum and not asset.sha256:
        raise InstallError(
            _("%(name)s has no published checksum, so it cannot be verified.")
            % {"name": asset.name},
            hint=_(
                "Pass --allow-unverified to install it anyway, knowing the archive is unchecked."
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _already_good(destination, asset):
        _log.debug("archive already downloaded and verified: %s", destination)
        if progress is not None:
            progress(asset.size, asset.size or None)
        return destination

    part = part_path(destination)
    last_error: NetworkError | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            _fetch_into(part, asset, fetcher=fetcher, progress=progress)
            break
        except NetworkError as exc:
            last_error = exc
            _log.debug("download attempt %d of %d failed: %s", attempt, attempts, exc.message)
    else:
        if last_error is not None:
            raise last_error
        raise NetworkError(
            _("could not download %(url)s: %(error)s")
            % {"url": asset.url, "error": _("the connection was closed")},
            hint=_("Run the command again; it continues from where it stopped."),
        )

    _verify(part, asset)
    part.replace(destination)
    return destination


def _already_good(destination: Path, asset: ReleaseAsset) -> bool:
    """Whether an archive already on disk is the one wanted, checksum and all."""
    if asset.sha256:
        return sha256_of(destination) == asset.sha256
    try:
        return bool(asset.size) and destination.stat().st_size == asset.size
    except OSError:
        return False


def _fetch_into(
    part: Path, asset: ReleaseAsset, *, fetcher: Fetcher, progress: ProgressCallback | None
) -> None:
    """One attempt: continue ``part`` from wherever it stopped.

    Raises:
        InstallError: The file could not be written; a full disk arrives here.
        NetworkError: The connection failed or stopped early.
    """
    start = part.stat().st_size if part.exists() else 0
    if asset.size and start > asset.size:
        # Longer than the file it claims to be: it is not a prefix of anything.
        part.unlink(missing_ok=True)
        start = 0
    with fetcher.open_range(asset.url, start=start) as body:
        if start and not body.resumed:
            _log.debug("server ignored the range request; restarting %s", part)
            part.unlink(missing_ok=True)
            start = 0
        total = body.total if body.total is not None else (asset.size or None)
        written = start
        try:
            with part.open("ab" if start else "wb") as handle:
                for chunk in body.chunks:
                    handle.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
        except OSError as exc:
            raise _write_failed(part, exc) from exc
    if asset.size and written != asset.size:
        raise NetworkError(
            _("the download of %(url)s stopped early: %(error)s")
            % {
                "url": asset.url,
                "error": _("%(got)s of %(want)s arrived")
                % {"got": format_bytes(written), "want": format_bytes(asset.size)},
            },
            hint=_("Run the command again; it continues from where it stopped."),
        )


def _write_failed(path: Path, exc: OSError) -> InstallError:
    """The error for a write that failed, naming a full disk when that is what it was."""
    if exc.errno == errno.ENOSPC:
        return InstallError(
            _("the disk holding %(path)s is full.") % {"path": path},
            hint=_("Free some space and run the command again; the download resumes."),
        )
    return InstallError(
        _("could not write %(path)s: %(error)s") % {"path": path, "error": exc},
        hint=_("Check that the directory exists and is writable."),
    )


def _verify(part: Path, asset: ReleaseAsset) -> None:
    """Check the downloaded file against the published checksum, before anything opens it.

    A file that does not match is deleted rather than kept. Keeping it would mean the
    next run resumes onto bytes already known to be wrong, and every run after that
    would fail the same way with nothing saying why.

    Raises:
        InstallError: The checksum does not match.
    """
    if not asset.sha256:
        _log.debug("no published checksum for %s; nothing to verify", asset.name)
        return
    actual = sha256_of(part)
    if actual == asset.sha256:
        return
    part.unlink(missing_ok=True)
    raise InstallError(
        _("%(name)s did not match its published checksum and was discarded.")
        % {"name": asset.name},
        hint=_(
            "Nothing was unpacked. Run the command again: the archive arrived damaged, "
            "which a second download usually fixes."
        ),
    )


# --------------------------------------------------------------------------------------
# Unpacking
# --------------------------------------------------------------------------------------


def _is_within(root: Path, candidate: Path) -> bool:
    """Whether ``candidate`` stays inside ``root`` once both are resolved."""
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _refuse_member(archive: Path, name: str) -> InstallError:
    return InstallError(
        _("%(archive)s contains %(name)s, which would be written outside the install directory.")
        % {"archive": archive.name, "name": name},
        hint=_("Do not install this archive; report it to the llama.cpp project."),
    )


def extract_archive(archive: Path, into: Path) -> Path:
    """Unpack ``archive`` into ``into``, refusing any member that would escape it.

    A member with an absolute path, one that climbs out with ``..``, or a link pointing
    outside is not unpacked and stops the whole install. Nothing in a llama.cpp release
    has ever looked like that; the check is here because unpacking an archive is
    running somebody else's instructions about where to write files.

    Executable bits are restored from the archive on POSIX, where a ``zip`` member's
    mode lives in a field Python's extractor does not apply. Without that,
    ``llama-server`` unpacks without its executable bit and cannot be run at all.

    Returns:
        ``into``, now holding the unpacked tree.

    Raises:
        InstallError: The archive is unreadable, of an unknown kind, or holds a member
            that would be written outside ``into``.
    """
    into.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            _extract_zip(archive, into)
        elif name.endswith((".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")):
            _extract_tar(archive, into)
        else:
            raise InstallError(
                _("%(name)s is not an archive LlamaFit knows how to unpack.")
                % {"name": archive.name},
                hint=_("Report this; the release layout has changed."),
            )
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        raise InstallError(
            _("%(name)s could not be unpacked: %(error)s") % {"name": archive.name, "error": exc},
            hint=_("Delete it and run the command again to download it afresh."),
        ) from exc
    except OSError as exc:
        raise _write_failed(into, exc) from exc
    return into


def _extract_zip(archive: Path, into: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = into / member.filename
            if os.path.isabs(member.filename) or not _is_within(into, target.parent):
                raise _refuse_member(archive, member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise _refuse_member(archive, member.filename)
            bundle.extract(member, into)
            if os.name != "nt" and not member.is_dir() and mode & 0o111:
                target.chmod(target.stat().st_mode | 0o111)


def _extract_tar(archive: Path, into: Path) -> None:
    # Python's own ``data`` filter is asked for as well, where the interpreter has it
    # (3.12, and the 3.10 and 3.11 security releases). It refuses the same members these
    # lines do; passing it says so to the reader and to a future interpreter, which from
    # 3.14 filters by default and warns until then. Keyword-splatted because the
    # parameter does not exist on the oldest interpreter LlamaFit supports.
    extra: dict[str, Any] = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = into / member.name
            escapes = os.path.isabs(member.name) or not _is_within(into, target.parent)
            if escapes or member.islnk() or member.issym() or member.isdev():
                raise _refuse_member(archive, member.name)
            bundle.extract(member, into, **extra)


def payload_dir(unpacked: Path, os_name: OsName) -> Path:
    """The directory inside an unpacked release that holds ``llama-server`` and its libraries.

    Releases have put the binaries at the root of the archive and inside ``build/bin``
    at different times, and both layouts are read the same way: find the server, and the
    directory it is in is the payload. Taking the whole directory rather than a list of
    known file names is deliberate — the ``ggml-*`` backend libraries beside it are what
    :func:`llamafit.llamacpp.detect.detect_backends` reads, and a release that adds one
    should not need this code changed.

    Raises:
        InstallError: The archive holds no ``llama-server``.
    """
    server = exe_name("llama-server", os_name)
    matches = sorted(unpacked.rglob(server), key=lambda p: len(p.parts))
    if not matches:
        raise InstallError(
            _("the archive holds no %(name)s.") % {"name": server},
            hint=_("Report this; the release layout has changed."),
        )
    return matches[0].parent


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallPlan:
    """Everything that is about to happen, before any of it does.

    This is what the command prints and asks about. It is built without writing a byte,
    and it carries the facts a person needs to say no: which build, which archive, how
    large, where it will go, what is already there, and whether the archive can be
    checked at all.
    """

    release: Release
    asset: ReleaseAsset
    companions: tuple[ReleaseAsset, ...]
    backend: Backend
    os_name: OsName
    arch: Arch
    target: Target
    bin_dir: Path
    archive_path: Path
    resume_bytes: int
    space: tuple[SpaceCheck, ...]

    @property
    def download_bytes(self) -> int:
        """How much still has to come down, after whatever is already on disk."""
        total = self.asset.size + sum(c.size for c in self.companions)
        return max(total - self.resume_bytes, 0)

    @property
    def verified(self) -> bool:
        """Whether every archive in the plan publishes a checksum to check it against."""
        return all(a.sha256 for a in (self.asset, *self.companions))

    @property
    def replaces(self) -> InstallMarker | None:
        """The LlamaFit install this one would replace, when there is one."""
        return self.target.marker if self.target.ownership == "ours" else None

    @property
    def short_space(self) -> tuple[SpaceCheck, ...]:
        """The volumes that do not have room."""
        return tuple(check for check in self.space if not check.enough)


def plan_install(
    release: Release,
    asset: ReleaseAsset,
    *,
    companions: Sequence[ReleaseAsset] = (),
    backend: Backend,
    os_name: OsName,
    arch: Arch,
    root: Path,
    cache_dir: Path,
    free: Callable[[Path], int | None] = free_bytes,
) -> InstallPlan:
    """Work out what installing this asset would do. Writes nothing.

    The download directory is a cache rather than a temporary one on purpose: a
    ``.part`` file that survives the process is what makes "run it again" continue a
    download instead of starting it over.
    """
    archive_dir = cache_dir / "llamacpp"
    archive_path = archive_dir / f"{release.tag}-{asset.name}"
    resume = part_path(archive_path).stat().st_size if part_path(archive_path).exists() else 0
    total_archive = asset.size + sum(c.size for c in companions)
    return InstallPlan(
        release=release,
        asset=asset,
        companions=tuple(companions),
        backend=backend,
        os_name=os_name,
        arch=arch,
        target=inspect_target(root),
        bin_dir=bin_dir_of(root),
        archive_path=archive_path,
        resume_bytes=resume,
        space=check_space(
            archive_dir=archive_dir,
            install_dir=root,
            download_bytes=max(total_archive - resume, 0),
            unpacked_bytes=int(total_archive * ARCHIVE_EXPANSION),
            free=free,
        ),
    )


# --------------------------------------------------------------------------------------
# Doing it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallResult:
    """What an install left behind.

    Args:
        root: The install directory.
        bin_dir: Where the binaries are, which is what a ``PATH`` entry would name.
        marker: The marker written into ``root``.
        backends: The backends read back out of the installed directory, by the same
            function ``llamafit doctor`` uses. When this does not contain the backend
            that was asked for, ``warnings`` says so.
        archives: The archives that were unpacked.
        replaced: The install that was overwritten, when one was.
        warnings: Translated sentences about anything that is installed but not right.
    """

    root: Path
    bin_dir: Path
    marker: InstallMarker
    backends: tuple[str, ...]
    archives: tuple[Path, ...]
    replaced: InstallMarker | None = None
    warnings: tuple[str, ...] = ()


def version_text(release: Release) -> str:
    """The contents of ``VERSION.txt``, in the form ``detect.parse_version`` reads.

    The first line is exactly the shape ``llama-server --version`` prints, because that
    is the shape the detector's first pattern matches, and a build whose binary cannot
    be executed (no CUDA runtime yet, a foreign architecture) must still report its
    build number rather than nothing.
    """
    build = release.build
    lines = []
    if build is not None and release.commit:
        lines.append(f"version: {build} ({release.commit[:12]})")
    elif build is not None:
        lines.append(f"version: {build}")
    lines.append(f"build: {release.tag}")
    if release.commit:
        lines.append(f"commit: {release.commit}")
    lines.append(f"installed-by: llamafit {__version__}")
    return "\n".join(lines) + "\n"


def _clear(path: Path) -> None:
    """Remove a file or a whole directory, tolerating its absence."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            path.unlink()


@contextmanager
def _staging(root: Path) -> Iterator[Path]:
    """A scratch directory inside ``root``, removed however the block ends.

    Inside ``root`` rather than in the system temporary directory so that moving the
    finished ``bin`` into place is a rename on the same volume, which either happens or
    does not. A copy across volumes can stop half way and leave a directory that is
    part old install and part new one.
    """
    root.mkdir(parents=True, exist_ok=True)
    staging = root / STAGING_NAME
    _clear(staging)
    staging.mkdir(parents=True)
    try:
        yield staging
    finally:
        _clear(staging)


def install_release(
    plan: InstallPlan,
    *,
    fetcher: Fetcher,
    progress: ProgressCallback | None = None,
    force: bool = False,
    require_checksum: bool = True,
    now: datetime | None = None,
) -> InstallResult:
    """Download, verify, unpack and move a release into place.

    Args:
        plan: What :func:`plan_install` worked out, including the target's ownership.
        fetcher: How to read the archives.
        progress: Called as bytes arrive.
        force: Install over a directory LlamaFit did not create. The only thing that
            makes that legal, and the caller is expected to have asked first.
        require_checksum: Refuse an archive with no published checksum.
        now: The moment recorded in the marker; injected so a test can pin it.

    Returns:
        Where everything landed and what was detected there afterwards.

    Raises:
        InstallError: The directory is not ours and ``force`` was not given, a volume
            has no room, an archive fails its checksum, or the tree cannot be written.
        NetworkError: The download failed after every attempt.
    """
    if not plan.target.replaceable and not force:
        raise InstallError(
            plan.target.detail
            or _("%(path)s was not installed by LlamaFit.") % {"path": plan.target.path},
            hint=_(
                "Pass --force to replace it, or --dir to install somewhere else. "
                "LlamaFit never overwrites an installation it did not create."
            ),
        )
    short = plan.short_space
    if short:
        check = short[0]
        raise InstallError(
            _("%(path)s has %(free)s free; this install needs %(needed)s.")
            % {
                "path": check.path,
                "free": format_bytes(check.free),
                "needed": format_bytes(check.needed),
            },
            hint=_("Free some space, or pass --dir to install on another volume."),
        )

    archives: list[Path] = []
    for asset in (plan.asset, *plan.companions):
        destination = plan.archive_path.with_name(f"{plan.release.tag}-{asset.name}")
        archives.append(
            download_asset(
                asset,
                destination,
                fetcher=fetcher,
                progress=progress,
                require_checksum=require_checksum,
            )
        )

    root = plan.target.path
    replaced = plan.replaces
    with _staging(root) as staging:
        unpacked = staging / "unpacked"
        for archive in archives:
            extract_archive(archive, unpacked)
        payload = payload_dir(unpacked, plan.os_name)
        new_bin = staging / "bin"
        shutil.move(str(payload), str(new_bin))
        # Companions unpack beside the build rather than into their own tree: the CUDA
        # runtime libraries have to sit next to llama-server.exe to be found at all.
        # When the archive was flat, the payload *was* the unpacked tree and there is
        # nothing left over to move.
        if unpacked.exists():
            for extra in sorted(unpacked.rglob("*")):
                if extra.is_file():
                    shutil.copy2(extra, new_bin / extra.name)
        (new_bin / VERSION_FILE).write_text(
            version_text(plan.release), encoding="utf-8", newline="\n"
        )
        _swap_in(new_bin, bin_dir_of(root))

    backends = tuple(detect_backends(bin_dir_of(root)))
    marker = InstallMarker(
        installed_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        tag=plan.release.tag,
        build=plan.release.build,
        commit=plan.release.commit,
        asset=plan.asset.name,
        sha256=plan.asset.sha256,
        backend=plan.backend,
        os_name=plan.os_name,
        arch=plan.arch,
        backends=backends,
    )
    try:
        (root / MARKER_NAME).write_text(marker.to_json(), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise _write_failed(root / MARKER_NAME, exc) from exc

    _log.debug("installed llama.cpp %s into %s", plan.release.tag, root)
    return InstallResult(
        root=root,
        bin_dir=bin_dir_of(root),
        marker=marker,
        backends=backends,
        archives=tuple(archives),
        replaced=replaced,
        warnings=_warnings_for(plan, backends),
    )


def _warnings_for(plan: InstallPlan, backends: Sequence[str]) -> tuple[str, ...]:
    """What is installed but not what was asked for, read back from the directory itself.

    The check is deliberately made with the detector's own function rather than with a
    list of expected file names: if what was installed is not what ``llamafit doctor``
    will report tomorrow, the person should hear it now, from the command that put it
    there.
    """
    warnings: list[str] = []
    if plan.backend not in ("cpu", "metal") and plan.backend not in backends:
        warnings.append(
            _("the %(backend)s libraries are not in the installed directory.")
            % {"backend": plan.backend}
        )
    if not backends:
        warnings.append(
            _("no ggml backend libraries were installed; the build may be a static one.")
        )
    return tuple(warnings)


def _swap_in(new_bin: Path, bin_dir: Path) -> None:
    """Move the assembled directory into place, keeping the old one until it has arrived.

    Raises:
        InstallError: The move failed; the previous directory is put back first.
    """
    backup = bin_dir.with_name(bin_dir.name + ".replaced")
    _clear(backup)
    had_previous = bin_dir.exists()
    try:
        if had_previous:
            bin_dir.rename(backup)
        shutil.move(str(new_bin), str(bin_dir))
    except OSError as exc:
        if had_previous and not bin_dir.exists():
            with contextlib.suppress(OSError):
                backup.rename(bin_dir)
        raise _write_failed(bin_dir, exc) from exc
    _clear(backup)


# --------------------------------------------------------------------------------------
# The user's PATH
# --------------------------------------------------------------------------------------

PROFILE_MARK = "# added by llamafit"
"""What the line appended to a shell profile is labelled with, so it can be found again."""


@dataclass(frozen=True)
class PathChange:
    """A proposed change to the user's ``PATH``, with the way to undo it.

    Nothing here is applied by building it. The command shows ``description`` and
    ``undo`` together, in the same breath, and only then asks — the environment belongs
    to the user and the change outlives the process that made it.

    Args:
        kind: ``registry`` on Windows, ``profile`` on a POSIX shell, ``present`` when
            the directory is already there and there is nothing to do, ``unsupported``
            when the current value could not even be read.
        where: The registry value or the file that would be edited.
        directory: What would be added.
        previous: The current value, kept so the undo can name it exactly.
        description: One translated sentence saying what would change.
        undo: One translated sentence saying how to put it back.
    """

    kind: Literal["registry", "profile", "present", "unsupported"]
    where: str
    directory: Path
    previous: str | None
    description: str
    undo: str

    @property
    def needed(self) -> bool:
        """Whether applying this would actually change anything."""
        return self.kind in ("registry", "profile")


def _windows_user_path(runner: Runner) -> str | None:
    """The user's own ``Path`` value, read from the registry; ``None`` when unreadable.

    The user value, never the machine one: a per-user change needs no administrator and
    can be undone by the person who made it.
    """
    result = runner.run(["reg", "query", "HKCU\\Environment", "/v", "Path"])
    if not result.ok:
        return None
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].lower() == "path" and parts[1].upper().startswith("REG_"):
            return parts[2].strip()
    return None


def _profile_file(home: Path, env: Mapping[str, str]) -> Path:
    """Which shell profile to append to: the one the user's shell actually reads."""
    shell = env.get("SHELL", "")
    if shell.endswith("zsh"):
        return home / ".zshrc"
    if shell.endswith("bash") and (home / ".bashrc").exists():
        return home / ".bashrc"
    return home / ".profile"


def plan_path_change(
    directory: Path, *, os_name: OsName, runner: Runner, home: Path, env: Mapping[str, str]
) -> PathChange:
    r"""Work out how ``directory`` would be added to the user's ``PATH``. Changes nothing.

    On Windows this is the ``HKCU\\Environment`` value, which is the user scope: no
    administrator, and reversible by the same person. On POSIX it is a labelled line
    appended to the profile the user's shell reads, which is the only mechanism that
    survives a new terminal without a package manager.
    """
    entry = str(directory)
    if os_name == "windows":
        current = _windows_user_path(runner)
        if current is None:
            return PathChange(
                kind="unsupported",
                where="HKCU\\Environment",
                directory=directory,
                previous=None,
                description=_("Your PATH could not be read, so it will not be changed."),
                undo=_("Nothing to undo."),
            )
        if entry.lower() in [part.strip().lower() for part in current.split(";")]:
            return PathChange(
                kind="present",
                where="HKCU\\Environment",
                directory=directory,
                previous=current,
                description=_("%(path)s is already on your PATH.") % {"path": entry},
                undo=_("Nothing to undo."),
            )
        return PathChange(
            kind="registry",
            where="HKCU\\Environment",
            directory=directory,
            previous=current,
            description=_("Add %(path)s to your user PATH (HKCU\\Environment).") % {"path": entry},
            undo=_(
                'To undo: run `setx PATH "%(previous)s"`, or remove the entry in '
                "System > About > Advanced system settings > Environment Variables."
            )
            % {"previous": current},
        )
    profile = _profile_file(home, env)
    try:
        text = profile.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = ""
    if entry in text:
        return PathChange(
            kind="present",
            where=str(profile),
            directory=directory,
            previous=None,
            description=_("%(path)s is already on your PATH.") % {"path": entry},
            undo=_("Nothing to undo."),
        )
    return PathChange(
        kind="profile",
        where=str(profile),
        directory=directory,
        previous=None,
        description=_("Append a PATH line for %(path)s to %(file)s.")
        % {"path": entry, "file": profile},
        undo=_("To undo: delete the two lines marked `%(mark)s` from %(file)s.")
        % {"mark": PROFILE_MARK, "file": profile},
    )


def apply_path_change(change: PathChange, *, runner: Runner) -> None:
    """Carry out a change the user has agreed to. Never called without that agreement.

    Raises:
        InstallError: The registry could not be written or the profile could not be
            appended to. Nothing else in the install is undone: llama.cpp is installed
            and usable by its full path, and only the convenience failed.
    """
    if not change.needed:
        return
    if change.kind == "registry":
        previous = change.previous or ""
        value = f"{previous};{change.directory}" if previous else str(change.directory)
        # A value ending in a backslash would escape the closing quote reg.exe expects.
        argv = [
            "reg",
            "add",
            "HKCU\\Environment",
            "/v",
            "Path",
            "/t",
            "REG_EXPAND_SZ",
            "/d",
            value.rstrip("\\"),
            "/f",
        ]
        result = runner.run(argv, timeout=15.0)
        if not result.ok:
            raise InstallError(
                _("your PATH could not be changed: %(error)s")
                % {"error": result.error or result.stderr.strip() or _("no output")},
                hint=change.undo,
                # The command as it was issued rather than as the runner echoed it:
                # what a reader needs is the line they can try by hand.
                command=" ".join(argv),
            )
        return
    profile = Path(change.where)
    line = f'\n{PROFILE_MARK}\nexport PATH="{change.directory}:$PATH"\n'
    try:
        with profile.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        raise _write_failed(profile, exc) from exc


def uninstall_note(root: Path) -> str:
    """One sentence saying how to remove what was installed, for the closing report."""
    return _("To remove it later, delete %(path)s.") % {"path": root}


__all__ = [
    "ARCHIVE_EXPANSION",
    "MARKER_NAME",
    "MARKER_TOOL",
    "PROFILE_MARK",
    "RETRY_ATTEMPTS",
    "STAGING_NAME",
    "VERSION_FILE",
    "Body",
    "FakeFetcher",
    "Fetcher",
    "HttpFetcher",
    "InstallError",
    "InstallMarker",
    "InstallPlan",
    "InstallResult",
    "Ownership",
    "PathChange",
    "ProgressCallback",
    "SpaceCheck",
    "Target",
    "apply_path_change",
    "bin_dir_of",
    "check_space",
    "download_asset",
    "extract_archive",
    "free_bytes",
    "inspect_target",
    "install_release",
    "managed_dir",
    "part_path",
    "payload_dir",
    "plan_install",
    "plan_path_change",
    "read_marker",
    "sha256_of",
    "uninstall_note",
    "version_text",
]
