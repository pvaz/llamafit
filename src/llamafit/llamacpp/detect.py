# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Find llama.cpp on this machine and describe the installation."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from llamafit.hardware.runner import Runner, probe
from llamafit.i18n import _
from llamafit.models.host import OsName, Probe
from llamafit.models.llamacpp import LlamaCpp, LocalModel

_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_BACKEND_ORDER = ["cuda", "hip", "metal", "vulkan", "sycl", "rpc", "cpu"]
# Three shapes, most specific first, because a build number is a number and the loose
# pattern would happily read one out of a release name.
#
#   version: 0.4.0-dev (build 10867, commit f3f1a8f27)   what llama-server prints today
#   version: 10867 (f3f1a8f27)                           what it printed before that
#   b10867                                               a release tag, and VERSION.txt
#
# The first was missing, which is why a build number was only ever known on a machine
# whose llama.cpp LlamaFit installed itself: the installer writes VERSION.txt, and the
# third pattern reads it back.
_VERSION_PATTERNS = [
    re.compile(r"\bbuild\s+(\d+)\s*,\s*commit\s+([0-9a-f]{6,})", re.IGNORECASE),
    re.compile(r"(?:version|build)\s*:?\s*(\d+)\s*\(([0-9a-f]{6,})\)", re.IGNORECASE),
    re.compile(r"\bb(\d{4,})\b"),
]


def exe_name(name: str, os_name: OsName) -> str:
    """``llama-server.exe`` on Windows, ``llama-server`` elsewhere."""
    return f"{name}.exe" if os_name == "windows" else name


def well_known_dirs(os_name: OsName, home: Path) -> list[Path]:
    """Directories where people commonly put llama.cpp, most likely first."""
    common = [
        home / ".llamafit" / "llama.cpp" / "bin",
        home / "llama.cpp" / "bin",
        home / "llama.cpp",
    ]
    if os_name == "windows":
        return [
            *common,
            Path("C:/llama.cpp/bin"),
            Path("D:/llama.cpp/bin"),
            Path("C:/llama.cpp"),
            Path("D:/llama.cpp"),
        ]
    if os_name == "macos":
        return [*common, Path("/opt/homebrew/bin"), Path("/usr/local/bin")]
    return [*common, Path("/usr/local/bin"), Path("/opt/llama.cpp/bin"), Path("/usr/bin")]


def find_llamacpp_dir(
    *,
    env: Mapping[str, str],
    path_dirs: Iterable[Path],
    well_known: Iterable[Path],
    os_name: OsName,
) -> Path | None:
    """Locate the directory that holds ``llama-server``.

    Precedence: ``LLAMA_CPP_PATH`` (a bin directory or an install root), then the
    ``PATH`` entries, then the well-known directories.
    """
    server = exe_name("llama-server", os_name)
    candidates: list[Path] = []
    configured = env.get("LLAMA_CPP_PATH")
    if configured:
        root = Path(configured).expanduser()
        candidates.extend([root, root / "bin"])
    candidates.extend(path_dirs)
    candidates.extend(well_known)
    for directory in candidates:
        if (directory / server).is_file():
            return directory
    return None


def parse_version(out: str) -> tuple[int | None, str | None]:
    """Extract the build number and commit hash from ``llama-server --version`` output.

    Args:
        out: Whatever the binary printed, on either stream.

    Returns:
        The build number and the commit, either of which may be ``None`` when the text
        does not carry it. Used directly on ``VERSION.txt``, where finding nothing is an
        ordinary answer; the probe uses :func:`_parsed_version` instead, which refuses.
    """
    for pattern in _VERSION_PATTERNS[:-1]:
        match = pattern.search(out)
        if match:
            return int(match.group(1)), match.group(2)
    match = _VERSION_PATTERNS[-1].search(out)
    if match:
        return int(match.group(1)), None
    return None, None


def _parsed_version(out: str) -> tuple[int, str | None]:
    """The same, for a probe, refusing rather than returning nothing.

    A probe whose parser hands back a hollow value is recorded ``ok``, and ``doctor`` then
    prints ``llama-server --version  ok`` about a run it learned nothing from. Raising is
    how a parser says "this ran and told me nothing", which is the honest record.

    Args:
        out: Whatever the binary printed, on either stream.

    Returns:
        The build number and the commit, when the text carries a build number.

    Raises:
        ValueError: When no pattern matched, naming the first line the binary printed so
            a reader can see what shape it is in.
    """
    build, commit = parse_version(out)
    if build is None:
        first = next((line.strip() for line in out.splitlines() if line.strip()), "")
        raise ValueError(
            _("no build number in the version output: %(text)s") % {"text": first or _("no output")}
        )
    return build, commit


def detect_backends(bin_dir: Path) -> list[str]:
    """Infer compiled backends from the ``ggml-*`` shared libraries next to the binaries.

    A directory that cannot be listed (it vanished, or the user may not read it) yields an
    empty list, which the caller reports as the usual "no ggml backend libraries" problem.
    """
    try:
        names = {p.name.lower() for p in bin_dir.iterdir() if p.is_file()}
    except OSError:  # unlistable directory: no backends is the honest answer
        return []
    found: set[str] = set()
    for name in names:
        if not name.startswith(("ggml-", "libggml-")):
            continue
        stem = name.split("ggml-", 1)[1]
        for backend in _BACKEND_ORDER:
            if stem.startswith(backend):
                found.add(backend)
    return [b for b in _BACKEND_ORDER if b in found]


def find_local_models(dirs: Iterable[Path], *, max_depth: int = 3) -> list[LocalModel]:
    """List GGUF files; split models appear once, under their first shard, with total bytes.

    An incomplete split model, one whose first shard is missing, is not listed.

    A model directory is exactly where files appear and disappear while it is being read,
    so a file whose size cannot be read is skipped and a directory walk that fails part
    way through keeps whatever it found.
    """
    models: dict[Path, int] = {}
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*.gguf"):
                if len(path.relative_to(root).parts) > max_depth:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:  # the file went away or cannot be read; skip it
                    continue
                match = _SHARD_RE.match(path.name)
                if match and match.group("index") != "00001":
                    stem, total = match.group("stem"), match.group("total")
                    first = path.with_name(f"{stem}-00001-of-{total}.gguf")
                    models[first] = models.get(first, 0) + size
                    continue
                models[path] = models.get(path, 0) + size
        except OSError:  # the directory walk itself failed; keep what was found
            continue
    return [LocalModel(path=str(p), bytes=b) for p, b in sorted(models.items()) if p.exists()]


def detect_install(
    runner: Runner,
    os_name: OsName,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    path_dirs: Iterable[Path] | None = None,
    well_known: Iterable[Path] | None = None,
) -> tuple[LlamaCpp, list[Probe]]:
    """Describe the llama.cpp installation, or explain why none was found.

    Args:
        runner: Executes ``llama-server --version``; a ``FakeRunner`` in tests.
        os_name: Selects the executable suffix and the platform's well-known directories.
        env: Environment mapping to read ``LLAMA_CPP_PATH`` and ``PATH`` from; defaults to
            ``os.environ``.
        home: The user's home directory, used to build the well-known directories; defaults
            to ``Path.home()``.
        path_dirs: Directories from ``PATH`` to search; defaults to splitting ``env["PATH"]``.
        well_known: Directories to treat as the well-known install locations. When ``None``
            (the default) this is ``well_known_dirs(os_name, home)``, matching what a real
            scan would search. Callers such as tests can pass ``well_known=[]`` so results do
            not depend on what happens to be installed on the machine running them.
    """
    env = os.environ if env is None else env
    home = home or Path.home()
    if path_dirs is None:
        path_dirs = [Path(p) for p in env.get("PATH", "").split(os.pathsep) if p]
    if well_known is None:
        well_known = well_known_dirs(os_name, home)
    probes: list[Probe] = []
    bin_dir = find_llamacpp_dir(
        env=env, path_dirs=path_dirs, well_known=well_known, os_name=os_name
    )
    if bin_dir is None:
        return LlamaCpp(
            installed=False,
            problems=[
                _(
                    "llama.cpp not found: no llama-server on PATH, "
                    "in LLAMA_CPP_PATH or in the usual directories"
                )
            ],
        ), probes

    server = bin_dir / exe_name("llama-server", os_name)
    version, rec = probe(
        "llama-server --version",
        runner,
        [str(server), "--version"],
        _parsed_version,
        include_stderr=True,
    )
    probes.append(rec)
    build, commit = version if version else (None, None)
    if build is None:
        try:
            text = (bin_dir / "VERSION.txt").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # absent, unreadable or not text: build unknown
            text = ""
        # Named rather than thrown away as `_`: this module imports the translator
        # under that name, and rebinding it inside a function is a string that is not
        # callable at the next translated call, with nothing to warn you first.
        build, _commit = parse_version(text)

    backends = detect_backends(bin_dir)
    problems: list[str] = []
    if not backends:
        problems.append(
            _("no ggml backend libraries found next to llama-server; the build may be static")
        )
    cache_dir = Path(env.get("LLAMA_CACHE", "")) if env.get("LLAMA_CACHE") else None
    model_dirs = [bin_dir.parent / "models", cache_dir]
    local_models = find_local_models([d for d in model_dirs if d])
    return LlamaCpp(
        installed=True,
        path=str(bin_dir),
        build=build,
        commit=commit,
        backends=backends,
        local_models=local_models,
        problems=problems,
    ), probes
