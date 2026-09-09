# LlamaFit Phase 1A — Foundation and Host Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable `llamafit` package that scans the host (OS, CPU, memory, GPUs, bandwidth, disks) and detects llama.cpp (binaries, build, backends, running servers, local GGUF files), exposed as `llamafit system` and `llamafit doctor` with Rich tables and `--json`, tested on Windows, macOS and Linux without hardware.

**Architecture:** Every OS probe runs through a `Runner` protocol so tests inject recorded command output. Probes return typed pydantic models plus a `Probe` record (ok, duration, error) and never abort the scan. The `scan()` function composes probes into a `Host`; `detect_llamacpp()` composes file-system and HTTP checks into a `LlamaCpp`. The CLI only renders.

**Tech Stack:** Python 3.10+, hatchling, pydantic 2, typer, rich, psutil, py-cpuinfo, platformdirs, httpx, pytest, ruff, mypy (strict), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-09-llamafit-design.md` (sections 3, 4, 5.1, 13.1 for `system` and `doctor`, 14, 17, 18, 19).

## Global Constraints

- Python floor 3.10; CI matrix Python 3.10 and 3.13 on Ubuntu, macOS and Windows.
- Dependencies limited to: `typer`, `rich`, `textual`, `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `platformdirs`, `psutil`, `py-cpuinfo`, `httpx`; `numpy` optional. This plan uses the subset it needs.
- Style: ruff (line length 100, isort rules), mypy strict, Google-style docstrings on every public function, type hints everywhere, no bare `except`, no `print` outside `cli/`, `tui/`, `web/`.
- Errors: every failure raises a `LlamaFitError` subclass with `message`, `hint`, `command`; rendered without a traceback unless `--verbose`.
- Probes never abort a scan; a failed probe becomes a `Probe` record with `ok=False` and an error string.
- Every assumed number carries a source label (`measured`, `estimated`, `assumed`, `unknown`).
- Package layout is `src/llamafit/...` exactly as in spec section 3.2.
- Commit after every task with a conventional-commit message ending in the attribution trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TstXuTB3ugCW29MycZGwbz
```

- Repository root is `C:\Dev\Projectos Pessoais\2026\llamaconfigurator` (the folder name is not the project name; the package and PyPI name are `llamafit`). Run all commands from the repository root. On Windows use PowerShell or Git Bash; the commands below are shell-neutral unless noted.

---

## File structure for this plan

| File | Responsibility |
|---|---|
| `pyproject.toml` | build, dependencies, entry point, ruff, mypy, pytest, coverage configuration |
| `src/llamafit/__init__.py` | `__version__`, public re-exports |
| `src/llamafit/__main__.py` | `python -m llamafit` |
| `src/llamafit/errors.py` | `LlamaFitError` hierarchy |
| `src/llamafit/units.py` | byte-size parsing and formatting |
| `src/llamafit/paths.py` | platform directories, `LLAMAFIT_HOME` override |
| `src/llamafit/logging.py` | file logger setup |
| `src/llamafit/models/__init__.py` | re-exports |
| `src/llamafit/models/host.py` | `Probe`, `Cpu`, `Memory`, `Gpu`, `Disk`, `Host` |
| `src/llamafit/models/llamacpp.py` | `RunningServer`, `LocalModel`, `LlamaCpp` |
| `src/llamafit/models/report.py` | `SystemReport` |
| `src/llamafit/hardware/runner.py` | `Runner` protocol, `CommandResult`, `SubprocessRunner`, `FakeRunner`, `probe()` |
| `src/llamafit/hardware/cpu.py` | CPU model, cores, ISA, performance cores |
| `src/llamafit/hardware/memory.py` | RAM totals, DDR type, speed, channels |
| `src/llamafit/hardware/gpu.py` | GPU probes for NVIDIA, AMD, Apple, generic |
| `src/llamafit/hardware/gputable.py` | bundled GPU table lookup |
| `src/llamafit/hardware/bandwidth.py` | RAM bandwidth measurement and estimation |
| `src/llamafit/hardware/disks.py` | free space for relevant paths |
| `src/llamafit/hardware/__init__.py` | `scan()` |
| `src/llamafit/data/gpus.json` | GPU specifications table |
| `src/llamafit/llamacpp/detect.py` | binary discovery, version, backends, local models |
| `src/llamafit/llamacpp/server.py` | running server discovery over HTTP |
| `src/llamafit/llamacpp/__init__.py` | `detect_llamacpp()` |
| `src/llamafit/services/scan.py` | `scan_system()` |
| `src/llamafit/services/doctor.py` | `diagnose()` |
| `src/llamafit/cli/app.py` | Typer application, global options, error rendering |
| `src/llamafit/cli/render.py` | Rich tables for host and llama.cpp |
| `src/llamafit/cli/system_cmd.py` | `system` command |
| `src/llamafit/cli/doctor_cmd.py` | `doctor` command |
| `tests/conftest.py` | fixtures directory helper, `FakeRunner` builders |
| `tests/fixtures/...` | recorded probe outputs |
| `tests/unit/...` | one test module per source module |
| `.github/workflows/ci.yml` | lint, type-check, test matrix |
| `README.md`, `LICENSE`, `NOTICE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.gitignore` | project documents |
| `docs/platform-support.md`, `docs/cli.md`, `docs/development.md` | user and contributor documentation for what this plan delivers |

---

### Task 1: Project scaffold and CI

**Files:**
- Create: `pyproject.toml`, `src/llamafit/__init__.py`, `src/llamafit/__main__.py`, `.gitignore`, `LICENSE`, `NOTICE`, `README.md`, `CHANGELOG.md`, `.github/workflows/ci.yml`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/unit/test_version.py`

**Interfaces:**
- Produces: `llamafit.__version__: str`; console script `llamafit` bound to `llamafit.cli.app:main` (created in Task 12; until then the entry point resolves but the module does not exist, which is fine for `pip install -e .`).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_version.py`:

```python
"""The package exposes a PEP 440 version."""

import re

import llamafit


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?", llamafit.__version__)
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "llamafit"
version = "0.1.0a1"
description = "Find, size, install and verify open-weight LLMs for llama.cpp on your own machine."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Pedro Vaz" }]
keywords = ["llama.cpp", "gguf", "llm", "local-ai", "hardware", "benchmark"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Environment :: Console",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
  "typer>=0.12",
  "rich>=13.7",
  "pydantic>=2.7",
  "platformdirs>=4.2",
  "psutil>=5.9",
  "py-cpuinfo>=9.0",
  "httpx>=0.27",
]

[project.optional-dependencies]
fast = ["numpy>=1.26"]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "ruff>=0.5",
  "mypy>=1.10",
  "types-psutil",
  "types-PyYAML",
]

[project.scripts]
llamafit = "llamafit.cli.app:main"

[project.urls]
Homepage = "https://github.com/pvaz/llamafit"
Documentation = "https://github.com/pvaz/llamafit/tree/main/docs"
Issues = "https://github.com/pvaz/llamafit/issues"
Changelog = "https://github.com/pvaz/llamafit/blob/main/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["src/llamafit"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "SIM", "RUF", "D"]
ignore = ["D203", "D213", "D107"]
pydocstyle = { convention = "google" }

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D"]

[tool.mypy]
python_version = "3.10"
strict = true
mypy_path = "src"
packages = ["llamafit"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["cpuinfo", "cpuinfo.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = ["hardware: needs the real machine, skipped in CI"]

[tool.coverage.run]
source = ["llamafit"]
branch = true

[tool.coverage.report]
fail_under = 85
show_missing = true
```

- [ ] **Step 3: Create the package files**

`src/llamafit/__init__.py`:

```python
"""LlamaFit: find, size, install and verify open-weight LLMs for llama.cpp."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llamafit")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
```

`src/llamafit/__main__.py`:

```python
"""Allow ``python -m llamafit``."""

from llamafit.cli.app import main

if __name__ == "__main__":
    main()
```

`tests/__init__.py` and `tests/unit/__init__.py`: empty files.

- [ ] **Step 4: Create project documents**

`.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
coverage.xml
htmlcov/
*.log
.DS_Store
Thumbs.db
```

`LICENSE`: the MIT license text with `Copyright (c) 2026 Pedro Vaz`.

`NOTICE`:

```
LlamaFit
Copyright (c) 2026 Pedro Vaz

This project adopts concepts, terminology and the command vocabulary of
llmfit (https://github.com/AlexsJones/llmfit), MIT License,
Copyright (c) the llmfit contributors. Its four-dimensional scoring,
fit verdicts, bandwidth speed model and confidence ladder inspired the
corresponding parts of LlamaFit, which re-implements them for llama.cpp.
The initial model catalog is seeded from llmfit's hand-curated MODELS.md.

llama.cpp (https://github.com/ggml-org/llama.cpp), MIT License, is the
runtime LlamaFit configures. LlamaFit does not bundle it.
```

`README.md` (initial; expanded in Task 13):

```markdown
# LlamaFit

Find, size, install and verify open-weight LLMs for **llama.cpp** on your own machine.

LlamaFit scans your computer, reads a curated catalog of GGUF models, computes exact memory
budgets, ranks what fits for what you need (coding, thinking, vision, tools, long context),
and turns the winner into a working `llama-server` configuration. Unlike
[llmfit](https://github.com/AlexsJones/llmfit), which tells you what fits across many runtimes,
LlamaFit goes all the way for llama.cpp: install it, download the model, write tuned launch
presets, and measure the real speed.

**Status:** alpha, phase 1 in progress. See `docs/superpowers/specs/` for the design.

## Quick start

```
pip install llamafit
llamafit system      # what this machine has
llamafit doctor      # what was detected, what failed, what would help
```

## License

MIT. See `LICENSE` and `NOTICE`.
```

`CHANGELOG.md`:

```markdown
# Changelog

All notable changes to LlamaFit are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold, host scan (`llamafit system`) and diagnostics (`llamafit doctor`).
```

- [ ] **Step 5: Create the CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    name: ${{ matrix.os }} / py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - name: Install
        run: python -m pip install --upgrade pip && pip install -e ".[dev]"
      - name: Lint
        run: ruff check . && ruff format --check .
      - name: Type-check
        run: mypy
      - name: Test
        run: pytest --cov --cov-report=xml -m "not hardware"
```

- [ ] **Step 6: Install and run the test**

Run: `python -m venv .venv` then activate it (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere), then `pip install -e ".[dev]"` and `pytest tests/unit/test_version.py -v`
Expected: PASS (version `0.1.0a1` matches the pattern).

- [ ] **Step 7: Run lint and type-check**

Run: `ruff check . && ruff format --check . && mypy`
Expected: ruff clean (run `ruff format .` first if formatting differs); mypy reports an error that `llamafit.cli.app` does not exist. That is expected until Task 12; to keep this task green, add `src/llamafit/cli/__init__.py` (empty) and a minimal `src/llamafit/cli/app.py`:

```python
"""Command-line entry point (filled in by later tasks)."""


def main() -> None:
    """Run the LlamaFit command-line interface."""
    raise SystemExit("llamafit CLI is not wired yet")
```

Re-run `mypy`. Expected: `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: project scaffold, packaging, CI and project documents"
```

(Append the attribution trailer from Global Constraints to every commit message in this plan.)

---

### Task 2: Foundation modules — errors, units, paths, logging

**Files:**
- Create: `src/llamafit/errors.py`, `src/llamafit/units.py`, `src/llamafit/paths.py`, `src/llamafit/logging.py`
- Test: `tests/unit/test_errors.py`, `tests/unit/test_units.py`, `tests/unit/test_paths.py`

**Interfaces:**
- Produces:
  - `LlamaFitError(message: str, *, hint: str | None = None, command: str | None = None)` with `.message`, `.hint`, `.command`, `.render() -> str`; subclasses `ProbeError`, `ConfigError`, `CatalogError`, `NetworkError`, `NotInstalledError`.
  - `parse_size(text: str) -> int` (bytes; accepts `8G`, `8GB`, `7.5GiB`, `512M`, `1T`, plain integers as bytes; binary units `GiB/MiB/KiB/TiB` are powers of 1024, decimal `GB/MB/KB/TB` and bare letters `G/M/K/T` are powers of 1000).
  - `format_bytes(n: int | None, *, binary: bool = True, digits: int = 1) -> str` (`"7.6 GiB"`, `"unknown"` for `None`).
  - `gib(n: int | None) -> float | None`.
  - `AppPaths` (pydantic) with `config_dir`, `data_dir`, `cache_dir`, `log_dir`, `downloads_dir` (all `Path`); `get_paths(env: Mapping[str, str] | None = None) -> AppPaths` honouring `LLAMAFIT_HOME`.
  - `setup_logging(log_dir: Path, *, verbose: bool = False) -> logging.Logger`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_errors.py`:

```python
from llamafit.errors import LlamaFitError, NotInstalledError, ProbeError


def test_render_includes_hint_and_command() -> None:
    err = ProbeError("nvidia-smi failed", hint="Install the NVIDIA driver", command="nvidia-smi -L")
    text = err.render()
    assert "nvidia-smi failed" in text
    assert "Hint: Install the NVIDIA driver" in text
    assert "Command: nvidia-smi -L" in text


def test_render_without_extras_is_just_the_message() -> None:
    assert LlamaFitError("boom").render() == "boom"


def test_subclasses_are_llamafit_errors() -> None:
    assert issubclass(NotInstalledError, LlamaFitError)
```

`tests/unit/test_units.py`:

```python
import pytest

from llamafit.units import format_bytes, gib, parse_size


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8G", 8_000_000_000),
        ("8GB", 8_000_000_000),
        ("8GiB", 8 * 1024**3),
        ("7.5GiB", int(7.5 * 1024**3)),
        ("512M", 512_000_000),
        ("512MiB", 512 * 1024**2),
        ("1T", 1_000_000_000_000),
        ("1024", 1024),
        (" 16 gb ", 16_000_000_000),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "8X", "-1G", "1.2.3G"])
def test_parse_size_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_size(bad)


def test_format_bytes_binary_and_decimal() -> None:
    assert format_bytes(8 * 1024**3) == "8.0 GiB"
    assert format_bytes(8 * 1024**3, binary=False) == "8.6 GB"
    assert format_bytes(1536 * 1024**2) == "1.5 GiB"
    assert format_bytes(900 * 1024) == "900.0 KiB"
    assert format_bytes(12) == "12 B"
    assert format_bytes(None) == "unknown"


def test_gib() -> None:
    assert gib(1024**3) == 1.0
    assert gib(None) is None
```

`tests/unit/test_paths.py`:

```python
from pathlib import Path

from llamafit.paths import get_paths


def test_llamafit_home_overrides_everything(tmp_path: Path) -> None:
    paths = get_paths({"LLAMAFIT_HOME": str(tmp_path)})
    assert paths.config_dir == tmp_path / "config"
    assert paths.data_dir == tmp_path / "data"
    assert paths.cache_dir == tmp_path / "cache"
    assert paths.log_dir == tmp_path / "log"
    assert paths.downloads_dir == tmp_path / "models"


def test_default_paths_are_absolute_and_named_llamafit() -> None:
    paths = get_paths({})
    for p in (paths.config_dir, paths.data_dir, paths.cache_dir, paths.log_dir):
        assert p.is_absolute()
        assert "llamafit" in str(p).lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_errors.py tests/unit/test_units.py tests/unit/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.errors'` (and the others).

- [ ] **Step 3: Implement `errors.py`**

```python
"""Error hierarchy. Every user-facing failure is a ``LlamaFitError``.

Interfaces catch ``LlamaFitError`` and print ``render()``; anything else is a bug
and is shown with a traceback only under ``--verbose``.
"""

from __future__ import annotations


class LlamaFitError(Exception):
    """Base class for failures LlamaFit knows how to explain.

    Args:
        message: What went wrong, in one sentence.
        hint: What the user can do about it, if anything.
        command: The exact external command that failed, if one did.
    """

    def __init__(self, message: str, *, hint: str | None = None, command: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.command = command

    def render(self) -> str:
        """Return the message with its hint and command on separate lines."""
        lines = [self.message]
        if self.command:
            lines.append(f"Command: {self.command}")
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        return "\n".join(lines)


class ProbeError(LlamaFitError):
    """A hardware or software probe could not run or returned unusable output."""


class ConfigError(LlamaFitError):
    """A configuration file or option is invalid."""


class CatalogError(LlamaFitError):
    """A catalog file is invalid or a model id is unknown."""


class NetworkError(LlamaFitError):
    """A network operation failed or the network is unavailable."""


class NotInstalledError(LlamaFitError):
    """A required external program (for example llama.cpp) is not installed."""
```

- [ ] **Step 4: Implement `units.py`**

```python
"""Byte sizes: parsing user input and formatting for display."""

from __future__ import annotations

import re

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*$", re.IGNORECASE)
_DECIMAL = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BINARY = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_BINARY_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]
_DECIMAL_UNITS = ["B", "KB", "MB", "GB", "TB"]


def parse_size(text: str) -> int:
    """Parse a human size such as ``8G``, ``7.5GiB`` or ``512MB`` into bytes.

    Bare letters and decimal units (``G``, ``GB``) are powers of 1000; binary units
    (``GiB``) are powers of 1024. A plain integer is taken as bytes.

    Raises:
        ValueError: If the text is not a size.
    """
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r}")
    number, prefix, binary = match.groups()
    prefix = prefix.lower()
    table = _BINARY if binary else _DECIMAL
    return int(float(number) * table[prefix])


def format_bytes(n: int | None, *, binary: bool = True, digits: int = 1) -> str:
    """Format a byte count for humans, ``"unknown"`` when ``n`` is ``None``."""
    if n is None:
        return "unknown"
    base = 1024 if binary else 1000
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    value = float(n)
    index = 0
    while value >= base and index < len(units) - 1:
        value /= base
        index += 1
    if index == 0:
        return f"{n} B"
    return f"{value:.{digits}f} {units[index]}"


def gib(n: int | None) -> float | None:
    """Convert bytes to GiB as a float, passing ``None`` through."""
    return None if n is None else n / 1024**3
```

- [ ] **Step 5: Implement `paths.py`**

```python
"""Where LlamaFit keeps its files, following each platform's conventions."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from platformdirs import PlatformDirs
from pydantic import BaseModel

APP_NAME = "llamafit"


class AppPaths(BaseModel):
    """Directories used by LlamaFit; all absolute, none created until needed."""

    config_dir: Path
    data_dir: Path
    cache_dir: Path
    log_dir: Path
    downloads_dir: Path


def get_paths(env: Mapping[str, str] | None = None) -> AppPaths:
    """Resolve the application directories.

    Setting ``LLAMAFIT_HOME`` puts everything under one directory, which is what
    tests and portable installations want. Otherwise the platform conventions apply
    (``%LOCALAPPDATA%`` on Windows, ``~/Library/Application Support`` on macOS, the
    XDG directories on Linux) and downloads default to ``~/llamafit/models``.
    """
    env = os.environ if env is None else env
    home = env.get("LLAMAFIT_HOME")
    if home:
        root = Path(home).expanduser().resolve()
        return AppPaths(
            config_dir=root / "config",
            data_dir=root / "data",
            cache_dir=root / "cache",
            log_dir=root / "log",
            downloads_dir=root / "models",
        )
    dirs = PlatformDirs(APP_NAME, appauthor=False)
    return AppPaths(
        config_dir=Path(dirs.user_config_dir),
        data_dir=Path(dirs.user_data_dir),
        cache_dir=Path(dirs.user_cache_dir),
        log_dir=Path(dirs.user_log_dir),
        downloads_dir=Path.home() / "llamafit" / "models",
    )
```

- [ ] **Step 6: Implement `logging.py`**

```python
"""File logging. The console stays quiet; details go to ``llamafit.log``."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "llamafit"


def setup_logging(log_dir: Path, *, verbose: bool = False) -> logging.Logger:
    """Configure the ``llamafit`` logger to write to ``log_dir/llamafit.log``.

    Returns the logger. Safe to call more than once; handlers are not duplicated.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "llamafit.log", maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, for example ``get_logger("hardware.gpu")``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
```

- [ ] **Step 7: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_errors.py tests/unit/test_units.py tests/unit/test_paths.py -v && ruff check . && mypy`
Expected: all PASS, ruff clean, mypy clean.

- [ ] **Step 8: Commit**

```bash
git add src/llamafit/errors.py src/llamafit/units.py src/llamafit/paths.py src/llamafit/logging.py tests/unit
git commit -m "feat: error hierarchy, size units, platform paths and file logging"
```

---

### Task 3: Data models — Host and LlamaCpp

**Files:**
- Create: `src/llamafit/models/__init__.py`, `src/llamafit/models/host.py`, `src/llamafit/models/llamacpp.py`, `src/llamafit/models/report.py`
- Test: `tests/unit/test_models_host.py`

**Interfaces:**
- Produces (all pydantic `BaseModel`, `frozen=False`, JSON round-trippable):
  - `Source = Literal["measured", "estimated", "assumed", "unknown"]`
  - `Probe(name: str, ok: bool, duration_ms: int, error: str | None = None)`
  - `Cpu(model: str, physical_cores: int, logical_cores: int, performance_cores: int | None = None, isa: list[str] = [])`
  - `Memory(total_bytes: int, available_bytes: int, type: str | None = None, speed_mts: int | None = None, channels: int | None = None, bandwidth_gbps: float | None = None, bandwidth_source: Source = "unknown")`
  - `Gpu(index: int, vendor: Literal["nvidia","amd","apple","intel","other"], name: str, vram_total_bytes: int | None = None, vram_used_bytes: int | None = None, bandwidth_gbps: float | None = None, compute_tflops_fp16: float | None = None, backend_hint: Literal["cuda","hip","metal","vulkan","sycl","cpu"] = "cpu", driver: str | None = None)` with property `vram_free_bytes -> int | None`
  - `Disk(path: str, free_bytes: int, total_bytes: int)`
  - `Host(os: Literal["windows","macos","linux"], os_version: str, arch: Literal["x86_64","arm64","other"], cpu: Cpu, memory: Memory, gpus: list[Gpu] = [], unified_memory: bool = False, disks: list[Disk] = [], probes: list[Probe] = [], scanned_at: datetime)` with properties `primary_gpu -> Gpu | None` (largest `vram_total_bytes`, else first) and `vram_available_bytes -> int | None`
  - `RunningServer(url: str, model: str | None = None, n_ctx: int | None = None, build: str | None = None)`
  - `LocalModel(path: str, bytes: int, sha256: str | None = None, catalog_id: str | None = None)`
  - `LlamaCpp(installed: bool, path: str | None = None, build: int | None = None, commit: str | None = None, backends: list[str] = [], running_servers: list[RunningServer] = [], local_models: list[LocalModel] = [], problems: list[str] = [], probes: list[Probe] = [])`
  - `SystemReport(host: Host, llamacpp: LlamaCpp, version: str)`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_host.py`:

```python
from datetime import datetime, timezone

from llamafit.models import Cpu, Gpu, Host, LlamaCpp, Memory, Probe, SystemReport


def make_host(**overrides: object) -> Host:
    base: dict[str, object] = {
        "os": "windows",
        "os_version": "11 (10.0.26200)",
        "arch": "x86_64",
        "cpu": Cpu(model="Intel i9-14900KF", physical_cores=24, logical_cores=32,
                   performance_cores=8, isa=["avx2", "avx512"]),
        "memory": Memory(total_bytes=128 * 1024**3, available_bytes=100 * 1024**3),
        "gpus": [
            Gpu(index=0, vendor="nvidia", name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * 1024**2, vram_used_bytes=550 * 1024**2,
                backend_hint="cuda"),
        ],
        "scanned_at": datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return Host(**base)  # type: ignore[arg-type]


def test_primary_gpu_is_largest() -> None:
    small = Gpu(index=1, vendor="intel", name="UHD", vram_total_bytes=1 * 1024**3)
    host = make_host(gpus=[small, *make_host().gpus])
    assert host.primary_gpu is not None
    assert host.primary_gpu.name == "NVIDIA GeForce RTX 4060"


def test_vram_available_is_total_minus_used() -> None:
    host = make_host()
    assert host.vram_available_bytes == (8188 - 550) * 1024**2


def test_no_gpu_means_no_vram() -> None:
    host = make_host(gpus=[])
    assert host.primary_gpu is None
    assert host.vram_available_bytes is None


def test_report_round_trips_through_json() -> None:
    report = SystemReport(host=make_host(), llamacpp=LlamaCpp(installed=False), version="0.1.0a1")
    again = SystemReport.model_validate_json(report.model_dump_json())
    assert again == report


def test_probe_defaults() -> None:
    probe = Probe(name="nvidia-smi", ok=False, duration_ms=3, error="not found")
    assert probe.error == "not found"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_models_host.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.models'`.

- [ ] **Step 3: Implement `models/host.py`**

```python
"""What LlamaFit knows about the machine it runs on."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["measured", "estimated", "assumed", "unknown"]
Vendor = Literal["nvidia", "amd", "apple", "intel", "other"]
Backend = Literal["cuda", "hip", "metal", "vulkan", "sycl", "cpu"]
OsName = Literal["windows", "macos", "linux"]
Arch = Literal["x86_64", "arm64", "other"]


class Probe(BaseModel):
    """Outcome of one detection step, kept so ``doctor`` can show what happened."""

    name: str
    ok: bool
    duration_ms: int
    error: str | None = None


class Cpu(BaseModel):
    """Processor facts relevant to llama.cpp thread choice and CPU inference."""

    model: str
    physical_cores: int
    logical_cores: int
    performance_cores: int | None = None
    isa: list[str] = Field(default_factory=list)


class Memory(BaseModel):
    """System memory; bandwidth carries the label of how it was obtained."""

    total_bytes: int
    available_bytes: int
    type: str | None = None
    speed_mts: int | None = None
    channels: int | None = None
    bandwidth_gbps: float | None = None
    bandwidth_source: Source = "unknown"


class Gpu(BaseModel):
    """One graphics device. Unknown values stay ``None``; they are never guessed here."""

    index: int
    vendor: Vendor
    name: str
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    bandwidth_gbps: float | None = None
    compute_tflops_fp16: float | None = None
    backend_hint: Backend = "cpu"
    driver: str | None = None

    @property
    def vram_free_bytes(self) -> int | None:
        """VRAM not in use at scan time, or ``None`` when either figure is unknown."""
        if self.vram_total_bytes is None or self.vram_used_bytes is None:
            return None
        return max(self.vram_total_bytes - self.vram_used_bytes, 0)


class Disk(BaseModel):
    """Free space at a path that matters (downloads, llama.cpp directory)."""

    path: str
    free_bytes: int
    total_bytes: int


class Host(BaseModel):
    """The scanned machine."""

    os: OsName
    os_version: str
    arch: Arch
    cpu: Cpu
    memory: Memory
    gpus: list[Gpu] = Field(default_factory=list)
    unified_memory: bool = False
    disks: list[Disk] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)
    scanned_at: datetime

    @property
    def primary_gpu(self) -> Gpu | None:
        """The GPU with the most VRAM, or the first one when sizes are unknown."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_total_bytes or -1)

    @property
    def vram_available_bytes(self) -> int | None:
        """Free VRAM on the primary GPU, ``None`` without a GPU or without figures."""
        gpu = self.primary_gpu
        return None if gpu is None else gpu.vram_free_bytes
```

- [ ] **Step 4: Implement `models/llamacpp.py` and `models/report.py`**

`models/llamacpp.py`:

```python
"""What LlamaFit knows about the llama.cpp installation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llamafit.models.host import Probe


class RunningServer(BaseModel):
    """A ``llama-server`` answering on a local port."""

    url: str
    model: str | None = None
    n_ctx: int | None = None
    build: str | None = None


class LocalModel(BaseModel):
    """A GGUF file found on disk."""

    path: str
    bytes: int
    sha256: str | None = None
    catalog_id: str | None = None


class LlamaCpp(BaseModel):
    """Installation status; ``problems`` explains anything that limits use."""

    installed: bool
    path: str | None = None
    build: int | None = None
    commit: str | None = None
    backends: list[str] = Field(default_factory=list)
    running_servers: list[RunningServer] = Field(default_factory=list)
    local_models: list[LocalModel] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)
```

`models/report.py`:

```python
"""Top-level result of ``llamafit system``."""

from __future__ import annotations

from pydantic import BaseModel

from llamafit.models.host import Host
from llamafit.models.llamacpp import LlamaCpp


class SystemReport(BaseModel):
    """Host plus llama.cpp status, stamped with the LlamaFit version that produced it."""

    host: Host
    llamacpp: LlamaCpp
    version: str
```

`models/__init__.py`:

```python
"""Typed data shared by every layer of LlamaFit."""

from llamafit.models.host import Arch, Backend, Cpu, Disk, Gpu, Host, Memory, OsName, Probe, Source
from llamafit.models.llamacpp import LlamaCpp, LocalModel, RunningServer
from llamafit.models.report import SystemReport

__all__ = [
    "Arch",
    "Backend",
    "Cpu",
    "Disk",
    "Gpu",
    "Host",
    "LlamaCpp",
    "LocalModel",
    "Memory",
    "OsName",
    "Probe",
    "RunningServer",
    "Source",
    "SystemReport",
]
```

- [ ] **Step 5: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_models_host.py -v && ruff check . && mypy`
Expected: PASS, clean, clean.

- [ ] **Step 6: Commit**

```bash
git add src/llamafit/models tests/unit/test_models_host.py
git commit -m "feat: host, llama.cpp and report data models"
```

---

### Task 4: Command runner and probe helper

**Files:**
- Create: `src/llamafit/hardware/__init__.py` (empty for now), `src/llamafit/hardware/runner.py`
- Test: `tests/unit/test_runner.py`

**Interfaces:**
- Produces:
  - `CommandResult(argv: list[str], returncode: int | None, stdout: str, stderr: str, duration_ms: int, error: str | None = None)` with property `ok -> bool` (`error is None and returncode == 0`).
  - `class Runner(Protocol): def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult`
  - `SubprocessRunner()` — real subprocesses, never raises: a missing program or a timeout becomes `error`.
  - `FakeRunner(responses: Mapping[str, str | CommandResult])` — keyed by the program name (`argv[0]`) or by the full command joined with spaces; unknown commands return `error="not found"`. Records `calls: list[list[str]]`.
  - `probe(name: str, runner: Runner, argv: Sequence[str], parse: Callable[[str], T], *, timeout: float = 10.0) -> tuple[T | None, Probe]` — runs, parses stdout, converts any exception into `Probe(ok=False, error=...)`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_runner.py`:

```python
import sys

from llamafit.hardware.runner import CommandResult, FakeRunner, SubprocessRunner, probe


def test_subprocess_runner_runs_python() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "print('hi')"])
    assert result.ok
    assert result.stdout.strip() == "hi"
    assert result.duration_ms >= 0


def test_subprocess_runner_missing_program_is_an_error_not_an_exception() -> None:
    result = SubprocessRunner().run(["definitely-not-a-program-xyz"])
    assert not result.ok
    assert result.error is not None
    assert result.returncode is None


def test_subprocess_runner_timeout() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
    assert not result.ok
    assert "timed out" in (result.error or "")


def test_fake_runner_matches_program_or_full_command() -> None:
    runner = FakeRunner({"nvidia-smi": "GPU 0", "rocm-smi --json": '{"ok": true}'})
    assert runner.run(["nvidia-smi", "-L"]).stdout == "GPU 0"
    assert runner.run(["rocm-smi", "--json"]).stdout == '{"ok": true}'
    assert not runner.run(["missing"]).ok
    assert runner.calls == [["nvidia-smi", "-L"], ["rocm-smi", "--json"], ["missing"]]


def test_probe_parses_and_records_success() -> None:
    runner = FakeRunner({"tool": "42"})
    value, rec = probe("tool", runner, ["tool"], int)
    assert value == 42
    assert rec.ok and rec.name == "tool" and rec.error is None


def test_probe_turns_parse_failure_into_probe_error() -> None:
    runner = FakeRunner({"tool": "forty-two"})
    value, rec = probe("tool", runner, ["tool"], int)
    assert value is None
    assert not rec.ok
    assert "invalid literal" in (rec.error or "")


def test_probe_reports_command_failure() -> None:
    runner = FakeRunner({"tool": CommandResult(argv=["tool"], returncode=1, stdout="", stderr="bad", duration_ms=1)})
    value, rec = probe("tool", runner, ["tool"], int)
    assert value is None
    assert rec.error == "exit code 1: bad"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.hardware.runner'`.

- [ ] **Step 3: Implement `hardware/runner.py`**

```python
"""Run external programs in a way tests can replace.

Every probe goes through a ``Runner`` so a test can feed recorded output from a real
machine instead of executing anything. ``SubprocessRunner`` never raises: a missing
program, a crash or a timeout all become a ``CommandResult`` with ``error`` set.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from llamafit.models.host import Probe

T = TypeVar("T")


@dataclass
class CommandResult:
    """Everything a probe needs to know about one command execution."""

    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the command ran and exited with status 0."""
        return self.error is None and self.returncode == 0


class Runner(Protocol):
    """Something that can run a command line and report what happened."""

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run ``argv`` and return its result without raising."""
        ...


class SubprocessRunner:
    """Runs real subprocesses with a timeout, capturing text output."""

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run ``argv``; a missing program or a timeout is reported, not raised."""
        args = list(argv)
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, check=False,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            return CommandResult(args, None, "", "", _elapsed_ms(start), error=f"{args[0]}: not found")
        except subprocess.TimeoutExpired:
            return CommandResult(args, None, "", "", _elapsed_ms(start), error=f"{args[0]}: timed out after {timeout}s")
        except OSError as exc:
            return CommandResult(args, None, "", "", _elapsed_ms(start), error=f"{args[0]}: {exc}")
        return CommandResult(args, completed.returncode, completed.stdout, completed.stderr, _elapsed_ms(start))


@dataclass
class FakeRunner:
    """Returns canned output; keys are a program name or a full command line."""

    responses: Mapping[str, str | CommandResult]
    calls: list[list[str]] = field(default_factory=list)

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Look up the command by full line first, then by program name."""
        args = list(argv)
        self.calls.append(args)
        response = self.responses.get(" ".join(args))
        if response is None:
            response = self.responses.get(args[0])
        if response is None:
            return CommandResult(args, None, "", "", 0, error=f"{args[0]}: not found")
        if isinstance(response, CommandResult):
            return response
        return CommandResult(args, 0, response, "", 0)


def probe(
    name: str,
    runner: Runner,
    argv: Sequence[str],
    parse: Callable[[str], T],
    *,
    timeout: float = 10.0,
) -> tuple[T | None, Probe]:
    """Run a command and parse its stdout, turning every failure into a ``Probe`` record.

    Returns:
        The parsed value (or ``None``) and the probe record for ``doctor``.
    """
    result = runner.run(argv, timeout=timeout)
    if result.error is not None:
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=result.error)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms,
                           error=f"exit code {result.returncode}: {detail}")
    try:
        value = parse(result.stdout)
    except Exception as exc:  # noqa: BLE001 - any parse failure must become a probe record
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=str(exc))
    return value, Probe(name=name, ok=True, duration_ms=result.duration_ms)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
```

`hardware/__init__.py`: a one-line docstring for now (`"""Host detection."""`); `scan()` is added in Task 10.

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_runner.py -v && ruff check . && mypy`
Expected: PASS (the timeout test takes about 0.2 s), clean, clean.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/hardware tests/unit/test_runner.py
git commit -m "feat: replaceable command runner and probe helper"
```

<!-- CONTINUE -->
