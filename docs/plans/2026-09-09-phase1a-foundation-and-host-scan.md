# LlamaFit Phase 1A — Foundation and Host Scan Implementation Plan

> Implementation plan: one task per section, each with its files, interfaces, tests, steps and commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable `llamafit` package that scans the host (OS, CPU, memory, GPUs, bandwidth, disks) and detects llama.cpp (binaries, build, backends, running servers, local GGUF files), exposed as `llamafit system` and `llamafit doctor` with Rich tables and `--json`, tested on Windows, macOS and Linux without hardware.

**Architecture:** Every OS probe runs through a `Runner` protocol so tests inject recorded command output. Probes return typed pydantic models plus a `Probe` record (ok, duration, error) and never abort the scan. The `scan()` function composes probes into a `Host`; `detect_llamacpp()` composes file-system and HTTP checks into a `LlamaCpp`. The CLI only renders.

**Tech Stack:** Python 3.10+, hatchling, pydantic 2, typer, rich, psutil, py-cpuinfo, platformdirs, httpx, pytest, ruff, mypy (strict), GitHub Actions.

**Spec:** `docs/specs/2026-09-09-llamafit-design.md` (sections 3, 4, 5.1, 13.1 for `system` and `doctor`, 14, 17, 18, 19).

## Global Constraints

- Python floor 3.10; CI matrix Python 3.10 and 3.13 on Ubuntu, macOS and Windows.
- Dependencies limited to: `typer`, `rich`, `textual`, `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `platformdirs`, `psutil`, `py-cpuinfo`, `httpx`; `numpy` optional. This plan uses the subset it needs.
- Style: ruff (line length 100, isort rules), mypy strict, Google-style docstrings on every public function, type hints everywhere, no bare `except`, no `print` outside `cli/`, `tui/`, `web/`.
- Errors: every failure raises a `LlamaFitError` subclass with `message`, `hint`, `command`; rendered without a traceback unless `--verbose`.
- Probes never abort a scan; a failed probe becomes a `Probe` record with `ok=False` and an error string.
- Every assumed number carries a source label (`measured`, `estimated`, `assumed`, `unknown`).
- Package layout is `src/llamafit/...` exactly as in spec section 3.2.
- Commit after every task with a conventional-commit message.

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
authors = [{ name = "Paulo Vaz" }]
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

- [ ] **Step 4: Check the project documents that already exist**

The repository already carries `.gitignore`, `LICENSE`, `NOTICE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `ROADMAP.md`, the `.github` templates and the `docs/` set, all written before implementation started. Do not replace them. In this step only:

1. Confirm `.gitignore` ignores `.venv/`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/` and `*.log` (it does; add anything the tools below create that it misses).
2. Change the **Status** blockquote in `README.md` to say that phase 1A (host scan and diagnostics) is in progress and link the plan file.
3. Under `## [Unreleased]` → `### Added` in `CHANGELOG.md`, add the line `- Project scaffold: packaging, CI matrix, lint, type-check and test configuration.`

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

---

### Task 5: CPU and memory detection

**Files:**
- Create: `src/llamafit/hardware/cpu.py`, `src/llamafit/hardware/memory.py`
- Test: `tests/unit/test_cpu.py`, `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: `Runner`, `probe()` (Task 4); `Cpu`, `Memory`, `Probe`, `OsName` (Task 3).
- Produces:
  - `detect_cpu(runner: Runner, os_name: OsName, *, cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None) -> tuple[Cpu, list[Probe]]`
  - `isa_from_flags(flags: Iterable[str]) -> list[str]`
  - `performance_cores_for(model: str, physical_cores: int) -> int | None`
  - `detect_memory(runner: Runner, os_name: OsName, *, vm_provider: Callable[[], tuple[int, int]] | None = None) -> tuple[Memory, list[Probe]]` — `vm_provider` returns `(total_bytes, available_bytes)`.
  - `theoretical_bandwidth_gbps(speed_mts: int, channels: int) -> float` = `speed_mts × 8 × channels / 1000`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_cpu.py`:

```python
from llamafit.hardware.cpu import detect_cpu, isa_from_flags, performance_cores_for
from llamafit.hardware.runner import FakeRunner


def test_isa_from_flags_maps_known_flags() -> None:
    flags = ["sse4_2", "avx2", "avx512f", "avx512_vnni", "amx_tile", "fma"]
    assert isa_from_flags(flags) == ["avx2", "avx512", "avx512_vnni", "amx"]


def test_isa_from_flags_arm() -> None:
    assert isa_from_flags(["asimd", "sve"]) == ["neon", "sve"]


def test_performance_cores_table() -> None:
    assert performance_cores_for("Intel(R) Core(TM) i9-14900KF", 24) == 8
    assert performance_cores_for("13th Gen Intel(R) Core(TM) i5-13600K", 14) == 6
    assert performance_cores_for("Intel(R) Core(TM) Ultra 9 285K", 24) == 8
    assert performance_cores_for("AMD Ryzen 9 7950X 16-Core Processor", 16) == 16
    assert performance_cores_for("Intel(R) Core(TM) i7-9700K", 8) == 8


def test_detect_cpu_uses_provider_and_psutil_counts() -> None:
    provider = lambda: {"brand_raw": "Intel(R) Core(TM) i9-14900KF", "flags": ["avx2", "avx512f"]}
    cpu, probes = detect_cpu(FakeRunner({}), "windows", cpuinfo_provider=provider)
    assert cpu.model == "Intel(R) Core(TM) i9-14900KF"
    assert cpu.isa == ["avx2", "avx512"]
    assert cpu.physical_cores >= 1 and cpu.logical_cores >= cpu.physical_cores
    assert cpu.performance_cores == 8
    assert [p.name for p in probes] == ["cpuinfo"]
    assert probes[0].ok


def test_detect_cpu_apple_perf_cores_from_sysctl() -> None:
    provider = lambda: {"brand_raw": "Apple M3 Max", "flags": ["asimd"]}
    runner = FakeRunner({"sysctl -n hw.perflevel0.physicalcpu": "12\n"})
    cpu, probes = detect_cpu(runner, "macos", cpuinfo_provider=provider)
    assert cpu.performance_cores == 12
    assert [p.name for p in probes] == ["cpuinfo", "sysctl-perflevel"]


def test_detect_cpu_survives_provider_failure() -> None:
    def broken() -> dict[str, object]:
        raise RuntimeError("no cpuinfo")

    cpu, probes = detect_cpu(FakeRunner({}), "linux", cpuinfo_provider=broken)
    assert cpu.model == "unknown"
    assert not probes[0].ok and "no cpuinfo" in (probes[0].error or "")
```

`tests/unit/test_memory.py`:

```python
import json

from llamafit.hardware.memory import detect_memory, theoretical_bandwidth_gbps
from llamafit.hardware.runner import FakeRunner

WIN_MODULES = json.dumps([
    {"SMBIOSMemoryType": 34, "Speed": 4800, "ConfiguredClockSpeed": 4200, "Capacity": 68719476736},
    {"SMBIOSMemoryType": 34, "Speed": 4800, "ConfiguredClockSpeed": 4200, "Capacity": 68719476736},
])

MAC_MEMORY = json.dumps({"SPMemoryDataType": [{"SPMemoryDataType": "64 GB", "dimm_type": "LPDDR5", "dimm_manufacturer": "Apple"}]})


def vm() -> tuple[int, int]:
    return 128 * 1024**3, 100 * 1024**3


def test_theoretical_bandwidth() -> None:
    assert theoretical_bandwidth_gbps(4200, 2) == 67.2


def test_windows_modules_give_type_speed_channels_and_estimate() -> None:
    runner = FakeRunner({"powershell": WIN_MODULES})
    memory, probes = detect_memory(runner, "windows", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type == "DDR5"
    assert memory.speed_mts == 4200
    assert memory.channels == 2
    assert memory.bandwidth_gbps == 67.2
    assert memory.bandwidth_source == "estimated"
    assert probes[-1].name == "memory-modules" and probes[-1].ok


def test_macos_memory_type_without_speed() -> None:
    runner = FakeRunner({"system_profiler": MAC_MEMORY})
    memory, _ = detect_memory(runner, "macos", vm_provider=vm)
    assert memory.type == "LPDDR5"
    assert memory.speed_mts is None
    assert memory.bandwidth_source == "unknown"


def test_failed_module_probe_keeps_totals() -> None:
    memory, probes = detect_memory(FakeRunner({}), "linux", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type is None
    assert not probes[-1].ok
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_cpu.py tests/unit/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `hardware/cpu.py`**

```python
"""CPU facts: model, cores, instruction sets, performance-core count."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import psutil

from llamafit.hardware.runner import Runner, probe
from llamafit.models.host import Cpu, OsName, Probe

_ISA_MAP: dict[str, str] = {
    "avx2": "avx2",
    "avx512f": "avx512",
    "avx512_vnni": "avx512_vnni",
    "amx_tile": "amx",
    "asimd": "neon",
    "neon": "neon",
    "sve": "sve",
}
_ISA_ORDER = ["avx2", "avx512", "avx512_vnni", "amx", "neon", "sve"]

# Intel hybrid parts: (regex on the model name, performance cores). Everything else is
# treated as homogeneous, so performance_cores == physical_cores.
_HYBRID_TABLE: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"i9-1[2-4]9\d\d"), 8),
    (re.compile(r"i7-1[2-4]7\d\d"), 8),
    (re.compile(r"i5-1[2-4]6\d\d"), 6),
    (re.compile(r"i5-1[34]5\d\d"), 6),
    (re.compile(r"i5-1[2-4]4\d\d"), 6),
    (re.compile(r"Ultra 9 2[89]\d"), 8),
    (re.compile(r"Ultra 7 2[5-7]\d"), 8),
    (re.compile(r"Ultra 5 2[2-4]\d"), 6),
]


def isa_from_flags(flags: Iterable[str]) -> list[str]:
    """Reduce raw CPU flags to the instruction sets llama.cpp builds care about."""
    found = {_ISA_MAP[f] for f in flags if f in _ISA_MAP}
    return [name for name in _ISA_ORDER if name in found]


def performance_cores_for(model: str, physical_cores: int) -> int | None:
    """Performance cores for hybrid Intel parts from a table; all cores otherwise."""
    if "Intel" in model and re.search(r"i[3579]-1[2-4]\d{3}|Ultra [579] 2\d\d", model):
        for pattern, cores in _HYBRID_TABLE:
            if pattern.search(model):
                return min(cores, physical_cores)
        return None
    return physical_cores


def _default_cpuinfo() -> Mapping[str, Any]:
    import cpuinfo  # imported lazily: it takes about a second to load

    info: Mapping[str, Any] = cpuinfo.get_cpu_info()
    return info


def detect_cpu(
    runner: Runner,
    os_name: OsName,
    *,
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[Cpu, list[Probe]]:
    """Detect the CPU. Never raises; failures are reported in the probes."""
    provider = cpuinfo_provider or _default_cpuinfo
    probes: list[Probe] = []
    start = time.perf_counter()
    model = "unknown"
    flags: list[str] = []
    try:
        info = provider()
        model = str(info.get("brand_raw") or info.get("brand") or "unknown").strip()
        flags = [str(f) for f in info.get("flags", [])]
        probes.append(Probe(name="cpuinfo", ok=True, duration_ms=_ms(start)))
    except Exception as exc:  # noqa: BLE001 - a broken cpuinfo must not stop the scan
        probes.append(Probe(name="cpuinfo", ok=False, duration_ms=_ms(start), error=str(exc)))

    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    perf = performance_cores_for(model, physical)
    if os_name == "macos":
        value, rec = probe("sysctl-perflevel", runner, ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                           lambda out: int(out.strip()))
        probes.append(rec)
        if value:
            perf = value
    return Cpu(model=model, physical_cores=physical, logical_cores=logical,
               performance_cores=perf, isa=isa_from_flags(flags)), probes


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
```

- [ ] **Step 4: Implement `hardware/memory.py`**

```python
"""System memory: totals from psutil, module facts from the OS, bandwidth estimate."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import psutil

from llamafit.hardware.runner import Runner, probe
from llamafit.models.host import Memory, OsName, Probe

_SMBIOS_TYPES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}

_WINDOWS_MODULES_CMD = [
    "powershell", "-NoProfile", "-Command",
    "Get-CimInstance Win32_PhysicalMemory | Select-Object SMBIOSMemoryType,Speed,"
    "ConfiguredClockSpeed,Capacity | ConvertTo-Json",
]
_MACOS_MEMORY_CMD = ["system_profiler", "SPMemoryDataType", "-json"]
_LINUX_MEMORY_CMD = ["dmidecode", "-t", "memory"]


def theoretical_bandwidth_gbps(speed_mts: int, channels: int) -> float:
    """Peak DDR bandwidth: transfers per second × 8 bytes × channels."""
    return round(speed_mts * 8 * channels / 1000, 1)


def _default_vm() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def detect_memory(
    runner: Runner,
    os_name: OsName,
    *,
    vm_provider: Callable[[], tuple[int, int]] | None = None,
) -> tuple[Memory, list[Probe]]:
    """Detect memory totals and, when the OS tells us, type, speed and channel count."""
    total, available = (vm_provider or _default_vm)()
    memory = Memory(total_bytes=total, available_bytes=available)
    probes: list[Probe] = []

    if os_name == "windows":
        facts, rec = probe("memory-modules", runner, _WINDOWS_MODULES_CMD, _parse_windows_modules)
    elif os_name == "macos":
        facts, rec = probe("memory-modules", runner, _MACOS_MEMORY_CMD, _parse_macos_memory)
    else:
        facts, rec = probe("memory-modules", runner, _LINUX_MEMORY_CMD, _parse_dmidecode)
    probes.append(rec)
    if facts:
        memory.type, memory.speed_mts, memory.channels = facts
        if memory.speed_mts and memory.channels:
            memory.bandwidth_gbps = theoretical_bandwidth_gbps(memory.speed_mts, memory.channels)
            memory.bandwidth_source = "estimated"
    return memory, probes


def _parse_windows_modules(out: str) -> tuple[str | None, int | None, int | None]:
    data: Any = json.loads(out)
    modules = data if isinstance(data, list) else [data]
    populated = [m for m in modules if m.get("Capacity")]
    if not populated:
        raise ValueError("no populated memory modules")
    first = populated[0]
    mem_type = _SMBIOS_TYPES.get(int(first.get("SMBIOSMemoryType") or 0))
    speed = first.get("ConfiguredClockSpeed") or first.get("Speed")
    return mem_type, int(speed) if speed else None, min(len(populated), 8)


def _parse_macos_memory(out: str) -> tuple[str | None, int | None, int | None]:
    data = json.loads(out)
    items = data.get("SPMemoryDataType", [])
    if not items:
        raise ValueError("no SPMemoryDataType")
    mem_type = items[0].get("dimm_type")
    return (str(mem_type) if mem_type else None), None, None


def _parse_dmidecode(out: str) -> tuple[str | None, int | None, int | None]:
    mem_type: str | None = None
    speed: int | None = None
    populated = 0
    in_device = False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Memory Device"):
            in_device = True
            continue
        if not in_device:
            continue
        if stripped.startswith("Size:") and "No Module" not in stripped:
            populated += 1
        elif stripped.startswith("Type:") and mem_type is None and "Unknown" not in stripped:
            mem_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Configured Memory Speed:") and speed is None:
            digits = "".join(ch for ch in stripped.split(":", 1)[1] if ch.isdigit())
            speed = int(digits) if digits else None
    if populated == 0:
        raise ValueError("no populated memory modules")
    return mem_type, speed, min(populated, 8)
```

- [ ] **Step 5: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_cpu.py tests/unit/test_memory.py -v && ruff check . && mypy`
Expected: PASS, clean, clean. (`ruff` may flag the `lambda` assignments in the tests under `E731`; tests ignore `D` only, so replace them with small `def` functions if flagged.)

- [ ] **Step 6: Commit**

```bash
git add src/llamafit/hardware/cpu.py src/llamafit/hardware/memory.py tests/unit/test_cpu.py tests/unit/test_memory.py
git commit -m "feat: CPU and memory detection with performance-core and DDR facts"
```

---

### Task 6: GPU detection

**Files:**
- Create: `src/llamafit/hardware/gpu.py`
- Test: `tests/unit/test_gpu.py`

**Interfaces:**
- Consumes: `Runner`, `probe()`; `Gpu`, `Probe`, `OsName`, `Vendor`, `Backend`.
- Produces:
  - `detect_gpus(runner: Runner, os_name: OsName) -> tuple[list[Gpu], list[Probe]]`
  - `vendor_from_name(name: str) -> Vendor`
  - `parse_nvidia_smi(out: str) -> list[Gpu]` (CSV from `--query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits`)
  - `parse_rocm_smi(out: str) -> list[Gpu]`, `parse_system_profiler(out: str) -> list[Gpu]`, `parse_wmi_video(out: str) -> list[Gpu]`, `parse_lspci(out: str) -> list[Gpu]`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_gpu.py`:

```python
import json

from llamafit.hardware.gpu import detect_gpus, parse_nvidia_smi, vendor_from_name
from llamafit.hardware.runner import FakeRunner

NVIDIA = "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n"
ROCM = json.dumps({"card0": {"Card Series": "Radeon RX 7900 XTX", "VRAM Total Memory (B)": "25753026560",
                              "VRAM Total Used Memory (B)": "1048576000"}})
APPLE = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M3 Max", "sppci_cores": "40"}]})
WMI = json.dumps([{"Name": "NVIDIA GeForce RTX 4060", "AdapterRAM": 4293918720},
                  {"Name": "Intel(R) UHD Graphics 770", "AdapterRAM": 1073741824}])
LSPCI = "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD107 [GeForce RTX 4060] [10de:2882]\n"


def test_vendor_from_name() -> None:
    assert vendor_from_name("NVIDIA GeForce RTX 4060") == "nvidia"
    assert vendor_from_name("AMD Radeon RX 7900 XTX") == "amd"
    assert vendor_from_name("Intel(R) Arc(TM) A770") == "intel"
    assert vendor_from_name("Apple M3 Max") == "apple"
    assert vendor_from_name("Matrox G200") == "other"


def test_parse_nvidia_smi_converts_mib_to_bytes() -> None:
    gpus = parse_nvidia_smi(NVIDIA)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "NVIDIA GeForce RTX 4060"
    assert gpu.vram_total_bytes == 8188 * 1024**2
    assert gpu.vram_used_bytes == 550 * 1024**2
    assert gpu.driver == "610.88"
    assert gpu.backend_hint == "cuda"


def test_windows_prefers_nvidia_smi_and_adds_others_from_wmi() -> None:
    runner = FakeRunner({"nvidia-smi": NVIDIA, "powershell": WMI})
    gpus, probes = detect_gpus(runner, "windows")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4060", "Intel(R) UHD Graphics 770"]
    assert gpus[0].vram_total_bytes == 8188 * 1024**2      # from nvidia-smi, not the WMI 4 GB cap
    assert gpus[1].vram_total_bytes is None                 # WMI AdapterRAM is unreliable: name only
    assert gpus[1].backend_hint == "vulkan"
    assert {p.name for p in probes} == {"nvidia-smi", "rocm-smi", "wmi-video"}


def test_linux_amd_from_rocm_smi() -> None:
    runner = FakeRunner({"rocm-smi": ROCM, "lspci": ""})
    gpus, _ = detect_gpus(runner, "linux")
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_total_bytes == 25753026560
    assert gpus[0].backend_hint == "hip"


def test_macos_apple_silicon_has_no_vram_figure() -> None:
    runner = FakeRunner({"system_profiler": APPLE})
    gpus, _ = detect_gpus(runner, "macos")
    assert gpus[0].vendor == "apple" and gpus[0].backend_hint == "metal"
    assert gpus[0].vram_total_bytes is None


def test_no_tools_means_no_gpus_but_probes_explain() -> None:
    gpus, probes = detect_gpus(FakeRunner({}), "linux")
    assert gpus == []
    assert probes and all(not p.ok for p in probes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_gpu.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `hardware/gpu.py`**

```python
"""GPU detection through vendor tools, with name-only fallbacks."""

from __future__ import annotations

import json
import re
from typing import Any

from llamafit.hardware.runner import Runner, probe
from llamafit.models.host import Backend, Gpu, OsName, Probe, Vendor

_NVIDIA_CMD = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
               "--format=csv,noheader,nounits"]
_ROCM_CMD = ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"]
_APPLE_CMD = ["system_profiler", "SPDisplaysDataType", "-json"]
_WMI_CMD = ["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"]
_LSPCI_CMD = ["lspci", "-nn"]


def vendor_from_name(name: str) -> Vendor:
    """Guess the vendor from a device name."""
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    if "apple" in lowered:
        return "apple"
    if "intel" in lowered:
        return "intel"
    return "other"


def _backend_for(vendor: Vendor, os_name: OsName) -> Backend:
    if vendor == "nvidia":
        return "cuda"
    if vendor == "apple":
        return "metal"
    if vendor == "amd":
        return "hip" if os_name == "linux" else "vulkan"
    if vendor == "intel":
        return "vulkan"
    return "cpu"


def parse_nvidia_smi(out: str) -> list[Gpu]:
    """Parse the CSV produced by ``_NVIDIA_CMD`` (memory figures are MiB)."""
    gpus: list[Gpu] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index, name, total, used, driver = parts[:5]
        gpus.append(Gpu(index=int(index), vendor="nvidia", name=name,
                        vram_total_bytes=int(float(total)) * 1024**2,
                        vram_used_bytes=int(float(used)) * 1024**2,
                        backend_hint="cuda", driver=driver))
    if not gpus:
        raise ValueError("nvidia-smi returned no GPUs")
    return gpus


def parse_rocm_smi(out: str) -> list[Gpu]:
    """Parse ``rocm-smi --json`` output keyed by ``cardN``."""
    data: dict[str, Any] = json.loads(out)
    gpus: list[Gpu] = []
    for key in sorted(k for k in data if k.startswith("card")):
        card = data[key]
        name = str(card.get("Card Series") or card.get("Card model") or "AMD GPU")
        total = card.get("VRAM Total Memory (B)")
        used = card.get("VRAM Total Used Memory (B)")
        gpus.append(Gpu(index=int(key[4:] or 0), vendor="amd", name=name,
                        vram_total_bytes=int(total) if total else None,
                        vram_used_bytes=int(used) if used else None, backend_hint="hip"))
    if not gpus:
        raise ValueError("rocm-smi returned no cards")
    return gpus


def parse_system_profiler(out: str) -> list[Gpu]:
    """Parse ``system_profiler SPDisplaysDataType -json``; Apple GPUs share system memory."""
    data = json.loads(out)
    items = data.get("SPDisplaysDataType", [])
    gpus = [Gpu(index=i, vendor=vendor_from_name(str(item.get("sppci_model", "Apple GPU"))),
                name=str(item.get("sppci_model", "Apple GPU")), backend_hint="metal")
            for i, item in enumerate(items)]
    if not gpus:
        raise ValueError("system_profiler returned no displays")
    return gpus


def parse_wmi_video(out: str) -> list[Gpu]:
    """Parse WMI video controllers: names only, AdapterRAM is capped at 4 GB and ignored."""
    data: Any = json.loads(out)
    items = data if isinstance(data, list) else [data]
    gpus = [Gpu(index=i, vendor=vendor_from_name(str(item.get("Name", ""))), name=str(item.get("Name", "")))
            for i, item in enumerate(items) if item.get("Name")]
    if not gpus:
        raise ValueError("WMI returned no video controllers")
    return gpus


def parse_lspci(out: str) -> list[Gpu]:
    """Parse ``lspci -nn`` for VGA and 3D controllers: names only."""
    gpus: list[Gpu] = []
    for i, line in enumerate(l for l in out.splitlines() if re.search(r"VGA|3D controller", l)):
        name = line.split(":", 2)[-1].split("[", 1)[0].strip()
        bracket = re.findall(r"\[([^\]]+)\]", line)
        pretty = bracket[1] if len(bracket) > 1 else name
        gpus.append(Gpu(index=i, vendor=vendor_from_name(line), name=pretty))
    if not gpus:
        raise ValueError("lspci found no display controllers")
    return gpus


def detect_gpus(runner: Runner, os_name: OsName) -> tuple[list[Gpu], list[Probe]]:
    """Run the probes that make sense for the OS and merge their results.

    Vendor tools win over generic listings: a device already reported by nvidia-smi
    or rocm-smi is not duplicated from WMI or lspci, and generic entries carry no
    VRAM figure because those sources are unreliable for it.
    """
    probes: list[Probe] = []
    found: list[Gpu] = []

    if os_name == "macos":
        gpus, rec = probe("system-profiler", runner, _APPLE_CMD, parse_system_profiler)
        probes.append(rec)
        return (gpus or []), probes

    for name, cmd, parser in (("nvidia-smi", _NVIDIA_CMD, parse_nvidia_smi),
                              ("rocm-smi", _ROCM_CMD, parse_rocm_smi)):
        gpus, rec = probe(name, runner, cmd, parser)
        probes.append(rec)
        if gpus:
            found.extend(gpus)

    if os_name == "windows":
        generic, rec = probe("wmi-video", runner, _WMI_CMD, parse_wmi_video)
    else:
        generic, rec = probe("lspci", runner, _LSPCI_CMD, parse_lspci)
    probes.append(rec)

    known = {g.name.lower() for g in found}
    for gpu in generic or []:
        if gpu.name.lower() in known or any(gpu.name.lower() in k or k in gpu.name.lower() for k in known):
            continue
        gpu.backend_hint = _backend_for(gpu.vendor, os_name)
        found.append(gpu)

    for i, gpu in enumerate(found):
        gpu.index = i
        if gpu.backend_hint == "cpu":
            gpu.backend_hint = _backend_for(gpu.vendor, os_name)
    return found, probes
```

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_gpu.py -v && ruff check . && mypy`
Expected: PASS, clean, clean.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/hardware/gpu.py tests/unit/test_gpu.py
git commit -m "feat: GPU detection for NVIDIA, AMD, Apple and generic listings"
```

---

### Task 7: GPU specification table, bandwidth and disks

**Files:**
- Create: `src/llamafit/data/__init__.py` (empty), `src/llamafit/data/gpus.json`, `src/llamafit/hardware/gputable.py`, `src/llamafit/hardware/bandwidth.py`, `src/llamafit/hardware/disks.py`
- Test: `tests/unit/test_gputable.py`, `tests/unit/test_bandwidth.py`, `tests/unit/test_disks.py`

**Interfaces:**
- Produces:
  - `GpuSpec(pattern: str, bandwidth_gbps: float, compute_tflops_fp16: float, pcie_gbps: float | None = None, unified: bool = False)`; `lookup_gpu(name: str) -> GpuSpec | None` (case-insensitive substring match on `pattern`, longest pattern wins); `enrich_gpu(gpu: Gpu) -> Gpu` (fills `bandwidth_gbps` and `compute_tflops_fp16` when unknown).
  - `measure_ram_bandwidth_gbps(*, duration_s: float = 0.05, buffer_mb: int = 256) -> float | None` — multi-pass copy of a buffer larger than the cache; uses NumPy when importable, else a `bytearray` slice copy with a ×1.6 correction documented in the docstring; returns `None` when it cannot allocate.
  - `resolve_memory_bandwidth(memory: Memory, *, measure: bool = True) -> Memory` — sets `bandwidth_gbps` and `bandwidth_source` with precedence measured (when plausible: 5 to 1000 GB/s) → estimated (already set) → assumed (`ASSUMED_RAM_BANDWIDTH_GBPS = 40.0`).
  - `detect_disks(paths: Iterable[Path]) -> list[Disk]` — one `Disk` per distinct mount, unreadable paths skipped.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_gputable.py`:

```python
from llamafit.hardware.gputable import enrich_gpu, lookup_gpu
from llamafit.models import Gpu


def test_lookup_prefers_the_longest_matching_pattern() -> None:
    spec = lookup_gpu("NVIDIA GeForce RTX 4060 Ti")
    assert spec is not None and spec.pattern == "rtx 4060 ti"
    spec = lookup_gpu("NVIDIA GeForce RTX 4060")
    assert spec is not None and spec.pattern == "rtx 4060"
    assert spec.bandwidth_gbps == 272


def test_lookup_unknown_returns_none() -> None:
    assert lookup_gpu("Matrox G200") is None


def test_enrich_fills_only_unknown_fields() -> None:
    gpu = Gpu(index=0, vendor="nvidia", name="NVIDIA GeForce RTX 4060", bandwidth_gbps=999)
    enriched = enrich_gpu(gpu)
    assert enriched.bandwidth_gbps == 999
    assert enriched.compute_tflops_fp16 is not None


def test_apple_entry_is_unified() -> None:
    spec = lookup_gpu("Apple M3 Max")
    assert spec is not None and spec.unified and spec.bandwidth_gbps >= 300
```

`tests/unit/test_bandwidth.py`:

```python
import pytest

from llamafit.hardware.bandwidth import ASSUMED_RAM_BANDWIDTH_GBPS, measure_ram_bandwidth_gbps, resolve_memory_bandwidth
from llamafit.models import Memory


def test_resolve_keeps_estimate_when_not_measuring() -> None:
    memory = Memory(total_bytes=1, available_bytes=1, bandwidth_gbps=67.2, bandwidth_source="estimated")
    out = resolve_memory_bandwidth(memory, measure=False)
    assert out.bandwidth_gbps == 67.2 and out.bandwidth_source == "estimated"


def test_resolve_assumes_when_nothing_known() -> None:
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=False)
    assert out.bandwidth_gbps == ASSUMED_RAM_BANDWIDTH_GBPS and out.bandwidth_source == "assumed"


@pytest.mark.hardware
def test_measurement_is_plausible_on_a_real_machine() -> None:
    value = measure_ram_bandwidth_gbps()
    assert value is None or 5 <= value <= 1000
```

`tests/unit/test_disks.py`:

```python
from pathlib import Path

from llamafit.hardware.disks import detect_disks


def test_detect_disks_dedupes_mounts_and_skips_missing(tmp_path: Path) -> None:
    disks = detect_disks([tmp_path, tmp_path / "sub", Path("/definitely/missing/xyz")])
    assert len(disks) == 1
    assert disks[0].free_bytes > 0 and disks[0].total_bytes >= disks[0].free_bytes
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_gputable.py tests/unit/test_bandwidth.py tests/unit/test_disks.py -v -m "not hardware"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `data/gpus.json`**

Values are vendor-published peak memory bandwidth (GB/s) and dense FP16 throughput (TFLOPS, without sparsity). `pcie_gbps` is the practical one-direction bandwidth of the card's slot (PCIe 4.0 x16 ≈ 25, x8 ≈ 12, PCIe 5.0 x16 ≈ 50). Extend by pull request; keep patterns lowercase and specific enough not to collide.

```json
{
  "_comment": "Peak memory bandwidth GB/s and dense FP16 TFLOPS from vendor specifications. Longest matching pattern wins.",
  "gpus": [
    {"pattern": "rtx 5090", "bandwidth_gbps": 1792, "compute_tflops_fp16": 105, "pcie_gbps": 50},
    {"pattern": "rtx 5080", "bandwidth_gbps": 960, "compute_tflops_fp16": 56, "pcie_gbps": 50},
    {"pattern": "rtx 5070 ti", "bandwidth_gbps": 896, "compute_tflops_fp16": 44, "pcie_gbps": 50},
    {"pattern": "rtx 5070", "bandwidth_gbps": 672, "compute_tflops_fp16": 31, "pcie_gbps": 50},
    {"pattern": "rtx 5060 ti", "bandwidth_gbps": 448, "compute_tflops_fp16": 24, "pcie_gbps": 25},
    {"pattern": "rtx 5060", "bandwidth_gbps": 448, "compute_tflops_fp16": 19, "pcie_gbps": 25},
    {"pattern": "rtx 4090", "bandwidth_gbps": 1008, "compute_tflops_fp16": 83, "pcie_gbps": 25},
    {"pattern": "rtx 4080 super", "bandwidth_gbps": 736, "compute_tflops_fp16": 52, "pcie_gbps": 25},
    {"pattern": "rtx 4080", "bandwidth_gbps": 717, "compute_tflops_fp16": 49, "pcie_gbps": 25},
    {"pattern": "rtx 4070 ti super", "bandwidth_gbps": 672, "compute_tflops_fp16": 44, "pcie_gbps": 25},
    {"pattern": "rtx 4070 ti", "bandwidth_gbps": 504, "compute_tflops_fp16": 40, "pcie_gbps": 25},
    {"pattern": "rtx 4070 super", "bandwidth_gbps": 504, "compute_tflops_fp16": 36, "pcie_gbps": 25},
    {"pattern": "rtx 4070", "bandwidth_gbps": 504, "compute_tflops_fp16": 29, "pcie_gbps": 25},
    {"pattern": "rtx 4060 ti", "bandwidth_gbps": 288, "compute_tflops_fp16": 22, "pcie_gbps": 12},
    {"pattern": "rtx 4060", "bandwidth_gbps": 272, "compute_tflops_fp16": 15, "pcie_gbps": 12},
    {"pattern": "rtx 3090", "bandwidth_gbps": 936, "compute_tflops_fp16": 36, "pcie_gbps": 25},
    {"pattern": "rtx 3080", "bandwidth_gbps": 760, "compute_tflops_fp16": 30, "pcie_gbps": 25},
    {"pattern": "rtx 3070", "bandwidth_gbps": 448, "compute_tflops_fp16": 20, "pcie_gbps": 25},
    {"pattern": "rtx 3060 ti", "bandwidth_gbps": 448, "compute_tflops_fp16": 16, "pcie_gbps": 25},
    {"pattern": "rtx 3060", "bandwidth_gbps": 360, "compute_tflops_fp16": 13, "pcie_gbps": 25},
    {"pattern": "rtx a6000", "bandwidth_gbps": 768, "compute_tflops_fp16": 39, "pcie_gbps": 25},
    {"pattern": "a100", "bandwidth_gbps": 2039, "compute_tflops_fp16": 312, "pcie_gbps": 25},
    {"pattern": "h100", "bandwidth_gbps": 3352, "compute_tflops_fp16": 989, "pcie_gbps": 50},
    {"pattern": "rx 7900 xtx", "bandwidth_gbps": 960, "compute_tflops_fp16": 123, "pcie_gbps": 25},
    {"pattern": "rx 7900 xt", "bandwidth_gbps": 800, "compute_tflops_fp16": 103, "pcie_gbps": 25},
    {"pattern": "rx 7800 xt", "bandwidth_gbps": 624, "compute_tflops_fp16": 75, "pcie_gbps": 25},
    {"pattern": "rx 7600", "bandwidth_gbps": 288, "compute_tflops_fp16": 43, "pcie_gbps": 12},
    {"pattern": "rx 9070 xt", "bandwidth_gbps": 640, "compute_tflops_fp16": 97, "pcie_gbps": 50},
    {"pattern": "arc a770", "bandwidth_gbps": 560, "compute_tflops_fp16": 39, "pcie_gbps": 25},
    {"pattern": "arc b580", "bandwidth_gbps": 456, "compute_tflops_fp16": 29, "pcie_gbps": 25},
    {"pattern": "apple m1 ultra", "bandwidth_gbps": 800, "compute_tflops_fp16": 21, "unified": true},
    {"pattern": "apple m1 max", "bandwidth_gbps": 400, "compute_tflops_fp16": 10.4, "unified": true},
    {"pattern": "apple m1 pro", "bandwidth_gbps": 200, "compute_tflops_fp16": 5.2, "unified": true},
    {"pattern": "apple m1", "bandwidth_gbps": 68, "compute_tflops_fp16": 2.6, "unified": true},
    {"pattern": "apple m2 ultra", "bandwidth_gbps": 800, "compute_tflops_fp16": 27, "unified": true},
    {"pattern": "apple m2 max", "bandwidth_gbps": 400, "compute_tflops_fp16": 13.6, "unified": true},
    {"pattern": "apple m2 pro", "bandwidth_gbps": 200, "compute_tflops_fp16": 6.8, "unified": true},
    {"pattern": "apple m2", "bandwidth_gbps": 100, "compute_tflops_fp16": 3.6, "unified": true},
    {"pattern": "apple m3 ultra", "bandwidth_gbps": 819, "compute_tflops_fp16": 28, "unified": true},
    {"pattern": "apple m3 max", "bandwidth_gbps": 400, "compute_tflops_fp16": 14.2, "unified": true},
    {"pattern": "apple m3 pro", "bandwidth_gbps": 150, "compute_tflops_fp16": 7.4, "unified": true},
    {"pattern": "apple m3", "bandwidth_gbps": 100, "compute_tflops_fp16": 3.5, "unified": true},
    {"pattern": "apple m4 max", "bandwidth_gbps": 546, "compute_tflops_fp16": 18.4, "unified": true},
    {"pattern": "apple m4 pro", "bandwidth_gbps": 273, "compute_tflops_fp16": 9.2, "unified": true},
    {"pattern": "apple m4", "bandwidth_gbps": 120, "compute_tflops_fp16": 4.6, "unified": true},
    {"pattern": "apple m5 max", "bandwidth_gbps": 614, "compute_tflops_fp16": 24, "unified": true},
    {"pattern": "apple m5 pro", "bandwidth_gbps": 307, "compute_tflops_fp16": 12, "unified": true},
    {"pattern": "apple m5", "bandwidth_gbps": 153, "compute_tflops_fp16": 6, "unified": true}
  ]
}
```

- [ ] **Step 4: Implement `hardware/gputable.py`**

```python
"""Bundled GPU specifications: bandwidth and compute the probes cannot read."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from pydantic import BaseModel

from llamafit.models.host import Gpu


class GpuSpec(BaseModel):
    """One row of ``data/gpus.json``."""

    pattern: str
    bandwidth_gbps: float
    compute_tflops_fp16: float
    pcie_gbps: float | None = None
    unified: bool = False


@lru_cache(maxsize=1)
def _table() -> list[GpuSpec]:
    text = resources.files("llamafit.data").joinpath("gpus.json").read_text(encoding="utf-8")
    return [GpuSpec(**row) for row in json.loads(text)["gpus"]]


def lookup_gpu(name: str) -> GpuSpec | None:
    """Find the row whose pattern is the longest substring of ``name`` (case-insensitive)."""
    lowered = name.lower()
    matches = [spec for spec in _table() if spec.pattern in lowered]
    if not matches:
        return None
    return max(matches, key=lambda spec: len(spec.pattern))


def enrich_gpu(gpu: Gpu) -> Gpu:
    """Fill bandwidth and compute from the table when the probe left them unknown."""
    spec = lookup_gpu(gpu.name)
    if spec is None:
        return gpu
    if gpu.bandwidth_gbps is None:
        gpu.bandwidth_gbps = spec.bandwidth_gbps
    if gpu.compute_tflops_fp16 is None:
        gpu.compute_tflops_fp16 = spec.compute_tflops_fp16
    return gpu
```

- [ ] **Step 5: Implement `hardware/bandwidth.py`**

```python
"""RAM bandwidth: measure it when possible, otherwise estimate or assume.

Generation speed with experts in RAM is bounded by this number, so it deserves a
real measurement. The probe copies a buffer larger than any CPU cache for a few tens
of milliseconds and reports bytes moved per second (read plus write counted once, as
llama.cpp's weight streaming is read-dominated).
"""

from __future__ import annotations

import time

from llamafit.models.host import Memory

ASSUMED_RAM_BANDWIDTH_GBPS = 40.0
PLAUSIBLE_RANGE_GBPS = (5.0, 1000.0)
# A bytearray slice copy in pure Python runs single-threaded through memcpy; measured
# against NumPy on the reference machine it under-reports by about 1.6x.
PURE_PYTHON_CORRECTION = 1.6


def measure_ram_bandwidth_gbps(*, duration_s: float = 0.05, buffer_mb: int = 256) -> float | None:
    """Measure copy bandwidth in GB/s, or ``None`` when the buffer cannot be allocated."""
    size = buffer_mb * 1024 * 1024
    try:
        import numpy as np

        src = np.ones(size, dtype=np.uint8)
        dst = np.empty_like(src)
        correction = 1.0

        def copy() -> None:
            np.copyto(dst, src)

    except ImportError:
        try:
            src_b = bytearray(size)
            dst_b = bytearray(size)
        except MemoryError:
            return None
        correction = PURE_PYTHON_CORRECTION

        def copy() -> None:
            dst_b[:] = src_b

    except MemoryError:
        return None

    copy()  # warm up, fault the pages in
    passes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        copy()
        passes += 1
    elapsed = time.perf_counter() - start
    if passes == 0 or elapsed <= 0:
        return None
    return round(passes * size / elapsed / 1e9 * correction, 1)


def resolve_memory_bandwidth(memory: Memory, *, measure: bool = True) -> Memory:
    """Set ``bandwidth_gbps`` with the best available source and label it."""
    if measure:
        measured = measure_ram_bandwidth_gbps()
        if measured is not None and PLAUSIBLE_RANGE_GBPS[0] <= measured <= PLAUSIBLE_RANGE_GBPS[1]:
            memory.bandwidth_gbps = measured
            memory.bandwidth_source = "measured"
            return memory
    if memory.bandwidth_gbps is not None and memory.bandwidth_source == "estimated":
        return memory
    memory.bandwidth_gbps = ASSUMED_RAM_BANDWIDTH_GBPS
    memory.bandwidth_source = "assumed"
    return memory
```

- [ ] **Step 6: Implement `hardware/disks.py`**

```python
"""Free disk space at the paths that matter."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from llamafit.models.host import Disk


def detect_disks(paths: Iterable[Path]) -> list[Disk]:
    """One ``Disk`` per distinct mount among ``paths``; missing paths fall back to parents."""
    seen: dict[str, Disk] = {}
    for path in paths:
        candidate = Path(path)
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        if not candidate.exists():
            continue
        anchor = str(candidate.anchor or candidate)
        try:
            usage = shutil.disk_usage(candidate)
        except OSError:
            continue
        if anchor not in seen:
            seen[anchor] = Disk(path=anchor, free_bytes=usage.free, total_bytes=usage.total)
    return list(seen.values())
```

- [ ] **Step 7: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_gputable.py tests/unit/test_bandwidth.py tests/unit/test_disks.py -v -m "not hardware" && ruff check . && mypy`
Expected: PASS, clean, clean. Then run `pytest -m hardware -v` once on the real machine: the measurement should be plausible (the reference machine reports roughly 30 to 60 GB/s).

- [ ] **Step 8: Commit**

```bash
git add src/llamafit/data src/llamafit/hardware/gputable.py src/llamafit/hardware/bandwidth.py src/llamafit/hardware/disks.py tests/unit
git commit -m "feat: GPU specification table, RAM bandwidth measurement and disk space"
```

---

### Task 8: The `scan()` function

**Files:**
- Modify: `src/llamafit/hardware/__init__.py`
- Create: `tests/unit/test_scan.py`, `tests/fixtures/reference_machine.py`

**Interfaces:**
- Consumes: everything from Tasks 4 to 7.
- Produces:
  - `scan(runner: Runner | None = None, *, os_name: OsName | None = None, measure_bandwidth: bool = True, extra_paths: Iterable[Path] = (), cpuinfo_provider=None, vm_provider=None) -> Host`
  - `current_os() -> OsName`, `current_arch() -> Arch`
  - Test fixture `reference_runner() -> FakeRunner` reproducing the reference machine (RTX 4060, i9-14900KF, 128 GB DDR5-4200, Windows 11) and `reference_cpuinfo()`, `reference_vm()`.

- [ ] **Step 1: Write the fixture and the failing test**

`tests/fixtures/__init__.py`: empty. `tests/fixtures/reference_machine.py`:

```python
"""Recorded probe outputs from the reference machine (2026-09-09)."""

import json

from llamafit.hardware.runner import FakeRunner

NVIDIA_SMI = "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n"
WMI_VIDEO = json.dumps([{"Name": "NVIDIA GeForce RTX 4060", "AdapterRAM": 4293918720}])
WMI_MEMORY = json.dumps([
    {"SMBIOSMemoryType": 34, "Speed": 4800, "ConfiguredClockSpeed": 4200, "Capacity": 68719476736},
    {"SMBIOSMemoryType": 34, "Speed": 4800, "ConfiguredClockSpeed": 4200, "Capacity": 68719476736},
])
POWERSHELL_MEMORY_CMD = (
    "powershell -NoProfile -Command Get-CimInstance Win32_PhysicalMemory | Select-Object "
    "SMBIOSMemoryType,Speed,ConfiguredClockSpeed,Capacity | ConvertTo-Json"
)
POWERSHELL_VIDEO_CMD = (
    "powershell -NoProfile -Command Get-CimInstance Win32_VideoController | Select-Object "
    "Name,AdapterRAM | ConvertTo-Json"
)


def reference_runner() -> FakeRunner:
    return FakeRunner({
        "nvidia-smi": NVIDIA_SMI,
        POWERSHELL_MEMORY_CMD: WMI_MEMORY,
        POWERSHELL_VIDEO_CMD: WMI_VIDEO,
    })


def reference_cpuinfo() -> dict[str, object]:
    return {"brand_raw": "Intel(R) Core(TM) i9-14900KF", "flags": ["avx2", "avx512f", "avx512_vnni"]}


def reference_vm() -> tuple[int, int]:
    return 128 * 1024**3, 100 * 1024**3
```

`tests/unit/test_scan.py`:

```python
from llamafit.hardware import scan
from tests.fixtures.reference_machine import reference_cpuinfo, reference_runner, reference_vm


def test_scan_reference_machine() -> None:
    host = scan(reference_runner(), os_name="windows", measure_bandwidth=False,
                cpuinfo_provider=reference_cpuinfo, vm_provider=reference_vm)
    assert host.os == "windows"
    assert host.cpu.performance_cores == 8
    assert host.memory.total_bytes == 128 * 1024**3
    assert host.memory.bandwidth_gbps == 67.2 and host.memory.bandwidth_source == "estimated"
    gpu = host.primary_gpu
    assert gpu is not None and gpu.name == "NVIDIA GeForce RTX 4060"
    assert gpu.bandwidth_gbps == 272 and gpu.compute_tflops_fp16 == 15
    assert host.vram_available_bytes == (8188 - 550) * 1024**2
    assert host.unified_memory is False
    assert host.disks, "the current working directory's disk is always reported"
    names = [p.name for p in host.probes]
    assert "cpuinfo" in names and "nvidia-smi" in names and "memory-modules" in names


def test_scan_with_nothing_available_still_returns_a_host() -> None:
    from llamafit.hardware.runner import FakeRunner

    def broken() -> dict[str, object]:
        raise RuntimeError("nope")

    host = scan(FakeRunner({}), os_name="linux", measure_bandwidth=False,
                cpuinfo_provider=broken, vm_provider=lambda: (8 * 1024**3, 4 * 1024**3))
    assert host.gpus == []
    assert host.memory.bandwidth_source == "assumed"
    assert any(not p.ok for p in host.probes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_scan.py -v`
Expected: FAIL with `ImportError: cannot import name 'scan'`.

- [ ] **Step 3: Implement `scan()` in `hardware/__init__.py`**

```python
"""Host detection: ``scan()`` composes every probe into a ``Host``."""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llamafit.hardware.bandwidth import resolve_memory_bandwidth
from llamafit.hardware.cpu import detect_cpu
from llamafit.hardware.disks import detect_disks
from llamafit.hardware.gpu import detect_gpus
from llamafit.hardware.gputable import enrich_gpu, lookup_gpu
from llamafit.hardware.memory import detect_memory
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.models.host import Arch, Host, OsName
from llamafit.paths import get_paths


def current_os() -> OsName:
    """Map ``platform.system()`` to LlamaFit's OS names."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def current_arch() -> Arch:
    """Map ``platform.machine()`` to ``x86_64``, ``arm64`` or ``other``."""
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "other"


def scan(
    runner: Runner | None = None,
    *,
    os_name: OsName | None = None,
    measure_bandwidth: bool = True,
    extra_paths: Iterable[Path] = (),
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
    vm_provider: Callable[[], tuple[int, int]] | None = None,
) -> Host:
    """Detect everything about this machine that the estimator needs.

    Every probe is optional: what fails is recorded in ``Host.probes`` and the rest
    of the scan continues. Pass a ``FakeRunner`` and providers to scan a recorded
    machine instead of the real one.
    """
    runner = runner or SubprocessRunner()
    os_name = os_name or current_os()
    cpu, cpu_probes = detect_cpu(runner, os_name, cpuinfo_provider=cpuinfo_provider)
    memory, memory_probes = detect_memory(runner, os_name, vm_provider=vm_provider)
    memory = resolve_memory_bandwidth(memory, measure=measure_bandwidth)
    gpus, gpu_probes = detect_gpus(runner, os_name)
    gpus = [enrich_gpu(gpu) for gpu in gpus]
    unified = any(spec.unified for spec in (lookup_gpu(g.name) for g in gpus) if spec) or (
        os_name == "macos" and current_arch() == "arm64"
    )
    paths = get_paths()
    disks = detect_disks([Path.cwd(), paths.downloads_dir, *extra_paths])
    return Host(
        os=os_name,
        os_version=platform.platform(),
        arch=current_arch(),
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        unified_memory=unified,
        disks=disks,
        probes=[*cpu_probes, *memory_probes, *gpu_probes],
        scanned_at=datetime.now(timezone.utc),
    )


__all__ = ["current_arch", "current_os", "scan"]
```

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_scan.py -v && ruff check . && mypy`
Expected: PASS, clean, clean. Then run `python -c "from llamafit.hardware import scan; print(scan().model_dump_json(indent=2))"` on the real machine and read the output: the GPU, memory, bandwidth source and probe list must look right.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/hardware/__init__.py tests/unit/test_scan.py tests/fixtures
git commit -m "feat: scan() composes CPU, memory, GPU, bandwidth and disk probes into a Host"
```

---

### Task 9: llama.cpp detection — binaries, build, backends, local models

**Files:**
- Create: `src/llamafit/llamacpp/__init__.py` (docstring only for now), `src/llamafit/llamacpp/detect.py`
- Test: `tests/unit/test_llamacpp_detect.py`

**Interfaces:**
- Consumes: `Runner`, `probe()`; `LocalModel`, `Probe`, `OsName`.
- Produces:
  - `BINARIES = ("llama-server", "llama-cli", "llama-bench", "llama-gguf")`
  - `find_llamacpp_dir(*, env: Mapping[str, str], path_dirs: Iterable[Path], well_known: Iterable[Path]) -> Path | None` — first directory containing `llama-server` (with `.exe` on Windows, decided by `os_name` passed separately); precedence `LLAMA_CPP_PATH` → PATH → well-known.
  - `well_known_dirs(os_name: OsName, home: Path) -> list[Path]`
  - `parse_version(out: str) -> tuple[int | None, str | None]` — build number and commit from `llama-server --version` output such as `version: 10867 (f3f1a8f27)` or `build: 10867 (f3f1a8f2)`.
  - `detect_backends(bin_dir: Path) -> list[str]` — from `ggml-*` shared libraries: `cuda`, `hip`, `metal`, `vulkan`, `sycl`, `rpc`, `cpu`; on macOS `metal` when `ggml-metal` or `libggml-metal` exists, and `cpu` when any `ggml-cpu*` exists.
  - `find_local_models(dirs: Iterable[Path]) -> list[LocalModel]` — every `*.gguf` under the given directories (recursive, depth 3), split files reported once by their first shard, `bytes` summed across shards.
  - `detect_install(runner: Runner, os_name: OsName, *, env: Mapping[str, str] | None = None, home: Path | None = None, path_dirs: Iterable[Path] | None = None) -> tuple[LlamaCpp, list[Probe]]` — fills `installed`, `path`, `build`, `commit`, `backends`, `local_models`, `problems` (not the running servers; Task 10 adds those).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llamacpp_detect.py`:

```python
from pathlib import Path

from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp.detect import (
    detect_backends,
    detect_install,
    find_llamacpp_dir,
    find_local_models,
    parse_version,
    well_known_dirs,
)


def make_install(root: Path, *, windows: bool = True) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    exe = ".exe" if windows else ""
    for name in ("llama-server", "llama-cli", "llama-bench"):
        (bin_dir / f"{name}{exe}").write_bytes(b"")
    suffix = ".dll" if windows else ".so"
    for lib in ("ggml-base", "ggml-cuda", "ggml-cpu-alderlake", "ggml-rpc"):
        (bin_dir / f"{lib}{suffix}").write_bytes(b"")
    (root / "bin" / "VERSION.txt").write_text("b10867\n")
    return bin_dir


def test_parse_version_variants() -> None:
    assert parse_version("version: 10867 (f3f1a8f27)\nbuilt with clang") == (10867, "f3f1a8f27")
    assert parse_version("build: 10867 (f3f1a8f2) with MSVC") == (10867, "f3f1a8f2")
    assert parse_version("llama-server b10867\n") == (10867, None)
    assert parse_version("nonsense") == (None, None)


def test_find_dir_precedence(tmp_path: Path) -> None:
    env_dir = make_install(tmp_path / "env")
    path_dir = make_install(tmp_path / "path")
    assert find_llamacpp_dir(env={"LLAMA_CPP_PATH": str(env_dir)}, path_dirs=[path_dir],
                             well_known=[], os_name="windows") == env_dir
    assert find_llamacpp_dir(env={}, path_dirs=[path_dir], well_known=[], os_name="windows") == path_dir
    assert find_llamacpp_dir(env={}, path_dirs=[], well_known=[tmp_path / "missing"], os_name="windows") is None


def test_well_known_dirs_include_platform_defaults(tmp_path: Path) -> None:
    dirs = well_known_dirs("windows", tmp_path)
    assert Path("D:/llama.cpp/bin") in dirs or Path("D:\\llama.cpp\\bin") in dirs
    assert tmp_path / ".llamafit" / "llama.cpp" / "bin" in dirs


def test_detect_backends_from_libraries(tmp_path: Path) -> None:
    bin_dir = make_install(tmp_path)
    assert detect_backends(bin_dir) == ["cuda", "rpc", "cpu"]


def test_find_local_models_groups_shards(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "a").mkdir(parents=True)
    (models / "a" / "Model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (models / "a" / "Big-UD-Q4_K_XL-00001-of-00002.gguf").write_bytes(b"x" * 5)
    (models / "a" / "Big-UD-Q4_K_XL-00002-of-00002.gguf").write_bytes(b"x" * 7)
    (models / "a" / "mmproj-F16.gguf").write_bytes(b"x" * 3)
    found = {Path(m.path).name: m.bytes for m in find_local_models([models])}
    assert found == {"Model-Q4_K_M.gguf": 10, "Big-UD-Q4_K_XL-00001-of-00002.gguf": 12, "mmproj-F16.gguf": 3}


def test_detect_install_reads_version_and_backends(tmp_path: Path) -> None:
    bin_dir = make_install(tmp_path)
    server = str(bin_dir / "llama-server.exe")
    runner = FakeRunner({f"{server} --version": "version: 10867 (f3f1a8f27)\n"})
    llamacpp, probes = detect_install(runner, "windows", env={"LLAMA_CPP_PATH": str(bin_dir)},
                                      home=tmp_path, path_dirs=[])
    assert llamacpp.installed and llamacpp.path == str(bin_dir)
    assert llamacpp.build == 10867 and llamacpp.commit == "f3f1a8f27"
    assert llamacpp.backends == ["cuda", "rpc", "cpu"]
    assert [p.name for p in probes] == ["llama-server --version"]


def test_detect_install_absent(tmp_path: Path) -> None:
    llamacpp, probes = detect_install(FakeRunner({}), "linux", env={}, home=tmp_path, path_dirs=[])
    assert not llamacpp.installed
    assert llamacpp.problems and "not found" in llamacpp.problems[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_llamacpp_detect.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `llamacpp/detect.py`**

```python
"""Find llama.cpp on this machine and describe the installation."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from llamafit.hardware.runner import Runner, probe
from llamafit.models.host import OsName, Probe
from llamafit.models.llamacpp import LlamaCpp, LocalModel

BINARIES = ("llama-server", "llama-cli", "llama-bench", "llama-gguf")
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_BACKEND_ORDER = ["cuda", "hip", "metal", "vulkan", "sycl", "rpc", "cpu"]
_VERSION_PATTERNS = [
    re.compile(r"(?:version|build)\s*:?\s*(\d+)\s*\(([0-9a-f]{6,})\)", re.IGNORECASE),
    re.compile(r"\bb(\d{4,})\b"),
]


def exe_name(name: str, os_name: OsName) -> str:
    """``llama-server.exe`` on Windows, ``llama-server`` elsewhere."""
    return f"{name}.exe" if os_name == "windows" else name


def well_known_dirs(os_name: OsName, home: Path) -> list[Path]:
    """Directories where people commonly put llama.cpp, most likely first."""
    common = [home / ".llamafit" / "llama.cpp" / "bin", home / "llama.cpp" / "bin", home / "llama.cpp"]
    if os_name == "windows":
        return [*common, Path("C:/llama.cpp/bin"), Path("D:/llama.cpp/bin"), Path("C:/llama.cpp"),
                Path("D:/llama.cpp")]
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
    """Extract the build number and commit hash from ``llama-server --version`` output."""
    match = _VERSION_PATTERNS[0].search(out)
    if match:
        return int(match.group(1)), match.group(2)
    match = _VERSION_PATTERNS[1].search(out)
    if match:
        return int(match.group(1)), None
    return None, None


def detect_backends(bin_dir: Path) -> list[str]:
    """Infer compiled backends from the ``ggml-*`` shared libraries next to the binaries."""
    names = {p.name.lower() for p in bin_dir.iterdir() if p.is_file()}
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
    """List GGUF files; split models appear once, under their first shard, with total bytes."""
    models: dict[Path, int] = {}
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.rglob("*.gguf"):
            if len(path.relative_to(root).parts) > max_depth:
                continue
            match = _SHARD_RE.match(path.name)
            if match and match.group("index") != "00001":
                first = path.with_name(f"{match.group('stem')}-00001-of-{match.group('total')}.gguf")
                models[first] = models.get(first, 0) + path.stat().st_size
                continue
            models[path] = models.get(path, 0) + path.stat().st_size
    return [LocalModel(path=str(p), bytes=b) for p, b in sorted(models.items())]


def detect_install(
    runner: Runner,
    os_name: OsName,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    path_dirs: Iterable[Path] | None = None,
) -> tuple[LlamaCpp, list[Probe]]:
    """Describe the llama.cpp installation, or explain why none was found."""
    env = os.environ if env is None else env
    home = home or Path.home()
    if path_dirs is None:
        path_dirs = [Path(p) for p in env.get("PATH", "").split(os.pathsep) if p]
    probes: list[Probe] = []
    bin_dir = find_llamacpp_dir(env=env, path_dirs=path_dirs, well_known=well_known_dirs(os_name, home),
                                os_name=os_name)
    if bin_dir is None:
        return LlamaCpp(installed=False, problems=[
            "llama.cpp not found: no llama-server on PATH, in LLAMA_CPP_PATH or in the usual directories"
        ]), probes

    server = bin_dir / exe_name("llama-server", os_name)
    version, rec = probe("llama-server --version", runner, [str(server), "--version"], parse_version)
    probes.append(rec)
    build, commit = version if version else (None, None)
    if build is None:
        text = (bin_dir / "VERSION.txt").read_text(encoding="utf-8") if (bin_dir / "VERSION.txt").exists() else ""
        build, _ = parse_version(text)

    backends = detect_backends(bin_dir)
    problems: list[str] = []
    if not backends:
        problems.append("no ggml backend libraries found next to llama-server; the build may be static")
    model_dirs = [bin_dir.parent / "models", Path(env.get("LLAMA_CACHE", "")) if env.get("LLAMA_CACHE") else None]
    local_models = find_local_models([d for d in model_dirs if d])
    return LlamaCpp(installed=True, path=str(bin_dir), build=build, commit=commit, backends=backends,
                    local_models=local_models, problems=problems), probes
```

`llamacpp/__init__.py`: `"""llama.cpp integration."""` (the composed `detect_llamacpp()` is added in Task 10).

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_llamacpp_detect.py -v && ruff check . && mypy`
Expected: PASS, clean, clean.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/llamacpp tests/unit/test_llamacpp_detect.py
git commit -m "feat: llama.cpp installation detection (binaries, build, backends, local GGUF files)"
```

---

### Task 10: Running-server discovery and `detect_llamacpp()`

**Files:**
- Create: `src/llamafit/llamacpp/server.py`
- Modify: `src/llamafit/llamacpp/__init__.py`
- Test: `tests/unit/test_llamacpp_server.py`

**Interfaces:**
- Consumes: `detect_install()` (Task 9); `RunningServer`, `LlamaCpp`, `Probe`.
- Produces:
  - `class HttpClient(Protocol): def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None` — returns parsed JSON or `None` on any failure.
  - `HttpxClient()` — real implementation over `httpx`; `FakeHttp(responses: Mapping[str, Any])` — keyed by full URL.
  - `discover_servers(http: HttpClient, ports: Iterable[int]) -> list[RunningServer]` — for each port: `GET /health` must return `{"status": "ok"}`; then `/v1/models` gives the model id, `/props` gives `default_generation_settings.n_ctx` and `build_info`.
  - `candidate_ports(env: Mapping[str, str]) -> list[int]` — `LLAMA_SERVER_PORT` first if set, then 8080, 8081, 8098.
  - `detect_llamacpp(runner: Runner | None = None, *, os_name: OsName | None = None, env=None, http: HttpClient | None = None) -> LlamaCpp` — composes install detection with server discovery; both sets of probes are stored in `LlamaCpp.probes`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llamacpp_server.py`:

```python
from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp import detect_llamacpp
from llamafit.llamacpp.server import FakeHttp, candidate_ports, discover_servers

HEALTH = {"status": "ok"}
MODELS = {"data": [{"id": "qwen3-coder-next", "object": "model"}]}
PROPS = {"default_generation_settings": {"n_ctx": 262144}, "build_info": "b10867-f3f1a8f27"}


def test_candidate_ports_env_first() -> None:
    assert candidate_ports({"LLAMA_SERVER_PORT": "9000"}) == [9000, 8080, 8081, 8098]
    assert candidate_ports({}) == [8080, 8081, 8098]


def test_discover_finds_one_server() -> None:
    http = FakeHttp({
        "http://127.0.0.1:8080/health": HEALTH,
        "http://127.0.0.1:8080/v1/models": MODELS,
        "http://127.0.0.1:8080/props": PROPS,
    })
    servers = discover_servers(http, [8080, 8081])
    assert len(servers) == 1
    server = servers[0]
    assert server.url == "http://127.0.0.1:8080"
    assert server.model == "qwen3-coder-next"
    assert server.n_ctx == 262144
    assert server.build == "b10867-f3f1a8f27"


def test_discover_ignores_non_llama_services() -> None:
    http = FakeHttp({"http://127.0.0.1:8080/health": {"status": "degraded"}})
    assert discover_servers(http, [8080]) == []


def test_detect_llamacpp_composes(tmp_path: object) -> None:
    http = FakeHttp({"http://127.0.0.1:8080/health": HEALTH, "http://127.0.0.1:8080/v1/models": MODELS,
                     "http://127.0.0.1:8080/props": PROPS})
    result = detect_llamacpp(FakeRunner({}), os_name="linux", env={"PATH": ""}, http=http)
    assert not result.installed
    assert [s.model for s in result.running_servers] == ["qwen3-coder-next"]
    assert any(p.name == "server:8080" and p.ok for p in result.probes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_llamacpp_server.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Implement `llamacpp/server.py`**

```python
"""Find ``llama-server`` instances already running on this machine."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from llamafit.models.host import Probe
from llamafit.models.llamacpp import RunningServer

DEFAULT_PORTS = [8080, 8081, 8098]


class HttpClient(Protocol):
    """Minimal HTTP GET returning parsed JSON, or ``None`` on any failure."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url`` and parse JSON; never raise."""
        ...


class HttpxClient:
    """Real HTTP client with short timeouts; connection refused is just ``None``."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url``; returns ``None`` for network errors, non-200 or invalid JSON."""
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None


@dataclass
class FakeHttp:
    """Canned JSON responses keyed by URL."""

    responses: Mapping[str, Any]

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Return the canned response or ``None``."""
        return self.responses.get(url)


def candidate_ports(env: Mapping[str, str]) -> list[int]:
    """``LLAMA_SERVER_PORT`` first when set, then the usual llama.cpp ports."""
    ports: list[int] = []
    configured = env.get("LLAMA_SERVER_PORT")
    if configured and configured.isdigit():
        ports.append(int(configured))
    return ports + [p for p in DEFAULT_PORTS if p not in ports]


def discover_servers(http: HttpClient, ports: Iterable[int]) -> list[RunningServer]:
    """Probe each port for a llama-server and describe what it is serving."""
    return [server for server, _ in _discover(http, ports) if server]


def _discover(http: HttpClient, ports: Iterable[int]) -> list[tuple[RunningServer | None, Probe]]:
    results: list[tuple[RunningServer | None, Probe]] = []
    for port in ports:
        start = time.perf_counter()
        base = f"http://127.0.0.1:{port}"
        health = http.get_json(f"{base}/health")
        duration = int((time.perf_counter() - start) * 1000)
        if not isinstance(health, dict) or health.get("status") != "ok":
            results.append((None, Probe(name=f"server:{port}", ok=False, duration_ms=duration,
                                        error="no llama-server answering")))
            continue
        model: str | None = None
        models = http.get_json(f"{base}/v1/models")
        if isinstance(models, dict) and models.get("data"):
            model = str(models["data"][0].get("id"))
        n_ctx: int | None = None
        build: str | None = None
        props = http.get_json(f"{base}/props")
        if isinstance(props, dict):
            settings = props.get("default_generation_settings") or {}
            n_ctx = int(settings["n_ctx"]) if settings.get("n_ctx") else None
            build = str(props["build_info"]) if props.get("build_info") else None
        results.append((RunningServer(url=base, model=model, n_ctx=n_ctx, build=build),
                        Probe(name=f"server:{port}", ok=True, duration_ms=duration)))
    return results


def discover_with_probes(http: HttpClient, ports: Iterable[int]) -> tuple[list[RunningServer], list[Probe]]:
    """Like ``discover_servers`` but also returns the probe records for ``doctor``."""
    pairs = _discover(http, ports)
    return [s for s, _ in pairs if s], [p for _, p in pairs]
```

- [ ] **Step 4: Implement `detect_llamacpp()` in `llamacpp/__init__.py`**

```python
"""llama.cpp integration: installation detection and running-server discovery."""

from __future__ import annotations

import os
from collections.abc import Mapping

from llamafit.hardware import current_os
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.llamacpp.detect import detect_install
from llamafit.llamacpp.server import HttpClient, HttpxClient, candidate_ports, discover_with_probes
from llamafit.models.host import OsName
from llamafit.models.llamacpp import LlamaCpp


def detect_llamacpp(
    runner: Runner | None = None,
    *,
    os_name: OsName | None = None,
    env: Mapping[str, str] | None = None,
    http: HttpClient | None = None,
) -> LlamaCpp:
    """Detect the installation and any running servers; never raises."""
    runner = runner or SubprocessRunner()
    os_name = os_name or current_os()
    env = os.environ if env is None else env
    http = http or HttpxClient()
    llamacpp, probes = detect_install(runner, os_name, env=env)
    servers, server_probes = discover_with_probes(http, candidate_ports(env))
    llamacpp.running_servers = servers
    llamacpp.probes = [*probes, *server_probes]
    return llamacpp


__all__ = ["detect_llamacpp"]
```

- [ ] **Step 5: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_llamacpp_server.py -v && ruff check . && mypy`
Expected: PASS, clean, clean.

- [ ] **Step 6: Commit**

```bash
git add src/llamafit/llamacpp tests/unit/test_llamacpp_server.py
git commit -m "feat: running llama-server discovery and detect_llamacpp()"
```

---

### Task 11: Services — `scan_system()` and `diagnose()`

**Files:**
- Create: `src/llamafit/services/__init__.py`, `src/llamafit/services/scan.py`, `src/llamafit/services/doctor.py`
- Test: `tests/unit/test_services.py`

**Interfaces:**
- Consumes: `scan()` (Task 8), `detect_llamacpp()` (Task 10), `SystemReport`, `__version__`.
- Produces:
  - `scan_system(*, runner: Runner | None = None, http: HttpClient | None = None, os_name: OsName | None = None, env: Mapping[str, str] | None = None, measure_bandwidth: bool = True, cpuinfo_provider=None, vm_provider=None) -> SystemReport`
  - `Finding(level: Literal["ok", "warn", "error"], title: str, detail: str, hint: str | None = None)`
  - `Diagnosis(report: SystemReport, findings: list[Finding])` with property `worst_level`.
  - `diagnose(report: SystemReport) -> Diagnosis`
  - `PROBE_HINTS: dict[str, str]` — what to do when a named probe fails.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_services.py`:

```python
from datetime import datetime, timezone

from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp.server import FakeHttp
from llamafit.models import Cpu, Gpu, Host, LlamaCpp, Memory, Probe, SystemReport
from llamafit.services.doctor import diagnose
from llamafit.services.scan import scan_system
from tests.fixtures.reference_machine import reference_cpuinfo, reference_runner, reference_vm


def test_scan_system_returns_report_with_version() -> None:
    report = scan_system(runner=reference_runner(), http=FakeHttp({}), os_name="windows",
                         env={"PATH": ""}, measure_bandwidth=False,
                         cpuinfo_provider=reference_cpuinfo, vm_provider=reference_vm)
    assert report.version
    assert report.host.primary_gpu is not None
    assert report.llamacpp.installed is False


def report_with(llamacpp: LlamaCpp, *, gpus: list[Gpu] | None = None, probes: list[Probe] | None = None,
                bandwidth_source: str = "measured") -> SystemReport:
    host = Host(
        os="windows", os_version="11", arch="x86_64",
        cpu=Cpu(model="i9", physical_cores=24, logical_cores=32, performance_cores=8),
        memory=Memory(total_bytes=128 * 1024**3, available_bytes=100 * 1024**3, bandwidth_gbps=60,
                      bandwidth_source=bandwidth_source),  # type: ignore[arg-type]
        gpus=gpus if gpus is not None else [Gpu(index=0, vendor="nvidia", name="RTX 4060",
                                                 vram_total_bytes=8 * 1024**3, vram_used_bytes=0,
                                                 backend_hint="cuda")],
        probes=probes or [], scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    return SystemReport(host=host, llamacpp=llamacpp, version="0.1.0a1")


def test_diagnose_missing_llamacpp_is_an_error_with_hint() -> None:
    diagnosis = diagnose(report_with(LlamaCpp(installed=False, problems=["llama.cpp not found: ..."])))
    errors = [f for f in diagnosis.findings if f.level == "error"]
    assert errors and "llama.cpp" in errors[0].title
    assert errors[0].hint and "install" in errors[0].hint.lower()
    assert diagnosis.worst_level == "error"


def test_diagnose_backend_mismatch_is_a_warning() -> None:
    llamacpp = LlamaCpp(installed=True, path="C:/llama.cpp/bin", build=10867, backends=["cpu"])
    diagnosis = diagnose(report_with(llamacpp))
    titles = [f.title for f in diagnosis.findings if f.level == "warn"]
    assert any("CUDA" in t for t in titles)


def test_diagnose_failed_probe_gets_hint() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=1, backends=["cuda"])
    probes = [Probe(name="nvidia-smi", ok=False, duration_ms=1, error="nvidia-smi: not found")]
    diagnosis = diagnose(report_with(llamacpp, probes=probes))
    finding = next(f for f in diagnosis.findings if f.title.startswith("Probe nvidia-smi"))
    assert finding.level == "warn" and finding.hint and "driver" in finding.hint.lower()


def test_diagnose_all_good() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cuda", "cpu"])
    diagnosis = diagnose(report_with(llamacpp))
    assert diagnosis.worst_level == "ok"
    assert any(f.level == "ok" and "llama.cpp" in f.title for f in diagnosis.findings)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_services.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `services/scan.py`**

`services/__init__.py`: `"""Operations the interfaces expose; each returns a pydantic model and never prints."""`

```python
"""``scan_system``: host scan plus llama.cpp detection in one report."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from llamafit import __version__
from llamafit.hardware import scan
from llamafit.hardware.runner import Runner
from llamafit.llamacpp import detect_llamacpp
from llamafit.llamacpp.server import HttpClient
from llamafit.models.host import OsName
from llamafit.models.report import SystemReport


def scan_system(
    *,
    runner: Runner | None = None,
    http: HttpClient | None = None,
    os_name: OsName | None = None,
    env: Mapping[str, str] | None = None,
    measure_bandwidth: bool = True,
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
    vm_provider: Callable[[], tuple[int, int]] | None = None,
) -> SystemReport:
    """Scan the host and detect llama.cpp; all arguments exist so tests can inject fakes."""
    host = scan(runner, os_name=os_name, measure_bandwidth=measure_bandwidth,
                cpuinfo_provider=cpuinfo_provider, vm_provider=vm_provider)
    llamacpp = detect_llamacpp(runner, os_name=host.os, env=env, http=http)
    return SystemReport(host=host, llamacpp=llamacpp, version=__version__)
```

- [ ] **Step 4: Implement `services/doctor.py`**

```python
"""``diagnose``: turn a report into findings a person can act on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from llamafit.models.report import SystemReport
from llamafit.units import format_bytes

Level = Literal["ok", "warn", "error"]
_ORDER: dict[Level, int] = {"ok": 0, "warn": 1, "error": 2}

PROBE_HINTS: dict[str, str] = {
    "nvidia-smi": "Install or repair the NVIDIA driver; nvidia-smi ships with it and must be on PATH.",
    "rocm-smi": "Install ROCm (Linux) to expose AMD VRAM figures; without it the GPU is listed by name only.",
    "wmi-video": "PowerShell could not list video controllers; run from a normal user session.",
    "lspci": "Install pciutils to list GPUs by name when no vendor tool is available.",
    "memory-modules": "DDR type and speed were not readable; on Linux run once with sudo (dmidecode) "
                      "or accept the measured bandwidth instead.",
    "cpuinfo": "py-cpuinfo failed; the CPU model and instruction sets are unknown.",
    "sysctl-perflevel": "Could not read performance-core count from sysctl.",
    "llama-server --version": "llama-server exists but did not report a version; the binary may be broken.",
    "system-profiler": "system_profiler failed; GPU information is unavailable.",
}

_BACKEND_FOR_VENDOR = {"nvidia": ("cuda", "CUDA"), "amd": ("hip", "ROCm/HIP"), "apple": ("metal", "Metal")}
_LOW_DISK_BYTES = 20 * 1024**3


class Finding(BaseModel):
    """One line of ``doctor`` output."""

    level: Level
    title: str
    detail: str
    hint: str | None = None


class Diagnosis(BaseModel):
    """All findings for a report, worst first."""

    report: SystemReport
    findings: list[Finding] = Field(default_factory=list)

    @property
    def worst_level(self) -> Level:
        """``error`` if any finding is an error, else ``warn``, else ``ok``."""
        worst = max((_ORDER[f.level] for f in self.findings), default=0)
        return next(level for level, rank in _ORDER.items() if rank == worst)


def diagnose(report: SystemReport) -> Diagnosis:
    """Apply the diagnostic rules to a report."""
    findings: list[Finding] = []
    host, llamacpp = report.host, report.llamacpp

    if llamacpp.installed:
        findings.append(Finding(level="ok", title="llama.cpp installed",
                                detail=f"build {llamacpp.build or 'unknown'} at {llamacpp.path}, "
                                       f"backends: {', '.join(llamacpp.backends) or 'none detected'}"))
    else:
        findings.append(Finding(level="error", title="llama.cpp not found",
                                detail="; ".join(llamacpp.problems) or "no llama-server binary was found",
                                hint="Install llama.cpp (LlamaFit phase 2 will do this: `llamafit install "
                                     "llama.cpp`); until then download a release from "
                                     "https://github.com/ggml-org/llama.cpp/releases and put its bin "
                                     "directory on PATH or in LLAMA_CPP_PATH."))

    for gpu in host.gpus:
        expected = _BACKEND_FOR_VENDOR.get(gpu.vendor)
        if expected and llamacpp.installed and expected[0] not in llamacpp.backends \
                and "vulkan" not in llamacpp.backends:
            findings.append(Finding(level="warn", title=f"{gpu.name}: no {expected[1]} backend in llama.cpp",
                                    detail=f"llama.cpp was built with: {', '.join(llamacpp.backends) or 'unknown'}",
                                    hint=f"Install a llama.cpp build with the {expected[1]} or Vulkan backend "
                                         f"to use this GPU."))
        if gpu.vram_total_bytes is None and not host.unified_memory:
            findings.append(Finding(level="warn", title=f"{gpu.name}: VRAM size unknown",
                                    detail="the vendor tool that reports memory was not available",
                                    hint=PROBE_HINTS["nvidia-smi"] if gpu.vendor == "nvidia" else PROBE_HINTS["rocm-smi"]))
    if not host.gpus:
        findings.append(Finding(level="warn", title="No GPU detected",
                                detail="models will run on the CPU only",
                                hint="If a GPU is present, check that its driver tools are installed."))

    if host.memory.bandwidth_source == "assumed":
        findings.append(Finding(level="warn", title="RAM bandwidth assumed",
                                detail=f"using {host.memory.bandwidth_gbps} GB/s as a default",
                                hint="Install numpy (`pip install llamafit[fast]`) so LlamaFit can measure it."))

    for probe in [*host.probes, *llamacpp.probes]:
        if not probe.ok and not probe.name.startswith("server:"):
            findings.append(Finding(level="warn", title=f"Probe {probe.name} failed",
                                    detail=probe.error or "unknown error", hint=PROBE_HINTS.get(probe.name)))

    for server in llamacpp.running_servers:
        findings.append(Finding(level="ok", title=f"llama-server running at {server.url}",
                                detail=f"model {server.model or 'unknown'}, context {server.n_ctx or 'unknown'}"))

    for disk in host.disks:
        if disk.free_bytes < _LOW_DISK_BYTES:
            findings.append(Finding(level="warn", title=f"Low disk space on {disk.path}",
                                    detail=f"{format_bytes(disk.free_bytes)} free",
                                    hint="Most useful models need 5 to 120 GB; free space or change the "
                                         "downloads directory."))

    findings.sort(key=lambda f: -_ORDER[f.level])
    return Diagnosis(report=report, findings=findings)
```

- [ ] **Step 5: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_services.py -v && ruff check . && mypy`
Expected: PASS, clean, clean.

- [ ] **Step 6: Commit**

```bash
git add src/llamafit/services tests/unit/test_services.py
git commit -m "feat: scan_system and diagnose services"
```

---

### Task 12: CLI — `llamafit system` and `llamafit doctor`

**Files:**
- Modify: `src/llamafit/cli/app.py`
- Create: `src/llamafit/cli/render.py`, `src/llamafit/cli/system_cmd.py`, `src/llamafit/cli/doctor_cmd.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `scan_system()`, `diagnose()`, `LlamaFitError`, `format_bytes`.
- Produces:
  - `CliState(json_output: bool, verbose: bool, no_color: bool)` stored in `ctx.obj`.
  - `app: typer.Typer`; `main() -> None` (entry point): runs `app`, maps `LlamaFitError` to exit code 1 (`NotInstalledError`, `ProbeError` to 2), prints `render()` to stderr.
  - `render_host(host) -> Table`, `render_llamacpp(llamacpp) -> Table`, `render_probes(probes) -> Table`, `render_findings(findings) -> Table` (Rich tables).
  - Commands: `system` (`--no-measure` skips the bandwidth probe), `doctor`. Both honour the global `--json`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli.py`:

```python
import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.models import Cpu, Gpu, Host, LlamaCpp, Memory, SystemReport

runner = CliRunner()


def fake_report() -> SystemReport:
    host = Host(os="windows", os_version="11", arch="x86_64",
                cpu=Cpu(model="Intel i9-14900KF", physical_cores=24, logical_cores=32, performance_cores=8,
                        isa=["avx2", "avx512"]),
                memory=Memory(total_bytes=128 * 1024**3, available_bytes=100 * 1024**3, type="DDR5",
                              speed_mts=4200, channels=2, bandwidth_gbps=67.2, bandwidth_source="estimated"),
                gpus=[Gpu(index=0, vendor="nvidia", name="NVIDIA GeForce RTX 4060",
                          vram_total_bytes=8188 * 1024**2, vram_used_bytes=550 * 1024**2,
                          bandwidth_gbps=272, compute_tflops_fp16=15, backend_hint="cuda", driver="610.88")],
                scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    llamacpp = LlamaCpp(installed=True, path="D:/llama.cpp/bin", build=10867, commit="f3f1a8f27",
                        backends=["cuda", "rpc", "cpu"])
    return SystemReport(host=host, llamacpp=llamacpp, version="0.1.0a1")


@pytest.fixture(autouse=True)
def patch_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.system_cmd.scan_system", lambda **kwargs: fake_report())
    monkeypatch.setattr("llamafit.cli.doctor_cmd.scan_system", lambda **kwargs: fake_report())


def test_system_table_mentions_gpu_and_memory() -> None:
    result = runner.invoke(app, ["system"])
    assert result.exit_code == 0, result.output
    assert "RTX 4060" in result.output
    assert "DDR5" in result.output
    assert "67.2" in result.output
    assert "b10867" in result.output


def test_system_json_is_the_report() -> None:
    result = runner.invoke(app, ["--json", "system"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["host"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4060"
    assert data["llamacpp"]["build"] == 10867
    assert data["version"] == "0.1.0a1"


def test_doctor_lists_findings_and_exit_code_zero_when_no_error() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "llama.cpp installed" in result.output


def test_doctor_json_has_findings() -> None:
    result = runner.invoke(app, ["--json", "doctor"])
    data = json.loads(result.output)
    assert isinstance(data["findings"], list)
    assert data["findings"][0]["level"] in {"ok", "warn", "error"}


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llamafit" in result.output
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_cli.py -v`
Expected: FAIL (the placeholder `app.py` has no `app`).

- [ ] **Step 3: Implement `cli/app.py`**

```python
"""Typer application: global options, error rendering, command registration."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import typer
from rich.console import Console

from llamafit import __version__
from llamafit.errors import LlamaFitError, NotInstalledError, ProbeError

app = typer.Typer(
    name="llamafit",
    help="Find, size, install and verify open-weight LLMs for llama.cpp on your own machine.",
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)


@dataclass
class CliState:
    """Options shared by every command, stored in ``ctx.obj``."""

    json_output: bool = False
    verbose: bool = False
    no_color: bool = False

    @property
    def console(self) -> Console:
        """A console for normal output, respecting ``--no-color``."""
        return Console(no_color=self.no_color, highlight=False)

    @property
    def err_console(self) -> Console:
        """A console for errors, on stderr."""
        return Console(stderr=True, no_color=self.no_color, highlight=False)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llamafit {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of tables."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log details and show tracebacks."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colours."),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True,
                                 help="Print the version and exit."),
) -> None:
    """LlamaFit command-line interface."""
    ctx.obj = CliState(json_output=json_output, verbose=verbose, no_color=no_color)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    """Entry point: run the app and turn LlamaFit errors into messages and exit codes."""
    try:
        app(standalone_mode=True)
    except LlamaFitError as exc:  # raised inside commands before Typer's own handling
        Console(stderr=True).print(f"[red]{exc.render()}[/red]")
        code = 2 if isinstance(exc, (NotInstalledError, ProbeError)) else 1
        sys.exit(code)


from llamafit.cli import doctor_cmd, system_cmd  # noqa: E402  (registers commands on import)

__all__ = ["CliState", "app", "main", "system_cmd", "doctor_cmd"]
```

- [ ] **Step 4: Implement `cli/render.py`**

```python
"""Rich tables for the host, llama.cpp status, probes and findings."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table

from llamafit.models.host import Host, Probe
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.doctor import Finding
from llamafit.units import format_bytes

_LEVEL_STYLE = {"ok": "green", "warn": "yellow", "error": "red"}


def render_host(host: Host) -> Table:
    """A two-column table with everything the scan found."""
    table = Table(title="Host", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("OS", f"{host.os} {host.os_version} ({host.arch})")
    cores = f"{host.cpu.physical_cores} cores / {host.cpu.logical_cores} threads"
    if host.cpu.performance_cores:
        cores += f", {host.cpu.performance_cores} performance cores"
    table.add_row("CPU", f"{host.cpu.model}; {cores}; {' '.join(host.cpu.isa) or 'isa unknown'}")
    mem = host.memory
    details = " ".join(x for x in (mem.type, f"{mem.speed_mts} MT/s" if mem.speed_mts else None,
                                   f"{mem.channels}-channel" if mem.channels else None) if x)
    bandwidth = f"{mem.bandwidth_gbps} GB/s ({mem.bandwidth_source})" if mem.bandwidth_gbps else "unknown"
    table.add_row("Memory", f"{format_bytes(mem.total_bytes)} total, {format_bytes(mem.available_bytes)} "
                            f"available; {details or 'type unknown'}; bandwidth {bandwidth}")
    if not host.gpus:
        table.add_row("GPU", "none detected")
    for gpu in host.gpus:
        vram = (f"{format_bytes(gpu.vram_total_bytes)} VRAM, {format_bytes(gpu.vram_free_bytes)} free"
                if gpu.vram_total_bytes else "VRAM unknown")
        specs = ", ".join(x for x in (f"{gpu.bandwidth_gbps} GB/s" if gpu.bandwidth_gbps else None,
                                      f"{gpu.compute_tflops_fp16} TFLOPS fp16" if gpu.compute_tflops_fp16 else None,
                                      f"driver {gpu.driver}" if gpu.driver else None) if x)
        table.add_row(f"GPU {gpu.index}", f"{gpu.name} ({gpu.backend_hint}); {vram}; {specs or 'no specs'}")
    if host.unified_memory:
        table.add_row("Memory pool", "unified (GPU shares system memory)")
    for disk in host.disks:
        table.add_row(f"Disk {disk.path}", f"{format_bytes(disk.free_bytes)} free of {format_bytes(disk.total_bytes)}")
    return table


def render_llamacpp(llamacpp: LlamaCpp) -> Table:
    """A two-column table describing the installation."""
    table = Table(title="llama.cpp", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    if not llamacpp.installed:
        table.add_row("Installed", "no")
    else:
        build = f"b{llamacpp.build}" if llamacpp.build else "unknown build"
        commit = f" ({llamacpp.commit})" if llamacpp.commit else ""
        table.add_row("Installed", f"yes, {build}{commit} at {llamacpp.path}")
        table.add_row("Backends", ", ".join(llamacpp.backends) or "none detected")
        table.add_row("Local models", str(len(llamacpp.local_models)))
    for server in llamacpp.running_servers:
        table.add_row("Running", f"{server.url}: {server.model or 'unknown model'}, "
                                 f"context {server.n_ctx or 'unknown'}")
    for problem in llamacpp.problems:
        table.add_row("Problem", problem)
    return table


def render_probes(probes: Iterable[Probe]) -> Table:
    """Probe-by-probe outcome."""
    table = Table(title="Probes", box=None, pad_edge=False)
    table.add_column("probe", style="bold")
    table.add_column("result")
    table.add_column("ms", justify="right")
    for probe in probes:
        status = "[green]ok[/green]" if probe.ok else f"[yellow]failed[/yellow] {probe.error or ''}"
        table.add_row(probe.name, status, str(probe.duration_ms))
    return table


def render_findings(findings: Iterable[Finding]) -> Table:
    """Findings with level colouring and hints."""
    table = Table(title="Findings", box=None, pad_edge=False)
    table.add_column("level")
    table.add_column("finding")
    for finding in findings:
        style = _LEVEL_STYLE[finding.level]
        text = f"[bold]{finding.title}[/bold]\n{finding.detail}"
        if finding.hint:
            text += f"\n[dim]Hint: {finding.hint}[/dim]"
        table.add_row(f"[{style}]{finding.level.upper()}[/{style}]", text)
    return table
```

- [ ] **Step 5: Implement `cli/system_cmd.py` and `cli/doctor_cmd.py`**

`cli/system_cmd.py`:

```python
"""``llamafit system``: print the host scan and llama.cpp status."""

from __future__ import annotations

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_host, render_llamacpp
from llamafit.services.scan import scan_system


@app.command("system")
def system_command(
    ctx: typer.Context,
    no_measure: bool = typer.Option(False, "--no-measure", help="Skip the RAM bandwidth measurement."),
) -> None:
    """Show what this machine has: CPU, memory, GPUs, disks and llama.cpp."""
    state: CliState = ctx.obj
    report = scan_system(measure_bandwidth=not no_measure)
    if state.json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    console = state.console
    console.print(render_host(report.host))
    console.print()
    console.print(render_llamacpp(report.llamacpp))
```

`cli/doctor_cmd.py`:

```python
"""``llamafit doctor``: probe-by-probe report and actionable findings."""

from __future__ import annotations

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_findings, render_probes
from llamafit.services.doctor import diagnose
from llamafit.services.scan import scan_system


@app.command("doctor")
def doctor_command(ctx: typer.Context) -> None:
    """Explain what was detected, what failed, and what would unlock more."""
    state: CliState = ctx.obj
    report = scan_system()
    diagnosis = diagnose(report)
    if state.json_output:
        typer.echo(diagnosis.model_dump_json(indent=2))
    else:
        console = state.console
        console.print(render_probes([*report.host.probes, *report.llamacpp.probes]))
        console.print()
        console.print(render_findings(diagnosis.findings))
    if diagnosis.worst_level == "error":
        raise typer.Exit(code=2)
```

- [ ] **Step 6: Run the tests, lint and type-check**

Run: `pytest tests/unit/test_cli.py -v && ruff check . && mypy`
Expected: PASS, clean, clean. If ruff complains about the import placement in `app.py` (`E402` is suppressed inline; `I001` may want the import sorted), keep the import at the bottom with the `noqa` comment: it must run after `app` exists to avoid a circular import. Then run the real thing: `llamafit system`, `llamafit --json system`, `llamafit doctor` on the reference machine and read the output critically.

- [ ] **Step 7: Commit**

```bash
git add src/llamafit/cli tests/unit/test_cli.py
git commit -m "feat: llamafit system and doctor commands with Rich tables and --json"
```

---

### Task 13: Documentation, community files and release readiness

**Files:**
- Modify: `docs/cli.md`, `docs/platform-support.md`, `docs/development.md`, `README.md`, `CHANGELOG.md` (all exist; this task brings them in line with the code)
- Create: `tests/unit/test_docs.py`, `scripts/record_fixtures.py`

**Interfaces:**
- Consumes: the commands and flags delivered by Task 12; the probe names and hints from Task 11.
- Produces: user-facing documentation that matches the code exactly (every flag and probe name in the docs must exist in the code; the test in Step 1 enforces it), and the fixture recorder that `docs/development.md` promises.

- [ ] **Step 1: Write the failing documentation test**

`tests/unit/test_docs.py`:

```python
"""Documentation must name every command and flag the CLI actually has."""

from pathlib import Path

from typer.main import get_command

from llamafit.cli.app import app

ROOT = Path(__file__).resolve().parents[2]


def test_cli_doc_mentions_every_command_and_option() -> None:
    text = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    command = get_command(app)
    for name in command.commands:  # type: ignore[attr-defined]
        assert f"`llamafit {name}`" in text, f"docs/cli.md lacks {name}"
    for option in ("--json", "--verbose", "--no-color", "--version", "--no-measure"):
        assert option in text, f"docs/cli.md lacks {option}"


def test_platform_doc_mentions_every_probe_hint() -> None:
    from llamafit.services.doctor import PROBE_HINTS

    text = (ROOT / "docs" / "platform-support.md").read_text(encoding="utf-8")
    for probe in PROBE_HINTS:
        assert probe in text, f"docs/platform-support.md lacks probe {probe}"


def test_readme_shows_the_two_commands_that_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "llamafit system" in text and "llamafit doctor" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_docs.py -v`
Expected: FAIL with `FileNotFoundError` for `docs/cli.md`.

- [ ] **Step 3: Bring `docs/cli.md` in line with the code**

The page already documents every planned command with a phase marker. For the two commands that now exist:

1. Replace the example block under `llamafit system` with the real output of `llamafit system` on the reference machine (trim the disk rows to two).
2. Check that every option the code accepts appears in the tables: `--json`, `--verbose`/`-v`, `--no-color`, `--version` (global) and `--no-measure` (`system`). Remove nothing that is documented for later phases; the test only requires that what exists is documented.
3. Confirm the exit codes match `cli/app.py` and `doctor_cmd.py` (0, 1, 2).

- [ ] **Step 4: Bring `docs/platform-support.md` in line with the code**

For each probe name in `PROBE_HINTS` (`services/doctor.py`) and each command constant in `hardware/cpu.py`, `hardware/memory.py`, `hardware/gpu.py` and `llamacpp/detect.py`, check the probe table: the probe name is exact, the command column shows the exact argument list, and the "Without it" column matches what the code does when the probe fails. Fix any difference in the documentation, not in the code, unless the code is wrong.

- [ ] **Step 5: Write `scripts/record_fixtures.py`**

`docs/development.md` promises this script. It runs every probe command on the current machine and prints a fixture module for `tests/fixtures/`.

```python
"""Record this machine's probe outputs as a test fixture module.

Usage:
    python scripts/record_fixtures.py my-machine > tests/fixtures/my_machine.py

The generated module exposes ``runner()``, ``cpuinfo()`` and ``vm()`` in the same shape as
``tests/fixtures/reference_machine.py`` so a scan test can reproduce the machine without it.
"""

from __future__ import annotations

import json
import pprint
import sys
from collections.abc import Mapping, Sequence

import psutil

from llamafit.hardware import current_os
from llamafit.hardware import gpu as gpu_probes
from llamafit.hardware import memory as memory_probes
from llamafit.hardware.runner import SubprocessRunner

# Private constants are imported on purpose: the recorder must run exactly what the probes run.
_COMMON: dict[str, Sequence[str]] = {
    "nvidia-smi": gpu_probes._NVIDIA_CMD,  # noqa: SLF001
    "rocm-smi": gpu_probes._ROCM_CMD,  # noqa: SLF001
}
_BY_OS: dict[str, dict[str, Sequence[str]]] = {
    "windows": {
        "wmi-video": gpu_probes._WMI_CMD,  # noqa: SLF001
        "memory-modules": memory_probes._WINDOWS_MODULES_CMD,  # noqa: SLF001
    },
    "macos": {
        "system-profiler": gpu_probes._APPLE_CMD,  # noqa: SLF001
        "memory-modules": memory_probes._MACOS_MEMORY_CMD,  # noqa: SLF001
        "sysctl-perflevel": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
    },
    "linux": {
        "lspci": gpu_probes._LSPCI_CMD,  # noqa: SLF001
        "memory-modules": memory_probes._LINUX_MEMORY_CMD,  # noqa: SLF001
    },
}


def collect(runner: SubprocessRunner, os_name: str) -> dict[str, str]:
    """Run every probe command for this OS and keep the stdout of the ones that worked."""
    commands = {**_COMMON, **_BY_OS[os_name]}
    responses: dict[str, str] = {}
    for name, argv in commands.items():
        result = runner.run(argv, timeout=20.0)
        if result.ok:
            responses[" ".join(argv)] = result.stdout
        else:
            print(f"# {name}: skipped ({result.error or f'exit code {result.returncode}'})", file=sys.stderr)
    return responses


def build_module(name: str, responses: Mapping[str, str], cpuinfo: Mapping[str, object],
                 vm: tuple[int, int], os_name: str) -> str:
    """Render the fixture module source for the recorded data."""
    lines = [
        f'"""Recorded probe outputs from {name} ({os_name}). Generated by scripts/record_fixtures.py."""',
        "",
        "from llamafit.hardware.runner import FakeRunner",
        "",
        f"OS_NAME = {os_name!r}",
        "",
        "RESPONSES = " + pprint.pformat(dict(responses), width=100, sort_dicts=True),
        "",
        "CPUINFO = " + pprint.pformat(dict(cpuinfo), width=100, sort_dicts=True),
        "",
        f"VM = {vm!r}",
        "",
        "",
        "def runner() -> FakeRunner:",
        "    return FakeRunner(RESPONSES)",
        "",
        "",
        "def cpuinfo() -> dict[str, object]:",
        "    return dict(CPUINFO)",
        "",
        "",
        "def vm() -> tuple[int, int]:",
        "    return VM",
        "",
    ]
    return chr(10).join(lines)


def main(argv: Sequence[str]) -> int:
    """Entry point."""
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    import cpuinfo as cpuinfo_lib

    os_name = current_os()
    info = cpuinfo_lib.get_cpu_info()
    cpu = {"brand_raw": info.get("brand_raw", ""), "flags": list(info.get("flags", []))}
    memory = psutil.virtual_memory()
    responses = collect(SubprocessRunner(), os_name)
    sys.stdout.write(build_module(argv[1], responses, cpu, (int(memory.total), int(memory.available)), os_name))
    json.dumps(responses)  # fail early if anything recorded is not plain text
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

Add to `tests/unit/test_record_fixtures.py`:

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("record_fixtures", ROOT / "scripts" / "record_fixtures.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_module_is_importable_python() -> None:
    script = load_script()
    source = script.build_module("box", {"nvidia-smi -L": "GPU 0"}, {"brand_raw": "x", "flags": ["avx2"]},
                                 (16, 8), "linux")
    namespace: dict[str, object] = {}
    exec(compile(source, "fixture", "exec"), namespace)  # noqa: S102 - generated by our own code
    assert namespace["runner"]().run(["nvidia-smi", "-L"]).stdout == "GPU 0"  # type: ignore[operator]
    assert namespace["vm"]() == (16, 8)  # type: ignore[operator]
```

Run: `pytest tests/unit/test_record_fixtures.py -v && python scripts/record_fixtures.py reference-check > /dev/null`
Expected: PASS; the script prints skipped probes to stderr for tools this machine lacks and a module to stdout.

- [ ] **Step 6: Check the community files**

`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md` and the `.github` templates already exist. Read `CONTRIBUTING.md` and `docs/development.md` once against the real workflow you just used (commands, extras, markers) and correct anything that differs.

- [ ] **Step 7: Bring `README.md` in line with what now exists**

The README was written before implementation and describes the whole roadmap. Edit, do not replace:

1. Set the **Status** paragraph to: phase 1A done (`llamafit system`, `llamafit doctor` work on Windows, macOS and Linux); catalog, scoring, TUI and web dashboard follow.
2. In the **Install** section replace "coming soon" wording with the real command `pip install llamafit` and the optional `[fast]` extra.
3. In the **Use** section make sure the three lines below appear exactly, with their comments:

```
llamafit system                 # CPU, memory, GPUs, disks, llama.cpp installation
llamafit doctor                 # what was detected, what failed, what would help
llamafit --json system          # the same as JSON for scripts
```

4. In the roadmap table mark phase 1A as done and leave the other rows unchanged.
5. Confirm the **Documentation** list links to `docs/cli.md`, `docs/platform-support.md`, `docs/development.md` and `CONTRIBUTING.md`, and that every linked file exists.

- [ ] **Step 8: Update `CHANGELOG.md`**

Under `[Unreleased]` → `### Added`, replace the single line with:

```markdown
- Host scan: OS, CPU (cores, performance cores, instruction sets), memory (totals, DDR facts,
  measured or estimated bandwidth), GPUs (NVIDIA, AMD, Apple, generic), disks.
- llama.cpp detection: binaries, build, backends, local GGUF files, running servers.
- `llamafit system` and `llamafit doctor` with Rich tables, `--json`, hints and exit codes.
- Documentation: CLI reference, platform support, development guide, contributing guide.
```

- [ ] **Step 9: Run everything**

Run: `pytest --cov -m "not hardware" && ruff check . && ruff format --check . && mypy`
Expected: all tests PASS including `test_docs.py`, coverage at or above 85 percent, ruff and mypy clean. Then on the reference machine: `pytest -m hardware`, `llamafit system`, `llamafit doctor`, and read both outputs against `docs/cli.md`; fix any wording that differs.

- [ ] **Step 10: Commit and push; check CI**

```bash
git add -A
git commit -m "docs: CLI reference, platform support, development and community files; README"
git push
```

Open the Actions tab on GitHub and confirm the six matrix jobs are green. Fix anything platform-specific that only CI reveals (path separators, PowerShell availability) in a follow-up commit before starting plan 1B.

---

## Plan self-review (done while writing)

- **Spec coverage:** section 4 (host detection) → Tasks 5 to 8; section 5.1 (llama.cpp detection) → Tasks 9 and 10; section 13.1 `system` and `doctor` → Task 12; section 14 paths → Task 2; section 17 error handling → Tasks 2, 4 and 12; section 18 testing (fixtures, golden reference machine, CI matrix) → Tasks 1, 8 and 13; section 19 repository files → Tasks 1 and 13. Hardware profiles (spec 4.4) and simulation flags are deferred to plan 1C, where the scoring that consumes them lives; `--profile`, `--memory`, `--ram`, `--cpu-cores`, `--max-context` are therefore not in this plan's CLI.
- **Placeholders:** none; every step carries its code or its exact content.
- **Type consistency:** `Probe`, `Cpu`, `Memory`, `Gpu`, `Host`, `LlamaCpp`, `RunningServer`, `LocalModel`, `SystemReport` are defined once in Task 3 and used with the same field names throughout; `Runner.run(argv, *, timeout)` and `probe(name, runner, argv, parse)` from Task 4 are used unchanged in Tasks 5, 6, 9; `scan()` keyword arguments in Task 8 match their use in Task 11; `detect_llamacpp()` in Task 10 matches Task 11; `scan_system()` keyword arguments match the tests in Tasks 11 and 12.
