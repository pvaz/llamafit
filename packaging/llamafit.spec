# -*- mode: python ; coding: utf-8 -*-
"""How the standalone executable is built, for people without a Python.

`pip install llamafit` is the normal way in and stays the recommended one. This file is
for the other reader: someone who wants to know whether a model runs on their machine and
should not have to acquire a language runtime to find out. PyInstaller bundles the
interpreter, the dependencies and the package's own data into one file that runs on a
machine with no Python at all.

Run it from the repository root, against an environment where the project is installed:

    pip install -e ".[web]" pyinstaller
    pyinstaller packaging/llamafit.spec

`.github/workflows/binaries.yml` does exactly that on four machines and attaches the
results to the release. Building by hand gives you the same binary for your own machine
and nothing else -- PyInstaller is not a cross-compiler, so each target is built on its
own operating system and architecture.

Three collections do the work, and each is here because leaving it out produces a binary
that starts and then fails at the moment somebody uses it:

- `collect_data_files("llamafit")` carries the catalog, the generated facts, the hardware
  profiles, the JSON schemas, the GPU table and the 38 message catalogs. These are read
  through `importlib.resources` at run time, never imported, so nothing in the bytecode
  mentions them and the analysis cannot infer them.
- `collect_submodules("llamafit")` is what makes those reads resolve. A frozen package is
  a set of modules, and `resources.files("llamafit.data.catalog")` needs
  `llamafit.data.catalog` to be one of them -- the directory alone is not enough.
- `collect_all` for `textual` and `uvicorn`, which both load classes by name at run time:
  Textual its widgets and CSS, uvicorn its HTTP and WebSocket protocol implementations.

The `fast` extra is deliberately absent. NumPy would add about half again to a file people
download, and roughly double the time each run spends unpacking itself, in exchange for
sharpening one measurement -- and without it LlamaFit still measures memory bandwidth and
still says which method it used. The `web` extra is present for the opposite reason:
without it `llamafit serve` would fail with advice to run `pip install`, which is the one
thing a person holding a standalone binary cannot do.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# SPECPATH and workpath are injected into this file's namespace by PyInstaller.
ROOT = Path(SPECPATH).parent  # noqa: F821
BUILD = Path(workpath)  # noqa: F821

VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "version"
]

datas = collect_data_files("llamafit")
hiddenimports = collect_submodules("llamafit")
binaries = []

for package in ("textual", "uvicorn"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


def _windows_version_resource() -> str | None:
    """The file-properties block Windows shows for an executable, or nothing elsewhere.

    Without it, Explorer's Properties dialog shows an unnamed file of unknown origin and
    no version, which is what a person is looking at when they are deciding whether to
    trust a download. It costs a few lines to say who wrote it and which version it is.

    Windows wants four integers, so `0.1.0` becomes `(0, 1, 0, 0)`; a pre-release suffix
    is dropped here and kept in the strings, where it is allowed to be a string.
    """
    if sys.platform != "win32":
        return None

    numbers = []
    for part in VERSION.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < 4:
        numbers.append(0)
    quad = tuple(numbers)

    resource = f"""\
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={quad},
    prodvers={quad},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'Paulo Vaz'),
        StringStruct('FileDescription',
                     'Find, size, install and verify open-weight LLMs for llama.cpp'),
        StringStruct('FileVersion', '{VERSION}'),
        StringStruct('InternalName', 'llamafit'),
        StringStruct('LegalCopyright',
                     'Paulo Vaz. GNU Affero General Public License v3 or later.'),
        StringStruct('OriginalFilename', 'llamafit.exe'),
        StringStruct('ProductName', 'LlamaFit'),
        StringStruct('ProductVersion', '{VERSION}'),
      ]),
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
"""
    BUILD.mkdir(parents=True, exist_ok=True)
    path = BUILD / "version-resource.txt"
    path.write_text(resource, encoding="utf-8")
    return str(path)


a = Analysis(  # noqa: F821
    [str(ROOT / "src" / "llamafit" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Nothing in this list is imported by the program. They arrive as dependencies of
    # dependencies, and each one that stays adds megabytes to a file someone downloads.
    excludes=[
        "tkinter",
        "unittest",
        "pydoc_data",
        "test",
        "lib2to3",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="llamafit",
    debug=False,
    bootloader_ignore_signals=False,
    # Stripping symbols saves a little and breaks binaries on some macOS toolchains; the
    # saving is not worth a file that will not run.
    strip=False,
    # UPX compression is off on purpose. It saves perhaps a third of the size and costs
    # far more than that: compressed executables are a long-standing heuristic in
    # antivirus products, and a first release whose Windows download is quarantined has
    # not shipped anything. A larger honest file gets installed.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=_windows_version_resource(),
)
