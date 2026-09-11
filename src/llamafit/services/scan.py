# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``scan_system``: host scan plus llama.cpp detection in one report."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from llamafit import __version__
from llamafit.hardware import current_os, scan
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
    refresh_bandwidth: bool = False,
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
    vm_provider: Callable[[], tuple[int, int]] | None = None,
    cores_provider: Callable[[], tuple[int, int]] | None = None,
    well_known: Iterable[Path] | None = None,
) -> SystemReport:
    """Scan the host and detect llama.cpp; all arguments exist so tests can inject fakes.

    llama.cpp is detected first so its install directory, and ``LLAMA_CACHE`` when set, can
    be added to the paths whose free space is reported: models and binaries usually live on
    a different volume from the working directory, and that is the volume a low-disk warning
    has to be about.

    ``cores_provider`` replaces psutil's CPU core counts; it is passed straight through
    to ``hardware.scan``. ``well_known`` replaces the platform's well-known llama.cpp
    install directories; it is passed straight through to ``detect_llamacpp``, so tests
    can pass ``[]`` to keep results independent of what happens to be installed on the
    machine running them.
    """
    os_name = os_name or current_os()
    llamacpp = detect_llamacpp(runner, os_name=os_name, env=env, http=http, well_known=well_known)
    environment = os.environ if env is None else env
    extra_paths = [
        Path(p)
        for p in (llamacpp.path, environment.get("LLAMA_CACHE"))
        if p  # an absent install or an unset LLAMA_CACHE adds no path
    ]
    host = scan(
        runner,
        os_name=os_name,
        measure_bandwidth=measure_bandwidth,
        refresh_bandwidth=refresh_bandwidth,
        extra_paths=extra_paths,
        cpuinfo_provider=cpuinfo_provider,
        vm_provider=vm_provider,
        cores_provider=cores_provider,
    )
    return SystemReport(host=host, llamacpp=llamacpp, version=__version__)
