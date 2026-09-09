"""``scan_system``: host scan plus llama.cpp detection in one report."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
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
    cores_provider: Callable[[], tuple[int, int]] | None = None,
    well_known: Iterable[Path] | None = None,
) -> SystemReport:
    """Scan the host and detect llama.cpp; all arguments exist so tests can inject fakes.

    ``cores_provider`` replaces psutil's CPU core counts; it is passed straight through
    to ``hardware.scan``. ``well_known`` replaces the platform's well-known llama.cpp
    install directories; it is passed straight through to ``detect_llamacpp``, so tests
    can pass ``[]`` to keep results independent of what happens to be installed on the
    machine running them.
    """
    host = scan(
        runner,
        os_name=os_name,
        measure_bandwidth=measure_bandwidth,
        cpuinfo_provider=cpuinfo_provider,
        vm_provider=vm_provider,
        cores_provider=cores_provider,
    )
    llamacpp = detect_llamacpp(runner, os_name=host.os, env=env, http=http, well_known=well_known)
    return SystemReport(host=host, llamacpp=llamacpp, version=__version__)
