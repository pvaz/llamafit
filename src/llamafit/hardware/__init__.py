# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Host detection: ``scan()`` composes every probe into a ``Host``."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llamafit.hardware.bandwidth import BandwidthCache, machine_key, resolve_memory_bandwidth
from llamafit.hardware.cpu import detect_cpu
from llamafit.hardware.disks import detect_disks
from llamafit.hardware.gpu import detect_gpus
from llamafit.hardware.gputable import enrich_gpu, lookup_gpu
from llamafit.hardware.memory import detect_memory
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.logging import get_logger
from llamafit.models.host import Arch, Cpu, Host, Memory, OsName, Probe
from llamafit.paths import get_paths

_log = get_logger("hardware")


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


def _bandwidth_cache(memory: Memory, cpu: Cpu) -> BandwidthCache | None:
    """Where this machine's kept bandwidth figure lives, or ``None`` when there is nowhere.

    The cache directory is resolved through :func:`llamafit.paths.get_paths`, which needs a
    home directory to resolve and does not always have one. A machine without one measures
    the bandwidth every time, which is what it did before any of this existed -- the same
    answer the disk probe below already gives to the same question.
    """
    try:
        return BandwidthCache(machine_key(memory, cpu))
    except Exception as exc:  # no home directory: there is nowhere to keep it, so nowhere
        _log.debug("no bandwidth cache: %s", exc)
        return None


def scan(
    runner: Runner | None = None,
    *,
    os_name: OsName | None = None,
    measure_bandwidth: bool = True,
    refresh_bandwidth: bool = False,
    extra_paths: Iterable[Path] = (),
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
    cores_provider: Callable[[], tuple[int, int]] | None = None,
    vm_provider: Callable[[], tuple[int, int]] | None = None,
) -> Host:
    """Detect everything about this machine that the estimator needs.

    Every probe is optional: what fails is recorded in ``Host.probes`` and the rest
    of the scan continues. Pass a ``FakeRunner`` and providers to scan a recorded
    machine instead of the real one. ``cpuinfo_provider`` and ``cores_provider`` let
    tests replace py-cpuinfo and psutil's core counts; ``vm_provider`` replaces
    psutil's memory totals.

    The memory bandwidth is looked for in the cache before it is timed, and
    ``refresh_bandwidth`` times it again and replaces what is there. A scan that is not
    measuring bandwidth at all never touches the cache, which is what keeps every test
    that passes ``measure_bandwidth=False`` independent of whatever this machine has
    stored.
    """
    runner = runner or SubprocessRunner()
    os_name = os_name or current_os()
    cpu, cpu_probes = detect_cpu(
        runner, os_name, cpuinfo_provider=cpuinfo_provider, cores_provider=cores_provider
    )
    memory, memory_probes = detect_memory(runner, os_name, vm_provider=vm_provider)
    memory = resolve_memory_bandwidth(
        memory,
        measure=measure_bandwidth,
        cache=_bandwidth_cache(memory, cpu) if measure_bandwidth else None,
        refresh=refresh_bandwidth,
    )
    gpus, gpu_probes = detect_gpus(runner, os_name)
    gpus = [enrich_gpu(gpu) for gpu in gpus]
    unified = any(spec.unified for spec in (lookup_gpu(g.name) for g in gpus) if spec) or (
        os_name == "macos" and current_arch() == "arm64"
    )
    paths_probes: list[Probe] = []
    disk_paths = [Path.cwd()]
    start = time.perf_counter()
    try:
        disk_paths.append(get_paths().downloads_dir)
    except Exception as exc:  # no home directory: the downloads path is simply unknown
        paths_probes.append(
            Probe(
                name="paths",
                ok=False,
                duration_ms=int((time.perf_counter() - start) * 1000),
                error=str(exc),
            )
        )
    disks = detect_disks([*disk_paths, *extra_paths])
    probes = [*cpu_probes, *memory_probes, *gpu_probes, *paths_probes]
    failed = sum(1 for p in probes if not p.ok)
    _log.debug(
        "scanned %s: cpu %s, memory %d bytes, %d gpus, %d probes failed",
        os_name,
        cpu.model,
        memory.total_bytes,
        len(gpus),
        failed,
    )
    return Host(
        os=os_name,
        os_version=platform.platform(),
        arch=current_arch(),
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        unified_memory=unified,
        disks=disks,
        probes=probes,
        scanned_at=datetime.now(timezone.utc),
    )


__all__ = ["current_arch", "current_os", "scan"]
