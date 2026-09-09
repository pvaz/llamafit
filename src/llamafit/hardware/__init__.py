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
from llamafit.logging import get_logger
from llamafit.models.host import Arch, Host, OsName
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


def scan(
    runner: Runner | None = None,
    *,
    os_name: OsName | None = None,
    measure_bandwidth: bool = True,
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
    """
    runner = runner or SubprocessRunner()
    os_name = os_name or current_os()
    cpu, cpu_probes = detect_cpu(
        runner, os_name, cpuinfo_provider=cpuinfo_provider, cores_provider=cores_provider
    )
    memory, memory_probes = detect_memory(runner, os_name, vm_provider=vm_provider)
    memory = resolve_memory_bandwidth(memory, measure=measure_bandwidth)
    gpus, gpu_probes = detect_gpus(runner, os_name)
    gpus = [enrich_gpu(gpu) for gpu in gpus]
    unified = any(spec.unified for spec in (lookup_gpu(g.name) for g in gpus) if spec) or (
        os_name == "macos" and current_arch() == "arm64"
    )
    paths = get_paths()
    disks = detect_disks([Path.cwd(), paths.downloads_dir, *extra_paths])
    probes = [*cpu_probes, *memory_probes, *gpu_probes]
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
