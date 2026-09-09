"""CPU facts: model, cores, instruction sets, performance-core count."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import psutil

from llamafit.hardware.runner import Runner, probe
from llamafit.i18n import pgettext
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


def _default_cores() -> tuple[int, int]:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    return physical, logical


def detect_cpu(
    runner: Runner,
    os_name: OsName,
    *,
    cpuinfo_provider: Callable[[], Mapping[str, Any]] | None = None,
    cores_provider: Callable[[], tuple[int, int]] | None = None,
) -> tuple[Cpu, list[Probe]]:
    """Detect the CPU. Never raises; failures are reported in the probes.

    ``cpuinfo_provider`` and ``cores_provider`` let tests replace py-cpuinfo and psutil's
    core counts; both default to the real thing.
    """
    provider = cpuinfo_provider or _default_cpuinfo
    probes: list[Probe] = []
    start = time.perf_counter()
    # A context, because "unknown" is also the answer three other rows give about
    # something else, and Portuguese inflects the word for the noun it is about.
    model = pgettext("CPU model", "unknown")
    flags: list[str] = []
    try:
        info = provider()
        model = str(
            info.get("brand_raw") or info.get("brand") or pgettext("CPU model", "unknown")
        ).strip()
        flags = [str(f) for f in info.get("flags", [])]
        probes.append(Probe(name="cpuinfo", ok=True, duration_ms=_ms(start)))
    except Exception as exc:  # a broken cpuinfo must not stop the scan
        probes.append(Probe(name="cpuinfo", ok=False, duration_ms=_ms(start), error=str(exc)))

    start = time.perf_counter()
    try:
        physical, logical = (cores_provider or _default_cores)()
        probes.append(Probe(name="cpu-cores", ok=True, duration_ms=_ms(start)))
    except Exception as exc:  # a failing core count must not stop the scan
        physical, logical = 1, 1
        probes.append(Probe(name="cpu-cores", ok=False, duration_ms=_ms(start), error=str(exc)))

    perf = performance_cores_for(model, physical)
    if os_name == "macos":
        value, rec = probe(
            "sysctl-perflevel",
            runner,
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            lambda out: int(out.strip()),
        )
        probes.append(rec)
        if value:
            perf = value
    return Cpu(
        model=model,
        physical_cores=physical,
        logical_cores=logical,
        performance_cores=perf,
        isa=isa_from_flags(flags),
    ), probes


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
