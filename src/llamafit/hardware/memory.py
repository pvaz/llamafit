"""System memory: totals from psutil, module facts from the OS, bandwidth estimate."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import psutil

from llamafit.hardware.runner import Runner, probe
from llamafit.models.host import Memory, OsName, Probe

_SMBIOS_TYPES = {
    20: "DDR",
    21: "DDR2",
    24: "DDR3",
    26: "DDR4",
    30: "LPDDR4",
    34: "DDR5",
    35: "LPDDR5",
}

_WINDOWS_MODULES_CMD = [
    "powershell",
    "-NoProfile",
    "-Command",
    "Get-CimInstance Win32_PhysicalMemory | Select-Object SMBIOSMemoryType,Speed,"
    "ConfiguredClockSpeed,Capacity | ConvertTo-Json",
]
_MACOS_MEMORY_CMD = ["system_profiler", "SPMemoryDataType", "-json"]
_LINUX_MEMORY_CMD = ["dmidecode", "-t", "memory"]


def theoretical_bandwidth_gbps(speed_mts: int, channels: int) -> float:
    """Peak DDR bandwidth: transfers per second x 8 bytes x channels."""
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
