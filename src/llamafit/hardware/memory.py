"""System memory: totals from psutil, module facts from the OS, bandwidth estimate."""

from __future__ import annotations

import json
import re
import time
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
    "ConfiguredClockSpeed,Capacity,BankLabel,DeviceLocator | ConvertTo-Json",
]
_MACOS_MEMORY_CMD = ["system_profiler", "SPMemoryDataType", "-json"]
_LINUX_MEMORY_CMD = ["dmidecode", "-t", "memory"]

# A channel letter next to the word "channel" (ChannelA-DIMM1, Channel A Slot 0, dmidecode's
# bare CHANNEL A), or next to "DIMM" as DIMM_A1 / DIMM A1, or leading the whole label as in
# A1_DIMM0. Deliberately narrow: an unrecognised label (BANK 0 alone, a bare DIMM index, or
# something this table has not seen yet) must come back None rather than a guess.
_CHANNEL_LETTER_PATTERNS = (
    re.compile(r"channel\s*([a-z])\b", re.IGNORECASE),
    re.compile(r"dimm[_ ]([a-z])\d", re.IGNORECASE),
    re.compile(r"^([a-z])\d+[_ ]dimm", re.IGNORECASE),
)
# Some boards report DeviceLocator as "ControllerN-DIMMx" instead of a channel letter: one
# integrated memory controller per channel. Confirmed against this project's own reference
# machine, whose four modules on a documented dual-channel board read Controller0-DIMM0,
# Controller0-DIMM1, Controller1-DIMM0 and Controller1-DIMM1 - two controllers, two channels.
_CONTROLLER_PATTERN = re.compile(r"controller(\d+)", re.IGNORECASE)


def channel_from_labels(bank_label: str | None, device_locator: str | None) -> str | None:
    """Parse a channel identifier out of one populated module's slot labels.

    ``bank_label`` and ``device_locator`` are the two label strings the OS reports for
    a memory module: ``BankLabel``/``DeviceLocator`` from ``Win32_PhysicalMemory`` on
    Windows, or ``Bank Locator``/``Locator`` from ``dmidecode -t memory`` on Linux.
    Recognises, case-insensitively, ``ChannelA-DIMM1``, ``Channel A Slot 0``,
    ``DIMM_A1``, ``DIMM A1``, ``A1_DIMM0`` and the bare ``CHANNEL A`` dmidecode also
    prints, returning the channel letter they encode (lower-cased); and
    ``Controller0-DIMM0`` style locators, returning ``"controllerN"`` for the
    controller number, since some boards report one memory controller per channel.

    Returns ``None`` for a label that does not encode a channel, such as ``BANK 0``
    (which numbers slots or ranks, not channels) or a bare ``DIMM 0``: an unrecognised
    label must never be silently treated as its own channel.
    """
    for label in (bank_label, device_locator):
        if not label:
            continue
        for pattern in _CHANNEL_LETTER_PATTERNS:
            match = pattern.search(label)
            if match:
                return match.group(1).lower()
        controller = _CONTROLLER_PATTERN.search(label)
        if controller:
            return f"controller{controller.group(1)}"
    return None


def _count_channels(pairs: list[tuple[str | None, str | None]]) -> int | None:
    """Distinct channel identifiers across a populated module's label pairs, or ``None``."""
    ids = {cid for bank, locator in pairs if (cid := channel_from_labels(bank, locator))}
    return len(ids) or None


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
    """Detect memory totals and, when the OS tells us, type, speed, modules and channels.

    The theoretical bandwidth estimate needs the channel count, which is not the module
    count. Windows and Linux report it indirectly: their slot labels (``BankLabel`` and
    ``DeviceLocator`` on Windows, ``Bank Locator`` and ``Locator`` from ``dmidecode`` on
    Linux) usually encode a channel letter or a per-controller locator, which
    ``channel_from_labels`` parses. The estimate is computed only when that parse
    genuinely found one; macOS's ``system_profiler`` reports no such labels, so channels
    stays unknown there, and the bandwidth then comes from the measurement or the assumed
    default.
    """
    probes: list[Probe] = []
    start = time.perf_counter()
    try:
        total, available = (vm_provider or _default_vm)()
        probes.append(Probe(name="memory-totals", ok=True, duration_ms=_ms(start)))
    except Exception as exc:  # a failing totals source must not stop the scan
        total, available = 0, 0
        probes.append(Probe(name="memory-totals", ok=False, duration_ms=_ms(start), error=str(exc)))
    memory = Memory(total_bytes=total, available_bytes=available)

    if os_name == "windows":
        facts, rec = probe("memory-modules", runner, _WINDOWS_MODULES_CMD, _parse_windows_modules)
    elif os_name == "macos":
        facts, rec = probe("memory-modules", runner, _MACOS_MEMORY_CMD, _parse_macos_memory)
    else:
        facts, rec = probe("memory-modules", runner, _LINUX_MEMORY_CMD, _parse_dmidecode)
    probes.append(rec)
    if facts:
        memory.type, memory.speed_mts, memory.modules, memory.channels = facts
        if memory.speed_mts and memory.channels:
            memory.bandwidth_gbps = theoretical_bandwidth_gbps(memory.speed_mts, memory.channels)
            memory.bandwidth_source = "estimated"
    return memory, probes


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _parse_windows_modules(out: str) -> tuple[str | None, int | None, int | None, int | None]:
    """Return the DDR type, configured speed, populated module count and channel count."""
    data: Any = json.loads(out)
    modules = data if isinstance(data, list) else [data]
    populated = [m for m in modules if m.get("Capacity")]
    if not populated:
        raise ValueError("no populated memory modules")
    first = populated[0]
    mem_type = _SMBIOS_TYPES.get(int(first.get("SMBIOSMemoryType") or 0))
    speed = first.get("ConfiguredClockSpeed") or first.get("Speed")
    channels = _count_channels([(m.get("BankLabel"), m.get("DeviceLocator")) for m in populated])
    return mem_type, int(speed) if speed else None, len(populated), channels


def _parse_macos_memory(out: str) -> tuple[str | None, int | None, int | None, int | None]:
    """Return the memory type and module count; speed and channel count are not reported."""
    data = json.loads(out)
    items = data.get("SPMemoryDataType", [])
    if not items:
        raise ValueError("no SPMemoryDataType")
    mem_type = items[0].get("dimm_type")
    return (str(mem_type) if mem_type else None), None, len(items), None


def _parse_dmidecode(out: str) -> tuple[str | None, int | None, int | None, int | None]:
    """Return the DDR type, configured speed, populated module count and channel count."""
    mem_type: str | None = None
    speed: int | None = None
    populated = 0
    pairs: list[tuple[str | None, str | None]] = []
    in_device = False
    is_populated = False
    locator: str | None = None
    bank_locator: str | None = None

    def flush() -> None:
        nonlocal populated
        if is_populated:
            populated += 1
            pairs.append((bank_locator, locator))

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Memory Device"):
            flush()
            in_device = True
            is_populated = False
            locator = None
            bank_locator = None
            continue
        if not in_device:
            continue
        if stripped.startswith("Size:"):
            is_populated = "No Module" not in stripped
        elif stripped.startswith("Bank Locator:"):
            bank_locator = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Locator:"):
            locator = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Type:") and mem_type is None and "Unknown" not in stripped:
            mem_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Configured Memory Speed:") and speed is None:
            digits = "".join(ch for ch in stripped.split(":", 1)[1] if ch.isdigit())
            speed = int(digits) if digits else None
    flush()
    if populated == 0:
        raise ValueError("no populated memory modules")
    return mem_type, speed, populated, _count_channels(pairs)
