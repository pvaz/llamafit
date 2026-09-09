"""GPU detection through vendor tools, with name-only fallbacks."""

from __future__ import annotations

import json
import re
from typing import Any

from llamafit.hardware.runner import Runner, probe
from llamafit.i18n import _
from llamafit.models.host import Backend, Gpu, OsName, Probe, Vendor

_NVIDIA_CMD = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,driver_version",
    "--format=csv,noheader,nounits",
]
_ROCM_CMD = ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"]
_APPLE_CMD = ["system_profiler", "SPDisplaysDataType", "-json"]
_WMI_CMD = [
    "powershell",
    "-NoProfile",
    "-Command",
    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json",
]
_LSPCI_CMD = ["lspci", "-nn"]
_PCI_CLASS_CODE_RE = re.compile(r"^[0-9a-f]{4}$", re.IGNORECASE)
_PCI_VENDOR_DEVICE_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{4}$", re.IGNORECASE)
_NORMALISE_TOKENS = [
    "nvidia",
    "amd",
    "intel",
    "corporation",
    "advanced micro devices",
    "(r)",
    "(tm)",
]


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


def _mib_to_bytes(text: str) -> int | None:
    """Convert an ``nvidia-smi`` MiB figure to bytes, or ``None`` for a non-numeric value.

    ``nvidia-smi`` reports unsupported fields (e.g. on some Tesla cards) as ``[N/A]``.
    """
    try:
        return int(float(text)) * 1024**2
    except ValueError:
        return None


def parse_nvidia_smi(out: str) -> list[Gpu]:
    """Parse the CSV produced by ``_NVIDIA_CMD`` (memory figures are MiB, sometimes ``[N/A]``)."""
    gpus: list[Gpu] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[1]:
            continue
        index, name, total, used, driver = parts[:5]
        gpus.append(
            Gpu(
                index=int(index),
                vendor="nvidia",
                name=name,
                vram_total_bytes=_mib_to_bytes(total),
                vram_used_bytes=_mib_to_bytes(used),
                backend_hint="cuda",
                driver=driver,
            )
        )
    if not gpus:
        raise ValueError(_("nvidia-smi returned no GPUs"))
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
        gpus.append(
            Gpu(
                index=int(key[4:] or 0),
                vendor="amd",
                name=name,
                vram_total_bytes=int(total) if total else None,
                vram_used_bytes=int(used) if used else None,
                backend_hint="hip",
            )
        )
    if not gpus:
        raise ValueError(_("rocm-smi returned no cards"))
    return gpus


def parse_system_profiler(out: str) -> list[Gpu]:
    """Parse ``system_profiler SPDisplaysDataType -json``; Apple GPUs share system memory."""
    data = json.loads(out)
    items = data.get("SPDisplaysDataType", [])
    gpus = [
        Gpu(
            index=i,
            vendor=vendor_from_name(str(item.get("sppci_model", "Apple GPU"))),
            name=str(item.get("sppci_model", "Apple GPU")),
            backend_hint="metal",
        )
        for i, item in enumerate(items)
    ]
    if not gpus:
        raise ValueError(_("system_profiler returned no displays"))
    return gpus


def parse_wmi_video(out: str) -> list[Gpu]:
    """Parse WMI video controllers: names only, AdapterRAM is capped at 4 GB and ignored."""
    data: Any = json.loads(out)
    items = data if isinstance(data, list) else [data]
    gpus: list[Gpu] = []
    for i, item in enumerate(items):
        if not item.get("Name"):
            continue
        name = str(item.get("Name", ""))
        gpus.append(Gpu(index=i, vendor=vendor_from_name(name), name=name))
    if not gpus:
        raise ValueError(_("WMI returned no video controllers"))
    return gpus


def _lspci_name(line: str) -> str:
    """Pick the marketing name from an ``lspci -nn`` line, skipping PCI class/id brackets."""
    brackets: list[str] = re.findall(r"\[([^\]]+)\]", line)
    names = [
        b
        for b in brackets
        if not _PCI_CLASS_CODE_RE.match(b) and not _PCI_VENDOR_DEVICE_RE.match(b)
    ]
    if names:
        return names[0]
    return line.rsplit(": ", 1)[-1].split("[", 1)[0].strip()


def parse_lspci(out: str) -> list[Gpu]:
    """Parse ``lspci -nn`` for VGA and 3D controllers: names only."""
    gpus: list[Gpu] = []
    for i, line in enumerate(ln for ln in out.splitlines() if re.search(r"VGA|3D controller", ln)):
        gpus.append(Gpu(index=i, vendor=vendor_from_name(line), name=_lspci_name(line)))
    if not gpus:
        raise ValueError(_("lspci found no display controllers"))
    return gpus


def _normalise(name: str) -> str:
    """Reduce a GPU name to a vendor-agnostic form so it can be matched across sources."""
    lowered = name.lower()
    for token in _NORMALISE_TOKENS:
        lowered = lowered.replace(token, " ")
    return " ".join(lowered.split())


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

    for name, cmd, parser in (
        ("nvidia-smi", _NVIDIA_CMD, parse_nvidia_smi),
        ("rocm-smi", _ROCM_CMD, parse_rocm_smi),
    ):
        gpus, rec = probe(name, runner, cmd, parser)
        probes.append(rec)
        if gpus:
            found.extend(gpus)

    if os_name == "windows":
        generic, rec = probe("wmi-video", runner, _WMI_CMD, parse_wmi_video)
    else:
        generic, rec = probe("lspci", runner, _LSPCI_CMD, parse_lspci)
    probes.append(rec)

    known = {_normalise(g.name) for g in found}
    for gpu in generic or []:
        if _normalise(gpu.name) in known:
            continue
        gpu.backend_hint = _backend_for(gpu.vendor, os_name)
        found.append(gpu)

    for i, gpu in enumerate(found):
        gpu.index = i
    return found, probes
