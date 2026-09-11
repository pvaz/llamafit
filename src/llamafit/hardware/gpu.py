# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""GPU detection through vendor tools, the Vulkan driver, and name-only fallbacks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
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
_VULKANINFO_CMD = ["vulkaninfo"]
"""The full text dump, not ``--summary``, and the difference is the whole point.

Section 4.2 of the design names ``vulkaninfo --summary`` as the source of "device names
and heap sizes". It is not: ``--summary`` prints the device name, the vendor and device
ids and the driver, and no memory at all. The heaps are in the default text output, under
``VkPhysicalDeviceMemoryProperties``, and that output is what this probe reads.
"""
_PCI_CLASS_CODE_RE = re.compile(r"^[0-9a-f]{4}$", re.IGNORECASE)
_PCI_VENDOR_DEVICE_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{4}$", re.IGNORECASE)
_VK_GPU_RE = re.compile(r"^GPU(\d+):\s*$")
_VK_FIELD_RE = re.compile(r"^\s+(\w+)\s*=\s*(.+?)\s*$")
_VK_HEAP_RE = re.compile(r"^\s+memoryHeaps\[\d+\]:\s*$")
_VK_HEAP_FIELD_RE = re.compile(r"^\s+(size|budget|usage)\s*=\s*(\d+)")
_VK_DEVICE_LOCAL = "MEMORY_HEAP_DEVICE_LOCAL_BIT"
_VK_DISCRETE = "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
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
                vram_source="measured",
                backend_hint="cuda",
                driver=driver,
            )
        )
    if not gpus:
        raise ValueError(_("nvidia-smi returned no GPUs"))
    return gpus


def _rocm_field(card: Mapping[str, Any], *keys: str) -> Any:
    """The first of ``keys`` the card reports, matched without regard to case.

    ROCm renamed its own keys between releases: ``--showproductname`` printed
    ``Card series`` for the whole of ROCm 5 and ``Card Series`` from 6 onwards. A lookup
    that spells one of them is a lookup that returns ``None`` on half the AMD machines
    there are, and an AMD card whose name came back ``None`` is an AMD card LlamaFit
    prints as "AMD GPU" -- which is how this went unnoticed, since the test fixture
    spelled it the way the code did rather than the way ROCm 5 did.
    """
    folded = {key.casefold(): value for key, value in card.items()}
    for key in keys:
        value = folded.get(key.casefold())
        if value is not None:
            return value
    return None


def parse_rocm_smi(out: str) -> list[Gpu]:
    """Parse ``rocm-smi --json`` output keyed by ``cardN``."""
    data: dict[str, Any] = json.loads(out)
    gpus: list[Gpu] = []
    for key in sorted(k for k in data if k.startswith("card")):
        card = data[key]
        name = str(_rocm_field(card, "Card Series", "Card Model") or "AMD GPU")
        total = _rocm_field(card, "VRAM Total Memory (B)")
        used = _rocm_field(card, "VRAM Total Used Memory (B)")
        gpus.append(
            Gpu(
                index=int(key[4:] or 0),
                vendor="amd",
                name=name,
                vram_total_bytes=int(total) if total else None,
                vram_used_bytes=int(used) if used else None,
                vram_source="measured" if total and used else "unknown",
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


def _vulkan_device_blocks(out: str) -> list[list[str]]:
    """Split ``vulkaninfo``'s text output into one block of lines per ``GPUn:`` heading.

    The heading is the only structure in that file that is guaranteed and cheap to find.
    Everything about one device -- its properties, its extensions, its memory heaps --
    is printed under it at column zero until the next heading, so a block is one device
    and the blocks are in the order the Vulkan loader enumerated them.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in out.splitlines():
        if _VK_GPU_RE.match(line):
            current = []
            blocks.append(current)
            continue
        if current is not None:
            current.append(line)
    return blocks


def _vulkan_device_local_heap(lines: Sequence[str]) -> tuple[int, int | None, int] | None:
    """The first device-local heap's size, budget and usage, in bytes.

    Returns:
        ``(size, budget, usage)`` for the first heap carrying
        ``MEMORY_HEAP_DEVICE_LOCAL_BIT``, with ``budget`` ``None`` when the driver does
        not support ``VK_EXT_memory_budget`` and ``vulkaninfo`` therefore printed none;
        or ``None`` when the device reports no device-local heap at all.

    The *first* such heap, because that is the one llama.cpp's Vulkan backend takes for
    the device's memory, and a figure that disagrees with the backend's own would be a
    budget computed for a card llama.cpp is not going to allocate on. Cards with ReBAR
    report a second, small, host-visible device-local heap after it; that one is a window
    onto the first, not more memory.
    """
    heap: list[str] | None = None
    for line in lines:
        if not line.strip():
            continue
        if line[:1] not in (" ", "\t"):
            # Column zero ends whatever was nested: the heap list, and the whole
            # VkPhysicalDeviceMemoryProperties section after it.
            if heap is not None and _VK_DEVICE_LOCAL in "\n".join(heap):
                break
            heap = None
            continue
        if _VK_HEAP_RE.match(line):
            if heap is not None and _VK_DEVICE_LOCAL in "\n".join(heap):
                break
            heap = []
            continue
        if heap is not None:
            heap.append(line)
    if heap is None or _VK_DEVICE_LOCAL not in "\n".join(heap):
        return None
    figures: dict[str, int] = {}
    for line in heap:
        field = _VK_HEAP_FIELD_RE.match(line)
        if field is not None:
            figures.setdefault(field.group(1), int(field.group(2)))
    if "size" not in figures:
        return None
    return figures["size"], figures.get("budget"), figures.get("usage", 0)


def parse_vulkaninfo(out: str) -> list[Gpu]:
    """Parse ``vulkaninfo``'s text output for discrete devices with a readable heap.

    Discrete devices only, and that restriction is the reason this probe can be trusted
    at all. An integrated GPU's device-local heap *is* system memory -- the same bytes
    :class:`~llamafit.models.host.Memory` already counts -- so reporting it as VRAM would
    hand the planner a machine with twice the memory it has and let it place weights in
    both pools at once. A wrong number that looks plausible is worse than the ``None``
    this probe was written to replace, so an integrated device is skipped and the card
    stays unsized.

    The free figure is the driver's own answer to the question the planner asks, which is
    not quite the question a vendor tool answers: ``heapBudget`` is what this process may
    allocate given everything else already on the card, and ``heapUsage`` is what this
    process has taken of it. On the reference machine that comes out about 450 MiB below
    ``nvidia-smi``'s free figure -- the driver keeps a reserve it will not promise away --
    which is the safe direction for a budget to be wrong in, and is why the figure is
    labelled ``estimated`` rather than ``measured``.

    Raises:
        ValueError: No discrete device reported a device-local heap, so there is nothing
            here the planner could use. Raised rather than returned empty so ``doctor``
            records a probe that ran and did not answer, instead of one that answered
            with nothing.
    """
    gpus: list[Gpu] = []
    for index, block in enumerate(_vulkan_device_blocks(out)):
        # First wins: ``deviceName`` is printed in VkPhysicalDeviceProperties, at the top
        # of the block, and a later structure that happens to carry the same key name is
        # describing something else.
        fields: dict[str, str] = {}
        for line in block:
            match = _VK_FIELD_RE.match(line)
            if match is not None:
                fields.setdefault(match.group(1), match.group(2))
        name = fields.get("deviceName")
        if not name or fields.get("deviceType") != _VK_DISCRETE:
            continue
        heap = _vulkan_device_local_heap(block)
        if heap is None:
            continue
        size, budget, usage = heap
        used = None if budget is None else size - max(min(budget, size) - usage, 0)
        gpus.append(
            Gpu(
                index=index,
                vendor=vendor_from_name(name),
                name=name,
                vram_total_bytes=size,
                vram_used_bytes=used,
                vram_source="estimated",
                backend_hint="vulkan",
            )
        )
    if not gpus:
        raise ValueError(_("vulkaninfo reported no discrete device with a memory heap"))
    return gpus


def _normalise(name: str) -> str:
    """Reduce a GPU name to a vendor-agnostic form so it can be matched across sources."""
    lowered = name.lower()
    for token in _NORMALISE_TOKENS:
        lowered = lowered.replace(token, " ")
    return " ".join(lowered.split())


def _take_vulkan_vram(gpu: Gpu, device: Gpu) -> None:
    """Copy one Vulkan device's memory figures onto the card the vendor tools could not size."""
    gpu.vram_total_bytes = device.vram_total_bytes
    gpu.vram_used_bytes = device.vram_used_bytes
    gpu.vram_source = device.vram_source


def fill_vram_from_vulkan(found: Sequence[Gpu], devices: Sequence[Gpu]) -> None:
    """Give every unsized card the figures of the Vulkan device that is the same card.

    Matching is by normalised name first, and a name is only allowed to match when
    exactly one Vulkan device carries it, so two identical cards in one machine are
    never sized from each other's heap.

    Then one fallback, for the case the name match cannot reach: ``lspci`` calls a
    7900 XTX ``Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]`` while Vulkan calls
    it ``AMD Radeon RX 7900 XTX (RADV NAVI31)``, and no normalisation is going to make
    those the same string. When exactly one card is still unsized and exactly one
    discrete Vulkan device is still unclaimed, they are the same device -- there is
    nothing else either of them could be. With two of either, the pairing is a guess, and
    a guess is what this probe exists to remove.
    """
    unsized = [gpu for gpu in found if gpu.vram_total_bytes is None]
    pool = list(devices)
    for gpu in unsized:
        same = [d for d in pool if _normalise(d.name) == _normalise(gpu.name)]
        if len(same) == 1:
            _take_vulkan_vram(gpu, same[0])
            pool.remove(same[0])
    remaining = [gpu for gpu in unsized if gpu.vram_total_bytes is None]
    if len(remaining) == 1 and len(pool) == 1:
        _take_vulkan_vram(remaining[0], pool[0])


def detect_gpus(runner: Runner, os_name: OsName) -> tuple[list[Gpu], list[Probe]]:
    """Run the probes that make sense for the OS and merge their results.

    Vendor tools win over generic listings: a device already reported by nvidia-smi
    or rocm-smi is not duplicated from WMI or lspci, and generic entries carry no
    VRAM figure because those sources are unreliable for it.

    ``vulkaninfo`` runs last and only when *no* card was sized at all, which is the
    ordering the two facts about it force. It is the weaker source -- a driver's
    allocation budget rather than the card's own memory -- so it must not overwrite a
    vendor tool's answer; and it is the source most often absent, so running it whenever
    anything was unsized would put a failed probe, and a ``doctor`` warning about a tool
    nobody needed, in front of every desktop that pairs a working nvidia-smi with an
    integrated chip WMI also lists. Nothing would be planned on that chip either way.
    Running it only when the machine has nothing else means the warning appears exactly
    when installing the tool is worth the reader's time.

    macOS never reaches it: MoltenVK is a developer installation rather than something a
    Mac has, and on Apple silicon there is no separate VRAM pool for it to report. An
    Intel Mac's discrete card is the one machine this leaves unsized; see
    ``docs/platform-support.md``.
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

    if found and all(gpu.vram_total_bytes is None for gpu in found):
        devices, rec = probe("vulkaninfo", runner, _VULKANINFO_CMD, parse_vulkaninfo)
        probes.append(rec)
        fill_vram_from_vulkan(found, devices or [])

    for i, gpu in enumerate(found):
        gpu.index = i
    return found, probes
