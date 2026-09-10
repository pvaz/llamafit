# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What LlamaFit knows about the machine it runs on -- or about one it does not.

A ``Host`` is normally what the probes found. It can also come from a hardware profile,
or from a scan with a value substituted for a what-if, and then it describes a machine
nobody is sitting at. That difference is carried by the host itself, in
:class:`Simulation` and the ``simulated`` flag beside it, and not by whichever screen
happens to be drawing it: every service takes a ``Host``, so a mark on the ``Host`` is
the only mark that reaches all of them, ``--json`` included.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

Source = Literal["measured", "estimated", "assumed", "unknown"]
Vendor = Literal["nvidia", "amd", "apple", "intel", "other"]
Backend = Literal["cuda", "hip", "metal", "vulkan", "sycl", "cpu"]
OsName = Literal["windows", "macos", "linux"]
Arch = Literal["x86_64", "arm64", "other"]
Override = Literal["gpu_memory", "ram", "cpu_cores"]


class Probe(BaseModel):
    """Outcome of one detection step, kept so ``doctor`` can show what happened."""

    name: str
    ok: bool
    duration_ms: int
    error: str | None = None


class Cpu(BaseModel):
    """Processor facts relevant to llama.cpp thread choice and CPU inference."""

    model: str
    physical_cores: int
    logical_cores: int
    performance_cores: int | None = None
    isa: list[str] = Field(default_factory=list)


class Memory(BaseModel):
    """System memory; bandwidth carries the label of how it was obtained."""

    total_bytes: int
    available_bytes: int
    type: str | None = None
    speed_mts: int | None = None
    modules: int | None = None
    """Number of populated memory modules, which is not the channel count."""
    channels: int | None = None
    """Memory channels in use, set only when a source genuinely reports one.

    Parsed from slot labels on Windows (``BankLabel``/``DeviceLocator``) and Linux
    (``dmidecode``'s ``Bank Locator``/``Locator``) when they encode a channel letter or a
    per-controller locator such as ``ControllerN-DIMMx``; macOS's ``system_profiler``
    reports no such labels, and a label that does not encode a channel leaves this
    ``None`` rather than guessing. Only then does the theoretical bandwidth estimate get
    computed from it.
    """
    bandwidth_gbps: float | None = None
    bandwidth_source: Source = "unknown"


class Gpu(BaseModel):
    """One graphics device. Unknown values stay ``None``; they are never guessed here."""

    index: int
    vendor: Vendor
    name: str
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    bandwidth_gbps: float | None = None
    compute_tflops_fp16: float | None = None
    backend_hint: Backend = "cpu"
    driver: str | None = None

    @property
    def vram_free_bytes(self) -> int | None:
        """VRAM not in use at scan time, or ``None`` when either figure is unknown."""
        if self.vram_total_bytes is None or self.vram_used_bytes is None:
            return None
        return max(self.vram_total_bytes - self.vram_used_bytes, 0)


class Disk(BaseModel):
    """Free space at a path that matters (downloads, llama.cpp directory)."""

    path: str
    free_bytes: int
    total_bytes: int


class Simulation(BaseModel):
    """Why this host is not the machine LlamaFit is running on.

    Present exactly when something was substituted: a whole machine read from a hardware
    profile, one or more values overridden for a what-if, or both. Absent on a scan, and
    there is no third state -- a host either describes the machine in front of the user or
    it says, in a field, that it does not.
    """

    profile: str | None = None
    """Name of the hardware profile the figures came from, when they came from one."""

    path: str | None = None
    """The file that profile was read from, so a reader can go and look at it."""

    overrides: list[Override] = Field(default_factory=list)
    """Which pools were substituted by hand, in the order the flags name them.

    Identifiers, not prose: a script filters on them. The substituted values are in the
    host's own fields, because a simulated machine has to be readable as a machine.
    """


class Host(BaseModel):
    """The scanned machine -- or, when ``simulation`` is set, a machine standing in for it."""

    os: OsName
    os_version: str
    arch: Arch
    cpu: Cpu
    memory: Memory
    gpus: list[Gpu] = Field(default_factory=list)
    unified_memory: bool = False
    disks: list[Disk] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)
    scanned_at: datetime
    simulation: Simulation | None = None
    """What was substituted, or ``None`` on a host the probes actually found."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def simulated(self) -> bool:
        """Whether these figures describe a machine other than the one running LlamaFit.

        A computed field rather than a plain one, so it cannot drift from
        ``simulation`` and cannot be left out of the serialised host: every ``--json``
        consumer gets one boolean at the top of the host, whatever else it ignores.
        """
        return self.simulation is not None

    @property
    def primary_gpu(self) -> Gpu | None:
        """The GPU with the most VRAM, or the first one when sizes are unknown."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_total_bytes or -1)

    @property
    def vram_available_bytes(self) -> int | None:
        """Free VRAM on the primary GPU, ``None`` without a GPU or without figures."""
        gpu = self.primary_gpu
        return None if gpu is None else gpu.vram_free_bytes
