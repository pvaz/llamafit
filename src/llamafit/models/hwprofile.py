# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""A machine written down instead of probed: the hardware profile.

A profile is a :class:`~llamafit.models.host.Host` a person can type. That is the whole
design (section 4.4): the scan produces a ``Host``, a profile produces the same ``Host``
from a file, and nothing downstream is told which it got. What downstream *is* told is
that the machine is not this one -- :class:`~llamafit.models.host.Simulation` rides along
on the host itself -- because a figure computed for a machine nobody is sitting at is the
weakest kind of figure this project produces.

Three rules shape the schema, and all three are the same rule seen from different sides.

**Sizes are written the way the flags are written.** ``"128GiB"``, ``"8188MiB"``, ``"8G"``
-- the syntax :func:`llamafit.units.parse_size` already accepts and ``--memory 24G``
already uses. A field called ``total_ram_gb`` would have had to mean gibibytes to describe
a 128 GiB machine and gigabytes to describe a 272 GB/s link, and a schema that means two
things by one suffix is a schema that will be filled in wrongly.

**A figure carries the label of how it was obtained, or it is not accepted.** A profile
that states a memory bandwidth has to say whether somebody measured it, derived it from
the module specification, or picked it because it sounded safe. There is no default,
because the default would be the label nobody chose. A profile that states no bandwidth is
fine: the host then says ``unknown`` and section 10.1's fallbacks stand in, which is a
worse estimate honestly labelled rather than a better one invented.

**A field nothing can read is a field that lies.** Every value here lands somewhere in
``Host``, or in the small ``calibration`` block that phase 3 writes and this build only
stores. Section 4.4's example carries ``pcie_bandwidth_gbps``; ``Host`` has nowhere to put
it, and :func:`llamafit.speed.resolve_bandwidths` reads the link's speed out of the
bundled GPU table by card name, so a profile that set it would have been quietly ignored.
It is not in this schema. ``docs/hardware-profiles.md`` says so where a profile author
will look.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from llamafit.models.host import Arch, Backend, OsName, Source, Vendor
from llamafit.units import parse_size

PROFILE_SCHEMA_VERSION = 1
"""The layout of the profile document this build reads.

It lives with the model that reads it, the way
:data:`llamafit.catalog.loader.FACTS_SCHEMA_VERSION` lives with the facts reader: a
document from a future version is refused whole rather than half-understood.
"""

NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")
"""What a profile may be called: lowercase, digits, hyphens and dots.

``--profile`` takes the name on a command line and the loader takes it as a file stem, so
a name with a space, a slash or a capital in it would be a name that is one thing in the
file and another on the terminal.
"""

_DEFAULT_BACKEND: dict[Vendor, Backend] = {
    "nvidia": "cuda",
    "amd": "hip",
    "apple": "metal",
    "intel": "sycl",
    "other": "vulkan",
}
"""Which llama.cpp backend a card of each vendor is assumed to use.

Only a default, and only so a profile need not repeat what its vendor already implies. A
machine running a Vulkan build on an NVIDIA card says ``backend_hint`` for itself.
"""


def _as_bytes(value: Any) -> Any:
    """Read a size written as ``128GiB`` or ``8G``; pass a plain byte count through.

    Raises:
        ValueError: The string is not a size, with the text that was not one.
    """
    if isinstance(value, str):
        return parse_size(value)
    return value


Bytes = Annotated[int, BeforeValidator(_as_bytes), Field(gt=0)]
"""A size in bytes, written as a number or as ``8G``, ``7.5GiB``, ``512M``."""

ByteCount = Annotated[int, BeforeValidator(_as_bytes), Field(ge=0)]
"""The same, where zero is a real answer -- VRAM in use on an idle card, say."""


class ProfileMatch(BaseModel):
    """Rules that let a live scan recognise itself in a profile.

    A machine that has been benchmarked gets its calibration written into a user profile
    whose rules fit it (section 16), and later runs pick that profile up without being
    told to. Every rule given must hold; a profile with no rules matches nothing, which
    is what a profile of somebody else's machine should do.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_name_contains: str | None = None
    """Matched case-insensitively against the name of any of the host's cards."""

    cpu_model_contains: str | None = None
    """Matched case-insensitively against the host's CPU model string."""

    total_ram_min: Bytes | None = None
    """The host must have at least this much system memory."""

    @property
    def empty(self) -> bool:
        """Whether no rule was given, in which case nothing matches."""
        return (
            self.gpu_name_contains is None
            and self.cpu_model_contains is None
            and self.total_ram_min is None
        )


class ProfileCpu(BaseModel):
    """The processor, as far as thread choice and CPU inference care."""

    model_config = ConfigDict(extra="forbid")

    model: str
    physical_cores: int = Field(gt=0)
    logical_cores: int | None = Field(default=None, gt=0)
    """Threads the OS offers. Absent means no simultaneous multithreading."""
    performance_cores: int | None = Field(default=None, gt=0)
    isa: list[str] = Field(default_factory=list)
    """Instruction sets llama.cpp cares about: ``avx2``, ``avx512``, ``neon``, ``amx``."""

    @model_validator(mode="after")
    def _cores_are_consistent(self) -> ProfileCpu:
        if self.logical_cores is not None and self.logical_cores < self.physical_cores:
            raise ValueError("logical_cores cannot be fewer than physical_cores")
        if self.performance_cores is not None and self.performance_cores > self.physical_cores:
            raise ValueError("performance_cores cannot be more than physical_cores")
        return self


class ProfileMemory(BaseModel):
    """The system memory pool, and how its bandwidth figure was arrived at."""

    model_config = ConfigDict(extra="forbid")

    total: Bytes
    available: ByteCount | None = None
    """Free at the moment described. Absent means the whole pool; see the module note."""
    type: str | None = None
    speed_mts: int | None = Field(default=None, gt=0)
    modules: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    bandwidth_gbps: float | None = Field(default=None, gt=0)
    bandwidth_source: Source | None = None
    """How ``bandwidth_gbps`` was obtained. Required whenever there is a figure."""

    @model_validator(mode="after")
    def _bandwidth_carries_its_label(self) -> ProfileMemory:
        if self.bandwidth_gbps is not None and self.bandwidth_source in (None, "unknown"):
            raise ValueError(
                "bandwidth_gbps needs a bandwidth_source of 'measured', 'estimated' or "
                "'assumed': a bandwidth without one is a number nobody can weigh"
            )
        if self.bandwidth_gbps is None and self.bandwidth_source not in (None, "unknown"):
            raise ValueError(
                f"bandwidth_source is {self.bandwidth_source!r} but there is no "
                "bandwidth_gbps for it to describe"
            )
        if self.available is not None and self.available > self.total:
            raise ValueError("available memory cannot be more than the total")
        return self


class ProfileGpu(BaseModel):
    """One graphics device.

    ``bandwidth_gbps`` and ``compute_tflops_fp16`` may be left out, and usually should
    be: :func:`llamafit.hardware.gputable.enrich_gpu` fills them from the bundled table
    of vendor specifications, which is sourced, dated and shared by every card of that
    name. A figure typed here instead is a figure whose provenance only the profile's
    ``provenance`` note can carry.
    """

    model_config = ConfigDict(extra="forbid")

    vendor: Vendor
    name: str
    vram_total: Bytes | None = None
    """Absent on a unified-memory machine, where the card draws on the system pool."""
    vram_used: ByteCount = 0
    bandwidth_gbps: float | None = Field(default=None, gt=0)
    compute_tflops_fp16: float | None = Field(default=None, gt=0)
    backend_hint: Backend | None = None
    """Which llama.cpp backend drives it. Absent takes the vendor's usual one."""
    driver: str | None = None

    @model_validator(mode="after")
    def _used_fits_in_total(self) -> ProfileGpu:
        if self.vram_total is not None and self.vram_used > self.vram_total:
            raise ValueError("vram_used cannot be more than vram_total")
        return self

    @property
    def backend(self) -> Backend:
        """The backend this card runs, stated or taken from the vendor."""
        return self.backend_hint or _DEFAULT_BACKEND[self.vendor]


class ProfileCalibration(BaseModel):
    """Factors fitted to one machine by ``llamafit bench``.

    Stored, not applied: nothing in this build reads them, and ``hardware show`` says so
    rather than letting a reader assume a number they typed here changed an estimate.
    Phase 3 (section 16) is where they start to count.
    """

    model_config = ConfigDict(extra="forbid")

    ram_efficiency: float | None = Field(default=None, gt=0, le=1)
    vram_efficiency: float | None = Field(default=None, gt=0, le=1)
    pcie_effective_gbps: float | None = Field(default=None, gt=0)
    layer_overhead_ms: float | None = Field(default=None, ge=0)
    fixed_overhead_ms: float | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    """Where these came from: a benchmark run, or a document in this repository."""


class HardwareProfile(BaseModel):
    """One machine, described well enough to score models against without being on it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    name: str
    description: str | None = None
    provenance: str = Field(min_length=1)
    """Where every figure in this profile came from, in the profile's own words.

    Required, and required to say something. It is the field that separates a profile of
    a machine somebody measured from a profile somebody imagined, and neither the loader
    nor the schema can tell those apart on its own.
    """
    recorded_at: datetime | None = None
    """When the machine was in this state, for a profile taken from a real scan.

    It becomes the host's ``scanned_at``, so a profile recorded from a scan reproduces
    that scan exactly rather than claiming to have been taken just now.
    """
    match: ProfileMatch = Field(default_factory=ProfileMatch)
    os: OsName
    os_version: str = Field(min_length=1)
    arch: Arch
    cpu: ProfileCpu
    memory: ProfileMemory
    gpus: list[ProfileGpu] = Field(default_factory=list)
    unified_memory: bool = False
    """One pool for everything, as on Apple silicon and AMD APUs."""
    backends: list[Backend] = Field(default_factory=list)
    """llama.cpp backends this machine can use, for the record; nothing filters on it yet."""
    calibration: ProfileCalibration | None = None

    @model_validator(mode="after")
    def _is_a_document_this_build_reads(self) -> HardwareProfile:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version is {self.schema_version}, but this build reads version "
                f"{PROFILE_SCHEMA_VERSION}"
            )
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"name {self.name!r} is not usable as a profile name: use lowercase "
                "letters, digits, hyphens and dots, as in 'reference-rtx4060-128gb'"
            )
        if self.unified_memory and any(gpu.vram_total is not None for gpu in self.gpus):
            raise ValueError(
                "a unified-memory machine has no separate VRAM pool, so no card may give "
                "vram_total; the memory total is the pool"
            )
        return self

    @property
    def primary_gpu(self) -> ProfileGpu | None:
        """The card with the most VRAM, or the first one when no size is stated."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: gpu.vram_total or -1)
