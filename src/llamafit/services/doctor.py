# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``diagnose``: turn a report into findings a person can act on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from llamafit.i18n import LazyString, _, lazy_gettext, pgettext
from llamafit.models.report import SystemReport
from llamafit.units import format_bytes, localise_number

Level = Literal["ok", "warn", "error"]
_ORDER: dict[Level, int] = {"ok": 0, "warn": 1, "error": 2}

PROBE_HINTS: dict[str, LazyString] = {
    "nvidia-smi": lazy_gettext(
        "Install or repair the NVIDIA driver; nvidia-smi ships with it and must be on PATH."
    ),
    "rocm-smi": lazy_gettext(
        "Install ROCm (Linux) to expose AMD VRAM figures; without it the GPU is "
        "listed by name only."
    ),
    "wmi-video": lazy_gettext(
        "PowerShell could not list video controllers; run from a normal user session."
    ),
    "lspci": lazy_gettext(
        "Install pciutils to list GPUs by name when no vendor tool is available."
    ),
    "memory-modules": lazy_gettext(
        "DDR type and speed were not readable; on Linux run once with sudo "
        "(dmidecode) or accept the measured bandwidth instead."
    ),
    "cpuinfo": lazy_gettext("py-cpuinfo failed; the CPU model and instruction sets are unknown."),
    "cpu-cores": lazy_gettext(
        "psutil could not count the CPU cores; thread choice falls back to one core. "
        "Reinstall psutil with `pip install --force-reinstall psutil`."
    ),
    "memory-totals": lazy_gettext(
        "psutil could not read the memory totals; budgets cannot be computed "
        "until this works. Reinstall psutil with `pip install --force-reinstall psutil`."
    ),
    "sysctl-perflevel": lazy_gettext("Could not read performance-core count from sysctl."),
    "llama-server --version": lazy_gettext(
        "llama-server exists but did not report a version; the binary may be broken."
    ),
    "system-profiler": lazy_gettext("system_profiler failed; GPU information is unavailable."),
    "vulkaninfo": lazy_gettext(
        "Install the Vulkan tools (vulkan-tools on Linux, the vendor's Vulkan runtime on "
        "Windows) so an unsized card's memory can be read from its driver. This probe "
        "runs only when no vendor tool sized the card, so it is the last source there is."
    ),
}
"""What would unlock each probe, deferred because this table is built at import time.

An eager ``_()`` here would look right and be wrong: the dictionary is filled while the
module loads, before any interface has chosen a language, so every hint would be English
for the life of the process and nothing anywhere would say so. The values are
``LazyString``, not ``str``, so every read of one passes through :func:`_hint` or an
explicit ``str()`` before it reaches a pydantic field.
"""

_BACKEND_FOR_VENDOR = {
    "nvidia": ("cuda", "CUDA"),
    "amd": ("hip", "ROCm/HIP"),
    "apple": ("metal", "Metal"),
    "intel": ("sycl", "SYCL"),
}
_VRAM_HINT_PROBE = {"nvidia": "nvidia-smi", "amd": "rocm-smi"}
_VENDOR_PROBE = {"nvidia-smi": "nvidia", "rocm-smi": "amd", "system-profiler": "apple"}
_GENERIC_VRAM_HINT = lazy_gettext(
    "No vendor tool reported the GPU's memory size; record the size in a hardware "
    "profile so budgets can be computed for this machine."
)
_LOW_DISK_BYTES = 20 * 1024**3


class Finding(BaseModel):
    """One line of ``doctor`` output."""

    level: Level
    title: str
    detail: str
    hint: str | None = None


class Diagnosis(BaseModel):
    """All findings for a report, worst first."""

    report: SystemReport
    findings: list[Finding] = Field(default_factory=list)

    @property
    def worst_level(self) -> Level:
        """``error`` if any finding is an error, else ``warn``, else ``ok``."""
        worst = max((_ORDER[f.level] for f in self.findings), default=0)
        return next(level for level, rank in _ORDER.items() if rank == worst)


def _hint(probe_name: str, fallback: LazyString | None = None) -> str | None:
    """One probe's hint as text, resolved now, in the language now installed.

    ``Finding.hint`` is a ``str`` and a :class:`LazyString` is deliberately not one, so
    the conversion has to happen somewhere. Here is the right somewhere: a diagnosis is
    built after the language has been chosen, which is exactly the render moment the
    deferred form was waiting for.
    """
    hint = PROBE_HINTS.get(probe_name, fallback)
    return None if hint is None else str(hint)


def diagnose(report: SystemReport) -> Diagnosis:
    """Apply the diagnostic rules to a report."""
    findings: list[Finding] = []
    host, llamacpp = report.host, report.llamacpp

    if llamacpp.installed:
        findings.append(
            Finding(
                level="ok",
                title=_("llama.cpp installed"),
                detail=_("build %(build)s at %(path)s, backends: %(backends)s")
                % {
                    "build": llamacpp.build or pgettext("llama.cpp build", "unknown"),
                    "path": llamacpp.path,
                    "backends": ", ".join(llamacpp.backends)
                    or pgettext("backends", "none detected"),
                },
            )
        )
    else:
        findings.append(
            Finding(
                level="error",
                title=_("llama.cpp not found"),
                detail="; ".join(llamacpp.problems) or _("no llama-server binary was found"),
                hint=_(
                    "Run `llamafit install llama.cpp`, which picks the right build for "
                    "this machine, checks it against its published checksum, and never "
                    "touches an installation LlamaFit did not create."
                ),
            )
        )

    # An unreadable memory total is not a missing detail: every budget LlamaFit will
    # compute starts from it, so the machine cannot be sized at all until it is fixed.
    if host.memory.total_bytes <= 0:
        findings.append(
            Finding(
                level="error",
                title=_("Memory size unknown"),
                detail=_(
                    "LlamaFit could not read this machine's memory totals, "
                    "so it cannot work out what will fit."
                ),
                hint=_hint("memory-totals"),
            )
        )

    for gpu in host.gpus:
        expected = _BACKEND_FOR_VENDOR.get(gpu.vendor)
        if (
            expected
            and llamacpp.installed
            and expected[0] not in llamacpp.backends
            and "vulkan" not in llamacpp.backends
        ):
            findings.append(
                Finding(
                    level="warn",
                    title=_("%(gpu)s: no %(backend)s backend in llama.cpp")
                    % {"gpu": gpu.name, "backend": expected[1]},
                    detail=_("llama.cpp was built with: %(backends)s")
                    % {"backends": ", ".join(llamacpp.backends) or pgettext("backends", "unknown")},
                    hint=_(
                        "Install a llama.cpp build with the %(backend)s or Vulkan backend "
                        "to use %(gpu)s."
                    )
                    % {"backend": expected[1], "gpu": gpu.name},
                )
            )
        if gpu.vram_total_bytes is None and not host.unified_memory:
            findings.append(
                Finding(
                    level="warn",
                    title=_("%(gpu)s: VRAM size unknown") % {"gpu": gpu.name},
                    detail=_("the vendor tool that reports memory was not available"),
                    hint=_hint(_VRAM_HINT_PROBE.get(gpu.vendor, ""), _GENERIC_VRAM_HINT),
                )
            )
    # The finding above says a size is missing. This one says what LlamaFit did about it,
    # and they are not the same thing: a reader told only that a figure is unknown has no
    # reason to suspect that the speeds on the board were produced by pretending their
    # card is not there. Every figure this program prints says where it came from, and a
    # figure that came from an assumption has to say that it did.
    unsized = host.unsized_gpus
    if unsized:
        findings.append(
            Finding(
                level="warn",
                title=_("%(gpus)s: planned around, not planned on")
                % {"gpus": ", ".join(gpu.name for gpu in unsized)},
                detail=_(
                    "with no reading of how much of the card is free, every budget and "
                    "every speed on this machine was computed as if there were no card: "
                    "they are CPU-only figures, and the real machine will be faster."
                ),
                hint=_(
                    "Install the Vulkan tools so the size can be read from the driver, or "
                    "record it in a hardware profile (`llamafit hardware path` says where "
                    "those go) and pass --profile."
                ),
            )
        )
    if not host.gpus:
        findings.append(
            Finding(
                level="warn",
                title=_("No GPU detected"),
                detail=_("models will run on the CPU only"),
                hint=_("If a GPU is present, check that its driver tools are installed."),
            )
        )

    if host.memory.bandwidth_source == "assumed":
        findings.append(
            Finding(
                level="warn",
                title=_("RAM bandwidth assumed"),
                detail=_("using %(gbps)s GB/s as a default")
                % {"gbps": localise_number(str(host.memory.bandwidth_gbps))},
                hint=_(
                    "Install numpy (`pip install llamafit[fast]`) so LlamaFit can measure "
                    "the RAM bandwidth."
                ),
            )
        )

    vendors = {g.vendor for g in host.gpus}
    for probe in [*host.probes, *llamacpp.probes]:
        if probe.ok or probe.name.startswith("server:"):
            continue
        vendor = _VENDOR_PROBE.get(probe.name)
        if vendor is not None and host.gpus and vendor not in vendors:
            continue
        findings.append(
            Finding(
                level="warn",
                title=_("Probe %(probe)s failed") % {"probe": probe.name},
                detail=probe.error or pgettext("probe error", "unknown error"),
                hint=_hint(probe.name),
            )
        )

    for server in llamacpp.running_servers:
        findings.append(
            Finding(
                level="ok",
                title=_("llama-server running at %(url)s") % {"url": server.url},
                detail=_("model %(model)s, context %(context)s")
                % {
                    "model": server.model or pgettext("model name", "unknown"),
                    "context": server.n_ctx or pgettext("context length", "unknown"),
                },
            )
        )

    for disk in host.disks:
        if disk.free_bytes < _LOW_DISK_BYTES:
            findings.append(
                Finding(
                    level="warn",
                    title=_("Low disk space on %(path)s") % {"path": disk.path},
                    detail=_("%(size)s free") % {"size": format_bytes(disk.free_bytes)},
                    hint=_(
                        "Most useful models need 5 to 120 GB; free space or change the "
                        "downloads directory."
                    ),
                )
            )

    findings.sort(key=lambda f: -_ORDER[f.level])
    return Diagnosis(report=report, findings=findings)
