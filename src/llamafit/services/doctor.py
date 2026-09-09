"""``diagnose``: turn a report into findings a person can act on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from llamafit.models.report import SystemReport
from llamafit.units import format_bytes

Level = Literal["ok", "warn", "error"]
_ORDER: dict[Level, int] = {"ok": 0, "warn": 1, "error": 2}

PROBE_HINTS: dict[str, str] = {
    "nvidia-smi": "Install or repair the NVIDIA driver; nvidia-smi ships with it and must be "
    "on PATH.",
    "rocm-smi": "Install ROCm (Linux) to expose AMD VRAM figures; without it the GPU is "
    "listed by name only.",
    "wmi-video": "PowerShell could not list video controllers; run from a normal user session.",
    "lspci": "Install pciutils to list GPUs by name when no vendor tool is available.",
    "memory-modules": "DDR type and speed were not readable; on Linux run once with sudo "
    "(dmidecode) or accept the measured bandwidth instead.",
    "cpuinfo": "py-cpuinfo failed; the CPU model and instruction sets are unknown.",
    "cpu-cores": "psutil could not count the CPU cores; thread choice falls back to one core. "
    "Reinstall psutil with `pip install --force-reinstall psutil`.",
    "memory-totals": "psutil could not read the memory totals; budgets cannot be computed "
    "until this works. Reinstall psutil with `pip install --force-reinstall psutil`.",
    "sysctl-perflevel": "Could not read performance-core count from sysctl.",
    "llama-server --version": "llama-server exists but did not report a version; the binary "
    "may be broken.",
    "system-profiler": "system_profiler failed; GPU information is unavailable.",
}

_BACKEND_FOR_VENDOR = {
    "nvidia": ("cuda", "CUDA"),
    "amd": ("hip", "ROCm/HIP"),
    "apple": ("metal", "Metal"),
    "intel": ("sycl", "SYCL"),
}
_VRAM_HINT_PROBE = {"nvidia": "nvidia-smi", "amd": "rocm-smi"}
_GENERIC_VRAM_HINT = (
    "No vendor tool reported this GPU's memory size; record it in a hardware profile "
    "so budgets can be computed for this machine."
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


def diagnose(report: SystemReport) -> Diagnosis:
    """Apply the diagnostic rules to a report."""
    findings: list[Finding] = []
    host, llamacpp = report.host, report.llamacpp

    if llamacpp.installed:
        findings.append(
            Finding(
                level="ok",
                title="llama.cpp installed",
                detail=f"build {llamacpp.build or 'unknown'} at {llamacpp.path}, "
                f"backends: {', '.join(llamacpp.backends) or 'none detected'}",
            )
        )
    else:
        findings.append(
            Finding(
                level="error",
                title="llama.cpp not found",
                detail="; ".join(llamacpp.problems) or "no llama-server binary was found",
                hint="Install llama.cpp (LlamaFit phase 2 will do this: `llamafit install "
                "llama.cpp`); until then download a release from "
                "https://github.com/ggml-org/llama.cpp/releases and put its bin "
                "directory on PATH or in LLAMA_CPP_PATH.",
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
                    title=f"{gpu.name}: no {expected[1]} backend in llama.cpp",
                    detail=f"llama.cpp was built with: {', '.join(llamacpp.backends) or 'unknown'}",
                    hint=f"Install a llama.cpp build with the {expected[1]} or Vulkan backend "
                    f"to use this GPU.",
                )
            )
        if gpu.vram_total_bytes is None and not host.unified_memory:
            findings.append(
                Finding(
                    level="warn",
                    title=f"{gpu.name}: VRAM size unknown",
                    detail="the vendor tool that reports memory was not available",
                    hint=PROBE_HINTS.get(_VRAM_HINT_PROBE.get(gpu.vendor, ""), _GENERIC_VRAM_HINT),
                )
            )
    if not host.gpus:
        findings.append(
            Finding(
                level="warn",
                title="No GPU detected",
                detail="models will run on the CPU only",
                hint="If a GPU is present, check that its driver tools are installed.",
            )
        )

    if host.memory.bandwidth_source == "assumed":
        findings.append(
            Finding(
                level="warn",
                title="RAM bandwidth assumed",
                detail=f"using {host.memory.bandwidth_gbps} GB/s as a default",
                hint="Install numpy (`pip install llamafit[fast]`) so LlamaFit can measure it.",
            )
        )

    for probe in [*host.probes, *llamacpp.probes]:
        if not probe.ok and not probe.name.startswith("server:"):
            findings.append(
                Finding(
                    level="warn",
                    title=f"Probe {probe.name} failed",
                    detail=probe.error or "unknown error",
                    hint=PROBE_HINTS.get(probe.name),
                )
            )

    for server in llamacpp.running_servers:
        findings.append(
            Finding(
                level="ok",
                title=f"llama-server running at {server.url}",
                detail=f"model {server.model or 'unknown'}, context {server.n_ctx or 'unknown'}",
            )
        )

    for disk in host.disks:
        if disk.free_bytes < _LOW_DISK_BYTES:
            findings.append(
                Finding(
                    level="warn",
                    title=f"Low disk space on {disk.path}",
                    detail=f"{format_bytes(disk.free_bytes)} free",
                    hint="Most useful models need 5 to 120 GB; free space or change the "
                    "downloads directory.",
                )
            )

    findings.sort(key=lambda f: -_ORDER[f.level])
    return Diagnosis(report=report, findings=findings)
