"""Recorded probe outputs from the reference machine (2026-09-09)."""

import json

from llamafit.hardware.runner import FakeRunner

NVIDIA_SMI = "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n"
WMI_VIDEO = json.dumps([{"Name": "NVIDIA GeForce RTX 4060", "AdapterRAM": 4293918720}])
WMI_MEMORY = json.dumps(
    [
        {
            "SMBIOSMemoryType": 34,
            "Speed": 4800,
            "ConfiguredClockSpeed": 4200,
            "Capacity": 68719476736,
        },
        {
            "SMBIOSMemoryType": 34,
            "Speed": 4800,
            "ConfiguredClockSpeed": 4200,
            "Capacity": 68719476736,
        },
    ]
)
POWERSHELL_MEMORY_CMD = (
    "powershell -NoProfile -Command Get-CimInstance Win32_PhysicalMemory | Select-Object "
    "SMBIOSMemoryType,Speed,ConfiguredClockSpeed,Capacity | ConvertTo-Json"
)
POWERSHELL_VIDEO_CMD = (
    "powershell -NoProfile -Command Get-CimInstance Win32_VideoController | Select-Object "
    "Name,AdapterRAM | ConvertTo-Json"
)


def reference_runner() -> FakeRunner:
    return FakeRunner(
        {
            "nvidia-smi": NVIDIA_SMI,
            POWERSHELL_MEMORY_CMD: WMI_MEMORY,
            POWERSHELL_VIDEO_CMD: WMI_VIDEO,
        }
    )


def reference_cpuinfo() -> dict[str, object]:
    return {
        "brand_raw": "Intel(R) Core(TM) i9-14900KF",
        "flags": ["avx2", "avx512f", "avx512_vnni"],
    }


def reference_vm() -> tuple[int, int]:
    return 128 * 1024**3, 100 * 1024**3


def reference_cores() -> tuple[int, int]:
    return 24, 32
