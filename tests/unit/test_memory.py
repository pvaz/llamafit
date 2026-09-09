import json

from llamafit.hardware.memory import detect_memory, theoretical_bandwidth_gbps
from llamafit.hardware.runner import FakeRunner

WIN_MODULES = json.dumps(
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

MAC_MEMORY = json.dumps(
    {
        "SPMemoryDataType": [
            {"SPMemoryDataType": "64 GB", "dimm_type": "LPDDR5", "dimm_manufacturer": "Apple"}
        ]
    }
)


def vm() -> tuple[int, int]:
    return 128 * 1024**3, 100 * 1024**3


def test_theoretical_bandwidth() -> None:
    assert theoretical_bandwidth_gbps(4200, 2) == 67.2


def test_windows_modules_give_type_speed_channels_and_estimate() -> None:
    runner = FakeRunner({"powershell": WIN_MODULES})
    memory, probes = detect_memory(runner, "windows", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type == "DDR5"
    assert memory.speed_mts == 4200
    assert memory.channels == 2
    assert memory.bandwidth_gbps == 67.2
    assert memory.bandwidth_source == "estimated"
    assert probes[-1].name == "memory-modules" and probes[-1].ok


def test_macos_memory_type_without_speed() -> None:
    runner = FakeRunner({"system_profiler": MAC_MEMORY})
    memory, _ = detect_memory(runner, "macos", vm_provider=vm)
    assert memory.type == "LPDDR5"
    assert memory.speed_mts is None
    assert memory.bandwidth_source == "unknown"


def test_failed_module_probe_keeps_totals() -> None:
    memory, probes = detect_memory(FakeRunner({}), "linux", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type is None
    assert not probes[-1].ok
