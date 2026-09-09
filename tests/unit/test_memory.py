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


DMIDECODE = """# dmidecode 3.4
Getting SMBIOS data from sysfs.

Handle 0x0009, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 64 GB
    Locator: DIMM 0
    Type: DDR5
    Type Detail: Synchronous Unbuffered (Unregistered)
    Speed: 4800 MT/s
    Configured Memory Speed: 4200 MT/s

Handle 0x000A, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 64 GB
    Locator: DIMM 1
    Type: DDR5
    Speed: 4800 MT/s
    Configured Memory Speed: 4200 MT/s

Handle 0x000B, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: No Module Installed
    Locator: DIMM 2
    Type: Unknown
"""


def vm() -> tuple[int, int]:
    return 128 * 1024**3, 100 * 1024**3


def test_theoretical_bandwidth() -> None:
    assert theoretical_bandwidth_gbps(4200, 2) == 67.2


def test_windows_modules_give_type_speed_and_module_count_but_no_channel_count() -> None:
    runner = FakeRunner({"powershell": WIN_MODULES})
    memory, probes = detect_memory(runner, "windows", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type == "DDR5"
    assert memory.speed_mts == 4200
    assert memory.modules == 2
    assert memory.channels is None, "no Windows source reports the channel count"
    assert memory.bandwidth_gbps is None
    assert memory.bandwidth_source == "unknown"
    assert probes[-1].name == "memory-modules" and probes[-1].ok


def test_macos_memory_type_without_speed() -> None:
    runner = FakeRunner({"system_profiler": MAC_MEMORY})
    memory, _ = detect_memory(runner, "macos", vm_provider=vm)
    assert memory.type == "LPDDR5"
    assert memory.speed_mts is None
    assert memory.modules == 1
    assert memory.channels is None
    assert memory.bandwidth_source == "unknown"


def test_dmidecode_gives_type_speed_and_module_count() -> None:
    runner = FakeRunner({"dmidecode": DMIDECODE})
    memory, _ = detect_memory(runner, "linux", vm_provider=vm)
    assert memory.type == "DDR5"
    assert memory.speed_mts == 4200
    assert memory.modules == 2, "the empty slot is not counted"
    assert memory.channels is None
    assert memory.bandwidth_source == "unknown"


def test_failed_module_probe_keeps_totals() -> None:
    memory, probes = detect_memory(FakeRunner({}), "linux", vm_provider=vm)
    assert memory.total_bytes == 128 * 1024**3
    assert memory.type is None
    assert not probes[-1].ok


def test_failed_totals_provider_is_a_probe_not_an_exception() -> None:
    def broken() -> tuple[int, int]:
        raise RuntimeError("no psutil")

    memory, probes = detect_memory(FakeRunner({}), "linux", vm_provider=broken)
    assert memory.total_bytes == 0 and memory.available_bytes == 0
    assert probes[0].name == "memory-totals" and not probes[0].ok
