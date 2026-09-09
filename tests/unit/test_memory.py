import json

import pytest

from llamafit.hardware.memory import channel_from_labels, detect_memory, theoretical_bandwidth_gbps
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


def _module(bank_label: str, device_locator: str) -> dict[str, object]:
    return {
        "SMBIOSMemoryType": 34,
        "Speed": 4800,
        "ConfiguredClockSpeed": 4200,
        "Capacity": 34359738368,
        "BankLabel": bank_label,
        "DeviceLocator": device_locator,
    }


WIN_MODULES_WITH_CHANNELS = json.dumps(
    [
        _module("BANK 0", "ChannelA-DIMM0"),
        _module("BANK 0", "ChannelA-DIMM1"),
        _module("BANK 0", "ChannelB-DIMM0"),
        _module("BANK 0", "ChannelB-DIMM1"),
    ]
)

# The four DIMMs on the actual dev machine this fix was verified on: a real, dual-channel,
# four-module board whose DeviceLocator uses "ControllerN-DIMMx" rather than any of the
# channel-letter forms above. channel_from_labels does not recognise it (see its docstring
# and tests below), so this stays an honest "channels unknown" rather than a forced guess.
WIN_MODULES_UNRECOGNISED_LOCATORS = json.dumps(
    [
        _module("BANK 0", "Controller0-DIMM0"),
        _module("BANK 0", "Controller0-DIMM1"),
        _module("BANK 0", "Controller1-DIMM0"),
        _module("BANK 0", "Controller1-DIMM1"),
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

DMIDECODE_WITH_CHANNELS = """# dmidecode 3.4
Handle 0x0009, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 32 GB
    Locator: DIMM_A1
    Bank Locator: BANK 0
    Type: DDR5
    Configured Memory Speed: 4200 MT/s

Handle 0x000A, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 32 GB
    Locator: DIMM_A2
    Bank Locator: BANK 1
    Type: DDR5
    Configured Memory Speed: 4200 MT/s

Handle 0x000B, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 32 GB
    Locator: DIMM_B1
    Bank Locator: BANK 2
    Type: DDR5
    Configured Memory Speed: 4200 MT/s

Handle 0x000C, DMI type 17, 92 bytes
Memory Device
    Array Handle: 0x0008
    Size: 32 GB
    Locator: DIMM_B2
    Bank Locator: BANK 3
    Type: DDR5
    Configured Memory Speed: 4200 MT/s
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


def test_windows_channel_labels_give_the_channel_count() -> None:
    runner = FakeRunner({"powershell": WIN_MODULES_WITH_CHANNELS})
    memory, _ = detect_memory(runner, "windows", vm_provider=vm)
    assert memory.modules == 4
    assert memory.channels == 2
    assert memory.bandwidth_gbps == 67.2
    assert memory.bandwidth_source == "estimated"


def test_windows_unrecognised_locators_leave_channels_unknown() -> None:
    """The real reference machine's own "ControllerN-DIMMx" form is not force-matched."""
    runner = FakeRunner({"powershell": WIN_MODULES_UNRECOGNISED_LOCATORS})
    memory, _ = detect_memory(runner, "windows", vm_provider=vm)
    assert memory.modules == 4
    assert memory.channels is None
    assert memory.bandwidth_gbps is None
    assert memory.bandwidth_source == "unknown"


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


def test_dmidecode_channel_labels_give_the_channel_count() -> None:
    runner = FakeRunner({"dmidecode": DMIDECODE_WITH_CHANNELS})
    memory, _ = detect_memory(runner, "linux", vm_provider=vm)
    assert memory.modules == 4
    assert memory.channels == 2
    assert memory.bandwidth_gbps == 67.2
    assert memory.bandwidth_source == "estimated"


@pytest.mark.parametrize(
    ("bank_label", "device_locator", "expected"),
    [
        # The forms the task calls out explicitly, in either field.
        (None, "ChannelA-DIMM1", "a"),
        ("Channel A Slot 0", None, "a"),
        (None, "DIMM_A1", "a"),
        (None, "DIMM A1", "a"),
        (None, "A1_DIMM0", "a"),
        # dmidecode's bare form and a different letter, for case-insensitivity.
        ("CHANNEL B", None, "b"),
        ("bank 1", "dimm_b1", "b"),
        # A board that puts the informative label in BankLabel and a plain index in
        # DeviceLocator (common on consumer boards, e.g. "P0 CHANNEL A" / "DIMM 0").
        ("P0 CHANNEL A", "DIMM 0", "a"),
        # Rejects: uninformative or unrecognised labels must come back None, not a guess.
        ("BANK 0", None, None),
        ("BANK 0", "BANK 0", None),
        (None, "DIMM0", None),
        (None, None, None),
        # This machine's real DeviceLocator convention: a genuine gap, not to be forced.
        ("BANK 0", "Controller0-DIMM0", None),
    ],
)
def test_channel_from_labels(
    bank_label: str | None, device_locator: str | None, expected: str | None
) -> None:
    assert channel_from_labels(bank_label, device_locator) == expected


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
