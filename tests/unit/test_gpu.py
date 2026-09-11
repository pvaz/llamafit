import json

import pytest

from llamafit.hardware.gpu import (
    detect_gpus,
    fill_vram_from_vulkan,
    parse_nvidia_smi,
    parse_vulkaninfo,
    vendor_from_name,
)
from llamafit.hardware.runner import FakeRunner
from llamafit.models.host import Gpu

NVIDIA = "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n"
ROCM = json.dumps(
    {
        "card0": {
            # "Card series", not "Card Series": ROCm 5 spells it this way and ROCm 6
            # renamed it, and this fixture used to spell it the way the parser did rather
            # than the way either release does. A fixture written from the code instead of
            # from the tool tests that the code agrees with itself.
            "Card series": "Radeon RX 7900 XTX",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "1048576000",
        }
    }
)
ROCM_6 = json.dumps(
    {
        "card0": {
            "Card Series": "Radeon RX 7900 XTX",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "1048576000",
        }
    }
)
APPLE = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M3 Max", "sppci_cores": "40"}]})
WMI = json.dumps(
    [
        {"Name": "NVIDIA GeForce RTX 4060", "AdapterRAM": 4293918720},
        {"Name": "Intel(R) UHD Graphics 770", "AdapterRAM": 1073741824},
    ]
)
LSPCI = (
    "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD107 "
    "[GeForce RTX 4060] [10de:2882]\n"
)


def test_vendor_from_name() -> None:
    assert vendor_from_name("NVIDIA GeForce RTX 4060") == "nvidia"
    assert vendor_from_name("AMD Radeon RX 7900 XTX") == "amd"
    assert vendor_from_name("Intel(R) Arc(TM) A770") == "intel"
    assert vendor_from_name("Apple M3 Max") == "apple"
    assert vendor_from_name("Matrox G200") == "other"


def test_parse_nvidia_smi_converts_mib_to_bytes() -> None:
    gpus = parse_nvidia_smi(NVIDIA)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "NVIDIA GeForce RTX 4060"
    assert gpu.vram_total_bytes == 8188 * 1024**2
    assert gpu.vram_used_bytes == 550 * 1024**2
    assert gpu.driver == "610.88"
    assert gpu.backend_hint == "cuda"


def test_windows_prefers_nvidia_smi_and_adds_others_from_wmi() -> None:
    runner = FakeRunner({"nvidia-smi": NVIDIA, "powershell": WMI})
    gpus, probes = detect_gpus(runner, "windows")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4060", "Intel(R) UHD Graphics 770"]
    assert gpus[0].vram_total_bytes == 8188 * 1024**2  # from nvidia-smi, not the WMI 4 GB cap
    assert gpus[1].vram_total_bytes is None  # WMI AdapterRAM is unreliable: name only
    assert gpus[1].backend_hint == "vulkan"
    assert {p.name for p in probes} == {"nvidia-smi", "rocm-smi", "wmi-video"}


def test_linux_amd_from_rocm_smi() -> None:
    runner = FakeRunner({"rocm-smi": ROCM, "lspci": ""})
    gpus, _ = detect_gpus(runner, "linux")
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_total_bytes == 25753026560
    assert gpus[0].backend_hint == "hip"


def test_macos_apple_silicon_has_no_vram_figure() -> None:
    runner = FakeRunner({"system_profiler": APPLE})
    gpus, _ = detect_gpus(runner, "macos")
    assert gpus[0].vendor == "apple" and gpus[0].backend_hint == "metal"
    assert gpus[0].vram_total_bytes is None


def test_no_tools_means_no_gpus_but_probes_explain() -> None:
    gpus, probes = detect_gpus(FakeRunner({}), "linux")
    assert gpus == []
    assert probes and all(not p.ok for p in probes)


def test_parse_lspci_prefers_the_marketing_name_and_skips_pci_ids() -> None:
    from llamafit.hardware.gpu import parse_lspci

    two = (
        "00:02.0 VGA compatible controller [0300]: Intel Corporation UHD Graphics 770 [8086:4680]\n"
    )
    assert [g.name for g in parse_lspci(LSPCI)] == ["GeForce RTX 4060"]
    assert [g.name for g in parse_lspci(two)] == ["Intel Corporation UHD Graphics 770"]
    assert parse_lspci(two)[0].vendor == "intel"


def test_generic_listing_keeps_a_distinct_model_with_a_shared_prefix() -> None:
    wmi = json.dumps([{"Name": "NVIDIA GeForce RTX 4060"}, {"Name": "NVIDIA GeForce RTX 4060 Ti"}])
    gpus, _ = detect_gpus(FakeRunner({"nvidia-smi": NVIDIA, "powershell": wmi}), "windows")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 4060 Ti"]


def test_lspci_name_without_vendor_prefix_is_deduplicated() -> None:
    gpus, _ = detect_gpus(FakeRunner({"nvidia-smi": NVIDIA, "lspci": LSPCI}), "linux")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4060"]


def test_parse_nvidia_smi_tolerates_na_fields() -> None:
    out = (
        "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n1, NVIDIA Tesla K80, [N/A], [N/A], 470.00\n"
    )
    gpus = parse_nvidia_smi(out)
    assert len(gpus) == 2
    assert gpus[1].vram_total_bytes is None and gpus[1].vram_used_bytes is None


def test_rocm_smi_is_read_whichever_way_the_release_spells_its_keys() -> None:
    """ROCm 5 prints ``Card series`` and ROCm 6 prints ``Card Series``; both are AMD cards."""
    from llamafit.hardware.gpu import parse_rocm_smi

    for out in (ROCM, ROCM_6):
        gpus = parse_rocm_smi(out)
        assert [g.name for g in gpus] == ["Radeon RX 7900 XTX"]
        assert gpus[0].vram_total_bytes == 25753026560
        assert gpus[0].vram_source == "measured"


# --- vulkaninfo: the last source there is for a card nobody else can size -------------

VULKANINFO = """\
==========
VULKANINFO
==========

Vulkan Instance Version: 1.3.280

Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
\tapiVersion         = 1.3.280
\tdriverVersion      = 2.0.310
\tvendorID           = 0x1002
\tdeviceID           = 0x744c
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = AMD Radeon RX 7900 XTX

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 3
\tmemoryHeaps[0]:
\t\tsize   = 25757220864 (0x5ff800000) (23.99 GiB)
\t\tbudget = 25204359168 (0x5de800000) (23.47 GiB)
\t\tusage  = 268435456 (0x10000000) (256.00 MiB)
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[1]:
\t\tsize   = 268435456 (0x10000000) (256.00 MiB)
\t\tbudget = 268435456 (0x10000000) (256.00 MiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[2]:
\t\tsize   = 33395412992 (0x7c6f79000) (31.10 GiB)
\t\tflags:
\t\t\tNone
memoryTypes: count = 2
\tmemoryTypes[0]:
\t\theapIndex     = 2
GPU1:
VkPhysicalDeviceProperties:
---------------------------
\tvendorID           = 0x8086
\tdeviceID           = 0xa780
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = Intel(R) UHD Graphics 770

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
\tmemoryHeaps[0]:
\t\tsize   = 33395412992 (0x7c6f79000) (31.10 GiB)
\t\tbudget = 30064771072 (0x700000000) (28.00 GiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 1
"""

VULKANINFO_NO_BUDGET = """\
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = AMD Radeon RX 7900 XTX

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
\tmemoryHeaps[0]:
\t\tsize   = 25757220864 (0x5ff800000) (23.99 GiB)
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 1
"""

VULKANINFO_INTEGRATED_ONLY = VULKANINFO.split("GPU1:")[0].replace(
    "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU", "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
)

# 25,204,359,168 promised minus the 268,435,456 already taken, and the 23.99 GiB heap
# minus that: the arithmetic written out, so a change to it has to be a deliberate one.
VULKAN_TOTAL = 25757220864
VULKAN_FREE = 24935923712
LSPCI_AMD = (
    "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
    "[AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c]\n"
)
WMI_AMD = json.dumps([{"Name": "AMD Radeon RX 7900 XTX"}])


def test_vulkaninfo_reads_the_device_local_heap_and_the_drivers_own_budget() -> None:
    gpus = parse_vulkaninfo(VULKANINFO)
    assert [g.name for g in gpus] == ["AMD Radeon RX 7900 XTX"], "integrated devices are skipped"
    gpu = gpus[0]
    assert gpu.vram_total_bytes == VULKAN_TOTAL
    assert gpu.vram_free_bytes == VULKAN_FREE
    assert gpu.vram_source == "estimated", "a driver's promise is not a vendor tool's reading"
    assert gpu.backend_hint == "vulkan"


def test_vulkaninfo_ignores_the_small_resizable_bar_heap_behind_the_first() -> None:
    """The 256 MiB host-visible heap is a window onto the first one, not more memory."""
    assert parse_vulkaninfo(VULKANINFO)[0].vram_total_bytes != 268435456


def test_vulkaninfo_without_the_memory_budget_extension_gives_a_size_and_no_free() -> None:
    gpu = parse_vulkaninfo(VULKANINFO_NO_BUDGET)[0]
    assert gpu.vram_total_bytes == VULKAN_TOTAL
    assert gpu.vram_free_bytes is None, "a size nobody can spend is not a budget"


def test_vulkaninfo_with_nothing_discrete_is_a_probe_that_ran_and_did_not_answer() -> None:
    """An integrated heap is system memory under another name, and would be counted twice."""
    with pytest.raises(ValueError):
        parse_vulkaninfo(VULKANINFO_INTEGRATED_ONLY)


def test_windows_amd_is_sized_from_the_vulkan_driver_when_rocm_is_not_there() -> None:
    """The machine the finding is about: no rocm-smi off Linux, so WMI knows only a name."""
    runner = FakeRunner({"powershell": WMI_AMD, "vulkaninfo": VULKANINFO})
    gpus, probes = detect_gpus(runner, "windows")
    assert [g.name for g in gpus] == ["AMD Radeon RX 7900 XTX"]
    assert gpus[0].vram_free_bytes == VULKAN_FREE
    assert gpus[0].vram_source == "estimated"
    assert [p.name for p in probes if p.ok] == ["wmi-video", "vulkaninfo"]


def test_linux_amd_is_sized_from_vulkan_though_lspci_calls_it_something_else() -> None:
    """``Navi 31 [Radeon RX ...]`` will never normalise to ``AMD Radeon RX 7900 XTX``.

    One unsized card, one unclaimed discrete device, and nothing else either of them
    could be: the fallback the name match cannot reach.
    """
    runner = FakeRunner({"lspci": LSPCI_AMD, "vulkaninfo": VULKANINFO})
    gpus, _ = detect_gpus(runner, "linux")
    assert len(gpus) == 1 and gpus[0].vram_free_bytes == VULKAN_FREE


def test_vulkaninfo_is_not_run_when_a_vendor_tool_already_sized_the_card() -> None:
    """A desktop with an NVIDIA card and an integrated chip must not be told to install it."""
    runner = FakeRunner({"nvidia-smi": NVIDIA, "powershell": WMI})
    gpus, probes = detect_gpus(runner, "windows")
    assert [p.name for p in probes] == ["nvidia-smi", "rocm-smi", "wmi-video"]
    assert gpus[0].vram_total_bytes == 8188 * 1024**2
    assert "vulkaninfo" not in [call[0] for call in runner.calls]


def test_two_identical_unsized_cards_are_not_sized_from_one_heap_each() -> None:
    """Ambiguity is left as it was found: a guess is what this probe exists to remove."""
    cards = [
        Gpu(index=0, vendor="amd", name="Mystery Card"),
        Gpu(index=1, vendor="amd", name="Mystery Card"),
    ]
    fill_vram_from_vulkan(cards, parse_vulkaninfo(VULKANINFO))
    assert [c.vram_total_bytes for c in cards] == [None, None]


def test_a_card_the_vulkan_probe_could_not_reach_stays_unsized() -> None:
    """An unsized card is left unsized rather than given the nearest number to hand."""
    runner = FakeRunner({"powershell": WMI_AMD, "vulkaninfo": ""})
    gpus, probes = detect_gpus(runner, "windows")
    assert gpus[0].vram_total_bytes is None
    assert gpus[0].vram_source == "unknown"
    assert [p.ok for p in probes if p.name == "vulkaninfo"] == [False]
