import json

from llamafit.hardware.gpu import detect_gpus, parse_nvidia_smi, vendor_from_name
from llamafit.hardware.runner import FakeRunner

NVIDIA = "0, NVIDIA GeForce RTX 4060, 8188, 550, 610.88\n"
ROCM = json.dumps(
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
