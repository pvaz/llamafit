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
