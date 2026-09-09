import pytest

from llamafit.hardware import scan
from tests.fixtures.reference_machine import (
    reference_cores,
    reference_cpuinfo,
    reference_runner,
    reference_vm,
)


def test_scan_reference_machine() -> None:
    host = scan(
        reference_runner(),
        os_name="windows",
        measure_bandwidth=False,
        cpuinfo_provider=reference_cpuinfo,
        vm_provider=reference_vm,
        cores_provider=reference_cores,
    )
    assert host.os == "windows"
    assert host.cpu.physical_cores == 24
    assert host.cpu.performance_cores == 8
    assert host.memory.total_bytes == 128 * 1024**3
    assert host.memory.modules == 2 and host.memory.channels is None
    assert host.memory.bandwidth_source == "assumed", "no source reports the channel count"
    gpu = host.primary_gpu
    assert gpu is not None and gpu.name == "NVIDIA GeForce RTX 4060"
    assert gpu.bandwidth_gbps == 272 and gpu.compute_tflops_fp16 == 15
    assert host.vram_available_bytes == (8188 - 550) * 1024**2
    assert host.unified_memory is False
    assert host.disks, "the current working directory's disk is always reported"
    names = [p.name for p in host.probes]
    assert "cpuinfo" in names and "nvidia-smi" in names and "memory-modules" in names


def test_scan_survives_an_environment_without_a_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llamafit.hardware as hardware_module

    def no_home() -> object:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(hardware_module, "get_paths", no_home)
    host = scan(
        reference_runner(),
        os_name="windows",
        measure_bandwidth=False,
        cpuinfo_provider=reference_cpuinfo,
        vm_provider=reference_vm,
        cores_provider=reference_cores,
    )
    paths_probe = next(p for p in host.probes if p.name == "paths")
    assert not paths_probe.ok and "home" in (paths_probe.error or "")
    assert host.disks, "the working directory is still reported without a downloads directory"


def test_scan_with_nothing_available_still_returns_a_host() -> None:
    from llamafit.hardware.runner import FakeRunner

    def broken() -> dict[str, object]:
        raise RuntimeError("nope")

    host = scan(
        FakeRunner({}),
        os_name="linux",
        measure_bandwidth=False,
        cpuinfo_provider=broken,
        vm_provider=lambda: (8 * 1024**3, 4 * 1024**3),
        cores_provider=lambda: (2, 4),
    )
    assert host.gpus == []
    assert host.memory.bandwidth_source == "assumed"
    assert any(not p.ok for p in host.probes)
