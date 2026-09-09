from datetime import datetime, timezone

from llamafit.models import Cpu, Gpu, Host, LlamaCpp, Memory, Probe, SystemReport


def make_host(**overrides: object) -> Host:
    base: dict[str, object] = {
        "os": "windows",
        "os_version": "11 (10.0.26200)",
        "arch": "x86_64",
        "cpu": Cpu(
            model="Intel i9-14900KF",
            physical_cores=24,
            logical_cores=32,
            performance_cores=8,
            isa=["avx2", "avx512"],
        ),
        "memory": Memory(total_bytes=128 * 1024**3, available_bytes=100 * 1024**3),
        "gpus": [
            Gpu(
                index=0,
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * 1024**2,
                vram_used_bytes=550 * 1024**2,
                backend_hint="cuda",
            ),
        ],
        "scanned_at": datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return Host(**base)  # type: ignore[arg-type]


def test_primary_gpu_is_largest() -> None:
    small = Gpu(index=1, vendor="intel", name="UHD", vram_total_bytes=1 * 1024**3)
    host = make_host(gpus=[small, *make_host().gpus])
    assert host.primary_gpu is not None
    assert host.primary_gpu.name == "NVIDIA GeForce RTX 4060"


def test_vram_available_is_total_minus_used() -> None:
    host = make_host()
    assert host.vram_available_bytes == (8188 - 550) * 1024**2


def test_no_gpu_means_no_vram() -> None:
    host = make_host(gpus=[])
    assert host.primary_gpu is None
    assert host.vram_available_bytes is None


def test_report_round_trips_through_json() -> None:
    report = SystemReport(host=make_host(), llamacpp=LlamaCpp(installed=False), version="0.1.0a1")
    again = SystemReport.model_validate_json(report.model_dump_json())
    assert again == report


def test_probe_defaults() -> None:
    probe = Probe(name="nvidia-smi", ok=False, duration_ms=3, error="not found")
    assert probe.error == "not found"
