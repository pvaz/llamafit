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


def test_probe_error_defaults_to_none() -> None:
    probe = Probe(name="nvidia-smi", ok=True, duration_ms=3)
    assert probe.error is None
    failed = Probe(name="nvidia-smi", ok=False, duration_ms=3, error="not found")
    assert failed.error == "not found"


def test_vram_free_clamps_at_zero_when_used_exceeds_total() -> None:
    gpu = Gpu(
        index=0,
        vendor="nvidia",
        name="RTX 4060",
        vram_total_bytes=8 * 1024**3,
        vram_used_bytes=9 * 1024**3,
    )
    assert gpu.vram_free_bytes == 0


# --- a card that is here and cannot be read ------------------------------------------


def unsized_card(**overrides: object) -> Gpu:
    """An AMD card as WMI reports one: a name, a backend, and no memory figures."""
    fields: dict[str, object] = {
        "index": 0,
        "vendor": "amd",
        "name": "AMD Radeon RX 7900 XTX",
        "backend_hint": "vulkan",
    }
    fields.update(overrides)
    return Gpu(**fields)  # type: ignore[arg-type]


def test_an_unsized_card_is_not_the_same_machine_as_no_card_at_all() -> None:
    host = make_host(gpus=[unsized_card()])
    assert host.vram_available_bytes is None
    assert [gpu.name for gpu in host.unsized_gpus] == ["AMD Radeon RX 7900 XTX"]
    assert make_host(gpus=[]).unsized_gpus == [], "no card is no apology"


def test_a_sized_card_beside_an_unsized_one_has_nothing_to_apologise_for() -> None:
    """A desktop with a working card and an unreadable integrated chip is a working desktop."""
    host = make_host(gpus=[make_host().gpus[0], unsized_card(index=1, vendor="intel")])
    assert host.unsized_gpus == []


def test_a_unified_memory_machine_has_no_separate_pool_to_have_failed_to_read() -> None:
    host = make_host(
        gpus=[unsized_card(vendor="apple", name="Apple M3 Max", backend_hint="metal")],
        unified_memory=True,
    )
    assert host.unsized_gpus == []


def test_a_card_read_as_completely_full_is_read_rather_than_unread() -> None:
    """Zero free is an answer; ``None`` free is the absence of one, and they differ."""
    full = unsized_card(vram_total_bytes=24 * 1024**3, vram_used_bytes=24 * 1024**3)
    assert make_host(gpus=[full]).unsized_gpus == []


def test_a_size_with_no_free_figure_is_still_a_card_nothing_can_be_planned_on() -> None:
    """What a Vulkan driver too old for VK_EXT_memory_budget leaves behind."""
    sized = unsized_card(vram_total_bytes=24 * 1024**3, vram_source="estimated")
    assert [g.name for g in make_host(gpus=[sized]).unsized_gpus] == ["AMD Radeon RX 7900 XTX"]
