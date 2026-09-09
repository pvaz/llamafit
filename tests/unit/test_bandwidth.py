import pytest

from llamafit.hardware.bandwidth import (
    ASSUMED_RAM_BANDWIDTH_GBPS,
    MAX_BANDWIDTH_WORKERS,
    PLAUSIBLE_RANGE_GBPS,
    PURE_PYTHON_CORRECTION,
    _default_workers,
    _measure_single_threaded,
    _time_passes,
    measure_ram_read_bandwidth_gbps,
    resolve_memory_bandwidth,
)
from llamafit.models import Memory


def test_resolve_keeps_estimate_when_not_measuring() -> None:
    memory = Memory(
        total_bytes=1, available_bytes=1, bandwidth_gbps=67.2, bandwidth_source="estimated"
    )
    out = resolve_memory_bandwidth(memory, measure=False)
    assert out.bandwidth_gbps == 67.2 and out.bandwidth_source == "estimated"


def test_resolve_assumes_when_nothing_known() -> None:
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=False)
    assert out.bandwidth_gbps == ASSUMED_RAM_BANDWIDTH_GBPS and out.bandwidth_source == "assumed"


def test_resolve_uses_the_label_the_measurement_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plausible fallback measurement is labelled ``estimated``, not forced to ``measured``."""
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (12.3, "estimated")
    )
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == 12.3 and out.bandwidth_source == "estimated"


def test_resolve_passes_through_a_measured_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (41.0, "measured")
    )
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == 41.0 and out.bandwidth_source == "measured"


def test_resolve_falls_back_when_measurement_is_implausible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps",
        lambda: (99999.0, "measured"),
    )
    memory = Memory(
        total_bytes=1, available_bytes=1, bandwidth_gbps=67.2, bandwidth_source="estimated"
    )
    out = resolve_memory_bandwidth(memory, measure=True)
    assert out.bandwidth_gbps == 67.2 and out.bandwidth_source == "estimated"


def test_resolve_falls_back_when_measurement_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: None)
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == ASSUMED_RAM_BANDWIDTH_GBPS and out.bandwidth_source == "assumed"


def test_default_workers_is_capped_at_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psutil.cpu_count", lambda logical=True: 32)
    assert _default_workers() == MAX_BANDWIDTH_WORKERS


def test_default_workers_is_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psutil.cpu_count", lambda logical=True: None)
    assert _default_workers() == 1


def test_pure_python_fallback_is_labelled_estimated() -> None:
    """The single-threaded fallback never claims to be a real measurement."""
    result = _measure_single_threaded(total_bytes=1024 * 1024, duration_s=0.01)
    assert result is not None
    value, source = result
    assert source == "estimated"
    assert value > 0


def test_time_passes_applies_the_correction_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin down the arithmetic: two passes of a 1 GB buffer over 1 s, scaled by 3.0x."""
    times = iter([0.0, 0.01, 0.02, 0.06, 1.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))
    result = _time_passes(
        lambda: None,
        total_bytes=1_000_000_000,
        duration_s=0.05,
        correction=PURE_PYTHON_CORRECTION,
        source="estimated",
    )
    assert result == (6.0, "estimated")


def test_numpy_measurement_is_labelled_measured() -> None:
    """NumPy is guaranteed present through the ``dev`` extra, so this exercises the real path."""
    result = measure_ram_read_bandwidth_gbps(duration_s=0.01, buffer_mb=8)
    assert result is not None
    value, source = result
    assert source == "measured"
    assert value > 0


def test_numpy_measurement_honours_an_explicit_worker_count() -> None:
    result = measure_ram_read_bandwidth_gbps(duration_s=0.01, buffer_mb=8, workers=1)
    assert result is not None
    value, source = result
    assert source == "measured"
    assert value > 0


@pytest.mark.hardware
def test_measurement_is_plausible_on_a_real_machine() -> None:
    result = measure_ram_read_bandwidth_gbps()
    assert result is None or PLAUSIBLE_RANGE_GBPS[0] <= result[0] <= PLAUSIBLE_RANGE_GBPS[1]


@pytest.mark.hardware
def test_read_bandwidth_scales_linearly_with_buffer_size_beyond_cache() -> None:
    """A cache-resident buffer would report an inflated, size-independent figure.

    Comparing well-beyond-cache buffers from 256 MiB up to 2 GiB confirms the reduction
    is genuinely reading from RAM each pass (time scaling with size, not a constant-time
    no-op the buffer size happened to multiply) rather than being served from cache.
    """
    results = [
        measure_ram_read_bandwidth_gbps(duration_s=0.1, buffer_mb=buffer_mb)
        for buffer_mb in (256, 1024, 2048)
    ]
    assert all(result is not None for result in results)
    values = [result[0] for result in results if result is not None]
    # Loose bounds: true DRAM bandwidth is roughly constant across buffer sizes once all
    # are well beyond the last-level cache, unlike a cache-resident buffer which would not
    # slow down at all as it grows. This only rules out a gross cache artifact.
    for value in values[1:]:
        assert 0.5 <= value / values[0] <= 2.0
