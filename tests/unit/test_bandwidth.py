import pytest

from llamafit.hardware.bandwidth import (
    ASSUMED_RAM_BANDWIDTH_GBPS,
    MAX_BANDWIDTH_WORKERS,
    PLAUSIBLE_RANGE_GBPS,
    PURE_PYTHON_CORRECTION,
    _default_workers,
    _measure_single_threaded,
    _time_copies,
    measure_ram_bandwidth_gbps,
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
        "llamafit.hardware.bandwidth.measure_ram_bandwidth_gbps", lambda: (12.3, "estimated")
    )
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == 12.3 and out.bandwidth_source == "estimated"


def test_resolve_passes_through_a_measured_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_bandwidth_gbps", lambda: (41.0, "measured")
    )
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == 41.0 and out.bandwidth_source == "measured"


def test_resolve_falls_back_when_measurement_is_implausible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_bandwidth_gbps", lambda: (99999.0, "measured")
    )
    memory = Memory(
        total_bytes=1, available_bytes=1, bandwidth_gbps=67.2, bandwidth_source="estimated"
    )
    out = resolve_memory_bandwidth(memory, measure=True)
    assert out.bandwidth_gbps == 67.2 and out.bandwidth_source == "estimated"


def test_resolve_falls_back_when_measurement_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamafit.hardware.bandwidth.measure_ram_bandwidth_gbps", lambda: None)
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
    result = _measure_single_threaded(size=1024 * 1024, duration_s=0.01)
    assert result is not None
    value, source = result
    assert source == "estimated"
    assert value > 0


def test_time_copies_applies_the_correction_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin down the arithmetic: two passes of a 1 GB buffer over 1 s, scaled by 1.6x."""
    times = iter([0.0, 0.01, 0.02, 0.06, 1.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))
    result = _time_copies(
        lambda: None,
        size=1_000_000_000,
        duration_s=0.05,
        correction=PURE_PYTHON_CORRECTION,
        source="estimated",
    )
    assert result == (3.2, "estimated")


def test_numpy_measurement_is_labelled_measured() -> None:
    """NumPy is guaranteed present through the ``dev`` extra, so this exercises the real path."""
    result = measure_ram_bandwidth_gbps(duration_s=0.01, buffer_mb=8)
    assert result is not None
    value, source = result
    assert source == "measured"
    assert value > 0


def test_numpy_measurement_honours_an_explicit_worker_count() -> None:
    result = measure_ram_bandwidth_gbps(duration_s=0.01, buffer_mb=8, workers=1)
    assert result is not None
    value, source = result
    assert source == "measured"
    assert value > 0


@pytest.mark.hardware
def test_measurement_is_plausible_on_a_real_machine() -> None:
    result = measure_ram_bandwidth_gbps()
    assert result is None or PLAUSIBLE_RANGE_GBPS[0] <= result[0] <= PLAUSIBLE_RANGE_GBPS[1]
