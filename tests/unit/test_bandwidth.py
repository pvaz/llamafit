import pytest

from llamafit.hardware.bandwidth import (
    ASSUMED_RAM_BANDWIDTH_GBPS,
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


@pytest.mark.hardware
def test_measurement_is_plausible_on_a_real_machine() -> None:
    value = measure_ram_bandwidth_gbps()
    assert value is None or 5 <= value <= 1000
