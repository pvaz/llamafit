"""The two repairs that stopped the same question giving a different answer every minute.

A single fifty-millisecond window, re-run on every scan, reported between 37 and 51 GB/s
on one machine inside an hour, and every speed on every board is divided by it. These
tests are about the two halves of the answer: take the best of several windows rather than
one reading, and then keep it, under a key that names the machine rather than the moment.

The measurement itself is in ``test_bandwidth.py``. What is here is what happens around it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llamafit.hardware.bandwidth import (
    BANDWIDTH_ROUNDS,
    CACHE_FILE,
    BandwidthCache,
    _best_of,
    machine_key,
    resolve_memory_bandwidth,
)
from llamafit.models import Cpu, Memory, Source

GIB = 1024**3


# --- the best of several windows, not the average -------------------------------------


def test_the_best_window_is_kept_and_not_the_average(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contention can only slow this down, so the largest reading is the machine.

    The three windows here stand in for a run where something else held the memory
    controller for two of them. A mean would report 30 GB/s about a machine that does 50.
    """
    readings = iter([(20.0, "measured"), (50.0, "measured"), (20.0, "measured")])
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth._time_passes", lambda *_a, **_k: next(readings)
    )
    result = _best_of(lambda: None, 1, 0.01, correction=1.0, source="measured", rounds=3)
    assert result == (50.0, "measured")


def test_a_window_that_completed_no_pass_does_not_sink_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([None, (33.0, "measured"), None])
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth._time_passes", lambda *_a, **_k: next(readings)
    )
    result = _best_of(lambda: None, 1, 0.01, correction=1.0, source="measured", rounds=3)
    assert result == (33.0, "measured")


def test_no_window_completing_a_pass_is_no_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.hardware.bandwidth._time_passes", lambda *_a, **_k: None)
    assert _best_of(lambda: None, 1, 0.01, correction=1.0, source="measured", rounds=3) is None


def test_the_buffer_is_warmed_up_once_rather_than_once_per_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Faulting the pages in is not part of the measurement, and is not paid five times."""
    warmups: list[int] = []
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth._time_passes", lambda *_a, **_k: (1.0, "measured")
    )
    _best_of(
        lambda: warmups.append(1),
        1,
        0.01,
        correction=1.0,
        source="measured",
        rounds=BANDWIDTH_ROUNDS,
    )
    assert warmups == [1], "the untimed pass runs once, whatever the round count"


# --- which machine a figure belongs to ------------------------------------------------


def _cpu() -> Cpu:
    return Cpu(model="Test CPU", physical_cores=8, logical_cores=16)


def _memory() -> Memory:
    return Memory(total_bytes=64 * GIB, available_bytes=32 * GIB, speed_mts=4800, channels=2)


def test_the_key_ignores_what_was_free_and_notices_what_was_installed() -> None:
    """Free memory moves while you watch it; a key built on it files every run separately."""
    busy = _memory()
    busy.available_bytes = 1024
    assert machine_key(_memory(), _cpu()) == machine_key(busy, _cpu())


@pytest.mark.parametrize("field,value", [("speed_mts", 6000), ("channels", 4), ("type", "DDR4")])
def test_a_change_that_could_move_the_answer_retires_the_stored_figure(
    field: str, value: object
) -> None:
    changed = _memory()
    setattr(changed, field, value)
    assert machine_key(_memory(), _cpu()) != machine_key(changed, _cpu())


def test_a_different_processor_is_a_different_machine() -> None:
    other = Cpu(model="Another CPU", physical_cores=8, logical_cores=16)
    assert machine_key(_memory(), _cpu()) != machine_key(_memory(), other)


# --- keeping it, saying so, and taking it again ---------------------------------------


def test_a_kept_figure_is_used_without_measuring_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding itself: nothing about a memory controller changes between two runs."""
    cache = BandwidthCache("machine-a", tmp_path / CACHE_FILE)
    cache.put(44.4, "measured")

    def refuse() -> tuple[float, Source] | None:
        raise AssertionError("a kept figure must not be measured again")

    monkeypatch.setattr("llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", refuse)
    out = resolve_memory_bandwidth(
        Memory(total_bytes=1, available_bytes=1), measure=True, cache=cache
    )
    assert (out.bandwidth_gbps, out.bandwidth_source) == (44.4, "measured")
    assert out.bandwidth_cached is True, "a kept figure has to say that it was kept"


def test_a_measurement_is_kept_for_next_time_and_says_it_was_taken_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (51.0, "measured")
    )
    cache = BandwidthCache("machine-a", tmp_path / CACHE_FILE)
    out = resolve_memory_bandwidth(
        Memory(total_bytes=1, available_bytes=1), measure=True, cache=cache
    )
    assert out.bandwidth_gbps == 51.0 and out.bandwidth_cached is False
    assert cache.get() == (51.0, "measured")


def test_refresh_measures_again_and_replaces_what_was_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kept figure has to be refusable, or it is a number nobody can ever correct."""
    cache = BandwidthCache("machine-a", tmp_path / CACHE_FILE)
    cache.put(37.0, "measured")
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (51.0, "measured")
    )
    out = resolve_memory_bandwidth(
        Memory(total_bytes=1, available_bytes=1), measure=True, cache=cache, refresh=True
    )
    assert out.bandwidth_gbps == 51.0 and out.bandwidth_cached is False
    assert cache.get() == (51.0, "measured")


def test_a_caller_that_passes_no_cache_measures_exactly_as_it_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (41.0, "measured")
    )
    out = resolve_memory_bandwidth(Memory(total_bytes=1, available_bytes=1), measure=True)
    assert out.bandwidth_gbps == 41.0 and out.bandwidth_cached is False


def test_an_implausible_kept_figure_is_no_more_trusted_than_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = BandwidthCache("machine-a", tmp_path / CACHE_FILE)
    cache.put(99999.0, "measured")
    monkeypatch.setattr(
        "llamafit.hardware.bandwidth.measure_ram_read_bandwidth_gbps", lambda: (41.0, "measured")
    )
    out = resolve_memory_bandwidth(
        Memory(total_bytes=1, available_bytes=1), measure=True, cache=cache
    )
    assert out.bandwidth_gbps == 41.0 and out.bandwidth_cached is False


@pytest.mark.parametrize(
    "written",
    ["not json at all", "[]", '{"machine-a": 42}', '{"machine-a": {"gbps": "fast"}}'],
)
def test_a_cache_that_cannot_be_read_is_a_cache_that_is_not_there(
    tmp_path: Path, written: str
) -> None:
    path = tmp_path / CACHE_FILE
    path.write_text(written, encoding="utf-8")
    assert BandwidthCache("machine-a", path).get() is None


def test_a_missing_cache_file_is_a_miss_rather_than_a_failure(tmp_path: Path) -> None:
    assert BandwidthCache("machine-a", tmp_path / "nowhere" / CACHE_FILE).get() is None


def test_two_machines_can_share_one_cache_file(tmp_path: Path) -> None:
    """A portable LLAMAFIT_HOME carried between machines holds both, not only the last."""
    path = tmp_path / CACHE_FILE
    BandwidthCache("machine-a", path).put(40.0, "measured")
    BandwidthCache("machine-b", path).put(90.0, "measured")
    assert BandwidthCache("machine-a", path).get() == (40.0, "measured")
    assert BandwidthCache("machine-b", path).get() == (90.0, "measured")


def test_an_unwritable_cache_is_a_figure_not_kept_rather_than_a_scan_that_fails(
    tmp_path: Path,
) -> None:
    in_the_way = tmp_path / "file"
    in_the_way.write_text("", encoding="utf-8")
    BandwidthCache("machine-a", in_the_way / "under-a-file" / CACHE_FILE).put(40.0, "measured")


def test_a_machine_with_nowhere_to_keep_a_figure_measures_it_every_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache directory needs a home directory, and not every machine has one."""
    from llamafit.hardware import _bandwidth_cache

    def no_home(*_a: object, **_k: object) -> object:
        raise RuntimeError("could not resolve a home directory")

    monkeypatch.setattr("llamafit.hardware.machine_key", no_home)
    assert _bandwidth_cache(_memory(), _cpu()) is None
