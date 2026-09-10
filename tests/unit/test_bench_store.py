"""The store: what it keeps, what it refuses, and whether two results can be confused."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llamafit.bench.fingerprint import conditions_hash, host_fingerprint, host_summary
from llamafit.bench.store import BenchStore, database_path, open_store
from llamafit.bench.types import BENCH_SCHEMA_VERSION, PagingCheck
from llamafit.errors import ProbeError
from tests.fixtures.bench import conditions, run_of, traffic
from tests.fixtures.speed import reference_host

FINGERPRINT = "0123456789abcdef"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[BenchStore]:
    opened = BenchStore(tmp_path / "benchmarks.sqlite")
    yield opened
    opened.close()


def test_a_recorded_run_comes_back_with_everything_it_was_given(store: BenchStore) -> None:
    run = store.record(
        kind="llama-bench-tg",
        conditions=conditions(),
        gen_tps=24.7,
        traffic=traffic(device_bytes=int(1e9)),
        estimated_gen_tps=21.0,
    )
    assert run.id and run.recorded_at and run.conditions_hash
    stored = store.runs()
    assert len(stored) == 1
    assert stored[0].gen_tps == 24.7
    assert stored[0].estimated_gen_tps == 21.0
    assert stored[0].traffic is not None
    assert stored[0].conditions.llama_cpp_build == 10867


def test_the_stored_flags_name_every_setting_that_changes_a_speed(store: BenchStore) -> None:
    """The gap this package exists to close.

    ``estimate_speed`` matches a benchmark to a placement by reading flags out of a
    recorded command line, and a flag the record does not mention cannot disagree with
    anything. A record that says ``-ngl 99`` and nothing about ``--n-cpu-moe`` matches
    every offload with that layer count.
    """
    settings = {"ngl": "99", "n-cpu-moe": "48", "ub": "1024", "t": "16", "ctk": "f16"}
    run = store.record(
        kind="llama-bench-tg", conditions=conditions(settings=settings), gen_tps=24.7
    )
    flags = run.conditions.flag_string
    assert "-ngl 99" in flags
    assert "--n-cpu-moe 48" in flags
    assert "-ub 1024" in flags
    assert "-ctk f16" in flags


def test_a_sweep_row_is_filed_under_the_micro_batch_it_ran_at_not_the_list_it_was_offered(
    store: BenchStore,
) -> None:
    """One command line, three rows, three different micro-batches.

    A reader parsing ``-ub`` out of the raw command line would take 512 and file a
    2048-token measurement under it, which is why the flags a record hands out are built
    from what the tool reported rather than from what it was asked.
    """
    argv = ("llama-bench", "-m", "model.gguf", "-ub", "512,1024,2048")
    run = store.record(
        kind="llama-bench-pp",
        conditions=conditions(argv=argv, settings={"ub": "2048"}, micro_batch=2048),
        pp_tps=323.0,
    )
    assert "-ub 2048" in run.conditions.flag_string
    assert "512,1024,2048" not in run.conditions.flag_string
    assert run.conditions.argv == argv


def test_two_runs_that_differ_in_anything_that_matters_hash_differently() -> None:
    base = conditions()
    same = conditions_hash("llama-bench-tg", base)
    assert conditions_hash("llama-bench-pp", base) != same
    assert conditions_hash("llama-bench-tg", conditions(quant="Q8_0")) != same
    assert conditions_hash("llama-bench-tg", conditions(model_id="other")) != same
    assert conditions_hash("llama-bench-tg", conditions(build=10868)) != same
    assert conditions_hash("llama-bench-tg", conditions(context=32768)) != same
    assert conditions_hash("llama-bench-tg", conditions(micro_batch=512)) != same
    assert conditions_hash("llama-bench-tg", conditions(n_gen=256)) != same
    assert conditions_hash("llama-bench-tg", conditions(settings={"ngl": "0"})) != same
    assert conditions_hash("llama-bench-tg", conditions(fingerprint="other")) != same


def test_a_repeat_of_the_same_run_shares_its_hash_and_stays_a_separate_row(
    store: BenchStore,
) -> None:
    first = store.record(kind="llama-bench-tg", conditions=conditions(), gen_tps=24.7)
    second = store.record(kind="llama-bench-tg", conditions=conditions(), gen_tps=23.8)
    assert first.conditions_hash == second.conditions_hash
    assert first.id != second.id
    assert len(store.runs()) == 2


def test_repeats_are_reduced_to_their_median_on_the_way_out_not_to_their_best(
    store: BenchStore,
) -> None:
    for speed in (23.8, 24.7, 30.0):
        store.record(kind="llama-bench-tg", conditions=conditions(), gen_tps=speed)
    measurements = store.measurements(
        host_fingerprint=FINGERPRINT, model_id="qwen3-coder-next", quant="UD-Q4_K_XL"
    )
    assert len(measurements) == 1
    assert measurements[0].gen_tps == 24.7


def test_a_run_that_paged_is_kept_as_evidence_and_never_handed_out_as_a_measurement(
    store: BenchStore,
) -> None:
    store.record(
        kind="server-1k",
        conditions=conditions(context=65536),
        gen_tps=6.2,
        paging=PagingCheck(paged=True, vram_ratio=0.99, speed_ratio=0.46, reason="paging"),
    )
    assert len(store.runs()) == 1
    assert store.measurements(host_fingerprint=FINGERPRINT, model_id="qwen3-coder-next") == []


def test_a_measurement_carries_the_date_and_a_source_that_leads_back_to_the_row(
    store: BenchStore,
) -> None:
    run = store.record(
        kind="llama-bench-tg", conditions=conditions(), gen_tps=24.7, peak_vram_bytes=7 * 1024**3
    )
    measurement = store.measurements(host_fingerprint=FINGERPRINT, model_id="qwen3-coder-next")[0]
    assert measurement.date == run.recorded_at.date()
    assert run.id in (measurement.source or "")
    assert measurement.peak_vram_gb == pytest.approx(7.516, rel=1e-3)


def test_a_run_whose_report_disagrees_with_its_command_line_is_not_stored(
    store: BenchStore,
) -> None:
    """The tool ran at a different micro-batch from the one it was given."""
    with pytest.raises(ProbeError, match="did not use the settings"):
        store.record(
            kind="llama-bench-pp",
            conditions=conditions(
                argv=("llama-bench", "-m", "model.gguf", "-ub", "1024"),
                settings={"ub": "512"},
            ),
            pp_tps=118.0,
        )
    assert store.runs() == []


def test_a_sweep_is_not_a_disagreement(store: BenchStore) -> None:
    run = store.record(
        kind="llama-bench-pp",
        conditions=conditions(
            argv=("llama-bench", "-m", "model.gguf", "-ub", "512,1024,2048"),
            settings={"ub": "1024"},
            micro_batch=1024,
        ),
        pp_tps=194.0,
    )
    assert run.conditions.micro_batch == 1024


def test_a_flag_the_command_line_never_named_is_not_a_disagreement(store: BenchStore) -> None:
    """llama.cpp's own default applied, and what it turned out to be is worth recording."""
    run = store.record(
        kind="llama-bench-tg",
        conditions=conditions(argv=("llama-bench", "-m", "model.gguf"), settings={"t": "16"}),
        gen_tps=24.7,
    )
    assert "-t 16" in run.conditions.flag_string


def test_a_simulated_machine_cannot_have_been_measured(store: BenchStore) -> None:
    with pytest.raises(ProbeError, match="simulated"):
        store.record(
            kind="llama-bench-tg", conditions=conditions(fingerprint="sim-abc"), gen_tps=24.7
        )


def test_a_database_from_a_newer_llamafit_is_refused_whole(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.sqlite"
    BenchStore(path).close()
    import sqlite3

    connection = sqlite3.connect(str(path))
    connection.execute(
        "UPDATE meta SET value = ? WHERE key = 'schema_version'",
        (str(BENCH_SCHEMA_VERSION + 1),),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ProbeError, match="newer LlamaFit"):
        BenchStore(path)


def test_runs_can_be_filtered_by_machine_model_and_quant(store: BenchStore) -> None:
    store.add(run_of(conditions_=conditions(model_id="a", quant="Q8_0"), gen_tps=1.0))
    store.add(run_of(conditions_=conditions(model_id="b", quant="Q4_K_M"), gen_tps=2.0))
    store.add(run_of(conditions_=conditions(model_id="a", fingerprint="elsewhere"), gen_tps=3.0))
    assert len(store.runs(model_id="a")) == 2
    assert len(store.runs(host_fingerprint=FINGERPRINT)) == 2
    assert len(store.runs(model_id="a", quant="Q8_0")) == 1


def test_a_calibration_is_kept_and_the_latest_one_comes_back(store: BenchStore) -> None:
    from llamafit.bench.calibrate import calibrate

    first = calibrate([], host_fingerprint=FINGERPRINT)
    store.save_calibration(first)
    assert store.latest_calibration(FINGERPRINT) is not None
    assert store.latest_calibration("nobody") is None


def test_the_database_lives_where_section_fourteen_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", "/tmp/llamafit-test")
    assert database_path().name == "benchmarks.sqlite"


def test_the_store_can_be_opened_and_closed_as_a_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path))
    with open_store() as opened:
        opened.record(kind="llama-bench-tg", conditions=conditions(), gen_tps=24.7)
    with open_store() as reopened:
        assert len(reopened.runs()) == 1


def test_a_machine_is_fingerprinted_by_what_does_not_change_between_runs() -> None:
    host = reference_host()
    assert host_fingerprint(host) == host_fingerprint(host)
    busy = host.model_copy(deep=True)
    busy.gpus[0].vram_used_bytes = 7 * 1024**3
    busy.memory.available_bytes = 1024**3
    assert host_fingerprint(busy) == host_fingerprint(host)


def test_a_driver_update_retires_the_measurements_taken_before_it() -> None:
    """The safe direction: the label falls back to `estimated`, which is true."""
    host = reference_host()
    updated = host.model_copy(deep=True)
    updated.gpus[0].driver = "611.10"
    assert host_fingerprint(updated) != host_fingerprint(host)


def test_a_summary_travels_with_the_fingerprint_so_a_digest_is_not_the_only_record() -> None:
    summary = host_summary(reference_host())
    assert "RTX 4060" in summary
    assert "i9-14900KF" in summary
