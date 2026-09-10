"""The estimate beside the measurement, and the small solver the calibration stands on."""

from __future__ import annotations

import pytest

from llamafit.bench.compare import compare, metric_label, within_tolerance
from llamafit.bench.lstsq import least_squares
from tests.fixtures.bench import conditions, run_of

GIB = 1024**3


def test_every_kind_of_run_gets_a_row_of_its_own() -> None:
    """The same configuration generates at three speeds; averaging them hides why."""
    runs = [
        run_of(kind="llama-bench-pp", pp_tps=77.0, estimated_pp_tps=74.0),
        run_of(kind="llama-bench-tg", gen_tps=13.0, estimated_gen_tps=12.9),
        run_of(kind="server-short", gen_tps=8.3, estimated_gen_tps=12.9),
        run_of(kind="server-1k", gen_tps=13.9, pp_tps=49.4, estimated_gen_tps=13.4),
    ]
    rows = compare(runs)
    assert [row.metric for row in rows] == [
        "prompt-bench",
        "generation-bench",
        "generation-short",
        "generation-1k",
        "prompt-1k",
    ]


def test_the_ratio_is_the_measurement_over_the_estimate_that_preceded_it() -> None:
    rows = compare([run_of(kind="llama-bench-tg", gen_tps=13.0, estimated_gen_tps=12.9)])
    assert rows[0].estimated == pytest.approx(12.9)
    assert rows[0].measured == pytest.approx(13.0)
    assert rows[0].ratio == pytest.approx(13.0 / 12.9)


def test_a_bad_prediction_is_shown_rather_than_replaced_by_the_measurement() -> None:
    """The easiest dishonest thing here would be to print only the right-hand column."""
    rows = compare([run_of(kind="llama-bench-tg", gen_tps=6.2, estimated_gen_tps=13.4)])
    assert rows[0].estimated == pytest.approx(13.4)
    assert rows[0].ratio == pytest.approx(6.2 / 13.4)
    assert within_tolerance(rows[0].ratio) is False


def test_a_measurement_with_no_estimate_has_no_ratio_rather_than_a_flattering_one() -> None:
    rows = compare([run_of(kind="llama-bench-tg", gen_tps=13.0)])
    assert rows[0].ratio is None
    assert within_tolerance(None) is None


def test_the_memory_prediction_is_checked_as_well_as_the_speed_one() -> None:
    runs = [run_of(kind="server-1k", gen_tps=13.9, peak_vram_bytes=7 * GIB)]
    rows = compare(runs, predicted_vram_bytes=6 * GIB)
    peak = rows[-1]
    assert peak.metric == "peak-vram"
    assert peak.unit == "bytes"
    assert peak.ratio == pytest.approx(7 / 6)


def test_a_run_with_no_comparable_figure_produces_no_row() -> None:
    assert compare([run_of(kind="server-toolcall", conditions_=conditions())]) == []


def test_the_tolerance_band_is_the_one_the_documentation_states() -> None:
    assert within_tolerance(0.8) is True
    assert within_tolerance(1.25) is True
    assert within_tolerance(0.79) is False
    assert within_tolerance(1.26) is False


def test_every_metric_has_a_name_a_person_can_read() -> None:
    for metric in (
        "generation-bench",
        "prompt-bench",
        "generation-short",
        "generation-1k",
        "prompt-1k",
        "peak-vram",
    ):
        assert metric_label(metric) != metric
    assert metric_label("something-new") == "something-new"


def test_the_solver_finds_the_line_through_two_points() -> None:
    solution = least_squares([[1.0, 0.0], [1.0, 2.0]], [3.0, 7.0])
    assert solution.values == pytest.approx((3.0, 2.0))
    assert solution.exactly_determined is True


def test_the_solver_refuses_two_columns_that_move_together() -> None:
    """Two measurements that vary the same way cannot separate two parameters."""
    solution = least_squares([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])
    assert solution.values is None
    assert solution.rank < solution.columns


def test_the_solver_refuses_fewer_rows_than_columns() -> None:
    solution = least_squares([[1.0, 1.0, 1.0]], [1.0])
    assert solution.values is None
    assert solution.rows == 1
    assert solution.columns == 3


def test_the_solver_refuses_a_column_nothing_varies() -> None:
    solution = least_squares([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [1.0, 2.0, 3.0])
    assert solution.values is None
    assert solution.rank == 1


def test_the_solver_reports_a_residual_when_the_points_disagree() -> None:
    solution = least_squares([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [0.0, 1.1, 1.9])
    assert solution.values is not None
    assert solution.residual > 0
    assert solution.exactly_determined is False


def test_the_solver_is_strict_about_shapes_that_are_a_programming_error() -> None:
    with pytest.raises(ValueError, match="one target per row"):
        least_squares([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one column"):
        least_squares([], [])
    with pytest.raises(ValueError, match="same number of columns"):
        least_squares([[1.0, 2.0], [1.0]], [1.0, 2.0])
