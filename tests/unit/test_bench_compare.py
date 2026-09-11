"""The estimate beside the measurement, and the small solver the calibration stands on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llamafit.bench.compare import compare, metric_label, within_tolerance
from llamafit.bench.lstsq import least_squares
from llamafit.bench.types import ComparisonRow
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


def test_every_row_names_the_conditions_both_of_its_figures_are_for() -> None:
    """The finding: ``tg128`` at 128 tokens of cache was divided by an estimate at 32,768.

    The quotient came out at 6.77 and the column called it a ratio. What it measured was
    the difference between a full context and an empty one, on a measurement the project's
    own calibration record agrees with to a tenth of a percent.
    """
    runs = [
        run_of(kind="llama-bench-pp", pp_tps=194.25, estimated_pp_tps=180.0),
        run_of(kind="llama-bench-tg", gen_tps=24.7, estimated_gen_tps=24.0),
        run_of(kind="server-1k", gen_tps=13.9, pp_tps=49.4, estimated_gen_tps=13.4),
    ]
    rows = {row.metric: row for row in compare(runs)}
    assert rows["prompt-bench"].context == 2048
    assert rows["prompt-bench"].micro_batch == 1024
    assert rows["generation-bench"].context == 128
    assert rows["generation-bench"].micro_batch == 1024
    assert rows["generation-1k"].context == 1056 + 64
    assert rows["prompt-1k"].micro_batch == 1024


def test_a_run_that_did_not_say_how_many_tokens_it_moved_gets_no_ratio() -> None:
    """An estimate is a statement about conditions; a guess at the conditions is not one."""
    silent = conditions(n_prompt=None, n_gen=None)
    rows = compare(
        [run_of(kind="llama-bench-tg", conditions_=silent, gen_tps=279.8, estimated_gen_tps=41.3)]
    )
    assert rows[0].measured == pytest.approx(279.8)
    assert rows[0].estimated == pytest.approx(41.3)
    assert rows[0].context is None
    assert rows[0].ratio is None


def test_a_ratio_cannot_be_carried_without_the_context_its_halves_were_taken_at() -> None:
    """The structural half of the fix: the row that shipped cannot be built any more."""
    with pytest.raises(ValidationError, match="must name the context"):
        ComparisonRow(metric="generation-bench", estimated=41.34, measured=279.76, ratio=6.77)
    allowed = ComparisonRow(
        metric="generation-bench", estimated=41.34, measured=279.76, ratio=6.77, context=128
    )
    assert allowed.context == 128


def test_the_memory_prediction_is_checked_as_well_as_the_speed_one() -> None:
    runs = [run_of(kind="server-1k", gen_tps=13.9, peak_vram_bytes=7 * GIB)]
    rows = compare(runs, predicted_vram_bytes=6 * GIB)
    peak = rows[-1]
    assert peak.metric == "peak-vram"
    assert peak.unit == "bytes"
    assert peak.ratio == pytest.approx(7 / 6)
    # The one row whose context is what the server allocated rather than what it filled:
    # the cache is allocated whole at load time, and the budget predicted that allocation.
    assert peak.context == 32768
    assert peak.context != runs[0].conditions.measured_context


def test_a_prediction_with_no_server_to_check_it_still_says_what_it_is_for() -> None:
    """`--no-server` measures no VRAM, so the budget figure stands alone -- with a context."""
    rows = compare(
        [run_of(kind="llama-bench-tg", gen_tps=279.9, estimated_gen_tps=274.0)],
        predicted_vram_bytes=6 * GIB,
        predicted_at_context=32768,
    )
    peak = rows[-1]
    assert peak.metric == "peak-vram"
    assert peak.measured is None
    assert peak.ratio is None
    assert peak.context == 32768


def test_the_server_own_context_beats_the_planned_one_for_the_memory_row() -> None:
    """llama.cpp clamps a context it cannot honour; the budget is then judged against the
    configuration that ran rather than the one that was asked for."""
    clamped = conditions(context=4096, n_prompt=1056, n_gen=64)
    rows = compare(
        [run_of(kind="server-1k", conditions_=clamped, gen_tps=13.9, peak_vram_bytes=7 * GIB)],
        predicted_vram_bytes=6 * GIB,
        predicted_at_context=32768,
    )
    assert rows[-1].context == 4096


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
