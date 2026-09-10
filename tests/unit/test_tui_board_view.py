"""Which columns fit, which rows show, and in what order -- without a terminal.

These are the decisions the board screen makes that have nothing to do with Textual, so
they are tested without one: at a hundred widths, against a real board built from the
bundled catalog and the recorded reference machine.

Nothing here re-tests a figure. The budget, the estimate and the score are tested
underneath; what is checked is that a column sort does not renumber a ranking, that a
speed never loses the word saying what kind of number it is, and that a row nobody could
size comes back as the word for unknown rather than as an empty cell somebody would read
as a zero.
"""

from __future__ import annotations

import pytest

from llamafit.cli.render_board import confidence_label, verdict_label
from llamafit.models.plan import Candidate, SpeedEstimate
from llamafit.services.recommend import BoardRow, build_board
from llamafit.tui import board_view
from llamafit.units import format_bytes
from tests.fixtures.board import catalog
from tests.fixtures.budget_hosts import reference_host

WIDTHS = range(20, 240, 3)


def a_board() -> object:
    """The bundled catalog ranked for a general request on the reference machine."""
    from llamafit.models.plan import Needs

    return build_board(catalog(), reference_host(), Needs())


def rows() -> list[BoardRow]:
    board = a_board()
    return list(board.rows)  # type: ignore[attr-defined]


# --- which columns fit -----------------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS)
def test_the_four_columns_that_identify_a_row_are_never_dropped(width: int) -> None:
    for uniform in (True, False):
        chosen = board_view.columns_for_width(width, uniform_confidence=uniform)
        assert chosen[: len(board_view.REQUIRED)] == board_view.REQUIRED


@pytest.mark.parametrize("width", WIDTHS)
def test_a_wider_terminal_never_shows_fewer_columns(width: int) -> None:
    narrow = board_view.columns_for_width(width)
    wide = board_view.columns_for_width(width + 1)
    assert set(narrow) <= set(wide)


@pytest.mark.parametrize("width", WIDTHS)
def test_a_speed_is_never_shown_without_its_label_when_the_rows_disagree(width: int) -> None:
    # Two rows carrying different labels make no single sentence true of the table, so the
    # word has to be beside the figure or the figure must not be there at all.
    chosen = board_view.columns_for_width(width, uniform_confidence=False)
    assert ("gen" in chosen) == ("confidence" in chosen)


def test_when_every_row_agrees_the_verdict_is_worth_more_than_a_repeated_label() -> None:
    # The band above the table already says it once, in a sentence, so the width buys the
    # column that actually varies between rows.
    at_eighty = board_view.columns_for_width(80, uniform_confidence=True)
    assert "gen" in at_eighty
    assert "verdict" in at_eighty
    assert "confidence" not in at_eighty


def test_a_wide_enough_terminal_shows_the_label_beside_the_figure_anyway() -> None:
    chosen = board_view.columns_for_width(200, uniform_confidence=True)
    assert "confidence" in chosen


def test_every_column_has_a_heading_and_a_width() -> None:
    for column in board_view.columns_for_width(400, uniform_confidence=False):
        assert board_view.heading(column)
        assert board_view.WIDTHS[column] > 0


def test_one_confidence_is_true_of_a_board_nothing_could_be_estimated_for() -> None:
    unsized = BoardRow(
        rank=1,
        model_id="x",
        name="X",
        quant="Q4_K_M",
        candidate=Candidate(model_id="x", quant="Q4_K_M"),
    )
    assert board_view.one_confidence([unsized])
    assert board_view.one_confidence([])


def test_rows_that_disagree_about_their_labels_are_not_uniform() -> None:
    estimated = _row("a", confidence="estimated")
    measured = _row("b", confidence="measured")
    assert not board_view.one_confidence([estimated, measured])


def _row(model_id: str, *, confidence: str = "estimated") -> BoardRow:
    return BoardRow(
        rank=1,
        model_id=model_id,
        name=model_id,
        quant="Q4_K_M",
        candidate=Candidate(
            model_id=model_id,
            quant="Q4_K_M",
            speed=SpeedEstimate(gen_tps=10.0, pp_tps=100.0, confidence=confidence),  # type: ignore[arg-type]
        ),
    )


# --- what a cell says ------------------------------------------------------------------


def test_a_row_nobody_could_size_says_unknown_rather_than_leaving_a_blank() -> None:
    unsized = BoardRow(
        rank=None,
        model_id="x",
        name="X",
        quant="Q4_K_M",
        candidate=Candidate(model_id="x", quant="Q4_K_M", excluded_because="no header was read"),
    )
    for column in ("rank", "score", "gen", "confidence", "verdict", "mode", "vram", "ram"):
        assert str(board_view.cell(unsized, column)) == format_bytes(None)  # type: ignore[arg-type]


def test_a_verdict_carries_its_word_as_well_as_its_colour() -> None:
    row = next(r for r in rows() if r.candidate.placement is not None)
    verdict = row.candidate.placement.budget.verdict  # type: ignore[union-attr]
    cell = board_view.cell(row, "verdict")
    # The colour is a style; the word is the cell. A screen read without colour reads the
    # same, which is the whole reason the word is there.
    assert str(cell) == verdict_label(verdict)
    assert cell.style


def test_a_speed_cell_and_its_label_come_from_the_same_estimate() -> None:
    row = next(r for r in rows() if r.candidate.speed is not None)
    assert str(board_view.cell(row, "confidence")) == confidence_label(
        row.candidate.speed.confidence  # type: ignore[union-attr]
    )


def test_a_model_that_is_not_on_disk_says_no_rather_than_nothing() -> None:
    row = rows()[0]
    assert str(board_view.cell(row, "have"))


# --- order and filters -----------------------------------------------------------------


def test_sorting_by_speed_reorders_the_screen_and_not_the_ranking() -> None:
    ranked = rows()
    by_speed = board_view.sorted_rows(ranked, "speed")
    assert {row.model_id for row in by_speed} == {row.model_id for row in ranked}
    assert len(by_speed) == len(ranked)
    # The rank travels with the row: a table that renumbered itself would be claiming the
    # column sort was a second opinion about which model is best.
    for row in by_speed:
        original = next(r for r in ranked if r.model_id == row.model_id)
        assert row.rank == original.rank
    speeds = [r.candidate.speed.gen_tps for r in by_speed if r.candidate.speed is not None]
    assert speeds == sorted(speeds, reverse=True)


def test_every_sort_and_every_filter_keeps_the_table_a_subset_of_the_board() -> None:
    ranked = rows()
    whole = {(row.model_id, row.quant) for row in ranked}
    for sort in board_view.SORTS:
        for fit in board_view.FILTERS:
            shown = board_view.visible_rows(ranked, sort=sort, fit=fit)
            assert {(row.model_id, row.quant) for row in shown} <= whole


def test_a_download_size_nobody_has_filled_in_sorts_last_not_first() -> None:
    small = _row("small")
    small.download_bytes = 1
    unknown = _row("unknown")
    ordered = board_view.sorted_rows([unknown, small], "size")
    assert [row.model_id for row in ordered] == ["small", "unknown"]


def test_the_fit_filter_keeps_only_the_verdicts_the_service_calls_running() -> None:
    from llamafit.services.recommend import MIN_FIT_VERDICTS

    ranked = rows()
    runs = board_view.visible_rows(ranked, fit="runs")
    for row in runs:
        assert row.candidate.placement is not None
        assert row.candidate.placement.budget.verdict in MIN_FIT_VERDICTS


def test_search_matches_the_id_the_name_and_the_quantisation() -> None:
    ranked = rows()
    wanted = ranked[0]
    assert wanted in board_view.visible_rows(ranked, search=wanted.model_id)
    assert wanted in board_view.visible_rows(ranked, search=wanted.name.lower())
    assert board_view.visible_rows(ranked, search="no such model") == []


def test_installed_only_hides_everything_when_nothing_is_on_disk() -> None:
    assert board_view.visible_rows(rows(), installed_only=True) == []


def test_the_state_line_says_how_many_of_how_many_and_why() -> None:
    line = board_view.state_line(2, 7, sort="speed", fit="runs", installed_only=False)
    assert "2" in line and "7" in line
    assert board_view.sort_label("speed") in line
    assert board_view.filter_label("runs") in line


def test_the_state_line_says_when_a_filter_is_hiding_models_a_reader_may_want() -> None:
    line = board_view.state_line(0, 7, sort="score", fit="all", installed_only=True)
    assert "0" in line and "7" in line
    assert line != board_view.state_line(0, 7, sort="score", fit="all", installed_only=False)


@pytest.mark.parametrize("sort", board_view.SORTS)
def test_every_sort_has_a_word(sort: str) -> None:
    assert board_view.sort_label(sort)  # type: ignore[arg-type]


@pytest.mark.parametrize("fit", board_view.FILTERS)
def test_every_filter_has_a_phrase(fit: str) -> None:
    assert board_view.filter_label(fit)  # type: ignore[arg-type]


def test_a_candidate_nobody_could_place_is_not_one_that_runs() -> None:
    # "Does it run" is a question about a placement, and a candidate with none has no
    # answer to it; counting it as running would put a model on a filtered list that the
    # filter exists to keep off.
    unplaced = BoardRow(
        rank=None,
        model_id="x",
        name="X",
        quant="Q4_K_M",
        candidate=Candidate(model_id="x", quant="Q4_K_M", excluded_because="no header was read"),
    )
    assert board_view.visible_rows([unplaced], fit="runs") == []
    assert board_view.visible_rows([unplaced], fit="roomy") == []
    assert board_view.visible_rows([unplaced], fit="all") == [unplaced]
