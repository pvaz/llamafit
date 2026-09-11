"""The board, budget, ladder and plan renderers, one piece at a time.

The commands are tested end to end elsewhere. These are the pieces that decide whether a
figure arrives with the thing that says how it was obtained, and each of them has a case
that only shows up at one console width, in one language, or for one verdict.
"""

from __future__ import annotations

from datetime import date

import pytest
from rich.console import Console

from llamafit.cli.render_board import (
    component_label,
    confidence_label,
    confidence_sentence,
    mode_label,
    pool_label,
    render_board,
    render_budget,
    render_excluded,
    render_fit,
    render_fit_excluded,
    render_measurements,
    render_notes,
    render_plan,
    render_speed,
    render_tiers,
    source_label,
    tier_state,
    verdict_label,
    verdict_sentence,
    verdict_style,
)
from llamafit.models.catalog import Measured
from llamafit.models.plan import ContextTier, Needs, SpeedEstimate
from llamafit.services.plan import plan_report
from llamafit.services.recommend import build_board, build_fit_board
from tests.fixtures.board import catalog, model_and_quant
from tests.fixtures.budget_hosts import machine, reference_host, unsized_card_host

GIB = 1024**3


def drawn(renderable: object, *, width: int = 100) -> str:
    """What a renderable actually puts on a console, as one flat string."""
    console = Console(width=width, no_color=True, highlight=False, record=True)
    console.print(renderable)
    return " ".join(console.export_text().split())


# --- the words a figure arrives with --------------------------------------------------


@pytest.mark.parametrize("verdict", ["comfortable", "fits", "tight", "too-tight", "does-not-fit"])
def test_every_verdict_has_a_short_word_a_colour_and_a_sentence(verdict: str) -> None:
    # "fits" and "tight" are already the word a column wants; the other three are not.
    assert verdict_label(verdict)
    assert verdict_style(verdict)
    assert verdict_sentence(verdict) != verdict


def test_the_paging_sentence_says_the_server_starts_anyway() -> None:
    # Section 8.4: the failure a person cannot diagnose is the one that looks like success.
    text = verdict_sentence("too-tight")
    assert "log looks healthy" in text
    assert "system memory" in text


@pytest.mark.parametrize("mode", ["gpu", "moe-offload", "hybrid", "cpu", "unsupported"])
def test_every_run_mode_has_a_word(mode: str) -> None:
    assert mode_label(mode)


def test_a_value_this_table_has_not_met_comes_back_in_english_rather_than_blank() -> None:
    assert mode_label("quantum") == "quantum"
    assert verdict_label("splendid") == "splendid"
    assert confidence_label("guessed") == "guessed"
    assert pool_label("tape") == "tape"
    assert component_label("flux-capacitor") == "flux-capacitor"


def test_a_budget_line_says_which_of_the_two_voices_it_speaks_in() -> None:
    assert source_label(True) != source_label(False)


@pytest.mark.parametrize("confidence", ["measured", "calibrated", "estimated", "unsupported"])
def test_every_confidence_has_a_word_and_a_sentence(confidence: str) -> None:
    assert confidence_label(confidence)
    assert confidence_sentence(confidence) != confidence


# --- the ladder -----------------------------------------------------------------------


def test_a_tier_reports_its_own_verdict_rather_than_a_guess_at_one() -> None:
    paging = ContextTier(tokens=32768, vram_required=8 * GIB, fits=False, verdict="too-tight")
    nothing = ContextTier(tokens=65536, vram_required=9 * GIB, fits=False, verdict="does-not-fit")
    roomy = ContextTier(tokens=16384, vram_required=4 * GIB, fits=True, verdict="comfortable")
    assert tier_state(paging) == verdict_label("too-tight")
    assert tier_state(nothing) == verdict_label("does-not-fit")
    assert tier_state(roomy) == verdict_label("comfortable")


def test_the_ladder_warns_once_when_any_rung_would_page() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host())
    text = drawn(render_tiers(report.placement))
    assert "Context tiers" in text
    assert verdict_label("too-tight") in text
    assert "log looks healthy" in text


# --- the budget -----------------------------------------------------------------------


def test_the_budget_shows_a_source_for_every_line_and_the_note_behind_it() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host())
    text = drawn(render_budget(report.placement.budget, title="Memory budget"))
    assert "Memory budget" in text
    assert source_label(True) in text
    assert source_label(False) in text
    assert "least certain line here" in text  # the compute buffer's own note


def test_the_notes_can_be_left_off_a_second_budget_for_the_same_model() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host())
    quiet = drawn(render_budget(report.placement.budget, notes=False))
    assert "least certain line here" not in quiet


# --- the speed breakdown --------------------------------------------------------------


def test_the_three_shares_of_a_token_are_shown_with_the_figure_they_add_up_to() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host())
    assert report.speed is not None
    text = drawn(render_speed(report.speed, context=32768))
    assert "A token's time" in text
    assert "reading the card" in text
    assert confidence_label("estimated") in text


def test_a_speed_with_no_breakdown_still_reports_the_two_figures() -> None:
    bare = SpeedEstimate(gen_tps=12.0, pp_tps=100.0, confidence="estimated")
    text = drawn(render_speed(bare, context=8192))
    assert "12.0" in text
    assert "A token's time" not in text
    assert "A prompt token's time" not in text


def test_the_prompt_terms_are_shown_beside_the_generation_ones() -> None:
    """Section 10.2's three terms, given the treatment section 10.1's three already had."""
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host())
    assert report.speed is not None
    text = drawn(render_speed(report.speed, context=32768))
    assert "A prompt token's time" in text
    assert "doing the arithmetic" in text
    assert "streaming the experts across the link" in text
    assert "reading the experts from system memory" in text


def test_the_link_term_carries_the_gap_it_is_known_to_have() -> None:
    """The constant covers two physical paths, and the reader is told beside the term."""
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host())
    assert report.speed is not None
    text = drawn(render_speed(report.speed, context=32768))
    assert "does not fit in system memory" in text
    assert "three times faster" in text


def test_a_generation_breakdown_alone_still_draws_its_table() -> None:
    """One trio may be there without the other, and the table for it is not conditional."""
    generation_only = SpeedEstimate(
        gen_tps=12.0,
        pp_tps=0.0,
        confidence="estimated",
        vram_seconds_per_token=0.05,
        ram_seconds_per_token=0.03,
        overhead_seconds_per_token=0.001,
    )
    text = drawn(render_speed(generation_only, context=8192))
    assert "A token's time" in text
    assert "A prompt token's time" not in text


def test_a_term_too_small_for_four_decimals_is_still_printed_as_a_number() -> None:
    """A breakdown that reads 0.0000 three times over looks like one and is not.

    A small dense model held on the card reads a prompt token in forty-six microseconds.
    Four decimals would round every term of that trio to zero, which is worse than showing
    no trio at all, because a reader checking the terms would find they add up to nothing.
    """
    fast = SpeedEstimate(
        gen_tps=279.5,
        pp_tps=21734.0,
        confidence="estimated",
        prompt_compute_seconds_per_token=1 / 21734.0,
    )
    text = drawn(render_speed(fast, context=8192))
    assert "0.0000460" in text


def test_a_measured_figure_carries_the_date_it_was_taken() -> None:
    measured = SpeedEstimate(
        gen_tps=13.9,
        pp_tps=49.0,
        confidence="measured",
        measured_on=date(2026, 9, 9),
        notes=("a note",),
    )
    text = drawn(render_speed(measured, context=40960))
    assert "2026-09-09" in text
    assert "a note" in text


# --- the recorded runs ----------------------------------------------------------------


def test_a_model_nobody_has_measured_shows_no_comparison_table() -> None:
    assert render_measurements(()) is None
    assert render_notes(()) is None


def test_the_recorded_runs_say_whose_machine_they_are_from() -> None:
    runs = [
        Measured(profile="somebody's bench", quant="Q4_K_M", gen_tps=10.0, date=date(2026, 1, 1)),
        Measured(profile="no figures at all", quant="Q4_K_M"),
    ]
    text = drawn(render_measurements(runs))
    assert "Recorded elsewhere" in text
    assert "not benchmarks of this one" in text


# --- the board ------------------------------------------------------------------------


def test_a_narrow_console_keeps_the_columns_a_row_cannot_be_read_without() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    narrow = drawn(render_board(board, console_width=60), width=60)
    wide = drawn(render_board(board, console_width=200), width=200)
    # The identity and the score survive any width; the run mode is bought by a wider one.
    for text in (narrow, wide):
        assert "qwen3-coder-next" in text
        assert "UD-Q4_K_XL" in text
    assert "Runs" in wide
    assert "Runs" not in narrow


def test_the_board_names_the_confidence_per_row_only_when_the_rows_disagree() -> None:
    # Drawn at the width it was laid out for, as the test above does. A board budgeted for
    # two hundred columns and printed into one hundred is squeezed by rich rather than by
    # `_board_columns`, and what a squeeze truncates first is not this test's subject.
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    text = drawn(render_board(board, console_width=200), width=200)
    # Every row is estimated today, so the label belongs under the table, not in it.
    assert "no figure here is a measurement" in text
    board.rows[0].candidate.speed = SpeedEstimate(
        gen_tps=1.0, pp_tps=1.0, confidence="measured", measured_on=date(2026, 9, 9)
    )
    mixed = drawn(render_board(board, console_width=200), width=200)
    assert confidence_label("measured") in mixed


def test_the_board_says_both_contexts_when_they_differ() -> None:
    general = build_board(catalog(), reference_host(), Needs(use_case="general"))
    text = drawn(render_board(general, console_width=200))
    assert "Sized for" in text and "scored for" in text
    coding = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    assert "Sized and scored" in drawn(render_board(coding, console_width=200))


def test_an_empty_exclusion_list_draws_nothing_at_all() -> None:
    assert render_excluded([]) is None
    assert render_fit_excluded([]) is None


def test_the_exclusions_carry_the_reason_beside_the_model() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    text = drawn(render_excluded(board.excluded), width=200)
    assert "Not ranked" in text
    assert "no coding capability" in text


def test_the_fit_listing_drops_its_optional_columns_on_a_narrow_console() -> None:
    board = build_fit_board(catalog(), reference_host())
    narrow = drawn(render_fit(board, console_width=60), width=60)
    assert "qwen3-coder-next" in narrow
    assert "Runs" not in narrow


def test_the_fit_listing_shows_the_models_it_could_not_place() -> None:
    tiny = machine(vram_total=None, ram_total=2 * GIB, ram_available=1 * GIB)
    board = build_fit_board(catalog(), tiny)
    text = drawn(render_fit_excluded(board.excluded), width=200)
    assert "Not placed" in text


# --- the plan -------------------------------------------------------------------------


def test_a_plan_for_a_file_nobody_has_says_where_a_download_would_put_it() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host(), local_files=[])
    text = drawn(render_plan(report), width=200)
    assert "not on this machine yet" in text
    assert "llama-server" in text


def test_a_plan_for_a_file_that_is_here_names_it_without_the_warning() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    bare = quant.files[0].rsplit("/", 1)[-1]
    report = plan_report(model, quant, reference_host(), local_files=[f"/models/{bare}"])
    text = drawn(render_plan(report), width=200)
    assert "not on this machine yet" not in text
    assert "/models/" in text


def test_a_target_that_a_shorter_context_would_reach_says_which_one() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host(), target_tps=24.0)
    assert report.target is not None
    text = drawn(render_plan(report), width=200)
    assert "tokens per second" in text


def test_a_score_expands_into_the_parts_the_weights_and_what_each_one_added() -> None:
    from llamafit.cli.render_board import render_score

    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    rows = [row for row in board.rows if row.candidate.score is not None]
    assert rows
    for row in rows:
        text = drawn(
            render_score(row.candidate, use_case="coding", requested_context=32768), width=200
        )
        for part in ("quality", "speed", "fit", "context"):
            assert part in text
        assert "Weight" in text and "Adds" in text
        # Every part says what it was measured against, not only what it scored.
        assert "from the curator" in text
        assert "this use case asks for" in text
        assert "tightest pool" in text
        assert "this request is scored against" in text


def test_a_candidate_with_no_score_has_nothing_to_expand() -> None:
    from llamafit.cli.render_board import render_score

    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    assert board.excluded
    assert (
        render_score(board.excluded[0].candidate, use_case="coding", requested_context=32768)
        is None
    )


def test_a_slow_prompt_says_what_it_cost_the_speed_score() -> None:
    from llamafit.cli.render_board import _speed_sentence

    slow = SpeedEstimate(gen_tps=30.0, pp_tps=20.0, confidence="estimated")
    quick = SpeedEstimate(gen_tps=30.0, pp_tps=900.0, confidence="estimated")
    assert "reads a prompt" in _speed_sentence(slow, "coding")
    assert "reads a prompt" not in _speed_sentence(quick, "coding")


# --- a card that is here and cannot be read -------------------------------------------


def test_the_board_says_its_speeds_are_cpu_only_even_where_the_runs_column_will_not_fit() -> None:
    """Eighty columns is where this finding was invisible: the ``Runs`` column is dropped.

    The caption is not an alternative to the column, it is the thing the column could not
    have said anyway -- that a card is here and none of these figures used it.
    """
    board = build_board(catalog(), unsized_card_host(), Needs())
    for width in (80, 200):
        text = drawn(render_board(board, console_width=width), width=width)
        assert "AMD Radeon RX 7900 XTX" in text
        assert "CPU-only speed" in text
        assert "llamafit doctor" in text


def test_the_fit_board_says_its_rows_were_sized_without_the_card() -> None:
    board = build_fit_board(catalog(), unsized_card_host())
    text = drawn(render_fit(board, console_width=200), width=200)
    assert "as if there were no card" in text


def test_a_board_for_a_machine_whose_card_was_read_prints_no_such_caption() -> None:
    board = build_board(catalog(), reference_host(), Needs())
    assert "CPU-only speed" not in drawn(render_board(board, console_width=200), width=200)


def test_a_board_for_a_machine_with_no_card_prints_no_such_caption() -> None:
    """A CPU-only machine is not apologised to for a card it does not have."""
    board = build_board(catalog(), machine(vram_total=None, ram_available=48 * GIB), Needs())
    assert "CPU-only speed" not in drawn(render_board(board, console_width=200), width=200)
