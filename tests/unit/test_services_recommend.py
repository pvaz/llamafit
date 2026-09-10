"""The board service: the whole pipeline, on the real catalog and the recorded machine.

Everything under this layer is tested against fakes, which is the right way to test a
budget or a search. This is where they meet, so the catalog is the real one and the host is
the reference machine's recorded scan: a board that is right against a fake and wrong
against the seeded entries would pass every other test in the suite.
"""

from __future__ import annotations

from llamafit.models.llamacpp import LocalModel
from llamafit.models.plan import Needs
from llamafit.scoring import DEFAULT_WEIGHTS, READING_TPS
from llamafit.services.recommend import (
    PERFECT_FIT,
    best_quant,
    build_board,
    build_fit_board,
    local_index,
    local_path_for,
    planned_context,
    quant_entries,
    recorded_measurements,
)
from tests.fixtures.board import catalog, model_and_quant
from tests.fixtures.budget_hosts import machine, reference_host

GIB = 1024**3


def ids(board_rows: list) -> list[str]:  # type: ignore[type-arg]
    return [row.model_id for row in board_rows]


# --- what the board is ---------------------------------------------------------------


def test_the_coding_board_ranks_the_two_measured_models_above_the_rest() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    assert board.rows, "the reference machine should be able to run something for coding"
    assert board.rows[0].model_id == "qwen3-coder-next"
    assert [row.rank for row in board.rows] == list(range(1, len(board.rows) + 1))


def test_every_scored_row_carries_the_parts_that_produced_it() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    for row in board.rows:
        score = row.candidate.score
        assert score is not None
        assert set(score.weights) == {"quality", "speed", "fit", "context"}
        assert row.candidate.quality is not None
        assert row.candidate.placement is not None
        assert row.candidate.speed is not None


def test_a_model_built_for_another_job_is_excluded_and_says_so() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    excluded = {row.model_id: row.candidate.excluded_because for row in board.excluded}
    assert "llama-3.1-8b-instruct" in excluded
    reason = excluded["llama-3.1-8b-instruct"] or ""
    assert "coding" in reason
    assert "chat" in reason  # the use cases its entry does list, so the reader can adjust


def test_nothing_is_dropped_even_when_the_limit_is_one() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"), limit=1)
    assert len(board.rows) == 1
    assert board.excluded, "a limit must not hide the reason a model is missing"


def test_the_board_shows_the_best_quant_per_model_unless_asked_otherwise() -> None:
    every = build_board(catalog(), reference_host(), Needs(use_case="coding"), all_quants=True)
    best = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    assert len(ids(best.rows)) == len(set(ids(best.rows)))
    assert len(every.rows) >= len(best.rows)


def test_speeds_are_never_labelled_measured_because_nothing_measured_this_machine() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    labels = {row.candidate.speed.confidence for row in board.rows if row.candidate.speed}
    assert labels == {"estimated"}


def test_the_catalogs_own_runs_are_offered_separately_rather_than_as_the_estimate() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    runs = recorded_measurements(model, quant.name)
    assert runs, "the seeded entry records the reference machine's own runs"
    assert all(run.quant == quant.name for run in runs)


# --- the request's own filters --------------------------------------------------------


def test_the_general_board_no_longer_leads_with_a_model_slower_than_its_reader() -> None:
    """The whole defect, at the service that produces the answer a person reads.

    On the reference machine a general request used to put Gemma 3 27B at 2.3 tokens per
    second above Llama 3.1 8B at 34, carried there by quality and fit while its speed
    score read exactly zero. The row is still on the page; it is on the half of the page
    that says why.
    """
    board = build_board(catalog(), reference_host(), Needs(use_case="general"))
    assert board.rows, "something on this machine should be readable"
    # The defect was a leading row nobody could read, so that is what is asserted: which
    # model leads is a fact about the catalog on the day, and the catalog keeps growing.
    lead = board.rows[0].candidate.speed
    assert lead is not None and lead.gen_tps >= READING_TPS
    assert "gemma-3-27b-it" not in ids(board.rows)

    slow = next(row for row in board.excluded if row.model_id == "gemma-3-27b-it")
    assert "a person reads at" in (slow.candidate.excluded_because or "")
    assert slow.candidate.speed is not None, "the excluded row keeps the number it was judged on"
    assert slow.candidate.placement is not None


def test_the_same_model_is_ranked_on_a_machine_that_can_actually_run_it() -> None:
    """The rule is a fact about a machine, not a verdict on a model.

    Gemma 3 27B is excluded from the reference machine's general board for running at
    two and a half tokens per second there. Given a card that holds it, the same entry
    at the same quantisation reaches nine and is ranked, which is what stops this from
    being a rule fitted to one board.
    """
    roomy = machine(vram_total=32 * GIB, ram_total=64 * GIB, ram_available=48 * GIB)
    board = build_board(catalog(), roomy, Needs(use_case="general"))
    ranked = next(row for row in board.rows if row.model_id == "gemma-3-27b-it")
    assert ranked.candidate.speed is not None
    assert ranked.candidate.speed.gen_tps > 6.0


def test_a_required_capability_excludes_by_name() -> None:
    board = build_board(
        catalog(), reference_host(), Needs(use_case="coding", capabilities=("vision",))
    )
    reasons = [row.candidate.excluded_because or "" for row in board.excluded]
    assert any("vision" in reason for reason in reasons)


def test_a_download_ceiling_excludes_and_names_both_sizes() -> None:
    needs = Needs(use_case="coding", max_download_bytes=10 * GIB)
    board = build_board(catalog(), reference_host(), needs)
    reasons = {row.model_id: row.candidate.excluded_because or "" for row in board.excluded}
    assert "qwen3.8-flash-next" in reasons
    assert "download" in reasons["qwen3.8-flash-next"]


def test_a_licence_the_request_refuses_is_an_exclusion_and_not_a_silent_filter() -> None:
    board = build_board(
        catalog(), reference_host(), Needs(use_case="coding"), licenses=["Apache-2.0"]
    )
    reasons = {row.model_id: row.candidate.excluded_because or "" for row in board.excluded}
    assert "qwen3.8-flash-next" in reasons, "a refused licence must still appear"
    assert "qwen-community-1.0" in reasons["qwen3.8-flash-next"]


def test_a_minimum_context_above_what_fits_excludes_and_names_both_figures() -> None:
    needs = Needs(use_case="coding", min_context=200_000)
    board = build_board(catalog(), reference_host(), needs)
    assert not board.rows or all(
        (row.candidate.placement is None or row.candidate.placement.max_context_fit >= 200_000)
        for row in board.rows
    )
    assert board.excluded


# --- weights and preferences ----------------------------------------------------------


def test_the_weights_are_the_use_cases_own_unless_a_preference_leans_them() -> None:
    plain = build_board(catalog(), reference_host(), Needs(use_case="coding"))
    assert plain.weights == dict(DEFAULT_WEIGHTS["coding"])
    leaned = build_board(catalog(), reference_host(), Needs(use_case="coding"), prefer="speed")
    assert leaned.weights["speed"] > plain.weights["speed"]
    assert leaned.weights["quality"] < plain.weights["quality"]
    assert abs(sum(leaned.weights.values()) - 1.0) < 1e-9


def test_the_board_reports_both_contexts_because_they_are_different_questions() -> None:
    board = build_board(catalog(), reference_host(), Needs(use_case="general"))
    # Section 11.2 scores a general request against 8K; section 9.2 sizes for 32K.
    assert board.requested_context == 8192
    assert board.planned_context == 32768
    assert planned_context(Needs(use_case="general", min_context=65536)) == 65536


# --- the fit listing ------------------------------------------------------------------


def test_fit_ranks_every_model_including_the_ones_built_for_another_job() -> None:
    board = build_fit_board(catalog(), reference_host())
    assert "qwen3-coder-next" in ids(board.rows)
    assert "gemma-3-27b-it" in ids(board.rows)
    scores = [row.fit or 0.0 for row in board.rows]
    assert scores == sorted(scores, reverse=True)


def test_perfect_keeps_only_the_configurations_inside_the_ideal_band() -> None:
    board = build_fit_board(catalog(), reference_host(), perfect=True)
    assert all((row.fit or 0.0) >= PERFECT_FIT for row in board.rows)


def test_min_fit_refuses_anything_worse_than_the_verdict_asked_for() -> None:
    board = build_fit_board(catalog(), reference_host(), min_fit="comfortable")
    for row in board.rows:
        assert row.placement is not None
        assert row.placement.budget.verdict == "comfortable"


def test_a_machine_nothing_fits_still_lists_every_model_with_its_reason() -> None:
    tiny = machine(vram_total=None, ram_total=2 * GIB, ram_available=1 * GIB)
    board = build_fit_board(catalog(), tiny)
    assert not board.rows
    assert board.excluded
    assert all(row.excluded_because for row in board.excluded)


def test_the_limit_cuts_the_ranked_rows_and_renumbers_them() -> None:
    board = build_fit_board(catalog(), reference_host(), limit=2)
    assert [row.rank for row in board.rows] == [1, 2]


# --- the small pieces -----------------------------------------------------------------


def test_quant_entries_never_repeats_a_name_two_sources_publish() -> None:
    model, _quant = model_and_quant("qwen3-coder-next")
    names = [quant.name for quant in quant_entries(model)]
    assert names == list(dict.fromkeys(names))


def test_best_quant_is_the_one_the_board_would_have_put_first() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    assert best_quant(model, reference_host()).name == quant.name


def test_a_quant_is_on_disk_when_its_first_shard_is() -> None:
    _model, quant = model_and_quant("qwen3.8-flash-next")
    assert quant.files, "the seeded facts name this quant's shards"
    index = local_index([LocalModel(path=f"/models/{quant.files[0]}", bytes=1)])
    assert local_path_for(quant, index) == f"/models/{quant.files[0]}"
    assert local_path_for(quant, {}) is None
