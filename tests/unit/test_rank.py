"""Ranking: the ordered board, and the candidates that did not make it onto it.

The board is built from the five seeded models with their real quant names and sizes,
placed on the reference machine, and with the measured speeds for the two models that
were measured there on 2026-09-09.
"""

from __future__ import annotations

import pytest

from llamafit.catalog.loader import load_catalog
from llamafit.models.catalog import Catalog, CatalogModel, Quant
from llamafit.models.plan import (
    Candidate,
    Needs,
    Placement,
    ScoreBreakdown,
    SpeedEstimate,
)
from llamafit.scoring import evaluate, evaluate_and_rank, rank
from llamafit.scoring.weights import PARTS
from tests.fixtures.scoring import BOARD, placement, speed

Entry = tuple[CatalogModel, Quant, Placement | None, SpeedEstimate | None]

CODING = Needs(use_case="coding", capabilities=("coding", "tools"), min_context=32768)


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    loaded, problems = load_catalog(custom_path=None)
    assert problems == []
    return loaded


def entries(catalog: Catalog) -> list[Entry]:
    """The whole seeded catalog, placed and estimated on the reference machine."""
    built: list[Entry] = []
    for model_id, (where, how_fast) in BOARD.items():
        model = catalog.by_id[model_id]
        built.append((model, model.sources[0].quants[0], where, how_fast))
    return built


def one(catalog: Catalog, model_id: str) -> Entry:
    model = catalog.by_id[model_id]
    where, how_fast = BOARD[model_id]
    return model, model.sources[0].quants[0], where, how_fast


def evaluate_one(catalog: Catalog, model_id: str, needs: Needs) -> Candidate:
    model, quant, where, how_fast = one(catalog, model_id)
    return evaluate(model, quant, needs, where, how_fast)


def ids(board: list[Candidate]) -> list[str]:
    return [candidate.model_id for candidate in board]


# --- the board ---------------------------------------------------------------------


def test_a_coding_request_on_the_reference_machine_puts_the_coding_model_first(
    catalog: Catalog,
) -> None:
    board = evaluate_and_rank(entries(catalog), CODING)
    assert ids(board)[0] == "qwen3-coder-next"
    assert ids(board)[:3] == ["qwen3-coder-next", "qwen3.8-flash-next", "qwen3-0.6b"]


def test_the_winner_is_the_model_the_measurements_favour(catalog: Catalog) -> None:
    # Qwen3-Coder-Next is measured at 24.7 t/s against Qwen3.8-Flash-Next's 13.9, and the
    # two are a point apart on quality, so speed is what separates them.
    board = evaluate_and_rank(entries(catalog), CODING)
    winner, second = board[0], board[1]
    assert winner.score is not None and second.score is not None
    assert winner.score.speed > second.score.speed
    assert winner.score.quality == pytest.approx(second.score.quality + 1.0)
    assert winner.speed is not None and winner.speed.confidence == "measured"


def test_the_fixture_speeds_are_the_catalogs_own_measurements(catalog: Catalog) -> None:
    for model_id in ("qwen3-coder-next", "qwen3.8-flash-next"):
        _, _, _, how_fast = one(catalog, model_id)
        assert how_fast is not None
        measured = catalog.by_id[model_id].measured
        assert how_fast.gen_tps in {m.gen_tps for m in measured}
        assert how_fast.pp_tps in {m.pp_tps for m in measured}


def test_the_same_board_reorders_when_the_request_changes(catalog: Catalog) -> None:
    # Two models offer themselves for chat, and the faster of them wins a use case that
    # weights speed at 0.40. The request is what decides, not a fixed order of merit.
    chat = evaluate_and_rank(entries(catalog), Needs(use_case="chat"))
    assert ids(chat)[0] == "llama-3.1-8b-instruct"
    coding = evaluate_and_rank(entries(catalog), Needs(use_case="coding"))
    assert ids(coding)[0] == "qwen3-coder-next"


def test_only_the_models_that_offer_themselves_for_the_job_compete(catalog: Catalog) -> None:
    coding = evaluate_and_rank(entries(catalog), Needs(use_case="coding"))
    ranked = [c.model_id for c in coding if c.score is not None]
    assert ranked == ["qwen3-coder-next", "qwen3.8-flash-next", "qwen3-0.6b"]
    # Llama 3.1 8B used to place second here on fit and a capped speed score, with a
    # catalog entry that never claimed to code.
    assert "llama-3.1-8b-instruct" not in ranked


MACHINE_ONLY = {"coding": {"quality": 0.0, "speed": 0.0, "fit": 1.0, "context": 0.0}}
"""Weights that ask nothing but "how well does this use the machine"."""


def test_overriding_the_weights_changes_the_order(catalog: Catalog) -> None:
    open_request = Needs(use_case="coding")
    assert ids(evaluate_and_rank(entries(catalog), open_request))[0] == "qwen3-coder-next"
    # A user who cares about nothing but how much of the machine a model uses is told what
    # that asks for, rather than being quietly given the project's opinion instead. On
    # 115 GB of memory it asks for the model that holds 82 GB of it, which is not the one
    # the balanced weights choose.
    machine_first = evaluate_and_rank(entries(catalog), open_request, weight_overrides=MACHINE_ONLY)
    assert ids(machine_first)[:3] == [
        "qwen3.8-flash-next",
        "qwen3-coder-next",
        "qwen3-0.6b",
    ]


def test_the_machine_question_alone_puts_the_smallest_model_last(catalog: Catalog) -> None:
    """The defect section 11.3 was written to prevent, guarded where it showed.

    Weighting fit alone, a 0.6B on a machine with 115 GB of memory used to come first: its
    0.6 GB of weights left the card ninety percent full of cache and buffers, and the
    score read the card. It comes last now, and scores nothing, because the machine it is
    being asked to fill is a hundred and eighty times its size.
    """
    machine_only = evaluate_and_rank(
        entries(catalog), Needs(use_case="coding"), weight_overrides=MACHINE_ONLY
    )
    scored = [candidate for candidate in machine_only if candidate.score is not None]
    assert scored[-1].model_id == "qwen3-0.6b"
    assert scored[-1].score is not None
    assert scored[-1].score.fit == 0.0


# --- the parts stay visible --------------------------------------------------------


def test_every_scored_candidate_carries_its_four_parts_and_the_weights(
    catalog: Catalog,
) -> None:
    for candidate in evaluate_and_rank(entries(catalog), CODING):
        if candidate.score is None:
            continue
        assert set(candidate.score.weights) == set(PARTS)
        assert candidate.quality is not None
        assert candidate.placement is not None
        assert candidate.speed is not None


def test_the_total_is_the_weighted_sum_of_the_parts_it_shows(catalog: Catalog) -> None:
    for candidate in evaluate_and_rank(entries(catalog), CODING):
        score = candidate.score
        if score is None:
            continue
        parts = {
            "quality": score.quality,
            "speed": score.speed,
            "fit": score.fit,
            "context": score.context,
        }
        expected = sum(score.weights[part] * parts[part] for part in PARTS)
        assert score.total == pytest.approx(expected)


def test_the_quality_breakdown_expands_into_the_catalogs_own_numbers(
    catalog: Catalog,
) -> None:
    candidate = evaluate_one(catalog, "qwen3-coder-next", CODING)
    assert candidate.quality is not None
    assert candidate.quality.baseline == 85.0
    assert candidate.quality.quant_penalty == 3.0
    assert candidate.quality.alignment_bonus == 5.0
    assert candidate.quality.quality == 87.0


# --- exclusions --------------------------------------------------------------------


def test_a_model_that_does_not_offer_itself_for_the_job_is_kept_with_its_reason(
    catalog: Catalog,
) -> None:
    board = evaluate_and_rank(entries(catalog), CODING)
    excluded = {c.model_id: c for c in board if c.excluded_because}
    assert set(excluded) == {"gemma-3-27b-it", "llama-3.1-8b-instruct"}
    reason = excluded["llama-3.1-8b-instruct"].excluded_because or ""
    assert "not a coding model" in reason
    # It names what the entry does say, so the reader can ask for one of those instead —
    # or correct the entry, which is a one-line change to a curated file.
    assert "general, chat, reasoning" in reason


def test_a_model_without_a_required_capability_is_kept_with_its_reason(
    catalog: Catalog,
) -> None:
    # Qwen3-Coder-Next is a coding model, so the use case lets it through; it cannot see.
    needs = Needs(use_case="coding", capabilities=("vision",))
    candidate = evaluate_one(catalog, "qwen3-coder-next", needs)
    reason = candidate.excluded_because or ""
    assert "no vision capability" in reason
    # The reason has to say what to change about the request, not merely that it failed.
    assert "drop it from the request" in reason


def test_an_excluded_candidate_keeps_what_was_learned_about_it(catalog: Catalog) -> None:
    board = evaluate_and_rank(entries(catalog), CODING)
    gemma = next(c for c in board if c.model_id == "gemma-3-27b-it")
    assert gemma.score is None
    assert gemma.quality is None
    assert gemma.placement is not None
    assert gemma.speed is not None
    assert gemma.quant == "Q4_K_M"


def test_nothing_is_dropped_from_the_board(catalog: Catalog) -> None:
    board = evaluate_and_rank(entries(catalog), CODING)
    assert len(board) == len(BOARD)


def test_a_context_below_the_minimum_is_excluded_with_both_numbers(catalog: Catalog) -> None:
    demanding = Needs(use_case="multimodal", min_context=131072)
    candidate = evaluate_one(catalog, "gemma-3-27b-it", demanding)
    reason = candidate.excluded_because or ""
    assert "16,384" in reason
    assert "131,072" in reason


def test_a_download_over_the_ceiling_is_excluded_with_both_sizes(catalog: Catalog) -> None:
    frugal = Needs(use_case="coding", max_download_bytes=20 * 1000**3)
    candidate = evaluate_one(catalog, "qwen3-coder-next", frugal)
    reason = candidate.excluded_because or ""
    assert "46.2 GiB" in reason
    assert "18.6 GiB" in reason


def test_a_download_size_nobody_knows_yet_cannot_exclude_anything(catalog: Catalog) -> None:
    model, _, where, how_fast = one(catalog, "qwen3-coder-next")
    unrefreshed = Quant(name="UD-Q4_K_XL")
    frugal = Needs(use_case="coding", max_download_bytes=1000)
    candidate = evaluate(model, unrefreshed, frugal, where, how_fast)
    assert candidate.excluded_because is None


def test_an_unknown_quantisation_is_excluded_rather_than_scored_as_free(
    catalog: Catalog,
) -> None:
    model, _, where, how_fast = one(catalog, "qwen3-0.6b")
    candidate = evaluate(model, Quant(name="banana"), Needs(), where, how_fast)
    assert "banana" in (candidate.excluded_because or "")


def test_a_model_with_nowhere_to_run_is_excluded_and_says_so(catalog: Catalog) -> None:
    model, quant, _, how_fast = one(catalog, "qwen3.8-flash-next")
    candidate = evaluate(model, quant, Needs(use_case="coding"), None, how_fast)
    assert "nowhere to put it" in (candidate.excluded_because or "")
    assert candidate.placement is None


def test_a_mode_llama_cpp_does_not_support_is_excluded(catalog: Catalog) -> None:
    model, quant, _, how_fast = one(catalog, "qwen3.8-flash-next")
    nowhere = placement(mode="unsupported", vram_required=0)
    candidate = evaluate(model, quant, Needs(use_case="coding"), nowhere, how_fast)
    assert "no run mode" in (candidate.excluded_because or "")


def test_a_budget_that_does_not_fit_is_excluded(catalog: Catalog) -> None:
    model, quant, _, how_fast = one(catalog, "qwen3.8-flash-next")
    too_big = placement(vram_required=int(9e9), verdict="does-not-fit")
    candidate = evaluate(model, quant, Needs(use_case="coding"), too_big, how_fast)
    assert "more memory than this machine has" in (candidate.excluded_because or "")


def test_a_candidate_with_no_speed_estimate_is_excluded(catalog: Catalog) -> None:
    model, quant, where, _ = one(catalog, "qwen3-0.6b")
    candidate = evaluate(model, quant, Needs(), where, None)
    assert "no speed estimate" in (candidate.excluded_because or "")


def test_a_model_slower_than_its_reader_is_excluded_and_the_number_is_in_the_reason(
    catalog: Catalog,
) -> None:
    """The defect this change exists to settle, at the model it was found on.

    Gemma 3 27B runs at 4.2 tokens per second on the reference machine's recorded
    placement, under the six a person reads at, and under both agents' earlier fixes it
    was still ranked for a general request: the speed score was zero, and quality, fit and
    context carried the remaining three quarters of the weight.
    """
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="general"))
    reason = candidate.excluded_because or ""
    assert "4.2 tokens per second" in reason
    assert "below the 6 a person reads at" in reason
    assert candidate.score is None


def test_the_excluded_model_keeps_the_placement_and_the_speed_it_was_judged_on(
    catalog: Catalog,
) -> None:
    """Nothing disappears: the row still says where it would run and how fast."""
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="chat"))
    assert candidate.placement is not None
    assert candidate.speed is not None
    assert candidate.speed.gen_tps == 4.2


def test_a_model_that_keeps_up_is_still_ranked_however_slow_it_looks(
    catalog: Catalog,
) -> None:
    """The rule is about the floor, not about being slower than the rest of the board."""
    candidate = evaluate_one(catalog, "qwen3.8-flash-next", Needs(use_case="multimodal"))
    assert candidate.excluded_because is None
    assert candidate.speed is not None and candidate.speed.gen_tps == 13.9


def test_an_embedding_request_never_excludes_a_model_for_generating_slowly(
    catalog: Catalog,
) -> None:
    """Nobody reads an embedding, so there is no reader for a model to fall behind.

    Gemma is not an embedding model and is excluded for that instead, which is the point:
    the reason a reader is given is about the catalog entry, never about a generation rate
    nothing in an embedding run produces.
    """
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="embedding"))
    assert "not a embedding model" in (candidate.excluded_because or "")


def test_being_too_slow_is_reported_after_everything_the_request_controls(
    catalog: Catalog,
) -> None:
    """A model that is both wrong for the job and too slow is told about the job.

    The reading floor is a fact about this machine and this model together. What the
    request asked for is the part the reader can change outright, so it is what they are
    told first.
    """
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="coding"))
    assert "not a coding model" in (candidate.excluded_because or "")


def test_the_slow_model_is_not_dropped_from_the_board_it_is_moved_to_the_end(
    catalog: Catalog,
) -> None:
    board = evaluate_and_rank(entries(catalog), Needs(use_case="general"))
    assert "gemma-3-27b-it" in ids(board)
    assert ids(board)[-1] == "gemma-3-27b-it"
    # Not which model leads: that is a fact about the catalog and moves when a curator
    # corrects an entry. What this test is named for is that the slow one is last and
    # still there, so the claim to make about the leader is that it is not the slow one.
    assert ids(board)[0] != "gemma-3-27b-it"


def test_a_request_with_nobody_waiting_ranks_the_model_it_would_have_excluded(
    catalog: Catalog,
) -> None:
    """``--min-tps 0`` is the batch case: no reader, so no reader to fall behind."""
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="general", min_tps=0))
    assert candidate.excluded_because is None
    assert candidate.score is not None


def test_nobody_waiting_scores_the_slow_model_on_the_straight_line_to_the_origin(
    catalog: Catalog,
) -> None:
    """With no floor the ramp is the one an embedding gets, and it separates slow from slower.

    The floor is what sends everything below it to zero. Take it away and 4.2 tokens per
    second against a general target of 25 is a sixth of the way there, which is the honest
    thing to say to somebody who is not sitting in front of it.
    """
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="general", min_tps=0))
    assert candidate.score is not None
    assert candidate.score.speed == pytest.approx(100.0 * 4.2 / 25.0)


def test_a_request_may_raise_the_bar_above_the_reading_floor(catalog: Catalog) -> None:
    """A reader who skims is entitled to disagree with six, upwards as well as downwards."""
    kept = evaluate_one(catalog, "qwen3.8-flash-next", Needs(use_case="multimodal"))
    assert kept.excluded_because is None
    raised = evaluate_one(catalog, "qwen3.8-flash-next", Needs(use_case="multimodal", min_tps=20))
    assert "13.9 tokens per second" in (raised.excluded_because or "")


def test_the_reason_names_the_figure_the_request_set_and_the_flag_that_set_it(
    catalog: Catalog,
) -> None:
    """A reader who moved a boundary meets the boundary they moved, not the default one."""
    candidate = evaluate_one(catalog, "gemma-3-27b-it", Needs(use_case="general", min_tps=12))
    reason = candidate.excluded_because or ""
    assert "below the 12 this request asks for" in reason
    assert "--min-tps" in reason
    assert "a person reads at" not in reason


def test_the_exclusion_and_the_score_are_decided_on_one_figure(catalog: Catalog) -> None:
    """A board that excluded on one number and scored on another would disagree with itself.

    Llama 3.1 8B runs at 33.9 here and a general request targets 25, so a floor of 30 keeps
    it and leaves its speed score at the rail. Nothing survives such a floor below the
    target, which is what makes the two readings impossible to tell apart from the outside —
    and exactly why they are taken from one call rather than two.
    """
    ranked = evaluate_one(catalog, "llama-3.1-8b-instruct", Needs(use_case="general", min_tps=30))
    assert ranked.excluded_because is None
    assert ranked.score is not None and ranked.score.speed == 100.0


def test_the_default_board_is_exactly_what_it_was_before_the_flag_existed(
    catalog: Catalog,
) -> None:
    """A request that says nothing is judged the way it was judged yesterday."""
    silent = evaluate_and_rank(entries(catalog), Needs(use_case="general"))
    spelled = evaluate_and_rank(entries(catalog), Needs(use_case="general", min_tps=6))
    assert ids(silent) == ids(spelled)
    assert [c.score is None for c in silent] == [c.score is None for c in spelled]


def test_what_the_request_asked_for_is_reported_before_what_the_machine_can_do(
    catalog: Catalog,
) -> None:
    # This candidate fails four ways at once. The job it does not offer itself for is the
    # broadest statement of why it is not here, so it is the one the reader is told.
    model, quant, _, _ = one(catalog, "gemma-3-27b-it")
    demanding = Needs(use_case="coding", capabilities=("coding",), min_context=131072)
    candidate = evaluate(model, quant, demanding, None, None)
    assert "not a coding model" in (candidate.excluded_because or "")


# --- ordering ----------------------------------------------------------------------


def scored(model_id: str, total: float, quality: float, gen_tps: float = 10.0) -> Candidate:
    return Candidate(
        model_id=model_id,
        quant="Q4_K_M",
        speed=speed(gen_tps, 500.0),
        score=ScoreBreakdown(
            quality=quality,
            speed=50.0,
            fit=50.0,
            context=50.0,
            weights={"quality": 0.4, "speed": 0.2, "fit": 0.2, "context": 0.2},
            total=total,
        ),
    )


def test_ties_keep_the_higher_quality() -> None:
    board = rank([scored("worse", 80.0, 60.0), scored("better", 80.0, 90.0)])
    assert ids(board) == ["better", "worse"]


def test_a_tie_on_quality_too_is_broken_by_speed_and_then_by_name() -> None:
    board = rank(
        [
            scored("slow", 80.0, 60.0, gen_tps=5.0),
            scored("fast", 80.0, 60.0, gen_tps=50.0),
        ]
    )
    assert ids(board) == ["fast", "slow"]
    same = rank([scored("b", 80.0, 60.0), scored("a", 80.0, 60.0)])
    assert ids(same) == ["a", "b"]


def test_excluded_candidates_come_last_in_the_order_they_arrived() -> None:
    left_out = [
        Candidate(model_id="second", quant="Q4_K_M", excluded_because="no vision capability"),
        Candidate(model_id="first", quant="Q4_K_M", excluded_because="no vision capability"),
    ]
    board = rank([left_out[0], scored("ranked", 50.0, 50.0), left_out[1]])
    assert ids(board) == ["ranked", "second", "first"]
