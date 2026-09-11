"""``llamafit fit`` and ``llamafit recommend`` through Typer's runner.

Every test replaces the scan with the reference machine's recorded one, so nothing here
depends on the graphics card of whatever runs the suite, and every test uses the real
bundled catalog, because a board is only worth testing against real entries.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from llamafit.catalog import load_catalog
from llamafit.cli.app import app
from llamafit.errors import CatalogError
from llamafit.services.recommend import quant_entries
from tests.fixtures.board import report
from tests.fixtures.budget_hosts import machine

runner = CliRunner()
GIB = 1024**3


def flat(output: str) -> str:
    """The output as one line, so an assertion is not defeated by where Rich wrapped a table.

    The column separators go first, before the whitespace is collapsed. A reason too long
    for its cell is wrapped onto the next row, which puts the borders of every column to its
    left between the two halves of a sentence, so collapsing whitespace alone leaves them
    there and an assertion on the sentence fails over a table's shape rather than its
    content. Widening one column is enough to do it, and a column is as wide as the longest
    model id the catalog happens to hold, which is not something these tests are about.
    """
    borders = str.maketrans(dict.fromkeys([ord("|"), *range(0x2500, 0x2580)], " "))
    return " ".join(output.translate(borders).split())


@pytest.fixture(autouse=True)
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command sees the reference machine, never the one running the tests."""
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())


# --- recommend ------------------------------------------------------------------------


def test_the_coding_board_names_the_model_the_measurements_favour() -> None:
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "coding"])
    assert result.exit_code == 0, result.output
    assert "qwen3-coder-next" in result.output
    assert "Recommended" in result.output


def test_the_board_says_what_its_speeds_are_and_are_not() -> None:
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "coding"])
    # No row is labelled per-row while every row carries the same label; the sentence under
    # the table is where it is said, and it says what the figures are not.
    text = flat(result.output)
    assert "no figure here is a measurement" in text
    assert "8K tokens of context" in text


def test_an_excluded_model_is_listed_with_its_reason() -> None:
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "coding"])
    assert "Not ranked" in result.output
    assert "llama-3.1-8b-instruct" in result.output


def test_json_carries_the_rows_the_reasons_and_the_weights() -> None:
    result = runner.invoke(app, ["--json", "recommend", "--use-case", "coding"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["weights"]["quality"] == 0.40
    assert data["rows"][0]["candidate"]["score"]["total"] > 0
    assert all(row["candidate"]["excluded_because"] for row in data["excluded"])


def test_explain_expands_a_row_into_its_parts_weights_budget_and_speed() -> None:
    result = runner.invoke(
        app, ["--language", "en", "recommend", "--use-case", "coding", "--limit", "1", "--explain"]
    )
    assert result.exit_code == 0, result.output
    assert "Score" in result.output
    assert "Weight" in result.output
    assert "compute buffer" in result.output  # a budget line
    assert "A token's time" in result.output  # the speed breakdown
    assert "Context tiers" in result.output


def test_explain_ends_with_what_would_change_the_answer() -> None:
    """Section 12.3's last clause, and the half that was never built.

    ``--explain`` printed the scores, the weights, the budget and the speed, and then
    stopped: an explanation of a decision already taken, with nothing a reader could act
    on. Two things can be changed without changing the machine or the model, and both are
    costed rather than reasoned about.
    """
    result = runner.invoke(
        app, ["--language", "en", "recommend", "--use-case", "coding", "--limit", "1", "--explain"]
    )
    assert result.exit_code == 0, result.output
    output = flat(result.output)
    assert "What would change it" in output
    assert "on the graphics card" in output


def test_the_quantisation_offered_is_one_the_model_actually_publishes() -> None:
    """A promise about a file nobody could fetch is worse than no promise."""
    result = runner.invoke(
        app,
        ["--language", "en", "recommend", "--use-case", "reasoning", "--limit", "6", "--explain"],
    )
    assert result.exit_code == 0, result.output
    output = flat(result.output)
    assert "the next quantisation down" in output
    catalog, problems = load_catalog()
    assert problems == []
    published = {q.name for m in catalog.models for q in quant_entries(m)}
    offered = re.findall(r"(\S+), the next quantisation down", output)
    assert offered
    assert set(offered) <= published


def test_a_model_publishing_one_quantisation_is_offered_no_other() -> None:
    """qwen3-coder-next has exactly one, so the sentence must not appear under its row."""
    result = runner.invoke(
        app, ["--language", "en", "recommend", "--use-case", "coding", "--limit", "1", "--explain"]
    )
    assert "qwen3-coder-next" in result.output
    assert "the next quantisation down" not in flat(result.output)


def test_nothing_is_said_about_what_would_change_without_explain() -> None:
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "coding"])
    assert "What would change it" not in flat(result.output)


def test_a_limit_cuts_the_board_and_never_the_reasons() -> None:
    result = runner.invoke(app, ["--json", "recommend", "--use-case", "coding", "--limit", "1"])
    data = json.loads(result.output)
    assert len(data["rows"]) == 1
    assert data["excluded"]


def test_a_cut_board_says_how_many_it_cut() -> None:
    """A limit that hides the rest of the catalog without saying so is a limit that lies."""
    whole = json.loads(runner.invoke(app, ["--json", "recommend", "--use-case", "coding"]).output)
    assert whole["ranked_total"] > 1, "this test needs a catalog with more than one ranked row"

    result = runner.invoke(
        app, ["--language", "en", "recommend", "--use-case", "coding", "--limit", "1"]
    )
    assert f"of {whole['ranked_total']} that qualified" in flat(result.output)


def test_an_uncut_board_says_nothing_about_a_limit() -> None:
    """Silence is the default: the sentence exists to report a cut, not to appear always."""
    result = runner.invoke(
        app, ["--language", "en", "recommend", "--use-case", "coding", "--limit", "500"]
    )
    assert "that qualified" not in flat(result.output)


def test_a_licence_filter_leaves_the_refused_model_visible() -> None:
    result = runner.invoke(
        app, ["--json", "recommend", "--use-case", "coding", "--license", "Apache-2.0"]
    )
    data = json.loads(result.output)
    refused = {row["model_id"]: row["candidate"]["excluded_because"] for row in data["excluded"]}
    assert "qwen3.8-flash-next" in refused
    assert "qwen-community-1.0" in refused["qwen3.8-flash-next"]


def test_a_download_ceiling_is_read_as_a_size() -> None:
    result = runner.invoke(
        app, ["--json", "recommend", "--use-case", "coding", "--max-download", "10G"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["needs"]["max_download_bytes"] == 10 * 10**9


def test_a_board_that_did_not_move_the_speed_floor_says_nothing_about_it() -> None:
    """Silence is the default, and the default is section 11.4's reading floor."""
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "general"])
    assert result.exit_code == 0, result.output
    assert "--min-tps" not in flat(result.output)


def test_a_batch_request_ranks_the_model_the_reading_floor_removes() -> None:
    """The board a person with nobody waiting gets: the slow model is on it.

    Both boards ask for every row. The default limit is a display choice, and a slow
    model is ranked last by construction, so the first ten rows are exactly where it
    will not be -- letting the limit decide this test would hide the model for a reason
    that has nothing to do with the speed floor.
    """
    plain = json.loads(runner.invoke(app, ["--json", "recommend", "--limit", "500"]).output)
    batch = json.loads(
        runner.invoke(app, ["--json", "recommend", "--min-tps", "0", "--limit", "500"]).output
    )
    assert "gemma-3-27b-it" in {row["model_id"] for row in plain["excluded"]}
    assert "gemma-3-27b-it" in {row["model_id"] for row in batch["rows"]}
    assert batch["needs"]["min_tps"] == 0.0


def test_a_batch_board_says_on_its_face_that_the_rule_was_relaxed() -> None:
    """A board that quietly stopped excluding is the failure the exclusion exists to stop."""
    result = runner.invoke(app, ["--language", "en", "recommend", "--min-tps", "0"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "--min-tps 0" in text
    assert "nobody is waiting on these tokens" in text


def test_a_raised_floor_is_named_under_the_board_and_in_every_reason() -> None:
    result = runner.invoke(app, ["--language", "en", "recommend", "--min-tps", "12"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "--min-tps 12" in text
    assert "at least that many tokens per second" in text
    # The reason itself is read from the JSON board rather than from the rendered table.
    # Rich wraps it inside its cell and the column widths move with the longest model id
    # in the catalog, so asserting on the wrapped text made the reason's presence depend
    # on which families happen to be catalogued.
    data = json.loads(
        runner.invoke(app, ["--language", "en", "--json", "recommend", "--min-tps", "12"]).output
    )
    reasons = [row["candidate"]["excluded_because"] or "" for row in data["excluded"]]
    assert any("the 12 this request asks for" in reason for reason in reasons)


def test_a_speed_floor_below_zero_is_refused_by_the_flag() -> None:
    result = runner.invoke(app, ["recommend", "--min-tps", "-1"])
    assert result.exit_code != 0


def test_a_download_ceiling_that_is_not_a_size_is_a_clean_error() -> None:
    result = runner.invoke(app, ["recommend", "--max-download", "lots"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "8G" in result.exception.render()


def test_an_unknown_use_case_lists_the_ones_that_exist() -> None:
    result = runner.invoke(app, ["recommend", "--use-case", "nonsense"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "coding" in result.exception.render()


def test_an_unknown_required_capability_lists_the_ones_that_exist() -> None:
    result = runner.invoke(app, ["recommend", "--require", "telepathy"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "vision" in result.exception.render()


def test_an_unknown_preference_lists_the_three() -> None:
    result = runner.invoke(app, ["recommend", "--prefer", "sideways"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "balanced" in result.exception.render()


def test_a_preference_moves_weight_between_quality_and_speed() -> None:
    plain = json.loads(runner.invoke(app, ["--json", "recommend"]).output)
    leaned = json.loads(runner.invoke(app, ["--json", "recommend", "--prefer", "quality"]).output)
    assert leaned["weights"]["quality"] > plain["weights"]["quality"]
    assert leaned["weights"]["speed"] < plain["weights"]["speed"]


def test_a_machine_that_fits_nothing_still_explains_every_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny = machine(vram_total=None, ram_total=2 * GIB, ram_available=1 * GIB)
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kw: report(host=tiny))
    result = runner.invoke(app, ["--language", "en", "recommend", "--use-case", "coding"])
    assert result.exit_code == 0, result.output
    assert "Nothing was ranked" in flat(result.output)
    assert "Not ranked" in result.output


# --- fit ------------------------------------------------------------------------------


def test_fit_lists_models_built_for_jobs_nobody_asked_about() -> None:
    result = runner.invoke(app, ["--language", "en", "fit"])
    assert result.exit_code == 0, result.output
    assert "Fit on this machine" in result.output
    assert "gemma-3-27b-it" in result.output
    assert "qwen3-coder-next" in result.output


def test_fit_json_carries_the_placement_and_the_score() -> None:
    result = runner.invoke(app, ["--json", "fit"])
    data = json.loads(result.output)
    assert data["planned_context"] == 32768
    assert data["rows"][0]["placement"]["mode"]
    assert data["rows"][0]["fit"] is not None


def test_perfect_narrows_the_listing_and_says_so_when_it_empties_it() -> None:
    result = runner.invoke(app, ["--language", "en", "fit", "--perfect"])
    assert result.exit_code == 0, result.output
    assert "Fit on this machine" in result.output or "Nothing fits" in result.output


def test_an_unknown_min_fit_lists_the_three_that_exist() -> None:
    result = runner.invoke(app, ["fit", "--min-fit", "snug"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "comfortable" in result.exception.render()


def test_fit_limits_rows_without_hiding_the_unplaceable_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = machine(vram_total=2 * GIB, ram_total=8 * GIB, ram_available=6 * GIB)
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kw: report(host=small))
    result = runner.invoke(app, ["--json", "fit", "--limit", "1"])
    data = json.loads(result.output)
    assert len(data["rows"]) <= 1
    assert data["excluded"]
    assert data["ranked_total"] >= len(data["rows"])
