"""Section 13.1's five substitution flags, through Typer's runner.

The four that replace a machine have one promise between them that matters more than any
of their arithmetic: an answer computed for a machine nobody is sitting at must be
impossible to mistake for a scan of this one, in the terminal *and* in ``--json``, where
there is no red line to read. Every command wired to them is checked for both halves here.

``--max-context`` is the fifth and is not a machine at all. It caps the context the
planner sizes for, the ladder a launch script would choose from, the largest context
reported and the denominator the context score is measured against, so the tests for it
are about numbers rather than about marking.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from tests.fixtures.board import report

runner = CliRunner()

PROFILE = "reference-rtx4060-128gb"
"""The one bundled profile, which describes the machine the estimator was fitted on."""


def flat(output: str) -> str:
    """The output as one line, so an assertion survives wherever Rich wrapped a table."""
    return " ".join(output.split())


@pytest.fixture
def scanned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command sees the recorded reference machine instead of the one under test."""
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())


@pytest.fixture
def unscannable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scan that fails if it is taken at all, which is what ``--profile`` promises."""

    def refuse(**_kwargs: object) -> None:
        raise AssertionError("the machine was probed for a run that named a profile")

    monkeypatch.setattr("llamafit.cli.common.scan", refuse)
    monkeypatch.setattr(
        "llamafit.cli.common.detect_llamacpp",
        lambda *_a, **_k: report().llamacpp,
    )


# --- a profile replaces the machine outright -------------------------------------------


def test_a_profile_answers_without_probing_this_machine(unscannable: None) -> None:
    """Section 4.4's rule, end to end: a profile is answered from the file and nothing else.

    The fixture is the assertion that matters: it raises if the machine is scanned at all.
    What is checked here is that a real answer came out of the file rather than an empty
    one -- named models, ranked. Which models they are is a fact about the catalog on the
    day, so the ids are not spelled out.
    """
    result = runner.invoke(app, ["--language", "en", "--profile", PROFILE, "fit", "--limit", "3"])
    assert result.exit_code == 0, result.output
    ranked = json.loads(
        runner.invoke(app, ["--json", "--profile", PROFILE, "fit", "--limit", "3"]).output
    )
    assert len(ranked["rows"]) == 3
    assert all(row["placement"] for row in ranked["rows"])
    assert ranked["rows"][0]["model_id"] in result.output


def test_a_board_from_a_profile_says_so_before_a_figure_is_read(unscannable: None) -> None:
    text = flat(
        runner.invoke(
            app, ["--language", "en", "--profile", PROFILE, "recommend", "--limit", "1"]
        ).output
    )
    assert "SIMULATED" in text
    assert f"hardware profile {PROFILE}" in text
    assert "they are not this machine" in text


def test_the_fit_heading_stops_claiming_to_be_this_machine(unscannable: None) -> None:
    """The one heading the banner would otherwise contradict in its own words."""
    text = flat(runner.invoke(app, ["--language", "en", "--profile", PROFILE, "fit"]).output)
    assert "Fit on the simulated machine" in text
    assert "Fit on this machine" not in text


def test_a_scanned_board_keeps_its_heading_and_carries_no_banner(scanned: None) -> None:
    text = flat(runner.invoke(app, ["--language", "en", "fit", "--limit", "3"]).output)
    assert "Fit on this machine" in text
    assert "SIMULATED" not in text


@pytest.mark.parametrize("command", ["recommend", "fit"])
def test_a_board_json_carries_the_simulation_a_script_can_read(
    unscannable: None, command: str
) -> None:
    """The half of the promise a red line cannot keep: no heading, no colour, one boolean."""
    result = runner.invoke(app, ["--json", "--profile", PROFILE, command, "--limit", "1"])
    assert result.exit_code == 0, result.output
    board = json.loads(result.output)
    assert board["simulated"] is True
    assert board["simulation"]["profile"] == PROFILE
    assert board["simulation"]["path"].endswith(".json")


def test_a_plan_json_carries_the_simulation_too(unscannable: None) -> None:
    result = runner.invoke(app, ["--json", "--profile", PROFILE, "plan", "qwen3-0.6b"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["simulated"] is True
    assert plan["simulation"]["profile"] == PROFILE


@pytest.mark.parametrize("command", ["recommend", "fit"])
def test_a_board_from_the_scan_says_it_is_not_a_simulation(scanned: None, command: str) -> None:
    """A scan asserting it is not a simulation is worth saying: absence is not evidence."""
    board = json.loads(runner.invoke(app, ["--json", command, "--limit", "1"]).output)
    assert board["simulated"] is False
    assert board["simulation"] is None


def test_system_prints_the_machine_the_profile_stands_in_for(unscannable: None) -> None:
    """``--profile ... system`` is ``hardware show ... --as-host`` with llama.cpp beside it."""
    result = runner.invoke(app, ["--json", "--profile", PROFILE, "system"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["host"]["simulated"] is True
    assert payload["host"]["simulation"]["profile"] == PROFILE
    # The llama.cpp half is never substituted: the binary on this disk is real.
    assert payload["llamacpp"]["installed"] is True


def test_a_profile_that_does_not_exist_is_a_clean_error() -> None:
    result = runner.invoke(app, ["--language", "en", "--profile", "no-such-profile", "fit"])
    assert result.exit_code == 1
    assert "no-such-profile" in result.output + str(result.exception or "")


# --- an override moves one pool of the machine that is there ----------------------------


def test_an_override_names_the_pool_it_moved_in_the_json(scanned: None) -> None:
    board = json.loads(
        runner.invoke(app, ["--json", "--memory", "24G", "fit", "--limit", "1"]).output
    )
    assert board["simulated"] is True
    assert board["simulation"]["profile"] is None
    assert board["simulation"]["overrides"] == ["gpu_memory"]


def test_every_override_is_recorded_in_the_order_the_flags_name_them(scanned: None) -> None:
    flags = ["--json", "--memory", "24G", "--ram", "64GiB", "--cpu-cores", "4"]
    board = json.loads(runner.invoke(app, [*flags, "fit", "--limit", "1"]).output)
    assert board["simulation"]["overrides"] == ["gpu_memory", "ram", "cpu_cores"]


def test_a_bigger_card_changes_the_answer_it_is_asked_about(scanned: None) -> None:
    """The point of the flag: the same catalog on a machine with three times the VRAM."""
    as_scanned = json.loads(runner.invoke(app, ["--json", "fit", "--limit", "50"]).output)
    bigger = json.loads(
        runner.invoke(app, ["--json", "--memory", "24G", "fit", "--limit", "50"]).output
    )

    def vram(board: dict[str, object]) -> int:
        rows = board["rows"]
        assert isinstance(rows, list)
        return max(int(row["placement"]["budget"]["vram_required"]) for row in rows)

    assert vram(bigger) > vram(as_scanned)


def test_an_override_that_is_not_a_size_is_refused_in_the_usual_words(scanned: None) -> None:
    result = runner.invoke(app, ["--language", "en", "--memory", "24Q", "fit"])
    assert result.exit_code == 1
    assert "not a size" in result.output + str(result.exception or "")


def test_a_command_that_reports_on_this_machine_refuses_a_stand_in(scanned: None) -> None:
    """``doctor`` is probes, and a profile has none: better refused than quietly ignored."""
    result = runner.invoke(app, ["--language", "en", "--profile", PROFILE, "doctor"])
    assert result.exit_code == 1
    combined = result.output + str(result.exception or "")
    assert "doctor" in combined
    assert "machine LlamaFit is running on" in combined


def test_a_context_ceiling_is_not_a_machine_and_doctor_accepts_it(scanned: None) -> None:
    """The refusal covers the four that replace a machine and must not cover the fifth."""
    result = runner.invoke(app, ["--language", "en", "--max-context", "16384", "doctor"])
    assert result.exit_code in (0, 2), result.output


# --- --max-context caps every context that is planned, reported or scored ---------------


def test_the_ceiling_caps_the_largest_context_reported(scanned: None) -> None:
    plain = json.loads(runner.invoke(app, ["--json", "plan", "qwen3-0.6b"]).output)
    capped = json.loads(
        runner.invoke(app, ["--json", "--max-context", "16384", "plan", "qwen3-0.6b"]).output
    )
    assert plain["placement"]["max_context_fit"] > 16384
    assert capped["placement"]["max_context_fit"] == 16384


def test_the_ceiling_caps_the_ladder_a_launch_script_would_choose_from(scanned: None) -> None:
    capped = json.loads(
        runner.invoke(app, ["--json", "--max-context", "16384", "plan", "qwen3-0.6b"]).output
    )
    tiers = capped["placement"]["tiers"]
    assert tiers, "a ceiling at the bottom rung still leaves that rung"
    assert max(tier["tokens"] for tier in tiers) <= 16384


def test_the_ceiling_reaches_the_board_and_the_score_it_is_measured_against(
    scanned: None,
) -> None:
    board = json.loads(
        runner.invoke(
            app, ["--json", "--max-context", "16384", "recommend", "--use-case", "coding"]
        ).output
    )
    assert board["needs"]["max_context"] == 16384
    # Coding asks for 32,768 by default; a request that forbade it is not scored against it.
    assert board["requested_context"] == 16384
    assert board["planned_context"] == 16384


def test_a_ceiling_below_the_floor_names_both_flags(scanned: None) -> None:
    result = runner.invoke(
        app,
        ["--language", "en", "--max-context", "8192", "recommend", "--min-context", "32768"],
    )
    assert result.exit_code == 1
    combined = result.output + str(result.exception or "")
    assert "--max-context" in combined
    assert "--min-context" in combined
