"""``llamafit info`` against the real catalog and the reference machine's recorded scan.

``test_cli_catalog.py`` next door checks ``info`` against a small made-up catalog, which is
the right shape for the facts half: a licence, a benchmark and a bracket in a model name do
not need a real file behind them. The other half does. Section 13.1 asks ``info`` for the
budget on this host per quantisation, and a budget needs a tensor table, so everything here
uses bundled entries that have one and sizes them against the machine the project records.

What these tests are mostly about is the line between this command and ``plan``. ``info``
answers "which of these quantisations can this machine run, and how fast", one line each;
``plan`` answers "how do I launch this one", with the budget component by component, the
ladder and the command. Two commands printing the same screen is a defect of its own, so
one test here asserts the things ``info`` deliberately does not print.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.errors import CatalogError
from tests.fixtures.board import report

runner = CliRunner()


def flat(output: str) -> str:
    """The output as one line, with the table borders gone first.

    A cell too long for its column is wrapped onto the next row, which leaves the borders
    of every column to its left sitting between the two halves of a sentence; collapsing
    whitespace alone would not remove them, and an assertion would then fail over a
    table's shape rather than its content.
    """
    borders = str.maketrans(dict.fromkeys([ord("|"), *range(0x2500, 0x2580)], " "))
    return " ".join(output.translate(borders).split())


@pytest.fixture(autouse=True)
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command sees the reference machine, never the one running the tests."""
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())


def test_info_sizes_every_quant_against_the_machine_it_was_asked_about() -> None:
    # The finding: `info` printed the catalog's facts and stopped. A person who had picked
    # a quantisation had nowhere to ask what it would cost here except `plan`, and `plan`
    # answers about one quantisation at a time.
    result = runner.invoke(app, ["--language", "en", "info", "gpt-oss-120b"])
    assert result.exit_code == 0, result.output
    output = flat(result.output)
    assert "On this machine" in output
    for name in ("MXFP4", "UD-Q4_K_XL", "UD-Q6_K_XL"):
        assert name in output
    assert "experts in RAM" in output
    assert "Sized for 32,768 tokens" in output


def test_info_stops_short_of_the_screen_plan_prints() -> None:
    """Where the line is drawn, asserted from the side that is easy to cross by accident."""
    result = runner.invoke(app, ["--language", "en", "info", "gpt-oss-120b"])
    output = flat(result.output)
    assert "Memory budget" not in output  # the component-by-component budget is plan's
    assert "Context tiers" not in output  # so is the ladder
    assert "A token's time" not in output  # so is the speed breakdown
    assert "llama-server" not in output  # and so is the command line
    assert "llamafit plan" in output  # but the caption says where all four are


def test_info_quant_narrows_both_tables_and_matches_any_case() -> None:
    result = runner.invoke(
        app, ["--language", "en", "info", "gpt-oss-120b", "--quant", "ud-q6_k_xl"]
    )
    assert result.exit_code == 0, result.output
    # From the first table heading on, so the assertion is about the two tables and not
    # about the curator's architecture note, which names MXFP4 in prose.
    tables = flat(result.output.partition("Quants")[2])
    assert tables.count("UD-Q6_K_XL") == 2
    assert "MXFP4" not in tables
    assert "UD-Q4_K_XL" not in tables


def test_a_quant_the_model_does_not_publish_lists_the_ones_it_does() -> None:
    result = runner.invoke(app, ["info", "gpt-oss-120b", "--quant", "Q2_K"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    rendered = result.exception.render()
    assert "MXFP4" in rendered
    assert "gpt-oss-120b" in rendered


def test_info_and_plan_refuse_an_unknown_quant_with_the_same_sentence() -> None:
    """One matcher behind both, so the two commands cannot come to disagree about a name."""
    from_info = runner.invoke(app, ["info", "gpt-oss-120b", "--quant", "Q2_K"]).exception
    from_plan = runner.invoke(app, ["plan", "gpt-oss-120b", "--quant", "Q2_K"]).exception
    assert isinstance(from_info, CatalogError)
    assert isinstance(from_plan, CatalogError)
    assert from_info.render() == from_plan.render()


def test_info_context_sizes_for_the_number_asked_for() -> None:
    result = runner.invoke(app, ["--language", "en", "info", "gpt-oss-120b", "--context", "65536"])
    assert result.exit_code == 0, result.output
    assert "Sized for 65,536 tokens" in flat(result.output)


def test_a_longer_context_costs_the_card_more_than_a_shorter_one() -> None:
    """The flag has to move the arithmetic, not only the caption above it."""
    short = json.loads(
        runner.invoke(
            app, ["--json", "info", "gpt-oss-120b", "--quant", "UD-Q4_K_XL", "--context", "8192"]
        ).output
    )
    long = json.loads(
        runner.invoke(
            app, ["--json", "info", "gpt-oss-120b", "--quant", "UD-Q4_K_XL", "--context", "65536"]
        ).output
    )
    assert (
        long["quants"][0]["placement"]["budget"]["vram_required"]
        > short["quants"][0]["placement"]["budget"]["vram_required"]
    )


def test_a_context_above_what_the_model_holds_is_captioned_at_what_it_holds() -> None:
    """A caption naming the figure the user typed would label one configuration with another."""
    result = runner.invoke(
        app, ["--language", "en", "info", "gpt-oss-120b", "--quant", "MXFP4", "--context", "999999"]
    )
    assert result.exit_code == 0, result.output
    output = flat(result.output)
    assert "Sized for 131,072 tokens" in output  # gpt-oss-120b's native length
    assert "999,999" not in output
    assert "999999" not in output


def test_info_json_carries_the_placement_the_table_was_drawn_from() -> None:
    result = runner.invoke(app, ["--json", "info", "gpt-oss-120b", "--quant", "UD-Q4_K_XL"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [q["name"] for q in data["quants"]] == ["UD-Q4_K_XL"]
    quant = data["quants"][0]
    assert quant["placement"]["mode"] == "moe-offload"
    assert quant["placement"]["budget"]["verdict"]
    assert quant["speed"]["confidence"] == "estimated"
    assert quant["unplaceable_because"] is None


def test_info_says_a_model_fits_nowhere_rather_than_leaving_the_row_blank() -> None:
    result = runner.invoke(app, ["--language", "en", "info", "kimi-k3"])
    assert result.exit_code == 0, result.output
    output = flat(result.output)
    assert "nowhere" in output  # the run mode
    assert "no room" in output  # the verdict
    # A configuration that fits nowhere is estimated at zero tokens per second, which is
    # arithmetic rather than a claim: a speed column reading 0.0 would tell a reader the
    # model runs very slowly here, when what is true is that it does not run.
    assert "0.0" not in output


def test_a_substituted_machine_is_marked_above_the_facts_and_not_only_above_the_budget() -> None:
    result = runner.invoke(app, ["--language", "en", "--memory", "24G", "info", "gpt-oss-120b"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert "SIMULATED" in lines[0]
    assert "On this machine" in flat(result.output)
