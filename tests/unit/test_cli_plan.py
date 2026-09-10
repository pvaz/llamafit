"""``llamafit plan`` through Typer's runner, against the reference machine's recorded scan.

The last line this command prints is pasted into a terminal by somebody who has decided to
trust it, so the tests below are mostly about that line and about the working shown above
it: the budget with the source of each figure, the ladder with the band that pages, and the
runs the catalog records beside the estimate rather than in place of it.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.errors import CatalogError
from llamafit.models.llamacpp import LocalModel
from tests.fixtures.board import model_and_quant, report
from tests.fixtures.budget_hosts import machine

runner = CliRunner()
GIB = 1024**3


def flat(output: str) -> str:
    """The output as one line, so an assertion is not defeated by where Rich wrapped a table."""
    return " ".join(output.split())


@pytest.fixture(autouse=True)
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command sees the reference machine, never the one running the tests."""
    monkeypatch.setattr("llamafit.cli.plan_cmd.scan", lambda **_kwargs: report())


def test_the_plan_ends_in_a_command_line_that_names_the_planned_flags() -> None:
    result = runner.invoke(app, ["--language", "en", "plan", "qwen3.8-flash-next"])
    assert result.exit_code == 0, result.output
    assert "llama-server" in result.output
    assert "--n-cpu-moe" in result.output
    assert "ffn_.*_shexp=CPU" in result.output
    assert "--fit off" in result.output.replace("\n", " ")


def test_every_budget_line_says_whether_it_is_a_file_figure_or_a_formula() -> None:
    result = runner.invoke(app, ["--language", "en", "plan", "qwen3-coder-next"])
    assert "Memory budget" in result.output
    assert "compute buffer" in result.output
    assert "formula" in result.output
    assert "file" in result.output


def test_the_ladder_names_the_rungs_that_would_page_rather_than_calling_them_a_failure() -> None:
    result = runner.invoke(app, ["--language", "en", "plan", "qwen3.8-flash-next"])
    assert "Context tiers" in result.output
    assert "pages" in result.output
    assert "the log looks healthy" in flat(result.output)


def test_the_recorded_runs_appear_beside_the_estimate_and_say_they_are_not_it() -> None:
    result = runner.invoke(app, ["--language", "en", "plan", "qwen3.8-flash-next"])
    assert "Recorded elsewhere" in result.output
    assert "not benchmarks of this one" in flat(result.output)
    assert "(estimated)" in flat(result.output)


def test_a_context_that_does_not_fit_is_costed_beside_the_one_that_does() -> None:
    result = runner.invoke(
        app, ["--language", "en", "plan", "qwen3.8-flash-next", "--context", "262144"]
    )
    assert result.exit_code == 0, result.output
    assert "would have cost" in flat(result.output)


def test_a_hand_set_micro_batch_reaches_the_command_line() -> None:
    result = runner.invoke(app, ["--json", "plan", "qwen3-coder-next", "--ub", "512"])
    data = json.loads(result.output)
    assert data["placement"]["micro_batch"] == 512
    assert data["command"][data["command"].index("-ub") + 1] == "512"


def test_a_target_speed_is_answered_one_way_or_the_other() -> None:
    result = runner.invoke(
        app, ["--language", "en", "plan", "qwen3.8-flash-next", "--target-tps", "500"]
    )
    assert result.exit_code == 0, result.output
    assert "500 tokens per second" in flat(result.output)


def test_json_is_the_same_report_the_table_was_drawn_from() -> None:
    result = runner.invoke(app, ["--json", "plan", "qwen3-coder-next"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_id"] == "qwen3-coder-next"
    assert data["placement"]["budget"]["lines"]
    assert data["placement"]["tiers"][0]["verdict"]
    assert data["speed"]["confidence"] == "estimated"


def test_a_quant_the_model_does_not_publish_lists_the_ones_it_does() -> None:
    result = runner.invoke(app, ["plan", "qwen3-coder-next", "--quant", "Q2_K"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "UD-Q4_K_XL" in result.exception.render()


def test_the_quant_can_be_named_in_any_case() -> None:
    result = runner.invoke(app, ["--json", "plan", "qwen3-coder-next", "--quant", "ud-q4_k_xl"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["quant"] == "UD-Q4_K_XL"


def test_a_model_that_is_not_in_the_catalog_suggests_the_closest_ids() -> None:
    result = runner.invoke(app, ["plan", "qwen3-coder"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "qwen3-coder-next" in result.exception.render()


def test_a_file_already_on_disk_is_the_one_the_command_line_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, quant = model_and_quant("qwen3-coder-next")
    bare = quant.files[0].rsplit("/", 1)[-1]
    monkeypatch.setattr(
        "llamafit.cli.plan_cmd.scan",
        lambda **_kw: report(local_models=[LocalModel(path=f"/models/{bare}", bytes=1)]),
    )
    result = runner.invoke(app, ["--json", "plan", "qwen3-coder-next"])
    data = json.loads(result.output)
    assert data["model_present"]
    assert data["model_path"] == f"/models/{bare}"


def test_a_machine_with_no_room_says_so_instead_of_printing_a_plan_that_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny = machine(vram_total=None, ram_total=4 * GIB, ram_available=2 * GIB)
    monkeypatch.setattr("llamafit.cli.plan_cmd.scan", lambda **_kw: report(host=tiny))
    result = runner.invoke(app, ["--language", "en", "plan", "qwen3.8-flash-next"])
    assert result.exit_code == 0, result.output
    assert "nowhere" in flat(result.output)


def test_no_vision_is_carried_into_the_plan() -> None:
    result = runner.invoke(app, ["--json", "plan", "gemma-3-27b-it", "--no-vision"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["placement"]["projector_pool"] is None
    assert data["projector_path"] is None
