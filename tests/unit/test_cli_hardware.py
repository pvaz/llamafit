"""Tests for the ``hardware`` command group.

Every test monkeypatches the names as ``hardware_cmd`` imports them, matching how the
catalog command tests work. Nothing here writes to the user's real data directory: the
tests that need a user profile directory point ``LLAMAFIT_PROFILES`` at ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.errors import ConfigError
from tests.fixtures import profiles

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_user_profiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test off whatever profiles the machine running it happens to have."""
    monkeypatch.setenv("LLAMAFIT_PROFILES", str(tmp_path / "no-profiles-here"))


def test_list_shows_the_bundled_profile() -> None:
    result = runner.invoke(app, ["hardware", "list"])
    assert result.exit_code == 0, result.output
    assert "reference-rtx4060" in result.output
    assert "bundled" in result.output


def test_list_json_carries_the_whole_profile() -> None:
    result = runner.invoke(app, ["--json", "hardware", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["name"] for row in payload] == ["reference-rtx4060-128gb"]
    assert payload[0]["bundled"] is True
    assert payload[0]["profile"]["memory"]["bandwidth_source"] == "measured"


def test_list_says_so_when_there_are_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("llamafit.cli.hardware_cmd.load_profiles", lambda: ([], []))
    result = runner.invoke(app, ["hardware", "list"])
    assert result.exit_code == 0
    assert "hardware path" in result.output


def test_show_prints_the_profile_with_its_provenance() -> None:
    result = runner.invoke(app, ["hardware", "show", "reference-rtx4060-128gb"])
    assert result.exit_code == 0, result.output
    assert "Provenance" in result.output
    assert "constants.py" in result.output.replace("\n", "")


def test_show_json_is_the_profile_document() -> None:
    result = runner.invoke(app, ["--json", "hardware", "show", "reference-rtx4060-128gb"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["memory"]["total"] == 128 * 1024**3


def test_show_as_host_marks_the_machine_as_simulated() -> None:
    result = runner.invoke(app, ["hardware", "show", "reference-rtx4060-128gb", "--as-host"])
    assert result.exit_code == 0, result.output
    assert "SIMULATED" in result.output
    assert "not this machine" in result.output.replace("\n", " ").replace("  ", " ")


def test_show_as_host_json_carries_the_flag_a_script_reads() -> None:
    result = runner.invoke(
        app, ["--json", "hardware", "show", "reference-rtx4060-128gb", "--as-host"]
    )
    assert result.exit_code == 0, result.output
    host = json.loads(result.output)
    assert host["simulated"] is True
    assert host["simulation"]["profile"] == "reference-rtx4060-128gb"
    assert host["simulation"]["overrides"] == []
    assert host["memory"]["total_bytes"] == 128 * 1024**3


def test_show_takes_a_path(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    result = runner.invoke(app, ["--json", "hardware", "show", str(tmp_path / "test-machine.json")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "test-machine"


def test_an_unknown_profile_exits_one_and_names_what_there_is() -> None:
    result = runner.invoke(app, ["hardware", "show", "nosuch"])
    assert result.exit_code == 1
    assert isinstance(result.exception, ConfigError)
    rendered = result.exception.render()
    assert "no hardware profile named 'nosuch'" in rendered
    assert "reference-rtx4060-128gb" in rendered


def test_validate_checks_the_bundled_profiles() -> None:
    result = runner.invoke(app, ["hardware", "validate"])
    assert result.exit_code == 0, result.output
    assert "no problems found" in result.output


def test_validate_one_file_that_is_wrong_exits_one(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(memory={"total": "lots"}))
    result = runner.invoke(app, ["hardware", "validate", str(tmp_path / "test-machine.json")])
    assert result.exit_code == 1
    assert "memory.total" in result.output


def test_validate_json_is_a_list_of_problems(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(match={"gpu_name_contains": "RTX 4090"}))
    result = runner.invoke(
        app, ["--json", "hardware", "validate", str(tmp_path / "test-machine.json")]
    )
    assert result.exit_code == 1
    problems = json.loads(result.output)
    assert [p["location"] for p in problems] == ["match"]
    assert problems[0]["profile"] == "test-machine"


def test_path_prints_the_user_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["hardware", "path"])
    assert result.exit_code == 0, result.output
    assert "no-profiles-here" in result.output
    assert "does not exist yet" in result.output


def test_path_json_names_both_directories() -> None:
    result = runner.invoke(app, ["--json", "hardware", "path"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["exists"] is False
    assert payload["suffix"] == ".json"
    assert Path(payload["bundled"]).is_dir()


def test_a_user_profile_is_listed_beside_the_bundled_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = profiles.write(tmp_path / "mine", profiles.document(name="my-laptop"))
    monkeypatch.setenv("LLAMAFIT_PROFILES", str(directory))
    result = runner.invoke(app, ["--json", "hardware", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["name"] for row in payload] == ["reference-rtx4060-128gb", "my-laptop"]
    assert [row["bundled"] for row in payload] == [True, False]


def test_a_broken_user_profile_warns_without_hiding_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "mine"
    directory.mkdir()
    (directory / "broken.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("LLAMAFIT_PROFILES", str(directory))
    result = runner.invoke(app, ["hardware", "list"])
    assert result.exit_code == 0
    assert "reference-rtx4060" in result.output, "browsing keeps working"
    assert "hardware validate" in result.output


def test_the_help_names_every_subcommand() -> None:
    result = runner.invoke(app, ["hardware", "--help"])
    assert result.exit_code == 0
    for name in ("list", "show", "validate", "path"):
        assert name in result.output
