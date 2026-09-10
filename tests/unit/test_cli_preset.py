"""``llamafit preset`` and ``llamafit launch`` through Typer's runner.

The machine is the recorded reference one and the server is a fake, so nothing here scans
hardware or starts a process. What is being checked is the part the other preset test
modules cannot see: that the command writes where it said it would, reports what became of
each file in words a person can act on, refuses to lose an edit, and hands ``launch`` the
script rather than a command line it rebuilt for itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.errors import BudgetError
from llamafit.presets import FakeLauncher, load_record
from tests.fixtures.board import report

runner = CliRunner()


def flat(output: str) -> str:
    """The output as one line, so an assertion is not defeated by where Rich wrapped it."""
    return " ".join(output.split())


@pytest.fixture(autouse=True)
def fixed_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLauncher]:
    """Every command sees the reference machine, a fake process and a fake server."""
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())
    launcher = FakeLauncher()
    monkeypatch.setattr("llamafit.cli.preset_cmd.launcher", lambda: launcher)
    monkeypatch.setattr(
        "llamafit.cli.preset_cmd.http_client", lambda: _HealthyOnceStarted(launcher)
    )
    yield launcher


def test_preset_writes_a_script_a_router_section_and_a_readme(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == [
        "README-qwen3-coder-next.md",
        "models-qwen3-coder-next.ini",
        "start-qwen3-coder-next.cmd",
    ]


def test_preset_shows_the_ladder_the_script_will_choose_from(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert "Context tiers the script may choose" in result.output
    assert "reads how much card memory is free" in flat(result.output)
    assert "http://127.0.0.1:8080" in flat(result.output)


def test_preset_says_what_became_of_each_file(tmp_path: Path) -> None:
    runner.invoke(app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    again = runner.invoke(
        app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert "already up to date" in flat(again.output)


def test_preset_keeps_a_file_you_edited_and_says_how_to_replace_it(tmp_path: Path) -> None:
    runner.invoke(app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text(script.read_text(encoding="utf-8") + "\nrem mine\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "--language",
            "en",
            "preset",
            "qwen3-coder-next",
            "--dir",
            str(tmp_path),
            "--port",
            "9001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "kept: you edited it" in flat(result.output)
    assert "--force" in flat(result.output)
    assert "rem mine" in script.read_text(encoding="utf-8")


def test_force_says_out_loud_that_it_discarded_your_edits(tmp_path: Path) -> None:
    runner.invoke(app, ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text(script.read_text(encoding="utf-8") + "\nrem mine\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["--language", "en", "preset", "qwen3-coder-next", "--dir", str(tmp_path), "--force"],
    )
    assert "your edits are gone" in flat(result.output)
    assert "rem mine" not in script.read_text(encoding="utf-8")


def test_the_port_asked_for_reaches_the_script_and_the_endpoint(tmp_path: Path) -> None:
    runner.invoke(
        app,
        ["preset", "qwen3-coder-next", "--dir", str(tmp_path), "--port", "8765"],
    )
    text = (tmp_path / "start-qwen3-coder-next.cmd").read_text(encoding="utf-8")
    assert "--port 8765" in text
    assert "http://127.0.0.1:8765" in text


def test_a_context_asked_for_becomes_the_top_of_the_ladder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--json", "preset", "qwen3-coder-next", "--dir", str(tmp_path), "--context", "16384"],
    )
    data = json.loads(result.output)
    assert data["planned_context"] == 16384
    assert data["rungs"][0]["tokens"] == 16384
    assert all(rung["tokens"] <= 16384 for rung in data["rungs"])


def test_json_names_every_file_and_what_happened_to_it(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--json", "preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_id"] == "qwen3-coder-next"
    assert [file["outcome"] for file in data["files"]] == ["created"] * 3
    assert data["probe_free_memory"] is True


def test_a_quant_the_model_does_not_publish_is_refused_by_name(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path), "--quant", "Q2_K"]
    )
    assert result.exit_code == 1
    assert "UD-Q4_K_XL" in result.exception.render()  # type: ignore[union-attr]


def test_a_model_that_fits_nowhere_gets_no_script_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset for a configuration that cannot run is a file that lies about a machine."""
    from llamafit.models.host import Memory

    tiny = report().host.model_copy(
        update={"gpus": [], "memory": Memory(total_bytes=2 * 1024**3, available_bytes=1024**3)}
    )
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report(host=tiny))
    result = runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert isinstance(result.exception, BudgetError)
    assert list(tmp_path.iterdir()) == []


def test_launch_runs_the_preset_script_itself(tmp_path: Path, fixed_machine: FakeLauncher) -> None:
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    result = runner.invoke(
        app, ["--language", "en", "launch", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert fixed_machine.started == [["cmd", "/c", str(tmp_path / "start-qwen3-coder-next.cmd")]]
    assert "is serving on" in flat(result.output)
    assert "/v1/chat/completions" in result.output


def test_launch_writes_the_preset_first_when_there_is_none(
    tmp_path: Path, fixed_machine: FakeLauncher
) -> None:
    result = runner.invoke(
        app, ["--language", "en", "launch", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Wrote a preset first" in flat(result.output)
    assert (tmp_path / "start-qwen3-coder-next.cmd").is_file()
    assert fixed_machine.started


def test_launch_honours_a_port_somebody_edited_into_their_own_script(
    tmp_path: Path,
) -> None:
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path), "--port", "9001"])
    result = runner.invoke(app, ["--json", "launch", "qwen3-coder-next", "--dir", str(tmp_path)])
    assert json.loads(result.output)["url"] == "http://127.0.0.1:9001"


def test_launch_remembers_what_it_started_so_stop_can_find_it(tmp_path: Path) -> None:
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    runner.invoke(app, ["launch", "qwen3-coder-next", "--dir", str(tmp_path)])
    record = load_record("qwen3-coder-next")
    assert record is not None
    assert record.url == "http://127.0.0.1:8080"


def test_stop_stops_what_was_started_and_says_so(
    tmp_path: Path, fixed_machine: FakeLauncher
) -> None:
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    runner.invoke(app, ["launch", "qwen3-coder-next", "--dir", str(tmp_path)])
    result = runner.invoke(app, ["--language", "en", "launch", "qwen3-coder-next", "--stop"])
    assert result.exit_code == 0, result.output
    assert "Stopped qwen3-coder-next" in flat(result.output)
    assert fixed_machine.stopped
    assert load_record("qwen3-coder-next") is None


def test_stopping_something_that_was_never_started_is_said_plainly() -> None:
    result = runner.invoke(app, ["--language", "en", "launch", "qwen3-coder-next", "--stop"])
    assert result.exit_code == 0, result.output
    assert "Nothing to stop" in flat(result.output)


def test_stop_in_json_says_whether_anything_was_stopped() -> None:
    result = runner.invoke(app, ["--json", "launch", "qwen3-coder-next", "--stop"])
    assert json.loads(result.output) == {"stopped": False}


def test_a_server_that_does_not_answer_is_reported_without_pretending_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llamafit.cli.preset_cmd.http_client", _Silent)
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    result = runner.invoke(
        app,
        [
            "--language",
            "en",
            "launch",
            "qwen3-coder-next",
            "--dir",
            str(tmp_path),
            "--timeout",
            "1",
        ],
    )
    assert "has not answered yet" in flat(result.output)


def test_a_preset_that_cannot_be_started_reports_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = FakeLauncher(fail_with=OSError("cmd is missing"))
    monkeypatch.setattr("llamafit.cli.preset_cmd.launcher", lambda: broken)
    monkeypatch.setattr("llamafit.cli.preset_cmd.http_client", _Silent)
    runner.invoke(app, ["preset", "qwen3-coder-next", "--dir", str(tmp_path)])
    result = runner.invoke(
        app, ["--language", "en", "launch", "qwen3-coder-next", "--dir", str(tmp_path)]
    )
    assert "could not be started" in flat(result.output)
    assert "cmd is missing" in flat(result.output)


class _HealthyOnceStarted:
    """A server that appears the moment the launcher is asked to start something."""

    def __init__(self, launcher: FakeLauncher) -> None:
        self.launcher = launcher

    def get_json(self, url: str, *, timeout: float = 1.5) -> object | None:
        return {"status": "ok"} if self.launcher.started else None


class _Silent:
    """A server that never answers."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> object | None:
        return None
