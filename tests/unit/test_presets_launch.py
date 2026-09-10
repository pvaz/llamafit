"""Running a preset, waiting for it, and stopping it -- with nothing actually started.

Everything with a side effect is behind a protocol: :class:`~llamafit.presets.FakeLauncher`
stands in for the process, :class:`~llamafit.llamacpp.server.FakeHttp` for the server, and
the clock is a counter. That is the same seam ``hardware/runner.py`` and
``llamacpp/server.py`` cut, and it is why this module runs in milliseconds and never leaves
a llama-server behind on the machine running the suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llamafit.llamacpp.server import FakeHttp
from llamafit.presets import (
    FakeLauncher,
    RunningPreset,
    clear_record,
    is_answering,
    launch_preset,
    load_record,
    port_of_script,
    record_path,
    save_record,
    script_command,
    stop_preset,
    wait_for_health,
)

URL = "http://127.0.0.1:8080"
HEALTHY = {f"{URL}/health": {"status": "ok"}}


@pytest.fixture(autouse=True)
def _own_state_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test writes its notes about running servers into its own directory."""
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    yield


class Clock:
    """A monotonic clock that only moves when something waits on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_a_batch_preset_is_run_through_cmd_because_it_is_not_an_executable() -> None:
    assert script_command(Path("C:/p/start-x.cmd")) == ["cmd", "/c", "C:\\p\\start-x.cmd"]


def test_a_shell_preset_is_run_through_sh_so_a_missing_execute_bit_does_not_stop_it() -> None:
    script = Path("/p/start-x.sh")
    assert script_command(script) == ["/bin/sh", str(script)]


def test_the_port_is_read_out_of_the_script_so_a_hand_edit_is_honoured() -> None:
    assert port_of_script("llama-server --host 127.0.0.1 --port 8080 -c 4096") == 8080
    assert port_of_script('  --port "9001" ^') == 9001
    assert port_of_script("--port 8080 ... later somebody wrote --port 9999") == 9999
    assert port_of_script("no port here") is None


def test_a_server_that_answers_at_once_is_not_waited_for() -> None:
    clock = Clock()
    assert wait_for_health(FakeHttp(HEALTHY), URL, sleep=clock.sleep, monotonic=clock.monotonic)
    assert clock.slept == []


def test_a_server_that_never_answers_is_given_up_on_rather_than_waited_for_ever() -> None:
    clock = Clock()
    assert not wait_for_health(
        FakeHttp({}), URL, timeout=2.0, sleep=clock.sleep, monotonic=clock.monotonic
    )
    assert clock.slept  # it did wait between asks rather than spinning


def test_a_server_that_is_still_reading_its_weights_is_waited_for() -> None:
    clock = Clock()
    http = _AnswersLater(after=3)
    assert wait_for_health(http, URL, timeout=30.0, sleep=clock.sleep, monotonic=clock.monotonic)
    assert http.calls == 4


def test_a_health_endpoint_that_is_not_ok_is_not_an_answer() -> None:
    assert not is_answering(FakeHttp({f"{URL}/health": {"status": "loading model"}}), URL)


def test_launching_runs_the_script_itself_and_not_a_rebuilt_command_line(
    tmp_path: Path,
) -> None:
    script = _script(tmp_path)
    launcher = FakeLauncher()
    outcome = launch_preset(
        "qwen3-coder-next", script, URL, launcher=launcher, http=_HealthyOnceStarted(launcher)
    )
    assert outcome.result == "started"
    assert launcher.started == [script_command(script)]


def test_a_started_server_is_written_down_with_its_start_time_not_only_its_pid(
    tmp_path: Path,
) -> None:
    script = _script(tmp_path)
    launcher = FakeLauncher(pid=99, created_at=1234.5)
    launch_preset(
        "qwen3-coder-next", script, URL, launcher=launcher, http=_HealthyOnceStarted(launcher)
    )
    record = load_record("qwen3-coder-next")
    assert record is not None
    assert (record.pid, record.created_at) == (99, 1234.5)
    assert record.url == URL


def test_a_server_already_serving_is_reported_rather_than_joined_by_a_second_one(
    tmp_path: Path,
) -> None:
    launcher = FakeLauncher()
    http = _HealthyOnceStarted(launcher)
    outcome = launch_preset(
        "qwen3-coder-next", _script(tmp_path), URL, launcher=launcher, http=http
    )
    assert outcome.result == "started"
    again = launch_preset("qwen3-coder-next", _script(tmp_path), URL, launcher=launcher, http=http)
    assert again.result == "already-running"
    assert len(launcher.started) == 1


def test_a_script_that_cannot_be_run_at_all_is_a_failure_with_its_reason(
    tmp_path: Path,
) -> None:
    launcher = FakeLauncher(fail_with=OSError("no such file"))
    outcome = launch_preset(
        "qwen3-coder-next", _script(tmp_path), URL, launcher=launcher, http=FakeHttp({})
    )
    assert outcome.result == "failed-to-start"
    assert "no such file" in (outcome.error or "")
    assert load_record("qwen3-coder-next") is None


def test_a_server_that_starts_but_does_not_answer_is_a_different_answer(
    tmp_path: Path,
) -> None:
    clock = Clock()
    outcome = launch_preset(
        "qwen3-coder-next",
        _script(tmp_path),
        URL,
        launcher=FakeLauncher(),
        http=FakeHttp({}),
        timeout=2.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert outcome.result == "no-health"
    # It is still written down: something is running, and --stop has to be able to find it.
    assert load_record("qwen3-coder-next") is not None


def test_stopping_uses_the_note_and_then_forgets_it(tmp_path: Path) -> None:
    launcher = FakeLauncher(pid=77, created_at=5.0)
    launch_preset(
        "qwen3-coder-next",
        _script(tmp_path),
        URL,
        launcher=launcher,
        http=_HealthyOnceStarted(launcher),
    )
    record = stop_preset("qwen3-coder-next", launcher=launcher)
    assert record is not None
    assert launcher.stopped == [(77, 5.0)]
    assert load_record("qwen3-coder-next") is None


def test_stopping_something_that_was_never_started_says_so() -> None:
    assert stop_preset("qwen3-coder-next", launcher=FakeLauncher()) is None


def test_a_note_for_a_process_that_has_gone_is_forgotten_rather_than_kept() -> None:
    """A note naming a dead pid is a pid somebody's next --stop would aim at."""
    save_record(
        RunningPreset(
            model_id="qwen3-coder-next",
            pid=1,
            created_at=1.0,
            url=URL,
            script="start.cmd",
            started_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )
    )
    assert stop_preset("qwen3-coder-next", launcher=FakeLauncher(alive=False)) is None
    assert load_record("qwen3-coder-next") is None


def test_a_note_that_cannot_be_parsed_is_treated_as_no_note() -> None:
    path = record_path("qwen3-coder-next")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_record("qwen3-coder-next") is None


def test_forgetting_a_note_that_is_not_there_is_not_an_error() -> None:
    clear_record("never-launched")


class _HealthyOnceStarted:
    """A server that appears the moment the launcher is asked to start something.

    Which is what makes the "already serving" test mean anything: a fake that always
    answered would report every launch as one that had already happened, and a fake that
    never answered could not tell the two apart either.
    """

    def __init__(self, launcher: FakeLauncher) -> None:
        self.launcher = launcher

    def get_json(self, url: str, *, timeout: float = 1.5) -> object | None:
        return {"status": "ok"} if self.launcher.started else None


class _AnswersLater:
    """A server that says it is loading until the nth ask."""

    def __init__(self, *, after: int) -> None:
        self.after = after
        self.calls = 0

    def get_json(self, url: str, *, timeout: float = 1.5) -> object | None:
        self.calls += 1
        return {"status": "ok"} if self.calls > self.after else {"status": "loading model"}


def _script(tmp_path: Path) -> Path:
    """A preset script on disk, which is all ``launch`` needs of one."""
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text("@echo off\n", encoding="utf-8")
    return script
