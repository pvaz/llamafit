import sys
from pathlib import Path

from llamafit.hardware.runner import CommandResult, FakeRunner, SubprocessRunner, probe


def test_subprocess_runner_runs_python() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "print('hi')"])
    assert result.ok
    assert result.stdout.strip() == "hi"
    assert result.duration_ms >= 0


def test_subprocess_runner_missing_program_is_an_error_not_an_exception() -> None:
    result = SubprocessRunner().run(["definitely-not-a-program-xyz"])
    assert not result.ok
    assert result.error is not None
    assert result.returncode is None


def test_subprocess_runner_timeout() -> None:
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2
    )
    assert not result.ok
    assert "timed out" in (result.error or "")


def test_fake_runner_matches_program_or_full_command() -> None:
    runner = FakeRunner({"nvidia-smi": "GPU 0", "rocm-smi --json": '{"ok": true}'})
    assert runner.run(["nvidia-smi", "-L"]).stdout == "GPU 0"
    assert runner.run(["rocm-smi", "--json"]).stdout == '{"ok": true}'
    assert not runner.run(["missing"]).ok
    assert runner.calls == [["nvidia-smi", "-L"], ["rocm-smi", "--json"], ["missing"]]


def test_probe_parses_and_records_success() -> None:
    runner = FakeRunner({"tool": "42"})
    value, rec = probe("tool", runner, ["tool"], int)
    assert value == 42
    assert rec.ok and rec.name == "tool" and rec.error is None


def test_probe_turns_parse_failure_into_probe_error() -> None:
    runner = FakeRunner({"tool": "forty-two"})
    value, rec = probe("tool", runner, ["tool"], int)
    assert value is None
    assert not rec.ok
    assert "invalid literal" in (rec.error or "")


def test_probe_reports_command_failure() -> None:
    runner = FakeRunner(
        {
            "tool": CommandResult(
                argv=["tool"],
                returncode=1,
                stdout="",
                stderr="bad",
                duration_ms=1,
            )
        }
    )
    value, rec = probe("tool", runner, ["tool"], int)
    assert value is None
    assert rec.error == "exit code 1: bad"


def test_empty_argv_is_an_error_not_an_exception() -> None:
    result = SubprocessRunner().run([])
    assert not result.ok and result.error == "empty command"
    fake = FakeRunner({})
    assert fake.run([]).error == "empty command"
    assert fake.calls == [[]]


def test_probe_writes_to_the_log_when_logging_is_set_up(tmp_path: Path) -> None:
    from llamafit.logging import setup_logging

    logger = setup_logging(tmp_path, verbose=True)
    try:
        probe("tool", FakeRunner({}), ["tool"], int)
        for handler in logger.handlers:
            handler.flush()
        assert "tool" in (tmp_path / "llamafit.log").read_text(encoding="utf-8")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
