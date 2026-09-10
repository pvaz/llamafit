"""``llamafit`` with no arguments: the dashboard, or the board it would have shown.

Section 13.1 gives that row two halves and both are load-bearing. The first is why the
dashboard exists at all -- somebody who has just installed this types the name of the
program, not a subcommand of it. The second is why a pipe, a redirect or a CI job still
gets an answer instead of a full-screen application fighting with a file handle.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.tui import entry
from tests.fixtures.board import report

runner = CliRunner()


class Stream:
    """A stand-in for stdin or stdout that knows only whether it is a terminal."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_a_terminal_is_one_only_when_both_ends_are_one() -> None:
    assert entry.interactive(stdin=Stream(True), stdout=Stream(True))
    assert not entry.interactive(stdin=Stream(True), stdout=Stream(False))
    assert not entry.interactive(stdin=Stream(False), stdout=Stream(True))


def test_a_stream_that_cannot_say_whether_it_is_a_terminal_is_not_one() -> None:
    assert not entry.interactive(stdin=object(), stdout=object())


def test_in_a_terminal_the_dashboard_is_what_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(entry, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(entry, "run_dashboard", lambda: opened.append(True))
    entry.open_dashboard(None)  # type: ignore[arg-type]
    assert opened == [True]


def test_run_dashboard_builds_the_application_and_runs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from llamafit.tui.app import LlamaFitApp

    ran: list[Any] = []
    monkeypatch.setattr(LlamaFitApp, "run", lambda self, *a, **k: ran.append(self))
    entry.run_dashboard()
    assert len(ran) == 1
    assert isinstance(ran[0], LlamaFitApp)


def test_with_no_terminal_it_prints_the_board_recommend_would_have_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Typer's runner is not a terminal, which is exactly the case being tested.
    monkeypatch.setattr("llamafit.cli.board_cmd.scan", lambda **_kwargs: report())
    result = runner.invoke(app, ["--language", "en"])
    assert result.exit_code == 0, result.output
    assert "Recommended" in result.output
    # And the same caption the command prints, so the two are the same board and not two.
    assert "no figure here is a measurement" in " ".join(result.output.split())


def test_the_fallback_says_why_it_is_not_a_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.board_cmd.scan", lambda **_kwargs: report())
    result = runner.invoke(app, ["--language", "en"])
    combined = " ".join((result.output + getattr(result, "stderr", "")).split())
    assert "no terminal" in combined or "recommend" in combined
