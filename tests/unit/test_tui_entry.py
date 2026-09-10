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

from llamafit.cli.app import CliState, app
from llamafit.errors import CatalogError
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


class Context:
    """A stand-in for Typer's context, carrying only the global options."""

    def __init__(self, obj: CliState | None = None) -> None:
        self.obj = obj or CliState()


def test_in_a_terminal_the_dashboard_is_what_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[Any] = []
    monkeypatch.setattr(entry, "interactive", lambda *a, **k: True)
    monkeypatch.setattr(entry, "run_dashboard", lambda dashboard=None: opened.append(dashboard))
    entry.open_dashboard(Context())  # type: ignore[arg-type]
    assert opened == [None]


def test_a_dashboard_asked_for_nothing_is_the_applications_own() -> None:
    """No flags, no seeded dashboard: the application builds its own, as it always did."""
    assert entry.seeded_dashboard(CliState()) is None


def test_the_dashboard_opens_on_the_machine_the_command_line_named() -> None:
    """``llamafit --profile ...`` opens simulating that machine, badge already showing."""
    dashboard = entry.seeded_dashboard(CliState(profile="fleet-node", memory="24G"))
    assert dashboard is not None
    assert dashboard.substitution.profile == "fleet-node"
    assert dashboard.substitution.gpu_memory == 24 * 1000**3
    assert dashboard.substitution.active


def test_a_context_ceiling_alone_still_seeds_the_dashboard() -> None:
    """``--max-context`` is not a machine, and it must reach the screens all the same."""
    dashboard = entry.seeded_dashboard(CliState(max_context=16384))
    assert dashboard is not None
    assert dashboard.request.needs.max_context == 16384
    assert not dashboard.substitution.active


def test_a_size_that_is_not_a_size_is_refused_before_a_screen_is_drawn() -> None:
    with pytest.raises(CatalogError, match="not a size"):
        entry.seeded_dashboard(CliState(memory="24Q"))


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
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())
    result = runner.invoke(app, ["--language", "en"])
    assert result.exit_code == 0, result.output
    assert "Recommended" in result.output
    # And the same caption the command prints, so the two are the same board and not two.
    assert "no figure here is a measurement" in " ".join(result.output.split())


def test_the_fallback_says_why_it_is_not_a_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: report())
    result = runner.invoke(app, ["--language", "en"])
    combined = " ".join((result.output + getattr(result, "stderr", "")).split())
    assert "no terminal" in combined or "recommend" in combined
