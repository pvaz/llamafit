"""``llamafit serve``: what it passes on, and what it says when the address is elsewhere."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app

runner = CliRunner()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what the command asked the server for, without starting one."""
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("llamafit.web.serve", fake)
    return seen


def test_it_serves_this_machine_by_default(served: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert served == {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": False,
        "allow_remote": False,
    }
    assert "http://127.0.0.1:8765/" in result.output


def test_a_port_and_the_browser_are_passed_on(served: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve", "--port", "9000", "--open"])
    assert result.exit_code == 0, result.output
    assert served["port"] == 9000
    assert served["open_browser"] is True


def test_typing_the_host_is_what_consents_to_a_remote_bind(served: dict[str, Any]) -> None:
    """A loopback default and a loopback address somebody chose must not look alike."""
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert served["allow_remote"] is True
    assert served["host"] == "0.0.0.0"


def test_a_remote_bind_says_what_it_exposes_before_it_starts(served: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"], color=False)
    output = " ".join(result.output.split())
    assert "There is no password" in output
    assert "your file paths" in output
    assert "Ctrl+C" in output


def test_asking_for_this_machine_by_name_says_nothing_alarming(served: dict[str, Any]) -> None:
    result = runner.invoke(app, ["serve", "--host", "localhost"])
    assert "no password" not in result.output
    assert served["allow_remote"] is True


def test_the_help_says_what_binding_elsewhere_costs() -> None:
    result = runner.invoke(app, ["serve", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    output = " ".join(result.output.split())
    assert "--host" in output
    assert "no password and no login" in output
    assert "--open" in output
