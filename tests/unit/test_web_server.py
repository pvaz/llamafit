"""Where the dashboard is allowed to listen, and what it says before it listens elsewhere.

The guard these tests exist for is not the default. A default that happens to be loopback
is worth nothing on the day somebody wires an address in from a config file; what is worth
something is that :func:`serve` refuses an address it was not told, in writing, to accept.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from llamafit.errors import ConfigError, NotInstalledError
from llamafit.web import server


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.1.5", "localhost", "LOCALHOST", "::1", "[::1]", " 127.0.0.1 "]
)
def test_these_addresses_reach_only_this_machine(host: str) -> None:
    assert server.is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com", ""])
def test_these_addresses_do_not(host: str) -> None:
    """A name that is not an address counts as remote: resolving it is somebody else's DNS."""
    assert not server.is_loopback(host)


def test_serving_elsewhere_is_refused_unless_it_was_asked_for() -> None:
    with pytest.raises(ConfigError) as failure:
        server.serve(host="0.0.0.0", runner=lambda *a, **k: None)
    rendered = failure.value.render()
    assert "0.0.0.0" in rendered
    assert "no authentication" in rendered


def test_the_warning_names_the_consequence_and_not_only_the_setting() -> None:
    warning = server.remote_warning("0.0.0.0", 8765)
    assert "0.0.0.0:8765" in warning
    for consequence in ("your hardware", "your file paths", "There is no password"):
        assert consequence in warning


def test_serving_here_binds_here() -> None:
    seen: dict[str, Any] = {}

    def runner(app: object, *, host: str, port: int) -> None:
        seen.update(app=app, host=host, port=port)

    server.serve(runner=runner)
    assert seen["host"] == server.DEFAULT_HOST
    assert seen["port"] == server.DEFAULT_PORT


def test_serving_elsewhere_on_purpose_is_allowed_and_still_addressable() -> None:
    seen: dict[str, Any] = {}

    def runner(app: object, *, host: str, port: int) -> None:
        seen.update(host=host, port=port, app=app)

    server.serve(host="192.168.1.10", port=9000, allow_remote=True, runner=runner)
    assert seen["host"] == "192.168.1.10"
    # The Host header check would otherwise refuse every request to the address just bound.
    assert "192.168.1.10" in seen["app"].user_middleware[0].kwargs["allowed_hosts"]


def test_opening_a_browser_is_asked_for_and_points_at_a_reachable_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(server, "_open_when_up", opened.append)
    server.serve(host="0.0.0.0", open_browser=True, allow_remote=True, runner=lambda *a, **k: None)
    # "every interface" is not somewhere a browser can go; loopback is one of them.
    assert opened == ["http://127.0.0.1:8765/"]


@pytest.mark.parametrize(
    ("host", "expected"),
    [("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"), ("::", "127.0.0.1"), ("::1", "[::1]")],
)
def test_the_browser_is_sent_somewhere_it_can_actually_go(host: str, expected: str) -> None:
    assert server._addressable(host) == expected


def test_a_machine_with_no_browser_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str) -> bool:
        raise RuntimeError("no browser here")

    monkeypatch.setattr(server.webbrowser, "open", refuse)
    server._open("http://127.0.0.1:8765/")  # must not raise


def test_the_browser_is_opened_once_the_server_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[float, object]] = []

    class Recorder:
        def __init__(self, delay: float, function: object, args: tuple[str, ...]) -> None:
            started.append((delay, args))

        def start(self) -> None:
            return None

    monkeypatch.setattr(server.threading, "Timer", Recorder)
    server._open_when_up("http://127.0.0.1:8765/")
    assert started and started[0][1] == ("http://127.0.0.1:8765/",)


def test_a_missing_web_extra_is_explained_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI is optional on purpose, so meeting it missing is normal and gets a command."""
    monkeypatch.setitem(sys.modules, "llamafit.web.api", None)
    with pytest.raises(NotInstalledError) as failure:
        server.build_app()
    assert 'pip install "llamafit[web]"' in failure.value.render()


def test_uvicorn_is_handed_the_application(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(app=app, **kwargs))
    marker = object()
    server._uvicorn_run(marker, host="127.0.0.1", port=1)  # type: ignore[arg-type]
    assert seen["app"] is marker
    assert seen["host"] == "127.0.0.1"
