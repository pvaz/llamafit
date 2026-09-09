"""Find ``llama-server`` instances already running on this machine."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from llamafit.models.host import Probe
from llamafit.models.llamacpp import RunningServer

DEFAULT_PORTS = [8080, 8081, 8098]


class HttpClient(Protocol):
    """Minimal HTTP GET returning parsed JSON, or ``None`` on any failure."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url`` and parse JSON; never raise."""
        ...


class HttpxClient:
    """Real HTTP client with short timeouts; connection refused is just ``None``."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url``; returns ``None`` for network errors, non-200 or invalid JSON."""
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None


@dataclass
class FakeHttp:
    """Canned JSON responses keyed by URL."""

    responses: Mapping[str, Any]

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Return the canned response or ``None``."""
        return self.responses.get(url)


def candidate_ports(env: Mapping[str, str]) -> list[int]:
    """``LLAMA_SERVER_PORT`` first when set, then the usual llama.cpp ports."""
    ports: list[int] = []
    configured = env.get("LLAMA_SERVER_PORT")
    if configured and configured.isdigit():
        ports.append(int(configured))
    return ports + [p for p in DEFAULT_PORTS if p not in ports]


def discover_servers(http: HttpClient, ports: Iterable[int]) -> list[RunningServer]:
    """Probe each port for a llama-server and describe what it is serving."""
    return [server for server, _ in _discover(http, ports) if server]


def _as_int(value: object) -> int | None:
    """Coerce ``value`` to ``int`` when it safely represents one, else ``None``.

    Accepts real ``int`` values (``bool`` excluded, since it is a subclass of ``int``)
    and digit-only strings; anything else, including floats, is not coerced.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _discover(http: HttpClient, ports: Iterable[int]) -> list[tuple[RunningServer | None, Probe]]:
    results: list[tuple[RunningServer | None, Probe]] = []
    for port in ports:
        start = time.perf_counter()
        base = f"http://127.0.0.1:{port}"
        health = http.get_json(f"{base}/health")
        duration = int((time.perf_counter() - start) * 1000)
        if not isinstance(health, dict) or health.get("status") != "ok":
            results.append(
                (
                    None,
                    Probe(
                        name=f"server:{port}",
                        ok=False,
                        duration_ms=duration,
                        error="no llama-server answering",
                    ),
                )
            )
            continue
        try:
            model: str | None = None
            models = http.get_json(f"{base}/v1/models")
            if isinstance(models, dict):
                data = models.get("data")
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    candidate = data[0].get("id")
                    model = candidate if isinstance(candidate, str) else None

            n_ctx: int | None = None
            build: str | None = None
            props = http.get_json(f"{base}/props")
            if isinstance(props, dict):
                settings = props.get("default_generation_settings")
                if isinstance(settings, dict) and settings.get("n_ctx") is not None:
                    n_ctx = _as_int(settings["n_ctx"])
                build_info = props.get("build_info")
                if isinstance(build_info, str):
                    build = build_info

            results.append(
                (
                    RunningServer(url=base, model=model, n_ctx=n_ctx, build=build),
                    Probe(name=f"server:{port}", ok=True, duration_ms=duration),
                )
            )
        except Exception as exc:  # a strange responder must not stop detection
            results.append(
                (
                    None,
                    Probe(
                        name=f"server:{port}",
                        ok=False,
                        duration_ms=duration,
                        error=f"unexpected response: {exc}",
                    ),
                )
            )
    return results


def discover_with_probes(
    http: HttpClient, ports: Iterable[int]
) -> tuple[list[RunningServer], list[Probe]]:
    """Like ``discover_servers`` but also returns the probe records for ``doctor``."""
    pairs = _discover(http, ports)
    return [s for s, _ in pairs if s], [p for _, p in pairs]
