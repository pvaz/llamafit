# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Start the dashboard, and refuse to put it anywhere but this machine by accident.

Section 1.3 is not a preference: remote hosts, multi-user access and authentication are
outside this project, and the dashboard binds to loopback. Everything here follows from
that one sentence.

A default is not a guard. :func:`serve` refuses a non-loopback address outright unless the
caller says, in a separate argument, that it meant it -- which the command line only says
when ``--host`` was actually typed. So the address cannot drift outwards through a config
file, an environment variable or a default nobody re-read, and a library caller has to
write the consent down. When the consent is there, :func:`remote_warning` says what was
just agreed to in plain words, because a warning that only repeats the address teaches
nobody anything: this server has no password, and it reads the machine's hardware, its
disks and its file paths.
"""

from __future__ import annotations

import ipaddress
import threading
import webbrowser
from typing import TYPE_CHECKING, Any

from llamafit.errors import ConfigError, NotInstalledError
from llamafit.i18n import _
from llamafit.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from fastapi import FastAPI

DEFAULT_HOST = "127.0.0.1"
"""Loopback. The dashboard is for the person sitting at this machine."""

DEFAULT_PORT = 8765
"""Section 21's choice: unassigned locally, and easy to remember."""

_log = get_logger("web")


def is_loopback(host: str) -> bool:
    """Whether an address reaches only this machine.

    Args:
        host: What to bind to, as an address or a name.

    Returns:
        True for ``127.0.0.1``, ``::1``, anything else in ``127.0.0.0/8``, and the name
        ``localhost``. A name that is not an address is False: this cannot resolve it
        without a DNS lookup, and a lookup that decides whether a security guard applies
        is a guard somebody else's resolver controls.
    """
    name = host.strip().strip("[]").casefold()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def remote_warning(host: str, port: int) -> str:
    """What binding away from loopback actually means, said before it happens.

    Args:
        host: The address that was asked for.
        port: The port it will listen on.

    Returns:
        The warning to print. It names the consequence rather than the setting: there is
        no password on this server, and what it serves is a description of this machine.
    """
    return _(
        "%(host)s:%(port)d is not this machine only. Anyone who can reach this computer "
        "on that port can read everything the dashboard shows -- your hardware, your "
        "disks, your file paths and the models you have downloaded. There is no password, "
        "because LlamaFit is meant for the person sitting at the keyboard. Press Ctrl+C "
        "now if you did not mean this."
    ) % {"host": host, "port": port}


def build_app(host: str = DEFAULT_HOST) -> FastAPI:
    """Build the application, explaining plainly when the web extra is not installed.

    Args:
        host: The address the server will bind to, so a deliberate non-loopback bind can
            still be addressed by that name through the ``Host`` header check.

    Returns:
        The FastAPI application.

    Raises:
        NotInstalledError: FastAPI or uvicorn is missing. They are an optional extra on
            purpose -- somebody who only ever types ``llamafit recommend`` should not be
            made to install a web server -- so this is a normal thing to meet, and it gets
            the command that fixes it rather than an ``ImportError``.
    """
    try:
        from llamafit.web.api import create_app
    except ImportError as exc:  # fastapi is an optional extra
        raise NotInstalledError(
            _("the web dashboard needs FastAPI and uvicorn, which are not installed"),
            hint=_('Install them with: pip install "llamafit[web]"'),
            command=str(exc),
        ) from exc
    extra = () if is_loopback(host) else (host,)
    return create_app(extra_hosts=extra)


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
    allow_remote: bool = False,
    runner: Any = None,
) -> None:
    """Run the dashboard until it is stopped.

    Args:
        host: The address to bind to.
        port: The port to listen on.
        open_browser: Open the page in the default browser once the server is up.
        allow_remote: Consent to a non-loopback address. The command line sets this only
            when ``--host`` was actually typed, so an address that arrived from a default
            can never be remote.
        runner: What actually serves, for tests; ``uvicorn.run`` when omitted.

    Raises:
        ConfigError: ``host`` is not loopback and ``allow_remote`` was not given.
        NotInstalledError: The ``web`` extra is not installed.
    """
    if not is_loopback(host) and not allow_remote:
        raise ConfigError(
            _("%(host)s is not a loopback address, and the dashboard has no authentication")
            % {"host": host},
            hint=_("Leave --host alone to serve this machine only, or pass it to mean it."),
        )
    app = build_app(host)
    if open_browser:
        _open_when_up(f"http://{_addressable(host)}:{port}/")
    serve_with = runner if runner is not None else _uvicorn_run
    serve_with(app, host=host, port=port)


def _addressable(host: str) -> str:
    """A host a browser can be pointed at.

    ``0.0.0.0`` and ``::`` mean "every interface" to a listening socket and nothing at all
    to a client, so the browser is sent to loopback, which is one of the interfaces the
    server is now on.
    """
    if host in {"0.0.0.0", "::", "[::]"}:  # compared, never bound to
        return DEFAULT_HOST
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _open_when_up(url: str) -> None:
    """Open the browser a moment after the server starts listening.

    ``uvicorn.run`` does not return until the server stops, so the browser has to be
    opened from somewhere else. A short-lived timer thread is that somewhere: half a
    second is longer than a local bind takes, and a browser that arrives fractionally
    early retries on its own.
    """
    threading.Timer(0.5, _open, args=(url,)).start()


def _open(url: str) -> None:
    """Open a URL in the default browser, logging rather than failing when there is none."""
    try:
        webbrowser.open(url)
    except Exception as exc:  # a headless machine has no browser; that is not a failure
        _log.debug("could not open a browser for %s: %s", url, exc)


def _uvicorn_run(app: FastAPI, *, host: str, port: int) -> None:
    """Hand the application to uvicorn.

    Imported here rather than at module scope so that ``llamafit --help`` on an
    installation without the ``web`` extra does not fail on an import nobody asked for.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
