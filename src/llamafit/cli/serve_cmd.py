# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit serve``: the dashboard and its JSON API, on this machine only.

The command is thin on purpose. Everything it knows is in :mod:`llamafit.web`; what
belongs here is the shape of the options and the one decision the option parser is the
only place that can make -- whether ``--host`` was *typed*. A default that happens to be
loopback and a loopback address somebody chose look identical by the time they reach
:func:`llamafit.web.server.serve`, so the consent to bind elsewhere is passed separately,
and only from the fact that the flag appeared.

``llamafit.web`` is imported inside the command rather than at the top of the file. The
web extra is optional, this module is imported every time anybody runs any command, and a
missing FastAPI must not break ``llamafit list`` for somebody who never wanted a server.
"""

from __future__ import annotations

from typing import cast

import typer
from rich.console import Console
from rich.text import Text

from llamafit.cli.app import CliState, app
from llamafit.cli.common import refuse_substitution
from llamafit.i18n import _, for_display, isolate, lazy_gettext

_HOST_OPTION: str | None = typer.Option(
    None,
    "--host",
    metavar="ADDRESS",
    help=cast(
        str,
        lazy_gettext(
            "Bind to this address instead of 127.0.0.1. Anything but loopback puts the "
            "dashboard on your network, where there is no password and no login; it "
            "prints what that exposes before it starts."
        ),
    ),
)
_PORT_OPTION: int = typer.Option(
    8765,
    "--port",
    min=1,
    max=65535,
    help=cast(str, lazy_gettext("Listen on this port.")),
)


@app.command(
    "serve",
    help=cast(
        str,
        lazy_gettext(
            "Open the dashboard in a browser: the board, the machine and the plans.\n\n"
            "It serves the same numbers `--json` prints, from the same code, on this "
            "machine only. Nothing is fetched from the internet to draw the page."
        ),
    ),
)
def serve_command(
    ctx: typer.Context,
    host: str | None = _HOST_OPTION,
    port: int = _PORT_OPTION,
    open_browser: bool = typer.Option(
        False, "--open", help=cast(str, lazy_gettext("Open the page in your browser."))
    ),
) -> None:
    """Open the dashboard in a browser: the board, the machine and the plans."""
    from llamafit.web import DEFAULT_HOST, is_loopback, remote_warning, serve

    state: CliState = ctx.obj
    # The page has a Simulate panel of its own, and the API takes the same four values per
    # request, so a global substitution here would set a default the panel then argues
    # with -- two places to say the same thing, disagreeing. Refusing says which one wins.
    refuse_substitution(
        state,
        command="serve",
        hint=_(
            "Drop --profile, --memory, --ram and --cpu-cores, then open the Simulate "
            "panel on the page: it substitutes a machine for as long as you want it, and "
            "marks every answer it produces."
        ),
    )
    address = host if host is not None else DEFAULT_HOST
    console = state.console
    if not is_loopback(address):
        # To stderr and in red: this is the one thing on the screen a reader must not
        # scroll past, and it must not join `--json` output on stdout.
        warning = Console(stderr=True, no_color=state.no_color, highlight=False)
        warning.print(Text(for_display(remote_warning(address, port)), style="bold red"))
    console.print(
        Text(
            for_display(
                _("LlamaFit is at %(url)s. Press Ctrl+C to stop it.")
                % {"url": isolate(f"http://{address}:{port}/")}
            )
        )
    )
    serve(
        host=address,
        port=port,
        open_browser=open_browser,
        # The address was typed, so binding away from loopback is a decision somebody
        # made rather than one that leaked in from a default.
        allow_remote=host is not None,
    )
