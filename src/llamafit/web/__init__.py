# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The dashboard: a JSON API over the services, and a static page that reads it.

Two halves, and the split is the point. :mod:`llamafit.web.api` serialises the same
pydantic models ``--json`` prints, by calling the same service functions the commands
call and computing nothing of its own; ``static/`` is plain HTML, CSS and JavaScript
that fetches those documents and draws them. Neither half knows a formula.

Everything ships inside the package. There is no build step, no package manager and no
framework: goal one of the specification is ``pip install llamafit`` with no compiler and
no Node, and a dashboard that needed either would take that away. Nothing on the page is
fetched from the internet at runtime, either — no font, no script, no stylesheet — so it
works with the network unplugged and opening it tells nobody that you did.

It binds to the loopback address. Section 1.3's non-goals put remote hosts, multi-user
access and authentication outside this project, and :func:`llamafit.web.server.serve`
enforces that rather than merely defaulting to it.

Only :mod:`llamafit.web.server` is imported here. FastAPI and uvicorn are an optional
extra, and importing :mod:`llamafit.web.api` from this file would turn "the dashboard is
not installed" into an ``ImportError`` raised by simply naming the package.
"""

from llamafit.web.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    build_app,
    is_loopback,
    remote_warning,
    serve,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "build_app",
    "is_loopback",
    "remote_warning",
    "serve",
]
