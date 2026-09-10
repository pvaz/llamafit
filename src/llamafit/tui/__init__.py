# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The terminal dashboard: what ``llamafit`` opens when it is given nothing to do.

Section 13.2. Five screens over one :class:`~llamafit.tui.state.Dashboard`, which is the
only thing in the package that calls a service; every screen reads what the services
answered and none of them works anything out.

Nothing here imports Textual. ``llamafit.cli.app`` imports this module on its way to every
command, and a full user-interface framework loaded to print a version string is a cost
every command would pay for one of them. :func:`~llamafit.tui.entry.open_dashboard` is
what pulls the application in, at the moment somebody actually asks for it.
"""

from __future__ import annotations

from llamafit.tui.entry import open_dashboard, run_dashboard
from llamafit.tui.state import Dashboard, Request, Substitution

__all__ = ["Dashboard", "Request", "Substitution", "open_dashboard", "run_dashboard"]
