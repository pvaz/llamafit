# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The two widgets the screens share: a pane holding a Rich renderable, and the top band."""

from __future__ import annotations

from llamafit.tui.widgets.machine_bar import MachineBar
from llamafit.tui.widgets.rich_pane import RichPane

__all__ = ["MachineBar", "RichPane"]
