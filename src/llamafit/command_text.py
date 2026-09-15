# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Copyable plan commands, distinct from the argv lists used to launch processes.

POSIX commands target ``sh``. Windows commands target PowerShell, including 5.1,
with literal quoting for model paths and llama.cpp's glob-like tensor patterns.
Legacy PowerShell still has native-argument limitations for empty strings and
embedded double quotes; this formatter does not change its argument-passing mode.
Preset scripts have their own dialect-specific renderers and do not use this module.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence
from typing import Literal

Shell = Literal["sh", "PowerShell"]


def command_shell() -> Shell:
    """The shell used for commands copied on the machine running LlamaFit."""
    return "PowerShell" if os.name == "nt" else "sh"


def render_command(command: Sequence[str], *, shell: Shell | None = None) -> str:
    """Protect a plan's paths and literal patterns in the named shell."""
    if not command:
        return ""
    if (shell or command_shell()) == "sh":
        return shlex.join(command)
    # PowerShell treats curly single quotes as delimiters too. The call operator
    # is required when the executable itself is a quoted path.
    escapes = {ord(quote): quote * 2 for quote in "'\u2018\u2019"}
    return "& " + " ".join("'" + argument.translate(escapes) + "'" for argument in command)
