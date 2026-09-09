"""Error hierarchy. Every user-facing failure is a ``LlamaFitError``.

Interfaces catch ``LlamaFitError`` and print ``render()``; anything else is a bug
and is shown with a traceback only under ``--verbose``.
"""

from __future__ import annotations


class LlamaFitError(Exception):
    """Base class for failures LlamaFit knows how to explain.

    Args:
        message: What went wrong, in one sentence.
        hint: What the user can do about it, if anything.
        command: The exact external command that failed, if one did.
    """

    def __init__(self, message: str, *, hint: str | None = None, command: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.command = command

    def render(self) -> str:
        """Return the message with its hint and command on separate lines."""
        lines = [self.message]
        if self.command:
            lines.append(f"Command: {self.command}")
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        return "\n".join(lines)


class ProbeError(LlamaFitError):
    """A hardware or software probe could not run or returned unusable output."""


class ConfigError(LlamaFitError):
    """A configuration file or option is invalid."""


class CatalogError(LlamaFitError):
    """A catalog file is invalid or a model id is unknown."""


class NetworkError(LlamaFitError):
    """A network operation failed or the network is unavailable."""


class NotInstalledError(LlamaFitError):
    """A required external program (for example llama.cpp) is not installed."""
