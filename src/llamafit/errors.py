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
        """Return the message with its hint and command on separate lines.

        The message and the hint arrive already translated, from wherever the failure
        was raised; what this adds is the two labels in front of them.
        """
        # Imported here rather than at module scope because the translation layer raises
        # ConfigError, so llamafit.i18n imports this module: asking for the translator at
        # module scope would be a cycle. Render time is when the language is known anyway.
        from llamafit.i18n import _

        lines = [str(self.message)]
        if self.command:
            lines.append(_("Command: %(command)s") % {"command": self.command})
        if self.hint:
            lines.append(_("Hint: %(hint)s") % {"hint": self.hint})
        return "\n".join(lines)


class ProbeError(LlamaFitError):
    """A hardware or software probe could not run or returned unusable output."""


class ConfigError(LlamaFitError):
    """A configuration file or option is invalid."""


class CatalogError(LlamaFitError):
    """A catalog file is invalid or a model id is unknown."""


class BudgetError(LlamaFitError):
    """A memory budget cannot be computed from what is known about a model.

    Raised rather than returning a budget with a hole in it. A missing component is not a
    component of zero bytes, and a budget that quietly left the key-value cache out
    because the file never said how many key/value heads it has would report a model
    fitting a card it would page off. The caller catches this and records why the
    candidate was excluded.
    """


class NetworkError(LlamaFitError):
    """A network operation failed or the network is unavailable."""


class NotInstalledError(LlamaFitError):
    """A required external program (for example llama.cpp) is not installed."""


class PackagedDataError(LlamaFitError):
    """Data that ships inside the package is missing from this installation.

    Everything under ``llamafit/data/`` is read through ``importlib.resources`` at run
    time, so a wheel built without one of those directories installs perfectly and fails
    only when a command reaches the data it needs. That is a broken installation, not
    something the reader did or can edit their way out of, which is why this is the one
    error whose hint is "reinstall".
    """
