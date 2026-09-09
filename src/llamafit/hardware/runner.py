"""Run external programs in a way tests can replace.

Every probe goes through a ``Runner`` so a test can feed recorded output from a real
machine instead of executing anything. ``SubprocessRunner`` never raises: a missing
program, a crash or a timeout all become a ``CommandResult`` with ``error`` set.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from llamafit.i18n import _
from llamafit.logging import get_logger
from llamafit.models.host import Probe

T = TypeVar("T")
_log = get_logger("hardware.runner")


@dataclass
class CommandResult:
    """Everything a probe needs to know about one command execution."""

    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the command ran and exited with status 0."""
        return self.error is None and self.returncode == 0


class Runner(Protocol):
    """Something that can run a command line and report what happened."""

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run ``argv`` and return its result without raising."""
        ...


class SubprocessRunner:
    """Runs real subprocesses with a timeout, capturing text output."""

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run ``argv``; a missing program or a timeout is reported, not raised."""
        args = list(argv)
        if not args:
            return CommandResult(args, None, "", "", 0, error=_("empty command"))
        start = time.perf_counter()
        _log.debug("running %s", " ".join(args))
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            return CommandResult(
                args,
                None,
                "",
                "",
                _elapsed_ms(start),
                error=_("%(program)s: not found") % {"program": args[0]},
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                args,
                None,
                "",
                "",
                _elapsed_ms(start),
                error=_("%(program)s: timed out after %(seconds)ss")
                % {"program": args[0], "seconds": timeout},
            )
        except OSError as exc:
            return CommandResult(
                args,
                None,
                "",
                "",
                _elapsed_ms(start),
                error=_("%(program)s: %(error)s") % {"program": args[0], "error": exc},
            )
        return CommandResult(
            args,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            _elapsed_ms(start),
        )


@dataclass
class FakeRunner:
    """Returns canned output; keys are a program name or a full command line."""

    responses: Mapping[str, str | CommandResult]
    calls: list[list[str]] = field(default_factory=list)

    def run(self, argv: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Look up the command by full line first, then by program name."""
        args = list(argv)
        self.calls.append(args)
        if not args:
            return CommandResult(args, None, "", "", 0, error=_("empty command"))
        response = self.responses.get(" ".join(args))
        if response is None:
            response = self.responses.get(args[0])
        if response is None:
            # The same two messages the real runner produces, so a recorded machine
            # reads exactly like the machine it was recorded from, in every language.
            return CommandResult(
                args, None, "", "", 0, error=_("%(program)s: not found") % {"program": args[0]}
            )
        if isinstance(response, CommandResult):
            return response
        return CommandResult(args, 0, response, "", 0)


def probe(
    name: str,
    runner: Runner,
    argv: Sequence[str],
    parse: Callable[[str], T],
    *,
    timeout: float = 10.0,
) -> tuple[T | None, Probe]:
    """Run a command and parse its stdout, turning every failure into a ``Probe`` record.

    Returns:
        The parsed value (or ``None``) and the probe record for ``doctor``.
    """
    result = runner.run(argv, timeout=timeout)
    if result.error is not None:
        _log.debug("probe %s failed: %s", name, result.error)
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=result.error)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or _("no output")
        error = _("exit code %(code)d: %(detail)s") % {
            "code": result.returncode,
            "detail": detail,
        }
        _log.debug("probe %s failed: %s", name, error)
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=error)
    try:
        value = parse(result.stdout)
    except Exception as exc:  # any parse failure must become a probe record
        _log.debug("probe %s failed: %s", name, exc)
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=str(exc))
    _log.debug("probe %s: ok in %d ms", name, result.duration_ms)
    return value, Probe(name=name, ok=True, duration_ms=result.duration_ms)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
