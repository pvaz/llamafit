# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
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

from llamafit.i18n import _, ngettext
from llamafit.logging import get_logger
from llamafit.models.host import Probe
from llamafit.units import localise_number

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


def _timed_out(program: str, seconds: float) -> str:
    """The message a command that ran out of time reports.

    The unit is a word in the sentence and not a letter welded to the number. Welded, a
    language that spells *seconds* out had to decide on its own whether it was allowed to
    drop the ``s``, and in Arabic or Hebrew that trailing Latin letter sat at the seam
    between two writing directions with nothing saying which side it belonged to.

    A timeout is a duration and not a count of things, so it can be fractional. The form
    is selected on the nearest whole second, while the number printed is the exact one.
    """
    return ngettext(
        "%(program)s: timed out after %(seconds)s second",
        "%(program)s: timed out after %(seconds)s seconds",
        round(seconds),
    ) % {"program": program, "seconds": localise_number(f"{seconds:g}")}


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
                error=_timed_out(args[0], timeout),
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
    include_stderr: bool = False,
) -> tuple[T | None, Probe]:
    """Run a command and parse its output, turning every failure into a ``Probe`` record.

    Args:
        name: What ``doctor`` calls this probe.
        runner: How to run it.
        argv: The command and its arguments.
        parse: Reads the output. Raising is how a parser says it found nothing; a parser
            that returns a hollow value instead leaves a probe reported ``ok`` after it
            learned nothing, which is the shape of every "the tool said it worked" bug.
        timeout: Seconds before the command is abandoned.
        include_stderr: Read the error stream as well as the output stream. Off by
            default, because most tools answer on stdout and stderr is where their noise
            goes. On for the ones that do not: llama.cpp prints its own version banner to
            stderr and nothing at all to stdout, which is why its build number read as
            unknown on every machine whose llama.cpp LlamaFit had not installed itself.

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
    text = "\n".join((result.stdout, result.stderr)) if include_stderr else result.stdout
    try:
        value = parse(text)
    except Exception as exc:  # any parse failure must become a probe record
        _log.debug("probe %s failed: %s", name, exc)
        return None, Probe(name=name, ok=False, duration_ms=result.duration_ms, error=str(exc))
    _log.debug("probe %s: ok in %d ms", name, result.duration_ms)
    return value, Probe(name=name, ok=True, duration_ms=result.duration_ms)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
