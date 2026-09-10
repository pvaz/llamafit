# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Run a preset, wait for ``/health``, and be able to stop what was started.

The preset script is what is run, never the flags directly. That is the whole point: the
script is where the context ladder lives, and a ``launch`` that rebuilt the command line
for itself would launch a configuration nobody had chosen at start time -- and would
quietly ignore whatever the user had tuned in the file.

Two seams, both cut the way the rest of the project cuts them. Starting a process goes
through :class:`Launcher`, the way every probe goes through
:class:`~llamafit.hardware.runner.Runner`, so a test substitutes :class:`FakeLauncher` and
nothing is spawned. Asking a server whether it is up goes through
:class:`~llamafit.llamacpp.server.HttpClient`, so a test substitutes
:class:`~llamafit.llamacpp.server.FakeHttp` and nothing is dialled. The clock is injected
for the same reason: a test that waits two minutes for a timeout is a test nobody runs.

``Runner`` itself is the wrong shape here and could not be reused. It runs a command *to
completion* and hands back its output, which is exactly right for ``nvidia-smi`` and
exactly wrong for a server meant to outlive the call.

A started server is remembered in a small file, because ``--stop`` runs in a later process
that has no memory of the earlier one. What is written down is the process id **and its
start time**: process ids are recycled, and a record holding only a number is a licence to
kill whatever inherited it. Every use of a record checks both.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import psutil
from pydantic import BaseModel, ConfigDict

from llamafit.llamacpp.server import HEALTH_TIMEOUT_S, HttpClient
from llamafit.logging import get_logger
from llamafit.paths import get_paths

_log = get_logger("presets.launch")

_PORT_RE = re.compile(r"--port\s+\"?(\d{1,5})")
"""How the port is found again in a rendered -- or hand-edited -- preset script."""

START_TIME_TOLERANCE_S = 1.0
"""How far a process's reported start time may differ from the recorded one.

Not zero. ``psutil`` reads a start time from the operating system with a resolution that
differs between platforms -- and on Linux it is derived from the boot time, which itself
drifts as the clock is disciplined. A second is far tighter than the recycling of a process
id and far looser than that drift.
"""

HEALTH_POLL_INTERVAL_S = 0.5
"""How often to ask a starting server whether it is ready."""

DEFAULT_HEALTH_TIMEOUT_S = 180.0
"""How long to wait for one.

Long, because the wait is dominated by reading the weights off the disk: a twenty-gigabyte
file on a mechanical disk is minutes, and a timeout that expired first would report a
healthy server as a failure and then leave it running.
"""


@dataclass(frozen=True)
class Started:
    """A process that has been started and not yet waited for.

    Attributes:
        pid: The process id.
        created_at: Its start time as the operating system reports it, which is what makes
            the pid safe to act on later.
        argv: What was run, for the log and for an error message.
    """

    pid: int
    created_at: float
    argv: tuple[str, ...]


class Launcher(Protocol):
    """Something that can start a long-running process and later stop it."""

    def start(self, argv: Sequence[str], *, cwd: Path | None = None) -> Started:
        """Start ``argv`` and return at once, without waiting for it."""
        ...

    def is_running(self, pid: int, created_at: float) -> bool:
        """Whether that exact process -- id and start time -- is still alive."""
        ...

    def stop(self, pid: int, created_at: float) -> bool:
        """Stop that exact process and everything it started; True when it is gone."""
        ...


class SubprocessLauncher:
    """Starts real processes, detached enough to survive the command that started them."""

    def start(self, argv: Sequence[str], *, cwd: Path | None = None) -> Started:
        """Start ``argv`` in its own process group, so Ctrl+C here does not stop it.

        Args:
            argv: The command, program first.
            cwd: The directory to start it in.

        Returns:
            The process id and start time.

        Raises:
            OSError: If the program cannot be started at all.

        A new process group on Windows and a new session on POSIX are the same intent
        spelled twice: the server is meant to keep running after ``llamafit launch``
        returns, and a Ctrl+C aimed at the terminal must not reach it by accident.
        """
        args = list(argv)
        _log.debug("starting %s", " ".join(args))
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if new_group:
            process = subprocess.Popen(args, cwd=cwd, creationflags=new_group)
        else:
            process = subprocess.Popen(args, cwd=cwd, start_new_session=True)
        return Started(pid=process.pid, created_at=_created_at(process.pid), argv=tuple(args))

    def is_running(self, pid: int, created_at: float) -> bool:
        """Whether the process with this id *and* this start time is still alive."""
        return _same_process(pid, created_at) is not None

    def stop(self, pid: int, created_at: float) -> bool:
        """Ask the process tree to stop, then insist.

        The children matter. On Windows the preset is run through ``cmd``, so the process
        id LlamaFit holds is the shell's and ``llama-server`` is its child; terminating
        only the recorded id would leave a server running with nothing left that knows
        about it.
        """
        process = _same_process(pid, created_at)
        if process is None:
            return False
        family = _family(process)
        for member in family:
            _ask_to_stop(member)
        _gone, alive = psutil.wait_procs(family, timeout=5)
        for member in alive:
            _insist(member)
        return True


@dataclass
class FakeLauncher:
    """Records what it was asked to start and pretends it worked.

    Attributes:
        pid: The process id to hand back.
        created_at: The start time to hand back.
        alive: Whether :meth:`is_running` says yes.
        fail_with: An error to raise from :meth:`start`, for the path where the script
            cannot be run at all.
        started: Every command it was asked to start.
        stopped: Every ``(pid, created_at)`` it was asked to stop.
    """

    pid: int = 4242
    created_at: float = 1_000_000.0
    alive: bool = True
    fail_with: OSError | None = None
    started: list[list[str]] = field(default_factory=list)
    stopped: list[tuple[int, float]] = field(default_factory=list)

    def start(self, argv: Sequence[str], *, cwd: Path | None = None) -> Started:
        """Record the command and return the canned process."""
        self.started.append(list(argv))
        if self.fail_with is not None:
            raise self.fail_with
        return Started(pid=self.pid, created_at=self.created_at, argv=tuple(argv))

    def is_running(self, pid: int, created_at: float) -> bool:
        """Whether this fake says the process is alive."""
        return self.alive and pid == self.pid and created_at == self.created_at

    def stop(self, pid: int, created_at: float) -> bool:
        """Record the request and report whether there was anything to stop."""
        self.stopped.append((pid, created_at))
        was_alive = self.is_running(pid, created_at)
        self.alive = False
        return was_alive


class RunningPreset(BaseModel):
    """What LlamaFit remembers about a server it started.

    Attributes:
        model_id: Which preset was launched.
        pid: The process id.
        created_at: Its start time, which is what makes the id safe to act on later.
        url: Where it answers.
        script: The preset script that was run.
        started_at: When LlamaFit started it, for the reader rather than for the check.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    pid: int
    created_at: float
    url: str
    script: str
    started_at: datetime


Result = Literal["started", "already-running", "no-health", "failed-to-start"]
"""How a launch turned out.

``no-health`` is separate from ``failed-to-start`` because they need different sentences:
one is a process that never appeared, the other is a process that is running and has not
answered, which on a large model usually means it is still reading weights.
"""


@dataclass(frozen=True)
class LaunchOutcome:
    """What ``launch`` did, for an interface to report.

    Attributes:
        result: Which of the four things happened.
        url: The endpoint, whether or not it answered.
        record: What was written down, when something was started.
        error: The reason the process could not be started, when that is what happened.
    """

    result: Result
    url: str
    record: RunningPreset | None = None
    error: str | None = None


def records_dir() -> Path:
    """Where the notes about running servers are kept."""
    return get_paths().data_dir / "running"


def record_path(model_id: str) -> Path:
    """The note for one model."""
    return records_dir() / f"{model_id}.json"


def save_record(record: RunningPreset) -> None:
    """Write the note, creating the directory if this is the first one."""
    path = record_path(record.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_record(model_id: str) -> RunningPreset | None:
    """The note for one model, or ``None`` when there is none or it cannot be read.

    A note that cannot be parsed is treated as no note. It is a cache of a fact about a
    process, not a source of truth: the process either exists or it does not, and refusing
    to run because a file from an older version has a field this one does not know would be
    a failure invented out of nothing.
    """
    path = record_path(model_id)
    try:
        return RunningPreset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_record(model_id: str) -> None:
    """Forget one note, if there is one."""
    record_path(model_id).unlink(missing_ok=True)


def script_command(script: Path) -> list[str]:
    """How to run a preset script on the platform whose script it is.

    A ``.cmd`` is not an executable and cannot be started directly; ``cmd /c`` is the thing
    that reads it. A ``.sh`` is run through ``/bin/sh`` rather than executed, so that a
    preset written onto a file system with no execute bit -- a FAT stick, a mounted share --
    still runs.
    """
    if script.suffix.lower() == ".cmd":
        return ["cmd", "/c", str(script)]
    return ["/bin/sh", str(script)]


def port_of_script(text: str) -> int | None:
    """The port a preset script will bind, read out of the script itself.

    Args:
        text: The script's contents.

    Returns:
        The port, or ``None`` when the script names none.

    Read rather than assumed, because the file is meant to be edited. Somebody who moved
    the port in their own script would otherwise have ``llamafit launch`` start the server
    and then poll an address nothing is listening on, and report a healthy server as one
    that never came up. The last occurrence wins: :mod:`llamafit.placement.flags` renders
    ``--port`` once, and anything after it is a hand edit, which is the one that counts.
    """
    matches = _PORT_RE.findall(text)
    return int(matches[-1]) if matches else None


def wait_for_health(
    http: HttpClient,
    url: str,
    *,
    timeout: float = DEFAULT_HEALTH_TIMEOUT_S,
    interval: float = HEALTH_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``/health`` until it says ok or the time runs out.

    Args:
        http: The client to ask with.
        url: The server's base URL.
        timeout: How long to keep asking.
        interval: How long to wait between asks.
        sleep: How to wait, injected so a test does not.
        monotonic: How to tell the time, injected for the same reason. Monotonic and not
            the wall clock: a machine that adjusts its clock mid-wait would otherwise
            either time out at once or never.

    Returns:
        True when the server answered ``{"status": "ok"}`` within the time.
    """
    deadline = monotonic() + timeout
    while True:
        payload = http.get_json(f"{url}/health", timeout=HEALTH_TIMEOUT_S)
        if isinstance(payload, dict) and payload.get("status") == "ok":
            return True
        if monotonic() >= deadline:
            return False
        sleep(interval)


def is_answering(http: HttpClient, url: str) -> bool:
    """Whether something is already serving on this endpoint, asked once and quickly."""
    payload = http.get_json(f"{url}/health", timeout=HEALTH_TIMEOUT_S)
    return isinstance(payload, dict) and payload.get("status") == "ok"


def launch_preset(
    model_id: str,
    script: Path,
    url: str,
    *,
    launcher: Launcher,
    http: HttpClient,
    timeout: float = DEFAULT_HEALTH_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> LaunchOutcome:
    """Run a preset and wait for the server it starts to answer.

    Args:
        model_id: Which preset this is, for the note that is written down.
        script: The preset script to run.
        url: Where the server will answer, which the preset itself fixed.
        launcher: How to start it.
        http: How to ask whether it is up.
        timeout: How long to wait for ``/health``.
        sleep: How to wait between asks.
        monotonic: How to tell the time.
        now: What to stamp the note with.

    Returns:
        What happened, with the note when something was started.

    A server already answering on that endpoint is reported rather than joined by a second
    one. Two ``llama-server`` processes on one port is a state where the second fails on a
    bound socket and the first keeps serving, and a launcher that produced it without
    saying so would leave a person believing they had just started what they are talking to.
    """
    if is_answering(http, url):
        return LaunchOutcome("already-running", url, record=load_record(model_id))
    try:
        started = launcher.start(script_command(script), cwd=script.parent)
    except OSError as exc:
        return LaunchOutcome("failed-to-start", url, error=str(exc))
    record = RunningPreset(
        model_id=model_id,
        pid=started.pid,
        created_at=started.created_at,
        url=url,
        script=str(script),
        started_at=now(),
    )
    save_record(record)
    healthy = wait_for_health(http, url, timeout=timeout, sleep=sleep, monotonic=monotonic)
    return LaunchOutcome("started" if healthy else "no-health", url, record=record)


def stop_preset(model_id: str, *, launcher: Launcher) -> RunningPreset | None:
    """Stop the server LlamaFit started for this model.

    Args:
        model_id: Which preset.
        launcher: How to stop it.

    Returns:
        The note describing what was stopped, or ``None`` when there was nothing to stop.

    The note is cleared either way. A note for a process that is gone is worse than no note:
    it is a process id somebody's next ``--stop`` would aim at.
    """
    record = load_record(model_id)
    if record is None:
        return None
    stopped = launcher.stop(record.pid, record.created_at)
    clear_record(model_id)
    return record if stopped else None


def _created_at(pid: int) -> float:
    """A process's start time, or zero when it cannot be read.

    Zero means "do not check", and it is reached only when the process has already exited
    between being started and being asked about -- a program that fails immediately. There
    is then nothing to protect from a recycled id, because there is nothing to stop.
    """
    try:
        return float(psutil.Process(pid).create_time())
    except psutil.Error:
        return 0.0


def _same_process(pid: int, created_at: float) -> psutil.Process | None:
    """The process with this id, but only if it is the one the id was written down for."""
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            # A zombie is an exit nobody has collected yet, not a server: psutil calls it
            # running because the entry is still in the table. Anything relying on that
            # would report a process that has finished as one it could still stop.
            return None
        if created_at and abs(process.create_time() - created_at) > START_TIME_TOLERANCE_S:
            return None
    except psutil.Error:
        return None
    return process


def _family(process: psutil.Process) -> list[psutil.Process]:
    """A process and everything it started, children first so parents outlive them."""
    try:
        children = list(process.children(recursive=True))
    except psutil.Error:
        children = []
    return [*children, process]


def _ask_to_stop(process: psutil.Process) -> None:
    """Terminate one process, ignoring one that has already gone."""
    # Already gone is the outcome that was wanted, so it is not an error to report.
    with contextlib.suppress(psutil.Error):
        process.terminate()


def _insist(process: psutil.Process) -> None:
    """Kill one process that did not take the hint."""
    # Already gone is the outcome that was wanted, so it is not an error to report.
    with contextlib.suppress(psutil.Error):
        process.kill()
