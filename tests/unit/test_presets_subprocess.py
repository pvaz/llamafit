"""The real launcher, against harmless Python processes rather than a language model.

Everything else about launching is tested through :class:`~llamafit.presets.FakeLauncher`,
which is the point of the protocol. These few tests exercise the implementation instead,
because two of the things it does cannot be checked any other way and both of them are the
kind that go wrong quietly.

**A process id is not an identity.** Ids are recycled, and a note that held only a number
would be a licence to stop whatever inherited it. The record carries the start time too and
every use checks both, which is only provable against a process that really exists.

**Stopping means the tree.** A preset is run through ``cmd`` or ``sh``, so the id LlamaFit
holds is the shell's and ``llama-server`` is its child. Terminating just the recorded id
would leave a server running with nothing left that knows about it -- the exact failure
``--stop`` exists to prevent.

Nothing here starts llama.cpp, downloads anything or touches a network: the subjects are
``python -c`` sleepers, and each test stops what it started.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from llamafit.presets import Started, SubprocessLauncher

SLEEPER = "import time; time.sleep(60)"
SPAWNER = (
    "import pathlib, subprocess, sys, time; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
    "time.sleep(60)"
)


@pytest.fixture
def launcher() -> Iterator[SubprocessLauncher]:
    """A real launcher, with whatever it started stopped again afterwards."""
    started: list[Started] = []
    real = SubprocessLauncher()
    original = real.start

    def remember(argv: list[str], *, cwd: Path | None = None) -> Started:
        result = original(argv, cwd=cwd)
        started.append(result)
        return result

    real.start = remember  # type: ignore[method-assign]
    yield real
    for process in started:
        real.stop(process.pid, process.created_at)


def test_a_started_process_is_running_and_can_be_stopped(
    launcher: SubprocessLauncher,
) -> None:
    started = launcher.start([sys.executable, "-c", SLEEPER])
    assert launcher.is_running(started.pid, started.created_at)
    assert launcher.stop(started.pid, started.created_at)
    _wait_until_gone(started.pid)
    assert not launcher.is_running(started.pid, started.created_at)


def test_a_note_whose_start_time_does_not_match_is_not_acted_on(
    launcher: SubprocessLauncher,
) -> None:
    """This is the guard against a recycled process id, and it has to fail closed."""
    started = launcher.start([sys.executable, "-c", SLEEPER])
    stale = started.created_at + 3600
    assert not launcher.is_running(started.pid, stale)
    assert not launcher.stop(started.pid, stale)
    assert launcher.is_running(started.pid, started.created_at)


def test_a_process_that_has_already_gone_is_not_running_and_cannot_be_stopped(
    launcher: SubprocessLauncher,
) -> None:
    started = launcher.start([sys.executable, "-c", "pass"])
    _wait_until_gone(started.pid)
    assert not launcher.is_running(started.pid, started.created_at)
    assert not launcher.stop(started.pid, started.created_at)


def test_stopping_takes_the_children_with_it(launcher: SubprocessLauncher, tmp_path: Path) -> None:
    """The preset is run through a shell, so the server is always somebody's child."""
    note = tmp_path / "child.pid"
    started = launcher.start([sys.executable, "-c", SPAWNER, str(note)])
    child = _wait_for_pid(note)
    assert psutil.pid_exists(child)
    assert launcher.stop(started.pid, started.created_at)
    _wait_until_gone(child)
    assert not psutil.pid_exists(child) or not psutil.Process(child).is_running()


def test_a_program_that_does_not_exist_raises_rather_than_reporting_a_process() -> None:
    with pytest.raises(OSError):
        SubprocessLauncher().start(["llamafit-no-such-program-anywhere"])


def test_the_started_process_is_the_one_that_was_asked_for(
    launcher: SubprocessLauncher,
) -> None:
    started = launcher.start([sys.executable, "-c", SLEEPER])
    assert started.argv == (sys.executable, "-c", SLEEPER)
    assert psutil.Process(started.pid).create_time() == pytest.approx(started.created_at, abs=0.01)


def _wait_until_gone(pid: int, *, timeout: float = 15.0) -> None:
    """Wait for a process to finish, reaping it so it does not linger as a zombie."""
    try:
        psutil.Process(pid).wait(timeout=timeout)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        return
    except subprocess.TimeoutExpired:  # pragma: no cover - psutil re-exports on some OSes
        return


def _wait_for_pid(note: Path, *, timeout: float = 15.0) -> int:
    """Wait for the spawner to write down the id of the child it started."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if note.is_file() and note.read_text(encoding="utf-8").strip():
            return int(note.read_text(encoding="utf-8").strip())
        time.sleep(0.05)
    raise AssertionError(f"{note} was never written")
