# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Giving the connection back: a rate limit, and a ceiling on requests in flight.

Two different politenesses live here, and they answer two different people.

:class:`RateLimiter` answers the user. A download that takes the whole line for six hours
makes a video call unusable in the next room, and the person who started it has no way to
say "use half". ``--limit-rate`` is that way. One bucket is shared by every worker, so the
figure is the figure whatever the concurrency is.

:class:`Governor` answers the server. It caps how many requests are in flight and, when the
server says 429 or 503, gives a permit up permanently rather than merely sleeping. Backing
off in time alone leaves the same sixteen workers arriving together after the delay; giving
up a permit is the only thing that actually reduces the pressure.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """A leaky bucket shared by every worker, or no limit at all.

    The clock and the sleep are injected so a test can prove the arithmetic without
    spending the seconds. ``take`` blocks the calling thread and nothing else: workers
    that are already writing to disk carry on.
    """

    def __init__(
        self,
        bytes_per_second: int | None,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Set the rate, or ``None`` for no limit at all.

        Args:
            bytes_per_second: The ceiling, or ``None`` to let every ``take`` through.
            monotonic: A clock, for tests.
            sleep: How to wait, for tests. Given a duration in seconds.
        """
        self._rate = bytes_per_second if bytes_per_second and bytes_per_second > 0 else None
        self._now = monotonic if monotonic is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._lock = threading.Lock()
        self._free_at = self._now()

    @property
    def bytes_per_second(self) -> int | None:
        """The ceiling in force, or ``None`` when there is none."""
        return self._rate

    def take(self, count: int) -> float:
        """Wait until ``count`` bytes may be transferred, and say how long that took.

        The reservation is made under the lock and the waiting is done outside it, so a
        worker that has to wait is not holding every other worker up while it does.

        Args:
            count: How many bytes are about to be transferred.

        Returns:
            The seconds this call waited, which is zero when there is no limit.
        """
        if self._rate is None or count <= 0:
            return 0.0
        with self._lock:
            now = self._now()
            start = max(now, self._free_at)
            self._free_at = start + count / self._rate
            delay = start - now
        if delay > 0:
            self._sleep(delay)
        return delay


class Governor:
    """How many requests may be in flight, lowered when the server pushes back.

    Starts at the worker count and never falls below one, so a server that rate-limits
    everything still gets its files fetched, one request at a time, instead of the whole
    command failing. A permit given up is never taken back during the run: a server that
    said 429 once will say it again, and creeping back up to sixteen workers to find out
    is exactly the rudeness this is here to stop.
    """

    def __init__(self, capacity: int) -> None:
        """Start with ``capacity`` permits, at least one."""
        self._capacity = max(1, capacity)
        self._semaphore = threading.Semaphore(self._capacity)
        self._lock = threading.Lock()
        self._retire = 0

    @property
    def capacity(self) -> int:
        """How many requests may be in flight now."""
        with self._lock:
            return self._capacity

    def acquire(self) -> None:
        """Wait for a permit."""
        self._semaphore.acquire()

    def release(self) -> None:
        """Give the permit back, unless :meth:`back_off` asked for one to be retired."""
        with self._lock:
            if self._retire > 0:
                self._retire -= 1
                self._capacity -= 1
                return
        self._semaphore.release()

    def back_off(self) -> bool:
        """Ask for one fewer request in flight from now on.

        Returns:
            Whether the ceiling will actually come down. ``False`` once it is at one,
            which is as polite as this can get.
        """
        with self._lock:
            if self._capacity - self._retire <= 1:
                return False
            self._retire += 1
            return True
