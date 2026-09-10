"""The two politenesses: a rate limit for the user's line, a ceiling for the server."""

from __future__ import annotations

import threading

from llamafit.download.throttle import Governor, RateLimiter


class _Clock:
    """A clock a test moves by hand, so the arithmetic is proved without the waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self.advance_on_sleep = True
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.slept.append(seconds)
            if self.advance_on_sleep:
                self.now += seconds


def test_no_limit_never_waits() -> None:
    limiter = RateLimiter(None)
    assert limiter.bytes_per_second is None
    assert limiter.take(10**9) == 0.0


def test_zero_is_the_same_as_no_limit() -> None:
    assert RateLimiter(0).bytes_per_second is None


def test_the_first_take_goes_straight_through() -> None:
    clock = _Clock()
    limiter = RateLimiter(1000, monotonic=clock.monotonic, sleep=clock.sleep)
    assert limiter.take(1000) == 0.0
    assert clock.slept == []


def test_the_bucket_makes_each_further_take_wait_its_share() -> None:
    clock = _Clock()
    limiter = RateLimiter(1000, monotonic=clock.monotonic, sleep=clock.sleep)
    limiter.take(1000)
    assert limiter.take(1000) == 1.0
    assert limiter.take(500) == 1.0
    assert clock.slept == [1.0, 1.0]


def test_time_that_has_already_passed_is_credited() -> None:
    clock = _Clock()
    limiter = RateLimiter(1000, monotonic=clock.monotonic, sleep=clock.sleep)
    limiter.take(1000)
    clock.now += 10
    assert limiter.take(1000) == 0.0


def test_nothing_is_taken_for_nothing() -> None:
    clock = _Clock()
    limiter = RateLimiter(1000, monotonic=clock.monotonic, sleep=clock.sleep)
    assert limiter.take(0) == 0.0


def test_one_bucket_is_shared_by_every_worker() -> None:
    # Four threads asking for a second's worth each spend three seconds between them,
    # not none: the figure a user typed is the figure whatever the concurrency is.
    clock = _Clock()
    clock.advance_on_sleep = False
    limiter = RateLimiter(1000, monotonic=clock.monotonic, sleep=clock.sleep)
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        limiter.take(1000)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Three of the four wait; the first goes straight through and never calls sleep.
    assert sorted(clock.slept) == [1.0, 2.0, 3.0]


def test_a_governor_starts_at_its_capacity() -> None:
    assert Governor(8).capacity == 8


def test_a_governor_never_starts_below_one() -> None:
    assert Governor(0).capacity == 1


def test_backing_off_retires_a_permit_when_it_is_given_back() -> None:
    governor = Governor(4)
    governor.acquire()
    assert governor.back_off() is True
    assert governor.capacity == 4  # not yet: the permit is still out
    governor.release()
    assert governor.capacity == 3


def test_the_ceiling_stops_at_one_however_often_the_server_complains() -> None:
    governor = Governor(3)
    for _ in range(10):
        governor.acquire()
        governor.back_off()
        governor.release()
    assert governor.capacity == 1
    assert governor.back_off() is False


def test_a_permit_that_was_not_retired_comes_back() -> None:
    governor = Governor(2)
    governor.acquire()
    governor.release()
    assert governor.capacity == 2
    governor.acquire()
    governor.acquire()
    governor.release()
    governor.release()
    assert governor.capacity == 2
