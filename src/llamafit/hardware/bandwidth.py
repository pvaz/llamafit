# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""RAM bandwidth: measure sequential read throughput, otherwise estimate or assume.

llama.cpp streams weights out of RAM during generation and writes almost nothing back, so
the number that predicts generation speed is sequential *read* bandwidth, not a copy figure.
A copy moves each byte twice, once read and once written (three times on a write-allocate
cache), so a copy benchmark that reports "bytes moved" as the buffer size once is reporting
roughly half (or a third) of the real memory traffic a read-only scan would see for the same
buffer, and the two numbers should not be compared directly. A single thread also cannot
saturate a multi-channel memory controller.

When NumPy is importable this module measures both properly: it splits a buffer larger than
any CPU cache into one slice per worker thread and runs a memory-bound reduction (``.max()``)
over each slice concurrently from a thread pool (NumPy releases the GIL during the reduction,
so this genuinely runs in parallel), reporting ``passes * total_bytes / elapsed`` as
``"measured"``.

Without NumPy, the fallback is still a single-threaded pure-Python ``bytearray`` slice
*copy*: it holds the GIL for the whole copy, so it cannot be parallelised across threads, and
because it is a copy rather than a read it under-reports the real read figure for two
independent reasons at once (single-threaded, and copy instead of read). Both are folded into
one documented ``PURE_PYTHON_CORRECTION``, and the result is labelled ``"estimated"`` rather
than ``"measured"``, since the correction is only an approximation. Install the optional
extra, ``pip install llamafit[fast]``, to bring in NumPy and get a real read-bandwidth
measurement instead.

**The figure is the best of several passes, and then it is kept.** One 50-millisecond
window run on every scan reported between 37 and 51 GB/s on the same machine inside an
hour, and every speed on every board is divided by it: the same question gave a different
answer, a different top row and a different tokens-per-second, minute to minute, with
nothing in the output admitting that anything had moved.

Two of the three things that could be done about that are done here, and the third is
deliberately not.

*Best of several, not longer, and not a mean.* What moves the number is not noise -- noise
averages out and a longer window would flatten it. It is contention: another process
holding the memory controller can only ever make this measurement come back slower than
the machine can go, never faster. So the readings have a hard ceiling at the truth and a
tail that runs away from it, the maximum of several is the estimator that walks towards
the machine while the mean walks towards how busy the machine happened to be, and
``BANDWIDTH_ROUNDS`` short windows beat one long one at the same cost. Running longer
would buy a more precise measurement of the wrong quantity.

*Cached per machine.* Best-of-five is a better single figure; it is not a still one, and
this is the half that actually stops the movement. Nothing about a memory controller
changes between two runs of a command, so the winning figure is written to the platform's
cache directory under a key built from the parts of the machine that could change the
answer -- the processor, the pool, and the modules' own type, speed and channel count --
and read back on every later scan. A figure that came back from the cache says so
(``Memory.bandwidth_cached``), and ``llamafit system --refresh-bandwidth`` times it again
and replaces it. Between them: one measurement per machine, taken as well as it can be.

*Free memory is left alone.* It is supposed to move between one run and the next; reading
it is the whole point of reading it. What that needed was for the board to say what it
used, which is :class:`~llamafit.models.host.MachineFacts`, not for anything here to hold
it still.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import psutil

from llamafit.logging import get_logger
from llamafit.models.host import Cpu, Memory, Source
from llamafit.paths import get_paths

_log = get_logger("hardware.bandwidth")

ASSUMED_RAM_BANDWIDTH_GBPS = 40.0
PLAUSIBLE_RANGE_GBPS = (5.0, 1000.0)
MAX_BANDWIDTH_WORKERS = 8
BANDWIDTH_ROUNDS = 5
"""Timed windows to run, keeping the fastest. See this module's own documentation.

Five rather than one because a window that happens to land while something else is on the
memory controller reads low and there is nothing in the window itself to say so; five
rather than twenty because the gain falls away as soon as one of them lands on a quiet
moment, and the quarter of a second is spent on a machine somebody is waiting at.

Measured on the reference machine, seven readings each. Idle, the old single window gave
49.6 to 55.7 GB/s and best-of-five gave 52.9 to 57.5 -- higher, as a maximum will be, and
no more spread out. With four threads deliberately copying half a gigabyte each in the
background, the single window gave 20.4 to 22.5 and best-of-five gave 22.4 to 24.7: about
a tenth of the machine recovered from whatever else was using it. Sustained contention
like that is the case neither of them can see through; what it cannot do is move the
answer from one run to the next, because by then the figure is cached.
"""

CACHE_FILE = "ram-bandwidth.json"
"""What the kept figures are written to, under the platform's cache directory."""
# The pure-Python fallback measures a single-threaded bytearray copy, not a multi-threaded
# read: it cannot be parallelised (it holds the GIL throughout) and a copy is not a read
# (it moves the data again on the way out, and reports only half of that traffic as "size").
# Measured on the reference machine: NumPy's multi-threaded read reduction averaged about
# 57 GB/s while the raw, uncorrected pure-Python copy averaged about 19 GB/s, a ratio of
# about 3x. Applied only on the no-NumPy fallback path (see the module docstring), which is
# labelled "estimated", not "measured", precisely because this correction is an approximation.
PURE_PYTHON_CORRECTION = 3.0


def _default_workers() -> int:
    """Physical core count capped at ``MAX_BANDWIDTH_WORKERS``, at least one."""
    return max(1, min(psutil.cpu_count(logical=False) or 1, MAX_BANDWIDTH_WORKERS))


def _time_passes(
    run: Callable[[], None],
    total_bytes: int,
    duration_s: float,
    *,
    correction: float,
    source: Source,
) -> tuple[float, Source] | None:
    """Run ``run`` repeatedly for at least ``duration_s`` seconds and report GB/s.

    ``run`` is expected to touch exactly ``total_bytes`` bytes each call, whether by
    copying or by reading them.

    Returns:
        The measured (or corrected) bandwidth and its source label, or ``None`` when
        no full pass completed.

    One window. :func:`_best_of` is what runs several of them and warms the buffer up
    first, so that the fault-in cost is paid once rather than charged to every window.
    """
    passes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        run()
        passes += 1
    elapsed = time.perf_counter() - start
    if passes == 0 or elapsed <= 0:
        return None
    return round(passes * total_bytes / elapsed / 1e9 * correction, 1), source


def _best_of(
    run: Callable[[], None],
    total_bytes: int,
    duration_s: float,
    *,
    correction: float,
    source: Source,
    rounds: int,
) -> tuple[float, Source] | None:
    """The fastest of ``rounds`` timed windows, which is the machine rather than its load.

    Args:
        run: One pass over exactly ``total_bytes`` bytes.
        total_bytes: What a pass touches.
        duration_s: How long each window runs for.
        correction: The factor the window's figure is scaled by.
        source: The label the figure carries.
        rounds: How many windows to run.

    Returns:
        The best figure and its label, or ``None`` when no window completed a pass.

    The maximum and not the mean, for the reason in the module docstring: another process
    on the memory controller can only push this number down, so the largest of several
    readings is the one least contaminated by whatever else the machine was doing, and
    averaging deliberately mixes that contamination back in.
    """
    run()  # warm up once, faulting the pages in before anything is timed
    best: float | None = None
    for _round in range(rounds):
        result = _time_passes(run, total_bytes, duration_s, correction=correction, source=source)
        if result is not None and (best is None or result[0] > best):
            best = result[0]
    return None if best is None else (best, source)


def _measure_single_threaded(
    total_bytes: int, duration_s: float, rounds: int = BANDWIDTH_ROUNDS
) -> tuple[float, Source] | None:
    """Pure-Python fallback: one thread, one ``bytearray`` slice copy, ``"estimated"``."""
    try:
        src_b = bytearray(total_bytes)
        dst_b = bytearray(total_bytes)
    except MemoryError:
        return None

    def copy() -> None:
        dst_b[:] = src_b

    return _best_of(
        copy,
        total_bytes,
        duration_s,
        correction=PURE_PYTHON_CORRECTION,
        source="estimated",
        rounds=rounds,
    )


def measure_ram_read_bandwidth_gbps(
    *,
    duration_s: float = 0.05,
    buffer_mb: int = 256,
    workers: int | None = None,
    rounds: int = BANDWIDTH_ROUNDS,
) -> tuple[float, Source] | None:
    """Measure sequential read bandwidth in GB/s and how it was obtained.

    Returns ``None`` when the buffer cannot be allocated or no full pass completed.

    ``rounds`` windows of ``duration_s`` are run and the fastest is kept, because what
    moves this figure between one run and the next is other processes on the memory
    controller and they can only ever slow it down. See the module docstring.

    With NumPy importable, splits a ``buffer_mb`` buffer into ``workers`` slices
    (default the physical core count capped at ``MAX_BANDWIDTH_WORKERS``) and runs a
    memory-bound reduction (``.max()``) over each slice concurrently from a thread
    pool, repeating for at least ``duration_s`` seconds; NumPy releases the GIL during
    the reduction, so this genuinely parallelises, and the result is labelled
    ``"measured"``. ``buffer_mb`` must stay well above the CPU's last-level cache or
    the reduction reads from cache instead of RAM and reports an inflated figure; 256
    MiB is comfortably larger than any current consumer or workstation LLC.

    Without NumPy, falls back to ``_measure_single_threaded``: a single-threaded
    pure-Python ``bytearray`` slice *copy* scaled by ``PURE_PYTHON_CORRECTION`` and
    labelled ``"estimated"`` rather than ``"measured"``, because a single thread
    cannot saturate a multi-channel memory controller and a copy is not a read; the
    correction folds in both gaps and is only an approximation. Install
    ``llamafit[fast]`` for a real measurement.

    The caller (``resolve_memory_bandwidth``) treats a result outside
    ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s) as implausible and falls back to an
    estimate or the assumed constant instead of trusting it, whatever the label.
    """
    total_bytes = buffer_mb * 1024 * 1024
    try:
        import numpy as np

        arr = np.ones(total_bytes, dtype=np.uint8)
    except ImportError:
        return _measure_single_threaded(total_bytes, duration_s, rounds)
    except MemoryError:
        return None

    n_workers = workers if workers is not None else _default_workers()
    bounds = [round(i * total_bytes / n_workers) for i in range(n_workers + 1)]
    slices = list(pairwise(bounds))

    def read_slice(bound: tuple[int, int]) -> int:
        lo, hi = bound
        return int(arr[lo:hi].max())

    with ThreadPoolExecutor(max_workers=n_workers) as pool:

        def read() -> None:
            list(pool.map(read_slice, slices))

        return _best_of(
            read, total_bytes, duration_s, correction=1.0, source="measured", rounds=rounds
        )


def machine_key(memory: Memory, cpu: Cpu) -> str:
    """A digest of the parts of the machine that could change its memory bandwidth.

    Args:
        memory: The pool as the probes described it, before a bandwidth was resolved.
        cpu: The processor.

    Returns:
        Sixteen hexadecimal characters.

    What goes in is the processor and its core count, the size of the pool, and the
    modules' own type, speed and channel count -- everything that decides how fast this
    machine can stream. What stays out is what was free at the time, which changes while
    you watch it and would file every run under a machine of its own. Reseating a module
    or changing an XMP profile moves the speed or the channel count, and so retires the
    stored figure by itself, which is the safe direction: the tool measures again rather
    than going on quoting a number about a machine that has changed underneath it.

    It is not :func:`llamafit.bench.fingerprint.host_fingerprint`, deliberately. That one
    needs a whole ``Host`` -- which does not exist yet at the point this is wanted, since
    the bandwidth goes *into* it -- and it includes the graphics driver, which has nothing
    to do with how fast the system memory reads.
    """
    parts = [
        cpu.model,
        str(cpu.physical_cores),
        str(memory.total_bytes),
        memory.type or "",
        str(memory.speed_mts or ""),
        str(memory.modules or ""),
        str(memory.channels or ""),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


class BandwidthCache:
    """Bandwidth figures kept between runs, filed under the machine each was taken on.

    One small JSON object keyed by :func:`machine_key`, rather than a file per key: there
    is one entry per machine and a portable ``LLAMAFIT_HOME`` carried between two of them
    should hold both rather than have the second overwrite the first.

    Every failure is a miss. A cache that cannot be read, cannot be parsed or holds
    something other than what was written is a cache that is not there, and the
    measurement runs; a cache that cannot be written is a measurement that simply is not
    kept. Neither is worth failing a scan over.
    """

    def __init__(self, key: str, path: Path | None = None) -> None:
        """Remember the machine and the file; the file is created on the first ``put``."""
        self.key = key
        self.path = path if path is not None else get_paths().cache_dir / CACHE_FILE

    def _entries(self) -> dict[str, dict[str, object]]:
        """Everything stored, or an empty mapping on any failure to read it."""
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {k: v for k, v in loaded.items() if isinstance(v, dict)}

    def get(self) -> tuple[float, Source] | None:
        """The figure kept for this machine, or ``None`` when there is none to trust."""
        entry = self._entries().get(self.key)
        if entry is None:
            return None
        gbps, source = entry.get("gbps"), entry.get("source")
        if not isinstance(gbps, int | float) or source not in ("measured", "estimated"):
            return None
        return float(gbps), source

    def put(self, gbps: float, source: Source) -> None:
        """Keep ``gbps`` for this machine, stamped with when it was taken."""
        entries = self._entries()
        entries[self.key] = {
            "gbps": gbps,
            "source": source,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except OSError as exc:  # an unwritable cache is a scan that measures again, not a failure
            _log.debug("could not write %s: %s", self.path, exc)


def resolve_memory_bandwidth(
    memory: Memory,
    *,
    measure: bool = True,
    cache: BandwidthCache | None = None,
    refresh: bool = False,
) -> Memory:
    """Set ``bandwidth_gbps`` with the best available source and label it.

    Args:
        memory: The pool, which is modified in place and returned.
        measure: Whether to consult a live measurement at all.
        cache: Where a figure for this machine is kept and looked for, or ``None`` to
            measure every time. A caller that passes nothing gets the old behaviour
            exactly, which is what every test that patches the measurement wants.
        refresh: Take the measurement again and replace what was kept, which is what
            ``llamafit system --refresh-bandwidth`` asks for.

    Returns:
        ``memory``, with a bandwidth and a label on it.

    Precedence is kept -> measured -> estimated -> assumed. A figure kept for this
    machine is used without measuring, because nothing about a memory controller changes
    between two runs of a command and re-timing it was what made the same question give
    different answers minute to minute; it carries ``bandwidth_cached``, so the board and
    the host table both say it was kept rather than taken. Failing that, a live
    measurement is used when it falls within ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s),
    labelled with whatever source ``measure_ram_read_bandwidth_gbps`` itself reports
    (``"measured"`` with NumPy, ``"estimated"`` on the pure-Python fallback), and kept for
    next time. Failing that, an existing ``"estimated"`` value already on ``memory`` is
    left alone; failing that, ``bandwidth_gbps`` is set to ``ASSUMED_RAM_BANDWIDTH_GBPS``
    and labelled ``"assumed"``.
    """
    if measure:
        kept = None if cache is None or refresh else cache.get()
        if kept is not None and _plausible(kept[0]):
            memory.bandwidth_gbps, memory.bandwidth_source = kept
            memory.bandwidth_cached = True
            return memory
        result = measure_ram_read_bandwidth_gbps()
        if result is not None and _plausible(result[0]):
            memory.bandwidth_gbps, memory.bandwidth_source = result
            memory.bandwidth_cached = False
            if cache is not None:
                cache.put(*result)
            return memory
    if memory.bandwidth_gbps is not None and memory.bandwidth_source == "estimated":
        return memory
    memory.bandwidth_gbps = ASSUMED_RAM_BANDWIDTH_GBPS
    memory.bandwidth_source = "assumed"
    return memory


def _plausible(gbps: float) -> bool:
    """Whether a figure is inside ``PLAUSIBLE_RANGE_GBPS``, whatever label it carries."""
    return PLAUSIBLE_RANGE_GBPS[0] <= gbps <= PLAUSIBLE_RANGE_GBPS[1]
