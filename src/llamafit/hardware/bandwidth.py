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
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import psutil

from llamafit.models.host import Memory, Source

ASSUMED_RAM_BANDWIDTH_GBPS = 40.0
PLAUSIBLE_RANGE_GBPS = (5.0, 1000.0)
MAX_BANDWIDTH_WORKERS = 8
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
    """
    run()  # warm up, fault the pages in
    passes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        run()
        passes += 1
    elapsed = time.perf_counter() - start
    if passes == 0 or elapsed <= 0:
        return None
    return round(passes * total_bytes / elapsed / 1e9 * correction, 1), source


def _measure_single_threaded(total_bytes: int, duration_s: float) -> tuple[float, Source] | None:
    """Pure-Python fallback: one thread, one ``bytearray`` slice copy, ``"estimated"``."""
    try:
        src_b = bytearray(total_bytes)
        dst_b = bytearray(total_bytes)
    except MemoryError:
        return None

    def copy() -> None:
        dst_b[:] = src_b

    return _time_passes(
        copy, total_bytes, duration_s, correction=PURE_PYTHON_CORRECTION, source="estimated"
    )


def measure_ram_read_bandwidth_gbps(
    *, duration_s: float = 0.05, buffer_mb: int = 256, workers: int | None = None
) -> tuple[float, Source] | None:
    """Measure sequential read bandwidth in GB/s and how it was obtained.

    Returns ``None`` when the buffer cannot be allocated or no full pass completed.

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
        return _measure_single_threaded(total_bytes, duration_s)
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

        return _time_passes(read, total_bytes, duration_s, correction=1.0, source="measured")


def resolve_memory_bandwidth(memory: Memory, *, measure: bool = True) -> Memory:
    """Set ``bandwidth_gbps`` with the best available source and label it.

    Precedence is measured -> estimated -> assumed: a live measurement is used only
    when it falls within ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s), labelled with
    whatever source ``measure_ram_read_bandwidth_gbps`` itself reports (``"measured"``
    with NumPy, ``"estimated"`` on the pure-Python fallback); otherwise an existing
    ``"estimated"`` value already on ``memory`` is kept; failing that, ``bandwidth_gbps``
    is set to ``ASSUMED_RAM_BANDWIDTH_GBPS`` and labelled ``"assumed"``.
    """
    if measure:
        result = measure_ram_read_bandwidth_gbps()
        if result is not None:
            value, source = result
            if PLAUSIBLE_RANGE_GBPS[0] <= value <= PLAUSIBLE_RANGE_GBPS[1]:
                memory.bandwidth_gbps = value
                memory.bandwidth_source = source
                return memory
    if memory.bandwidth_gbps is not None and memory.bandwidth_source == "estimated":
        return memory
    memory.bandwidth_gbps = ASSUMED_RAM_BANDWIDTH_GBPS
    memory.bandwidth_source = "assumed"
    return memory
