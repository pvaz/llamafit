"""RAM bandwidth: measure it when possible, otherwise estimate or assume.

Generation speed with experts in RAM is bounded by this number, so it deserves a real
measurement. A single thread cannot saturate a multi-channel memory controller, so
when NumPy is importable this module measures properly: it splits a buffer larger
than any CPU cache into one slice per worker thread and copies the slices
concurrently from a thread pool (NumPy releases the GIL during the copy, so this
genuinely runs in parallel), reporting the aggregate bytes moved per second (read
plus write counted once, as llama.cpp's weight streaming is read-dominated) as
``"measured"``.

Without NumPy, the fallback is a single pure-Python ``bytearray`` slice copy. It
holds the GIL for the whole copy, so it cannot be parallelised across threads, and a
single thread under-reports the bandwidth a saturating, multi-threaded copy would
find. That result is scaled by ``PURE_PYTHON_CORRECTION`` (about 1.6x, measured
against the NumPy multi-threaded figure on the reference machine) and labelled
``"estimated"`` rather than ``"measured"``, since the correction is only an
approximation. Install the optional extra, ``pip install llamafit[fast]``, to bring
in NumPy and get a real measurement instead.
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
# The pure-Python bytearray copy runs single-threaded and holds the GIL throughout, so it
# cannot be parallelised across threads; measured against a NumPy multi-threaded copy on
# the reference machine, it under-reports by about 1.6x. Applied only on the no-NumPy
# fallback path (see the module docstring), which is labelled "estimated", not "measured",
# precisely because this correction is an approximation rather than a real measurement.
PURE_PYTHON_CORRECTION = 1.6


def _default_workers() -> int:
    """Physical core count capped at ``MAX_BANDWIDTH_WORKERS``, at least one."""
    return max(1, min(psutil.cpu_count(logical=False) or 1, MAX_BANDWIDTH_WORKERS))


def _time_copies(
    copy: Callable[[], None],
    size: int,
    duration_s: float,
    *,
    correction: float,
    source: Source,
) -> tuple[float, Source] | None:
    """Run ``copy`` repeatedly for at least ``duration_s`` seconds and report GB/s.

    Returns:
        The measured (or corrected) bandwidth and its source label, or ``None`` when
        no full pass completed.
    """
    copy()  # warm up, fault the pages in
    passes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        copy()
        passes += 1
    elapsed = time.perf_counter() - start
    if passes == 0 or elapsed <= 0:
        return None
    return round(passes * size / elapsed / 1e9 * correction, 1), source


def _measure_single_threaded(size: int, duration_s: float) -> tuple[float, Source] | None:
    """Pure-Python fallback: one thread, one ``bytearray`` slice copy, ``"estimated"``."""
    try:
        src_b = bytearray(size)
        dst_b = bytearray(size)
    except MemoryError:
        return None

    def copy() -> None:
        dst_b[:] = src_b

    return _time_copies(
        copy, size, duration_s, correction=PURE_PYTHON_CORRECTION, source="estimated"
    )


def measure_ram_bandwidth_gbps(
    *, duration_s: float = 0.05, buffer_mb: int = 256, workers: int | None = None
) -> tuple[float, Source] | None:
    """Measure copy bandwidth in GB/s and how it was obtained.

    Returns ``None`` when the buffer cannot be allocated or no full pass completed.

    With NumPy importable, splits a ``buffer_mb`` buffer into ``workers`` slices
    (default the physical core count capped at ``MAX_BANDWIDTH_WORKERS``) and copies
    them concurrently from a thread pool, repeating for at least ``duration_s``
    seconds; NumPy releases the GIL during the copy, so this genuinely parallelises,
    and the result is labelled ``"measured"``.

    Without NumPy, falls back to ``_measure_single_threaded``: a single-threaded
    pure-Python ``bytearray`` slice copy scaled by ``PURE_PYTHON_CORRECTION`` and
    labelled ``"estimated"`` rather than ``"measured"``, because a single thread
    cannot saturate a multi-channel memory controller and the correction is only an
    approximation for it. Install ``llamafit[fast]`` for a real measurement.

    The caller (``resolve_memory_bandwidth``) treats a result outside
    ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s) as implausible and falls back to an
    estimate or the assumed constant instead of trusting it, whatever the label.
    """
    size = buffer_mb * 1024 * 1024
    try:
        import numpy as np

        src = np.ones(size, dtype=np.uint8)
        dst = np.empty_like(src)
    except ImportError:
        return _measure_single_threaded(size, duration_s)
    except MemoryError:
        return None

    n_workers = workers if workers is not None else _default_workers()
    bounds = [round(i * size / n_workers) for i in range(n_workers + 1)]
    slices = list(pairwise(bounds))

    def copy_slice(bound: tuple[int, int]) -> None:
        lo, hi = bound
        np.copyto(dst[lo:hi], src[lo:hi])

    with ThreadPoolExecutor(max_workers=n_workers) as pool:

        def copy() -> None:
            list(pool.map(copy_slice, slices))

        return _time_copies(copy, size, duration_s, correction=1.0, source="measured")


def resolve_memory_bandwidth(memory: Memory, *, measure: bool = True) -> Memory:
    """Set ``bandwidth_gbps`` with the best available source and label it.

    Precedence is measured -> estimated -> assumed: a live measurement is used only
    when it falls within ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s), labelled with
    whatever source ``measure_ram_bandwidth_gbps`` itself reports (``"measured"`` with
    NumPy, ``"estimated"`` on the pure-Python fallback); otherwise an existing
    ``"estimated"`` value already on ``memory`` is kept; failing that, ``bandwidth_gbps``
    is set to ``ASSUMED_RAM_BANDWIDTH_GBPS`` and labelled ``"assumed"``.
    """
    if measure:
        result = measure_ram_bandwidth_gbps()
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
