"""RAM bandwidth: measure it when possible, otherwise estimate or assume.

Generation speed with experts in RAM is bounded by this number, so it deserves a
real measurement. The probe copies a buffer larger than any CPU cache for a few tens
of milliseconds and reports bytes moved per second (read plus write counted once, as
llama.cpp's weight streaming is read-dominated).
"""

from __future__ import annotations

import time

from llamafit.models.host import Memory

ASSUMED_RAM_BANDWIDTH_GBPS = 40.0
PLAUSIBLE_RANGE_GBPS = (5.0, 1000.0)
# A bytearray slice copy in pure Python runs single-threaded through memcpy; measured
# against NumPy on the reference machine it under-reports by about 1.6x.
PURE_PYTHON_CORRECTION = 1.6


def measure_ram_bandwidth_gbps(*, duration_s: float = 0.05, buffer_mb: int = 256) -> float | None:
    """Measure copy bandwidth in GB/s, or ``None`` when the buffer cannot be allocated.

    Repeatedly copies a ``buffer_mb`` buffer for at least ``duration_s`` seconds and
    reports bytes moved per second. Uses NumPy when importable; otherwise falls back
    to a pure-Python ``bytearray`` slice copy scaled by ``PURE_PYTHON_CORRECTION``.
    The caller (``resolve_memory_bandwidth``) treats a result outside
    ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s) as implausible and falls back to an
    estimate or the assumed constant instead of trusting it.
    """
    size = buffer_mb * 1024 * 1024
    try:
        import numpy as np

        src = np.ones(size, dtype=np.uint8)
        dst = np.empty_like(src)
        correction = 1.0

        def copy() -> None:
            np.copyto(dst, src)

    except ImportError:
        try:
            src_b = bytearray(size)
            dst_b = bytearray(size)
        except MemoryError:
            return None
        correction = PURE_PYTHON_CORRECTION

        def copy() -> None:
            dst_b[:] = src_b

    except MemoryError:
        return None

    copy()  # warm up, fault the pages in
    passes = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        copy()
        passes += 1
    elapsed = time.perf_counter() - start
    if passes == 0 or elapsed <= 0:
        return None
    return round(passes * size / elapsed / 1e9 * correction, 1)


def resolve_memory_bandwidth(memory: Memory, *, measure: bool = True) -> Memory:
    """Set ``bandwidth_gbps`` with the best available source and label it.

    Precedence is measured -> estimated -> assumed: a live measurement is used only
    when it falls within ``PLAUSIBLE_RANGE_GBPS`` (5 to 1000 GB/s); otherwise an
    existing ``"estimated"`` value already on ``memory`` is kept; failing that,
    ``bandwidth_gbps`` is set to ``ASSUMED_RAM_BANDWIDTH_GBPS`` and labelled
    ``"assumed"``.
    """
    if measure:
        measured = measure_ram_bandwidth_gbps()
        if measured is not None and PLAUSIBLE_RANGE_GBPS[0] <= measured <= PLAUSIBLE_RANGE_GBPS[1]:
            memory.bandwidth_gbps = measured
            memory.bandwidth_source = "measured"
            return memory
    if memory.bandwidth_gbps is not None and memory.bandwidth_source == "estimated":
        return memory
    memory.bandwidth_gbps = ASSUMED_RAM_BANDWIDTH_GBPS
    memory.bandwidth_source = "assumed"
    return memory
