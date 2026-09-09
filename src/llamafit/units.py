"""Byte sizes: parsing user input and formatting for display."""

from __future__ import annotations

import re

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*$", re.IGNORECASE)
_DECIMAL = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BINARY = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_BINARY_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]
_DECIMAL_UNITS = ["B", "KB", "MB", "GB", "TB"]


def parse_size(text: str) -> int:
    """Parse a human size such as ``8G``, ``7.5GiB`` or ``512MB`` into bytes.

    Bare letters and decimal units (``G``, ``GB``) are powers of 1000; binary units
    (``GiB``) are powers of 1024. A plain integer is taken as bytes.

    Raises:
        ValueError: If the text is not a size.
    """
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r}")
    number, prefix, binary = match.groups()
    prefix = prefix.lower()
    table = _BINARY if binary else _DECIMAL
    return int(float(number) * table[prefix])


def format_bytes(n: int | None, *, binary: bool = True, digits: int = 1) -> str:
    """Format a byte count for humans, ``"unknown"`` when ``n`` is ``None``."""
    if n is None:
        return "unknown"
    base = 1024 if binary else 1000
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    value = float(n)
    index = 0
    while value >= base and index < len(units) - 1:
        value /= base
        index += 1
    if index == 0:
        return f"{n} B"
    return f"{value:.{digits}f} {units[index]}"


def gib(n: int | None) -> float | None:
    """Convert bytes to GiB as a float, passing ``None`` through."""
    return None if n is None else n / 1024**3
