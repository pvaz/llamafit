"""Numbers for people: byte sizes, and the punctuation a language writes numbers with.

Every figure LlamaFit prints is formatted the English way first and then passed through
:func:`localise_number`, which swaps the two separators for whatever the catalog says this
language uses. The separators are catalog entries rather than a locale lookup on purpose:
the process-wide locale is global state that would reach code with no opinion about it,
and an extra dependency for two characters is not a trade worth making.

It is not cosmetic. English writes a model's context as ``32,768``; Portuguese and German
read that as a fraction, and would be told the model holds thirty-two tokens and a bit.
"""

from __future__ import annotations

import re

from llamafit.i18n import pgettext

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*$", re.IGNORECASE)
_DECIMAL = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BINARY = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_BINARY_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]
_DECIMAL_UNITS = ["B", "KB", "MB", "GB", "TB"]


def group_separator() -> str:
    """What this language puts between a number's groups of three digits."""
    # Translators: this is punctuation, not prose. It is what your language puts between
    # a number's groups of three digits: English writes 32,768 and German 32.768. Write
    # the character your readers expect. An empty translation does not mean "no
    # separator"; it means untranslated, and falls back to the English comma. A
    # translation that is only a space counts as empty too, so a language that groups
    # with a space (French, Russian, Swedish and others) cannot say so here yet: please
    # open an issue rather than working around it, because the fix belongs in the reader.
    return pgettext("thousands separator", ",")


def decimal_separator() -> str:
    """What this language puts between a number's whole and fractional parts."""
    # Translators: this is punctuation, not prose. It is what your language puts before
    # the fractional part of a number: English writes 127.8 and Portuguese 127,8. An empty
    # translation means untranslated, and falls back to the English point.
    return pgettext("decimal separator", ".")


def billions_suffix() -> str:
    """The abbreviation this language uses for a thousand million, as in ``27B``."""
    # Translators: this is an abbreviation, not a word. It marks a parameter count of a
    # thousand million, as in 27B. Do not translate the English word "billion": the long
    # and short scales disagree about what a billion is, so write the abbreviation your
    # own readers expect for a thousand million.
    return pgettext("parameter count", "B")


def localise_number(text: str) -> str:
    """Rewrite a number written the English way with this language's separators.

    Both separators are swapped in one pass over the text, not one after the other: a
    language that groups with a point would otherwise re-read its own output and turn every
    group separator it had just written into a decimal one.

    Args:
        text: A number already formatted the English way, and nothing else. Never a
            sentence: it would take the points out of a file path and a version number too.

    Returns:
        The same number with its separators replaced.
    """
    group, decimal = group_separator(), decimal_separator()
    if group == "," and decimal == ".":
        return text
    return "".join(group if c == "," else decimal if c == "." else c for c in text)


def format_grouped(value: int) -> str:
    """A whole number with its digits grouped, in this language's punctuation."""
    return localise_number(f"{value:,}")


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
    """Format a byte count for humans, ``"unknown"`` when ``n`` is ``None``.

    The word carries a context because two other rows say *unknown* about something
    else, and Portuguese inflects it for the noun each row is about.
    """
    if n is None:
        return pgettext("size", "unknown")
    base = 1024 if binary else 1000
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    value = float(n)
    index = 0
    while value >= base and index < len(units) - 1:
        value /= base
        index += 1
    if index == 0:
        return f"{n} B"
    return f"{localise_number(f'{value:.{digits}f}')} {units[index]}"


def gib(n: int | None) -> float | None:
    """Convert bytes to GiB as a float, passing ``None`` through."""
    return None if n is None else n / 1024**3
