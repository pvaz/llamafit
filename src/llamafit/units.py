# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Numbers for people: byte sizes, and the punctuation a language writes numbers with.

Every figure LlamaFit prints is formatted the English way first and then passed through
:func:`localise_number`, which swaps the two separators for whatever the catalog says this
language uses. The separators are catalog entries rather than a locale lookup on purpose:
the process-wide locale is global state that would reach code with no opinion about it,
and an extra dependency for two characters is not a trade worth making.

It is not cosmetic. English writes a model's context as ``32,768``; Portuguese and German
read that as a fraction, and would be told the model holds thirty-two tokens and a bit.

The three entries are read with :func:`llamafit.i18n.pgettext_literal` rather than with
``pgettext``, because a separator can be a space. Half the languages here group digits
with one, and the ordinary rule that a blank translation means untranslated -- the right
rule for a sentence, which must never degrade to a blank line -- would have handed every
one of them the English comma with nothing anywhere to say so.
"""

from __future__ import annotations

import re

from llamafit.i18n import pgettext, pgettext_literal

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*$", re.IGNORECASE)
_DECIMAL = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BINARY = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_BINARY_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]
_DECIMAL_UNITS = ["B", "KB", "MB", "GB", "TB"]


def group_separator() -> str:
    """What this language puts between a number's groups of three digits."""
    # Translators: this is punctuation, not prose. It is what your language puts between
    # a number's groups of three digits: English writes 32,768, German 32.768 and French
    # 32 768. Write the character your readers expect. A space is a character like any
    # other here and is read as one, so a language that groups with a space says so by
    # writing that space, and a no-break space if that is what it wants. Only a
    # translation with nothing in it at all means untranslated, and that falls back to
    # the English comma rather than to no separator.
    return pgettext_literal("thousands separator", ",")


def decimal_separator() -> str:
    """What this language puts between a number's whole and fractional parts."""
    # Translators: this is punctuation, not prose. It is what your language puts before
    # the fractional part of a number: English writes 127.8 and Portuguese 127,8. Write
    # the character your readers expect; a space counts as a character here, the same as
    # in the group separator above. Only a translation with nothing in it at all means
    # untranslated, and that falls back to the English point.
    return pgettext_literal("decimal separator", ".")


def billions_suffix() -> str:
    """The mark a parameter count of a thousand million carries, as in ``27B``."""
    # Translators: leave this as B. It is not the English word "billion" — the long and
    # short scales disagree about what one of those is — and it is not a number word at
    # all. It is how these models are named: the file is Qwen3-27B, the vendor announces
    # a 27B model, and this cell sits in the same row as that identifier, so a localised
    # abbreviation would make one row disagree with itself. Readers of every language
    # meet the B in the model's own name before they meet this table.
    # The entry is here for the language whose readers genuinely would not recognise it.
    # If yours is one, write what they do use; whitespace is read as written, for a form
    # your language separates from the digits. Otherwise leave it, and leave it filled in
    # rather than empty, so the next reader can see the question was asked.
    return pgettext_literal("parameter count", "B")


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


def format_decimals(value: float, places: int) -> str:
    """A number to a fixed number of decimal places, in this language's punctuation.

    Args:
        value: The number.
        places: How many digits after the separator.

    A message that writes ``%(gb).2f`` formats in the C locale, so it always writes a
    point -- which put ``0.92 GB`` in a sentence directly underneath a table that had just
    written ``0,0165`` for a reader whose language uses the comma. The numeric conversion
    is the whole problem: it cannot be reached by the catalog, because the catalog carries
    a separator and not a format. So those messages take ``%(gb)s`` instead and are handed
    the result of this, which is what every number that reaches a reader already goes
    through.

    Grouping is deliberately not applied. These are quantities under a thousand -- gigabytes
    per token, an efficiency, a correction factor -- and the digit-grouping limitation
    ``docs/translations.md`` records is a separate thing that needs a pattern per language.
    """
    return localise_number(f"{value:.{places}f}")


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
