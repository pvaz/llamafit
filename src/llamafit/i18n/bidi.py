"""Directional isolates for the three languages LlamaFit writes right to left.

Arabic, Hebrew and Urdu run right to left, and every identifier LlamaFit prints stays in
Latin script by the project's own rule: a flag, a command name, a path, a model id, a
repository, a URL, a unit. A sentence that mixes the two is laid out by the Unicode
bidirectional algorithm (UAX #9), and the algorithm gets one thing wrong for us by
design: a leading hyphen is direction-neutral, so it takes the direction of whatever
surrounds it. Inside an Arabic paragraph the two hyphens of ``--verbose`` join the Arabic
run and the flag reaches the screen as ``verbose--``. A reader who retypes what they see
has typed a command that does not run.

The fix is one character each side. U+2068 FIRST STRONG ISOLATE opens a run whose
direction is decided by its own first strong character and which the surrounding text
cannot reach into; U+2069 POP DIRECTIONAL ISOLATE closes it. That is what the standard
prescribes for embedding text whose direction is not known in advance, which is exactly
what a value read off a machine is: a device name, a path, an error from a vendor tool.
An isolate also fixes the full stop that follows a Latin token at the end of an Arabic
sentence, because the isolated run counts as one neutral object and the stop then
resolves with the paragraph rather than with the Latin letters.

Three rules shape what is here.

**Nothing goes in the catalogs.** Two translators said so independently and both were
right: an invisible character breaks comparison and search over a ``.po`` file, several
terminals draw it as a box, and it cannot reorder a table's columns anyway. Marks are
emitted at render time, by the renderer, once.

**Nothing changes for a left-to-right language.** Every function here returns its input
unchanged unless the language now installed is one of :data:`RTL_LANGUAGES`, so thirty-four
of the thirty-seven catalogs, English included, are byte for byte what they were.

**Nothing reaches machine-readable output.** Only ``llamafit.cli.render`` imports this
module. ``--json`` is built from the services, which never call any of it, so a program
parsing LlamaFit never receives a direction mark.

What this cannot do is give a terminal a bidirectional algorithm it does not have. Most
do not have one, and there the marks are inert; ``docs/translations.md`` says so plainly
rather than implying a fix that only some terminals honour.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import TypeVar

from llamafit.i18n.tags import normalise
from llamafit.i18n.translator import current_language

FIRST_STRONG_ISOLATE = "\u2068"
"""Opens a run whose direction its own first strong character decides."""

POP_DIRECTIONAL_ISOLATE = "\u2069"
"""Closes the innermost open isolate."""

RIGHT_TO_LEFT_MARK = "\u200f"
"""A strong right-to-left character of no width, used to fix a line's base direction."""

_ISOLATE_INITIATORS = frozenset("\u2066\u2067\u2068")
"""LRI, RLI and FSI: the three characters that open an isolate."""

RTL_LANGUAGES = frozenset(
    {
        "ar",  # Arabic, shipped
        "he",  # Hebrew, shipped
        "ur",  # Urdu, shipped
        "iw",  # Hebrew under its retired code, which locale.normalize still answers with
        "fa",  # Persian
        "ps",  # Pashto
        "sd",  # Sindhi
        "ug",  # Uyghur
        "yi",  # Yiddish
        "dv",  # Divehi
        "ckb",  # Central Kurdish
        "syr",  # Syriac
    }
)
"""Language subtags written right to left, by subtag rather than by catalog.

The three LlamaFit ships are the first three. The rest are here so that a catalog
contributed tomorrow is laid out correctly on the day it lands rather than on the day
somebody remembers this set exists. A script property would be the principled way to
decide this, and the standard library has no table for it; a list of twelve subtags that
a reader can check against CLDR is better than a dependency for one predicate.
"""

_RTL_BIDI_CLASSES = frozenset({"R", "AL"})
"""The bidirectional classes that make a character strongly right to left."""

# What the renderer marks inside a sentence a translator wrote. The values a call site
# interpolates are handled by `isolate`, which needs no pattern because the code already
# knows they are data; this is for the identifiers a translator had to keep verbatim
# *inside* prose, where nothing in the string says which words are English and which are
# a command. The three shapes are the ones the English messages actually use, and each is
# unmistakable in Arabic, Hebrew or Urdu text:
#
#   `pip install --force-reinstall psutil`   a command, in the backticks the messages use
#   https://github.com/ggml-org/llama.cpp    a URL
#   --verbose, -v                            a command-line option
#
# Deliberately absent: a bare hyphenated command name such as `nvidia-smi`. Its hyphen
# sits between two Latin letters, which the algorithm resolves as Latin, so it is not at
# risk; guessing at which unmarked Latin words are identifiers would mark ordinary words
# in a translator's own sentence for no gain.
_IDENTIFIER_IN_PROSE = re.compile(
    r"""
      `[^`\n]+`                          # a command line between backticks
    | \b[A-Za-z][A-Za-z0-9+.-]*://\S+    # a URL
    | (?<![\w-])--?[A-Za-z][A-Za-z0-9_-]*  # a long or short option
    """,
    re.VERBOSE,
)

_TRAILING_PUNCTUATION = ".,;:!?)]}\u060c\u061b\u061f"
"""Punctuation a URL must not swallow, Arabic comma, semicolon and question mark included."""

T = TypeVar("T")


def is_rtl(language: str | None = None) -> bool:
    """Whether a language is written right to left.

    Args:
        language: The tag to ask about; the language now installed when ``None``.

    Returns:
        True when the tag's language subtag is one of :data:`RTL_LANGUAGES`.
    """
    tag = current_language() if language is None else language
    normalised = normalise(tag) or tag
    return normalised.split("_", 1)[0].lower() in RTL_LANGUAGES


def isolate(value: object) -> str:
    """Render one untranslatable value as its own directional island.

    Call this wherever the code interpolates something it already treats as data rather
    than as words: a flag, a command name, a path, a model or repository id, a URL, a
    size with its unit. The value comes back wrapped in FSI and PDI when the language now
    installed runs right to left, and untouched otherwise.

    One thing it must not be given is a value the message glues a letter onto — the
    number in ``%(total)sB``, say. Isolating that number would cut it off from the ``B``
    that follows and the two would swap places on screen, which is worse than the problem
    it was meant to solve. Isolate a value the message sets off with a space or with
    punctuation; leave a glued one alone.

    Args:
        value: The value; anything that can be turned into a string.

    Returns:
        The value as a string, isolated when the reader's language needs it to be.
    """
    text = str(value)
    if not text or not is_rtl():
        return text
    return f"{FIRST_STRONG_ISOLATE}{text}{POP_DIRECTIONAL_ISOLATE}"


def isolate_identifiers(text: str) -> str:
    """Isolate each command, URL and option a translator kept verbatim inside prose.

    A hint such as *reinstall psutil with ``pip install --force-reinstall psutil``* is one
    message: the flag inside it never passes through an interpolation, so :func:`isolate`
    can never reach it and the catalogs are not where a mark may go. This is the one place
    a pattern is used instead of knowledge, and it is kept to three shapes that no Arabic,
    Hebrew or Urdu word can be mistaken for.

    Text already inside an isolate is left alone, so a value :func:`isolate` has wrapped
    is not wrapped a second time.

    Args:
        text: A finished sentence, in whatever language is installed.

    Returns:
        The sentence with each identifier isolated, or unchanged for a left-to-right
        language.
    """
    if not text or not is_rtl():
        return text
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in _ISOLATE_INITIATORS:
            if depth == 0:
                parts.append(_mark_identifiers(text[start:index]))
                start = index
            depth += 1
        elif char == POP_DIRECTIONAL_ISOLATE and depth:
            depth -= 1
            if depth == 0:
                parts.append(text[start : index + 1])
                start = index + 1
    rest = text[start:]
    parts.append(rest if depth else _mark_identifiers(rest))
    return "".join(parts)


def for_display(text: str) -> str:
    """Prepare a finished piece of interface text for a terminal.

    Two things happen, both only for a right-to-left language. Every identifier a
    translator kept verbatim is isolated, per :func:`isolate_identifiers`. Then every line
    that contains a right-to-left letter is opened with U+200F RIGHT-TO-LEFT MARK, which
    fixes the line's base direction: without it a line that happens to begin with a Latin
    identifier is laid out left to right by rule P2 of the algorithm, and the Arabic that
    follows comes out running the wrong way with its full stop adrift.

    A line with no right-to-left letter in it — a model id, a size, a column of quant
    names — is left as it is. It is already unambiguous, and a mark nothing needs is a
    character some terminal will draw as a box.

    Args:
        text: The finished cell, title or caption, which may hold several lines.

    Returns:
        The text as the terminal should receive it.
    """
    if not text or not is_rtl():
        return text
    return "\n".join(_directed_line(line) for line in isolate_identifiers(text).split("\n"))


def reading_order(items: Sequence[T]) -> list[T]:
    """Put a table's columns, or one row's cells, in the order they are read.

    Rich lays a table out from the left, which is where a right-to-left reader finishes.
    Reversing the sequence puts the first column against the right edge, where the eye
    starts.

    Args:
        items: The columns or cells, first-read first.

    Returns:
        A new list, reversed for a right-to-left language and copied otherwise.
    """
    return list(reversed(items)) if is_rtl() else list(items)


def mirror_justify(justify: str) -> str:
    """Mirror a column's alignment, so padding falls at the end of the line and not the start.

    A left-aligned column pads on the right. Read right to left that is padding before the
    first character rather than after the last, which is what puts the ragged edge at the
    start of reading. Left becomes right and right becomes left; centre is its own mirror,
    and so is anything else Rich understands.

    Args:
        justify: The alignment the column would have in a left-to-right language.

    Returns:
        The alignment to give the column in the language now installed.
    """
    if not is_rtl():
        return justify
    return {"left": "right", "right": "left"}.get(justify, justify)


def _mark_identifiers(segment: str) -> str:
    """Isolate every identifier in one stretch of text known not to be inside an isolate."""
    return _IDENTIFIER_IN_PROSE.sub(_wrap_match, segment)


def _wrap_match(match: re.Match[str]) -> str:
    """Isolate one matched identifier, leaving any punctuation it ran into outside.

    A URL is matched as far as the next space, which swallows the full stop that ends the
    sentence around it; that stop belongs to the Arabic and not to the URL, so it is put
    back outside the island. What is left can never be empty: every shape the pattern
    matches opens with a backtick, a letter or a hyphen, and none of those is punctuation
    this trims.
    """
    token = match.group()
    identifier = token.rstrip(_TRAILING_PUNCTUATION)
    trailing = token[len(identifier) :]
    return f"{FIRST_STRONG_ISOLATE}{identifier}{POP_DIRECTIONAL_ISOLATE}{trailing}"


def _directed_line(line: str) -> str:
    """Open a line with a right-to-left mark when it holds right-to-left letters."""
    if line.startswith(RIGHT_TO_LEFT_MARK) or not _has_rtl_letter(line):
        return line
    return RIGHT_TO_LEFT_MARK + line


def _has_rtl_letter(line: str) -> bool:
    """Whether any character in the line is strongly right to left."""
    return any(unicodedata.bidirectional(char) in _RTL_BIDI_CLASSES for char in line)
