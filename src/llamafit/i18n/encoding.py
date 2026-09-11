# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What happens when the stream the output goes to cannot carry the language.

A translation is chosen by looking at the person: what they asked for, what their
environment says, what their operating system says. None of that looks at where the
words are going. On Windows a console window takes any character, but a file or a pipe
takes the system code page — ``cp1252`` on a Western install — and Python's text streams
raise ``UnicodeEncodeError`` on the first character the page does not have. So
``llamafit --language ja list > out.txt`` used to exit 1 with nothing in the file, and the
error saying so was itself in Japanese and could not be written either, so it arrived as
a run of backslash-u escapes. A right-to-left language failed the same way one step later,
on the direction isolates :mod:`llamafit.i18n.bidi` puts around a Latin identifier, which
even the Arabic code page does not have.

Two decisions, made here and nowhere else, and they are different decisions because a
character and a language are different things.

**A character the stream cannot write is replaced, never raised over.** Whatever else
goes wrong, the program finishes and says what it can. A format character — a direction
mark, a joiner — carries no text and is dropped; a space of an unusual width becomes an
ordinary space; anything else becomes ``?``, and each ``?`` is counted so the interface
can say once, at the end, how many there were and what would have avoided them. This is
done with a :mod:`codecs` error handler installed on the stream rather than with a
wrapper around it, because Rich, Click and ``print`` all write to ``sys.stdout``
directly and the handler is the one seam under all three.

**A language the stream cannot write is not spoken.** Replacing is right for a character
and wrong for a language. A Latin letter that loses its diacritic to a ``?`` is a letter
with a hole in it, and a reader fills the hole: *n?o* is still *não*. A letter of
another script that becomes ``?`` is a hole where a letter was, and a screen of those is
not a translation with holes, it is holes. So before a catalog is installed, the stream
is asked whether it can write the letters of the language's own script — the ones that
are not Latin — and when it cannot write most of them the interface speaks English and
says why. That sentence is English on purpose: it is the one sentence that has to be
readable on a stream that has just been found unable to write the language, and English
is the one language every encoding Python offers can write. The same rule keeps a
Portuguese catalog in Portuguese on a Polish code page, where only *ã* and *õ* are
missing, and takes a Japanese one to English on a Western code page, where everything
is. Between those two there is no realistic middle: a catalog is written in one script,
and a code page either has that script or does not. The threshold of half is where
*more lost than kept* begins, and it is deliberately not a knob.

What this cannot do is give a terminal a glyph it does not have. A console that decodes
UTF-8 correctly and draws a box for every kanji is showing the language as well as it
can, and nothing here can tell that box from a character.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from collections import Counter

ERROR_HANDLER = "llamafit-replace"
"""The name the replacement policy is registered under with :mod:`codecs`."""

_BEYOND_ASCII = re.compile(r"[^\x00-\x7f]")
"""Every character that is not ASCII, which is every character an encoding might lack."""

_replaced = 0
"""How many visible characters have been written as ``?`` so far in this process."""


def _replace(error: UnicodeError) -> tuple[str, int]:
    """Stand in for what the stream cannot encode, and count what a reader will miss.

    Only the encoding direction is handled; a decoding error is somebody else's and is
    raised again, which is what :mod:`codecs` asks a handler to do with an error it does
    not understand.
    """
    if not isinstance(error, UnicodeEncodeError):
        raise error
    global _replaced
    replacement: list[str] = []
    for char in error.object[error.start : error.end]:
        category = unicodedata.category(char)
        if category == "Cf":
            continue
        if category == "Zs":
            replacement.append(" ")
            continue
        replacement.append("?")
        _replaced += 1
    return "".join(replacement), error.end


codecs.register_error(ERROR_HANDLER, _replace)


def tolerate(stream: object) -> None:
    """Make a text stream replace what it cannot encode instead of raising.

    The stream keeps its encoding and its identity: only its error policy changes, so a
    console cached by Rich or captured by a test harness goes on working. A stream that
    cannot be reconfigured — one that is not a text stream, one already closed, one an
    embedding application made up — is left alone rather than complained about, because
    this runs before anything can be said and the worst case is the behaviour it had.

    Args:
        stream: ``sys.stdout`` or ``sys.stderr``, or anything else with a ``reconfigure``.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(errors=ERROR_HANDLER)
    except (ValueError, OSError):  # closed, or a stream that only says it can
        return


def encoding_of(stream: object) -> str | None:
    """The encoding a text stream writes in, or ``None`` when it does not say.

    Args:
        stream: ``sys.stdout`` or anything else that may carry an ``encoding``.

    Returns:
        The encoding's name as the stream reports it, which is what a person should be
        shown, since it is the name they can look up.
    """
    encoding = getattr(stream, "encoding", None)
    return encoding if isinstance(encoding, str) and encoding else None


def replacements() -> int:
    """How many visible characters have been written as ``?`` so far in this process."""
    return _replaced


def clear_replacements() -> None:
    """Forget the count. Only tests need this; a real process reports once and exits."""
    global _replaced
    _replaced = 0


def can_write(text: str, encoding: str) -> bool:
    """Whether a stream encoded as ``encoding`` can carry the language ``text`` is written in.

    The letters that decide it are the ones outside the Latin script, because those are
    the ones a reader cannot do without; the module docstring says why. ``text`` is meant
    to be everything a catalog would say, so that a language is judged on its whole
    vocabulary and not on one sentence that happened to be all identifiers.

    Args:
        text: The translations, joined; the more the better.
        encoding: The stream's encoding, as :func:`encoding_of` reports it.

    Returns:
        True when the text has no letters outside the Latin script, or when the encoding
        can write at least half of the ones it has. An encoding Python does not know
        cannot be judged and is trusted, since the error handler catches whatever it
        turns out not to be able to do.
    """
    try:
        codecs.lookup(encoding)
    except LookupError:
        return True
    own = 0
    lost = 0
    for char, count in Counter(_BEYOND_ASCII.findall(text)).items():
        if not char.isalpha() or unicodedata.name(char, "").startswith("LATIN"):
            continue
        own += count
        try:
            char.encode(encoding)
        except UnicodeEncodeError:
            lost += count
    return lost * 2 <= own
