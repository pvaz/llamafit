# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The ``Plural-Forms`` header: how many forms a language has, and which one a count picks.

The rule is a C expression written by a translator and read from a data file, so it is
never evaluated here. The standard library already parses that expression language
safely: :func:`gettext.c2py` tokenises it against a whitelist and returns a function of
``n``. This module only splits the header into its two halves and hands the expression
over.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from gettext import c2py

DEFAULT_PLURAL_FORMS = "nplurals=2; plural=(n != 1);"
"""English: one form for exactly one, another for every other count.

It is what the extracted template declares, for a translator to replace with their
own language's rule. It is not a fallback: a catalog that declares no ``Plural-Forms``
at all is refused rather than quietly given this one, because a three-form language
given English's rule reads real words in the wrong grammar and nothing says so.
"""

_HEADER_RE = re.compile(r"nplurals\s*=\s*(\d+)\s*;\s*plural\s*=(.+)", re.DOTALL)
_MAX_FORMS = 10


class PluralFormsError(ValueError):
    """A ``Plural-Forms`` header is missing, malformed, or names an unusable rule."""


class PluralRule:
    """How many plural forms a language has and which form a count selects.

    Attributes:
        nplurals: The number of forms, at least one.
        expression: The C expression the header declared, without its trailing ``;``.
    """

    def __init__(self, nplurals: int, expression: str) -> None:
        if nplurals < 1 or nplurals > _MAX_FORMS:
            raise PluralFormsError(f"nplurals must be between 1 and {_MAX_FORMS}, not {nplurals}")
        try:
            select: Callable[[int], int] = c2py(expression)
        except ValueError as exc:
            raise PluralFormsError(f"invalid plural expression {expression!r}: {exc}") from exc
        self.nplurals = nplurals
        self.expression = expression
        self._select = select

    def index(self, n: int) -> int:
        """Return the plural form ``n`` selects, always inside ``0..nplurals - 1``.

        A rule that returns a form the header did not declare is clamped rather than
        allowed to reach into a catalog past its end.
        """
        try:
            chosen = int(self._select(n))
        except (ArithmeticError, TypeError, ValueError):
            chosen = 0
        return min(max(chosen, 0), self.nplurals - 1)

    def __repr__(self) -> str:
        """Show the rule as it would be written in a header."""
        return f"PluralRule(nplurals={self.nplurals}, expression={self.expression!r})"


def parse_plural_forms(value: str) -> PluralRule:
    """Parse a ``Plural-Forms`` header value such as ``nplurals=2; plural=(n != 1);``.

    Returns:
        The rule the header declares.

    Raises:
        PluralFormsError: If the header is not in that shape or the expression is not
            one the standard library's parser accepts.
    """
    match = _HEADER_RE.match(value.strip())
    if match is None:
        raise PluralFormsError(f"not a Plural-Forms header: {value.strip()!r}")
    expression = match.group(2).strip().rstrip(";").strip()
    if not expression:
        raise PluralFormsError(f"no plural expression in {value.strip()!r}")
    return PluralRule(int(match.group(1)), expression)
