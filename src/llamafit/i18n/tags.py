"""Language tags: putting them in one shape and matching a request to a catalog.

Nothing here reads a file or asks the operating system anything; it is string algebra
over tags such as ``pt``, ``pt_PT``, ``pt-BR.UTF-8`` and ``Portuguese_Portugal``.
"""

from __future__ import annotations

import locale
import re
from collections.abc import Iterable

SOURCE_LANGUAGE = "en"
"""The language the messages are written in, which needs no catalog."""

_TAG_RE = re.compile(r"^([A-Za-z]{2,3})(?:_([A-Za-z]{2}|[0-9]{3}))?$")
_NEUTRAL = {"C", "POSIX", "C.UTF-8", "C.UTF8"}


def normalise(tag: str) -> str | None:
    """Put a locale tag into ``ll`` or ``ll_CC`` form.

    Codeset and modifier suffixes (``.UTF-8``, ``@euro``) are dropped, ``-`` becomes
    ``_``, the language is lower-cased and the region upper-cased. The neutral locales
    (``C``, ``POSIX``) mean the source language. A Windows-style name such as
    ``German_Germany`` is looked up in the standard library's alias table.

    Returns:
        The normalised tag, or ``None`` when the text names no language.
    """
    text = tag.strip()
    if not text:
        return None
    if text.upper() in _NEUTRAL:
        return SOURCE_LANGUAGE
    base = text.replace("-", "_").split(".", 1)[0].split("@", 1)[0]
    formatted = _format(base)
    if formatted is not None:
        return formatted
    aliased = locale.normalize(base.lower())
    if aliased == base.lower():
        return None
    return _format(aliased.split(".", 1)[0].split("@", 1)[0])


def match(requested: str, available: Iterable[str]) -> str | None:
    """Choose the catalog that best serves a requested tag.

    An exact match wins. Failing that, any catalog for the same language will do, in
    this order: the region-less one when a region was asked for (``pt_PT`` takes
    ``pt``), then whichever region exists (``pt`` and ``pt_BR`` both take ``pt_PT``).

    Serving one region's catalog to another is a substitution, not a match: European and
    Brazilian Portuguese are not the same translation. It is still worth far more to a
    Brazilian reader than English, so it is offered and then said out loud rather than
    withheld; :func:`llamafit.i18n.select.resolve_language` is what says it.

    Returns:
        The matching tag as it appears in ``available``, or ``None`` when no catalog is
        written in that language at all.
    """
    wanted = normalise(requested)
    if wanted is None:
        return None
    catalogs = {normalised: tag for tag in available if (normalised := normalise(tag)) is not None}
    if wanted in catalogs:
        return catalogs[wanted]
    language = wanted.split("_", 1)[0]
    same = sorted(tag for tag in catalogs if tag.split("_", 1)[0] == language)
    if not same:
        return None
    if "_" in wanted and language in same:
        return catalogs[language]
    return catalogs[same[0]]


def is_substitution(requested: str, found: str) -> bool:
    """True when ``found`` is a different region of the language that was asked for.

    ``pt_BR`` served by ``pt_PT`` is a substitution; ``pt`` served by ``pt_PT``, or
    ``pt_PT`` served by ``pt``, is not, because only one of the two names a region.
    """
    wanted, chosen = normalise(requested), normalise(found)
    if wanted is None or chosen is None:
        return False
    asked_region, given_region = wanted.partition("_")[2], chosen.partition("_")[2]
    return bool(asked_region) and bool(given_region) and asked_region != given_region


def _format(base: str) -> str | None:
    """Return ``ll`` or ``ll_CC`` when ``base`` already has that shape."""
    parsed = _TAG_RE.match(base)
    if parsed is None:
        return None
    language, region = parsed.group(1).lower(), parsed.group(2)
    return language if region is None else f"{language}_{region.upper()}"
