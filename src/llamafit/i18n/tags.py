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

    An exact match wins. Otherwise a request without a region takes the most specific
    catalog for that language (``pt`` takes ``pt_PT``), and a request with a region
    takes the region-less catalog for the same language (``pt_PT`` takes ``pt``). A
    request for one region is never served by another: ``pt_BR`` does not take
    ``pt_PT``, because European and Brazilian Portuguese are not the same translation.

    Returns:
        The matching tag as it appears in ``available``, or ``None``.
    """
    wanted = normalise(requested)
    if wanted is None:
        return None
    catalogs = {normalised: tag for tag in available if (normalised := normalise(tag)) is not None}
    if wanted in catalogs:
        return catalogs[wanted]
    language, separator, _region = wanted.partition("_")
    if not separator:
        wider = sorted(tag for tag in catalogs if tag.split("_", 1)[0] == language)
        return catalogs[wider[0]] if wider else None
    return catalogs.get(language)


def _format(base: str) -> str | None:
    """Return ``ll`` or ``ll_CC`` when ``base`` already has that shape."""
    parsed = _TAG_RE.match(base)
    if parsed is None:
        return None
    language, region = parsed.group(1).lower(), parsed.group(2)
    return language if region is None else f"{language}_{region.upper()}"
