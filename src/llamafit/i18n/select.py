"""Decide which language to speak, and say so when the answer is not what was asked.

The order is most explicit first:

1. the ``--language`` option, passed in as ``requested``;
2. the ``LLAMAFIT_LANGUAGE`` environment variable;
3. the operating system's locale;
4. English.

The first two are somebody asking. When what they asked for has no catalog the choice
carries a notice, because quietly ignoring a request is worse than doing nothing. The
operating system's locale is not a request, so a machine set to a language LlamaFit does
not speak simply gets English without a remark on every run.

There is a third outcome between those two. A request for ``pt_BR`` when only ``pt_PT``
exists is served the Portuguese catalog: a Brazilian reader gets far more out of European
Portuguese than out of English, even where the words are unfamiliar. That substitution
carries a notice whatever asked for it, the operating system included, because the reader
deserves to know why some of the wording looks foreign, and because it is the moment to
invite a catalog for their own region.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from llamafit.i18n.catalogs import available_languages
from llamafit.i18n.detect import LocaleProvider, SystemLocale
from llamafit.i18n.tags import SOURCE_LANGUAGE, is_substitution, match, normalise

LANGUAGE_ENV_VAR = "LLAMAFIT_LANGUAGE"

Source = Literal["option", "environment", "system", "default"]
"""Where the chosen language came from."""


@dataclass(frozen=True)
class LanguageChoice:
    """The language LlamaFit will speak, and how it was arrived at.

    Attributes:
        language: The chosen tag, ``"en"`` when nothing else could be honoured.
        source: Which step of the order decided it.
        requested: What was asked for, when it was not what was given; ``None`` otherwise.
        available: Every language that does have a catalog, English included.
        notice: One sentence for the user when the request was not met exactly, or when
            what was loaded to meet it turned out not to be wholly usable.
        hint: The action that would fix it, when there is one.
    """

    language: str
    source: Source
    requested: str | None = None
    available: tuple[str, ...] = ()
    notice: str | None = None
    hint: str | None = None

    @property
    def honoured(self) -> bool:
        """True when nothing was asked for, or what was asked for was available."""
        return self.requested is None


def resolve_language(
    requested: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    locale_provider: LocaleProvider | None = None,
    available: Sequence[str] | None = None,
) -> LanguageChoice:
    """Work out which language to speak, without loading anything.

    Args:
        requested: The value of the ``--language`` option, if one was given.
        env: The environment to read; the real one when ``None``.
        locale_provider: Where the operating system's locale comes from.
        available: The languages to choose among; the packaged ones when ``None``.

    Returns:
        The choice, carrying a notice when the request was not met exactly: refused
        outright, or served by a catalog for another region of the same language.
    """
    environment = os.environ if env is None else env
    catalogs = tuple(available) if available is not None else available_languages()

    asked_for: tuple[tuple[str | None, Source], ...] = (
        (requested, "option"),
        (environment.get(LANGUAGE_ENV_VAR), "environment"),
    )
    for asked, source in asked_for:
        if asked is None or not asked.strip():
            continue
        found = match(asked, catalogs)
        if found is not None:
            return _chosen(asked, found, source, catalogs)
        return _refused(asked, source, catalogs)

    provider = SystemLocale() if locale_provider is None else locale_provider
    for tag in provider.locale_tags():
        found = match(tag, catalogs)
        if found is not None:
            return _chosen(tag, found, "system", catalogs)
    return LanguageChoice(SOURCE_LANGUAGE, "default", available=catalogs)


def substitution_notice(requested: str, using: str) -> str:
    """The sentence said when one region's catalog stands in for another region's.

    ``using`` is the tag at first, and the catalog's ``Language-Team`` once the catalog
    has been read and can say what it calls itself. That header is data from a file, so
    the sentence it lands in is data too: print it with ``console.print(Text(notice))``
    and never as markup, or a team name holding a square-bracket tag raises at start-up.
    ``docs/translations.md`` shows the call.
    """
    return f"LlamaFit has no {requested} translation, so it is using the {using} one."


def _chosen(asked: str, found: str, source: Source, catalogs: tuple[str, ...]) -> LanguageChoice:
    """Build the choice for a language that was found, substitution noted if it is one."""
    if not is_substitution(asked, found):
        return LanguageChoice(found, source, available=catalogs)
    name = normalise(asked) or asked.strip()
    return LanguageChoice(
        language=found,
        source=source,
        requested=name,
        available=catalogs,
        notice=substitution_notice(name, found),
        hint=f"Contribute a {name} catalog: docs/translations.md says how.",
    )


def _refused(asked: str, source: Source, catalogs: tuple[str, ...]) -> LanguageChoice:
    """Build the choice for a request LlamaFit cannot honour at all."""
    name = normalise(asked) or asked.strip()
    where = "--language" if source == "option" else LANGUAGE_ENV_VAR
    return LanguageChoice(
        language=SOURCE_LANGUAGE,
        source=source,
        requested=name,
        available=catalogs,
        notice=f"LlamaFit does not speak {name}, so it is using English.",
        hint=f"Pass {where} with one of: {', '.join(catalogs)}.",
    )
