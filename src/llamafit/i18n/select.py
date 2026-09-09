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
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from llamafit.i18n.catalogs import available_languages
from llamafit.i18n.detect import LocaleProvider, SystemLocale
from llamafit.i18n.tags import SOURCE_LANGUAGE, match, normalise

LANGUAGE_ENV_VAR = "LLAMAFIT_LANGUAGE"

Source = Literal["option", "environment", "system", "default"]
"""Where the chosen language came from."""


@dataclass(frozen=True)
class LanguageChoice:
    """The language LlamaFit will speak, and how it was arrived at.

    Attributes:
        language: The chosen tag, ``"en"`` when nothing else could be honoured.
        source: Which step of the order decided it.
        requested: What was asked for, when it could not be honoured; ``None`` otherwise.
        available: Every language that does have a catalog, English included.
        notice: One sentence for the user when a request was not honoured.
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
        The choice, carrying a notice when an explicit request could not be honoured.
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
            return LanguageChoice(found, source, available=catalogs)
        return _refused(asked, source, catalogs)

    provider = SystemLocale() if locale_provider is None else locale_provider
    for tag in provider.locale_tags():
        found = match(tag, catalogs)
        if found is not None:
            return LanguageChoice(found, "system", available=catalogs)
    return LanguageChoice(SOURCE_LANGUAGE, "default", available=catalogs)


def _refused(asked: str, source: Source, catalogs: tuple[str, ...]) -> LanguageChoice:
    """Build the choice for a request LlamaFit cannot honour."""
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
