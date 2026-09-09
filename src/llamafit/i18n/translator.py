"""The active translation and the two functions every other module will call.

``_`` translates one message; ``ngettext`` translates a message that counts something.
Both are plain module-level functions taking literal strings, so the extractor in
``scripts/gen_messages.py`` can find every call by reading the syntax tree.

Choosing a language is a one-off at start-up: ``set_language`` installs the translator
and returns what it decided, including the sentence the interface should print when a
request could not be honoured. That sentence is handed over once per process, so a
request for a language LlamaFit does not speak is reported and not repeated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from llamafit.errors import ConfigError
from llamafit.i18n.catalogs import load_language
from llamafit.i18n.detect import LocaleProvider
from llamafit.i18n.po import PoCatalog
from llamafit.i18n.select import LanguageChoice, resolve_language
from llamafit.i18n.tags import SOURCE_LANGUAGE
from llamafit.logging import get_logger

_log = get_logger("i18n")


class Translator(Protocol):
    """Something that can turn an English message into the user's language."""

    @property
    def language(self) -> str:
        """The tag of the language this translator speaks."""
        ...

    def gettext(self, message: str) -> str:
        """Translate one message, falling back to the English original."""
        ...

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Translate a message that counts something, falling back to English."""
        ...


@dataclass(frozen=True)
class EnglishTranslator:
    """The source language: every lookup returns the message it was given."""

    @property
    def language(self) -> str:
        """Always ``"en"``."""
        return SOURCE_LANGUAGE

    def gettext(self, message: str) -> str:
        """Return the message unchanged."""
        return message

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Return the singular for one, the plural for every other count."""
        return singular if n == 1 else plural


@dataclass(frozen=True)
class CatalogTranslator:
    """Translates through a parsed catalog, under the tag the catalog was chosen as.

    Attributes:
        tag: The language tag, which is what ``current_language`` reports.
        catalog: The parsed ``.po`` file.
    """

    tag: str
    catalog: PoCatalog

    @property
    def language(self) -> str:
        """The tag this translator was loaded as."""
        return self.tag

    def gettext(self, message: str) -> str:
        """Translate one message through the catalog."""
        return self.catalog.gettext(message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Translate a counting message through the catalog."""
        return self.catalog.ngettext(singular, plural, n)


_active: Translator = EnglishTranslator()
_announced: set[str] = set()


def get_translator() -> Translator:
    """Return the translator currently installed."""
    return _active


def set_translator(translator: Translator) -> None:
    """Install a translator directly, which is what tests do."""
    global _active
    _active = translator


def current_language() -> str:
    """Return the tag of the language LlamaFit is speaking."""
    return _active.language


def reset() -> None:
    """Go back to English and forget which notices have been given.

    Only tests need this; a real process chooses its language once.
    """
    global _active
    _active = EnglishTranslator()
    _announced.clear()


def set_language(
    requested: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    locale_provider: LocaleProvider | None = None,
    available: Sequence[str] | None = None,
    directory: Path | None = None,
) -> LanguageChoice:
    """Choose a language, load its catalog and install it.

    A catalog that cannot be read is a broken install, not a reason to fail: it is
    logged, English is installed instead, and the choice comes back with a notice.

    Args:
        requested: The value of the ``--language`` option, if one was given.
        env: The environment to read; the real one when ``None``.
        locale_provider: Where the operating system's locale comes from.
        available: The languages to choose among; the packaged ones when ``None``.
        directory: Where the catalogs live; the packaged directory when ``None``.

    Returns:
        The choice. Its ``notice`` is set only the first time a given problem is met in
        this process, so the interface prints it once.
    """
    choice = resolve_language(
        requested, env=env, locale_provider=locale_provider, available=available
    )
    if choice.language == SOURCE_LANGUAGE:
        set_translator(EnglishTranslator())
        return _once(choice)
    try:
        catalog = load_language(choice.language, directory=directory)
    except (ConfigError, OSError) as exc:
        _log.debug("could not read the %s catalog: %s", choice.language, exc)
        set_translator(EnglishTranslator())
        return _once(
            replace(
                choice,
                language=SOURCE_LANGUAGE,
                requested=choice.language,
                notice=f"LlamaFit could not read its {choice.language} translation, "
                "so it is using English.",
                hint="Reinstall LlamaFit, or report the file the log names.",
            )
        )
    set_translator(CatalogTranslator(choice.language, catalog))
    _log.debug("speaking %s, chosen by %s", choice.language, choice.source)
    return _once(choice)


def gettext(message: str) -> str:
    """Translate one message into the language currently installed.

    Returns:
        The translation, or the English original when there is none.
    """
    return _active.gettext(message)


def ngettext(singular: str, plural: str, n: int) -> str:
    """Translate a message that counts something, choosing the form ``n`` needs.

    Args:
        singular: The English message for one thing.
        plural: The English message for several.
        n: The count the sentence is about.

    Returns:
        The translated form, or the matching English form when there is none.
    """
    return _active.ngettext(singular, plural, n)


_ = gettext
"""The short name for :func:`gettext`, which is what call sites use."""


def _once(choice: LanguageChoice) -> LanguageChoice:
    """Blank the notice when this process has already given it.

    The notice is logged at debug level only. Saying it out loud is the interface's job:
    a library that writes to the console would talk over ``--json``, and the logger has
    no handler until ``--verbose`` installs one.
    """
    if choice.notice is None:
        return choice
    key = f"{choice.source}:{choice.requested}"
    if key in _announced:
        return replace(choice, notice=None, hint=None)
    _announced.add(key)
    _log.debug("%s", choice.notice)
    return choice
