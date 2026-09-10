# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The active translation and the functions every other module will call.

``_`` translates one message; ``ngettext`` translates a message that counts something.
``pgettext`` and ``npgettext`` are the same two with a context: a short word saying where
the message is used, so one English word can be translated two ways. *none detected* is
feminine in the GPU row and masculine in the Backends row, and no single Portuguese
string is right in both.

``pgettext_literal`` is ``pgettext`` for the handful of entries that hold punctuation
rather than prose. It differs in one thing: a translation of nothing but whitespace is a
translation, because the space French groups a number's digits with is the answer to that
entry and not the absence of one.

All five are plain module-level functions taking literal strings, so the extractor in
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
from llamafit.i18n.select import LanguageChoice, resolve_language, substitution_notice
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

    def pgettext(self, context: str, message: str) -> str:
        """Translate one message as used in ``context``, falling back to English."""
        ...

    def pgettext_literal(self, context: str, message: str) -> str:
        """Translate one message as used in ``context``, whitespace counting as a value."""
        ...

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Translate a message that counts something, falling back to English."""
        ...

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        """Translate a counting message as used in ``context``, falling back to English."""
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

    def pgettext(self, context: str, message: str) -> str:
        """Return the message unchanged; the context only ever picks a translation."""
        return message

    def pgettext_literal(self, context: str, message: str) -> str:
        """Return the message unchanged: English is the message, blank or not."""
        return message

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Return the singular for one, the plural for every other count."""
        return singular if n == 1 else plural

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        """Return the English form the count selects, whatever the context."""
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

    def pgettext(self, context: str, message: str) -> str:
        """Translate one message through the catalog, under its context."""
        return self.catalog.pgettext(context, message)

    def pgettext_literal(self, context: str, message: str) -> str:
        """Read one entry of the catalog as it stands, whitespace included."""
        return self.catalog.pgettext_literal(context, message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Translate a counting message through the catalog."""
        return self.catalog.ngettext(singular, plural, n)

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        """Translate a counting message through the catalog, under its context."""
        return self.catalog.npgettext(context, singular, plural, n)


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
                # The catalog errors know what is wrong with the file and say what would
                # fix it. Only a missing or unreadable file has nothing to add, and only
                # for that one is a broken install the likeliest explanation.
                hint=(isinstance(exc, ConfigError) and exc.hint)
                or "Reinstall LlamaFit, or report the file the log names.",
            )
        )
    set_translator(CatalogTranslator(choice.language, catalog))
    _log.debug("speaking %s, chosen by %s", choice.language, choice.source)
    return _once(_unusable(_named(choice, catalog), catalog))


def _unusable(choice: LanguageChoice, catalog: PoCatalog) -> LanguageChoice:
    """Say out loud that part of the catalog could not be used, and log each reason.

    A translation whose placeholders would not fill is dropped when the catalog is read,
    so those messages are already showing in English and nothing is going to crash. That
    is the fallback, not the whole answer: whoever wrote the catalog has to be told, and a
    catalog somebody writes themselves never goes near the test suite that would have told
    them. The reasons go to the log, which is where this layer puts detail, and one
    sentence goes to the interface, which is what a user actually reads.

    A notice the choice already carries is left alone. A substitution notice says the
    reader is getting another region's translation, which they need more than this.
    """
    if not catalog.problems:
        return choice
    for problem in catalog.problems:
        _log.debug("%s", problem)
    if choice.notice is not None:
        return choice
    count = len(catalog.problems)
    messages = "message" if count == 1 else "messages"
    return replace(
        choice,
        notice=f"LlamaFit could not use {count} {messages} in its {choice.language} "
        "translation, so those are in English.",
        hint="Run again with --verbose; the log names each one. "
        "docs/translations.md explains placeholders.",
    )


def _named(choice: LanguageChoice, catalog: PoCatalog) -> LanguageChoice:
    """Let a substitution notice call the catalog what the catalog calls itself.

    ``resolve_language`` only knows the tag, because it has read no file yet. Once the
    catalog is open it can say ``Portuguese (Portugal)`` instead of ``pt_PT``, which is
    the difference between a reader understanding why the wording looks foreign and not.
    """
    team = catalog.headers.get("Language-Team", "").strip()
    if choice.notice is None or choice.requested is None or not team:
        return choice
    return replace(choice, notice=substitution_notice(choice.requested, team))


def gettext(message: str) -> str:
    """Translate one message into the language currently installed.

    Returns:
        The translation, or the English original when there is none.
    """
    return _active.gettext(message)


def pgettext(context: str, message: str) -> str:
    """Translate one message as it is used in ``context``.

    Use this, not :func:`gettext`, wherever an English word has to be translated two
    ways because two places use it differently. The context is a short note naming the
    place, written for the translator: ``pgettext("GPU", "none detected")``.

    Args:
        context: Where the message is used, which is part of what identifies it.
        message: The English message.

    Returns:
        The translation filed under that context, or the English original.
    """
    return _active.pgettext(context, message)


def pgettext_literal(context: str, message: str) -> str:
    """Translate one message under ``context``, with whitespace counting as a value.

    The lookup for an entry that holds punctuation rather than prose. :func:`pgettext`
    treats a translation of nothing but whitespace as untranslated, because a blank line
    is not what a half-finished sentence should degrade to; that rule makes the space
    French, Russian, Swedish and Polish group a number's digits with unsayable. This
    lookup takes the catalog at its word, and only an entry with no characters at all
    falls back to the English.

    Use it only where the code is asking for a character. Everything a reader reads as
    words goes through :func:`pgettext`.

    Args:
        context: Where the message is used, which is part of what identifies it.
        message: The English message.

    Returns:
        The translation filed under that context, exactly as the catalog spells it, or
        the English original when the catalog leaves it empty.
    """
    return _active.pgettext_literal(context, message)


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


def npgettext(context: str, singular: str, plural: str, n: int) -> str:
    """Translate a counting message as it is used in ``context``.

    Args:
        context: Where the message is used, which is part of what identifies it.
        singular: The English message for one thing.
        plural: The English message for several.
        n: The count the sentence is about.

    Returns:
        The translated form filed under that context, or the matching English form.
    """
    return _active.npgettext(context, singular, plural, n)


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
