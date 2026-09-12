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
request could not be honoured, when it was honoured by another region's catalog, or when
the catalog that was loaded turns out to be only part written. That sentence is handed
over once per process, so a request for a language LlamaFit does not speak is reported
and not repeated.

The choice looks at the person, and, when the interface says where the words are going,
at the stream too. A catalog whose script the stream's encoding cannot write is not
installed: a screen of ``?`` is not a translation, so English is spoken and the notice
says which encoding refused which language. :mod:`llamafit.i18n.encoding` argues that
rule and draws the line; this module only asks it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from llamafit.errors import ConfigError
from llamafit.i18n.catalogs import load_language
from llamafit.i18n.completeness import (
    SHORTFALL_HINT,
    completeness,
    shortfall_after,
    shortfall_notice,
)
from llamafit.i18n.detect import LocaleProvider
from llamafit.i18n.encoding import can_write
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
    encoding: str | None = None,
) -> LanguageChoice:
    """Choose a language, load its catalog and install it.

    A catalog that cannot be read is a broken install, not a reason to fail: it is
    logged, English is installed instead, and the choice comes back with a notice. A
    catalog the output stream cannot write is not a reason to fail either, and ends the
    same way: English, and a notice naming the encoding and the language it refused.

    Args:
        requested: The value of the ``--language`` option, if one was given.
        env: The environment to read; the real one when ``None``.
        locale_provider: Where the operating system's locale comes from.
        available: The languages to choose among; the packaged ones when ``None``.
        directory: Where the catalogs live; the packaged directory when ``None``.
        encoding: What the output stream writes in, when the interface has one to name;
            ``None`` skips the question, which is right for an interface that draws its
            own screen and for a test that has no stream.

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
    if encoding is not None and not can_write(_everything_said(catalog), encoding):
        _log.debug("%s cannot write the %s catalog", encoding, choice.language)
        set_translator(EnglishTranslator())
        return _once(_unwritable(choice, catalog, encoding))
    set_translator(CatalogTranslator(choice.language, catalog))
    _log.debug("speaking %s, chosen by %s", choice.language, choice.source)
    named = _named(choice, catalog)
    return _once(_unusable(_incomplete(named, catalog, directory=directory), catalog))


def _incomplete(
    choice: LanguageChoice, catalog: PoCatalog, *, directory: Path | None = None
) -> LanguageChoice:
    """Say out loud when the catalog that was loaded is only part written.

    Every catalog that ships is finished today, and this is here for the ones that will
    not be: a catalog is filled in from the top of the template down, so what a
    part-written one has is the short fragments and what it lacks is the prose. A reader
    who asked for such a language meets their own words in the column headings and English
    in every sentence that explains anything, with nothing anywhere to say which of the
    two they are looking at. That is the notice.

    A substitution notice is not dropped for this one and does not swallow it either: a
    request for a region no catalog covers, served by a catalog that is itself a quarter
    written, is two things wrong with the same run, and a reader told only the first would
    put the English down to the region they did not get. The second sentence is added to
    the first, and the hint stays the substitution's, which already points at the file
    this one would have pointed at.

    This runs before :func:`_unusable` rather than after it, so when a catalog is both
    part written and carrying a message the reader had to drop, the line goes to the one
    about most of the interface instead of the one about a handful of entries.
    """
    measured = completeness(catalog, directory=directory)
    _log.debug(
        "the %s catalog has %d of %d messages",
        choice.language,
        measured.translated,
        measured.total,
    )
    if measured.nearly_complete:
        return choice
    if choice.notice is not None:
        return replace(choice, notice=shortfall_after(choice.notice, measured))
    team = catalog.headers.get("Language-Team", "").strip()
    return replace(
        choice,
        notice=shortfall_notice(team or choice.language, measured),
        hint=SHORTFALL_HINT,
    )


def _everything_said(catalog: PoCatalog) -> str:
    """Every translation in the catalog as one text, which is what a stream is judged on."""
    return "\n".join(form for message in catalog.messages.values() for form in message.translations)


def _unwritable(choice: LanguageChoice, catalog: PoCatalog, encoding: str) -> LanguageChoice:
    """The choice for a catalog the stream cannot write: English, and the reason.

    The notice names the language the way the catalog names itself, as the substitution
    notice does, because *Japanese* tells a reader more than *ja*. It names the encoding
    the way the stream reports it, because that is the name they can search for. And the
    hint names the one setting that makes a Python stream write UTF-8 whatever the
    platform's default is, since the stream is the thing at fault and the person can fix
    it without touching LlamaFit.

    Whatever notice the choice carried before is replaced rather than kept. A substitution
    notice explains why the wording looks foreign, and no wording is about to.
    """
    name = catalog.headers.get("Language-Team", "").strip() or choice.language
    return replace(
        choice,
        language=SOURCE_LANGUAGE,
        requested=choice.requested or choice.language,
        notice=f"This output is written as {encoding}, which cannot carry {name}, "
        "so LlamaFit is using English.",
        hint="Set PYTHONUTF8=1 in the environment and run again: Python then writes "
        "UTF-8, which carries every language.",
    )


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
