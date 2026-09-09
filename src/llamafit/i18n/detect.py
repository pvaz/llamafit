"""Ask the operating system which language the user reads.

This is the one part of translation that touches the operating system, so it goes behind
a ``LocaleProvider`` and a test replaces it with ``FixedLocale``, the same way every
external command goes behind ``Runner``.

The three platforms disagree about where the answer lives:

* On Linux and macOS the environment holds it: ``LANGUAGE`` (a colon-separated list of
  preferences), then ``LC_ALL``, ``LC_MESSAGES`` and ``LANG``.
* On Windows those variables are normally unset. The answer is the user interface
  language, ``GetUserDefaultUILanguage`` in ``kernel32``, whose language identifier the
  standard library's ``locale.windows_locale`` table turns into a tag such as ``pt_PT``.
  ``locale.getlocale()`` is no help there: it returns ``Portuguese_Portugal``, a name the
  alias table does not know.
* ``locale.getlocale()`` is consulted last, for the platforms where it does answer.
"""

from __future__ import annotations

import locale
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

_ENVIRONMENT_LISTS = ("LANGUAGE",)
_ENVIRONMENT_SINGLES = ("LC_ALL", "LC_MESSAGES", "LANG")


class LocaleProvider(Protocol):
    """Something that can say which languages the user prefers."""

    def locale_tags(self) -> tuple[str, ...]:
        """Return the user's locale tags, most preferred first, unnormalised."""
        ...


@dataclass(frozen=True)
class FixedLocale:
    """A provider that answers with whatever a test gave it."""

    tags: Sequence[str] = ()

    def locale_tags(self) -> tuple[str, ...]:
        """Return the tags this provider was built with."""
        return tuple(self.tags)


def windows_ui_language() -> str | None:
    """Return the Windows user interface language as a locale tag.

    Returns:
        A tag such as ``pt_PT``, or ``None`` off Windows and whenever the call or the
        lookup does not produce one.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            language_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
        except (AttributeError, OSError, ValueError):  # a locked-down or unusual host
            return None
        return locale.windows_locale.get(language_id)
    return None


def current_locale() -> str | None:
    """Return the process locale's language tag, or ``None`` when it has none."""
    try:
        return locale.getlocale()[0]
    except ValueError:  # an unparsable locale setting is not worth raising over
        return None


@dataclass(frozen=True)
class SystemLocale:
    """Reads the user's languages from the environment, then from the operating system.

    Attributes:
        env: The environment to read; the real one when ``None``.
        ui_language: Returns the Windows interface language; replaced in tests.
        process_locale: Returns the process locale; replaced in tests.
    """

    env: Mapping[str, str] | None = None
    ui_language: Callable[[], str | None] = windows_ui_language
    process_locale: Callable[[], str | None] = current_locale

    def locale_tags(self) -> tuple[str, ...]:
        """Return every tag the operating system offers, most preferred first."""
        env = os.environ if self.env is None else self.env
        tags: list[str] = []
        for name in _ENVIRONMENT_LISTS:
            tags.extend(part for part in env.get(name, "").split(":") if part)
        for name in _ENVIRONMENT_SINGLES:
            value = env.get(name, "").strip()
            if value:
                tags.append(value)
        for reported in (self.ui_language(), self.process_locale()):
            if reported:
                tags.append(reported)
        return tuple(tags)
