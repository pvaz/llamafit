# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Every key this dashboard answers to, written down once.

Two things need this list and they must not be able to disagree: the bindings Textual
dispatches, and the bar along the bottom that tells a reader the key exists. A screen whose
bar advertises a key nothing is bound to is worse than one with no bar, so both are built
from :data:`BOARD_KEYS` and its neighbours rather than typed out twice.

Every description is deferred. These tables are filled while the module is imported, which
on the command line's own account happens long before ``--language`` has been read, so an
eager ``_()`` here would freeze the key bar in English while the rest of the screen spoke
Portuguese and nothing anywhere would fail. :class:`~llamafit.i18n.LazyString` is rendered
by :func:`bar`, which runs at draw time, after a language has been chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llamafit.i18n import LazyString, lazy_gettext


@dataclass(frozen=True)
class Key:
    """One key: what Textual calls it, what it does, and what a reader is told.

    Attributes:
        name: Textual's own name for the key, which is what a binding takes.
        action: The action method it calls, without the ``action_`` prefix. Empty for a
            key a widget already handles on its own, such as Enter inside a table, which
            the bar still has to mention because a reader cannot see a binding.
        shown: What a reader presses, as they would say it: ``/``, not ``slash``.
        description: What it does, translated when the bar is drawn.
    """

    name: str
    action: str
    shown: str
    description: LazyString


GLOBAL_KEYS: tuple[Key, ...] = (
    Key("question_mark", "help", "?", lazy_gettext("keys")),
    Key("t", "next_theme", "t", lazy_gettext("theme")),
    Key("q", "quit", "q", lazy_gettext("quit")),
)
"""What works on every screen. Tab and Shift+Tab move between screens; Textual binds those."""

BOARD_KEYS: tuple[Key, ...] = (
    Key("", "", "↑↓", lazy_gettext("choose")),
    Key("", "", "Enter", lazy_gettext("why")),
    Key("p", "plan", "p", lazy_gettext("command line")),
    Key("slash", "search", "/", lazy_gettext("search")),
    Key("f", "cycle_fit", "f", lazy_gettext("fit")),
    Key("s", "cycle_sort", "s", lazy_gettext("sort")),
    Key("S", "previous_sort", "S", lazy_gettext("sort back")),
    Key("o", "reverse_sort", "o", lazy_gettext("reverse")),
    Key("a", "toggle_installed", "a", lazy_gettext("on disk")),
    Key("A", "toggle_quants", "A", lazy_gettext("all quants")),
    Key("e", "toggle_excluded", "e", lazy_gettext("unranked rows")),
    Key("c", "toggle_columns", "c", lazy_gettext("all columns")),
    Key("x", "toggle_why", "x", lazy_gettext("explanation")),
    Key("n", "not_ranked", "n", lazy_gettext("not ranked")),
)

NEEDS_KEYS: tuple[Key, ...] = (
    Key("ctrl+s", "apply", "Ctrl+S", lazy_gettext("apply")),
    Key("ctrl+r", "reset", "Ctrl+R", lazy_gettext("reset")),
)

HOST_KEYS: tuple[Key, ...] = (Key("R", "rescan", "R", lazy_gettext("rescan")),)

PLAN_KEYS: tuple[Key, ...] = (
    Key("plus", "longer", "+", lazy_gettext("more context")),
    Key("minus", "shorter", "-", lazy_gettext("less context")),
    Key("v", "toggle_vision", "v", lazy_gettext("vision")),
    Key("y", "copy", "y", lazy_gettext("copy")),
)

SIMULATE_KEYS: tuple[Key, ...] = (
    Key("ctrl+s", "apply", "Ctrl+S", lazy_gettext("apply")),
    Key("ctrl+r", "reset", "Ctrl+R", lazy_gettext("this machine")),
)


def bindable(keys: Sequence[Key]) -> list[Key]:
    """The keys that need a binding: the ones with an action of their own."""
    return [key for key in keys if key.name and key.action]


def bar(keys: Sequence[Key]) -> str:
    """The keys as one line, in the reader's language, for the bottom of a screen."""
    return "   ".join(f"{key.shown} {key.description}" for key in keys)
