"""Messages translated when they are rendered, not when they are defined.

Anything built while a module is imported is built before an interface has chosen a
language: a module-level constant, a table of hints, an argument to a decorator.
Translating such a string with :func:`llamafit.i18n.gettext` at that moment freezes it in
English for the life of the process, and does so **silently** — the interface comes out
half translated, nothing raises, and nobody goes looking.

:func:`lazy_gettext` and :func:`lazy_ngettext` defer the lookup to the moment the text is
rendered, so a constant defined at import time obeys a language chosen afterwards.

The result is a :class:`LazyString`, not a ``str``. That is deliberate. A ``str``
subclass would carry a frozen English buffer that C-level fast paths such as ``str.join``
would use in preference to any method this module could override, and the failure would
be another silent English message. A separate type fails loudly instead: ``", ".join``
over a lazy message raises ``TypeError`` at the call site, where the fix is to render it
with ``str()`` — which is the right thing to do there anyway, because that call site *is*
the render moment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from llamafit.i18n.translator import gettext, ngettext, npgettext, pgettext


class LazyString:
    """A message whose translation is looked up every time the text is needed.

    It stands in for a string: it renders, formats, compares, sorts, concatenates,
    indexes, and forwards every string method to the text it currently resolves to.
    It is not a ``str``, so anything that demands one raises rather than quietly
    using English.

    Because the text it resolves to changes with the language, so does its hash. Do not
    use one as a dictionary key across a call to
    :func:`llamafit.i18n.translator.set_language`.

    Attributes:
        message: The English message, which never changes; the extractor reads this.
    """

    __slots__ = ("_render", "message")

    def __init__(self, message: str, render: Callable[[], str]) -> None:
        self.message = message
        self._render = render

    def __str__(self) -> str:
        """The message in the language installed right now."""
        return self._render()

    def __repr__(self) -> str:
        """Show that the message is deferred, and what it resolves to at the moment."""
        return f"LazyString({self.message!r} -> {self._render()!r})"

    def __format__(self, spec: str) -> str:
        """Render inside an f-string or ``format`` call."""
        return format(self._render(), spec)

    def __eq__(self, other: object) -> bool:
        """Equal to the text it resolves to, and to another lazy message that matches."""
        if isinstance(other, LazyString):
            return self._render() == other._render()
        if isinstance(other, str):
            return self._render() == other
        return NotImplemented

    def __hash__(self) -> int:
        """Hash the text it resolves to, so it matches that text in a set."""
        return hash(self._render())

    def __lt__(self, other: str | LazyString) -> bool:
        """Sort by the rendered text."""
        return self._render() < str(other)

    def __le__(self, other: str | LazyString) -> bool:
        """Sort by the rendered text."""
        return self._render() <= str(other)

    def __gt__(self, other: str | LazyString) -> bool:
        """Sort by the rendered text."""
        return self._render() > str(other)

    def __ge__(self, other: str | LazyString) -> bool:
        """Sort by the rendered text."""
        return self._render() >= str(other)

    def __len__(self) -> int:
        """The length of the rendered text."""
        return len(self._render())

    def __bool__(self) -> bool:
        """True when the rendered text is not empty."""
        return bool(self._render())

    def __iter__(self) -> Iterator[str]:
        """Iterate the characters of the rendered text."""
        return iter(self._render())

    def __getitem__(self, index: int | slice) -> str:
        """Index or slice the rendered text."""
        return self._render()[index]

    def __contains__(self, part: str) -> bool:
        """Search the rendered text."""
        return part in self._render()

    def __add__(self, other: str) -> str:
        """Concatenate onto the rendered text."""
        return self._render() + other

    def __radd__(self, other: str) -> str:
        """Concatenate the rendered text onto something else."""
        return other + self._render()

    def __mod__(self, values: object) -> str:
        """Fill the rendered text's placeholders, as ``"%(count)d" % {...}`` does."""
        return self._render() % values

    def __getattr__(self, name: str) -> Any:
        """Forward every string method to the text it currently resolves to."""
        return getattr(self._render(), name)


def lazy_gettext(message: str) -> LazyString:
    """Translate ``message`` every time it is rendered, rather than now.

    Use this, not :func:`llamafit.i18n.gettext`, for anything built while a module is
    imported: a module-level constant, a dictionary of hints, an argument to a decorator.

    Returns:
        A :class:`LazyString` standing in for the message.
    """
    return LazyString(message, lambda: gettext(message))


def lazy_pgettext(context: str, message: str) -> LazyString:
    """Translate ``message`` under ``context`` every time it is rendered, rather than now.

    Returns:
        A :class:`LazyString` standing in for the message in that context.
    """
    return LazyString(message, lambda: pgettext(context, message))


def lazy_ngettext(singular: str, plural: str, n: int) -> LazyString:
    """Translate a counting message every time it is rendered, rather than now.

    Returns:
        A :class:`LazyString` standing in for the form ``n`` selects.
    """
    return LazyString(singular, lambda: ngettext(singular, plural, n))


def lazy_npgettext(context: str, singular: str, plural: str, n: int) -> LazyString:
    """Translate a counting message under ``context`` when it is rendered, not now.

    Returns:
        A :class:`LazyString` standing in for the form ``n`` selects in that context.
    """
    return LazyString(singular, lambda: npgettext(context, singular, plural, n))
