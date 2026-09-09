"""Read a GNU gettext ``.po`` catalog.

The reader understands what a translator's tool writes: ``msgctxt``, ``msgid``,
``msgstr``, ``msgid_plural``, ``msgstr[n]``, a string split over adjacent lines, the
standard backslash escapes, ``#`` comment lines (obsolete ``#~`` entries included) and
the header entry whose ``msgid`` is empty.

``msgctxt`` is how one English word is translated two ways. *none detected* is the value
of the GPU row, where the noun is feminine, and of the Backends row, where it is
masculine, and no single Portuguese string is right in both. An entry is therefore keyed
by its context and its message together. A message written without a context is keyed
under a context of ``None``, which is not the same thing as a context of ``""``.

An empty ``msgstr`` means untranslated. Lookup then returns the English original,
never the empty string: a half-finished translation degrades to English, never to a
blank line on someone's screen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from llamafit.errors import ConfigError
from llamafit.i18n.plurals import (
    PluralFormsError,
    PluralRule,
    default_plural_rule,
    parse_plural_forms,
)

_ESCAPES = {
    "\\": "\\",
    '"': '"',
    "'": "'",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}
_KEYWORD_RE = re.compile(r"^(msgctxt|msgid_plural|msgid|msgstr)(?:\[(\d+)\])?\s*(.*)$")
_WORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_STARTS_AN_ENTRY = frozenset({"msgctxt", "msgid"})
_PLURAL_FORMS = "Plural-Forms"
_LANGUAGE = "Language"
_MAX_PLURAL_INDEX = 9

MessageKey = tuple[str | None, str]
"""What identifies an entry: its context, ``None`` when it has none, and its ``msgid``."""


class PoSyntaxError(ConfigError):
    """A ``.po`` file could not be read.

    Attributes:
        source: The file the problem is in, or ``"<string>"`` for parsed text.
        line: The 1-based line the problem is on.
    """

    def __init__(self, source: str, line: int, message: str) -> None:
        super().__init__(
            f"{source}: line {line}: {message}",
            hint="Correct the .po file, then check it with `msgfmt --check`.",
        )
        self.source = source
        self.line = line


@dataclass(frozen=True)
class Message:
    """One entry in a catalog.

    Attributes:
        msgid: The English message, which with the context is the lookup key.
        plural: The English plural message, or ``None`` for a singular-only entry.
        translations: One string per plural form; an empty string means untranslated.
        line: The line the entry starts on, for error messages.
        context: What the ``msgctxt`` said, or ``None`` for an entry that has none.
    """

    msgid: str
    plural: str | None
    translations: tuple[str, ...]
    line: int
    context: str | None = None

    @property
    def key(self) -> MessageKey:
        """What this entry is filed under: its context and its message together."""
        return (self.context, self.msgid)

    @property
    def translated(self) -> bool:
        """True when at least one form carries a non-empty translation."""
        return any(self.translations)


@dataclass(frozen=True)
class PoCatalog:
    """A parsed catalog, ready to answer lookups.

    Attributes:
        language: The tag the ``Language`` header names, or the empty string.
        headers: The header entry's fields, in the order they were written.
        messages: Every entry except the header, keyed by context and ``msgid``.
        plural_forms: The raw ``Plural-Forms`` header, or ``None`` when it is absent.
        plural_rule: The rule to apply; English's when the header is absent.
        source: Where the catalog was read from, for error messages.
    """

    language: str
    headers: Mapping[str, str]
    messages: Mapping[MessageKey, Message]
    plural_forms: str | None
    plural_rule: PluralRule
    source: str = "<string>"

    def gettext(self, message: str) -> str:
        """Translate one message that carries no context.

        Returns:
            The translation, or the English original when the catalog has no entry for
            it or the entry is untranslated.
        """
        return self._one(None, message)

    def pgettext(self, context: str, message: str) -> str:
        """Translate one message as it is used in ``context``.

        A translation filed without a context is not used here, and the other way round:
        the whole point of the context is that the two are different sentences.

        Returns:
            The translation filed under that context, or the English original.
        """
        return self._one(context, message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        """Translate a message that counts something, picking the form ``n`` selects.

        Returns:
            The translated form, or the English singular or plural when the catalog has
            no usable translation for the form ``n`` selects.
        """
        return self._counted(None, singular, plural, n)

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        """Translate a counting message as it is used in ``context``.

        Returns:
            The translated form filed under that context, or the matching English form.
        """
        return self._counted(context, singular, plural, n)

    def untranslated(self) -> tuple[MessageKey, ...]:
        """Return the key of every entry still untranslated, in file order."""
        return tuple(key for key, entry in self.messages.items() if not entry.translated)

    def _one(self, context: str | None, message: str) -> str:
        """Look one message up under ``context``, falling back to the English."""
        entry = self.messages.get((context, message))
        if entry is None:
            return message
        return _usable(entry.translations, 0) or message

    def _counted(self, context: str | None, singular: str, plural: str, n: int) -> str:
        """Look a counting message up under ``context``, falling back to the English."""
        entry = self.messages.get((context, singular))
        if entry is not None:
            translation = _usable(entry.translations, self.plural_rule.index(n))
            if translation:
                return translation
        return singular if n == 1 else plural


@dataclass
class _Entry:
    """One entry under construction while the file is being read."""

    line: int
    context: str | None = None
    msgid: str | None = None
    plural: str | None = None
    translations: dict[int, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """True while nothing has been read into this entry yet."""
        return (
            self.context is None
            and self.msgid is None
            and self.plural is None
            and not self.translations
        )


def parse_po(text: str, *, source: str = "<string>") -> PoCatalog:
    """Parse the text of a ``.po`` file.

    Args:
        text: The whole file, already decoded as UTF-8.
        source: The name to put in error messages, usually the path.

    Returns:
        The catalog, with its header parsed and its plural rule resolved.

    Raises:
        PoSyntaxError: If a line cannot be read, a message is defined twice, or the
            ``Plural-Forms`` header is malformed.
    """
    entries: list[_Entry] = []
    current = _Entry(line=1)
    target: tuple[str, int] | None = None

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            if not current.empty:
                entries.append(current)
            current, target = _Entry(line=number + 1), None
            continue
        if line.startswith("#"):
            continue
        if line.startswith('"'):
            if target is None:
                raise PoSyntaxError(source, number, "a string continues nothing")
            _append(current, target, _unquote(line, source, number))
            continue
        keyword, index, rest = _split(line, source, number)
        if keyword in _STARTS_AN_ENTRY and current.translations:
            entries.append(current)
            current = _Entry(line=number)
        target = _open(current, keyword, index, source, number)
        _append(current, target, _unquote(rest, source, number))

    if not current.empty:
        entries.append(current)
    return _build(entries, source)


def read_po(path: Path) -> PoCatalog:
    """Read and parse a ``.po`` file, always as UTF-8 whatever the platform default is.

    Returns:
        The parsed catalog.

    Raises:
        PoSyntaxError: If the file is not valid UTF-8 or its contents are malformed.
        OSError: If the file cannot be opened.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PoSyntaxError(str(path), 1, f"the file is not valid UTF-8: {exc}") from exc
    return parse_po(text, source=str(path))


def _usable(translations: tuple[str, ...], index: int) -> str:
    """Return the form at ``index``, or the empty string when there is no such form."""
    if index >= len(translations):
        return ""
    return translations[index]


def _split(line: str, source: str, number: int) -> tuple[str, int, str]:
    """Split ``msgstr[1] "text"`` into its keyword, its form index and the string."""
    match = _KEYWORD_RE.match(line)
    if match is None:
        word = _WORD_RE.match(line)
        found = word.group(0) if word else line
        raise PoSyntaxError(source, number, f"unknown keyword {found!r}")
    keyword, index, rest = match.group(1), match.group(2), match.group(3)
    if not rest.startswith('"'):
        raise PoSyntaxError(source, number, f"expected a string after {keyword}")
    form = int(index) if index is not None else 0
    if form > _MAX_PLURAL_INDEX:
        raise PoSyntaxError(source, number, f"plural form {form} is out of range")
    return keyword, form, rest


def _open(entry: _Entry, keyword: str, index: int, source: str, number: int) -> tuple[str, int]:
    """Start a field on the entry and return the key that later string lines extend."""
    if keyword == "msgctxt":
        if entry.context is not None:
            raise PoSyntaxError(source, number, "the entry already has a msgctxt")
        if entry.msgid is not None:
            raise PoSyntaxError(source, number, "a msgctxt must come before its msgid")
        entry.context = ""
        return ("context", 0)
    if keyword == "msgid":
        if entry.msgid is not None:
            raise PoSyntaxError(source, number, "the entry already has a msgid")
        entry.msgid = ""
        return ("msgid", 0)
    if keyword == "msgid_plural":
        if entry.plural is not None:
            raise PoSyntaxError(source, number, "the entry already has a msgid_plural")
        entry.plural = ""
        return ("plural", 0)
    if index in entry.translations:
        raise PoSyntaxError(source, number, f"the entry already has msgstr[{index}]")
    entry.translations[index] = ""
    return ("msgstr", index)


def _append(entry: _Entry, target: tuple[str, int], value: str) -> None:
    """Append text to the field ``target`` names; adjacent strings concatenate."""
    name, index = target
    if name == "context":
        entry.context = (entry.context or "") + value
    elif name == "msgid":
        entry.msgid = (entry.msgid or "") + value
    elif name == "plural":
        entry.plural = (entry.plural or "") + value
    else:
        entry.translations[index] = entry.translations.get(index, "") + value


def _unquote(raw: str, source: str, number: int) -> str:
    """Turn one quoted ``.po`` string literal into the text it stands for."""
    text = raw.strip()
    if len(text) < 2 or not text.startswith('"') or not text.endswith('"'):
        raise PoSyntaxError(source, number, "expected a double-quoted string")
    body = text[1:-1]
    out: list[str] = []
    position = 0
    while position < len(body):
        char = body[position]
        if char == '"':
            raise PoSyntaxError(source, number, "an unescaped quote ends the string early")
        if char != "\\":
            out.append(char)
            position += 1
            continue
        if position + 1 >= len(body):
            raise PoSyntaxError(source, number, "the string ends with a lone backslash")
        escape = body[position + 1]
        if escape not in _ESCAPES:
            raise PoSyntaxError(source, number, f"unknown escape \\{escape}")
        out.append(_ESCAPES[escape])
        position += 2
    return "".join(out)


def _forms(entry: _Entry) -> tuple[str, ...]:
    """Flatten the entry's ``msgstr[n]`` map into one string per form, gaps empty."""
    if not entry.translations:
        return ()
    return tuple(entry.translations.get(i, "") for i in range(max(entry.translations) + 1))


def _headers(text: str) -> dict[str, str]:
    """Parse the header entry's ``Name: value`` lines; lines without a colon are ignored."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip():
            fields[name.strip()] = value.strip()
    return fields


def _describe(key: MessageKey) -> str:
    """Name an entry in an error message, saying which context it is in when it has one."""
    context, msgid = key
    return f"{msgid!r}" if context is None else f"{msgid!r} in context {context!r}"


def _build(entries: list[_Entry], source: str) -> PoCatalog:
    """Turn the accumulated entries into a catalog, validating as it goes."""
    headers: dict[str, str] = {}
    header_line = 1
    seen_header = False
    messages: dict[MessageKey, Message] = {}

    for entry in entries:
        if entry.msgid is None:
            raise PoSyntaxError(source, entry.line, "an entry has a translation but no msgid")
        forms = _forms(entry)
        if entry.msgid == "" and entry.plural is None and entry.context is None:
            if seen_header:
                raise PoSyntaxError(source, entry.line, "the file has two header entries")
            seen_header, header_line = True, entry.line
            headers = _headers(forms[0] if forms else "")
            continue
        key = (entry.context, entry.msgid)
        if key in messages:
            raise PoSyntaxError(source, entry.line, f"duplicate message {_describe(key)}")
        messages[key] = Message(
            msgid=entry.msgid,
            plural=entry.plural,
            translations=forms,
            line=entry.line,
            context=entry.context,
        )

    plural_forms = headers.get(_PLURAL_FORMS)
    if plural_forms is None:
        rule = default_plural_rule()
    else:
        try:
            rule = parse_plural_forms(plural_forms)
        except PluralFormsError as exc:
            raise PoSyntaxError(source, header_line, str(exc)) from exc
    return PoCatalog(
        language=headers.get(_LANGUAGE, ""),
        headers=headers,
        messages=messages,
        plural_forms=plural_forms,
        plural_rule=rule,
        source=source,
    )
