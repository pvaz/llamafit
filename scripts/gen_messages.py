"""Write the message template by reading the syntax tree of every Python source.

Usage:
    python scripts/gen_messages.py [path ...]

Finds every call to the translation functions and writes ``messages.pot``, the catalog a
translator copies to start a language. Parsing is done with :mod:`ast`, never with a
regular expression: a regular expression finds matches inside comments and strings and
misses real calls that span several lines.

A call whose argument is not a literal string cannot be extracted and cannot be
translated, so it is reported with its file and line and the script exits non-zero. So is
a contextual call whose context is empty: a message with no context is written with
``_()``, and ``pgettext("", ...)`` would file it under a third thing that is neither.

A context is part of what identifies a message, so ``pgettext("GPU", "none detected")``
and ``pgettext("backends", "none detected")`` are two entries, written into the template
with a ``msgctxt`` line each, and a translator sees two strings to translate differently.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llamafit import __version__
from llamafit.i18n import DEFAULT_PLURAL_FORMS, TEMPLATE_NAME

ROOT = Path(__file__).resolve().parents[1]

SINGULAR_NAMES = frozenset({"_", "gettext", "lazy_gettext"})
"""Names that translate one message; the first argument is the ``msgid``."""

PLURAL_NAMES = frozenset({"ngettext", "lazy_ngettext"})
"""Names that translate a counting message; the first two arguments are the forms.

The ``lazy_`` pair defers the lookup to render time but holds the same literals, so a
message wrapped for a module-level constant is extracted like any other.
"""

CONTEXT_SINGULAR_NAMES = frozenset({"pgettext", "lazy_pgettext"})
"""Names that translate one message under a context; the context comes first."""

CONTEXT_PLURAL_NAMES = frozenset({"npgettext", "lazy_npgettext"})
"""Names that translate a counting message under a context; the context comes first."""

SOURCE_ROOTS: tuple[Path, ...] = (
    ROOT / "src" / "llamafit",
    # The message sweep has not landed yet, so the only calls in the tree are the ones
    # the translation tests make. Drop this root once the sweep wraps the real messages.
    ROOT / "tests" / "fixtures" / "messages.py",
)

EXCLUDED: tuple[Path, ...] = (ROOT / "src" / "llamafit" / "i18n",)
"""The translation machinery itself: it defines ``gettext``, it does not call it.

Its own messages stay in English on purpose. A notice saying LlamaFit does not speak a
language cannot be written in the language it does not speak.
"""

DEST = ROOT / "src" / "llamafit" / "data" / "locale" / TEMPLATE_NAME
"""The template file this script writes, in the repository rather than the install."""

# No creation date: the template is committed and a test compares it byte for byte, so a
# field that changes on every run would make the check fail for no reason.
_HEADER_FIELDS = (
    ("Project-Id-Version", "llamafit {version}"),
    ("Report-Msgid-Bugs-To", "https://github.com/pvaz/llamafit/issues"),
    ("MIME-Version", "1.0"),
    ("Content-Type", "text/plain; charset=UTF-8"),
    ("Content-Transfer-Encoding", "8bit"),
    ("Language", ""),
    ("Plural-Forms", DEFAULT_PLURAL_FORMS),
)
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


@dataclass
class Entry:
    """One message the extractor found.

    Attributes:
        msgid: The English message.
        plural: The English plural form, for a counting message.
        context: What the call said the message is used for, or ``None`` for a call that
            said nothing. A message with no context is not the same entry as the same
            message with one, and neither is the same as a context of ``""``.
        references: ``file:line`` for every call site, in the order they were found.
        first: The file and line the message was first seen at, which orders the file.
    """

    msgid: str
    plural: str | None = None
    context: str | None = None
    references: list[str] = field(default_factory=list)
    first: tuple[str, int] = ("", 0)


@dataclass
class Extraction:
    """Everything one run found.

    Attributes:
        entries: The messages, in the order they were first seen.
        problems: ``file:line: message`` for every call that cannot be extracted.
    """

    entries: list[Entry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def python_files(roots: Iterable[Path], excluded: Iterable[Path] = EXCLUDED) -> Iterator[Path]:
    """Yield every Python file under the roots, sorted, skipping caches and exclusions."""
    skip = tuple(excluded)
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in candidates:
            if "__pycache__" in path.parts:
                continue
            if any(path == folder or folder in path.parents for folder in skip):
                continue
            yield path


def extract(roots: Iterable[Path] = SOURCE_ROOTS) -> Extraction:
    """Read every source under the roots and collect the messages it asks to translate.

    Returns:
        The entries ordered by where each was first seen, and one problem line per call
        whose argument is not a literal string.
    """
    found = Extraction()
    by_key: dict[tuple[str | None, str], Entry] = {}
    for path in python_files(roots):
        _extract_file(path, found, by_key)
    found.entries.sort(key=lambda entry: entry.first)
    return found


def render_template(found: Extraction, *, version: str) -> str:
    """Render an extraction as the text of a ``.pot`` file.

    Returns:
        The template, with a header entry and one entry per message.
    """
    lines = ["# Translation template for LlamaFit.", "# Generated by scripts/gen_messages.py.", "#"]
    lines.append('msgid ""')
    lines.append('msgstr ""')
    for name, value in _HEADER_FIELDS:
        lines.append(f'"{name}: {value.format(version=version)}\\n"')
    for entry in found.entries:
        lines.append("")
        for reference in entry.references:
            lines.append(f"#: {reference}")
        if entry.context is not None:
            lines.append(f"msgctxt {_quote(entry.context)}")
        lines.append(f"msgid {_quote(entry.msgid)}")
        if entry.plural is None:
            lines.append('msgstr ""')
        else:
            lines.append(f"msgid_plural {_quote(entry.plural)}")
            lines.append('msgstr[0] ""')
            lines.append('msgstr[1] ""')
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the template, or report every call that cannot be extracted.

    Refuses to write anything when a call passes something other than a literal string:
    such a message can never reach a translator, so the run fails loudly and names each
    one instead of writing a template that quietly omits it.

    Returns:
        ``0`` on success, ``1`` when a call could not be extracted.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    roots = tuple(Path(a).resolve() for a in arguments) if arguments else SOURCE_ROOTS
    found = extract(roots)
    if found.problems:
        for problem in found.problems:
            print(problem, file=sys.stderr)
        print(
            f"refusing to write {TEMPLATE_NAME}: "
            f"{len(found.problems)} call(s) do not pass a literal string",
            file=sys.stderr,
        )
        return 1
    # Newlines are forced to "\n": the template is committed and compared byte for byte,
    # so it must not depend on the platform it was generated on.
    DEST.write_text(render_template(found, version=__version__), encoding="utf-8", newline="\n")
    print(f"wrote {DEST} with {len(found.entries)} message(s)")
    return 0


def _extract_file(
    path: Path, found: Extraction, by_key: dict[tuple[str | None, str], Entry]
) -> None:
    """Parse one file and add everything it asks to translate."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        found.problems.append(f"{_where(path, exc.lineno or 1)}: cannot parse: {exc.msg}")
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name in SINGULAR_NAMES:
            _collect(path, node, found, by_key, arity=1, contextual=False)
        elif name in PLURAL_NAMES:
            _collect(path, node, found, by_key, arity=2, contextual=False)
        elif name in CONTEXT_SINGULAR_NAMES:
            _collect(path, node, found, by_key, arity=1, contextual=True)
        elif name in CONTEXT_PLURAL_NAMES:
            _collect(path, node, found, by_key, arity=2, contextual=True)


def _collect(
    path: Path,
    node: ast.Call,
    found: Extraction,
    by_key: dict[tuple[str | None, str], Entry],
    *,
    arity: int,
    contextual: bool,
) -> None:
    """Record one call, or report it when its literal arguments are not literals."""
    reference = _where(path, node.lineno)
    wanted = arity + 1 if contextual else arity
    if len(node.args) < wanted:
        found.problems.append(f"{reference}: the call is missing its message argument")
        return
    literals: list[str] = []
    for position in range(wanted):
        value = _literal(node.args[position])
        if value is None:
            found.problems.append(
                f"{reference}: argument {position + 1} is not a literal string, "
                "so it cannot be extracted or translated"
            )
            return
        literals.append(value)
    context = literals.pop(0) if contextual else None
    if context == "":
        found.problems.append(
            f"{reference}: the context is empty; a message with no context is written "
            "with _() or ngettext(), not with an empty one"
        )
        return
    msgid = literals[0]
    plural = literals[1] if arity > 1 else None
    entry = by_key.get((context, msgid))
    if entry is None:
        entry = Entry(
            msgid=msgid, plural=plural, context=context, first=(_relative(path), node.lineno)
        )
        by_key[(context, msgid)] = entry
        found.entries.append(entry)
    elif entry.plural != plural and plural is not None:
        found.problems.append(f"{reference}: {msgid!r} already has a different plural form")
        return
    entry.references.append(reference)


def _called_name(func: ast.expr) -> str | None:
    """The bare name a call uses, so ``i18n._(...)`` counts the same as ``_(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.expr) -> str | None:
    """The text of a string literal, or ``None`` for anything else.

    Adjacent string literals are one constant by the time the parser is done, so a
    message split over several lines extracts as the whole sentence.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _where(path: Path, line: int) -> str:
    """A ``file:line`` reference, relative to the repository and with forward slashes."""
    return f"{_relative(path)}:{line}"


def _relative(path: Path) -> str:
    """The path as written in the template: repository-relative, forward slashes."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _quote(text: str) -> str:
    """Render a string as ``.po`` literals, broken after each newline as msgfmt does.

    Returns:
        One quoted string, or an empty one followed by a quoted string per line, which
        is how a multi-line message is written so a translator sees its shape.
    """
    lines = text.split("\n")
    parts = [line + "\n" for line in lines[:-1]]
    if lines[-1] or not parts:
        parts.append(lines[-1])
    quoted = ['"' + "".join(_ESCAPES.get(c, c) for c in part) + '"' for part in parts]
    if len(quoted) == 1:
        return quoted[0]
    return '""\n' + "\n".join(quoted)


if __name__ == "__main__":
    raise SystemExit(main())
