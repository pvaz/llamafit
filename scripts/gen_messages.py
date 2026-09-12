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

A comment block immediately above a call, whose first line opens with ``Translators:``, is
copied into the template as ``#.`` lines. It is how a call site says something a translator
cannot work out from the message alone -- that a message is punctuation rather than prose,
say. Only a block carrying that marker is copied, so a comment written for whoever
maintains the code stays in the code.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llamafit.i18n import DEFAULT_PLURAL_FORMS, TEMPLATE_NAME

ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    """The version in ``pyproject.toml``, which is the one this template belongs to.

    Not ``llamafit.__version__``. That reads the installed distribution's metadata, which
    is right at run time -- a wheel's version is its metadata -- and wrong here. In an
    editable install the metadata is written once and does not move when the version in
    ``pyproject.toml`` does, so the moment after a release bump this script would stamp
    the template with the version that was current before it, `--check` would agree with
    itself, and the first clean checkout would disagree with both. That is not a
    hypothetical: it failed all six CI jobs on the 0.1.1 bump while passing locally.

    The file is the source of truth the release already checks the tag against, so it is
    the one to read. It is read with a regular expression and not a TOML parser for the
    reason `tests/unit/test_docs_truth.py` gives for doing the same: Python 3.10 has no
    `tomllib`, this script runs on 3.10 in CI, and this needs one line out of one file.

    Raises:
        SystemExit: `pyproject.toml` has no version line, which means the file this was
            pointed at is not the one it was written for.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    if match is None:
        raise SystemExit("pyproject.toml has no version line")
    return match.group(1)


SINGULAR_NAMES = frozenset({"_", "gettext", "lazy_gettext"})
"""Names that translate one message; the first argument is the ``msgid``."""

PLURAL_NAMES = frozenset({"ngettext", "lazy_ngettext"})
"""Names that translate a counting message; the first two arguments are the forms.

The ``lazy_`` pair defers the lookup to render time but holds the same literals, so a
message wrapped for a module-level constant is extracted like any other.
"""

CONTEXT_SINGULAR_NAMES = frozenset({"pgettext", "lazy_pgettext", "pgettext_literal"})
"""Names that translate one message under a context; the context comes first.

``pgettext_literal`` reads its entry with whitespace counting as a value, for an entry
holding punctuation rather than prose. That is a difference at lookup time and none at
all here: it takes the same two literals and files the same entry.
"""

CONTEXT_PLURAL_NAMES = frozenset({"npgettext", "lazy_npgettext"})
"""Names that translate a counting message under a context; the context comes first."""

SOURCE_ROOTS: tuple[Path, ...] = (ROOT / "src" / "llamafit",)
"""What the template is built from: the shipped package, and nothing else.

The test fixtures wrap the same messages a second time, to exercise the machinery
without freezing a real call site into a test, but they are not what LlamaFit ships and
a message that only a fixture asks for has no business reaching a translator."""

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

TRANSLATOR_MARKER = "Translators:"
"""What a comment block must open with before it is copied into the template.

The marker is the convention every gettext toolchain already uses, and it is what keeps
notes written for whoever maintains the code out of a file whose reader cannot act on them.
"""


@dataclass
class Entry:
    """One message the extractor found.

    Attributes:
        msgid: The English message.
        plural: The English plural form, for a counting message.
        context: What the call said the message is used for, or ``None`` for a call that
            said nothing. A message with no context is not the same entry as the same
            message with one, and neither is the same as a context of ``""``.
        comments: Notes for the translator, from the ``Translators:`` blocks above the call
            sites, in the order they were found and without repeats: a message wrapped in
            two places under the same note is not told it twice.
        references: ``file:line`` for every call site, in the order they were found.
        first: The file and line the message was first seen at, which orders the file.
    """

    msgid: str
    plural: str | None = None
    context: str | None = None
    comments: list[str] = field(default_factory=list)
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
        for comment in entry.comments:
            lines.append(f"#. {comment}" if comment else "#.")
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

    Args:
        argv: Arguments, for a test. ``None`` reads the real ones.

    Returns:
        ``0`` on success, ``1`` when a call could not be extracted, and ``1`` under
        ``--check`` when the committed template is not what this run would write.

    Every argument used to be taken as a source root, so a stray word produced a template
    with almost nothing in it and the script said it had succeeded. Roots are now named by
    ``--root``, which nothing passes in practice, and a bare argument is an error. The
    other two generators are checked in CI and this one was not, which is how a stale
    template reached the main branch twice in one week; ``--check`` is what CI runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the committed template is not what this run produces",
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        metavar="DIR",
        help="a directory to extract from, repeatable (default: the package and the scripts)",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    roots = tuple(root.resolve() for root in args.root) if args.root else SOURCE_ROOTS
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
    rendered = render_template(found, version=project_version())
    if args.check:
        try:
            current = DEST.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current == rendered:
            print(f"{TEMPLATE_NAME} is current with {len(found.entries)} message(s)")
            return 0
        print(
            f"{TEMPLATE_NAME} is not what the source says it should be; "
            f"run `python scripts/gen_messages.py` and commit the result",
            file=sys.stderr,
        )
        return 1
    DEST.write_text(rendered, encoding="utf-8", newline="\n")
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
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name in SINGULAR_NAMES:
            _collect(path, node, found, by_key, lines, arity=1, contextual=False)
        elif name in PLURAL_NAMES:
            _collect(path, node, found, by_key, lines, arity=2, contextual=False)
        elif name in CONTEXT_SINGULAR_NAMES:
            _collect(path, node, found, by_key, lines, arity=1, contextual=True)
        elif name in CONTEXT_PLURAL_NAMES:
            _collect(path, node, found, by_key, lines, arity=2, contextual=True)


def _collect(
    path: Path,
    node: ast.Call,
    found: Extraction,
    by_key: dict[tuple[str | None, str], Entry],
    lines: Sequence[str],
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
    for comment in translator_comment(lines, node.lineno):
        if comment not in entry.comments:
            entry.comments.append(comment)
    entry.references.append(reference)


def translator_comment(lines: Sequence[str], lineno: int) -> list[str]:
    """The ``Translators:`` note written immediately above line ``lineno``, if there is one.

    The block has to touch the call: a blank line, or any code, ends it. Only a block whose
    first line opens with :data:`TRANSLATOR_MARKER` comes back, so a comment explaining the
    code to whoever maintains it never reaches a translator who could not act on it.

    Args:
        lines: The file's lines, without their endings.
        lineno: The call's line number, counting from one.

    Returns:
        The note's lines, marker included, each with its ``#`` and one following space
        stripped; empty when the lines above the call are not a marked block.
    """
    block: list[str] = []
    index = lineno - 2  # the line above the call, counting from zero
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped.startswith("#"):
            break
        block.append(stripped[1:].removeprefix(" "))
        index -= 1
    block.reverse()
    if not block or not block[0].startswith(TRANSLATOR_MARKER):
        return []
    return block


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
