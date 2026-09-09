"""Print one version's section of ``CHANGELOG.md``, for a GitHub release's notes.

Usage:
    python scripts/changelog_section.py 0.1.0
    python scripts/changelog_section.py 0.1.0 --changelog path/to/CHANGELOG.md

The release workflow runs this *before* it uploads anything, so a tag whose version has
no changelog section stops the release rather than publishing a release with empty notes:
a version number on PyPI can never be reused, so the cheap failure has to come first.

The changelog follows Keep a Changelog, so a section starts at ``## [0.1.0] - 2026-09-09``
and ends at the next ``##`` heading. The link-reference block at the foot of the file
(``[0.1.0]: https://...``) belongs to no section and is dropped from the last one.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANGELOG = ROOT / "CHANGELOG.md"

_HEADING = re.compile(r"^##\s+(?!#)")
_LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:\s")


class SectionNotFoundError(LookupError):
    """The changelog has no usable section for the version asked for."""


def heading_version(line: str) -> str | None:
    """Return the version a level-two heading introduces, or ``None``.

    Args:
        line: One line of the changelog.

    Returns:
        ``"0.1.0"`` for ``## [0.1.0] - 2026-09-09`` and ``## 0.1.0``, ``"Unreleased"``
        for ``## [Unreleased]``, and ``None`` for anything that is not a level-two
        heading.
    """
    if not _HEADING.match(line):
        return None
    rest = _HEADING.sub("", line).strip()
    if not rest:
        return None
    return rest.split(" - ", 1)[0].split()[0].strip("[]") or None


def extract_section(text: str, version: str) -> str:
    """Return the body of one version's changelog section.

    Args:
        text: The whole changelog.
        version: The version to look for, without a leading ``v``.

    Returns:
        The section's body, with the surrounding blank lines and any trailing
        link-reference definitions removed, ending in a newline.

    Raises:
        SectionNotFoundError: No heading names that version, or the section is empty.
    """
    lines = text.splitlines()
    start = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if (found := heading_version(line)) is not None and found.lower() == version.lower()
        ),
        None,
    )
    if start is None:
        raise SectionNotFoundError(
            f"CHANGELOG.md has no '## [{version}]' section. Move the Unreleased entries "
            f"under a {version} heading before tagging."
        )
    end = next(
        (index for index in range(start, len(lines)) if heading_version(lines[index]) is not None),
        len(lines),
    )
    body = lines[start:end]
    while body and (not body[-1].strip() or _LINK_DEFINITION.match(body[-1])):
        body.pop()
    while body and not body[0].strip():
        body.pop(0)
    if not body:
        raise SectionNotFoundError(
            f"the '## [{version}]' section of CHANGELOG.md is empty; a release needs notes"
        )
    return "\n".join(body) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the section named on the command line, or explain why there is none."""
    parser = argparse.ArgumentParser(description="Print one version's changelog section.")
    parser.add_argument("version", help="the version to print, without a leading 'v'")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help="the changelog to read (default: the one in the repository root)",
    )
    args = parser.parse_args(argv)
    try:
        section = extract_section(args.changelog.read_text(encoding="utf-8"), args.version)
    except SectionNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(section, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
