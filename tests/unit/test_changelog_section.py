"""Unit tests for the changelog section reader the release workflow uses for its notes."""

from pathlib import Path

import pytest
from scripts.changelog_section import SectionNotFoundError, extract_section, heading_version

ROOT = Path(__file__).resolve().parents[2]

CHANGELOG = """# Changelog

Preamble that belongs to no version.

## [Unreleased]

### Added
- Something not released yet.

## [0.2.0] - 2026-10-01

### Added
- Second release.

## [0.1.0] - 2026-09-09

### Added
- First release.

### Fixed
- A bug.

[Unreleased]: https://example.com/compare/v0.2.0...HEAD
[0.2.0]: https://example.com/releases/tag/v0.2.0
[0.1.0]: https://example.com/releases/tag/v0.1.0
"""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("## [0.1.0] - 2026-09-09", "0.1.0"),
        ("## 0.1.0 - 2026-09-09", "0.1.0"),
        ("## [Unreleased]", "Unreleased"),
        ("# Changelog", None),
        ("### Added", None),
        ("- ## not a heading", None),
    ],
)
def test_heading_version_reads_the_keep_a_changelog_forms(line: str, expected: str | None) -> None:
    assert heading_version(line) == expected


def test_section_stops_at_the_next_version() -> None:
    assert extract_section(CHANGELOG, "0.2.0") == "### Added\n- Second release.\n"


def test_last_section_drops_the_link_reference_block() -> None:
    assert extract_section(CHANGELOG, "0.1.0") == (
        "### Added\n- First release.\n\n### Fixed\n- A bug.\n"
    )


def test_unreleased_is_readable_too() -> None:
    assert extract_section(CHANGELOG, "unreleased") == "### Added\n- Something not released yet.\n"


def test_a_missing_version_says_what_to_do() -> None:
    with pytest.raises(SectionNotFoundError, match=r"no '## \[9\.9\.9\]' section"):
        extract_section(CHANGELOG, "9.9.9")


def test_an_empty_section_is_refused() -> None:
    with pytest.raises(SectionNotFoundError, match="is empty"):
        extract_section("# Changelog\n\n## [0.1.0] - 2026-09-09\n\n## [0.0.1]\n\n- old\n", "0.1.0")


def test_the_committed_changelog_still_parses() -> None:
    """The real file, in its real format: this is what fails if the changelog drifts."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = extract_section(text, "Unreleased")
    assert section.startswith("### ")
    assert not section.rstrip().endswith("]: https://github.com/pvaz/llamafit/commits/main")
