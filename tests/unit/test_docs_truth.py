"""The documentation must not describe an older program than the one in this tree.

`test_docs.py` next door checks that the documentation *mentions* every command and field.
This file checks the other half, which is the half that went wrong: that it does not say a
shipped command is being built, does not print sample output the command stopped producing,
and does not carry anybody's home directory into a public file.

Every assertion here reads the software — the registered commands, the version in
`pyproject.toml`, the catalog that ships — rather than a list typed out beside it, so a
command added tomorrow is documented or fails, and a version bumped tomorrow has a changelog
section or fails.
"""

import re
from pathlib import Path

from typer.main import get_command

from llamafit.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [
    ROOT / "README.md",
    ROOT / "ROADMAP.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]

# The four sentences the 0.1.0 truth pass retired, each of which was false when it was
# written down or became false and nobody noticed. They are listed by their exact wording
# because that is what a revert would put back.
RETIRED_CLAIMS = [
    "are being built now",
    "no interface message goes through it yet",
    "the bundled catalog carries no refreshed facts",
    "there is nothing to download with and nothing to benchmark with yet",
]

# Sample output is generated on somebody's machine, and an absolute path is where their name
# rides along into a public file. The rule is an allow-list rather than a search for account
# names, because the leak this was written for was `C:\Dev\<a personal folder>\...`, which no
# pattern for "home directory" would have caught. A drive path in the documentation starts
# with one of these; a POSIX home path is never allowed at all. Add a root here when the
# documentation honestly needs one, and look twice at it while you do.
DOCUMENTABLE_ROOTS = frozenset({"llama.cpp", "models", "projects"})
# The lookbehind keeps `https://` out of it: the drive letter of a path is not preceded by a
# letter, and the `s` of `https` is.
DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]+([^\s\\/`\"'<>|,;]*)")
POSIX_HOME = re.compile(r"(?<![\w-])/(?:home|Users)/[A-Za-z0-9._-]+/")


def command_names() -> list[str]:
    """Every subcommand `llamafit` registers, as a person types it."""
    command = get_command(app)
    return sorted(command.commands)  # type: ignore[attr-defined]


def pyproject_version() -> str:
    """The version `pyproject.toml` declares, read without a TOML parser.

    Python 3.10 has no `tomllib`, and this needs one line out of one file.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_the_changelog_has_a_section_for_the_version_being_shipped() -> None:
    """The release stops itself when this is not true, after the whole CI matrix has run.

    `release.yml` calls `scripts/changelog_section.py` with the tag's version before it
    uploads anything, so a version with no changelog section fails the release rather than
    publishing one with empty notes. It fails there in about twenty minutes; it fails here
    in about twenty milliseconds.
    """
    from scripts.changelog_section import extract_section

    version = pyproject_version()
    section = extract_section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    assert section.strip(), f"the '## [{version}]' section of CHANGELOG.md is empty"


def test_no_document_names_a_path_off_somebody_machine() -> None:
    """Every absolute path in the documentation starts somewhere a reader could too."""
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        where = path.relative_to(ROOT).as_posix()
        home = POSIX_HOME.search(text)
        assert home is None, f"{where} names a home directory: {home.group(0)!r}"
        for root in DRIVE_PATH.findall(text):
            assert root in DOCUMENTABLE_ROOTS, (
                f"{where} names a drive path rooted at {root!r}; the documentation's roots are"
                f" {sorted(DOCUMENTABLE_ROOTS)}"
            )


def test_no_document_names_the_checkout_it_was_written_in() -> None:
    """The other half of the same rule, for a path no pattern could guess."""
    for anchor in {str(ROOT), ROOT.as_posix()}:
        for path in DOC_FILES:
            assert anchor not in path.read_text(encoding="utf-8"), (
                f"{path.relative_to(ROOT).as_posix()} names this checkout's own path"
            )


def test_the_cli_reference_marks_every_registered_command_shipped() -> None:
    """A command a person can run is not a command whose documentation calls it unbuilt."""
    lines = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8").splitlines()
    headings = [line for line in lines if line.startswith("#")]
    for name in command_names():
        owned = [line for line in headings if f"`llamafit {name}`" in line]
        assert owned, f"docs/cli.md has no heading for `llamafit {name}`"
        for heading in owned:
            assert "shipped" in heading, f"docs/cli.md calls a shipped command unbuilt: {heading}"


def test_the_readme_command_table_marks_every_registered_command_done() -> None:
    """The README's table has a phase column; every row of it names a command that answers."""
    rows = [
        line
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("| `llamafit")
    ]
    assert rows, "README.md has no command table"
    for name in command_names():
        owns = re.compile(rf"`llamafit {re.escape(name)}[`\s]|`{re.escape(name)}`")
        owning = [row for row in rows if owns.search(row.split("|")[1])]
        assert owning, f"README.md's command table does not name {name}"
        for row in owning:
            phase = row.rstrip("| ").rsplit("|", 1)[-1]
            assert "done" in phase, f"README.md leaves a shipped command unmarked: {row}"


def test_the_roadmap_does_not_call_a_shipped_command_unbuilt() -> None:
    """Whatever is still being built, it is not a command that answers today.

    The check is on the region a "being built" marker governs: a table row is its own
    region, a heading's runs to the next heading. A phase that genuinely has not started
    names no command, so this stays quiet until one ships under a heading that still says
    it is coming.
    """
    lines = (ROOT / "ROADMAP.md").read_text(encoding="utf-8").splitlines()
    names = command_names()
    for index, line in enumerate(lines):
        if "being built" not in line.lower():
            continue
        if line.startswith("|"):
            region = [line]
        else:
            end = next(
                (later for later in range(index + 1, len(lines)) if lines[later].startswith("## ")),
                len(lines),
            )
            region = lines[index:end]
        text = "\n".join(region)
        for name in names:
            assert f"`{name}`" not in text and f"`llamafit {name}" not in text, (
                f"ROADMAP.md line {index + 1} calls `{name}` unbuilt, and it answers"
            )


def fenced_blocks(text: str) -> list[str]:
    """Every fenced block in a Markdown file: the parts that claim to be real output."""
    blocks, current = [], None
    for line in text.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def test_no_sample_shows_a_quant_the_shipped_catalog_has_facts_for() -> None:
    """The wheel carries a `.facts.json` beside every catalog file; `info` prints real figures.

    `not read yet` is what the renderer prints for a quant nobody has refreshed. Explaining
    that string in prose is fine and this page does it; printing it in a sample, while the
    shipped catalog has been refreshed, is showing output no reader can reproduce.
    """
    from llamafit.catalog import load_catalog

    catalog, problems = load_catalog()
    assert problems == [], problems
    refreshed = any(
        quant.bytes_
        for model in catalog.models
        for source in model.sources
        for quant in source.quants
    )
    assert refreshed, "the bundled catalog ships no refreshed facts; this test's premise is gone"
    for path in DOC_FILES:
        for block in fenced_blocks(path.read_text(encoding="utf-8")):
            assert "not read yet" not in block, (
                f"{path.relative_to(ROOT).as_posix()} shows a quant with no facts, and the"
                " catalog ships with them"
            )


def test_no_document_repeats_a_claim_this_pass_retired() -> None:
    """Four sentences that were false. A revert puts them back; this is what catches it."""
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for claim in RETIRED_CLAIMS:
            assert claim not in text, f"{path.relative_to(ROOT).as_posix()} says {claim!r} again"
