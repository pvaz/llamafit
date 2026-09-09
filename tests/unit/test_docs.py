"""Documentation must name every command and flag the CLI actually has."""

import re
from pathlib import Path

from typer.main import get_command

from llamafit.cli.app import app

ROOT = Path(__file__).resolve().parents[2]


def test_cli_doc_mentions_every_command_and_option() -> None:
    text = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    command = get_command(app)
    for name in command.commands:  # type: ignore[attr-defined]
        assert f"`llamafit {name}`" in text, f"docs/cli.md lacks {name}"
    for option in ("--json", "--verbose", "--no-color", "--version", "--no-measure"):
        assert option in text, f"docs/cli.md lacks {option}"


def test_platform_doc_mentions_every_probe_hint() -> None:
    from llamafit.services.doctor import PROBE_HINTS

    text = (ROOT / "docs" / "platform-support.md").read_text(encoding="utf-8")
    for probe in PROBE_HINTS:
        assert probe in text, f"docs/platform-support.md lacks probe {probe}"


def test_development_doc_only_names_workflows_that_exist() -> None:
    text = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    named = set(re.findall(r"\.github/workflows/([A-Za-z0-9_.-]+\.ya?ml)", text))
    assert named, "docs/development.md should name the workflow that runs the checks"
    for workflow in sorted(named):
        assert (ROOT / ".github" / "workflows" / workflow).is_file(), (
            f"docs/development.md names .github/workflows/{workflow}, which does not exist"
        )


def test_readme_shows_the_two_commands_that_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "llamafit system" in text and "llamafit doctor" in text
