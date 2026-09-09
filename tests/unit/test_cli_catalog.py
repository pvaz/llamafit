"""Tests for ``list``, ``search``, ``info`` and the ``catalog`` command group.

Every test monkeypatches ``llamafit.cli.catalog_cmd.load_catalog`` (or
``validate_files`` / ``refresh_file``) directly, by name, rather than the module,
matching how those names are imported in ``catalog_cmd.py``. No test touches the
bundled catalog or the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llamafit.catalog.loader import Problem
from llamafit.catalog.refresh import RefreshResult
from llamafit.cli.app import app
from llamafit.errors import CatalogError
from llamafit.models.catalog import Architecture, Catalog, License, ModelSource, Quant
from llamafit.models.gguf import GgufFacts
from llamafit.services.catalog import ModelSummary
from tests.unit.test_models_catalog import minimal

runner = CliRunner()


def _catalog() -> Catalog:
    coder_with_tools = minimal(
        id="coder-with-tools",
        name="Coder With Tools",
        vendor="Acme Robotics",
        capabilities=["coding", "tools"],
        use_cases=["coding"],
        quality={
            "baseline": 90,
            "benchmarks": [
                {"name": "Made-Up-Bench", "score": 42.0, "source": "https://example.invalid/bench"}
            ],
        },
        context={"native": 32768, "extended": 131072, "extended_method": "yarn"},
        architecture=Architecture(
            **{"class": "moe-hybrid", "gguf_arch": "qwen3next", "notes": "48 layers, 512 experts"}
        ),
        sources=[
            ModelSource(
                repo="acme/coder-gguf",
                quants=[
                    Quant(
                        name="Q4_K_M",
                        files=["coder-Q4_K_M.gguf"],
                        bytes=1_000_000_000,
                        bpw=4.5,
                        gguf_facts=GgufFacts(
                            arch="qwen3next", n_layer=48, n_expert=512, n_expert_used=10
                        ),
                    )
                ],
            )
        ],
    )
    coder_only = minimal(
        id="coder-only",
        name="Coder Only",
        vendor="Acme Robotics",
        capabilities=["coding"],
        use_cases=["coding"],
        quality={"baseline": 70},
    )
    chat_model = minimal(
        id="chat-model",
        name="Chat Model",
        vendor="Other Corp",
        capabilities=["multilingual"],
        use_cases=["chat"],
        quality={"baseline": 50},
        license=License(spdx="MIT", url="https://example.invalid/mit"),
    )
    return Catalog(models=[coder_with_tools, coder_only, chat_model])


@pytest.fixture(autouse=True)
def patch_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test sees the same small, fixed catalog and file list, never the bundled ones."""
    monkeypatch.setattr("llamafit.cli.catalog_cmd.load_catalog", lambda: (_catalog(), []))
    monkeypatch.setattr(
        "llamafit.cli.catalog_cmd._default_catalog_paths", lambda: [Path("fake-catalog.yaml")]
    )


# --- list / search -----------------------------------------------------------


def test_list_table_contains_a_models_id_and_quality() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert "coder-with-tools" in result.output
    assert "90" in result.output  # coder-with-tools' quality baseline
    assert "Vendor" not in result.output  # dropped to keep rows to one line at 80 cols


def test_list_json_carries_every_id() -> None:
    result = runner.invoke(app, ["--json", "list"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert {m["id"] for m in data} == {"coder-with-tools", "coder-only", "chat-model"}


def test_a_capability_filter_narrows_the_rows() -> None:
    result = runner.invoke(app, ["list", "--capability", "tools"])
    assert result.exit_code == 0, result.output
    assert "coder-with-tools" in result.output
    assert "coder-only" not in result.output
    assert "chat-model" not in result.output


def test_two_capabilities_require_both() -> None:
    args = ["--json", "list", "--capability", "coding", "--capability", "tools"]
    result = runner.invoke(app, args)
    data = json.loads(result.output)
    assert [m["id"] for m in data] == ["coder-with-tools"]


def test_use_case_filter() -> None:
    result = runner.invoke(app, ["--json", "list", "--use-case", "chat"])
    data = json.loads(result.output)
    assert [m["id"] for m in data] == ["chat-model"]


def test_vendor_filter() -> None:
    result = runner.invoke(app, ["--json", "list", "--vendor", "other corp"])
    data = json.loads(result.output)
    assert [m["id"] for m in data] == ["chat-model"]


def test_license_filter() -> None:
    result = runner.invoke(app, ["--json", "list", "--license", "MIT"])
    data = json.loads(result.output)
    assert [m["id"] for m in data] == ["chat-model"]


def test_limit_caps_the_result_count() -> None:
    result = runner.invoke(app, ["--json", "list", "--limit", "1"])
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["id"] == "coder-with-tools"  # highest quality_baseline sorts first


def test_an_invalid_capability_is_a_clean_error_not_a_traceback() -> None:
    result = runner.invoke(app, ["list", "--capability", "nonsense"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "nonsense" in result.exception.render()


def test_an_invalid_use_case_fails_the_same_way_as_an_invalid_capability() -> None:
    """--use-case and --capability are the same kind of mistake and must fail alike."""
    result = runner.invoke(app, ["list", "--use-case", "nonsense"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    rendered = result.exception.render()
    assert "nonsense" in rendered
    assert "coding" in rendered  # a valid use case, named in the hint


def test_search_is_list_search() -> None:
    result = runner.invoke(app, ["search", "chat"])
    assert result.exit_code == 0, result.output
    assert "chat-model" in result.output
    assert "coder-only" not in result.output


def test_no_matches_says_so_instead_of_printing_an_empty_table() -> None:
    result = runner.invoke(app, ["list", "--vendor", "Nobody Makes This"])
    assert result.exit_code == 0, result.output
    assert "No models matched" in result.output
    # a hint at what to try, not just a bare "nothing found"
    assert "--vendor" in result.output


def test_no_matches_with_json_is_still_an_empty_array() -> None:
    result = runner.invoke(app, ["--json", "list", "--vendor", "Nobody Makes This"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_the_list_table_labels_the_quality_score_with_its_caveat() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Quality" in output
    assert "editorial baseline" in output
    assert "quantisation penalty" in output
    assert "info" in output  # points at where the sourced benchmarks live


@pytest.mark.parametrize(
    "args",
    [["list", "--help"], ["search", "--help"], ["catalog", "show", "--help"]],
)
def test_help_screens_have_no_backticks(args: list[str]) -> None:
    """Typer prints a command's docstring verbatim in --help; it must be plain text."""
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "`" not in result.output


# --- info ----------------------------------------------------------------------


def test_info_shows_vendor_licence_and_quant_details() -> None:
    result = runner.invoke(app, ["info", "coder-with-tools"])
    assert result.exit_code == 0, result.output
    assert "Acme Robotics" in result.output
    assert "Apache-2.0" in result.output
    assert "Q4_K_M" in result.output
    assert "4.50" in result.output  # bits per weight


def test_info_matches_an_id_regardless_of_case() -> None:
    result = runner.invoke(app, ["info", "CODER-WITH-TOOLS"])
    assert result.exit_code == 0, result.output
    assert "Acme Robotics" in result.output


def test_info_matches_an_id_with_surrounding_whitespace() -> None:
    result = runner.invoke(app, ["info", " coder-with-tools \t"])
    assert result.exit_code == 0, result.output
    assert "Acme Robotics" in result.output


def test_info_matches_an_id_with_mixed_case_and_whitespace() -> None:
    result = runner.invoke(app, ["info", "  Coder-With-Tools  "])
    assert result.exit_code == 0, result.output
    assert "Acme Robotics" in result.output


def test_info_shows_extended_context_notes_benchmarks_and_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A wide console so long cells (the facts summary, the notes) render on one line
    # instead of wrapping across Rich's table borders, which would split the very
    # text this test is checking for.
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["info", "coder-with-tools"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "extended via yarn" in output
    assert "48 layers, 512 experts" in output  # architecture notes
    assert "Made-Up-Bench" in output
    assert "48 layers" in output  # from the quant's gguf facts
    assert "512/10 experts" in output


def test_info_json_is_a_model_detail_with_aliased_quant_bytes() -> None:
    result = runner.invoke(app, ["--json", "info", "coder-with-tools"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model"]["id"] == "coder-with-tools"
    assert data["quants"][0]["bytes"] == 1_000_000_000
    assert "bytes_" not in data["quants"][0]


def test_info_on_an_unknown_id_exits_1_and_names_the_close_id() -> None:
    result = runner.invoke(app, ["info", "coder-with-tool"])  # missing trailing "s"
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "coder-with-tools" in result.exception.render()


def test_info_on_nonsense_suggests_nothing() -> None:
    result = runner.invoke(app, ["info", "zzz-not-even-close"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    rendered = result.exception.render()
    assert "coder" not in rendered and "chat" not in rendered


def test_info_ambiguous_prefix_lists_every_tied_match() -> None:
    result = runner.invoke(app, ["info", "coder-"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    rendered = result.exception.render()
    assert "coder-only" in rendered
    assert "coder-with-tools" in rendered


def test_a_bracket_in_catalog_text_does_not_crash_list_or_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A lone closing tag actually matches Rich's markup grammar and raises
    # ``MarkupError`` if this text is ever handed to Rich as a plain string
    # instead of wrapped in ``Text`` (verified directly:
    # ``Text.from_markup("Model [/bold] v2")`` raises). "Hostile [Name]" (capital
    # N) does not match that grammar and passes through even with the bug
    # present, so it proves nothing; this uses a form that actually bites.
    hostile = minimal(
        id="hostile-model",
        name="Model [/bold] v2",
        vendor="Vendor [/red]",
        license=License(spdx="weird-[/bold]licence", url="https://example.invalid/l"),
    )
    monkeypatch.setattr(
        "llamafit.cli.catalog_cmd.load_catalog", lambda: (Catalog(models=[hostile]), [])
    )

    # The list table no longer shows vendor or name (dropped to keep rows to one
    # line), so it carries no free-text field a bracket could break; this just
    # proves a hostile catalog entry still renders without crashing it.
    list_result = runner.invoke(app, ["list"])
    assert list_result.exit_code == 0, list_result.output
    assert "hostile-model" in list_result.output

    info_result = runner.invoke(app, ["info", "hostile-model"])
    assert info_result.exit_code == 0, info_result.output
    assert "Model [/bold] v2" in info_result.output  # the table's title
    assert "Vendor [/red]" in info_result.output
    assert "weird-[/bold]licence" in info_result.output


# --- catalog problems: reported, never fatal, never dumped into a table -------


def test_a_catalog_problem_is_reported_but_browsing_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = Problem(
        file="custom_models.yaml",
        model_id="typo-model",
        location="quality.baseline",
        message="field required",
    )
    monkeypatch.setattr("llamafit.cli.catalog_cmd.load_catalog", lambda: (_catalog(), [problem]))

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0, result.output
    assert "coder-with-tools" in result.output  # the entries that did load still show
    assert "1 catalog problem found" in result.output
    assert "llamafit catalog validate" in result.output
    # the problem's own detail is not dumped into the table; `validate` is where it lives
    assert "field required" not in result.output
    assert "typo-model" not in result.output


def test_multiple_catalog_problems_are_reported_as_a_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = [
        Problem(file="a.yaml", model_id="x", location="id", message="bad"),
        Problem(file="b.yaml", model_id="y", location="id", message="also bad"),
    ]
    monkeypatch.setattr("llamafit.cli.catalog_cmd.load_catalog", lambda: (_catalog(), problems))

    result = runner.invoke(app, ["info", "coder-with-tools"])

    assert result.exit_code == 0, result.output
    assert "2 catalog problems found" in result.output


# --- catalog validate ------------------------------------------------------


def test_catalog_validate_exits_0_on_a_clean_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.catalog_cmd.validate_files", lambda paths: [])
    result = runner.invoke(app, ["catalog", "validate"])
    assert result.exit_code == 0, result.output
    assert "no problems" in result.output


def test_catalog_validate_exits_1_and_prints_the_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = Problem(file="bad.yaml", model_id="x", location="id", message="duplicate id 'x'")
    monkeypatch.setattr("llamafit.cli.catalog_cmd.validate_files", lambda paths: [problem])
    result = runner.invoke(app, ["catalog", "validate"])
    assert result.exit_code == 1
    assert "bad.yaml" in result.output
    assert "duplicate id 'x'" in result.output


def test_catalog_validate_json_is_the_problem_list(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = Problem(file="bad.yaml", model_id=None, location="file", message="invalid YAML")
    monkeypatch.setattr("llamafit.cli.catalog_cmd.validate_files", lambda paths: [problem])
    result = runner.invoke(app, ["--json", "catalog", "validate"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data == [
        {"file": "bad.yaml", "model_id": None, "location": "file", "message": "invalid YAML"}
    ]


# --- catalog refresh --------------------------------------------------------


def test_catalog_refresh_check_exits_1_when_something_would_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = RefreshResult(
        model_id="coder-with-tools",
        file="x.yaml",
        changed=True,
        fields=["sources[0].quants[0].bytes"],
    )
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [changed])
    result = runner.invoke(app, ["catalog", "refresh", "--check"])
    assert result.exit_code == 1
    assert "coder-with-tools" in result.output


def test_catalog_refresh_prints_no_changes_when_nothing_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchanged = RefreshResult(model_id="coder-with-tools", file="x.yaml", changed=False, fields=[])
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [unchanged])
    result = runner.invoke(app, ["catalog", "refresh"])
    assert result.exit_code == 0, result.output
    assert "No changes." in result.output


def test_catalog_refresh_reports_an_error_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = RefreshResult(
        model_id="coder-with-tools", file="x.yaml", changed=False, fields=[], error="network down"
    )
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [failed])
    result = runner.invoke(app, ["catalog", "refresh"])
    assert result.exit_code == 1
    assert "network down" in result.output


def test_catalog_refresh_does_not_repeat_a_repo_name_and_adds_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = "acme/coder-gguf"
    failed = RefreshResult(
        model_id="coder-with-tools",
        file="x.yaml",
        changed=False,
        fields=[],
        error=f"{repo}: Hugging Face returned 404 while listing files for {repo}.",
    )
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [failed])
    result = runner.invoke(app, ["catalog", "refresh"])

    assert result.exit_code == 1
    assert result.output.count(repo) == 1  # the repo name appears once, not twice
    assert "Hint" in result.output
    assert "network" in result.output.lower()


def test_catalog_refresh_groups_an_identical_warning_across_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_warning = "facts.json could not be parsed; recorded facts were left as they were"
    affected = [
        SimpleNamespace(
            model_id=model_id,
            file="x.yaml",
            changed=False,
            fields=[],
            error=None,
            warnings=[shared_warning],
        )
        for model_id in ("coder-with-tools", "coder-only", "chat-model")
    ]
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: affected)

    result = runner.invoke(app, ["catalog", "refresh"])

    assert result.exit_code == 0, result.output
    assert result.output.count(shared_warning) == 1  # said once, not once per model
    assert "3 models" in result.output
    for model_id in ("coder-with-tools", "coder-only", "chat-model"):
        assert model_id in result.output  # each affected model is still named


def test_catalog_refresh_json_is_the_result_list(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = RefreshResult(model_id="coder-with-tools", file="x.yaml", changed=True, fields=["a"])
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [changed])
    result = runner.invoke(app, ["--json", "catalog", "refresh"])
    data = json.loads(result.output)
    assert data == [
        {
            "model_id": "coder-with-tools",
            "file": "x.yaml",
            "changed": True,
            "fields": ["a"],
            "error": None,
        }
    ]


def test_catalog_refresh_prints_a_warning_even_without_a_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duck-typed stand-in for the ``warnings`` field ``RefreshResult`` will soon carry.

    The field does not exist on ``RefreshResult`` yet, so this uses a plain namespace with
    the same attributes to prove the printing path already surfaces it, with nothing left
    to change here once the real field lands.
    """
    quiet_but_suspicious = SimpleNamespace(
        model_id="coder-with-tools",
        file="x.yaml",
        changed=False,
        fields=[],
        error=None,
        warnings=["quant Q4_K_M matched no files in acme/coder-gguf"],
    )
    monkeypatch.setattr(
        "llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [quiet_but_suspicious]
    )
    result = runner.invoke(app, ["catalog", "refresh"])
    assert result.exit_code == 0, result.output
    assert "quant Q4_K_M matched no files" in result.output
    assert "No changes." not in result.output


def test_catalog_refresh_with_an_unknown_model_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.catalog_cmd.refresh_file", lambda *a, **k: [])
    result = runner.invoke(app, ["catalog", "refresh", "--model", "does-not-exist"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)


# --- catalog show ------------------------------------------------------------


def test_catalog_show_json_is_the_raw_entry_with_aliases() -> None:
    result = runner.invoke(app, ["catalog", "show", "coder-with-tools"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["id"] == "coder-with-tools"
    assert data["architecture"]["class"] == "moe-hybrid"
    assert "class_" not in data["architecture"]
    assert data["sources"][0]["quants"][0]["bytes"] == 1_000_000_000


def test_catalog_show_yaml_is_yaml_not_json() -> None:
    result = runner.invoke(app, ["catalog", "show", "coder-with-tools", "--yaml"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().startswith("id: coder-with-tools")
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_catalog_show_unknown_id_exits_1() -> None:
    result = runner.invoke(app, ["catalog", "show", "coder-with-tool"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "coder-with-tools" in result.exception.render()


# --- _default_catalog_paths, unpatched -----------------------------------------


def test_default_catalog_paths_lists_the_real_bundled_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one test that calls the real, un-mocked helper, so its filesystem logic runs too."""
    monkeypatch.undo()  # this test wants the real _default_catalog_paths, not the fixture's fake
    from llamafit.cli.catalog_cmd import _default_catalog_paths

    paths = _default_catalog_paths()
    assert paths
    assert all(path.suffix == ".yaml" for path in paths)


# --- _fmt_capabilities: whole items only, marker always survives --------------


def test_a_narrow_width_drops_whole_capabilities_never_cuts_one_in_half() -> None:
    from llamafit.cli.render import _fmt_capabilities

    # "vision, multilingual, long-context" (the eligible top 3) is 34 characters and
    # does not fit in 23; "vision, multilingual" plus a marker for the two dropped
    # (long-context, and tools beyond the top-3 cap) is exactly 23.
    capabilities = ["vision", "multilingual", "long-context", "tools"]
    result = _fmt_capabilities(capabilities, width=23)

    assert result == "vision, multilingual +2"
    assert len(result) <= 23
    assert "…" not in result  # no ellipsis, ever
    shown, _, marker = result.partition(" +")
    shown_items = shown.split(", ") if shown else []
    for item in shown_items:
        assert item in capabilities, f"{item!r} is not a complete capability name"
    assert marker == "2"  # long-context and tools were dropped


def test_when_nothing_whole_fits_only_the_marker_is_shown() -> None:
    from llamafit.cli.render import _fmt_capabilities

    result = _fmt_capabilities(["multilingual", "long-context"], width=3)

    assert result == "+2"
    assert "…" not in result


def test_a_wide_enough_width_shows_the_full_top_three_and_a_marker() -> None:
    from llamafit.cli.render import _fmt_capabilities

    capabilities = ["coding", "thinking", "vision", "tools", "multilingual", "long-context"]
    result = _fmt_capabilities(capabilities, width=100)

    assert result == "coding, thinking, vision +3"


def test_no_marker_when_every_capability_already_fits() -> None:
    from llamafit.cli.render import _fmt_capabilities

    assert _fmt_capabilities(["coding", "tools"], width=100) == "coding, tools"
    assert _fmt_capabilities([], width=100) == ""


# --- render_catalog_list end to end: a long id must never blank another column -


def _rendered(summaries: list[ModelSummary], width: int) -> str:
    """Render the real table through Rich at ``width`` and return the plain text."""
    from rich.console import Console

    from llamafit.cli.render import render_catalog_list

    console = Console(width=width)
    with console.capture() as capture:
        console.print(render_catalog_list(summaries, console_width=width))
    return capture.get()


def _id_column_cells(output: str) -> list[str]:
    """The id (first) column's text from every content line, in top-to-bottom order.

    A row whose id folds onto extra lines contributes one cell per physical line;
    concatenating them in order reconstructs the id without whatever a shorter
    quality or capabilities cell on the very same line would otherwise interleave
    into a reading that just strips separators from the whole block of text.
    """
    cells = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] not in "|│":
            continue
        inner = stripped[1:]
        end = next((i for i, ch in enumerate(inner) if ch in "|│"), len(inner))
        cell = inner[:end].strip()
        if any(ch.isalnum() for ch in cell):  # skip pure separator lines
            cells.append(cell)
    return cells


def _summary(
    model_id: str, quality: int, total_b: float, active_b: float, context: int
) -> ModelSummary:
    return ModelSummary(
        id=model_id,
        name=model_id,
        vendor="Acme",
        params_total_b=total_b,
        params_active_b=active_b,
        capabilities=["vision", "multilingual", "long-context", "tools"],
        context_native=context,
        license_spdx="MIT",
        quant_names=["Q4_K_M"],
        largest_quant_bytes=None,
        quality_baseline=quality,
    )


def test_a_53_character_id_never_blanks_or_cuts_another_column_at_80_or_50() -> None:
    from llamafit.cli.render import (
        _column_budget,
        _fmt_billions,
        _fmt_capabilities,
        _fmt_context_compact,
    )

    # Built rather than hand-counted, to match the reviewer's 53-character report exactly.
    long_id = "custom-model-" + "a" * 37 + "-53"
    assert len(long_id) == 53
    summaries = [
        _summary("short-id", 61, 8, 8, 131072),
        _summary(long_id, 74, 27, 27, 131072),
    ]

    for width in (50, 80):
        _id_width, included, capabilities_width = _column_budget(summaries, width)
        output = _rendered(summaries, width)

        assert "…" not in output  # never an ellipsis
        assert "�" not in output  # never a replacement character either

        for summary in summaries:
            if "quality" in included:
                assert str(summary.quality_baseline) in output, (
                    f"quality {summary.quality_baseline} missing or cut at width {width}"
                )
            if "params" in included:
                expected = (
                    f"{_fmt_billions(summary.params_total_b)}/"
                    f"{_fmt_billions(summary.params_active_b)}B"
                )
                assert expected in output, f"params {expected!r} missing or cut at width {width}"
            if "context" in included:
                expected_ctx = _fmt_context_compact(summary.context_native)
                assert expected_ctx in output, (
                    f"context {expected_ctx!r} missing or cut at width {width}"
                )
            if capabilities_width >= 3:
                expected_caps = _fmt_capabilities(summary.capabilities, width=capabilities_width)
                assert expected_caps in output, (
                    f"capabilities {expected_caps!r} missing or cut at width {width}"
                )

        # The id folds onto extra lines rather than losing characters. Extracting
        # only the first (id) cell of every content line, in order, and joining
        # them reconstructs each row's id without whatever the quality or
        # capabilities cell on the same physical line would otherwise interleave
        # into a naive strip-everything-and-concatenate reading of the output.
        assert long_id in "".join(_id_column_cells(output))
