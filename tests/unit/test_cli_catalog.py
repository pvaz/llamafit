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


def test_search_is_list_search() -> None:
    result = runner.invoke(app, ["search", "chat"])
    assert result.exit_code == 0, result.output
    assert "chat-model" in result.output
    assert "coder-only" not in result.output


# --- info ----------------------------------------------------------------------


def test_info_shows_vendor_licence_and_quant_details() -> None:
    result = runner.invoke(app, ["info", "coder-with-tools"])
    assert result.exit_code == 0, result.output
    assert "Acme Robotics" in result.output
    assert "Apache-2.0" in result.output
    assert "Q4_K_M" in result.output
    assert "4.50" in result.output  # bits per weight


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
    hostile = minimal(
        id="hostile-model",
        name="Hostile [Name]",
        vendor="Vendor [Co]",
        license=License(spdx="weird-[licence]", url="https://example.invalid/l"),
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
    assert "Vendor [Co]" in info_result.output
    assert "weird-[licence]" in info_result.output


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
