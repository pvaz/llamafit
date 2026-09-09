from pathlib import Path

import pytest

from llamafit.catalog.loader import custom_models_path, load_catalog, load_models_from_file
from llamafit.errors import CatalogError

ENTRY = """
- id: tiny-1b
  name: Tiny 1B
  vendor: Example
  family: tiny
  release_date: 2026-01-01
  license: {spdx: MIT, url: "https://example.invalid/l"}
  params: {total_b: 1.0, active_b: 1.0}
  architecture: {class: dense, gguf_arch: llama}
  context: {native: 8192}
  capabilities: [coding]
  use_cases: [coding]
  quality: {baseline: 60}
  sources:
    - repo: example/tiny-1b-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""


def write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_valid_file(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "tiny.yaml", ENTRY))
    assert problems == []
    assert [m.id for m in models] == ["tiny-1b"]


def test_broken_yaml_becomes_a_problem_not_an_exception(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "bad.yaml", "- id: [unclosed"))
    assert models == []
    assert len(problems) == 1 and "bad.yaml" in problems[0].file


def test_an_invalid_entry_names_its_field(tmp_path: Path) -> None:
    text = ENTRY.replace("baseline: 60", "baseline: 900")
    models, problems = load_models_from_file(write(tmp_path, "x.yaml", text))
    assert models == []
    assert any("baseline" in p.location for p in problems)


def test_a_custom_entry_replaces_a_bundled_one_and_new_ids_are_added(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "tiny.yaml", ENTRY)
    custom = write(
        tmp_path,
        "custom.yaml",
        ENTRY.replace("Tiny 1B", "Mine")
        + ENTRY.replace("tiny-1b", "other-2b").replace("Tiny 1B", "Other"),
    )
    catalog, problems = load_catalog(bundled_dir=bundled, custom_path=custom)
    assert problems == []
    assert catalog.by_id["tiny-1b"].name == "Mine"
    assert "other-2b" in catalog.by_id
    assert len(catalog.models) == 2


def test_a_duplicate_id_inside_the_bundled_set_is_a_problem(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "a.yaml", ENTRY)
    write(bundled, "b.yaml", ENTRY)
    _, problems = load_catalog(bundled_dir=bundled, custom_path=None)
    assert any("duplicate" in p.message for p in problems)


def test_strict_mode_raises_with_every_problem_listed(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "bad.yaml", "- id: [unclosed")
    with pytest.raises(CatalogError) as info:
        load_catalog(bundled_dir=bundled, custom_path=None, strict=True)
    assert "bad.yaml" in str(info.value)


def test_a_missing_custom_file_is_not_a_problem(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "tiny.yaml", ENTRY)
    catalog, problems = load_catalog(bundled_dir=bundled, custom_path=tmp_path / "absent.yaml")
    assert problems == [] and len(catalog.models) == 1


def test_an_empty_file_has_no_entries_and_no_problems(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "empty.yaml", ""))
    assert models == [] and problems == []


def test_a_non_list_top_level_is_a_problem(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "mapping.yaml", "id: tiny-1b"))
    assert models == []
    assert len(problems) == 1 and "list" in problems[0].message


def test_custom_models_path_honours_the_environment_override(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.yaml"
    assert custom_models_path({"LLAMAFIT_CUSTOM_MODELS": str(target)}) == target


def test_custom_models_path_defaults_to_the_data_directory() -> None:
    path = custom_models_path({})
    assert path.name == "custom_models.yaml"
