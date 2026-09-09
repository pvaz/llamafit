import json
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


def test_a_missing_facts_file_is_not_a_problem_and_leaves_fields_empty(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "tiny.yaml", ENTRY))
    assert problems == []
    quant = models[0].sources[0].quants[0]
    assert quant.files == []
    assert quant.bytes_ is None
    assert quant.sha256 == []
    assert quant.bpw is None
    assert quant.gguf_facts is None


def test_the_loader_merges_a_facts_file_into_the_quants(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    facts = {
        "schema_version": 1,
        "refreshed_at": "2026-09-09T00:00:00+00:00",
        "models": {
            "tiny-1b": {
                "quants": {
                    "Q4_K_M": {
                        "files": ["tiny-1b-Q4_K_M.gguf"],
                        "bytes": 700_000_000,
                        "sha256": ["abc123"],
                        "bpw": 5.6,
                        "gguf_facts": {"arch": "llama", "n_layer": 16},
                    }
                }
            }
        },
    }
    write(tmp_path, "tiny.facts.json", json.dumps(facts))

    models, problems = load_models_from_file(path)

    assert problems == []
    quant = models[0].sources[0].quants[0]
    assert quant.files == ["tiny-1b-Q4_K_M.gguf"]
    assert quant.bytes_ == 700_000_000
    assert quant.sha256 == ["abc123"]
    assert quant.bpw == 5.6
    assert quant.gguf_facts is not None and quant.gguf_facts.n_layer == 16


def test_a_facts_file_naming_an_unknown_quant_is_a_problem_naming_both(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    facts = {
        "schema_version": 1,
        "refreshed_at": "2026-09-09T00:00:00+00:00",
        "models": {"tiny-1b": {"quants": {"Q9_TYPO": {"bytes": 1}}}},
    }
    write(tmp_path, "tiny.facts.json", json.dumps(facts))

    models, problems = load_models_from_file(path)

    assert [m.id for m in models] == ["tiny-1b"]
    assert any(p.model_id == "tiny-1b" and "Q9_TYPO" in p.message for p in problems)


def test_a_facts_file_naming_an_unknown_model_is_a_problem_naming_both(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    facts = {
        "schema_version": 1,
        "refreshed_at": "2026-09-09T00:00:00+00:00",
        "models": {"no-such-model": {}},
    }
    write(tmp_path, "tiny.facts.json", json.dumps(facts))

    _, problems = load_models_from_file(path)

    assert any(p.model_id == "no-such-model" and "no-such-model" in p.message for p in problems)


def test_a_quant_name_reused_across_sources_is_a_problem_naming_both_repos(
    tmp_path: Path,
) -> None:
    text = """
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
    - repo: official/tiny-1b-GGUF
      trust: official
      quants:
        - {name: Q4_K_M}
    - repo: community/tiny-1b-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""
    models, problems = load_models_from_file(write(tmp_path, "dup.yaml", text))

    assert [m.id for m in models] == ["tiny-1b"]
    matches = [p for p in problems if p.model_id == "tiny-1b"]
    assert len(matches) == 1
    assert matches[0].location == "quants.Q4_K_M"
    assert "official/tiny-1b-GGUF" in matches[0].message
    assert "community/tiny-1b-GGUF" in matches[0].message
    assert "unique within a model" in matches[0].message


def test_distinct_quant_names_across_sources_are_not_a_problem(tmp_path: Path) -> None:
    text = """
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
    - repo: official/tiny-1b-GGUF
      trust: official
      quants:
        - {name: Q4_K_M}
    - repo: community/tiny-1b-GGUF
      trust: community
      quants:
        - {name: UD-Q4_K_XL}
"""
    _, problems = load_models_from_file(write(tmp_path, "ok.yaml", text))
    assert problems == []
