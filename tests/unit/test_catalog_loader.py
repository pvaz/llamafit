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


ENTRY_WITH_EXTRA = (
    ENTRY
    + """      extras:
        - {role: mmproj, file: mmproj-F16.gguf}
"""
)

DUPLICATE_EXTRA_ENTRY = """
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
      extras:
        - {role: mmproj, file: mmproj-F16.gguf}
    - repo: community/tiny-1b-GGUF
      trust: community
      quants:
        - {name: UD-Q4_K_XL}
      extras:
        - {role: mmproj, file: mmproj-F16.gguf}
"""

DUPLICATE_QUANT_ENTRY = """
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

GOOD_QUANT_FACTS = {
    "files": ["tiny-1b-Q4_K_M.gguf"],
    "bytes": 700_000_000,
    "sha256": ["abc123"],
    "bpw": 5.6,
    "gguf_facts": {"arch": "llama", "n_layer": 16},
}


def write_document(directory: Path, document: object) -> Path:
    return write(directory, "tiny.facts.json", json.dumps(document))


def write_facts(directory: Path, models: object, schema_version: object = 1) -> Path:
    return write_document(
        directory,
        {
            "schema_version": schema_version,
            "refreshed_at": "2026-09-09T00:00:00+00:00",
            "models": models,
        },
    )


@pytest.mark.parametrize("version", [999, None, True, "1"])
def test_a_facts_file_this_build_cannot_read_merges_nothing(
    tmp_path: Path, version: object
) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, {"tiny-1b": {"quants": {"Q4_K_M": GOOD_QUANT_FACTS}}}, version)

    models, problems = load_models_from_file(path)

    assert len(problems) == 1
    assert problems[0].location == "schema_version"
    assert "tiny.facts.json" in problems[0].file
    assert "1" in problems[0].message
    assert models[0].sources[0].quants[0].bytes_ is None


def test_a_facts_file_with_no_schema_version_merges_nothing(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_document(tmp_path, {"models": {"tiny-1b": {"quants": {"Q4_K_M": GOOD_QUANT_FACTS}}}})

    models, problems = load_models_from_file(path)

    assert len(problems) == 1 and problems[0].location == "schema_version"
    assert models[0].sources[0].quants[0].bytes_ is None


def test_a_facts_file_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_document(tmp_path, ["not", "an", "object"])

    _, problems = load_models_from_file(path)

    assert len(problems) == 1 and "models" in problems[0].message


def test_a_facts_file_without_a_models_mapping_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, ["tiny-1b"])

    _, problems = load_models_from_file(path)

    assert len(problems) == 1 and "models" in problems[0].message


def test_a_model_entry_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, {"tiny-1b": "not an object"})

    models, problems = load_models_from_file(path)

    assert [m.id for m in models] == ["tiny-1b"]
    assert len(problems) == 1
    assert problems[0].model_id == "tiny-1b"
    assert "a string" in problems[0].message


def test_a_quants_section_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, {"tiny-1b": {"quants": ["Q4_K_M"]}})

    _, problems = load_models_from_file(path)

    assert len(problems) == 1
    assert problems[0].location == "quants"
    assert "a list" in problems[0].message


def test_a_quant_entry_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, {"tiny-1b": {"quants": {"Q4_K_M": 700}}})

    models, problems = load_models_from_file(path)

    assert len(problems) == 1
    assert problems[0].location == "quants.Q4_K_M"
    assert "a number" in problems[0].message
    assert models[0].sources[0].quants[0].bytes_ is None


def test_every_malformed_quant_field_is_a_problem_and_stays_unset(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(
        tmp_path,
        {
            "tiny-1b": {
                "quants": {
                    "Q4_K_M": {
                        "files": "tiny-1b-Q4_K_M.gguf",
                        "bytes": "700000000",
                        "sha256": "abc123",
                        "bpw": "5.6",
                        "gguf_facts": ["llama"],
                    }
                }
            }
        },
    )

    models, problems = load_models_from_file(path)

    assert {p.location for p in problems} == {
        "quants.Q4_K_M.files",
        "quants.Q4_K_M.bytes",
        "quants.Q4_K_M.sha256",
        "quants.Q4_K_M.bpw",
        "quants.Q4_K_M.gguf_facts",
    }
    assert all(p.model_id == "tiny-1b" for p in problems)
    assert all("Q4_K_M" in p.message for p in problems)
    quant = models[0].sources[0].quants[0]
    assert quant.files == []
    assert quant.bytes_ is None
    assert quant.sha256 == []
    assert quant.bpw is None
    assert quant.gguf_facts is None


def test_gguf_facts_that_fail_validation_are_a_problem_not_a_silent_drop(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(
        tmp_path,
        {"tiny-1b": {"quants": {"Q4_K_M": {"gguf_facts": {"arch": "llama", "n_layer": "many"}}}}},
    )

    models, problems = load_models_from_file(path)

    assert len(problems) == 1
    assert problems[0].location == "quants.Q4_K_M.gguf_facts"
    assert "n_layer" in problems[0].message
    assert models[0].sources[0].quants[0].gguf_facts is None


def test_one_malformed_field_does_not_stop_the_others(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(
        tmp_path,
        {"tiny-1b": {"quants": {"Q4_K_M": {**GOOD_QUANT_FACTS, "bytes": "700000000"}}}},
    )

    models, problems = load_models_from_file(path)

    assert [p.location for p in problems] == ["quants.Q4_K_M.bytes"]
    quant = models[0].sources[0].quants[0]
    assert quant.bytes_ is None
    assert quant.files == ["tiny-1b-Q4_K_M.gguf"]
    assert quant.sha256 == ["abc123"]
    assert quant.bpw == 5.6
    assert quant.gguf_facts is not None


def test_a_null_field_is_not_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(
        tmp_path,
        {
            "tiny-1b": {
                "quants": {
                    "Q4_K_M": {
                        "files": [],
                        "bytes": None,
                        "sha256": [],
                        "bpw": None,
                        "gguf_facts": None,
                    }
                }
            }
        },
    )

    models, problems = load_models_from_file(path)

    assert problems == []
    assert models[0].sources[0].quants[0].bytes_ is None


def test_a_malformed_extra_field_is_a_problem_and_stays_unset(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY_WITH_EXTRA)
    write_facts(
        tmp_path,
        {"tiny-1b": {"extras": {"mmproj-F16.gguf": {"bytes": "900", "sha256": ["def456"]}}}},
    )

    models, problems = load_models_from_file(path)

    assert {p.location for p in problems} == {
        "extras.mmproj-F16.gguf.bytes",
        "extras.mmproj-F16.gguf.sha256",
    }
    assert all("mmproj-F16.gguf" in p.message for p in problems)
    extra = models[0].sources[0].extras[0]
    assert extra.bytes_ is None
    assert extra.sha256 is None


def test_an_extra_entry_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY_WITH_EXTRA)
    write_facts(tmp_path, {"tiny-1b": {"extras": {"mmproj-F16.gguf": "900"}}})

    _, problems = load_models_from_file(path)

    assert len(problems) == 1 and problems[0].location == "extras.mmproj-F16.gguf"


def test_an_extra_file_name_reused_across_sources_is_a_problem_naming_both_repos(
    tmp_path: Path,
) -> None:
    models, problems = load_models_from_file(write(tmp_path, "dup.yaml", DUPLICATE_EXTRA_ENTRY))

    assert [m.id for m in models] == ["tiny-1b"]
    assert len(problems) == 1
    assert problems[0].location == "extras.mmproj-F16.gguf"
    assert "official/tiny-1b-GGUF" in problems[0].message
    assert "community/tiny-1b-GGUF" in problems[0].message
    assert "unique within a model" in problems[0].message


def test_distinct_extra_file_names_across_sources_are_not_a_problem(tmp_path: Path) -> None:
    text = DUPLICATE_EXTRA_ENTRY.replace(
        "- {role: mmproj, file: mmproj-F16.gguf}\n    - repo: community",
        "- {role: mmproj, file: mmproj-official-F16.gguf}\n    - repo: community",
    )
    _, problems = load_models_from_file(write(tmp_path, "ok.yaml", text))
    assert problems == []


def test_a_model_whose_sources_reuse_a_quant_name_merges_no_facts(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", DUPLICATE_QUANT_ENTRY)
    write_facts(tmp_path, {"tiny-1b": {"quants": {"Q4_K_M": GOOD_QUANT_FACTS}}})

    models, problems = load_models_from_file(path)

    assert [m.id for m in models] == ["tiny-1b"]
    assert models[0].name == "Tiny 1B"
    assert [p.location for p in problems] == ["quants.Q4_K_M"]
    for source in models[0].sources:
        quant = source.quants[0]
        assert quant.files == []
        assert quant.bytes_ is None
        assert quant.sha256 == []
        assert quant.bpw is None
        assert quant.gguf_facts is None


def test_a_model_whose_sources_reuse_an_extra_name_merges_no_facts(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", DUPLICATE_EXTRA_ENTRY)
    write_facts(
        tmp_path,
        {
            "tiny-1b": {
                "quants": {"Q4_K_M": GOOD_QUANT_FACTS},
                "extras": {"mmproj-F16.gguf": {"bytes": 900, "sha256": "def456"}},
            }
        },
    )

    models, problems = load_models_from_file(path)

    assert [p.location for p in problems] == ["extras.mmproj-F16.gguf"]
    assert models[0].sources[0].quants[0].bytes_ is None
    assert models[0].sources[0].extras[0].bytes_ is None


def test_a_facts_file_naming_an_unknown_extra_is_a_problem_naming_both(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY_WITH_EXTRA)
    write_facts(tmp_path, {"tiny-1b": {"extras": {"mmproj-TYPO.gguf": {"bytes": 1}}}})

    models, problems = load_models_from_file(path)

    assert [m.id for m in models] == ["tiny-1b"]
    assert len(problems) == 1
    assert problems[0].model_id == "tiny-1b"
    assert "mmproj-TYPO.gguf" in problems[0].message


def test_an_extra_field_that_was_never_refreshed_is_not_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY_WITH_EXTRA)
    write_facts(tmp_path, {"tiny-1b": {"extras": {"mmproj-F16.gguf": {"bytes": None}}}})

    models, problems = load_models_from_file(path)

    assert problems == []
    extra = models[0].sources[0].extras[0]
    assert extra.bytes_ is None and extra.sha256 is None


def test_a_facts_file_that_is_not_json_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write(tmp_path, "tiny.facts.json", "{not json")

    models, problems = load_models_from_file(path)

    assert [m.id for m in models] == ["tiny-1b"]
    assert len(problems) == 1 and "invalid JSON" in problems[0].message


def test_a_quant_name_reused_by_a_third_source_is_reported_once(tmp_path: Path) -> None:
    text = (
        DUPLICATE_QUANT_ENTRY
        + """    - repo: third/tiny-1b-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""
    )
    _, problems = load_models_from_file(write(tmp_path, "dup3.yaml", text))

    assert [p.location for p in problems] == ["quants.Q4_K_M"]
