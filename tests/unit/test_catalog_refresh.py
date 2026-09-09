import json
from pathlib import Path

import pytest

from llamafit.catalog.hf import FakeHfClient, RepoFile
from llamafit.catalog.loader import load_models_from_file
from llamafit.catalog.refresh import refresh_file
from llamafit.errors import CatalogError, NetworkError
from llamafit.models.gguf import GgufFacts
from tests.unit.test_catalog_loader import ENTRY, write

FILES = {
    "example/tiny-1b-GGUF": [
        RepoFile(path="tiny-1b-Q4_K_M.gguf", size=700_000_000, sha256="abc123"),
        RepoFile(path="README.md", size=10, sha256=None),
    ]
}

# Carries a comment, so a refresh that touches the YAML at all would show up as a
# diff even where the data itself did not change.
ENTRY_WITH_COMMENT = """# Sourced from the vendor's own model card; see the license URL below.
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
        - {name: Q4_K_M}  # the only quant this repo publishes
"""

MULTI_ENTRY = """
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
      extras:
        - {role: mmproj, file: mmproj-tiny.gguf}
    - kind: local
      path: D:/models/tiny.gguf
      quants:
        - {name: Q4_K_M-local}
- id: other-2b
  name: Other 2B
  vendor: Example
  family: tiny
  release_date: 2026-01-01
  license: {spdx: MIT, url: "https://example.invalid/l"}
  params: {total_b: 2.0, active_b: 2.0}
  architecture: {class: dense, gguf_arch: llama}
  context: {native: 8192}
  capabilities: [coding]
  use_cases: [coding]
  quality: {baseline: 60}
  sources:
    - repo: example/other-2b-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""

MULTI_FILES = {
    "example/tiny-1b-GGUF": [
        RepoFile(path="tiny-1b-Q4_K_M.gguf", size=700_000_000, sha256="abc123"),
        RepoFile(path="mmproj-tiny.gguf", size=1_000_000, sha256="def456"),
        RepoFile(path="README.md", size=10, sha256=None),
    ],
    "example/other-2b-GGUF": [
        RepoFile(path="other-2b-Q4_K_M.gguf", size=1_400_000_000, sha256="ghi789"),
    ],
}

TWO_SOURCE_ENTRY = """
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
    - repo: example/missing-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""


class RaisingHfClient:
    """A stand-in ``HfClient`` whose listing always raises, for error-path tests."""

    def list_files(self, repo: str) -> list[RepoFile]:
        raise NetworkError(f"could not reach {repo}")

    def file_url(self, repo: str, path: str) -> str:
        return f"https://huggingface.co/{repo}/resolve/main/{path}"


def facts_stub(*args: object, **kwargs: object) -> GgufFacts:
    return GgufFacts(
        arch="llama",
        n_layer=16,
        n_embd=2048,
        n_vocab=32000,
        n_head=16,
        n_head_kv=8,
        head_dim=128,
        attention_layers=16,
        attention_layers_source="tensors",
        bytes_expert_weights=0,
        bytes_attention_weights=1,
        bytes_output_head=1,
        bytes_token_embd=1,
        bytes_lazy_tables=0,
        bytes_total=3,
        has_shared_experts=False,
    )


def facts_path_of(path: Path) -> Path:
    return path.with_name(f"{path.stem}.facts.json")


def test_refresh_fills_the_facts_file(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert [r.changed for r in results] == [True]
    assert results[0].warnings == []

    assert facts_path_of(path).is_file()
    models, problems = load_models_from_file(path)
    assert problems == []
    quant = models[0].sources[0].quants[0]
    assert quant.files == ["tiny-1b-Q4_K_M.gguf"]
    assert quant.bytes_ == 700_000_000
    assert quant.sha256 == ["abc123"]
    assert quant.bpw is not None and 5.5 < quant.bpw < 5.7
    assert quant.gguf_facts is not None and quant.gguf_facts.n_layer == 16


def test_a_refresh_never_touches_the_curated_yaml_bytes(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY_WITH_COMMENT)
    before = path.read_bytes()
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert results[0].changed is True
    assert path.read_bytes() == before


def test_a_dry_run_writes_no_facts_file(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub, dry_run=True)
    assert results[0].changed is True
    assert not facts_path_of(path).exists()


def test_two_refreshes_over_unchanged_data_produce_a_byte_identical_facts_file(
    tmp_path: Path,
) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    first = facts_path_of(path).read_bytes()

    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)

    assert [r.changed for r in results] == [False]
    assert facts_path_of(path).read_bytes() == first


def test_a_failing_repository_leaves_everything_untouched(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    before = path.read_bytes()
    results = refresh_file(path, hf=FakeHfClient({}), read_facts_fn=facts_stub)
    assert results[0].error is not None
    assert path.read_bytes() == before
    assert not facts_path_of(path).exists()


def test_a_listing_exception_leaves_the_model_untouched(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    before = path.read_bytes()
    results = refresh_file(path, hf=RaisingHfClient(), read_facts_fn=facts_stub)
    assert results[0].error is not None and "could not reach" in results[0].error
    assert path.read_bytes() == before
    assert not facts_path_of(path).exists()


def test_only_limits_the_refresh_to_one_model(tmp_path: Path) -> None:
    path = write(tmp_path, "multi.yaml", MULTI_ENTRY)
    results = refresh_file(
        path, hf=FakeHfClient(MULTI_FILES), read_facts_fn=facts_stub, only="tiny-1b"
    )
    assert [r.model_id for r in results] == ["tiny-1b"]
    models, _ = load_models_from_file(path)
    by_id = {m.id: m for m in models}
    assert by_id["tiny-1b"].sources[0].quants[0].files == ["tiny-1b-Q4_K_M.gguf"]
    assert by_id["other-2b"].sources[0].quants[0].files == []

    facts = json.loads(facts_path_of(path).read_text(encoding="utf-8"))
    assert "other-2b" not in facts["models"]


def test_a_local_source_is_left_alone(tmp_path: Path) -> None:
    path = write(tmp_path, "multi.yaml", MULTI_ENTRY)
    refresh_file(path, hf=FakeHfClient(MULTI_FILES), read_facts_fn=facts_stub, only="tiny-1b")
    models, _ = load_models_from_file(path)
    local_source = models[0].sources[1]
    assert local_source.kind == "local"
    assert local_source.quants[0].files == []


def test_an_extra_is_filled_by_exact_file_name(tmp_path: Path) -> None:
    path = write(tmp_path, "multi.yaml", MULTI_ENTRY)
    refresh_file(path, hf=FakeHfClient(MULTI_FILES), read_facts_fn=facts_stub, only="tiny-1b")
    models, _ = load_models_from_file(path)
    extra = models[0].sources[0].extras[0]
    assert extra.bytes_ == 1_000_000
    assert extra.sha256 == "def456"


def test_a_second_failing_source_aborts_the_whole_model(tmp_path: Path) -> None:
    path = write(tmp_path, "two.yaml", TWO_SOURCE_ENTRY)
    before = path.read_bytes()
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert results[0].error is not None
    assert results[0].changed is False
    assert path.read_bytes() == before
    assert not facts_path_of(path).exists()


def test_a_quant_with_no_matching_files_is_a_warning_not_a_no_op(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY.replace("name: Q4_K_M", "name: Q9_TYPO"))
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert results[0].error is None
    assert results[0].changed is False
    assert any("Q9_TYPO" in warning for warning in results[0].warnings)


def test_a_file_with_load_problems_raises_a_catalog_error(tmp_path: Path) -> None:
    path = write(tmp_path, "bad.yaml", "- id: [unclosed")
    with pytest.raises(CatalogError):
        refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
