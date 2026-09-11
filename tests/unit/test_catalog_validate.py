from pathlib import Path

from llamafit.catalog.validate import validate_files
from tests.unit.test_catalog_loader import ENTRY, write, write_facts


def test_a_clean_set_of_files_has_no_problems(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    assert validate_files([path]) == []


def test_broken_yaml_is_reported_by_file(tmp_path: Path) -> None:
    path = write(tmp_path, "bad.yaml", "- id: [unclosed")
    problems = validate_files([path])
    assert len(problems) == 1
    assert "bad.yaml" in problems[0].file


def test_a_duplicate_id_across_two_files_is_a_problem(tmp_path: Path) -> None:
    first = write(tmp_path, "a.yaml", ENTRY)
    second = write(tmp_path, "b.yaml", ENTRY)
    problems = validate_files([first, second])
    assert any("duplicate" in p.message for p in problems)


def facts_for(context_length: int | None, quant: str = "Q4_K_M") -> dict[str, object]:
    return {
        "tiny-1b": {
            "quants": {
                quant: {
                    "files": ["tiny-1b-Q4_K_M.gguf"],
                    "bytes": 700_000_000,
                    "sha256": ["abc123"],
                    "bpw": 5.6,
                    "gguf_facts": {
                        "arch": "llama",
                        "n_layer": 16,
                        "context_length": context_length,
                        "bytes_dense_block_weights": 699_000_000,
                        "bytes_total": 699_000_000,
                    },
                }
            }
        }
    }


def test_a_curated_context_longer_than_the_file_declares_is_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY.replace("native: 8192", "native: 131072"))
    write_facts(tmp_path, facts_for(8192))

    problems = validate_files([path])

    assert [p.location for p in problems] == ["context.native"]
    assert "131,072" in problems[0].message and "8,192" in problems[0].message
    assert "Q4_K_M" in problems[0].message


def test_a_curated_context_shorter_than_the_file_declares_is_not_a_problem(tmp_path: Path) -> None:
    # Qwen3-0.6B really is curated at its documented 32,768 against a header that
    # declares 40,960: a deliberate conservative choice, not a mistake.
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, facts_for(40960))

    assert validate_files([path]) == []


def test_a_curated_context_equal_to_the_file_is_not_a_problem(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, facts_for(8192))

    assert validate_files([path]) == []


def test_a_file_that_declares_no_context_is_not_compared(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(tmp_path, facts_for(None))

    assert validate_files([path]) == []


def test_an_unrefreshed_model_is_not_compared(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY.replace("native: 8192", "native: 999999"))

    assert validate_files([path]) == []


def test_a_measured_context_is_never_compared_with_the_curated_one(tmp_path: Path) -> None:
    # measured[].context is the context one benchmark ran at, not a claim about the
    # model, so a short measurement must not read as a contradiction.
    text = (
        ENTRY
        + """  measured:
    - {profile: bench, quant: Q4_K_M, context: 4096}
"""
    )
    path = write(tmp_path, "tiny.yaml", text)
    write_facts(tmp_path, facts_for(8192))

    assert validate_files([path]) == []


def test_facts_whose_tensors_do_not_account_for_the_files_are_reported(tmp_path: Path) -> None:
    # `catalog validate` said "no problems" over a tree whose gpt-oss facts summed a 63 GB
    # file to 2.4 GB. The loader now refuses such facts, and validate is where it shows.
    path = write(tmp_path, "tiny.yaml", ENTRY)
    write_facts(
        tmp_path,
        {
            "tiny-1b": {
                "quants": {
                    "Q4_K_M": {
                        "files": ["tiny-1b-Q4_K_M.gguf"],
                        "bytes": 700_000_000,
                        "gguf_facts": {
                            "arch": "llama",
                            "bytes_expert_weights": 3,
                            "bytes_total": 3,
                        },
                    }
                }
            }
        },
    )

    problems = validate_files([path])

    assert [p.location for p in problems] == ["quants.Q4_K_M.gguf_facts"]
    assert problems[0].model_id == "tiny-1b"
    assert "3 bytes" in problems[0].message and "700,000,000" in problems[0].message
