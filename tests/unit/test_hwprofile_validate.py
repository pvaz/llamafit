"""The checks a single file cannot make about itself."""

from __future__ import annotations

import json
from pathlib import Path

from llamafit.hwprofile.validate import validate_files
from tests.fixtures import profiles


def test_a_good_file_has_no_problems(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    assert validate_files([tmp_path / "test-machine.json"]) == []


def test_the_bundled_profiles_validate() -> None:
    from llamafit.hwprofile import bundled_profiles_dir, profile_files

    assert validate_files(profile_files(bundled_profiles_dir())) == []


def test_a_broken_file_is_reported_and_does_not_stop_the_next_one(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    profiles.write(tmp_path, profiles.MINIMAL)
    problems = validate_files([tmp_path / "broken.json", tmp_path / "test-machine.json"])
    assert len(problems) == 1 and problems[0].file.endswith("broken.json")


def test_two_files_claiming_one_name(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(profiles.MINIMAL), encoding="utf-8")
    problems = validate_files([tmp_path / "test-machine.json", other])
    duplicates = [p for p in problems if "duplicate profile name" in p.message]
    assert len(duplicates) == 1
    assert "test-machine.json" in duplicates[0].message


def test_a_gpu_figure_that_contradicts_the_bundled_table(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[
            {
                "vendor": "nvidia",
                "name": "NVIDIA GeForce RTX 4060",
                "vram_total": "8GiB",
                "bandwidth_gbps": 500,
            }
        ]
    )
    profiles.write(tmp_path, document)
    problems = validate_files([tmp_path / "test-machine.json"])
    assert [p.location for p in problems] == ["gpus.0.bandwidth_gbps"]
    assert "272" in problems[0].message


def test_a_figure_within_the_tolerance_is_not_reported(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[
            {
                "vendor": "nvidia",
                "name": "NVIDIA GeForce RTX 4060",
                "vram_total": "8GiB",
                "bandwidth_gbps": 272.5,
                "compute_tflops_fp16": 60.9,
            }
        ]
    )
    profiles.write(tmp_path, document)
    assert validate_files([tmp_path / "test-machine.json"]) == []


def test_a_card_the_table_does_not_know_is_never_contradicted(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[{"vendor": "other", "name": "Some Future Card", "bandwidth_gbps": 4000}]
    )
    profiles.write(tmp_path, document)
    assert validate_files([tmp_path / "test-machine.json"]) == []


def test_a_unified_part_described_as_having_its_own_pool(tmp_path: Path) -> None:
    document = profiles.document(gpus=[{"vendor": "apple", "name": "Apple M3 Max"}])
    profiles.write(tmp_path, document)
    problems = validate_files([tmp_path / "test-machine.json"])
    assert [p.location for p in problems] == ["unified_memory"]


def test_match_rules_that_could_never_fire(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(match={"gpu_name_contains": "RTX 4090"}))
    problems = validate_files([tmp_path / "test-machine.json"])
    assert [p.location for p in problems] == ["match"]
    assert "never select it" in problems[0].message


def test_match_rules_that_fit_the_profile_are_fine(tmp_path: Path) -> None:
    document = profiles.with_gpu(match={"gpu_name_contains": "RTX 4090", "total_ram_min": "32GiB"})
    profiles.write(tmp_path, document)
    assert validate_files([tmp_path / "test-machine.json"]) == []


def test_no_match_rules_is_not_a_problem(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    assert validate_files([tmp_path / "test-machine.json"]) == []
