from pathlib import Path

from llamafit.catalog.validate import validate_files
from tests.unit.test_catalog_loader import ENTRY, write


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
