"""Reading profile files: what loads, what becomes a problem, and what raises.

The division is the catalog loader's, deliberately: a malformed file is a ``Problem``
somebody can be shown, and only a reference that resolves to nothing raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamafit.errors import ConfigError
from llamafit.hwprofile.loader import (
    bundled_profiles_dir,
    load_profile_file,
    load_profiles,
    profile_files,
    resolve_profile,
    user_profiles_dir,
)
from tests.fixtures import profiles


def test_a_minimal_document_loads(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert problems == []
    assert profile is not None
    assert profile.name == "test-machine"
    assert profile.memory.total == 64 * 1024**3
    assert profile.gpus == []


def test_sizes_may_be_written_the_way_the_flags_are(tmp_path: Path) -> None:
    profiles.write(
        tmp_path,
        profiles.document(memory={"total": "8G", "available": 4_000_000_000}),
    )
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert problems == []
    assert profile is not None
    assert profile.memory.total == 8_000_000_000
    assert profile.memory.available == 4_000_000_000


def test_a_size_that_is_not_a_size_is_a_problem_not_an_exception(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(memory={"total": "lots"}))
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert [p.location for p in problems] == ["memory.total"]
    assert "not a size" in problems[0].message


def test_invalid_json_is_a_problem(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    profile, problems = load_profile_file(path)
    assert profile is None
    assert problems[0].location == "file" and "invalid JSON" in problems[0].message


def test_a_document_that_is_not_an_object_is_a_problem(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    profile, problems = load_profile_file(path)
    assert profile is None
    assert "one machine" in problems[0].message


def test_a_file_that_cannot_be_read_is_a_problem(tmp_path: Path) -> None:
    profile, problems = load_profile_file(tmp_path / "absent.json")
    assert profile is None
    assert problems[0].location == "file"


def test_an_unknown_field_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    """A field nobody reads would look exactly like a field that never applied."""
    profiles.write(tmp_path, profiles.document(pcie_bandwidth_gbps=12))
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert [p.location for p in problems] == ["pcie_bandwidth_gbps"]


def test_a_bandwidth_without_a_label_is_refused(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(memory={"total": "64GiB", "bandwidth_gbps": 57.0}))
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert "bandwidth_source" in problems[0].message


def test_a_label_without_a_bandwidth_is_refused(tmp_path: Path) -> None:
    profiles.write(
        tmp_path, profiles.document(memory={"total": "64GiB", "bandwidth_source": "measured"})
    )
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert "no bandwidth_gbps" in problems[0].message


def test_a_schema_version_this_build_does_not_read_is_refused(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.document(schema_version=99))
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert "reads version 1" in problems[0].message


def test_a_name_that_could_not_be_typed_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(profiles.document(name="My Machine")), encoding="utf-8")
    profile, problems = load_profile_file(path)
    assert profile is None
    assert "not usable as a profile name" in problems[0].message


def test_a_name_that_disagrees_with_its_file_still_loads_but_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps(profiles.MINIMAL), encoding="utf-8")
    profile, problems = load_profile_file(path)
    assert profile is not None, "the machine is usable; only its file name is wrong"
    assert [p.location for p in problems] == ["name"]
    assert "rename one to match the other" in problems[0].message


def test_a_unified_machine_may_not_give_a_card_its_own_vram(tmp_path: Path) -> None:
    profiles.write(
        tmp_path,
        profiles.document(
            unified_memory=True,
            gpus=[{"vendor": "apple", "name": "Apple M3 Max", "vram_total": "48GiB"}],
        ),
    )
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert "no separate VRAM pool" in problems[0].message


def test_the_user_directory_wins_over_a_bundled_profile_of_the_same_name(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.document(description="theirs"))
    user = profiles.write(tmp_path / "user", profiles.document(description="mine"))
    loaded, problems = load_profiles(bundled_dir=bundled, user_dir=user)
    assert problems == []
    assert [p.name for p in loaded] == ["test-machine"]
    assert loaded[0].profile.description == "mine"
    assert loaded[0].bundled is False


def test_a_user_profile_with_a_new_name_is_appended(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.MINIMAL)
    user = profiles.write(tmp_path / "user", profiles.document(name="mine"))
    loaded, _problems = load_profiles(bundled_dir=bundled, user_dir=user)
    assert [p.name for p in loaded] == ["test-machine", "mine"]
    assert [p.bundled for p in loaded] == [True, False]


def test_two_bundled_files_claiming_one_name_report_it_and_keep_the_first(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    profiles.write(bundled, profiles.document(description="first"))
    # Named so it sorts after the first: the loader reads a directory in name order.
    (bundled / "zz-second.json").write_text(
        json.dumps(profiles.document(description="second")), encoding="utf-8"
    )
    loaded, problems = load_profiles(bundled_dir=bundled, user_dir=tmp_path / "none")
    assert len(loaded) == 1 and loaded[0].profile.description == "first"
    # Two problems: the second file's stem disagrees with its name, and the name is taken.
    assert {p.location for p in problems} == {"name"}
    assert any("duplicate" in p.message for p in problems)


def test_a_missing_directory_is_simply_no_profiles(tmp_path: Path) -> None:
    loaded, problems = load_profiles(bundled_dir=tmp_path / "nope", user_dir=tmp_path / "nor")
    assert loaded == [] and problems == []
    assert profile_files(tmp_path / "nope") == []


def test_resolve_by_name(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.MINIMAL)
    loaded, problems = resolve_profile(
        "test-machine", bundled_dir=bundled, user_dir=tmp_path / "none"
    )
    assert loaded.name == "test-machine" and problems == []


def test_resolve_by_path(tmp_path: Path) -> None:
    profiles.write(tmp_path, profiles.MINIMAL)
    loaded, problems = resolve_profile(str(tmp_path / "test-machine.json"))
    assert loaded.name == "test-machine" and problems == []
    assert loaded.bundled is False


def test_an_unknown_name_names_the_ones_that_exist(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.MINIMAL)
    with pytest.raises(ConfigError) as info:
        resolve_profile("nosuch", bundled_dir=bundled, user_dir=tmp_path / "none")
    rendered = info.value.render()
    assert "no hardware profile named 'nosuch'" in rendered
    assert "test-machine" in rendered


def test_an_unknown_name_with_no_profiles_at_all_points_at_the_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no hardware profile named"):
        resolve_profile("nosuch", bundled_dir=tmp_path / "a", user_dir=tmp_path / "b")


def test_a_path_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no hardware profile file at"):
        resolve_profile(str(tmp_path / "absent.json"))


def test_a_path_to_a_broken_file_says_what_is_broken(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        resolve_profile(str(path))
    assert "invalid JSON" in info.value.render()


def test_the_bundled_directory_resolves_and_holds_profiles() -> None:
    assert profile_files(bundled_profiles_dir())


def test_the_user_directory_follows_the_environment(tmp_path: Path) -> None:
    assert user_profiles_dir({"LLAMAFIT_PROFILES": str(tmp_path)}) == tmp_path
    assert user_profiles_dir({"LLAMAFIT_HOME": str(tmp_path)}) == tmp_path / "data" / "profiles"


@pytest.mark.parametrize(
    ("document", "location", "fragment"),
    [
        (
            profiles.document(
                cpu={"model": "c", "physical_cores": 8, "logical_cores": 4},
            ),
            "cpu",
            "logical_cores cannot be fewer",
        ),
        (
            profiles.document(
                cpu={"model": "c", "physical_cores": 8, "performance_cores": 12},
            ),
            "cpu",
            "performance_cores cannot be more",
        ),
        (
            profiles.document(memory={"total": "8GiB", "available": "16GiB"}),
            "memory",
            "available memory cannot be more",
        ),
        (
            profiles.document(
                gpus=[
                    {
                        "vendor": "nvidia",
                        "name": "NVIDIA GeForce RTX 4090",
                        "vram_total": "8GiB",
                        "vram_used": "16GiB",
                    }
                ]
            ),
            "gpus.0",
            "vram_used cannot be more",
        ),
    ],
)
def test_a_figure_that_contradicts_its_neighbour_is_refused(
    tmp_path: Path, document: dict[str, object], location: str, fragment: str
) -> None:
    """A pool cannot hold more than it has; the schema says so rather than the reader."""
    profiles.write(tmp_path, document)
    profile, problems = load_profile_file(tmp_path / "test-machine.json")
    assert profile is None
    assert [p.location for p in problems] == [location]
    assert fragment in problems[0].message


def test_a_broken_file_in_a_directory_does_not_stop_the_others(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.MINIMAL)
    (bundled / "broken.json").write_text("{", encoding="utf-8")
    loaded, problems = load_profiles(bundled_dir=bundled, user_dir=tmp_path / "none")
    assert [p.name for p in loaded] == ["test-machine"]
    assert len(problems) == 1 and problems[0].file.endswith("broken.json")


def test_a_broken_user_file_does_not_stop_the_others(tmp_path: Path) -> None:
    bundled = profiles.write(tmp_path / "bundled", profiles.MINIMAL)
    user = tmp_path / "user"
    user.mkdir()
    (user / "broken.json").write_text("{", encoding="utf-8")
    loaded, problems = load_profiles(bundled_dir=bundled, user_dir=user)
    assert [p.name for p in loaded] == ["test-machine"]
    assert len(problems) == 1
