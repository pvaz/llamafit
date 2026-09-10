"""Which profile, if any, a live scan recognises itself in."""

from __future__ import annotations

from pathlib import Path

from llamafit.hwprofile.loader import LoadedProfile, load_profile_file
from llamafit.hwprofile.match import best_match, matches
from llamafit.models.hwprofile import HardwareProfile
from tests.fixtures import profiles
from tests.fixtures.budget_hosts import GIB, machine, reference_host


def _profile(tmp_path: Path, document: dict[str, object]) -> HardwareProfile:
    profiles.write(tmp_path, document)
    profile, problems = load_profile_file(tmp_path / f"{document['name']}.json")
    assert problems == [] and profile is not None
    return profile


def _loaded(profile: HardwareProfile, *, bundled: bool) -> LoadedProfile:
    return LoadedProfile(profile=profile, path=Path(f"{profile.name}.json"), bundled=bundled)


def test_a_profile_with_no_rules_matches_nothing(tmp_path: Path) -> None:
    """A profile of somebody else's machine should not claim to be yours."""
    assert matches(_profile(tmp_path, profiles.MINIMAL), machine()) is False


def test_every_rule_must_hold(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        profiles.document(match={"gpu_name_contains": "RTX 4060", "total_ram_min": "100GiB"}),
    )
    assert matches(profile, reference_host()) is True


def test_one_rule_that_fails_fails_the_match(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        profiles.document(match={"gpu_name_contains": "RTX 4060", "total_ram_min": "512GiB"}),
    )
    assert matches(profile, reference_host()) is False


def test_the_card_name_is_matched_case_insensitively(tmp_path: Path) -> None:
    profile = _profile(tmp_path, profiles.document(match={"gpu_name_contains": "rtx 4060"}))
    assert matches(profile, reference_host()) is True


def test_the_cpu_model_is_matched_case_insensitively(tmp_path: Path) -> None:
    profile = _profile(tmp_path, profiles.document(match={"cpu_model_contains": "I9-14900KF"}))
    assert matches(profile, reference_host()) is True


def test_a_card_rule_fails_on_a_machine_with_no_card(tmp_path: Path) -> None:
    profile = _profile(tmp_path, profiles.document(match={"gpu_name_contains": "RTX 4060"}))
    assert matches(profile, machine(vram_total=None)) is False


def test_no_profile_matches_is_no_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path, profiles.document(match={"gpu_name_contains": "RTX 4060"}))
    assert best_match([_loaded(profile, bundled=True)], machine(vram_total=None)) is None


def test_the_most_specific_match_wins(tmp_path: Path) -> None:
    loose = _profile(
        tmp_path / "a", profiles.document(name="loose", match={"total_ram_min": "1GiB"})
    )
    exact = _profile(
        tmp_path / "b",
        profiles.document(
            name="exact",
            match={
                "gpu_name_contains": "RTX 4060",
                "cpu_model_contains": "i9-14900KF",
                "total_ram_min": "100GiB",
            },
        ),
    )
    chosen = best_match(
        [_loaded(loose, bundled=True), _loaded(exact, bundled=True)], reference_host()
    )
    assert chosen is not None and chosen.name == "exact"


def test_a_profile_you_wrote_beats_one_that_shipped(tmp_path: Path) -> None:
    theirs = _profile(
        tmp_path / "a", profiles.document(name="theirs", match={"total_ram_min": "1GiB"})
    )
    mine = _profile(tmp_path / "b", profiles.document(name="mine", match={"total_ram_min": "1GiB"}))
    chosen = best_match(
        [_loaded(theirs, bundled=True), _loaded(mine, bundled=False)],
        machine(ram_total=64 * GIB),
    )
    assert chosen is not None and chosen.name == "mine"


def test_the_bundled_reference_profile_recognises_the_reference_machine() -> None:
    """The point of a match block: this machine's own scan finds its own profile."""
    from llamafit.hwprofile import load_profiles

    loaded, problems = load_profiles()
    assert problems == []
    chosen = best_match(loaded, reference_host())
    assert chosen is not None and chosen.name == "reference-rtx4060-128gb"


def test_it_does_not_recognise_a_different_machine() -> None:
    from llamafit.hwprofile import load_profiles

    loaded, _problems = load_profiles()
    assert best_match(loaded, machine()) is None


def test_a_processor_rule_that_does_not_fit_fails_the_match(tmp_path: Path) -> None:
    profile = _profile(tmp_path, profiles.document(match={"cpu_model_contains": "Threadripper"}))
    assert matches(profile, reference_host()) is False
