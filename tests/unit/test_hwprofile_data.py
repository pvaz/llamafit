"""The bundled profile has to be the machine this project measured, not a sketch of it.

Every constant in ``llamafit/constants.py`` was fitted on the reference machine and every
``measured`` block in the catalog was taken on it. ``tests/fixtures/speed.py`` states that
machine for the estimator tests; the bundled profile states it for a stranger who does not
own it. Two independent statements of one machine are only worth having if they agree, and
this is where they are made to.
"""

from __future__ import annotations

from importlib import resources

from llamafit.hwprofile import bundled_profiles_dir, load_profiles, profile_files
from llamafit.hwprofile.simulate import host_from_profile
from llamafit.models.host import Host
from llamafit.speed import resolve_bandwidths
from tests.fixtures.speed import reference_host

REFERENCE = "reference-rtx4060-128gb"

_NOT_COMPARED = {
    # `platform.platform()` on the machine, against the shorter form the fixture carries.
    "os_version",
    # Only the profile-built host has one, which is the whole point of it.
    "simulation",
    "simulated",
}


def _reference_profile_host() -> Host:
    loaded, problems = load_profiles()
    assert problems == [], problems
    for profile in loaded:
        if profile.name == REFERENCE:
            return host_from_profile(profile)
    raise AssertionError(f"{REFERENCE} does not ship")


def test_the_bundled_profile_is_the_machine_the_estimator_was_fitted_to() -> None:
    from_profile = _reference_profile_host().model_dump(mode="json")
    from_fixture = reference_host().model_dump(mode="json")
    for field in _NOT_COMPARED:
        from_profile.pop(field, None)
        from_fixture.pop(field, None)
    assert from_profile == from_fixture


def test_the_speed_estimator_reads_the_same_bandwidths_off_both() -> None:
    """What the agreement is for: the estimates a stranger gets are the documented ones."""
    from_profile = resolve_bandwidths(_reference_profile_host())
    assert from_profile == resolve_bandwidths(reference_host())
    assert from_profile.assumed is False, "nothing about this machine had to be guessed"


def test_the_bundled_profile_is_marked_simulated_all_the_same() -> None:
    """Measured figures, and still not the machine the reader is sitting at."""
    host = _reference_profile_host()
    assert host.simulated is True
    assert host.memory.bandwidth_source == "measured"


def test_the_card_specifications_come_from_the_bundled_table_not_the_file() -> None:
    text = (
        resources.files("llamafit.data.profiles")
        .joinpath(f"{REFERENCE}.json")
        .read_text(encoding="utf-8")
    )
    assert "bandwidth_gbps" not in text.split('"gpus"', 1)[1]
    host = _reference_profile_host()
    gpu = host.primary_gpu
    assert gpu is not None
    assert gpu.bandwidth_gbps == 272.0 and gpu.compute_tflops_fp16 == 15.0


def test_every_bundled_profile_says_where_its_figures_came_from() -> None:
    """The one field neither the schema nor the loader can check the truth of."""
    loaded, _problems = load_profiles()
    assert loaded, "at least the reference machine ships"
    for profile in loaded:
        provenance = profile.profile.provenance
        assert len(provenance) > 80, f"{profile.name} says too little about its figures"


def test_only_measured_machines_ship() -> None:
    """Bundling an aspirational profile would ship invented data that reads as measured."""
    assert [path.stem for path in profile_files(bundled_profiles_dir())] == [REFERENCE]
