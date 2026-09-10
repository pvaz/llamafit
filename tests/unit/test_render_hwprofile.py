"""What a reader sees: the simulated banner, and the profile tables.

The banner is the part that matters. Question one of this feature is how somebody can tell
a simulated board from a measured one, and on a terminal the answer is the first row of
the host table -- so these tests assert it is there, before the figures, in every shape a
simulation can take.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from llamafit.cli.render import render_host, render_profile, render_profiles
from llamafit.hwprofile.loader import LoadedProfile, load_profile_file
from llamafit.hwprofile.simulate import host_from_profile, override_host
from tests.fixtures import profiles
from tests.fixtures.budget_hosts import GIB, machine


def _drawn(renderable: object) -> str:
    console = Console(width=120, no_color=True, highlight=False, record=True)
    console.print(renderable)  # type: ignore[arg-type]
    return " ".join(console.export_text().split())


def _loaded(tmp_path: Path, document: dict[str, object]) -> LoadedProfile:
    profiles.write(tmp_path, document)
    path = tmp_path / f"{document['name']}.json"
    profile, problems = load_profile_file(path)
    assert problems == [] and profile is not None
    return LoadedProfile(profile=profile, path=path, bundled=False)


def test_a_scanned_host_has_no_banner() -> None:
    assert "SIMULATED" not in _drawn(render_host(machine()))


def test_a_profile_host_says_which_profile_before_any_figure(tmp_path: Path) -> None:
    text = _drawn(render_host(host_from_profile(_loaded(tmp_path, profiles.MINIMAL))))
    assert text.index("SIMULATED") < text.index("Memory")
    assert "test-machine" in text
    assert "not this machine" in text


def test_an_overridden_scan_says_which_pool_was_replaced() -> None:
    text = _drawn(render_host(override_host(machine(), gpu_memory=24 * GIB)))
    assert "SIMULATED" in text
    assert "VRAM was overridden" in text
    assert "not what was scanned" in text


def test_several_overrides_agree_in_number() -> None:
    text = _drawn(render_host(override_host(machine(), ram=8 * GIB, cpu_cores=4)))
    assert "system memory, CPU cores were overridden" in text


def test_a_profile_with_overrides_names_both(tmp_path: Path) -> None:
    host = override_host(host_from_profile(_loaded(tmp_path, profiles.MINIMAL)), cpu_cores=2)
    text = _drawn(render_host(host))
    assert "test-machine" in text and "CPU cores" in text


def test_the_profile_list_names_the_machine(tmp_path: Path) -> None:
    text = _drawn(render_profiles([_loaded(tmp_path, profiles.with_gpu())]))
    assert "RTX 4090" in text and "24.0 GiB" in text


def test_the_profile_list_says_when_there_is_no_card(tmp_path: Path) -> None:
    assert "no graphics card" in _drawn(render_profiles([_loaded(tmp_path, profiles.MINIMAL)]))


def test_the_profile_list_says_when_the_pool_is_shared(tmp_path: Path) -> None:
    document = profiles.document(
        unified_memory=True, gpus=[{"vendor": "apple", "name": "Apple M3 Max"}]
    )
    assert "shared with" in _drawn(render_profiles([_loaded(tmp_path, document)]))


def test_one_profile_shows_its_provenance_and_its_match_rules(tmp_path: Path) -> None:
    document = profiles.with_gpu(match={"gpu_name_contains": "RTX 4090"})
    text = _drawn(render_profile(_loaded(tmp_path, document)))
    assert "Invented for a test" in text
    assert "name contains 'RTX 4090'" in text


def test_a_profile_with_no_rules_says_it_is_never_chosen(tmp_path: Path) -> None:
    text = _drawn(render_profile(_loaded(tmp_path, profiles.MINIMAL)))
    assert "never chosen for you" in text
    assert "none detected" in text, "and that it describes no graphics card"


def test_every_match_rule_is_spelled_out(tmp_path: Path) -> None:
    document = profiles.with_gpu(
        match={
            "gpu_name_contains": "RTX 4090",
            "cpu_model_contains": "Test CPU",
            "total_ram_min": "32GiB",
        }
    )
    text = _drawn(render_profile(_loaded(tmp_path, document)))
    assert "graphics card whose name contains" in text
    assert "processor whose model contains" in text
    assert "at least 32.0 GiB of memory" in text


def test_a_card_with_no_figures_says_the_table_supplied_them(tmp_path: Path) -> None:
    text = _drawn(render_profile(_loaded(tmp_path, profiles.with_gpu())))
    assert "from the specification table" in text


def test_a_card_with_its_own_figures_says_so(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[
            {
                "vendor": "nvidia",
                "name": "NVIDIA GeForce RTX 4090",
                "vram_total": "24GiB",
                "bandwidth_gbps": 1008,
                "compute_tflops_fp16": 83,
            }
        ]
    )
    text = _drawn(render_profile(_loaded(tmp_path, document)))
    assert "from the profile" in text


def test_a_card_the_table_does_not_know_says_it_has_no_figures(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[{"vendor": "other", "name": "Some Future Card", "vram_total": "16GiB"}]
    )
    text = _drawn(render_profile(_loaded(tmp_path, document)))
    assert "no bandwidth or compute figure" in text


def test_a_profile_with_no_bandwidth_says_unknown_rather_than_nothing(tmp_path: Path) -> None:
    assert "bandwidth unknown" in _drawn(render_profile(_loaded(tmp_path, profiles.MINIMAL)))


def test_a_profile_with_a_bandwidth_shows_its_label(tmp_path: Path) -> None:
    document = profiles.document(
        memory={"total": "64GiB", "bandwidth_gbps": 57.0, "bandwidth_source": "estimated"}
    )
    assert "57.0 GB/s (estimated)" in _drawn(render_profile(_loaded(tmp_path, document)))


def test_a_calibration_block_says_it_is_not_applied_yet(tmp_path: Path) -> None:
    document = profiles.document(
        calibration={"ram_efficiency": 0.7, "source": "docs/calibration/example.md"}
    )
    text = _drawn(render_profile(_loaded(tmp_path, document)))
    assert "does not apply it yet" in text


def test_the_bundled_profile_draws_every_row() -> None:
    from llamafit.hwprofile import load_profiles

    loaded, _problems = load_profiles()
    text = _drawn(render_profile(loaded[0]))
    assert "Recorded" in text and "Backends" in text and "cuda, cpu" in text
