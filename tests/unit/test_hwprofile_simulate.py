"""Substituting a machine, and the mark that says one was substituted.

The mark is the point of these tests. A figure computed for a machine nobody is sitting
at is the weakest figure this project produces, so every path that makes one has to leave
``simulated`` true behind it, and the one path that does not make one -- a flag nobody
passed -- has to leave a scan alone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llamafit.errors import ConfigError
from llamafit.hwprofile.loader import LoadedProfile, load_profile_file
from llamafit.hwprofile.simulate import host_from_profile, override_host, resolve_host
from llamafit.models.host import Cpu, Gpu, Host, Memory
from tests.fixtures import profiles
from tests.fixtures.budget_hosts import GIB, machine

MIB = 1024**2


def _loaded(tmp_path: Path, document: dict[str, object]) -> LoadedProfile:
    """Write a document to a temporary directory and load it as a profile."""
    profiles.write(tmp_path, document)
    path = tmp_path / f"{document['name']}.json"
    profile, problems = load_profile_file(path)
    assert problems == [] and profile is not None
    return LoadedProfile(profile=profile, path=path, bundled=False)


def test_a_profile_becomes_a_host_that_says_it_is_one(tmp_path: Path) -> None:
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert isinstance(host, Host)
    assert host.simulated is True
    assert host.simulation is not None
    assert host.simulation.profile == "test-machine"
    assert host.simulation.overrides == []
    assert "test-machine.json" in (host.simulation.path or "")


def test_the_flag_is_in_the_serialised_host(tmp_path: Path) -> None:
    """A computed field, so no ``--json`` consumer can be handed a host without it."""
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert host.model_dump(mode="json")["simulated"] is True
    assert '"simulated": true' in host.model_dump_json(indent=2)


def test_a_scanned_host_is_not_simulated() -> None:
    assert machine().simulated is False
    assert machine().model_dump(mode="json")["simulated"] is False


def test_a_card_with_no_figures_gets_them_from_the_bundled_table(tmp_path: Path) -> None:
    host = host_from_profile(_loaded(tmp_path, profiles.with_gpu()))
    gpu = host.primary_gpu
    assert gpu is not None
    assert gpu.bandwidth_gbps == 1008 and gpu.compute_tflops_fp16 == 330
    assert gpu.backend_hint == "cuda"


def test_a_backend_is_taken_from_the_vendor_when_it_is_not_stated(tmp_path: Path) -> None:
    document = profiles.document(
        gpus=[{"vendor": "amd", "name": "Radeon RX 7900 XTX", "vram_total": "24GiB"}]
    )
    host = host_from_profile(_loaded(tmp_path, document))
    assert host.gpus[0].backend_hint == "hip"


def test_an_absent_available_figure_means_the_whole_pool(tmp_path: Path) -> None:
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert host.memory.available_bytes == host.memory.total_bytes


def test_an_absent_bandwidth_is_unknown_rather_than_assumed(tmp_path: Path) -> None:
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert host.memory.bandwidth_gbps is None
    assert host.memory.bandwidth_source == "unknown"


def test_recorded_at_becomes_the_scan_time(tmp_path: Path) -> None:
    document = profiles.document(recorded_at="2026-09-09T20:00:00Z")
    host = host_from_profile(_loaded(tmp_path, document))
    assert host.scanned_at == datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)


def test_without_recorded_at_the_host_is_stamped_now(tmp_path: Path) -> None:
    before = datetime.now(timezone.utc)
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert host.scanned_at >= before


def test_a_profile_host_has_no_probes_or_disks(tmp_path: Path) -> None:
    """Nothing probed it and nothing looked at its disks; both stay empty rather than faked."""
    host = host_from_profile(_loaded(tmp_path, profiles.MINIMAL))
    assert host.probes == [] and host.disks == []


def test_overriding_vram_marks_the_host_and_keeps_what_is_in_use() -> None:
    host = machine(vram_total=8 * GIB, vram_used=1 * GIB)
    bigger = override_host(host, gpu_memory=24 * GIB)
    assert bigger.primary_gpu is not None
    assert bigger.primary_gpu.vram_total_bytes == 24 * GIB
    assert bigger.primary_gpu.vram_used_bytes == 1 * GIB
    assert bigger.simulated is True
    assert bigger.simulation is not None and bigger.simulation.overrides == ["gpu_memory"]
    assert host.primary_gpu is not None
    assert host.primary_gpu.vram_total_bytes == 8 * GIB, "the original is untouched"


def test_a_smaller_card_is_a_smaller_card_and_not_a_full_one() -> None:
    """``--memory`` names a card, so a card a quarter the size is a quarter as busy.

    The rule this replaces carried the whole 4 GiB across and clamped it, which left a
    card with nothing free on it at all -- a card nothing could ever be placed on,
    returned as the answer to a question somebody asked in good faith.
    """
    host = machine(vram_total=24 * GIB, vram_used=4 * GIB)
    smaller = override_host(host, gpu_memory=6 * GIB)
    assert smaller.primary_gpu is not None
    assert smaller.primary_gpu.vram_used_bytes == 1 * GIB, "a sixth of the card, as before"
    assert smaller.primary_gpu.vram_free_bytes == 5 * GIB


def test_a_card_the_size_it_already_is_moves_nothing() -> None:
    host = machine(vram_total=8 * GIB, vram_used=3 * GIB)
    same = override_host(host, gpu_memory=8 * GIB)
    assert same.primary_gpu is not None
    assert same.primary_gpu.vram_used_bytes == 3 * GIB
    assert same.primary_gpu.vram_free_bytes == 5 * GIB


def test_a_card_whose_size_nobody_could_read_is_still_clamped() -> None:
    """No total is no share to scale, so the load is held to the new size and no further."""
    host = machine(vram_total=8 * GIB, vram_used=0)
    host.gpus[0] = host.gpus[0].model_copy(
        update={"vram_total_bytes": None, "vram_used_bytes": 4 * GIB}
    )
    smaller = override_host(host, gpu_memory=2 * GIB)
    assert smaller.primary_gpu is not None
    assert smaller.primary_gpu.vram_used_bytes == 2 * GIB


def test_only_the_primary_card_is_resized() -> None:
    host = machine(vram_total=8 * GIB, vram_used=0)
    host.gpus.append(
        Gpu(index=1, vendor="nvidia", name="Second", vram_total_bytes=4 * GIB, vram_used_bytes=0)
    )
    bigger = override_host(host, gpu_memory=24 * GIB)
    assert [g.vram_total_bytes for g in bigger.gpus] == [24 * GIB, 4 * GIB]


def test_overriding_memory_keeps_what_is_in_use_rather_than_what_is_free() -> None:
    host = machine(ram_total=64 * GIB, ram_available=48 * GIB)
    bigger = override_host(host, ram=128 * GIB)
    assert bigger.memory.total_bytes == 128 * GIB
    assert bigger.memory.available_bytes == 112 * GIB, "16 GiB was in use and still is"


def test_a_smaller_machine_has_memory_on_it() -> None:
    """The finding itself: ``--ram`` below what this machine is using left nothing free.

    56 GiB of a 64 GiB machine in use is seven eighths of it; a quarter-sized machine is
    seven eighths busy too, which is 14 GiB in use and 2 GiB free. What it is not is zero,
    which is what subtracting 56 from 16 and clamping produced, and which emptied every
    board on every machine smaller than this one.
    """
    host = machine(ram_total=64 * GIB, ram_available=8 * GIB)
    smaller = override_host(host, ram=16 * GIB)
    assert smaller.memory.total_bytes == 16 * GIB
    assert smaller.memory.available_bytes == 2 * GIB


def test_a_machine_the_size_it_already_is_moves_nothing() -> None:
    """The rule has to hold for a machine exactly this size, and holding means byte-exact."""
    host = machine(ram_total=64 * GIB, ram_available=8 * GIB)
    same = override_host(host, ram=64 * GIB)
    assert same.memory.total_bytes == 64 * GIB
    assert same.memory.available_bytes == 8 * GIB


def test_an_idle_machine_shrinks_to_an_idle_one() -> None:
    """Nothing in use scales to nothing in use, whatever the size asked for."""
    host = machine(ram_total=128 * GIB, ram_available=128 * GIB)
    smaller = override_host(host, ram=8 * GIB)
    assert smaller.memory.available_bytes == 8 * GIB


def test_overriding_cores_keeps_the_threads_per_core_ratio() -> None:
    host = machine()
    assert host.cpu.physical_cores == 8 and host.cpu.logical_cores == 16
    fewer = override_host(host, cpu_cores=4)
    assert fewer.cpu.physical_cores == 4 and fewer.cpu.logical_cores == 8
    assert fewer.cpu.performance_cores == 4, "cannot have more performance cores than cores"


def test_a_machine_without_multithreading_does_not_acquire_it() -> None:
    host = machine()
    host.cpu = Cpu(model="No SMT", physical_cores=8, logical_cores=8)
    more = override_host(host, cpu_cores=16)
    assert more.cpu.logical_cores == 16


def test_every_override_is_recorded_in_order() -> None:
    host = override_host(machine(), gpu_memory=24 * GIB, ram=128 * GIB, cpu_cores=4)
    assert host.simulation is not None
    assert host.simulation.overrides == ["gpu_memory", "ram", "cpu_cores"]
    assert host.simulation.profile is None


def test_overrides_on_a_profile_host_keep_the_profile_name(tmp_path: Path) -> None:
    host = override_host(
        host_from_profile(_loaded(tmp_path, profiles.with_gpu())),
        ram=8 * GIB,
    )
    assert host.simulation is not None
    assert host.simulation.profile == "test-machine"
    assert host.simulation.overrides == ["ram"]


def test_no_override_leaves_a_scan_a_scan() -> None:
    """A flag nobody passed must not turn a measured host into a simulated one."""
    host = machine()
    assert override_host(host) is host
    assert override_host(host).simulated is False


def test_overriding_vram_on_a_machine_with_no_card_is_refused() -> None:
    with pytest.raises(ConfigError) as info:
        override_host(machine(vram_total=None), gpu_memory=24 * GIB)
    assert "no graphics card" in info.value.render()
    assert "--profile" in info.value.render()


def test_overriding_vram_on_a_unified_machine_points_at_the_right_flag() -> None:
    host = machine(vram_total=8 * GIB, unified=True)
    with pytest.raises(ConfigError) as info:
        override_host(host, gpu_memory=24 * GIB)
    assert "--ram" in info.value.render()


@pytest.mark.parametrize("field", ["gpu_memory", "ram", "cpu_cores"])
def test_a_value_of_zero_or_less_is_refused(field: str) -> None:
    with pytest.raises(ConfigError, match="above zero"):
        override_host(machine(), **{field: 0})


def test_resolve_host_scans_when_no_profile_is_named() -> None:
    scanned = machine()
    assert resolve_host(scan_host=lambda: scanned) is scanned


def test_resolve_host_does_not_scan_when_a_profile_is_named(tmp_path: Path) -> None:
    """A run scoring against somebody else's machine has no business probing this one."""
    profiles.write(tmp_path, profiles.MINIMAL)

    def refuse_to_scan() -> Host:
        raise AssertionError("the scan must not run when --profile was given")

    host = resolve_host(
        scan_host=refuse_to_scan, profile=str(tmp_path / "test-machine.json"), ram=8 * GIB
    )
    assert host.simulation is not None
    assert host.simulation.profile == "test-machine"
    assert host.simulation.overrides == ["ram"]


def test_resolve_host_applies_overrides_to_the_scan() -> None:
    host = resolve_host(scan_host=machine, cpu_cores=2)
    assert host.cpu.physical_cores == 2 and host.simulated is True


def test_a_host_built_by_hand_needs_no_simulation_block() -> None:
    """The field is optional, so nothing that builds a Host today has to change."""
    host = Host(
        os="linux",
        os_version="1",
        arch="x86_64",
        cpu=Cpu(model="c", physical_cores=1, logical_cores=1),
        memory=Memory(total_bytes=GIB, available_bytes=GIB),
        scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    assert host.simulation is None and host.simulated is False
