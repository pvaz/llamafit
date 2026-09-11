# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Substituting a machine for the one LlamaFit is running on.

Two substitutions, and they answer different questions.

A **profile** replaces the machine outright: a fleet nobody is standing in front of, or
the reference machine this project's numbers were measured on. What comes out is a
``Host`` built field for field from the file, so every service downstream takes it exactly
as it takes a scan -- that is the point, and it is why nothing here returns a second type.

An **override** changes one pool of the machine that *is* there: what would a 24 GB card
run, what would half the memory cost me. It keeps everything the scan knows and moves one
figure, because a person asking "what if my card were bigger" is not asking to have their
processor and their memory bandwidth replaced by somebody's guesses too.

Both leave the same mark. :class:`~llamafit.models.host.Simulation` goes on the host, so
``host.simulated`` is ``True`` for the rest of the run wherever it travels -- into a
budget, into a board, into ``--json``. Nothing in this module can produce a substituted
host without it: the mark is set in the two constructors below and nowhere else.

The overrides' arithmetic is written down rather than guessed at each call site:

* ``gpu_memory`` resizes the primary card's VRAM and ``ram`` resizes the system pool.
  Both move what is *in use* by :func:`_carried_load`, which is the one rule below.
* ``cpu_cores`` sets the physical core count and scales the thread count by the ratio the
  host had, so a machine with simultaneous multithreading keeps it and one without does
  not acquire it. Performance cores cannot outnumber the cores that remain.

**A size names a machine.** ``--ram 16GiB`` describes a 16 GiB machine, and ``--memory
4GiB`` a 4 GiB card; neither asks for this machine's occupancy to be subtracted from the
figure typed. The flag's own help says *pretend the machine has this much system memory*,
and a reader who says 16 GiB means a 16 GiB machine, not their 128 GiB one with 112 GiB
taken off the top. :func:`_carried_load` is the whole of how that is honoured, and it is
one rule for both pools rather than one per flag, because a person who has understood
``--ram`` has already understood ``--memory``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from llamafit.errors import ConfigError
from llamafit.hardware.gputable import enrich_gpu
from llamafit.hwprofile.loader import LoadedProfile, resolve_profile
from llamafit.i18n import _
from llamafit.models.host import Cpu, Gpu, Host, Memory, Override, Simulation


def host_from_profile(loaded: LoadedProfile) -> Host:
    """Build the ``Host`` a profile describes.

    A card that states no bandwidth or compute figure is filled from the bundled GPU
    table by name, which is where those numbers come from on a real scan too: a profile
    author should not have to retype a vendor specification LlamaFit already ships, and a
    figure retyped is a figure that can drift from the one every other machine uses.

    ``scanned_at`` is the profile's ``recorded_at`` when it has one, so a profile taken
    from a real scan reproduces that scan rather than claiming to be current.

    Returns:
        A host marked as simulated, naming the profile and the file it came from.
    """
    profile = loaded.profile
    gpus = [
        enrich_gpu(
            Gpu(
                index=index,
                vendor=card.vendor,
                name=card.name,
                vram_total_bytes=card.vram_total,
                vram_used_bytes=card.vram_used,
                # A stated size is a size somebody read off a machine and wrote down, and
                # the profile's required ``provenance`` is where they say which machine.
                # A card the profile leaves unsized is left unsized here too, rather than
                # acquiring a provenance for a figure that does not exist.
                vram_source="measured" if card.vram_total is not None else "unknown",
                bandwidth_gbps=card.bandwidth_gbps,
                compute_tflops_fp16=card.compute_tflops_fp16,
                backend_hint=card.backend,
                driver=card.driver,
            )
        )
        for index, card in enumerate(profile.gpus)
    ]
    memory = Memory(
        total_bytes=profile.memory.total,
        available_bytes=(
            profile.memory.total if profile.memory.available is None else profile.memory.available
        ),
        type=profile.memory.type,
        speed_mts=profile.memory.speed_mts,
        modules=profile.memory.modules,
        channels=profile.memory.channels,
        bandwidth_gbps=profile.memory.bandwidth_gbps,
        bandwidth_source=profile.memory.bandwidth_source or "unknown",
    )
    cpu = Cpu(
        model=profile.cpu.model,
        physical_cores=profile.cpu.physical_cores,
        logical_cores=profile.cpu.logical_cores or profile.cpu.physical_cores,
        performance_cores=profile.cpu.performance_cores,
        isa=list(profile.cpu.isa),
    )
    return Host(
        os=profile.os,
        os_version=profile.os_version,
        arch=profile.arch,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        unified_memory=profile.unified_memory,
        disks=[],
        probes=[],
        scanned_at=profile.recorded_at or datetime.now(timezone.utc),
        simulation=Simulation(profile=profile.name, path=str(loaded.path)),
    )


def _carried_load(in_use: int, was_total: int, now_total: int) -> int:
    """How much of a resized pool the machine's own load takes, in bytes.

    Args:
        in_use: What was in use when the pool was its real size.
        was_total: The real size of the pool.
        now_total: The size the flag asked for.

    Returns:
        The same load on a pool that is the same size or larger, and the same *share* of
        one that is smaller.

    Three directions and one rule, which is the point.

    Growing, the load does not grow with the machine: the browser and the desktop do not
    swell to fill a bigger card, so a 24 GiB card asked about from an 8 GiB one with 700
    MiB in use has 700 MiB in use and 23.3 GiB free. This is the rule the module was
    written with and it was right about this direction.

    Shrinking, that same load cannot be carried over unchanged, and subtracting it is what
    made a 16 GiB machine asked about from a 128 GiB one come back with nothing free at
    all: 112 GiB of somebody's open applications charged against a machine that could
    never have held them, an empty board for a four-billion-parameter model that would
    have fitted on the card. So the *share* is carried instead -- a machine asked to be a
    quarter of the size is a quarter as busy -- which is the only reading of "pretend the
    machine has this much memory" that leaves a small machine usable and still says
    something true about how busy it is.

    Exactly the same size, in either arm, gives exactly the figure that was scanned: the
    division is integer and ``in_use * total // total`` is ``in_use``. A flag naming the
    size the machine already has must not move a single byte of the answer.
    """
    if was_total <= 0:
        return 0
    return min(in_use, in_use * now_total // was_total)


def _overridden_gpu(host: Host, vram_total_bytes: int) -> list[Gpu]:
    """Every card, with the primary one resized to ``vram_total_bytes``.

    What the desktop was using is carried across by :func:`_carried_load`, so a card
    asked to be smaller than the one in the machine comes back with room on it rather
    than filled to the brim by a compositor that never ran on it.

    Raises:
        ConfigError: There is no card to resize, or the machine has one memory pool and
            resizing "the card" would mean resizing the system memory behind its back.
    """
    primary = host.primary_gpu
    if primary is None:
        raise ConfigError(
            _("this machine has no graphics card, so there is no VRAM to override"),
            hint=_("Describe the whole machine in a hardware profile and pass --profile."),
        )
    if host.unified_memory:
        raise ConfigError(
            _("this machine has one memory pool shared with the graphics card"),
            hint=_("Use --ram to resize the pool; --memory would change only half of it."),
        )
    return [
        gpu.model_copy(
            update={
                "vram_total_bytes": vram_total_bytes,
                "vram_used_bytes": _carried_load(
                    gpu.vram_used_bytes or 0,
                    # A card whose size nobody could read has no share to scale, so the
                    # load is carried whole and clamped by the new size -- which is what
                    # the equal-size arm of the rule does anyway.
                    gpu.vram_total_bytes or (gpu.vram_used_bytes or 0),
                    vram_total_bytes,
                ),
                # Nothing read this card; a person asked for it to be this size. That is
                # what ``assumed`` means everywhere else a figure carries a label, and it
                # keeps ``--memory`` from dressing a what-if as a measurement.
                "vram_source": "assumed",
            }
        )
        if gpu is primary
        else gpu
        for gpu in host.gpus
    ]


def _overridden_memory(memory: Memory, total_bytes: int) -> Memory:
    """The system pool resized, carrying this machine's load across by :func:`_carried_load`."""
    in_use = _carried_load(
        max(memory.total_bytes - memory.available_bytes, 0), memory.total_bytes, total_bytes
    )
    return memory.model_copy(
        update={
            "total_bytes": total_bytes,
            "available_bytes": max(total_bytes - in_use, 0),
        }
    )


def _overridden_cpu(cpu: Cpu, physical_cores: int) -> Cpu:
    """The processor with a different core count, keeping its threads-per-core ratio."""
    per_core = cpu.logical_cores / cpu.physical_cores if cpu.physical_cores else 1.0
    logical = max(physical_cores, round(physical_cores * per_core))
    performance = (
        None if cpu.performance_cores is None else min(cpu.performance_cores, physical_cores)
    )
    return cpu.model_copy(
        update={
            "physical_cores": physical_cores,
            "logical_cores": logical,
            "performance_cores": performance,
        }
    )


def override_host(
    host: Host,
    *,
    gpu_memory: int | None = None,
    ram: int | None = None,
    cpu_cores: int | None = None,
) -> Host:
    """Substitute one or more of a host's pools for a what-if.

    Args:
        host: The machine to start from -- a live scan, or a host built from a profile.
        gpu_memory: VRAM the primary card should be pretended to have, in bytes.
        ram: System memory the machine should be pretended to have, in bytes.
        cpu_cores: Physical cores the processor should be pretended to have.

    Returns:
        The host with those figures replaced and
        :class:`~llamafit.models.host.Simulation` recording which ones. Given nothing to
        override, the host comes back untouched, still marked exactly as it arrived: a
        flag nobody passed must not turn a scan into a simulation.

    Raises:
        ConfigError: ``gpu_memory`` was asked for on a machine with no separate VRAM
            pool, or a figure was not positive.
    """
    for label, value in (("--memory", gpu_memory), ("--ram", ram), ("--cpu-cores", cpu_cores)):
        if value is not None and value <= 0:
            raise ConfigError(
                _("%(option)s needs a value above zero") % {"option": label},
            )

    overrides: list[Override] = []
    update: dict[str, object] = {}
    if gpu_memory is not None:
        update["gpus"] = _overridden_gpu(host, gpu_memory)
        overrides.append("gpu_memory")
    if ram is not None:
        update["memory"] = _overridden_memory(host.memory, ram)
        overrides.append("ram")
    if cpu_cores is not None:
        update["cpu"] = _overridden_cpu(host.cpu, cpu_cores)
        overrides.append("cpu_cores")
    if not overrides:
        return host

    simulation = host.simulation or Simulation()
    update["simulation"] = simulation.model_copy(
        update={"overrides": [*simulation.overrides, *overrides]}
    )
    return host.model_copy(update=update)


def resolve_host(
    *,
    scan_host: Callable[[], Host],
    profile: str | None = None,
    gpu_memory: int | None = None,
    ram: int | None = None,
    cpu_cores: int | None = None,
) -> Host:
    """The machine a command should work against, given the simulation options.

    This is what ``--profile``, ``--memory``, ``--ram`` and ``--cpu-cores`` call. The
    scan is a callable rather than a host so that ``--profile`` never probes: a run
    scoring against somebody else's machine has no business reading this one's, and on a
    machine whose graphics driver hangs a probe, the profile has to work anyway.

    Args:
        scan_host: Produces the live scan; called only when no profile is named.
        profile: A bundled or user profile's name, or a path to a profile file.
        gpu_memory: VRAM override in bytes.
        ram: System memory override in bytes.
        cpu_cores: Physical core count override.

    Returns:
        A ``Host``: the scan, a profile, or either with pools substituted. Marked as
        simulated whenever it is not simply the scan.

    Raises:
        ConfigError: The profile could not be found or read, or an override cannot
            apply to this machine.
    """
    host = host_from_profile(resolve_profile(profile)[0]) if profile else scan_host()
    return override_host(host, gpu_memory=gpu_memory, ram=ram, cpu_cores=cpu_cores)
