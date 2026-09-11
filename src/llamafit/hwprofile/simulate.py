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

* ``gpu_memory`` resizes the primary card's VRAM. What the desktop was already using
  stays used, capped at the new size, because a bigger card does not empty itself.
* ``ram`` resizes the system pool and keeps *what is in use* rather than what is free:
  the same applications are still open on the larger machine, so the free figure moves by
  the whole difference.
* ``cpu_cores`` sets the physical core count and scales the thread count by the ratio the
  host had, so a machine with simultaneous multithreading keeps it and one without does
  not acquire it. Performance cores cannot outnumber the cores that remain.
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


def _overridden_gpu(host: Host, vram_total_bytes: int) -> list[Gpu]:
    """Every card, with the primary one resized to ``vram_total_bytes``.

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
                "vram_used_bytes": min(gpu.vram_used_bytes or 0, vram_total_bytes),
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
    """The system pool resized, keeping what was in use rather than what was free."""
    in_use = max(memory.total_bytes - memory.available_bytes, 0)
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
