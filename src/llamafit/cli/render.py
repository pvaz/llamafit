"""Rich tables for the host, llama.cpp status, probes and findings."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table
from rich.text import Text

from llamafit.models.host import Host, Probe
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.doctor import Finding
from llamafit.units import format_bytes

_LEVEL_STYLE = {"ok": "green", "warn": "yellow", "error": "red"}


def render_host(host: Host) -> Table:
    """A two-column table with everything the scan found.

    Values are built as ``Text`` rather than markup strings: a device name or a path
    may contain square brackets, which Rich would otherwise try to parse as a tag.
    """
    table = Table(title="Host", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("OS", Text(f"{host.os} {host.os_version} ({host.arch})"))
    cores = f"{host.cpu.physical_cores} cores / {host.cpu.logical_cores} threads"
    if host.cpu.performance_cores:
        cores += f", {host.cpu.performance_cores} performance cores"
    table.add_row(
        "CPU", Text(f"{host.cpu.model}; {cores}; {' '.join(host.cpu.isa) or 'isa unknown'}")
    )
    mem = host.memory
    details = ", ".join(
        x
        for x in (
            " ".join(
                y for y in (mem.type, f"{mem.speed_mts} MT/s" if mem.speed_mts else None) if y
            ),
            f"{mem.modules} module{'' if mem.modules == 1 else 's'}" if mem.modules else None,
            f"{mem.channels} channel{'' if mem.channels == 1 else 's'}" if mem.channels else None,
        )
        if x
    )
    bandwidth = (
        f"{mem.bandwidth_gbps} GB/s ({mem.bandwidth_source})" if mem.bandwidth_gbps else "unknown"
    )
    table.add_row(
        "Memory",
        Text(
            f"{format_bytes(mem.total_bytes)} total, "
            f"{format_bytes(mem.available_bytes)} available; "
            f"{details or 'type unknown'}; bandwidth {bandwidth}"
        ),
    )
    if not host.gpus:
        table.add_row("GPU", "none detected")
    for gpu in host.gpus:
        vram = (
            f"{format_bytes(gpu.vram_total_bytes)} VRAM, {format_bytes(gpu.vram_free_bytes)} free"
            if gpu.vram_total_bytes
            else "VRAM unknown"
        )
        # Bandwidth and compute come from the bundled specification table, not from this
        # machine, so they are labelled: no number reaches the user without its source.
        from_table = " and ".join(
            x
            for x in (
                f"{gpu.bandwidth_gbps} GB/s" if gpu.bandwidth_gbps else None,
                f"{gpu.compute_tflops_fp16} TFLOPS fp16" if gpu.compute_tflops_fp16 else None,
            )
            if x
        )
        specs = ", ".join(
            x
            for x in (
                f"{from_table} (spec)" if from_table else None,
                f"driver {gpu.driver}" if gpu.driver else None,
            )
            if x
        )
        table.add_row(
            f"GPU {gpu.index}",
            Text(f"{gpu.name} ({gpu.backend_hint}); {vram}; {specs or 'no specs'}"),
        )
    if host.unified_memory:
        table.add_row("Memory pool", "unified (GPU shares system memory)")
    for disk in host.disks:
        table.add_row(
            "Disk",
            Text(
                f"{disk.path}: {format_bytes(disk.free_bytes)} free of "
                f"{format_bytes(disk.total_bytes)}"
            ),
        )
    return table


def render_llamacpp(llamacpp: LlamaCpp) -> Table:
    """A two-column table describing the installation, with paths never read as markup."""
    table = Table(title="llama.cpp", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    if not llamacpp.installed:
        table.add_row("Installed", "no")
    else:
        build = f"b{llamacpp.build}" if llamacpp.build else "unknown build"
        commit = f" ({llamacpp.commit})" if llamacpp.commit else ""
        table.add_row("Installed", Text(f"yes, {build}{commit} at {llamacpp.path}"))
        table.add_row("Backends", Text(", ".join(llamacpp.backends) or "none detected"))
        table.add_row("Local models", str(len(llamacpp.local_models)))
    for server in llamacpp.running_servers:
        table.add_row(
            "Running",
            Text(
                f"{server.url}: {server.model or 'unknown model'}, "
                f"context {server.n_ctx or 'unknown'}"
            ),
        )
    for problem in llamacpp.problems:
        table.add_row("Problem", Text(problem))
    return table


def render_probes(probes: Iterable[Probe]) -> Table:
    """Probe-by-probe outcome; the error text is never parsed as markup.

    A ``server:<port>`` probe that found nothing is the ordinary case on a machine with no
    llama-server running, so it is shown dim rather than as a failure.
    """
    table = Table(title="Probes", box=None, pad_edge=False)
    table.add_column("probe", style="bold")
    table.add_column("result")
    table.add_column("ms", justify="right")
    for probe in probes:
        if probe.ok:
            status = Text("ok", style="green")
        elif probe.name.startswith("server:"):
            status = Text("no server", style="dim")
        else:
            status = Text("failed", style="yellow")
            if probe.error:
                status.append(f" {probe.error}")
        table.add_row(probe.name, status, str(probe.duration_ms))
    return table


def render_findings(findings: Iterable[Finding]) -> Table:
    """Findings with level colouring and hints; the finding text is never parsed as markup."""
    table = Table(title="Findings", box=None, pad_edge=False)
    table.add_column("level")
    table.add_column("finding")
    for finding in findings:
        text = Text.assemble((finding.title, "bold"), "\n", finding.detail)
        if finding.hint:
            text.append("\n")
            text.append(f"Hint: {finding.hint}", style="dim")
        table.add_row(Text(finding.level.upper(), style=_LEVEL_STYLE[finding.level]), text)
    return table
