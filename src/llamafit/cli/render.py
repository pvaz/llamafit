"""Rich tables for the host, llama.cpp status, probes, findings and the catalog."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rich.table import Table
from rich.text import Text

from llamafit.models.catalog import CatalogModel
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Host, Probe
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.catalog import ModelSummary, QuantDetail
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


def _fmt_billions(value: float) -> str:
    """Format a parameter count in billions, dropping a trailing ``.0``."""
    return f"{value:g}"


def _fmt_context_compact(tokens: int) -> str:
    """Format a context length compactly, for example ``256K`` for ``262144``.

    Falls back to the plain number when it is not a clean multiple of a mebi- or
    kibi-token, which every context length in the catalog so far is.
    """
    if tokens % (1024 * 1024) == 0:
        return f"{tokens // (1024 * 1024)}M"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return str(tokens)


def _fmt_capabilities(capabilities: Sequence[str], *, limit: int = 3) -> str:
    """The first ``limit`` capabilities, joined, with a ``+N`` marker for the rest.

    A model with many capabilities would otherwise wrap its row across several
    lines, which makes five models harder to compare, not easier: the eye can scan
    down a column of one-line rows but not across a staircase of wrapped ones. The
    full list is one ``info`` or ``--json`` away.
    """
    shown = list(capabilities[:limit])
    text = ", ".join(shown)
    remaining = len(capabilities) - len(shown)
    return f"{text} +{remaining}" if remaining > 0 else text


def render_catalog_list(summaries: Sequence[ModelSummary], *, console_width: int = 80) -> Table:
    """A table listing models: enough to tell them apart, not everything about them.

    Columns are id, parameters, native context, quality and capabilities (the first
    three, with a ``+N`` marker for the rest; ``info`` or ``--json`` has every one):
    what separates one candidate from another at a glance, and, for quality, *why*
    they are ordered the way they are (``filter_models`` sorts by it, so a reader
    should not have to take the order on faith). Vendor, licence and quant count are
    left out to keep every row on one line at 80 columns; an id already carries the
    family (``qwen3-coder-next``), so vendor is the one of the three that costs
    least to drop.

    Every column is ``no_wrap``: an id is one hyphenated word with no space to wrap
    on, and a wrapped row anywhere turns a table meant to be scanned down a column
    back into the staircase this whole layout exists to avoid. The other four
    columns are content-sized and fully rigid (id, a number or a short code), so
    capabilities is deliberately the one elastic column: its ``max_width`` is
    computed from ``console_width`` minus what those four need, so a wide terminal
    shows every one of the first three capabilities in full and only a genuinely
    80-column terminal falls back to Rich's own ellipsis mid-list. That trade keeps
    every row one line, which a fuller but wrapped cell would not. Every cell built
    from catalog text goes through ``Text``, not an f-string handed to
    ``console.print``, since a model name or id is never guaranteed free of
    characters Rich would try to parse as markup.

    Args:
        summaries: The rows to render, already filtered and sorted.
        console_width: The console's width, used only to size the capabilities
            column; defaults to 80, the narrowest width this table is designed for.
    """
    # 59 is what the other four columns need for the catalog as it stands today
    # (an id up to "llama-3.1-8b-instruct" long, plus three short numeric columns)
    # and every border and padding character around all five columns; it will
    # drift a little as ids grow, which only ever costs capabilities a character
    # or two of headroom, never a wrapped row.
    capabilities_width = max(15, console_width - 59)
    table = Table(title="Models")
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Params", justify="right", no_wrap=True)
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Quality", justify="right", no_wrap=True)
    table.add_column("Capabilities", no_wrap=True, max_width=capabilities_width)
    for summary in summaries:
        total = _fmt_billions(summary.params_total_b)
        active = _fmt_billions(summary.params_active_b)
        table.add_row(
            Text(summary.id),
            f"{total}/{active}B",
            _fmt_context_compact(summary.context_native),
            str(summary.quality_baseline),
            Text(_fmt_capabilities(summary.capabilities)),
        )
    return table


def render_model_facts(model: CatalogModel) -> Table:
    """A two-column table of one model's curated facts: everything but its quants.

    As with every other table here, catalog text (licence slug, architecture notes,
    a source's repository or path) is wrapped in ``Text`` rather than interpolated
    into a markup string.
    """
    table = Table(title=f"{model.name} ({model.id})", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("Vendor", Text(model.vendor))
    table.add_row("Family", Text(model.family))
    table.add_row("Release date", str(model.release_date))
    table.add_row("Licence", Text(f"{model.license.spdx} ({model.license.url})"))
    table.add_row(
        "Parameters",
        Text(
            f"{_fmt_billions(model.params.total_b)}B total, "
            f"{_fmt_billions(model.params.active_b)}B active"
        ),
    )
    context = f"{model.context.native:,} tokens native"
    if model.context.extended:
        method = model.context.extended_method or "unspecified method"
        context += f", {model.context.extended:,} extended via {method}"
    table.add_row("Context", Text(context))
    table.add_row(
        "Architecture",
        Text(f"{model.architecture.class_}, gguf_arch={model.architecture.gguf_arch}"),
    )
    if model.architecture.notes:
        table.add_row("Notes", Text(model.architecture.notes))
    table.add_row("Capabilities", Text(", ".join(model.capabilities)))
    table.add_row("Use cases", Text(", ".join(model.use_cases)))
    table.add_row("Quality baseline", str(model.quality.baseline))
    for benchmark in model.quality.benchmarks:
        table.add_row(
            "Benchmark", Text(f"{benchmark.name}: {benchmark.score} ({benchmark.source})")
        )
    for index, source in enumerate(model.sources):
        location = source.repo if source.kind == "gguf" else source.path
        table.add_row(
            f"Source {index + 1}",
            Text(f"{location} ({source.kind}, trust={source.trust})"),
        )
    return table


def _facts_summary(facts: GgufFacts | None) -> str:
    """A one-line summary of a quant's architecture facts, or a note that none are known yet."""
    if facts is None:
        return "not read yet"
    parts = [facts.arch]
    if facts.n_layer is not None:
        parts.append(f"{facts.n_layer} layers")
    if facts.n_expert:
        used = f"/{facts.n_expert_used}" if facts.n_expert_used is not None else ""
        parts.append(f"{facts.n_expert}{used} experts")
    return ", ".join(parts)


def render_quants(quants: Sequence[QuantDetail]) -> Table:
    """A table of every quant with its size, bits per weight and architecture facts."""
    table = Table(title="Quants")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("BPW", justify="right")
    table.add_column("Downloaded")
    table.add_column("Facts")
    for quant in quants:
        table.add_row(
            Text(quant.name),
            format_bytes(quant.bytes_),
            f"{quant.bpw:.2f}" if quant.bpw is not None else "unknown",
            "yes" if quant.downloaded else "no",
            Text(_facts_summary(quant.facts)),
        )
    return table
