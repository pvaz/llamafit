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


def _fmt_capabilities(capabilities: Sequence[str], *, width: int, limit: int = 3) -> str:
    """As many complete capability names (up to ``limit``) as fit in ``width``.

    Items are dropped whole, never cut mid-name: handing Rich a longer string and
    letting its ellipsis crop it would risk cropping a name in half, or cropping
    the ``+N`` marker itself away, leaving a reader with less information than a
    shorter, complete answer would have given them (``coding, tools +4`` beats
    ``coding, tools, long…`` even though it names fewer capabilities, because
    nothing in it is a guess). This starts from the full (up to ``limit``) list and
    drops one whole item at a time from the end, each time re-adding a marker for
    everything dropped so far, until the result fits; when even one name plus the
    marker will not fit, the marker alone is shown. Empty input is the only case
    with no marker and no items.
    """
    eligible = list(capabilities[:limit])
    total = len(capabilities)
    for shown_count in range(len(eligible), 0, -1):
        shown = eligible[:shown_count]
        dropped = total - shown_count
        text = ", ".join(shown)
        candidate = f"{text} +{dropped}" if dropped else text
        if len(candidate) <= width:
            return candidate
    return f"+{total}" if total else ""


_ID_COLUMN_MAX_WIDTH = 28
# What one more column costs beyond its own content: a border and the padding on
# each side of it. The table's own leading border is the one extra "+1" charged
# once, before any column, in the budget below.
_COLUMN_OVERHEAD = 3
_TABLE_OVERHEAD = 1
_QUALITY_CAPTION = (
    "Quality is the editorial baseline, before any quantisation penalty; "
    "run `llamafit info <model>` for the sourced benchmarks behind it."
)
_PARAMS_CAPTION = (
    "Params is total/active billions for a mixture-of-experts model, "
    "or one number when they are equal."
)


def _fmt_params(total_b: float, active_b: float) -> str:
    """One parameter count when dense, a total/active pair when they differ.

    Total and active parameters are equal by definition for a dense model, so a
    pair like ``27/27B`` reads like a typo rather than a fact; printing the single
    number it actually is says the same thing without inviting that doubt. A
    mixture-of-experts model, where the two genuinely differ, keeps the pair,
    which is the only case the difference is telling anyone something.
    """
    if total_b == active_b:
        return f"{_fmt_billions(total_b)}B"
    return f"{_fmt_billions(total_b)}/{_fmt_billions(active_b)}B"


def _table_caption(included: Sequence[str]) -> str | None:
    """The footnote explaining whichever of quality and params is shown, or none."""
    parts = [
        caption
        for name, caption in (("quality", _QUALITY_CAPTION), ("params", _PARAMS_CAPTION))
        if name in included
    ]
    return " ".join(parts) if parts else None


def _column_budget(
    summaries: Sequence[ModelSummary], console_width: int
) -> tuple[int, list[str], int]:
    """Decide the id column's width and which optional columns fit, in priority order.

    The identifier is never dropped, only capped and allowed to wrap onto a second
    line: a truncated id cannot be typed back into ``info`` or ``search``, so
    folding it is the only acceptable way to shrink it. Quality, params, context
    and capabilities are tried in that order, each added only while there is
    room for its whole content plus its own border and padding; the first one
    that does not fit, and everything after it, is dropped rather than shrunk,
    because a blank column or a number missing a digit is worse than one column
    fewer.

    Returns:
        The id column's width, the list of optional column names to include (a
        subset, in priority order, of ``["quality", "params", "context"]``), and
        the width left over for capabilities (which may be too small to use).
    """
    id_width = min(_ID_COLUMN_MAX_WIDTH, max((len(s.id) for s in summaries), default=len("ID")))
    # Each width starts from its header's own length, so the list is never empty
    # even when there are no rows, and a short header never lets a column shrink
    # smaller than its own name.
    quality_width = max([len("Quality*"), *(len(str(s.quality_baseline)) for s in summaries)])
    params_width = max(
        [
            len("Params"),
            *(len(_fmt_params(s.params_total_b, s.params_active_b)) for s in summaries),
        ]
    )
    context_width = max(
        [len("Context"), *(len(_fmt_context_compact(s.context_native)) for s in summaries)]
    )

    remaining = console_width - _TABLE_OVERHEAD - (id_width + _COLUMN_OVERHEAD)
    included: list[str] = []
    columns = (("quality", quality_width), ("params", params_width), ("context", context_width))
    for name, width in columns:
        cost = width + _COLUMN_OVERHEAD
        if cost > remaining:
            break
        included.append(name)
        remaining -= cost
    return id_width, included, remaining - _COLUMN_OVERHEAD


def render_catalog_list(summaries: Sequence[ModelSummary], *, console_width: int = 80) -> Table:
    """A table listing models: enough to tell them apart, not everything about them.

    Columns are id, quality, parameters (one number for a dense model, a
    total/active pair for a mixture-of-experts one, per :func:`_fmt_params`),
    native context and capabilities (as many complete names as fit, up to three,
    plus a ``+N`` marker for the rest; ``info`` or ``--json`` has every one): what
    separates one candidate from another at a glance, and, for quality, *why*
    they are ordered the way they are (``filter_models`` sorts by it, so a reader
    should not have to take the order on faith — see the caption for what that
    number, and the params pair, are and are not). Vendor, licence and quant
    count are left out entirely; an id already carries the family
    (``qwen3-coder-next``), so vendor is the cheapest of the three to drop.

    Fitting the rest to ``console_width`` is :func:`_column_budget`'s job, in a
    fixed priority: id first, then quality, params, context, and capabilities
    last, each included only whole. A column that cannot fit its content in
    full is dropped rather than shrunk: Rich's own width negotiation, left to
    itself across several ``no_wrap`` columns, can render one of them completely
    blank or as a single ellipsis once an unusually long id crowds the rest, and
    a blank column is a worse failure than a missing one. The id column alone is
    allowed to grow past one line (capped, folding onto a second) rather than
    ellipsize, because a truncated id cannot be typed back into ``info`` or
    ``search``, so nothing else is preserved by cutting it short. Every cell
    built from catalog text goes through ``Text``, not an f-string handed to
    ``console.print``, since a model name or id is never guaranteed free of
    characters Rich would try to parse as markup.

    Args:
        summaries: The rows to render, already filtered and sorted.
        console_width: The console's width; defaults to 80, the narrowest width
            this table is designed for.
    """
    id_width, included, capabilities_width = _column_budget(summaries, console_width)
    include_capabilities = capabilities_width >= 3  # room for at least a bare "+N" marker

    table = Table(
        title="Models",
        caption=_table_caption(included),
    )
    table.add_column("ID", style="bold", max_width=id_width, overflow="fold")
    if "quality" in included:
        table.add_column("Quality*", justify="right", no_wrap=True)
    if "params" in included:
        table.add_column("Params", justify="right", no_wrap=True)
    if "context" in included:
        table.add_column("Context", justify="right", no_wrap=True)
    if include_capabilities:
        table.add_column("Capabilities", no_wrap=True)

    for summary in summaries:
        row: list[Text | str] = [Text(summary.id)]
        if "quality" in included:
            row.append(str(summary.quality_baseline))
        if "params" in included:
            row.append(_fmt_params(summary.params_total_b, summary.params_active_b))
        if "context" in included:
            row.append(_fmt_context_compact(summary.context_native))
        if include_capabilities:
            row.append(Text(_fmt_capabilities(summary.capabilities, width=capabilities_width)))
        table.add_row(*row)
    return table


def render_model_facts(model: CatalogModel) -> Table:
    """A two-column table of one model's curated facts: everything but its quants.

    As with every other table here, catalog text (licence slug, architecture notes,
    a source's repository or path) is wrapped in ``Text`` rather than interpolated
    into a markup string.
    """
    table = Table(
        title=Text(f"{model.name} ({model.id})"), show_header=False, box=None, pad_edge=False
    )
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
    """A table of every quant with its size, bits per weight and architecture facts.

    There is deliberately no "downloaded" column: see :class:`QuantDetail` for why
    that field does not exist yet either. Every other cell here is honest about not
    knowing something (``unknown``, ``not read yet``); a column that can only ever
    say ``no`` would not be.
    """
    table = Table(title="Quants")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("BPW", justify="right")
    table.add_column("Facts")
    for quant in quants:
        table.add_row(
            Text(quant.name),
            format_bytes(quant.bytes_),
            f"{quant.bpw:.2f}" if quant.bpw is not None else "unknown",
            Text(_facts_summary(quant.facts)),
        )
    return table
