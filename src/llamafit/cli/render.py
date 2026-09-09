"""Rich tables for the host, llama.cpp status, probes, findings and the catalog.

Every sentence here is built at render time, so the eager translation functions are the
right ones: by the time a table is drawn the language has been chosen. Nothing on this
page is a module-level constant holding a message, which is the shape that would need the
deferred pair.

Two habits run through the file. A counted noun goes through :func:`ngettext` rather than
an ``s`` bolted onto a word, because a suffix is an English rule that most languages do
not share. A word short enough for two rows to share — *unknown*, *none detected*, *ok* —
carries a :func:`pgettext` context naming its row, because the Portuguese for it is
inflected and one translation cannot be right in both places.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rich.cells import cell_len
from rich.table import Table
from rich.text import Text

from llamafit.i18n import _, ngettext, pgettext
from llamafit.models.catalog import CatalogModel
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Host, Probe, Source
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.catalog import ModelSummary, QuantDetail
from llamafit.services.doctor import Finding
from llamafit.units import billions_suffix, format_bytes, format_grouped, localise_number

_LEVEL_STYLE = {"ok": "green", "warn": "yellow", "error": "red"}


def _level_label(level: str) -> str:
    """The word in the level column, which is a severity a reader scans, not an identifier."""
    labels = {
        "ok": pgettext("finding level", "OK"),
        "warn": pgettext("finding level", "WARN"),
        "error": pgettext("finding level", "ERROR"),
    }
    return labels.get(level, level.upper())


def _bandwidth_source_label(source: Source) -> str:
    """How a bandwidth figure was arrived at, in the reader's language.

    The label is half of what the figure says: an assumed number and a measured one are
    not the same claim, and this project shows neither without saying which it is.
    """
    labels = {
        "measured": pgettext("bandwidth source", "measured"),
        "estimated": pgettext("bandwidth source", "estimated"),
        "assumed": pgettext("bandwidth source", "assumed"),
        "unknown": pgettext("bandwidth source", "unknown"),
    }
    return labels.get(source, source)


def _from_table(bandwidth_gbps: float | None, compute_tflops_fp16: float | None) -> str:
    """The specification-table figures a GPU has, as one phrase in the reader's language.

    The three shapes are written out rather than joined from pieces: a translator cannot
    move a word across a join they never see, and ``and`` is a word, not punctuation.
    """
    if bandwidth_gbps and compute_tflops_fp16:
        return _("%(bandwidth)s GB/s and %(compute)s TFLOPS fp16") % {
            "bandwidth": localise_number(str(bandwidth_gbps)),
            "compute": localise_number(str(compute_tflops_fp16)),
        }
    if bandwidth_gbps:
        return _("%(bandwidth)s GB/s") % {"bandwidth": localise_number(str(bandwidth_gbps))}
    return _("%(compute)s TFLOPS fp16") % {"compute": localise_number(str(compute_tflops_fp16))}


def render_host(host: Host) -> Table:
    """A two-column table with everything the scan found.

    Values are built as ``Text`` rather than markup strings: a device name or a path
    may contain square brackets, which Rich would otherwise try to parse as a tag.
    """
    table = Table(title=_("Host"), show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row(
        _("OS"),
        Text(
            _("%(os)s %(version)s (%(arch)s)")
            % {"os": host.os, "version": host.os_version, "arch": host.arch}
        ),
    )
    cores = _("%(physical)d cores / %(logical)d threads") % {
        "physical": host.cpu.physical_cores,
        "logical": host.cpu.logical_cores,
    }
    if host.cpu.performance_cores:
        cores += ", " + ngettext(
            "%(count)d performance core",
            "%(count)d performance cores",
            host.cpu.performance_cores,
        ) % {"count": host.cpu.performance_cores}
    table.add_row(
        _("CPU"),
        Text(
            _("%(model)s; %(cores)s; %(isa)s")
            % {
                "model": host.cpu.model,
                "cores": cores,
                "isa": " ".join(host.cpu.isa) or _("isa unknown"),
            }
        ),
    )
    mem = host.memory
    details = ", ".join(
        x
        for x in (
            " ".join(
                y
                for y in (
                    mem.type,
                    _("%(speed)d MT/s") % {"speed": mem.speed_mts} if mem.speed_mts else None,
                )
                if y
            ),
            ngettext("%(count)d module", "%(count)d modules", mem.modules) % {"count": mem.modules}
            if mem.modules
            else None,
            ngettext("%(count)d channel", "%(count)d channels", mem.channels)
            % {"count": mem.channels}
            if mem.channels
            else None,
        )
        if x
    )
    bandwidth = (
        _("%(gbps)s GB/s (%(source)s)")
        % {
            "gbps": localise_number(str(mem.bandwidth_gbps)),
            "source": _bandwidth_source_label(mem.bandwidth_source),
        }
        if mem.bandwidth_gbps
        else pgettext("memory bandwidth", "unknown")
    )
    table.add_row(
        _("Memory"),
        Text(
            _("%(total)s total, %(available)s available; %(details)s; bandwidth %(bandwidth)s")
            % {
                "total": format_bytes(mem.total_bytes),
                "available": format_bytes(mem.available_bytes),
                "details": details or _("type unknown"),
                "bandwidth": bandwidth,
            }
        ),
    )
    if not host.gpus:
        table.add_row(_("GPU"), pgettext("GPU", "none detected"))
    for gpu in host.gpus:
        vram = (
            _("%(total)s VRAM, %(free)s free")
            % {
                "total": format_bytes(gpu.vram_total_bytes),
                "free": format_bytes(gpu.vram_free_bytes),
            }
            if gpu.vram_total_bytes
            else _("VRAM unknown")
        )
        # Bandwidth and compute come from the bundled specification table, not from this
        # machine, so they are labelled: no number reaches the user without its source.
        specs = ", ".join(
            x
            for x in (
                _("%(specs)s (spec)")
                % {"specs": _from_table(gpu.bandwidth_gbps, gpu.compute_tflops_fp16)}
                if gpu.bandwidth_gbps or gpu.compute_tflops_fp16
                else None,
                _("driver %(driver)s") % {"driver": gpu.driver} if gpu.driver else None,
            )
            if x
        )
        table.add_row(
            _("GPU %(index)d") % {"index": gpu.index},
            Text(
                _("%(name)s (%(backend)s); %(vram)s; %(specs)s")
                % {
                    "name": gpu.name,
                    "backend": gpu.backend_hint,
                    "vram": vram,
                    "specs": specs or _("no specs"),
                }
            ),
        )
    if host.unified_memory:
        table.add_row(_("Memory pool"), _("unified (GPU shares system memory)"))
    for disk in host.disks:
        table.add_row(
            _("Disk"),
            Text(
                _("%(path)s: %(free)s free of %(total)s")
                % {
                    "path": disk.path,
                    "free": format_bytes(disk.free_bytes),
                    "total": format_bytes(disk.total_bytes),
                }
            ),
        )
    return table


def render_llamacpp(llamacpp: LlamaCpp) -> Table:
    """A two-column table describing the installation, with paths never read as markup."""
    table = Table(title="llama.cpp", show_header=False, box=None, pad_edge=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    if not llamacpp.installed:
        table.add_row(_("Installed"), pgettext("llama.cpp is installed", "no"))
    else:
        if llamacpp.build and llamacpp.commit:
            build = _("b%(build)d (%(commit)s)") % {
                "build": llamacpp.build,
                "commit": llamacpp.commit,
            }
        elif llamacpp.build:
            build = _("b%(build)d") % {"build": llamacpp.build}
        else:
            build = _("unknown build")
        table.add_row(
            _("Installed"),
            Text(_("yes, %(build)s at %(path)s") % {"build": build, "path": llamacpp.path}),
        )
        table.add_row(
            _("Backends"),
            Text(", ".join(llamacpp.backends) or pgettext("backends", "none detected")),
        )
        table.add_row(_("Local models"), str(len(llamacpp.local_models)))
    for server in llamacpp.running_servers:
        table.add_row(
            _("Running"),
            Text(
                _("%(url)s: %(model)s, context %(context)s")
                % {
                    "url": server.url,
                    "model": server.model or _("unknown model"),
                    "context": server.n_ctx or pgettext("context length", "unknown"),
                }
            ),
        )
    for problem in llamacpp.problems:
        table.add_row(_("Problem"), Text(problem))
    return table


def render_probes(probes: Iterable[Probe]) -> Table:
    """Probe-by-probe outcome; the error text is never parsed as markup.

    A ``server:<port>`` probe that found nothing is the ordinary case on a machine with no
    llama-server running, so it is shown dim rather than as a failure.
    """
    table = Table(title=_("Probes"), box=None, pad_edge=False)
    table.add_column(_("probe"), style="bold")
    table.add_column(_("result"))
    table.add_column(_("ms"), justify="right")
    for probe in probes:
        if probe.ok:
            status = Text(pgettext("probe result", "ok"), style="green")
        elif probe.name.startswith("server:"):
            status = Text(_("no server"), style="dim")
        else:
            status = Text(pgettext("probe result", "failed"), style="yellow")
            if probe.error:
                status.append(f" {probe.error}")
        table.add_row(probe.name, status, str(probe.duration_ms))
    return table


def render_findings(findings: Iterable[Finding]) -> Table:
    """Findings with level colouring and hints; the finding text is never parsed as markup."""
    table = Table(title=_("Findings"), box=None, pad_edge=False)
    table.add_column(_("level"))
    table.add_column(_("finding"))
    for finding in findings:
        text = Text.assemble((finding.title, "bold"), "\n", finding.detail)
        if finding.hint:
            text.append("\n")
            text.append(_("Hint: %(hint)s") % {"hint": finding.hint}, style="dim")
        table.add_row(Text(_level_label(finding.level), style=_LEVEL_STYLE[finding.level]), text)
    return table


def _fmt_billions(value: float) -> str:
    """Format a parameter count in billions, dropping a trailing ``.0``."""
    return localise_number(f"{value:g}")


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

    A capability name is a catalog identifier, the same word ``--capability`` is typed
    with, so it is never translated.

    ``width`` is a count of terminal cells, so the fit is measured with ``cell_len`` and
    not with ``len``: a Japanese character is one of those and two of these.
    """
    eligible = list(capabilities[:limit])
    total = len(capabilities)
    for shown_count in range(len(eligible), 0, -1):
        shown = eligible[:shown_count]
        dropped = total - shown_count
        text = ", ".join(shown)
        candidate = f"{text} +{dropped}" if dropped else text
        if cell_len(candidate) <= width:
            return candidate
    return f"+{total}" if total else ""


_ID_COLUMN_MAX_WIDTH = 28
# What one more column costs beyond its own content: a border and the padding on
# each side of it. The table's own leading border is the one extra "+1" charged
# once, before any column, in the budget below.
_COLUMN_OVERHEAD = 3
_TABLE_OVERHEAD = 1


def _list_headings() -> dict[str, str]:
    """The list table's column headings, translated once and read twice.

    :func:`_column_budget` measures these and :func:`render_catalog_list` prints them,
    and the two have to agree: a heading measured in English and printed in Portuguese
    would give a column a width its own title does not fit into.
    """
    return {
        "id": _("ID"),
        "quality": _("Quality*"),
        "params": _("Params"),
        "context": _("Context"),
        "capabilities": _("Capabilities"),
    }


def _table_caption(included: Sequence[str]) -> str | None:
    """The footnote explaining whichever of quality and params is shown, or none."""
    captions = {
        "quality": _(
            "Quality is the editorial baseline, before any quantisation penalty; "
            "run `llamafit info <model>` for the sourced benchmarks behind it."
        ),
        "params": _(
            "Params is total/active billions for a mixture-of-experts model, "
            "or one number when they are equal."
        ),
    }
    parts = [caption for name, caption in captions.items() if name in included]
    return " ".join(parts) if parts else None


def _fmt_params(total_b: float, active_b: float) -> str:
    """One parameter count when dense, a total/active pair when they differ.

    Total and active parameters are equal by definition for a dense model, so a
    pair like ``27/27B`` reads like a typo rather than a fact; printing the single
    number it actually is says the same thing without inviting that doubt. A
    mixture-of-experts model, where the two genuinely differ, keeps the pair,
    which is the only case the difference is telling anyone something.
    """
    if total_b == active_b:
        return f"{_fmt_billions(total_b)}{billions_suffix()}"
    return f"{_fmt_billions(total_b)}/{_fmt_billions(active_b)}{billions_suffix()}"


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

    Every width here is a count of terminal cells, measured with ``cell_len`` rather
    than with ``len``: a Japanese heading is two characters and four columns wide, and a
    Devanagari one counts its combining marks as characters that occupy no column at all.
    A budget that measures the wrong thing is not a budget, and the whole point of this
    function is that a column is admitted only when its content fits whole.

    Returns:
        The id column's width, the list of optional column names to include (a
        subset, in priority order, of ``["quality", "params", "context"]``), and
        the width left over for capabilities (which may be too small to use).
    """
    headings = _list_headings()
    id_width = min(
        _ID_COLUMN_MAX_WIDTH,
        max((cell_len(s.id) for s in summaries), default=cell_len(headings["id"])),
    )
    # Each width starts from its heading's own width, so the list is never empty
    # even when there are no rows, and a short heading never lets a column shrink
    # smaller than its own name.
    quality_width = max(
        [cell_len(headings["quality"]), *(cell_len(str(s.quality_baseline)) for s in summaries)]
    )
    params_width = max(
        [
            cell_len(headings["params"]),
            *(cell_len(_fmt_params(s.params_total_b, s.params_active_b)) for s in summaries),
        ]
    )
    context_width = max(
        [
            cell_len(headings["context"]),
            *(cell_len(_fmt_context_compact(s.context_native)) for s in summaries),
        ]
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
    headings = _list_headings()

    table = Table(
        title=_("Models"),
        caption=_table_caption(included),
    )
    table.add_column(headings["id"], style="bold", max_width=id_width, overflow="fold")
    if "quality" in included:
        table.add_column(headings["quality"], justify="right", no_wrap=True)
    if "params" in included:
        table.add_column(headings["params"], justify="right", no_wrap=True)
    if "context" in included:
        table.add_column(headings["context"], justify="right", no_wrap=True)
    if include_capabilities:
        table.add_column(headings["capabilities"], no_wrap=True)

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
    into a markup string. That text is data: it is shown as it was written and never
    translated, and the row labels around it are what a reader's language reaches.
    """
    table = Table(
        title=Text(f"{model.name} ({model.id})"), show_header=False, box=None, pad_edge=False
    )
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row(_("Vendor"), Text(model.vendor))
    table.add_row(_("Family"), Text(model.family))
    table.add_row(_("Release date"), str(model.release_date))
    table.add_row(
        _("Licence"),
        Text(_("%(spdx)s (%(url)s)") % {"spdx": model.license.spdx, "url": model.license.url}),
    )
    table.add_row(
        _("Parameters"),
        Text(
            _("%(total)sB total, %(active)sB active")
            % {
                "total": _fmt_billions(model.params.total_b),
                "active": _fmt_billions(model.params.active_b),
            }
        ),
    )
    context = _("%(tokens)s tokens native") % {"tokens": format_grouped(model.context.native)}
    if model.context.extended:
        # This second message is a fragment: it opens with a comma and continues the one
        # above rather than standing on its own, which is exactly what a translator cannot
        # work with. It should be two whole alternative sentences. It is deliberately left
        # alone for now: catalogs are being translated against the committed template, and
        # changing a message id orphans that work silently, with a green build. Fix it once
        # they have landed, not helpfully in passing.
        context += _(", %(tokens)s extended via %(method)s") % {
            "tokens": format_grouped(model.context.extended),
            "method": model.context.extended_method or _("unspecified method"),
        }
    table.add_row(_("Context"), Text(context))
    table.add_row(
        _("Architecture"),
        Text(f"{model.architecture.class_}, gguf_arch={model.architecture.gguf_arch}"),
    )
    if model.architecture.notes:
        table.add_row(_("Notes"), Text(model.architecture.notes))
    table.add_row(_("Capabilities"), Text(", ".join(model.capabilities)))
    table.add_row(_("Use cases"), Text(", ".join(model.use_cases)))
    table.add_row(_("Quality baseline"), str(model.quality.baseline))
    for benchmark in model.quality.benchmarks:
        table.add_row(
            _("Benchmark"),
            Text(f"{benchmark.name}: {localise_number(str(benchmark.score))} ({benchmark.source})"),
        )
    for index, source in enumerate(model.sources):
        location = source.repo if source.kind == "gguf" else source.path
        table.add_row(
            _("Source %(index)d") % {"index": index + 1},
            Text(f"{location} ({source.kind}, trust={source.trust})"),
        )
    return table


def _facts_summary(facts: GgufFacts | None) -> str:
    """A one-line summary of a quant's architecture facts, or a note that none are known yet."""
    if facts is None:
        return _("not read yet")
    parts = [facts.arch]
    if facts.n_layer is not None:
        parts.append(
            ngettext("%(count)d layer", "%(count)d layers", facts.n_layer)
            % {"count": facts.n_layer}
        )
    if facts.n_expert:
        if facts.n_expert_used is not None:
            parts.append(
                ngettext("%(count)d/%(used)d expert", "%(count)d/%(used)d experts", facts.n_expert)
                % {"count": facts.n_expert, "used": facts.n_expert_used}
            )
        else:
            parts.append(
                ngettext("%(count)d expert", "%(count)d experts", facts.n_expert)
                % {"count": facts.n_expert}
            )
    return ", ".join(parts)


def render_quants(quants: Sequence[QuantDetail]) -> Table:
    """A table of every quant with its size, bits per weight and architecture facts.

    There is deliberately no "downloaded" column: see :class:`QuantDetail` for why
    that field does not exist yet either. Every other cell here is honest about not
    knowing something (``unknown``, ``not read yet``); a column that can only ever
    say ``no`` would not be.
    """
    table = Table(title=_("Quants"))
    table.add_column(_("Name"))
    table.add_column(_("Size"), justify="right")
    table.add_column(_("BPW"), justify="right")
    table.add_column(_("Facts"))
    for quant in quants:
        table.add_row(
            Text(quant.name),
            format_bytes(quant.bytes_),
            localise_number(f"{quant.bpw:.2f}")
            if quant.bpw is not None
            else pgettext("bits per weight", "unknown"),
            Text(_facts_summary(quant.facts)),
        )
    return table
