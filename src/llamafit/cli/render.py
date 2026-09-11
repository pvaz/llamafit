# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Rich tables for the host, llama.cpp status, probes, findings and the catalog.

Every sentence here is built at render time, so the eager translation functions are the
right ones: by the time a table is drawn the language has been chosen. Nothing on this
page is a module-level constant holding a message, which is the shape that would need the
deferred pair.

Three habits run through the file. A counted noun goes through :func:`ngettext` rather
than an ``s`` bolted onto a word, because a suffix is an English rule that most languages
do not share, and every count that a noun has to agree with gets an entry of its own: one
entry can select on one number, so two numbers in one sentence means one of the two nouns
is left to guess. A word short enough for two rows to share — *unknown*, *none detected*,
*ok* — carries a :func:`pgettext` context naming its row, because the Portuguese for it
is inflected and one translation cannot be right in both places. And a line assembled
from optional pieces is written out as one whole message per shape rather than joined
with a separator in Python: four of the languages here punctuate a list four different
ways, and a translator can move a word across a join they can see and not across one
they never do.

A fourth habit is what makes this page readable in Arabic, Hebrew and Urdu. Every value
that is not words — a flag, a command name, a path, a model or repository id, a URL, a
size with its unit — goes through :func:`llamafit.i18n.isolate` on its way into a
sentence, and every finished cell goes through :func:`llamafit.i18n.for_display`. The
first makes each identifier a directional island, so a leading hyphen cannot drift and
``--verbose`` cannot reach a reader as ``verbose--``; the second fixes the line's base
direction and catches the identifiers a translator had to keep verbatim inside prose.
Columns and their alignment are mirrored the same way. All of it is inert for a
left-to-right language, and none of it is anywhere near ``--json``: the marks are added
here, at the last moment before Rich draws, and the services that build the
machine-readable output never see one. :mod:`llamafit.i18n.bidi` explains the characters,
and ``docs/translations.md`` is honest about which terminals act on them.

The last two habits answer to each other, and the island always goes around the value the
sentence *now* interpolates. Writing a whole shape out instead of joining fragments moved
several values into a placeholder that used to be spliced beside one, and each of those is
an island the older shape had no way to make: the parameter count carries its own ``B``
into ``%(total)s`` rather than having the letter welded on after it, so ``27B`` is isolated
whole, and the extended-context sentence stopped being a fragment that opened with a comma,
so its token count is reached the same way the native one is. The reverse holds too. A
capability is a word now rather than an enum value, so it is translated rather than
treated as an identifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from llamafit.hardware.gputable import lookup_gpu
from llamafit.hwprofile.loader import LoadedProfile
from llamafit.i18n import (
    _,
    for_display,
    isolate,
    mirror_justify,
    ngettext,
    pgettext,
    reading_order,
)
from llamafit.models.catalog import CatalogModel
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Cpu, Gpu, Host, Memory, Override, Probe, Simulation, Source
from llamafit.models.hwprofile import ProfileCpu, ProfileGpu, ProfileMemory
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.catalog import ModelSummary, QuantDetail
from llamafit.services.doctor import Finding
from llamafit.units import billions_suffix, format_bytes, format_grouped, localise_number

_LEVEL_STYLE = {"ok": "green", "warn": "yellow", "error": "red"}

Cell = str | Text
"""What one cell of a table may be: markup-free text, or a plain string of labels."""


def _cell(text: str) -> Text:
    """One cell's finished text, prepared for the reader's writing direction.

    ``Text`` rather than a markup string for the reason every other value here is: a
    device name or a path may hold square brackets Rich would try to parse as a tag.
    """
    return Text(for_display(text))


def _size(n: int | None) -> str:
    """A byte size as one directional island, or the word for a size nobody could read.

    ``format_bytes`` answers with a translated *unknown* when it is given ``None``, and
    that is prose, not an identifier: isolating it would put a mark around a word in the
    reader's own language for no reason. Only a real ``7.6 GiB`` becomes an island, which
    is what keeps the figure and its unit from being parted in an Arabic sentence.
    """
    return isolate(format_bytes(n)) if n is not None else format_bytes(n)


def _add_columns(table: Table, columns: Sequence[Mapping[str, Any]]) -> None:
    """Add a table's columns in reading order, with their alignment mirrored to match.

    Each mapping is the ``header`` plus whatever :meth:`rich.table.Table.add_column`
    should be told about that column. For a right-to-left language the columns are added
    last-read first, so the first one lands against the right edge, and a left-aligned
    column becomes right-aligned so its padding falls after the last character rather than
    before the first. Neither happens in a left-to-right language, where this is exactly
    the sequence of ``add_column`` calls that was here before.

    Args:
        table: The table being built.
        columns: The columns, in the order a reader meets them.
    """
    for column in reading_order(columns):
        options = dict(column)
        header = str(options.pop("header"))
        options["justify"] = mirror_justify(str(options.get("justify", "left")))
        table.add_column(for_display(header), **options)


def _add_row(table: Table, *cells: Cell) -> None:
    """Add one row, its cells in the same order :func:`_add_columns` put the columns.

    A plain string is a row label this file wrote, so it is prepared for the writing
    direction here rather than at thirty call sites; a ``Text`` has already been through
    :func:`_cell` and is left alone.
    """
    prepared: list[Cell] = [c if isinstance(c, Text) else for_display(c) for c in cells]
    table.add_row(*reading_order(prepared))


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


def bandwidth_sentence(gbps: float | None, source: Source, *, cached: bool = False) -> str:
    """A memory bandwidth figure with how it was arrived at, and whether it was kept.

    Args:
        gbps: The figure, or ``None`` when nothing produced one.
        source: How it was arrived at.
        cached: Whether it was read back from an earlier run rather than timed on this one.

    Returns:
        The phrase to drop into whichever line is naming it.

    Written once and public because two places name this figure and they have to name it
    identically: the host table, where it is one of the machine's properties, and the
    board, where it is the constant every speed on the table was divided by. A number that
    read ``measured`` in one of them and ``measured, cached`` in the other would be two
    numbers as far as anybody comparing two answers is concerned.
    """
    if not gbps:
        return pgettext("memory bandwidth", "unknown")
    values = {"gbps": localise_number(str(gbps)), "source": _bandwidth_source_label(source)}
    if cached:
        return _("%(gbps)s GB/s (%(source)s, cached)") % values
    return _("%(gbps)s GB/s (%(source)s)") % values


def _override_label(field: Override) -> str:
    """The name of a pool somebody substituted by hand, in the reader's language.

    A word, not the identifier: ``gpu_memory`` is what a script filters on and *VRAM* is
    what the sentence beside it has to read as. The identifiers stay in the JSON, where
    the script is.
    """
    labels = {
        "gpu_memory": pgettext("simulation override", "VRAM"),
        "ram": pgettext("simulation override", "system memory"),
        "cpu_cores": pgettext("simulation override", "CPU cores"),
    }
    return labels.get(field, field)


def _simulation_note(simulation: Simulation) -> str:
    """Why the host below this line is not the machine the reader is sitting at.

    The three shapes are written out whole rather than assembled from a stem and a
    clause: which of them applies is what the sentence is *about*, and a translator who
    can see all three can put "not this machine" wherever their language puts it.
    """
    count = len(simulation.overrides)
    fields = ", ".join(_override_label(field) for field in simulation.overrides)
    if simulation.profile and count:
        return ngettext(
            "these figures come from the hardware profile %(profile)s with %(fields)s "
            "overridden; they are not this machine",
            "these figures come from the hardware profile %(profile)s with %(fields)s "
            "overridden; they are not this machine",
            count,
        ) % {"profile": isolate(simulation.profile), "fields": fields}
    if simulation.profile:
        return _(
            "these figures come from the hardware profile %(profile)s; they are not this machine"
        ) % {"profile": isolate(simulation.profile)}
    if count:
        return ngettext(
            "%(fields)s was overridden for a what-if; these figures are not what was scanned",
            "%(fields)s were overridden for a what-if; these figures are not what was scanned",
            count,
        ) % {"fields": fields}
    return _("these figures are not this machine")


def render_simulation(simulation: Simulation | None) -> Text | None:
    """One red line saying the figures under it are not this machine, or ``None``.

    Args:
        simulation: What was substituted, straight off whatever is about to be printed.

    Returns:
        The line, or ``None`` when nothing was substituted and there is nothing to say.

    :func:`render_host` says this inside its own table, because a host is a table. A
    board, a fit listing and a plan are not, and each of them is a page of numbers
    computed *for* a machine rather than a description of one -- so the warning goes
    above them, in the two messages the host already uses, before the reader meets a
    figure. The serialised half of the same promise is the ``simulated`` field on each of
    those documents; a red line is no use to a script.
    """
    if simulation is None:
        return None
    line = Text(for_display(_("SIMULATED")), style="bold red")
    line.append("  ")
    line.append(for_display(_simulation_note(simulation)), style="red")
    return line


def simulated_answer(simulation: Simulation | None, *parts: RenderableType) -> Group:
    """Whatever a command is about to print, under the red line when it is not this machine.

    Args:
        simulation: What was substituted, off the answer itself rather than off the
            command's own flags: a document that says which machine it describes is the
            only thing that can still say so once it has been handed somewhere else.
        parts: The answer, in the order it should be drawn.

    Returns:
        The parts, with the banner above them when there is one to draw.

    This exists because :func:`render_simulation` was being consulted by each renderer that
    had a table to put it over, and a *renderer* is not what the promise can hang on: a
    board that ranked nothing has no table, so the branch that printed "nothing was ranked"
    printed it without the line and a reader met an empty answer with no way of knowing it
    was empty for a machine that is not theirs. Every path that prints something computed
    from a substituted host goes through this function, including the paths where what is
    printed is a sentence saying there is nothing to print.
    """
    banner = render_simulation(simulation)
    return Group(*([] if banner is None else [banner]), *parts)


def _capability_label(capability: str) -> str:
    """One of a model's capabilities, in the reader's language.

    These are ordinary words — *coding*, *vision*, *audio* — and a reader meets them as
    words, not as identifiers: they are the answer to "what is this model for", which is
    the question the table exists to answer. They reach the catalog as enum values, which
    is why they were English everywhere for as long as they were, and why each one is
    written out here as its own literal call rather than looked up from the enum: the
    extractor reads the syntax tree, so a message it can find is a message somebody can
    translate.

    A name the enum grows before this table does comes back unchanged, in English, rather
    than disappearing from the row.

    ``--capability`` still takes the English identifier, and the hint on a mistyped one
    lists those, so nothing here is a value a reader has to type back.

    Both tables still isolate the list these build, and being words now is the reason
    rather than an objection to it. A cell that is Latin until a catalog answers these
    eight entries and the reader's own script afterwards is a run whose direction is not
    known in advance, which is the case U+2068 FIRST STRONG ISOLATE exists for; the mark
    also keeps the commas between the names from being drawn into the Arabic around the
    cell, whichever script the names themselves are in.
    """
    labels = {
        "coding": pgettext("model capability", "coding"),
        "thinking": pgettext("model capability", "thinking"),
        "vision": pgettext("model capability", "vision"),
        "tools": pgettext("model capability", "tools"),
        "multilingual": pgettext("model capability", "multilingual"),
        "long-context": pgettext("model capability", "long-context"),
        "embeddings": pgettext("model capability", "embeddings"),
        "audio": pgettext("model capability", "audio"),
    }
    return labels.get(capability, capability)


def _from_table(bandwidth_gbps: float | None, compute_tflops_fp16: float | None) -> str:
    """The specification-table figures a GPU has, as one phrase in the reader's language.

    The three shapes are written out rather than joined from pieces: a translator cannot
    move a word across a join they never see, and ``and`` is a word, not punctuation.

    The figures stay bare. Each is a number the message glues its unit to from the
    outside — ``%(bandwidth)s GB/s`` — so an island around the number alone would part it
    from the unit it is meaningless without. ``docs/translations.md`` lists that as
    residue, with the message change that would clear it.
    """
    if bandwidth_gbps and compute_tflops_fp16:
        return _("%(bandwidth)s GB/s and %(compute)s TFLOPS fp16") % {
            "bandwidth": localise_number(str(bandwidth_gbps)),
            "compute": localise_number(str(compute_tflops_fp16)),
        }
    if bandwidth_gbps:
        return _("%(bandwidth)s GB/s") % {"bandwidth": localise_number(str(bandwidth_gbps))}
    return _("%(compute)s TFLOPS fp16") % {"compute": localise_number(str(compute_tflops_fp16))}


def _cores(cpu: Cpu) -> str:
    """What the CPU offers, as one whole phrase in the reader's language.

    Cores, threads and performance cores are three counted nouns, and a ``.po`` entry
    selects its form on one number: a single message carrying all three would leave two
    of the nouns agreeing with a count that is not theirs, which in Czech or Russian is
    the difference between a word and a misspelling. Each count therefore gets its own
    counted entry, and the two shapes the row can take are written out whole.

    Nothing here is isolated: every piece of it is a number sitting next to a word in the
    reader's own language, which is prose and not an identifier.
    """
    cores = ngettext("%(count)d core", "%(count)d cores", cpu.physical_cores) % {
        "count": cpu.physical_cores
    }
    threads = ngettext("%(count)d thread", "%(count)d threads", cpu.logical_cores) % {
        "count": cpu.logical_cores
    }
    if not cpu.performance_cores:
        return _("%(cores)s / %(threads)s") % {"cores": cores, "threads": threads}
    performance = ngettext(
        "%(count)d performance core",
        "%(count)d performance cores",
        cpu.performance_cores,
    ) % {"count": cpu.performance_cores}
    return _("%(cores)s / %(threads)s, %(performance)s") % {
        "cores": cores,
        "threads": threads,
        "performance": performance,
    }


def _memory_kind(memory: Memory) -> str | None:
    """What the memory is, as one whole phrase, or ``None`` when nothing is known.

    The type is data read off the machine — ``DDR5``, ``LPDDR5X`` — so it is never
    translated; the sentence it sits in is. Being data is also what makes it an island:
    it is a Latin run with digits in it, and an Arabic sentence would otherwise decide
    for itself which way round to lay it. The speed stays bare, because ``MT/s`` is
    written outside its placeholder.

    The three shapes are written out rather than joined, the same way :func:`_from_table`
    writes out its three.
    """
    if memory.type and memory.speed_mts:
        return _("%(type)s at %(speed)d MT/s") % {
            "type": isolate(memory.type),
            "speed": memory.speed_mts,
        }
    if memory.type:
        return isolate(memory.type)
    if memory.speed_mts:
        return _("%(speed)d MT/s") % {"speed": memory.speed_mts}
    return None


def _memory_details(memory: Memory) -> str:
    """The middle of the memory row: what the modules are and how many there are.

    Every shape is a message of its own. The alternative — building a list and joining it
    with ``", "`` — hands the translator three fragments and keeps the punctuation between
    them in Python, where Japanese cannot reach it to write ``、``, Arabic cannot reach it
    to write ``، ``, and no language can put a conjunction before the last item.

    The module and channel counts stay two counted entries rather than one message holding
    both numbers, so each noun agrees with the count next to it; what varies between the
    shapes is the sentence they are set into.
    """
    kind = _memory_kind(memory)
    modules = (
        ngettext("%(count)d module", "%(count)d modules", memory.modules)
        % {"count": memory.modules}
        if memory.modules
        else None
    )
    channels = (
        ngettext("%(count)d channel", "%(count)d channels", memory.channels)
        % {"count": memory.channels}
        if memory.channels
        else None
    )
    if kind and modules and channels:
        return _("%(kind)s, %(modules)s across %(channels)s") % {
            "kind": kind,
            "modules": modules,
            "channels": channels,
        }
    if kind and modules:
        return _("%(kind)s, %(modules)s") % {"kind": kind, "modules": modules}
    if kind and channels:
        return _("%(kind)s, %(channels)s") % {"kind": kind, "channels": channels}
    if kind:
        return kind
    if modules and channels:
        return _("%(modules)s across %(channels)s") % {"modules": modules, "channels": channels}
    if modules:
        return modules
    if channels:
        return channels
    return pgettext("memory type", "type unknown")


def _gpu_specs(gpu: Gpu) -> str:
    """The bundled specification figures and the driver, as one whole phrase.

    Bandwidth and compute come from the bundled specification table, not from this
    machine, so they are labelled: no number reaches the user without its source. The four
    shapes are written out, because the comma between the two halves is punctuation a
    language chooses and not punctuation this file does.

    The driver version is an island in each shape that names it: it is a dotted Latin run,
    and the dots are as direction-neutral as a flag's hyphens.
    """
    from_table = (
        _from_table(gpu.bandwidth_gbps, gpu.compute_tflops_fp16)
        if gpu.bandwidth_gbps or gpu.compute_tflops_fp16
        else None
    )
    if from_table and gpu.driver:
        return _("%(specs)s (spec), driver %(driver)s") % {
            "specs": from_table,
            "driver": isolate(gpu.driver),
        }
    if from_table:
        return _("%(specs)s (spec)") % {"specs": from_table}
    if gpu.driver:
        return _("driver %(driver)s") % {"driver": isolate(gpu.driver)}
    return pgettext("GPU specifications", "no specs")


def _vram(gpu: Gpu) -> str:
    """One card's memory, saying which of the three things is known about it.

    A size and a free figure is the whole answer, and a size the Vulkan driver supplied
    says so: it is the driver's allocation budget rather than the card's own memory, and
    the two are not the same claim any more than a measured bandwidth and an assumed one
    are. A size with no free figure is the shape a driver without ``VK_EXT_memory_budget``
    leaves behind -- it is worth printing, and it is not enough to plan on, which is what
    the row below the cards is for.
    """
    if gpu.vram_total_bytes and gpu.vram_free_bytes is not None:
        if gpu.vram_source == "estimated":
            return _("%(total)s VRAM, %(free)s free (Vulkan driver)") % {
                "total": _size(gpu.vram_total_bytes),
                "free": _size(gpu.vram_free_bytes),
            }
        return _("%(total)s VRAM, %(free)s free") % {
            "total": _size(gpu.vram_total_bytes),
            "free": _size(gpu.vram_free_bytes),
        }
    if gpu.vram_total_bytes:
        return _("%(total)s VRAM, free unknown") % {"total": _size(gpu.vram_total_bytes)}
    return pgettext("GPU VRAM", "VRAM unknown")


def render_host(host: Host) -> Table:
    """A two-column table with everything the scan found.

    Values are built as ``Text`` rather than markup strings: a device name or a path
    may contain square brackets, which Rich would otherwise try to parse as a tag. Each
    of them is also isolated, because a device name, an architecture and a path are
    identifiers, and an Arabic sentence lays one out backwards without a mark to say so.

    A host that did not come from the probes says so in its first row, in red, before a
    reader has met a single figure. That row is here rather than at the commands that
    print a simulated host because this is the one function all of them go through, and a
    warning that has to be remembered at each call site is a warning that will be
    forgotten at one of them.
    """
    table = Table(title=for_display(_("Host")), show_header=False, box=None, pad_edge=False)
    _add_columns(table, [{"header": "key", "style": "bold"}, {"header": "value"}])
    if host.simulation is not None:
        _add_row(
            table,
            Text(for_display(_("SIMULATED")), style="bold red"),
            Text(for_display(_simulation_note(host.simulation)), style="red"),
        )
    _add_row(
        table,
        _("OS"),
        _cell(
            _("%(os)s %(version)s (%(arch)s)")
            % {
                "os": isolate(host.os),
                "version": isolate(host.os_version),
                "arch": isolate(host.arch),
            }
        ),
    )
    _add_row(
        table,
        _("CPU"),
        _cell(
            _("%(model)s; %(cores)s; %(isa)s")
            % {
                "model": isolate(host.cpu.model),
                "cores": _cores(host.cpu),
                "isa": isolate(" ".join(host.cpu.isa))
                if host.cpu.isa
                else pgettext("CPU instruction sets", "isa unknown"),
            }
        ),
    )
    mem = host.memory
    bandwidth = bandwidth_sentence(
        mem.bandwidth_gbps, mem.bandwidth_source, cached=mem.bandwidth_cached
    )
    _add_row(
        table,
        _("Memory"),
        _cell(
            _("%(total)s total, %(available)s available; %(details)s; bandwidth %(bandwidth)s")
            % {
                "total": _size(mem.total_bytes),
                "available": _size(mem.available_bytes),
                "details": _memory_details(mem),
                "bandwidth": bandwidth,
            }
        ),
    )
    if not host.gpus:
        _add_row(table, _("GPU"), pgettext("GPU", "none detected"))
    for gpu in host.gpus:
        _add_row(
            table,
            _("GPU %(index)d") % {"index": gpu.index},
            _cell(
                _("%(name)s (%(backend)s); %(vram)s; %(specs)s")
                % {
                    "name": isolate(gpu.name),
                    "backend": isolate(gpu.backend_hint),
                    "vram": _vram(gpu),
                    "specs": _gpu_specs(gpu),
                }
            ),
        )
    unsized = host.unsized_gpus
    if unsized:
        _add_row(
            table,
            _("Card memory"),
            _cell(
                _(
                    "%(gpus)s: nothing here could read how much of the card is free, so "
                    "every budget on this machine was computed as if there were no card "
                    "at all. Install the Vulkan tools, or write the size into a hardware "
                    "profile."
                )
                % {"gpus": ", ".join(isolate(gpu.name) for gpu in unsized)}
            ),
        )
    if host.unified_memory:
        _add_row(table, _("Memory pool"), _("unified (GPU shares system memory)"))
    for disk in host.disks:
        _add_row(
            table,
            _("Disk"),
            _cell(
                _("%(path)s: %(free)s free of %(total)s")
                % {
                    "path": isolate(disk.path),
                    "free": _size(disk.free_bytes),
                    "total": _size(disk.total_bytes),
                }
            ),
        )
    return table


def render_llamacpp(llamacpp: LlamaCpp) -> Table:
    """A two-column table describing the installation, with paths never read as markup.

    The build number keeps its ``b`` un-isolated on purpose: the message glues the letter
    to the digits, and isolating the digits alone would part ``b`` from ``10867`` and let
    the two swap places in an Arabic line. See :func:`llamafit.i18n.isolate`.
    """
    table = Table(title="llama.cpp", show_header=False, box=None, pad_edge=False)
    _add_columns(table, [{"header": "key", "style": "bold"}, {"header": "value"}])
    if not llamacpp.installed:
        _add_row(table, _("Installed"), pgettext("llama.cpp is installed", "no"))
    else:
        if llamacpp.build and llamacpp.commit:
            build = _("b%(build)d (%(commit)s)") % {
                "build": llamacpp.build,
                "commit": isolate(llamacpp.commit),
            }
        elif llamacpp.build:
            build = _("b%(build)d") % {"build": llamacpp.build}
        else:
            build = pgettext("llama.cpp build", "unknown build")
        _add_row(
            table,
            _("Installed"),
            _cell(
                _("yes, %(build)s at %(path)s") % {"build": build, "path": isolate(llamacpp.path)}
            ),
        )
        _add_row(
            table,
            _("Backends"),
            _cell(
                isolate(", ".join(llamacpp.backends))
                if llamacpp.backends
                else pgettext("backends", "none detected")
            ),
        )
        _add_row(table, _("Local models"), str(len(llamacpp.local_models)))
    for server in llamacpp.running_servers:
        _add_row(
            table,
            _("Running"),
            _cell(
                _("%(url)s: %(model)s, context %(context)s")
                % {
                    "url": isolate(server.url),
                    "model": isolate(server.model)
                    if server.model
                    else pgettext("model name", "unknown model"),
                    "context": server.n_ctx or pgettext("context length", "unknown"),
                }
            ),
        )
    for problem in llamacpp.problems:
        _add_row(table, _("Problem"), _cell(problem))
    return table


def render_probes(probes: Iterable[Probe]) -> Table:
    """Probe-by-probe outcome; the error text is never parsed as markup.

    A ``server:<port>`` probe that found nothing is the ordinary case on a machine with no
    llama-server running, so it is shown dim rather than as a failure.
    """
    table = Table(title=for_display(_("Probes")), box=None, pad_edge=False)
    _add_columns(
        table,
        [
            {"header": _("probe"), "style": "bold"},
            {"header": _("result")},
            {"header": _("ms"), "justify": "right"},
        ],
    )
    for probe in probes:
        if probe.ok:
            status = Text(for_display(pgettext("probe result", "ok")), style="green")
        elif probe.name.startswith("server:"):
            status = Text(for_display(pgettext("probe result", "no server")), style="dim")
        else:
            status = Text(for_display(pgettext("probe result", "failed")), style="yellow")
            if probe.error:
                status.append(f" {for_display(probe.error)}")
        # A probe name is an identifier and half of them carry a hyphen or a flag
        # (`nvidia-smi`, `llama-server --version`), which is exactly what drifts.
        _add_row(table, _cell(isolate(probe.name)), status, str(probe.duration_ms))
    return table


def render_findings(findings: Iterable[Finding]) -> Table:
    """Findings with level colouring and hints; the finding text is never parsed as markup.

    A finding arrives here already written out: ``diagnose`` composed the sentence, and
    the same object is what ``--json`` serialises, so nothing may be marked before this
    point. The identifiers inside it are therefore the ones a translator kept verbatim in
    their own sentence — ``--verbose``, a URL, a backticked command line — and
    :func:`llamafit.i18n.for_display` is what finds them. The interpolated values a
    finding carries (a path, a probe name, a GPU) cannot be reached from here at all; that
    is written up in ``docs/translations.md`` rather than guessed at with a wider pattern.
    """
    table = Table(title=for_display(_("Findings")), box=None, pad_edge=False)
    _add_columns(table, [{"header": _("level")}, {"header": _("finding")}])
    for finding in findings:
        text = Text.assemble(
            (for_display(finding.title), "bold"), "\n", for_display(finding.detail)
        )
        if finding.hint:
            text.append("\n")
            text.append(for_display(_("Hint: %(hint)s") % {"hint": finding.hint}), style="dim")
        _add_row(
            table,
            Text(for_display(_level_label(finding.level)), style=_LEVEL_STYLE[finding.level]),
            text,
        )
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

    A capability name reaches a reader as a word and goes through
    :func:`_capability_label`, so the widths measured here are the widths of the
    translated names and not of the catalog's identifiers.

    ``width`` is a count of terminal cells, so the fit is measured with ``cell_len`` and
    not with ``len``: a Japanese character is one of those and two of these. What is
    measured is the bare text: the caller isolates the answer, and an isolate is two
    characters of no width at all.
    """
    eligible = [_capability_label(name) for name in capabilities[:limit]]
    total = len(capabilities)
    for shown_count in range(len(eligible), 0, -1):
        shown = eligible[:shown_count]
        dropped = total - shown_count
        text = ", ".join(shown)
        candidate = f"{text} +{dropped}" if dropped else text
        if cell_len(candidate) <= width:
            return candidate
    return f"+{total}" if total else ""


# Every width in this section is a count of terminal cells, never of characters, and
# `cell_len` is the only thing that may measure one. The id column is where getting that
# wrong shows up, because it is the only column allowed to fold: measure a heading in
# characters and the budget hands the capabilities column a leftover that is too generous,
# Rich re-measures for real when it draws, finds the row too wide, and takes the shortfall
# out of the id -- the one column that has to stay typable. Measured with `len`, four of
# the five shipped ids fold onto a second line at eighty columns under Japanese headings,
# and Chinese does it too although its headings are genuinely narrower than the English
# ones, which is what says the mis-measurement alone is the cause.
#
# `cell_len` is right and still not the whole story for every script. Thai vowel and tone
# marks are zero width by Unicode, so a Thai heading measures narrower than it looks and
# the box only lines up in a terminal that also gives those marks no advance. Terminals
# disagree about that, and nothing this code can measure would settle it.
_ID_COLUMN_MAX_WIDTH = 28
"""How wide the id column may grow, in terminal cells."""

# What one more column costs beyond its own content: a border and the padding on
# each side of it. The table's own leading border is the one extra "+1" charged
# once, before any column, in the budget below.
_COLUMN_OVERHEAD = 3
_TABLE_OVERHEAD = 1

_FOOTNOTE_MARKER = "*"
"""What marks the quality heading as having a footnote under the table.

It is punctuation and not a word, so it is not part of any message: a translator who met
it inside one would have to decide whether it is theirs to keep, and a heading that lost
it would point at a caption for no visible reason.
"""


def _list_headings() -> dict[str, str]:
    """The list table's column headings, translated once and read three times.

    :func:`_column_budget` measures these, :func:`render_catalog_list` prints them and
    :func:`_table_caption` names two of them, and all three have to agree: a heading
    measured in English and printed in Portuguese would give a column a width its own
    title does not fit into, and a caption that spelled the word out a second time would
    be a second entry a translator has to keep in step by hand, with nothing checking.

    Each carries a ``column heading`` context. Two of them — ``Context`` and
    ``Capabilities`` — are also row labels in :func:`render_model_facts`, where there is
    no width budget at all; one entry serving both jobs forces a language whose full word
    is long to choose between a heading that crowds the table and a label that reads
    clipped. With a context on each, a translator answers the two questions separately.
    """
    return {
        "id": pgettext("column heading", "ID"),
        "quality": pgettext("column heading", "Quality"),
        "params": pgettext("column heading", "Params"),
        "context": pgettext("column heading", "Context"),
        "capabilities": pgettext("column heading", "Capabilities"),
    }


def _quality_heading(headings: Mapping[str, str]) -> str:
    """The quality column's heading with its footnote marker, as printed and as measured."""
    return f"{headings['quality']}{_FOOTNOTE_MARKER}"


def _table_caption(included: Sequence[str]) -> str | None:
    """The footnote explaining whichever of quality and params is shown, or none.

    Each caption is handed the heading it is about rather than spelling it out, so the
    heading has one source. Written out twice, the two would drift the first time a
    translator shortened a heading to fit the column and left the caption naming the
    longer word, and nothing in the build would notice.

    The backticked command inside the quality caption is not marked here. It is an
    identifier a translator keeps verbatim inside their own sentence, which is exactly
    what :func:`llamafit.i18n.for_display` picks out when the caption is printed.
    """
    headings = _list_headings()
    captions = {
        "quality": _(
            "%(heading)s is the editorial baseline, before any quantisation penalty; "
            "run `llamafit info <model>` for the sourced benchmarks behind it."
        )
        % {"heading": headings["quality"]},
        "params": _(
            "%(heading)s is total/active billions for a mixture-of-experts model, "
            "or one number when they are equal."
        )
        % {"heading": headings["params"]},
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
        [
            cell_len(_quality_heading(headings)),
            *(cell_len(str(s.quality_baseline)) for s in summaries),
        ]
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

    In a right-to-left language the columns are added in the reverse order and every
    alignment is mirrored, so the id lands against the right edge where reading starts and
    the ragged edge falls at the left where it ends. The budget above is untouched by
    that: a column is the same width whichever end of the line it is drawn from, and the
    marks that make each cell an island have no width at all.

    Args:
        summaries: The rows to render, already filtered and sorted.
        console_width: The console's width; defaults to 80, the narrowest width
            this table is designed for.
    """
    id_width, included, capabilities_width = _column_budget(summaries, console_width)
    include_capabilities = capabilities_width >= 3  # room for at least a bare "+N" marker
    headings = _list_headings()
    caption = _table_caption(included)

    table = Table(
        title=for_display(_("Models")),
        caption=for_display(caption) if caption else None,
    )
    columns: list[Mapping[str, Any]] = [
        {"header": headings["id"], "style": "bold", "max_width": id_width, "overflow": "fold"}
    ]
    if "quality" in included:
        columns.append({"header": _quality_heading(headings), "justify": "right", "no_wrap": True})
    for name in ("params", "context"):
        if name in included:
            columns.append({"header": headings[name], "justify": "right", "no_wrap": True})
    if include_capabilities:
        columns.append({"header": headings["capabilities"], "no_wrap": True})
    _add_columns(table, columns)

    for summary in summaries:
        # An id, a score, a `27/3B` pair and a `256K` context are identifiers and bare
        # figures, never words, and each is isolated so that neither the slash nor the
        # `+N` marker can be drawn into the Arabic around it. The capabilities cell is
        # words now and is isolated for a different reason: see `_capability_label`.
        row: list[Cell] = [_cell(isolate(summary.id))]
        if "quality" in included:
            row.append(isolate(summary.quality_baseline))
        if "params" in included:
            row.append(isolate(_fmt_params(summary.params_total_b, summary.params_active_b)))
        if "context" in included:
            row.append(isolate(_fmt_context_compact(summary.context_native)))
        if include_capabilities:
            row.append(
                _cell(isolate(_fmt_capabilities(summary.capabilities, width=capabilities_width)))
            )
        _add_row(table, *row)
    return table


def render_model_facts(model: CatalogModel) -> Table:
    """A two-column table of one model's curated facts: everything but its quants.

    As with every other table here, catalog text (licence slug, architecture notes,
    a source's repository or path) is wrapped in ``Text`` rather than interpolated
    into a markup string. That text is data: it is shown as it was written and never
    translated, and the row labels around it are what a reader's language reaches. Being
    data is also what makes each of them an isolate: an id, a licence slug, a URL, a
    release date and an architecture name are all Latin runs with a hyphen, a slash or a
    dot in them, and every one of those is a character the algorithm would otherwise let
    the surrounding Arabic claim.

    The parameter count is an island too, which it could not be while the message wrote
    ``%(total)sB`` and welded the suffix on outside the placeholder. It now arrives whole,
    ``27B`` and its localised mark together, so the mark goes round both. What is left
    unmarked is a figure whose unit is still written in the message — the memory speed,
    the GPU's bandwidth and compute — and ``docs/translations.md`` lists those.
    """
    table = Table(
        title=Text(for_display(isolate(f"{model.name} ({model.id})"))),
        show_header=False,
        box=None,
        pad_edge=False,
    )
    _add_columns(table, [{"header": "key", "style": "bold"}, {"header": "value"}])
    _add_row(table, _("Vendor"), _cell(isolate(model.vendor)))
    _add_row(table, _("Family"), _cell(isolate(model.family)))
    _add_row(table, _("Release date"), isolate(model.release_date))
    _add_row(
        table,
        _("Licence"),
        _cell(
            _("%(spdx)s (%(url)s)")
            % {"spdx": isolate(model.license.spdx), "url": isolate(model.license.url)}
        ),
    )
    # The B is the suffix a parameter count carries in the model's own name, and it is the
    # one `billions_suffix` hands out, so the two tables that print a parameter count
    # print the same mark. Welded to the placeholder it was neither: a language that
    # answered that entry got its suffix in the list table and the English one here.
    _add_row(
        table,
        _("Parameters"),
        _cell(
            _("%(total)s total, %(active)s active")
            % {
                "total": isolate(f"{_fmt_billions(model.params.total_b)}{billions_suffix()}"),
                "active": isolate(f"{_fmt_billions(model.params.active_b)}{billions_suffix()}"),
            }
        ),
    )
    if model.context.extended:
        context = _("%(tokens)s tokens native, %(extended)s extended via %(method)s") % {
            "tokens": isolate(format_grouped(model.context.native)),
            "extended": isolate(format_grouped(model.context.extended)),
            "method": isolate(model.context.extended_method)
            if model.context.extended_method
            else pgettext("context extension method", "unspecified method"),
        }
    else:
        context = _("%(tokens)s tokens native") % {
            "tokens": isolate(format_grouped(model.context.native))
        }
    _add_row(table, pgettext("table row label", "Context"), _cell(context))
    _add_row(
        table,
        _("Architecture"),
        _cell(isolate(f"{model.architecture.class_}, gguf_arch={model.architecture.gguf_arch}")),
    )
    if model.architecture.notes:
        _add_row(table, _("Notes"), _cell(model.architecture.notes))
    _add_row(
        table,
        pgettext("table row label", "Capabilities"),
        _cell(isolate(", ".join(_capability_label(name) for name in model.capabilities))),
    )
    _add_row(table, _("Use cases"), _cell(isolate(", ".join(model.use_cases))))
    _add_row(table, _("Quality baseline"), str(model.quality.baseline))
    for benchmark in model.quality.benchmarks:
        _add_row(
            table,
            _("Benchmark"),
            _cell(
                isolate(
                    f"{benchmark.name}: {localise_number(str(benchmark.score))} "
                    f"({benchmark.source})"
                )
            ),
        )
    for index, source in enumerate(model.sources):
        location = source.repo if source.kind == "gguf" else source.path
        _add_row(
            table,
            _("Source %(index)d") % {"index": index + 1},
            _cell(isolate(f"{location} ({source.kind}, trust={source.trust})")),
        )
    return table


def _facts_summary(facts: GgufFacts | None) -> str:
    """A one-line summary of a quant's architecture facts, or a note that none are known yet.

    The architecture is data — ``qwen3moe``, ``llama`` — and never translated, so it is
    the one island here. The layer and expert counts are counted entries, and the shapes
    they can combine into are written out rather than joined, for the same reason
    :func:`_memory_details` writes its out: the comma is punctuation a language chooses.
    """
    if facts is None:
        return pgettext("GGUF facts", "not read yet")
    arch = isolate(facts.arch)
    layers = (
        ngettext("%(count)d layer", "%(count)d layers", facts.n_layer) % {"count": facts.n_layer}
        if facts.n_layer is not None
        else None
    )
    experts = _experts(facts)
    if layers and experts:
        return _("%(arch)s, %(layers)s, %(experts)s") % {
            "arch": arch,
            "layers": layers,
            "experts": experts,
        }
    if layers:
        return _("%(arch)s, %(layers)s") % {"arch": arch, "layers": layers}
    if experts:
        return _("%(arch)s, %(experts)s") % {"arch": arch, "experts": experts}
    return arch


def _experts(facts: GgufFacts) -> str | None:
    """How many experts the model has, and how many of them each token uses.

    The form is selected on the count the noun stands next to, which for the ratio is the
    *used* count and not the total: Russian and Polish inflect *experts* to agree with the
    numeral immediately before it, so ``128/2 эксперта`` and ``128/5 экспертов`` take
    different forms even though the total is 128 in both.

    Neither number is isolated. ``%(count)d/%(used)d`` puts the slash between two
    placeholders, so marking each number apart would be marking the slash out of the
    middle of them.
    """
    if not facts.n_expert:
        return None
    if facts.n_expert_used is None:
        return ngettext("%(count)d expert", "%(count)d experts", facts.n_expert) % {
            "count": facts.n_expert
        }
    return ngettext(
        "%(count)d/%(used)d expert", "%(count)d/%(used)d experts", facts.n_expert_used
    ) % {"count": facts.n_expert, "used": facts.n_expert_used}


def render_quants(quants: Sequence[QuantDetail]) -> Table:
    """A table of every quant with its size, bits per weight and architecture facts.

    There is deliberately no "downloaded" column: see :class:`QuantDetail` for why
    that field does not exist yet either. Every other cell here is honest about not
    knowing something (``unknown``, ``not read yet``); a column that can only ever
    say ``no`` would not be.
    """
    table = Table(title=for_display(_("Quants")))
    _add_columns(
        table,
        [
            {"header": _("Name")},
            {"header": _("Size"), "justify": "right"},
            {"header": _("BPW"), "justify": "right"},
            {"header": _("Facts")},
        ],
    )
    for quant in quants:
        _add_row(
            table,
            _cell(isolate(quant.name)),
            _size(quant.bytes_),
            isolate(localise_number(f"{quant.bpw:.2f}"))
            if quant.bpw is not None
            else pgettext("bits per weight", "unknown"),
            _cell(_facts_summary(quant.facts)),
        )
    return table


def _profile_origin(loaded: LoadedProfile) -> str:
    """Whether a profile shipped with LlamaFit or the reader wrote it."""
    return (
        pgettext("profile origin", "bundled")
        if loaded.bundled
        else pgettext("profile origin", "yours")
    )


def _profile_machine(loaded: LoadedProfile) -> str:
    """One line saying what machine a profile describes: its pools and its card.

    Both shapes are written out whole. A machine with no graphics card is not a machine
    with an empty GPU column, and joining "no GPU" onto the memory with a comma would
    leave a translator with a fragment instead of a sentence.
    """
    profile = loaded.profile
    ram = _size(profile.memory.total)
    gpu = profile.primary_gpu
    if gpu is None:
        return _("%(ram)s RAM, no graphics card") % {"ram": ram}
    if gpu.vram_total is None:
        return _("%(ram)s RAM shared with %(gpu)s") % {"ram": ram, "gpu": isolate(gpu.name)}
    return _("%(ram)s RAM, %(gpu)s with %(vram)s") % {
        "ram": ram,
        "gpu": isolate(gpu.name),
        "vram": _size(gpu.vram_total),
    }


def _profile_cpu(cpu: ProfileCpu) -> str:
    """A profile's processor as one phrase: model, cores, threads, instruction sets."""
    threads = cpu.logical_cores or cpu.physical_cores
    cores = ngettext(
        "%(cores)d core / %(threads)d thread",
        "%(cores)d cores / %(threads)d threads",
        cpu.physical_cores,
    ) % {"cores": cpu.physical_cores, "threads": threads}
    isa = isolate(" ".join(cpu.isa)) if cpu.isa else pgettext("CPU instruction sets", "isa unknown")
    return _("%(model)s; %(cores)s; %(isa)s") % {
        "model": isolate(cpu.model),
        "cores": cores,
        "isa": isa,
    }


def _profile_memory(memory: ProfileMemory) -> str:
    """A profile's memory pool as one phrase, its bandwidth carrying its label.

    A profile that states no bandwidth says so rather than showing a blank: an absent
    figure means section 10.1's fallback stands in, which is a different claim from a
    figure somebody wrote down, and the reader is entitled to know which they have.
    """
    if memory.bandwidth_gbps is None:
        bandwidth = pgettext("memory bandwidth", "unknown")
    else:
        bandwidth = _("%(gbps)s GB/s (%(source)s)") % {
            "gbps": localise_number(str(memory.bandwidth_gbps)),
            "source": _bandwidth_source_label(memory.bandwidth_source or "unknown"),
        }
    return _("%(total)s total, %(available)s available; bandwidth %(bandwidth)s") % {
        "total": _size(memory.total),
        "available": _size(memory.total if memory.available is None else memory.available),
        "bandwidth": bandwidth,
    }


def _profile_gpu(card: ProfileGpu) -> str:
    """A profile's graphics card as one phrase, saying where its specifications come from.

    A card that states no bandwidth or compute figure gets both from the bundled table by
    name, exactly as a scanned one does, and the line says ``from the specification
    table`` so nobody reads a vendor's number as this machine's measurement.
    """
    spec = lookup_gpu(card.name)
    stated = card.bandwidth_gbps is not None or card.compute_tflops_fp16 is not None
    bandwidth = card.bandwidth_gbps or (spec.bandwidth_gbps if spec else None)
    compute = card.compute_tflops_fp16 or (spec.compute_tflops_fp16 if spec else None)
    if bandwidth is None and compute is None:
        specs = pgettext("GPU specifications", "no bandwidth or compute figure")
    elif stated:
        specs = _("%(specs)s (from the profile)") % {"specs": _from_table(bandwidth, compute)}
    else:
        specs = _("%(specs)s (from the specification table)") % {
            "specs": _from_table(bandwidth, compute)
        }
    vram = (
        pgettext("GPU VRAM", "VRAM from the shared pool")
        if card.vram_total is None
        else _("%(total)s VRAM, %(free)s free")
        % {
            "total": _size(card.vram_total),
            "free": _size(max(card.vram_total - card.vram_used, 0)),
        }
    )
    return _("%(name)s (%(backend)s); %(vram)s; %(specs)s") % {
        "name": isolate(card.name),
        "backend": isolate(card.backend),
        "vram": vram,
        "specs": specs,
    }


def render_profiles(profiles: Sequence[LoadedProfile]) -> Table:
    """Every hardware profile LlamaFit can reach, bundled ones first.

    The name is the one cell on this screen a reader has to retype, into
    ``--profile``, so it is the one cell that must never be shortened. At eighty
    columns Rich was giving every column an equal share and the only bundled profile
    arrived as ``reference-rtx4060-1…``. It folds now, and the two columns that are
    prose give way for it: a description that wraps is read the same, and a name that
    wraps is still a name, while a name with a full stop in the middle of it is a
    command that does not run.
    """
    table = Table(title=for_display(_("Hardware profiles")))
    _add_columns(
        table,
        [
            {"header": _("Name"), "overflow": "fold", "ratio": None, "no_wrap": False},
            {"header": _("From"), "no_wrap": True},
            {"header": _("Machine"), "ratio": 2},
            {"header": _("Description"), "ratio": 2},
        ],
    )
    for loaded in profiles:
        _add_row(
            table,
            _cell(isolate(loaded.name)),
            _profile_origin(loaded),
            _cell(_profile_machine(loaded)),
            _cell(loaded.profile.description or ""),
        )
    return table


def _profile_match_rules(loaded: LoadedProfile) -> str:
    """The rules that would let a live scan recognise itself in this profile."""
    rules = loaded.profile.match
    parts: list[str] = []
    if rules.gpu_name_contains is not None:
        parts.append(
            _("a graphics card whose name contains %(text)s")
            % {"text": isolate(repr(rules.gpu_name_contains))}
        )
    if rules.cpu_model_contains is not None:
        parts.append(
            _("a processor whose model contains %(text)s")
            % {"text": isolate(repr(rules.cpu_model_contains))}
        )
    if rules.total_ram_min is not None:
        parts.append(_("at least %(ram)s of memory") % {"ram": _size(rules.total_ram_min)})
    if not parts:
        return pgettext("profile match rules", "none, so this profile is never chosen for you")
    return ", ".join(parts)


def render_profile(loaded: LoadedProfile) -> Table:
    """One profile in full, provenance included.

    The provenance line is not decoration. Every other row is a figure, and a figure
    somebody typed into a file is worth exactly what its source is worth; this is the
    row that says what that source was.
    """
    profile = loaded.profile
    table = Table(
        title=for_display(isolate(profile.name)), show_header=False, box=None, pad_edge=False
    )
    _add_columns(table, [{"header": "key", "style": "bold"}, {"header": "value"}])
    if profile.description:
        _add_row(table, _("Description"), _cell(profile.description))
    _add_row(table, _("From"), _profile_origin(loaded))
    _add_row(table, _("File"), _cell(isolate(str(loaded.path))))
    _add_row(
        table,
        _("OS"),
        _cell(
            _("%(os)s %(version)s (%(arch)s)")
            % {
                "os": isolate(profile.os),
                "version": isolate(profile.os_version),
                "arch": isolate(profile.arch),
            }
        ),
    )
    _add_row(table, _("CPU"), _cell(_profile_cpu(profile.cpu)))
    _add_row(table, _("Memory"), _cell(_profile_memory(profile.memory)))
    if not profile.gpus:
        _add_row(table, _("GPU"), pgettext("GPU", "none detected"))
    for index, card in enumerate(profile.gpus):
        _add_row(table, _("GPU %(index)d") % {"index": index}, _cell(_profile_gpu(card)))
    if profile.unified_memory:
        _add_row(table, _("Memory pool"), _("unified (GPU shares system memory)"))
    if profile.backends:
        _add_row(table, _("Backends"), _cell(isolate(", ".join(profile.backends))))
    _add_row(table, _("Matches"), _cell(_profile_match_rules(loaded)))
    if profile.recorded_at is not None:
        _add_row(table, _("Recorded"), _cell(isolate(profile.recorded_at.isoformat())))
    _add_row(table, _("Provenance"), _cell(profile.provenance))
    if profile.calibration is not None:
        _add_row(
            table,
            _("Calibration"),
            _cell(
                _("recorded from %(source)s; this build stores it but does not apply it yet")
                % {"source": isolate(profile.calibration.source)}
            ),
        )
    return table
