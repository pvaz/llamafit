# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The two lines that are always on screen: what this machine is, and what a speed is.

A dashboard has one line of chrome at the top and everything else is a table. Whatever
goes in that line is the only thing every screen says, so it has to be the two things a
reader must not be allowed to forget: which machine these figures are about, and what
kind of number a speed is.

The second is the reason this file exists at all. Nothing has been benchmarked on any
machine yet -- section 10.3 reserves ``measured`` for a run taken *here*, and phase 3 is
what will store one -- so every speed on the board today is a formula on default
constants. The command line says that in a caption under the table.  A caption under a
table is a footnote, and a footnote is a thing nobody reads, so on this screen the same
sentence is a band of its own that does not scroll away.

The sentence is not written here. :func:`llamafit.cli.render_board.confidence_sentence`
already holds one per label, they are already translated into thirty-seven catalogs, and
an interface that wrote its own would be one edit away from the two screens disagreeing
about how much a number is worth.
"""

from __future__ import annotations

from collections.abc import Sequence

from llamafit.cli.render_board import confidence_sentence
from llamafit.i18n import _, isolate, ngettext, pgettext, pgettext_literal
from llamafit.models.host import Host
from llamafit.models.llamacpp import LlamaCpp
from llamafit.services.recommend import BoardRow
from llamafit.tui.format import size


def separator() -> str:
    """What comes between two parts of the header line.

    Punctuation rather than prose, which is why it is a literal entry: a language that
    parts a list with something other than a middle dot -- or that wants a space on only
    one side of it -- has to be able to say so, and a separator hidden in a Python join
    is one a translator never sees.
    """
    return pgettext_literal("dashboard header separator", " · ")


def gpu_phrase(host: Host) -> str:
    """The card and what is free on it, or the words for a machine that has neither.

    Three shapes rather than a stem with clauses bolted on. A machine with no card is not
    a machine whose card has an unknown size, and neither is a machine whose card is
    known down to the byte; a reader deciding whether to trust the board's verdicts needs
    to be able to tell those three apart at a glance.
    """
    gpu = host.primary_gpu
    if gpu is None:
        return pgettext("dashboard header", "no GPU detected")
    if gpu.vram_total_bytes is None or gpu.vram_free_bytes is None:
        return _("%(name)s, VRAM unknown") % {"name": isolate(gpu.name)}
    return _("%(name)s, %(free)s free of %(total)s") % {
        "name": isolate(gpu.name),
        "free": size(gpu.vram_free_bytes),
        "total": size(gpu.vram_total_bytes),
    }


def memory_phrase(host: Host) -> str:
    """System memory: what is free, out of what there is."""
    return _("%(available)s free of %(total)s RAM") % {
        "available": size(host.memory.available_bytes),
        "total": size(host.memory.total_bytes),
    }


def cpu_phrase(host: Host) -> str:
    """The processor and how many cores it has, as one counted phrase."""
    return ngettext(
        "%(model)s, %(count)d core",
        "%(model)s, %(count)d cores",
        host.cpu.physical_cores,
    ) % {"model": isolate(host.cpu.model), "count": host.cpu.physical_cores}


def llamacpp_phrase(llamacpp: LlamaCpp) -> str:
    """Whether llama.cpp is here and which build, in the width of a header cell.

    A build number keeps its ``b`` glued to its digits inside one placeholder-free
    message, for the reason :func:`llamafit.cli.render_llamacpp` gives: isolating the
    digits alone would part the letter from the number and let the two swap places in an
    Arabic line.
    """
    if not llamacpp.installed:
        return pgettext("dashboard header", "llama.cpp not installed")
    if llamacpp.build is None:
        return pgettext("dashboard header", "llama.cpp installed")
    return _("llama.cpp b%(build)d") % {"build": llamacpp.build}


def machine_line(host: Host | None, llamacpp: LlamaCpp | None) -> str:
    """The header's machine line: the card, the memory, the processor, llama.cpp.

    Before the scan has come back there is nothing true to say, so it says that rather
    than drawing an empty line the reader would take for a machine with nothing in it.
    """
    if host is None:
        return pgettext("dashboard header", "scanning this machine…")
    parts = [gpu_phrase(host), memory_phrase(host), cpu_phrase(host)]
    if llamacpp is not None:
        parts.append(llamacpp_phrase(llamacpp))
    return separator().join(parts)


def simulated_badge(host: Host | None) -> str:
    """``SIMULATED`` when these figures are not this machine, and nothing when they are.

    The same word :func:`llamafit.cli.render.render_host` puts in a simulated host's
    first row, so the two interfaces mark a substituted machine with one word between
    them.
    """
    return _("SIMULATED") if host is not None and host.simulated else ""


def speed_band(rows: Sequence[BoardRow]) -> str:
    """What kind of number the speed column holds, said where it cannot be scrolled past.

    One label across every row means one sentence, and it is the command line's own. Rows
    that disagree mean no single sentence is true of the table, and then the only honest
    thing to say is that the column beside each figure is where its label is.
    """
    labels = {row.candidate.speed.confidence for row in rows if row.candidate.speed is not None}
    if not labels:
        return ""
    if len(labels) == 1:
        return confidence_sentence(next(iter(labels)))
    return _(
        "These speeds were not all arrived at the same way; the How column says which each row is."
    )
