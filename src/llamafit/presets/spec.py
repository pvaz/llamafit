# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Everything a launch script needs, and the one decision it makes for itself.

A :class:`PresetSpec` is a plan frozen into the shape a script can be rendered from: the
paths, the flags, the endpoint, what the machine looked like when the plan was made, and
the context ladder. Both renderers read it, and neither reads a
:class:`~llamafit.models.plan.Placement`, so the Windows script and the POSIX script
cannot drift apart in what they were told.

The ladder is the reason this package exists at all. Section 15.3 asks for "a context tier
chosen from live free VRAM at start", and :func:`choose_context` is that choice, written
once here as arithmetic on integers. The scripts do not reimplement it; they unroll it,
rung by rung, into their own dialect. That is what lets the rule be tested as a rule --
give it a free-memory figure, get a context back -- separately from testing that each
script spells the same rule correctly.

Two rules shape the ladder itself, and both are about not being clever.

**It only ever steps down.** The top rung is the context the planner chose, and no rung
above it is offered even when the tier table says one would fit. The planner weighed system
memory, the user's request and the mode when it chose; the script measures one pool, the
card, and a script that promoted itself to a longer context on the strength of the only
number it can see would be sizing against pools it never looked at.

**A rung the machine could not hold as scanned is not on it.** ``fits`` is false for a rung
that overflows the card *or* system memory, and the script cannot tell those apart from
free VRAM alone. Leaving them off costs nothing -- they were not reachable anyway -- and
keeps every rung on the ladder one the planner has already said yes to.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from llamafit.constants import MIB, VRAM_RESERVE_BYTES
from llamafit.models.catalog import CatalogModel
from llamafit.models.host import Host, OsName, Vendor
from llamafit.models.plan import Placement, RunMode

CONTEXT_TOKEN = "@@LLAMAFIT_CONTEXT@@"
"""What stands in for the context in :attr:`PresetSpec.flags` while a script is rendered.

The flags are rendered once, by :mod:`llamafit.placement.flags`, in the order section 9.5
fixes. A script that rebuilt them to substitute its own ``-c`` would be a second renderer
of the same command line and would eventually disagree with the first. So the value after
``-c`` is swapped for this marker and each script replaces the marker with its own way of
naming a variable -- ``%CTX%`` or ``"$CTX"`` -- and nothing else about the line moves.
"""


def mib_ceiling(byte_count: int) -> int:
    """``byte_count`` in whole mebibytes, rounded up.

    Up, never down: every figure here is a memory requirement, and a requirement rounded
    down is a requirement understated. The scripts compare in mebibytes because that is
    the unit ``nvidia-smi`` answers in, and a comparison done in the units of the
    measurement needs no floating point in a batch file.
    """
    return -(-byte_count // MIB)


class Rung(BaseModel):
    """One step of the ladder a script chooses from: a context and what it costs the card.

    Attributes:
        tokens: The context length.
        vram_required: What that context needs on the graphics card, in bytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: int = Field(gt=0)
    vram_required: int = Field(ge=0)

    @property
    def vram_required_mib(self) -> int:
        """What this rung needs on the card, in the unit the scripts compare in."""
        return mib_ceiling(self.vram_required)


def choose_context(rungs: Sequence[Rung], free_mib: int, *, reserve_mib: int) -> int | None:
    """The largest rung that fits under ``free_mib``, or ``None`` when none does.

    Args:
        rungs: The ladder, largest context first.
        free_mib: Card memory free right now, in mebibytes.
        reserve_mib: What to hold back for whatever else the desktop is doing.

    Returns:
        A context length, or ``None`` when even the smallest rung needs more than is free.

    ``None`` is a real answer and not a failure to produce one. A script that has been
    handed it must stop and say so: the alternative is to start anyway at a rung that does
    not fit, which on an NVIDIA driver does not fail but pages into system memory, and a
    server that runs at a third of its speed while its log looks healthy is the one outcome
    this whole project exists to prevent.
    """
    usable = free_mib - reserve_mib
    for rung in rungs:
        if rung.vram_required_mib <= usable:
            return rung.tokens
    return None


class PresetSpec(BaseModel):
    """A plan in the shape a launch script is rendered from.

    Attributes:
        model_id: The catalog id, which is also the alias the server answers to.
        name: The model's display name, for the header a person reads.
        quant: The quantisation planned.
        mode: How the weights are divided, for the header.
        planned_context: The context the planner chose, and the top of the ladder.
        rungs: The ladder, largest first. Never empty, and never above ``planned_context``.
        reserve_mib: Held back from free card memory before the ladder is walked.
        server_path: The ``llama-server`` this machine has, or ``None`` when it has none
            and the script must find one on ``PATH``.
        server_program: The program name to look for on ``PATH``, with ``.exe`` on Windows.
        model_path: The GGUF file, or where ``llamafit install model`` would put it.
        projector_path: The vision projector the flags name, when they name one.
        flags: The ``llama-server`` arguments with :data:`CONTEXT_TOKEN` where the context
            goes, so a script substitutes one value and rebuilds nothing.
        host_address: The address the server binds. Loopback; llama-server has no
            authentication.
        port: The port the server binds.
        probe_free_memory: Whether the script should read live free card memory at all.
            False when the placement puts nothing on a card, where there is no ladder to
            walk and a probe would only be a way to fail.
        gpu_index: Which device to ask about, as the vendor tool numbers them.
        gpu_vendor: Which vendor tool the script should reach for.
        gpu_name: The card the plan was made for, so the script can notice another one.
        gpu_total_mib: That card's size, which is what the script actually compares.
        gpu_free_at_plan_mib: What was free when the plan was made, for the header. This is
            the number the whole ladder exists because of: it was true once.
        unified_memory: Whether card memory and system memory are the same pool, which
            changes how a script has to ask what is free.
        os_name: The machine the script is written for.
        llamacpp_build: The llama.cpp build the flags were rendered against, named in the
            script so an upgrade that rejects a flag has an explanation waiting.
        generated: The day the script was written.
        version: The LlamaFit that wrote it.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    name: str
    quant: str
    mode: RunMode
    planned_context: int = Field(gt=0)
    rungs: tuple[Rung, ...] = Field(min_length=1)
    reserve_mib: int = Field(ge=0)
    server_path: str | None = None
    server_program: str = "llama-server"
    model_path: str
    projector_path: str | None = None
    flags: tuple[str, ...]
    host_address: str
    port: int
    probe_free_memory: bool
    gpu_index: int | None = None
    gpu_vendor: Vendor | None = None
    gpu_name: str | None = None
    gpu_total_mib: int | None = None
    gpu_free_at_plan_mib: int | None = None
    unified_memory: bool = False
    os_name: OsName
    llamacpp_build: int | None = None
    generated: date
    version: str

    @property
    def script_stem(self) -> str:
        """What the generated files are called, without an extension."""
        return f"start-{self.model_id}"

    @property
    def endpoint(self) -> str:
        """The base URL the server answers on."""
        return f"http://{self.host_address}:{self.port}"

    @property
    def smallest_rung(self) -> Rung:
        """The bottom of the ladder, which is what a script reports when nothing fits."""
        return self.rungs[-1]

    @property
    def card_sentence(self) -> str:
        """One line describing the card as it was when the plan was made.

        Written here rather than in each renderer, because the two scripts saying slightly
        different things about the same machine is exactly the drift this module exists to
        stop, and a header is the first thing anyone reads.
        """
        if self.gpu_name is None:
            return "No graphics card was detected on this machine."
        parts = [f"Card: {self.gpu_name}"]
        if self.gpu_total_mib is not None:
            parts.append(f"{self.gpu_total_mib} MiB total")
        if self.gpu_free_at_plan_mib is not None:
            parts.append(f"{self.gpu_free_at_plan_mib} MiB free at plan time")
        return ", ".join(parts) + "."

    @property
    def build_phrase(self) -> str:
        """``, llama.cpp build N`` when a build was read, and nothing when it was not."""
        return "" if self.llamacpp_build is None else f", llama.cpp build {self.llamacpp_build}"


def can_probe_free_memory(vendor: Vendor | None, os_name: OsName) -> bool:
    """Whether a script on this machine has any way to ask how much memory is free.

    Args:
        vendor: The primary card's vendor, or ``None`` when there is no card.
        os_name: The machine the script will run on.

    Returns:
        True when a shell can read a free-memory figure without installing anything.

    The three yeses are the three tools that are already there when the driver is: NVIDIA's
    ``nvidia-smi`` on every platform, amdgpu's sysfs files on Linux, and ``vm_stat`` on
    macOS, where the graphics memory is the system memory and so the question is a
    different one with the same answer. Everything else -- an AMD card on Windows, Intel
    anywhere -- has no such tool, and this returning False is what makes the generated
    script say so once, in its header, instead of printing the same apology on every run.
    """
    if vendor == "nvidia":
        return True
    if vendor == "amd":
        return os_name == "linux"
    if vendor == "apple":
        return os_name == "macos"
    return False


MESSAGE_LABEL = "[llamafit] "
"""What every line a generated script prints at run time opens with.

So that a person reading a console can tell what came from the launch script apart from
what came from llama.cpp, which prints a great deal.
"""


def message_block(paragraphs: Sequence[str], *, width: int = 68) -> list[str]:
    """Run-time messages wrapped to one width and labelled, for either dialect to echo.

    Args:
        paragraphs: What to say. A paragraph opening with a space is passed through
            unwrapped, which is how an indented command keeps its indent.
        width: How wide the text may be, not counting the label.

    Returns:
        The lines, each already labelled and none of them wrapped mid-command.
    """
    lines: list[str] = []
    for paragraph in paragraphs:
        if paragraph.startswith(" "):
            lines.append(MESSAGE_LABEL + paragraph)
            continue
        lines += [
            MESSAGE_LABEL + line
            for line in textwrap.wrap(
                paragraph, width=width, break_long_words=False, break_on_hyphens=False
            )
        ]
    return lines


def section(prefix: str, title: str, *, width: int = 78) -> str:
    """A section heading padded to one width, so a generated file looks laid out.

    Args:
        prefix: The comment marker with its trailing space, ``"rem "`` or ``"# "``.
        title: What the section is.
        width: The column the rule runs to.

    Returns:
        One comment line.

    Counting the dashes by hand is how a file ends up with headings of five different
    lengths, which is a small thing that makes a generated file look generated.
    """
    head = f"{prefix}--- {title} "
    return head + "-" * max(width - len(head), 3)


def comment_block(prefix: str, paragraphs: Sequence[str], *, width: int = 74) -> list[str]:
    """Wrap prose into comment lines of an even width, for a header a person will read.

    Args:
        prefix: What opens each line, including its trailing spaces -- ``"rem  "`` or
            ``"#  "``.
        paragraphs: The paragraphs, separated in the output by a bare comment line.
        width: How wide the text may be, not counting the prefix.

    Returns:
        The lines, ready to be joined.

    Long words are never broken. A path, a flag or a command like
    ``llamafit preset qwen3-coder-next`` split across two comment lines is a thing a
    reader cannot copy, and a header exists to be copied from.
    """
    lines: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if index:
            lines.append(prefix.rstrip())
        lines += [
            prefix + line
            for line in textwrap.wrap(
                paragraph, width=width, break_long_words=False, break_on_hyphens=False
            )
        ]
    return lines


def why_a_ladder(spec: PresetSpec) -> list[str]:
    """The paragraphs the whole package exists for, written once for both scripts.

    Here rather than in either renderer because it is the same argument in both files, and
    two copies of an explanation drift the moment one of them is improved. Each renderer
    wraps it behind its own comment marker with :func:`comment_block`.
    """
    free = spec.gpu_free_at_plan_mib
    when = f"while the card had {free} MiB free" if free is not None else "on an idle machine"
    pool = "system memory" if spec.unified_memory else "the graphics card"
    return [
        "WHY THIS SCRIPT CHOOSES ITS CONTEXT WHEN IT RUNS",
        f"The placement above was computed {when}. By the time you run this, a browser, a "
        f"game or another model may be holding a gigabyte of {pool}. Ask llama.cpp for "
        "more of it than is free and the driver does not refuse the allocation: it pages "
        "the overflow into system memory. The server starts, /health answers, the log "
        "looks healthy, and generation runs at a fraction of its speed -- for weeks, if "
        "nobody thinks to measure it.",
        "So the context is not written in. The ladder below lists each context this plan "
        "can run at and what it costs. The script reads what is free at the moment it "
        f"runs, keeps {spec.reserve_mib} MiB back for everything else, and takes the "
        "largest rung that still fits underneath.",
        f"It never climbs above the planned {spec.planned_context}. The planner weighed "
        "system memory and what you asked for as well; this script can measure one pool, "
        "and sizing against the pool you can see is how the other one overflows.",
    ]


def why_no_ladder(spec: PresetSpec) -> list[str]:
    """The paragraphs for a machine where no ladder can be walked, and which reason it is.

    Two reasons, and they are not the same news. A placement that puts nothing on a card
    has nothing to step down for, which is fine. A card nobody publishes a tool for means
    the context below is a number that was true once and nothing will notice when it stops
    being true, which the reader had better know.
    """
    if spec.gpu_vendor is None or spec.mode == "cpu":
        return [
            "WHY THIS SCRIPT DOES NOT CHOOSE A CONTEXT WHEN IT RUNS",
            f"This placement ({spec.mode}) keeps no weights and no cache on a graphics "
            "card, so there is no free-card-memory figure for a ladder to read and "
            f"nothing to step down for. The context is the planned {spec.planned_context}, "
            f"system memory is what limits it, and `llamafit plan {spec.model_id}` knows "
            "how far it goes.",
            "Set LLAMAFIT_CONTEXT to run at another one for a single run.",
        ]
    return [
        "WHY THIS SCRIPT DOES NOT CHOOSE A CONTEXT WHEN IT RUNS",
        f"LlamaFit has no way to read free memory from a {spec.gpu_vendor} card on "
        f"{spec.os_name}: there is no vendor tool a script can call for it here. So the "
        f"context below is the planned {spec.planned_context}, and that is a number that "
        f"was true on {spec.generated.isoformat()} rather than one this script works out "
        "for itself.",
        "That is worth knowing. If something else is holding the card when you start, the "
        "driver may page part of this into system memory and generation will be slow "
        f"while the log looks healthy. `llamafit plan {spec.model_id}` says what fits "
        "today, and LLAMAFIT_CONTEXT sets a smaller context for a single run.",
    ]


def nothing_fits(spec: PresetSpec, *, free_memory: str) -> list[str]:
    """What a script says when even its smallest rung needs more than is free.

    Args:
        spec: The plan.
        free_memory: How this dialect spells the variable holding the free figure --
            ``%FREE_MIB%`` or ``$FREE_MIB``.

    Returns:
        The paragraphs to print before stopping.

    Stopping is the point of the message. The alternative is to start at a rung that does
    not fit, which does not fail: it pages, and runs at a fraction of the speed while the
    log looks healthy.
    """
    smallest = spec.smallest_rung
    return [
        f"Only {free_memory} MiB of card memory is free. The smallest configuration in "
        f"this plan needs {smallest.vram_required_mib} MiB, plus the {spec.reserve_mib} "
        "MiB this script keeps back for everything else.",
        "Starting anyway would page into system memory and run at a fraction of the "
        "speed, so this script stops here instead.",
        "Close whatever is holding the card, or see what fits today:",
        f"  llamafit plan {spec.model_id}",
    ]


def cannot_measure(spec: PresetSpec) -> list[str]:
    """What a script says when the free-memory probe gave it nothing to work with.

    It falls back to the planned context and says so out loud, which is the honest end of
    a bad choice between two. Falling back to the smallest rung would leave every machine
    whose tool failed once running at the shortest context for ever; falling back silently
    to the planned one would be the baked-in number this package exists to remove. So it
    uses the number that was true when the plan was made, and tells the reader that is what
    it did.
    """
    return [
        "Could not read how much card memory is free, so this falls back to the planned "
        f"{spec.planned_context} tokens, which fit when this file was written on "
        f"{spec.generated.isoformat()}.",
        "If the card is busier now, the driver will page and generation will be slow. "
        "What fits today:",
        f"  llamafit plan {spec.model_id}",
    ]


def why_it_is_yours(spec: PresetSpec) -> list[str]:
    """The paragraph that tells a reader the file is theirs and what protects it."""
    return [
        "Yours to edit -- that is the point of writing a file rather than printing a "
        "command. LlamaFit will not overwrite this once you have changed it: the stamp "
        "below is a checksum of everything else in the file, and a stamp that no longer "
        f"matches makes `llamafit preset {spec.model_id}` keep your version and say so.",
    ]


def group_flags(words: Sequence[str]) -> list[list[str]]:
    """Pair each flag with the value that belongs to it, one pair to a rendered line.

    Args:
        words: The rendered argument list, already quoted for its shell.

    Returns:
        Groups of one or two words: a flag with its value where it has one, and a lone
        word otherwise.

    The scripts print one group per line so that a person can read the command, and delete
    or change one flag, without counting words. The rule is the command line's own: a word
    that does not open with ``-`` belongs to the flag before it.
    """
    groups: list[list[str]] = []
    for word in words:
        previous = groups[-1] if groups else None
        opens_a_pair = previous is not None and len(previous) == 1 and not _is_value(previous[0])
        if opens_a_pair and _is_value(word):
            groups[-1].append(word)
            continue
        groups.append([word])
    return groups


def _is_value(word: str) -> bool:
    """Whether a word reads as a value rather than as a flag.

    A quoted word is always a value: only :mod:`llamafit.placement.flags` produces the
    flags, and it never quotes one, so anything the renderers wrapped in quotes or turned
    into a variable reference is something a flag was waiting for.
    """
    return not word.startswith("-") or word == "-"


def _ladder(placement: Placement) -> tuple[Rung, ...]:
    """The rungs a script may take, largest first, topped by the planned configuration.

    The planned context is put on the ladder from the placement's own budget rather than
    looked up among the tiers, because it need not be a tier at all: ``plan --context
    50000`` is a context nobody's ladder contains, and a preset written for it that
    silently started at 49,152 would be a preset for a different plan.
    """
    rungs = [
        Rung(tokens=tier.tokens, vram_required=tier.vram_required)
        for tier in placement.tiers
        if tier.fits and tier.tokens < placement.context
    ]
    rungs.append(Rung(tokens=placement.context, vram_required=placement.budget.vram_required))
    return tuple(sorted(rungs, key=lambda rung: rung.tokens, reverse=True))


def build_spec(
    *,
    model: CatalogModel,
    quant: str,
    placement: Placement,
    flags: Sequence[str],
    model_path: str,
    projector_path: str | None,
    host: Host,
    server_path: str | None,
    llamacpp_build: int | None,
    port: int,
    generated: date,
    version: str,
) -> PresetSpec:
    """Freeze one plan into the shape both renderers read.

    Args:
        model: The catalog entry.
        quant: The quantisation planned.
        placement: Where the bytes go, with the context ladder.
        flags: The rendered ``llama-server`` arguments, in section 9.5's order.
        model_path: The GGUF file the flags name.
        projector_path: The vision projector the flags name, when there is one.
        host: The machine as it was scanned, which is what the script later checks itself
            against.
        server_path: The ``llama-server`` binary found on this machine, if one was.
        llamacpp_build: Its build number, if it was read.
        port: The port to bind.
        generated: The day this is being written.
        version: The LlamaFit writing it.

    Returns:
        The specification, with the context in ``flags`` replaced by
        :data:`CONTEXT_TOKEN`.
    """
    gpu = host.primary_gpu
    on_card = placement.mode != "cpu" and placement.gpu_layers > 0
    free_at_plan = host.vram_available_bytes
    return PresetSpec(
        model_id=model.id,
        name=model.name,
        quant=quant,
        mode=placement.mode,
        planned_context=placement.context,
        rungs=_ladder(placement),
        reserve_mib=mib_ceiling(VRAM_RESERVE_BYTES),
        server_path=server_path,
        server_program="llama-server.exe" if host.os == "windows" else "llama-server",
        model_path=model_path,
        projector_path=projector_path,
        flags=_with_context_token(flags),
        host_address=_flag_value(flags, "--host") or "127.0.0.1",
        port=port,
        probe_free_memory=(
            on_card and gpu is not None and can_probe_free_memory(gpu.vendor, host.os)
        ),
        gpu_index=None if gpu is None else gpu.index,
        gpu_vendor=None if gpu is None else gpu.vendor,
        gpu_name=None if gpu is None else gpu.name,
        gpu_total_mib=(
            None if gpu is None or gpu.vram_total_bytes is None else gpu.vram_total_bytes // MIB
        ),
        gpu_free_at_plan_mib=None if free_at_plan is None else free_at_plan // MIB,
        unified_memory=host.unified_memory,
        os_name=host.os,
        llamacpp_build=llamacpp_build,
        generated=generated,
        version=version,
    )


def _with_context_token(flags: Sequence[str]) -> tuple[str, ...]:
    """The flags with the value after ``-c`` replaced by :data:`CONTEXT_TOKEN`.

    A command line with no ``-c`` at all is left alone rather than repaired: it would mean
    section 9.5's renderer had changed under this module, and quietly inserting a flag the
    planner did not render is how a script comes to launch a configuration nobody costed.
    """
    out = list(flags)
    for index, flag in enumerate(out[:-1]):
        if flag == "-c":
            out[index + 1] = CONTEXT_TOKEN
            break
    return tuple(out)


def _flag_value(flags: Sequence[str], name: str) -> str | None:
    """The value that follows ``name`` in a rendered argument list, or ``None``."""
    for index, flag in enumerate(flags[:-1]):
        if flag == name:
            return flags[index + 1]
    return None
