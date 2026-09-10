"""Turn a placement into the ``llama-server`` arguments a person will actually paste.

This is the end of the whole pipeline, and the place where a small mistake costs the most.
Everything before it can be re-read and argued with; a flag list is copied into a terminal
by somebody who has already decided to trust it, and a flag that is subtly wrong fails at
exactly that moment. So the order is section 9.5's order, written once, and the tests
check it against the flags the reference machine's own measured configurations recorded.

Two rules follow from that and are worth stating.

Nothing here is ever localised. Every other number LlamaFit prints goes through
:func:`~llamafit.units.localise_number`, which would write ``--temp 0,95`` for a Portuguese
reader and hand llama.cpp a number it does not parse. A command line is not prose.

A default llama.cpp already applies is not rendered. ``-ctk``/``-ctv`` appear only for a
cache that is not ``f16``, because the recorded configurations do not name the default
either, and a rendered line that matches the measured one is a line that can be compared
with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from llamafit.constants import (
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    KV_TYPE_DEFAULT,
)
from llamafit.models.catalog import CatalogModel
from llamafit.models.plan import Placement

_SERVER_EXECUTABLE = "llama-server"

_LAZY_MODE_VALUES = frozenset({"recommended", "required", "yes", "true", "on"})
"""What ``llama_cpp.requires["lazy_mode"]`` has to say for ``--lazy-mode`` to be rendered.

Section 9.5 renders the flag "when the catalog recommends it", and section 6.2's entry
schema is where the catalog says so: ``requires: {mmproj: optional, lazy_mode: recommended}``.
"""


@dataclass(frozen=True)
class LaunchOptions:
    """Everything a command line needs that a :class:`~llamafit.models.plan.Placement` does not.

    A placement says where the bytes go; it does not know where the files are, what the
    server should be called or which port is free. Those come from here, so the same
    placement can be rendered for a preset, for the ``plan`` output and for a benchmark
    run without any of them inventing paths.

    Attributes:
        model_path: The GGUF file, or the first shard of a split one.
        projector_path: The vision projector file, when the model has one. Supplying it
            is what lets the renderer say ``--no-mmproj`` when the placement left vision
            out, which is the difference between vision being off and being forgotten.
        alias: What the server calls this model; the catalog id when left unset.
        host: The address to bind. Loopback by default, and deliberately: llama-server
            has no authentication.
        port: The port to bind.
        tensor_overrides: ``-ot`` patterns, each one already in llama.cpp's
            ``pattern=POOL`` form, for example ``ffn_.*_shexp=CPU``. The reference
            machine's winning configuration is exactly this flag, so it is a first-class
            option rather than something to be smuggled in through ``extra_args``.
        lazy_mode: Render ``--lazy-mode``; ``None`` follows the catalog's
            ``llama_cpp.requires``.
        thinking: Turn the model's thinking mode on or off through its chat template;
            ``None`` leaves the template's own default alone.
        cache_ram_mib: ``--cache-ram``, in MiB. Left unset, the flag is not rendered and
            llama.cpp's own default applies: pinning a number nobody has chosen would
            change how much system memory the server takes without anybody asking.
        parallel: ``-np``. One slot, because every budget in section 8 is computed for
            one sequence.
        extra_args: Anything else, appended last so a caller can always have the final
            word.
    """

    model_path: str
    projector_path: str | None = None
    alias: str | None = None
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT
    tensor_overrides: tuple[str, ...] = ()
    lazy_mode: bool | None = None
    thinking: bool | None = None
    cache_ram_mib: int | None = None
    parallel: int = DEFAULT_PARALLEL_SLOTS
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def render_flags(placement: Placement, model: CatalogModel, options: LaunchOptions) -> list[str]:
    """Render one placement as ``llama-server`` arguments, in section 9.5's order.

    Args:
        placement: Where the bytes go.
        model: The catalog entry, which supplies the alias, the sampling parameters, the
            chat-template options and the quirks.
        options: The paths, the endpoint and everything else a placement does not know.

    Returns:
        The argument list, without the program name. Use :func:`command_line` for the
        whole thing.
    """
    args: list[str] = ["-m", options.model_path]
    args += _projector_flags(placement, options)
    args += ["--alias", options.alias or model.id]
    args += ["--host", options.host, "--port", str(options.port)]
    args += ["-c", str(placement.context)]
    args += ["-fa", "on"]
    if placement.kv_type != KV_TYPE_DEFAULT:
        args += ["-ctk", placement.kv_type, "-ctv", placement.kv_type]
    args += ["-ngl", str(placement.gpu_layers)]
    if placement.cpu_moe_layers is not None:
        args += ["--n-cpu-moe", str(placement.cpu_moe_layers)]
    for override in options.tensor_overrides:
        args += ["-ot", override]
    # Always: llama.cpp's own automatic placement was measured at half the speed of the
    # planned one on the reference machine (11.9 tokens per second against 24.7), and a
    # plan whose numbers are computed for one placement must not be launched into another.
    args += ["--fit", "off"]
    if _wants_lazy_mode(model, options):
        args += ["--lazy-mode"]
    args += ["-t", str(placement.threads), "-tb", str(placement.threads)]
    args += ["-b", str(placement.batch), "-ub", str(placement.micro_batch)]
    args += _sampling_flags(model)
    args += ["--jinja"]
    args += _reasoning_flags(model, options)
    args += ["-np", str(options.parallel)]
    if options.cache_ram_mib is not None:
        args += ["--cache-ram", str(options.cache_ram_mib)]
    args += quirk_arguments(model.llama_cpp.quirks)
    args += list(options.extra_args)
    return args


def command_line(
    placement: Placement,
    model: CatalogModel,
    options: LaunchOptions,
    *,
    executable: str = _SERVER_EXECUTABLE,
) -> list[str]:
    """The whole command, program name first, ready to be run or printed."""
    return [executable, *render_flags(placement, model, options)]


def _projector_flags(placement: Placement, options: LaunchOptions) -> list[str]:
    """The vision projector: its path, where it lives, or that it was deliberately left out."""
    if options.projector_path is None:
        return []
    if placement.projector_pool is None:
        # Said out loud rather than left implicit. Some files carry a projector inside
        # them, and a plan that budgeted no VRAM for one must not let it load anyway.
        return ["--no-mmproj"]
    flags = ["--mmproj", options.projector_path]
    if placement.projector_pool != "vram":
        flags.append("--no-mmproj-offload")
    return flags


def _wants_lazy_mode(model: CatalogModel, options: LaunchOptions) -> bool:
    """Whether ``--lazy-mode`` is rendered: the caller's word first, then the catalog's."""
    if options.lazy_mode is not None:
        return options.lazy_mode
    declared = model.llama_cpp.requires.get("lazy_mode", "").strip().lower()
    return declared in _LAZY_MODE_VALUES


def _sampling_flags(model: CatalogModel) -> list[str]:
    """The vendor's recommended sampling, in the order the catalog declares it.

    Only what the curator actually recorded. An unset parameter is not the same as a
    parameter set to llama.cpp's default, and rendering a guess as a flag would put the
    project's opinion where the vendor's belongs.
    """
    sampling = model.sampling
    pairs: tuple[tuple[str, float | int | None], ...] = (
        ("--temp", sampling.temp),
        ("--top-p", sampling.top_p),
        ("--top-k", sampling.top_k),
        ("--min-p", sampling.min_p),
        ("--presence-penalty", sampling.presence_penalty),
        ("--repeat-penalty", sampling.repeat_penalty),
    )
    flags: list[str] = []
    for flag, value in pairs:
        if value is not None:
            flags += [flag, _number(value)]
    return flags


def _reasoning_flags(model: CatalogModel, options: LaunchOptions) -> list[str]:
    """The chat template's reasoning format, and the thinking toggle when one is asked for."""
    template = model.chat_template
    flags: list[str] = []
    if template.reasoning_format:
        flags += ["--reasoning-format", template.reasoning_format]
    if template.thinking_toggle and options.thinking is not None:
        flags += [
            "--chat-template-kwargs",
            json.dumps({template.thinking_toggle: options.thinking}, separators=(",", ":")),
        ]
    return flags


def quirk_arguments(quirks: Iterable[str]) -> list[str]:
    """The arguments among a model's curated quirks, ignoring the ones that are prose.

    Section 6.2's own example of a quirk is
    ``"--lazy-mode on streams the n-gram table from disk"``: a sentence for a human that
    happens to open with a flag. Splitting that on spaces and appending it to a command
    line would paste ``streams the n-gram table from disk`` after the flag, and a command
    line that fails to parse is the best of the ways that could go wrong.

    So an entry is rendered only when the whole of it reads as arguments, by the rule in
    :func:`_is_arguments`, and is otherwise left to :func:`prose_quirks`, which puts it in
    front of the reader as a note instead.

    Args:
        quirks: The catalog's ``llama_cpp.quirks`` entries.

    Returns:
        The words of every entry that is entirely arguments, in order.
    """
    rendered: list[str] = []
    for quirk in quirks:
        words = quirk.split()
        if _is_arguments(words):
            rendered.extend(words)
    return rendered


def prose_quirks(model: CatalogModel) -> tuple[str, ...]:
    """The curated quirks that are notes for a reader rather than arguments for a program."""
    return tuple(q for q in model.llama_cpp.quirks if not _is_arguments(q.split()))


def _is_arguments(words: Sequence[str]) -> bool:
    """Whether an entry reads as arguments rather than as a sentence.

    The test is a command line's own grammar rather than a guess about English: an entry
    has to open with a flag, and every word that is not a flag has to be the value of the
    flag immediately before it. One word after a flag is a value; a second one in a row is
    a sentence, which is what ``--lazy-mode on streams the n-gram table from disk`` turns
    out to be at its third word.

    What it cannot catch is a two-word note whose second word looks like a value, and
    nothing lexical could: ``--flash-attn recommended`` is indistinguishable from
    ``--rope-scaling yarn``. That residue is the reason a curator writing a note about a
    flag should write a sentence about it, and the reason this function is small enough to
    be read by whoever finds out the hard way.
    """
    if not words or not _is_flag(words[0]):
        return False
    return all(_is_flag(word) or _is_flag(previous) for previous, word in pairwise(words))


def _is_flag(word: str) -> bool:
    """Whether a word is a command-line flag, as opposed to a hyphenated English word."""
    return len(word) > 1 and word.startswith("-")


def _number(value: float | int) -> str:
    """A number as a command line writes it: no grouping, no localised separator.

    Python's own repr, which writes back what the curator wrote: ``0.95`` stays 0.95 and
    ``1.0`` stays 1.0, so the rendered flag and the catalog entry read the same.
    """
    return str(value)
