# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Which machine a measurement belongs to, and whether the run did what it was told.

Two questions, and both of them are about not attributing a number to something it does
not describe.

**Which machine.** A fingerprint is a digest of the parts of a host that do not change
between one run and the next: the operating system, the processor, the size of the memory
pool, the card, and the driver. Free memory is not in it, nor is VRAM in use, because
those change while you watch them and a fingerprint that moved would file every run under
a machine of its own.

The driver *is* in it, and that is a deliberate cost. A driver update can move a card's
speed by more than the tolerance anybody would notice, so a measurement taken before one
is not a measurement of the machine afterwards. Including it means an update quietly
retires every stored result, which is the safe direction: the tool falls back to saying
``estimated``, which is true, rather than carrying on saying ``measured`` about a machine
that has changed underneath it.

**Whether the run did what it was told.** llama.cpp clamps a context larger than the model
allows, rounds a batch, and silently ignores a flag a build was not compiled for. A result
whose stored flags describe something other than what ran is worse than no result, because
the flags are the whole basis on which a later reader decides the measurement applies to
them. :func:`conditions_conflicts` compares what was asked for against what the tool said
it did, and the store refuses a run with a conflict rather than writing a plausible lie.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import PurePath

from llamafit.bench.types import BenchKind, RunConditions
from llamafit.i18n import _
from llamafit.models.host import Host
from llamafit.units import format_bytes

_FINGERPRINT_LENGTH = 16
"""Hexadecimal characters kept from the digest. Sixty-four bits of machine identity."""

_SEPARATOR = "\x1f"
"""The unit separator, which cannot appear in any of the fields it joins."""

_CHECKED_FLAGS: dict[str, str] = {
    "-ngl": "ngl",
    "--n-gpu-layers": "ngl",
    "-ub": "ub",
    "--ubatch-size": "ub",
    "-b": "b",
    "--batch-size": "b",
    "-t": "t",
    "--threads": "t",
    "-c": "c",
    "--ctx-size": "c",
    "-ctk": "ctk",
    "--cache-type-k": "ctk",
    "-ctv": "ctv",
    "--cache-type-v": "ctv",
    "--n-cpu-moe": "n-cpu-moe",
    "-ncmoe": "n-cpu-moe",
}
"""Flags whose value the tool reports back, so the two can be compared.

Only these. A flag nothing reports back cannot be checked, and pretending otherwise by
comparing it against itself would turn a check into a formality.
"""


def host_fingerprint(host: Host) -> str:
    """A stable digest of the machine, for filing measurements under.

    Args:
        host: The scanned machine, or one standing in for it.

    Returns:
        Sixteen hexadecimal characters. A simulated host is prefixed ``sim-``, because a
        measurement can never have been taken on a machine nobody is sitting at, and a
        fingerprint that let one look as though it had would be the worst kind of bug this
        package could have.
    """
    gpu = host.primary_gpu
    parts = [
        host.os,
        host.os_version,
        host.arch,
        host.cpu.model,
        str(host.cpu.physical_cores),
        str(host.memory.total_bytes),
        "" if gpu is None else gpu.name,
        "" if gpu is None else str(gpu.vram_total_bytes or 0),
        "" if gpu is None else gpu.backend_hint,
        "" if gpu is None else (gpu.driver or ""),
    ]
    digest = hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()
    short = digest[:_FINGERPRINT_LENGTH]
    return f"sim-{short}" if host.simulated else short


def host_summary(host: Host) -> str:
    """The machine in one line, so a fingerprint is not the only record of what it was.

    Args:
        host: The scanned machine.

    Returns:
        A line naming the operating system, the processor, the memory and the card. It is
        stored beside the fingerprint because a digest nobody can expand is a digest that
        stops meaning anything the moment the machine is gone.
    """
    gpu = host.primary_gpu
    card = _("no graphics card") if gpu is None else gpu.name
    driver = "" if gpu is None or not gpu.driver else f" ({gpu.driver})"
    return _("%(os)s %(version)s, %(cpu)s, %(ram)s, %(gpu)s%(driver)s") % {
        "os": host.os,
        "version": host.os_version,
        "cpu": host.cpu.model,
        "ram": format_bytes(host.memory.total_bytes),
        "gpu": card,
        "driver": driver,
    }


def conditions_hash(kind: BenchKind, conditions: RunConditions) -> str:
    """A digest of everything two results have to share before they are the same result.

    Args:
        kind: What was measured, which is part of the identity and not a detail of it.
        conditions: The run that happened.

    Returns:
        Sixty-four hexadecimal characters.

    What goes in is the machine, the llama.cpp build, the model file by name and size, the
    complete settings the run actually used, the prompt and generation lengths, and the
    kind. What stays out is the port, the alias, the directory the model happened to live
    in and the run's identifier: two people benchmarking the same file on the same machine
    from different directories measured the same thing, and a hash that said otherwise
    would keep repeats from ever being recognised as repeats.
    """
    parts = [
        kind,
        conditions.host_fingerprint,
        str(conditions.llama_cpp_build or ""),
        conditions.llama_cpp_commit or "",
        conditions.model_id,
        conditions.quant,
        PurePath(conditions.model_file).name,
        str(conditions.model_bytes or ""),
        conditions.flag_string,
        # Named separately as well as being inside `flag_string`, which normally carries
        # them. Belt and braces on purpose: these two are the settings most likely to be
        # swept, and a caller that filled the field without also putting the value in the
        # settings map would otherwise give two genuinely different runs one hash, which
        # is the one failure this digest exists to make impossible.
        str(conditions.context or ""),
        str(conditions.micro_batch or ""),
        str(conditions.n_prompt or ""),
        str(conditions.n_gen or ""),
    ]
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


def flags_from_argv(argv: Sequence[str]) -> dict[str, str]:
    """Read the settings a command line asks for, keyed the way ``settings`` is keyed.

    Args:
        argv: The command line, program name first or not.

    Returns:
        The settings named on it. A repeated flag keeps its last value, which is what
        llama.cpp itself does. ``-ot`` may be repeated meaningfully, so its values are
        joined with a comma rather than overwriting one another.
    """
    found: dict[str, str] = {}
    overrides: list[str] = []
    items = list(argv)
    for index, word in enumerate(items):
        if word in ("-ot", "--override-tensor") and index + 1 < len(items):
            overrides.append(items[index + 1])
            continue
        key = _CHECKED_FLAGS.get(word)
        if key is not None and index + 1 < len(items):
            found[key] = items[index + 1]
        elif word in ("-fa", "--flash-attn") and index + 1 < len(items):
            found["fa"] = items[index + 1]
    if overrides:
        found["ot"] = ",".join(overrides)
    return found


def conditions_conflicts(conditions: RunConditions) -> tuple[str, ...]:
    """Every setting the command line asked for that the tool says it did not use.

    Args:
        conditions: The run that happened, with both its command line and its own report.

    Returns:
        One sentence per disagreement, already translated, and an empty tuple when the run
        did what it was told.

    A value the command line offers as a list -- ``-ub 512,1024,2048``, which llama-bench
    accepts and expands into a run each -- is satisfied when the reported value is one of
    them, because a sweep asks for all of them and each row answers for one.

    A setting the command line does not mention is not a conflict: llama.cpp's own default
    applied, and the reported value is the record of what that default turned out to be,
    which is exactly what wants storing.

    Two settings are recorded but never compared. ``-ot`` is a list of patterns and the
    tool reports which tensors moved rather than which patterns it was given. ``-fa auto``
    resolves to on or off while the model loads, so a disagreement there is the flag doing
    its job and not the run ignoring an instruction.
    """
    asked = flags_from_argv(conditions.argv)
    problems: list[str] = []
    for key, wanted in sorted(asked.items()):
        if key in ("ot", "fa"):
            continue
        actual = conditions.settings.get(key)
        if actual is None or actual == "":
            continue
        if _matches(wanted, actual):
            continue
        problems.append(
            _("%(flag)s was asked for as %(wanted)s but the run reports %(actual)s")
            % {"flag": _flag_for(key), "wanted": wanted, "actual": actual}
        )
    return tuple(problems)


def _flag_for(key: str) -> str:
    """The flag a settings key is written as on a command line."""
    return f"--{key}" if len(key) > 3 else f"-{key}"


_BOOLEAN_TRUE = frozenset({"1", "on", "true", "yes", "enabled"})
_BOOLEAN_FALSE = frozenset({"0", "off", "false", "no", "disabled"})
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _matches(wanted: str, actual: str) -> bool:
    """Whether a reported value answers for a requested one.

    Handles the three ways llama.cpp writes the same answer differently from the way it
    was asked: a comma-separated sweep, a boolean spelled ``on`` in one place and ``1`` in
    the other, and a number with a different amount of whitespace around it.
    """
    for candidate in (piece.strip() for piece in wanted.split(",")):
        if not candidate:
            continue
        if candidate == actual.strip():
            return True
        lowered, other = candidate.lower(), actual.strip().lower()
        if lowered in _BOOLEAN_TRUE and other in _BOOLEAN_TRUE:
            return True
        if lowered in _BOOLEAN_FALSE and other in _BOOLEAN_FALSE:
            return True
        if _NUMBER.match(candidate) and _NUMBER.match(other) and float(candidate) == float(other):
            return True
    return False
