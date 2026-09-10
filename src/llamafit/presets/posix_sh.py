# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
r"""Render a plan as ``start-<id>.sh``: a POSIX shell script, not a bash script.

``#!/bin/sh`` is a promise, and on Debian and Ubuntu it is kept by ``dash``. So there are
no arrays, no ``[[``, no ``local`` and no ``${var,,}`` here: only what
IEEE Std 1003.1 defines, which is also what macOS's ``sh`` and every BusyBox will run.

Quoting is the opposite problem from the Windows renderer's and much the smaller one. A
single-quoted string in ``sh`` has no escapes at all -- every byte is itself until the next
quote -- so one rule covers a path with spaces, dollars, backslashes, asterisks and
newlines alike: put it in single quotes, and write an embedded quote as ``'\\''``, which
closes the string, adds an escaped quote and opens it again. :func:`sh_quote` is those two
sentences.

The other half of getting this right is expansion at *use*. ``"$MODEL"`` in double quotes
is one word however many spaces it holds; ``$MODEL`` bare is however many words the spaces
make of it, and the failure is a llama.cpp that reports a file it was never asked for. Every
use of every variable below is quoted for that reason.
"""

from __future__ import annotations

import re

from llamafit.presets.spec import (
    CONTEXT_TOKEN,
    PresetSpec,
    cannot_measure,
    comment_block,
    group_flags,
    message_block,
    nothing_fits,
    section,
    why_a_ladder,
    why_it_is_yours,
    why_no_ladder,
)
from llamafit.presets.windows_cmd import STAMP_MARKER

_RULE = "# "
"""What opens a section heading."""

_COMMENT = "#  "
"""What opens a comment line, with the indent the header blocks are laid out on."""

_SAFE_WORD = re.compile(r"^[A-Za-z0-9_.:@%+/=,-]+$")
"""A word ``sh`` passes through unchanged, so quoting it would be noise.

No ``*``, no ``?``, no ``[``: the shell expands those against the working directory, and
``-ot 'ffn_.*_shexp=CPU'`` becomes a filename the moment one happens to match.
"""


def sh_quote(word: str) -> str:
    r"""One argument as ``sh`` has to spell it: single quotes, with ``'`` as ``'\\''``.

    Args:
        word: The argument as ``llama-server`` should receive it.

    Returns:
        The argument, quoted unless it is plainly safe unquoted.
    """
    if word and _SAFE_WORD.match(word):
        return word
    return "'" + word.replace("'", "'\\''") + "'"


def render_sh(spec: PresetSpec, *, stamp: str) -> str:
    r"""The whole shell script, with ``\\n`` line endings, which are the only kind it may have.

    Args:
        spec: The plan to render.
        stamp: The value to write on the stamp line; see
            :func:`llamafit.presets.render.stamp_of`.

    Returns:
        The file's text. A shell script written with carriage returns fails at its own
        shebang -- the kernel looks for ``/bin/sh\\r`` and reports that it does not exist --
        so :func:`llamafit.presets.render.write_files` never translates line endings for
        this one, on any platform.
    """
    lines = ["#!/bin/sh"]
    lines += _header(spec, stamp)
    lines += [
        "",
        "# Unset variables are a bug here, not an empty string: every one of them is a",
        "# path or a memory figure, and continuing with an empty one is how a script",
        "# launches something other than what it was written to launch.",
        "set -u",
        "",
    ]
    lines += _paths(spec)
    lines += _machine_check(spec)
    lines += _ladder(spec)
    lines += _launch(spec)
    return "\n".join(lines) + "\n"


def _header(spec: PresetSpec, stamp: str) -> list[str]:
    """The comment block: what this is, what it decides at start time, and why it does."""
    rule = "# " + "=" * 76
    prose = why_a_ladder(spec) if spec.probe_free_memory else why_no_ladder(spec)
    return [
        rule,
        f"#  {spec.script_stem}.sh",
        f"#  Launch {spec.name} ({spec.quant}) with llama.cpp on this machine.",
        "#",
        f"#  Written by LlamaFit {spec.version} on {spec.generated.isoformat()},",
        f"#  by:  llamafit preset {spec.model_id}",
        f"#  Placement: {spec.mode}, {spec.planned_context} tokens{spec.build_phrase}.",
        f"#  {spec.card_sentence}",
        "#",
        *comment_block(_COMMENT, prose),
        "#",
        *comment_block(_COMMENT, why_it_is_yours(spec)),
        "#",
        f"#  {STAMP_MARKER} {stamp}",
        rule,
    ]


def _paths(spec: PresetSpec) -> list[str]:
    """The two or three paths, named once so the rest of the file reads as English."""
    lines = [section(_RULE, "where things are")]
    lines.append(f"LLAMA_SERVER={sh_quote(spec.server_path or spec.server_program)}")
    lines.append(f"MODEL={sh_quote(spec.model_path)}")
    if spec.projector_path is not None:
        lines.append(f"MMPROJ={sh_quote(spec.projector_path)}")
    lines.append("")
    return lines


def _machine_check(spec: PresetSpec) -> list[str]:
    """Refuse to start when a file has moved; say so when the card has changed."""
    stem = spec.model_id
    return [
        section(_RULE, f"has this machine changed since {spec.generated.isoformat()}?"),
        "# A model that has moved, or a llama.cpp that is gone, is fatal: there is",
        "# nothing to run. A different card is not, because the ladder below adapts,",
        "# but the layer split was chosen for the card named at the top of this file.",
        'if [ ! -f "$MODEL" ]; then',
        '    echo "[llamafit] The model file is not where this script expects it:" >&2',
        '    echo "[llamafit]   $MODEL" >&2',
        f'    echo "[llamafit] Fetch it with:      llamafit install model {stem}" >&2',
        f'    echo "[llamafit] Re-write this with: llamafit preset {stem} --force" >&2',
        "    exit 3",
        "fi",
        "",
        'if [ ! -x "$LLAMA_SERVER" ]; then',
        f"    if command -v {spec.server_program} >/dev/null 2>&1; then",
        '        echo "[llamafit] llama.cpp is no longer at the path in this script;" >&2',
        f'        echo "[llamafit] using the {spec.server_program} found on PATH." >&2',
        f"        LLAMA_SERVER={spec.server_program}",
        "    else",
        '        echo "[llamafit] llama-server was not found, neither here nor on PATH:" >&2',
        '        echo "[llamafit]   $LLAMA_SERVER" >&2',
        '        echo "[llamafit] Install it with: llamafit install llama.cpp" >&2',
        "        exit 3",
        "    fi",
        "fi",
        "",
    ]


def _ladder(spec: PresetSpec) -> list[str]:
    """Read free memory, then walk the rungs; or, with nothing to read, do neither."""
    if not spec.probe_free_memory:
        return [
            section(_RULE, "the context"),
            "# Nothing to measure here; see the note at the top of this file.",
            f"CTX={spec.planned_context}",
            "",
            *_context_override(),
        ]
    return [
        *_free_memory_function(spec),
        *_card_identity(spec),
        *_rungs(spec),
        *_context_override(),
    ]


def _free_memory_function(spec: PresetSpec) -> list[str]:
    """The one shell function in the file: how much memory is free, in MiB, or nothing.

    Which body it gets depends on the card, because there is no portable answer.
    :func:`llamafit.presets.spec.can_probe_free_memory` decides whether there is an answer
    at all, so by the time this runs there is exactly one to write.
    """
    lines = [
        section(_RULE, "how much memory is free right now?"),
        "# Prints a whole number of MiB, or nothing at all when it cannot tell. Nothing",
        "# is a real answer and the caller treats it as one: guessing a figure here is",
        "# how a script talks itself into a configuration that pages.",
        "free_mib() {",
    ]
    lines += _probe_body(spec)
    lines += ["}", "", "FREE_MIB=$(free_mib)", ""]
    return lines


def _probe_body(spec: PresetSpec) -> list[str]:
    """The body of ``free_mib`` for the card this plan was made for."""
    if spec.gpu_vendor == "nvidia":
        index = 0 if spec.gpu_index is None else spec.gpu_index
        return [
            "    # nvidia-smi ships with the NVIDIA driver and answers in MiB. -i pins the",
            "    # card the plan was made for: on a machine with two, the other one's free",
            "    # memory would be a true number about the wrong device. tr keeps only the",
            "    # digits, so the [N/A] some cards report for an unsupported field becomes",
            "    # the empty string rather than a zero to compare against.",
            "    command -v nvidia-smi >/dev/null 2>&1 || return 0",
            f"    nvidia-smi -i {index} --query-gpu=memory.free --format=csv,noheader,nounits \\",
            "        2>/dev/null | head -n 1 | tr -dc '0-9'",
        ]
    if spec.gpu_vendor == "amd":
        return [
            "    # amdgpu publishes what the card has and what is on it in sysfs, in bytes.",
            "    # There is no rocm-smi dependency here on purpose: sysfs is always readable",
            "    # and always present, and rocm-smi is neither.",
            "    for device in /sys/class/drm/card*/device; do",
            '        [ -r "$device/mem_info_vram_total" ] || continue',
            '        [ -r "$device/mem_info_vram_used" ] || continue',
            '        total=$(cat "$device/mem_info_vram_total")',
            '        used=$(cat "$device/mem_info_vram_used")',
            "        echo $(( (total - used) / 1048576 ))",
            "        return 0",
            "    done",
        ]
    return [
        "    # On Apple silicon the graphics memory is the system memory, so what a model",
        "    # can be given is what vm_stat calls free, inactive and speculative: inactive",
        "    # pages are cache the kernel will hand over on demand, and leaving them out",
        "    # would understate what is available by most of it. The page size is in the",
        "    # header because it is 16 KiB on Apple silicon and 4 KiB elsewhere.",
        "    command -v vm_stat >/dev/null 2>&1 || return 0",
        "    vm_stat | awk '",
        "        /page size of/     { page = $8 }",
        "        /^Pages free/      { free = $3 }",
        "        /^Pages inactive/  { inactive = $3 }",
        "        /^Pages speculative/ { speculative = $3 }",
        "        END {",
        '            if (page == "") page = 4096',
        '            gsub(/\\./, "", free)',
        '            gsub(/\\./, "", inactive)',
        '            gsub(/\\./, "", speculative)',
        '            printf "%d", (free + inactive + speculative) * page / 1048576',
        "        }'",
    ]


def _card_identity(spec: PresetSpec) -> list[str]:
    """Notice a swapped card by its size, which is the figure the flags were chosen for."""
    if spec.gpu_total_mib is None or spec.gpu_vendor != "nvidia":
        return []
    index = 0 if spec.gpu_index is None else spec.gpu_index
    name = spec.gpu_name or "the card this was planned for"
    return [
        section(_RULE, "is this still the same card?"),
        "TOTAL_MIB=$(nvidia-smi -i "
        f"{index} --query-gpu=memory.total --format=csv,noheader,nounits \\",
        "    2>/dev/null | head -n 1 | tr -dc '0-9')",
        f'if [ -n "$TOTAL_MIB" ] && [ "$TOTAL_MIB" != "{spec.gpu_total_mib}" ]; then',
        '    echo "[llamafit] This card reports $TOTAL_MIB MiB, not the '
        f'{spec.gpu_total_mib} MiB of" >&2',
        f'    echo "[llamafit] {name}, which this plan was made for. The context" >&2',
        '    echo "[llamafit] below still adapts, but the layer split does not:" >&2',
        f'    echo "[llamafit]   llamafit preset {spec.model_id} --force" >&2',
        "fi",
        "",
    ]


def _rungs(spec: PresetSpec) -> list[str]:
    """The ladder, unrolled largest first, and the two ways of not choosing from it."""
    lines = [
        section(_RULE, "the ladder"),
        "# Each rung was costed by the same budget that produced the plan.",
        "#",
        "#      context   needs on the card",
    ]
    lines += [f"#      {rung.tokens:>7}   {rung.vram_required_mib:>6} MiB" for rung in spec.rungs]
    lines += [
        "#",
        "CTX=''",
        'if [ -n "$FREE_MIB" ]; then',
        f"    USABLE_MIB=$(( FREE_MIB - {spec.reserve_mib} ))",
    ]
    first = True
    for rung in spec.rungs:
        keyword = "if  " if first else "elif"
        lines.append(
            f'    {keyword} [ "$USABLE_MIB" -ge {rung.vram_required_mib} ]; then CTX={rung.tokens}'
        )
        first = False
    stuck = message_block(nothing_fits(spec, free_memory="$FREE_MIB"))
    lines += [
        "    fi",
        '    if [ -z "$CTX" ]; then',
        *(_echo(line, indent=8) for line in stuck),
        "        exit 4",
        "    fi",
        "else",
        *(_echo(line, indent=4) for line in message_block(cannot_measure(spec))),
        f"    CTX={spec.planned_context}",
        "fi",
        "",
    ]
    return lines


def _echo(text: str, *, indent: int) -> str:
    """One ``echo`` to standard error, indented to the block it sits in.

    Standard error and not standard output: everything this script says about itself is a
    remark about a launch, and a person piping llama-server's output into a file should get
    the model's tokens and not LlamaFit's commentary.
    """
    return " " * indent + f'echo "{text}" >&2'


def _context_override() -> list[str]:
    """The one hand-hold: an environment variable that overrules the ladder for a run."""
    return [
        "# LLAMAFIT_CONTEXT overrules all of the above, for a deliberate experiment.",
        'if [ -n "${LLAMAFIT_CONTEXT:-}" ]; then',
        '    CTX="$LLAMAFIT_CONTEXT"',
        "fi",
        "",
    ]


def _launch(spec: PresetSpec) -> list[str]:
    """The command itself, one flag to a line, and what to say if it comes straight back."""
    lines = [
        section(_RULE, "go"),
        f'echo "[llamafit] {spec.name} ({spec.quant}) at $CTX tokens on {spec.endpoint}"',
        'echo "[llamafit] Press Ctrl+C to stop it."',
        "",
        "# Anything you pass to this script is appended to the command below, so you",
        f"# always have the last word:  ./{spec.script_stem}.sh --verbose",
        "#",
        "# No exec: the shell stays alive so that a non-zero exit can be explained, which",
        "# is worth one idle process. Ctrl+C reaches llama-server either way, because it",
        "# goes to the whole foreground process group.",
    ]
    lines += _command_lines(spec)
    lines += [
        "RC=$?",
        'if [ "$RC" -ne 0 ]; then',
        *_upgrade_hint(spec),
        "fi",
        'exit "$RC"',
    ]
    return lines


def _upgrade_hint(spec: PresetSpec) -> list[str]:
    """What a non-zero exit most often means on a machine whose llama.cpp has moved on."""
    build = (
        f"since build {spec.llamacpp_build}" if spec.llamacpp_build is not None else "since then"
    )
    return [
        '    echo "[llamafit] llama-server exited with code $RC." >&2',
        f'    echo "[llamafit] If llama.cpp has been upgraded {build}, a flag above" >&2',
        '    echo "[llamafit] may no longer be one it accepts. Re-render them with:" >&2',
        f'    echo "[llamafit]   llamafit preset {spec.model_id} --force" >&2',
    ]


def _command_lines(spec: PresetSpec) -> list[str]:
    """``llama-server`` with one flag and its value to a line, continued with backslashes."""
    words = ['"$LLAMA_SERVER"']
    words += [f"    {group}" for group in _grouped_arguments(spec)]
    words.append('    "$@"')
    return [word + " \\" for word in words[:-1]] + [words[-1]]


def _grouped_arguments(spec: PresetSpec) -> list[str]:
    """The flags, each with the value that belongs to it, ready for one line each."""
    return [" ".join(group) for group in group_flags(_shell_words(spec))]


def _shell_words(spec: PresetSpec) -> list[str]:
    """The rendered flags with the three long values replaced by the variables above."""
    out: list[str] = []
    for word in spec.flags:
        if word == CONTEXT_TOKEN:
            out.append('"$CTX"')
        elif word == spec.model_path:
            out.append('"$MODEL"')
        elif spec.projector_path is not None and word == spec.projector_path:
            out.append('"$MMPROJ"')
        else:
            out.append(sh_quote(word))
    return out
