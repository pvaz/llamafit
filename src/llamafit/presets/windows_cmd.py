# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
r"""Render a plan as ``start-<id>.cmd``: a batch file somebody can double-click.

Batch is a language with three separate quoting rules layered on top of one another, and a
model path with a space in it meets all three. This module keeps them apart on purpose.

``cmd`` expands ``%NAME%`` **before** it parses the line, in a batch file and inside double
quotes alike, so a literal per cent sign has to be written ``%%`` -- and a directory called
``100% Mine`` is not exotic. It also treats ``&``, ``|``, ``<``, ``>`` and ``^`` as syntax
outside quotes, which is why every argument that is not plainly safe is quoted rather than
trusted. Inside the quotes, ``llama-server`` is a C program and parses its own tail by the
Microsoft C rules, where a quote has to arrive as ``\\"``: that is what
:func:`cmd_argument` implements, and it is why the JSON of ``--chat-template-kwargs``
survives the trip.

Two things this file deliberately does not use.

**No delayed expansion.** ``setlocal enabledelayedexpansion`` would make ``!`` a
metacharacter, and ``!`` is a legal character in a Windows path. The ladder is the only
thing that would have wanted it, and the ladder is unrolled instead -- one ``if`` per rung,
in descending order, guarded by ``if not defined CTX`` so the first match wins. Unrolled it
needs no expansion at all, and a person opening the file sees the ladder written out
instead of a loop they have to run in their head.

**No parenthesised blocks.** ``if exist "%X%" ( ... )`` is where batch's parse-then-expand
order bites hardest. Every branch here is a ``goto`` to a label, which is longer and cannot
surprise anyone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

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

STAMP_MARKER = "llamafit-preset-stamp:"
"""The label the overwrite check looks for; see :mod:`llamafit.presets.render`."""

_SAFE_WORD = re.compile(r"^[A-Za-z0-9_.:@#=+/\\*?~,;\[\]{}-]+$")
"""A word cmd will pass through untouched, so quoting it would only add noise.

Deliberately narrow. ``*`` and ``?`` are on it because ``cmd`` does not expand wildcards --
the program does -- and ``llama-server`` never treats an argument as a glob. Anything with
a space, a quote, a per cent sign or a redirection character is off it and gets quoted.
"""

_RULE = "rem "
"""What opens a section heading."""

_COMMENT = "rem  "
"""What opens a comment line, with the indent the header blocks are laid out on."""

_ECHO_SPECIAL = "^&|<>()"
"""Characters ``echo`` would read as syntax; each is escaped with a caret."""


def cmd_argument(word: str) -> str:
    r"""One command-line argument as a batch file has to spell it.

    Args:
        word: The argument as ``llama-server`` should receive it.

    Returns:
        The same argument, quoted and escaped for both ``cmd`` and the C runtime.

    The two layers are applied in the order they are undone. First the Microsoft C quoting
    rules, which is where a backslash run before a quote doubles and a quote becomes
    ``\\"``; then the per cent sign is doubled, because ``cmd`` expands it earlier than
    anything else and does so inside quotes too.
    """
    if word and _SAFE_WORD.match(word) and "%" not in word:
        return word
    return _double_percent(_c_quote(word))


def cmd_assignment(name: str, value: str) -> str:
    """A ``set "NAME=value"`` line, safe for a value with spaces or per cent signs.

    The quotes go around the whole assignment rather than around the value: written
    ``set NAME="value"`` the quotes become part of the value, and every later use of it
    would carry a pair of quotes into the middle of a path.
    """
    return f'set "{name}={_double_percent(value)}"'


def cmd_echo(text: str) -> str:
    """An ``echo`` line whose text survives cmd's own syntax.

    Args:
        text: What to print. ``%NAME%`` in it is left alone and expands at run time, which
            is what every message here that quotes a figure relies on. Only LlamaFit's own
            sentences and catalog identifiers are echoed, so nothing arrives holding a per
            cent sign it wanted printed literally; a path is echoed through a variable.

    Returns:
        The ``echo`` line. An empty text becomes ``echo.``, because a bare ``echo`` reports
        whether echoing is on rather than printing a blank line.
    """
    if not text:
        return "echo."
    return f"echo {_escape_echo(text)}"


def render_cmd(spec: PresetSpec, *, stamp: str) -> str:
    r"""The whole batch file, with ``\\n`` line endings for the writer to translate.

    Args:
        spec: The plan to render.
        stamp: The value to write on the stamp line; see
            :func:`llamafit.presets.render.stamp_of`.

    Returns:
        The file's text. Line endings are left as ``\\n`` here and turned into ``\\r\\n``
        by :func:`llamafit.presets.render.write_files`, so nothing in this repository has
        to store a carriage return and every golden file compares as text.
    """
    lines = ["@echo off"]
    lines += _code_page(spec)
    lines += _header(spec, stamp)
    lines += ["", "setlocal", ""]
    lines += _paths(spec)
    lines += _machine_check(spec)
    lines += _ladder(spec)
    lines += _launch(spec)
    return "\n".join(lines) + "\n"


def _code_page(spec: PresetSpec) -> list[str]:
    """``chcp 65001`` only when something in the file is not ASCII.

    A batch file has no way to declare its encoding: ``cmd`` reads it in the console's
    code page, so a path with an accent in it comes out as mojibake and a byte-order mark
    at the top is printed rather than skipped. Everything LlamaFit writes into these files
    is ASCII by choice for exactly that reason -- see the package docstring -- but a user
    directory is not LlamaFit's choice, and when one is not ASCII this is the only thing
    that makes the file work at all.
    """
    if _is_ascii(spec):
        return []
    return [
        "rem A path below is not ASCII, and a batch file cannot declare its encoding.",
        "rem This file is UTF-8, so the console has to be told before it reads the path.",
        "chcp 65001 >nul",
    ]


def _header(spec: PresetSpec, stamp: str) -> list[str]:
    """The comment block: what this is, what it does at start time, and why.

    Long, and meant to be. Whoever opens this file in six months will not have had the
    conversation that produced it, and the one thing they must not conclude is that the
    context could just as well have been written in as a number.
    """
    rule = "rem " + "=" * 74
    prose = why_a_ladder(spec) if spec.probe_free_memory else why_no_ladder(spec)
    return [
        rule,
        f"rem  {spec.script_stem}.cmd",
        f"rem  Launch {spec.name} ({spec.quant}) with llama.cpp on this machine.",
        "rem",
        f"rem  Written by LlamaFit {spec.version} on {spec.generated.isoformat()},",
        f"rem  by:  llamafit preset {spec.model_id}",
        f"rem  Placement: {spec.mode}, {spec.planned_context} tokens{spec.build_phrase}.",
        f"rem  {spec.card_sentence}",
        "rem",
        *comment_block(_COMMENT, prose),
        "rem",
        *comment_block(_COMMENT, why_it_is_yours(spec)),
        "rem",
        f"rem  {STAMP_MARKER} {stamp}",
        rule,
    ]


def _paths(spec: PresetSpec) -> list[str]:
    """The two or three paths, named once so the rest of the file reads as English."""
    lines = [section(_RULE, "where things are")]
    lines.append(cmd_assignment("LLAMA_SERVER", spec.server_path or spec.server_program))
    lines.append(cmd_assignment("MODEL", spec.model_path))
    if spec.projector_path is not None:
        lines.append(cmd_assignment("MMPROJ", spec.projector_path))
    lines.append("")
    return lines


def _machine_check(spec: PresetSpec) -> list[str]:
    """Refuse to start when a file has moved; say so when the card has changed.

    Two different answers to "the machine is not what it was", because they are two
    different situations. A missing model or a missing server means there is nothing to
    run and guessing would only produce a worse error later. A different card is survivable
    -- the ladder is what makes it survivable -- but the layer split was chosen for the old
    one, so it is worth a sentence on the way past.
    """
    stem = spec.model_id
    lines = [
        section(_RULE, f"has this machine changed since {spec.generated.isoformat()}?"),
        "rem A model that has moved, or a llama.cpp that is gone, is fatal: there is",
        "rem nothing to run. A different card is not, because the ladder below adapts,",
        "rem but the layer split was chosen for the card named at the top of this file.",
        'if exist "%MODEL%" goto model_found',
        cmd_echo("[llamafit] The model file is not where this script expects it:"),
        "echo [llamafit]   %MODEL%",
        cmd_echo(f"[llamafit] Fetch it with:      llamafit install model {stem}"),
        cmd_echo(f"[llamafit] Re-write this with: llamafit preset {stem} --force"),
        "exit /b 3",
        ":model_found",
        "",
        'if exist "%LLAMA_SERVER%" goto server_found',
        f"where {spec.server_program} >nul 2>nul",
        "if errorlevel 1 goto no_server",
        "rem llama.cpp is not where it was, but PATH still has one; use that and say so.",
        cmd_echo("[llamafit] llama.cpp is no longer at the path in this script;"),
        cmd_echo(f"[llamafit] using the {spec.server_program} found on PATH instead."),
        cmd_assignment("LLAMA_SERVER", spec.server_program),
        "goto server_found",
        ":no_server",
        cmd_echo("[llamafit] llama-server was not found, neither here nor on PATH:"),
        "echo [llamafit]   %LLAMA_SERVER%",
        cmd_echo("[llamafit] Install it with: llamafit install llama.cpp"),
        "exit /b 3",
        ":server_found",
        "",
    ]
    return lines


def _ladder(spec: PresetSpec) -> list[str]:
    """Read free card memory, then walk the rungs; or, with no card, do neither."""
    if not spec.probe_free_memory:
        return [
            section(_RULE, "the context"),
            "rem Nothing of this placement lives on a graphics card, so there is no free",
            "rem card memory to read and no rung to step down to.",
            cmd_assignment("CTX", str(spec.planned_context)),
            "",
            *_context_override(spec),
        ]
    return [
        *_free_memory_probe(spec),
        *_card_identity(spec),
        *_rungs(spec),
        *_context_override(spec),
    ]


def _free_memory_probe(spec: PresetSpec) -> list[str]:
    """Ask the vendor tool how much card memory is free, and validate what comes back.

    ``nvidia-smi`` is the only tool this reaches for on Windows, and that is a statement
    about Windows rather than a preference: it ships with the driver and answers in one
    number, while AMD and Intel publish nothing equivalent that a batch file can call. On
    a card it cannot ask about, the script says so and falls back -- see ``:no_probe``.
    """
    query = _nvidia_query(spec, "memory.free")
    return [
        section(_RULE, "how much card memory is free right now?"),
        "rem nvidia-smi ships with the NVIDIA driver and answers in MiB. -i pins the card",
        "rem the plan was made for: on a machine with two, the other one's free memory",
        "rem would be a true number about the wrong device.",
        'set "FREE_MIB="',
        f'for /f "usebackq tokens=1 delims= " %%f in (`{query}`) do set "FREE_MIB=%%f"',
        "if not defined FREE_MIB goto no_probe",
        "rem Some cards report a field they do not support as [N/A], which is an answer",
        "rem but not a number, and comparing against it would compare against zero.",
        'echo %FREE_MIB%| findstr /r /c:"^[0-9][0-9]*$" >nul',
        "if errorlevel 1 goto no_probe",
        "",
    ]


def _card_identity(spec: PresetSpec) -> list[str]:
    """Notice a swapped card by its size, which is the figure the flags were chosen for."""
    if spec.gpu_total_mib is None:
        return []
    query = _nvidia_query(spec, "memory.total")
    name = spec.gpu_name or "the card this was planned for"
    return [
        section(_RULE, "is this still the same card?"),
        'set "TOTAL_MIB="',
        f'for /f "usebackq tokens=1 delims= " %%t in (`{query}`) do set "TOTAL_MIB=%%t"',
        "if not defined TOTAL_MIB goto card_checked",
        f'if "%TOTAL_MIB%"=="{spec.gpu_total_mib}" goto card_checked',
        cmd_echo(
            f"[llamafit] This card reports %TOTAL_MIB% MiB, not the {spec.gpu_total_mib} MiB of"
        ),
        cmd_echo(f"[llamafit] {name}, which this plan was made for. The context below"),
        cmd_echo("[llamafit] still adapts, but the layer split does not:"),
        cmd_echo(f"[llamafit]   llamafit preset {spec.model_id} --force"),
        ":card_checked",
        "",
    ]


def _rungs(spec: PresetSpec) -> list[str]:
    """The ladder, unrolled largest first, and the two ways of not choosing from it."""
    lines = [
        section(_RULE, "the ladder"),
        "rem Each rung was costed by the same budget that produced the plan.",
        "rem",
        "rem      context   needs on the card",
    ]
    lines += [f"rem      {rung.tokens:>7}   {rung.vram_required_mib:>6} MiB" for rung in spec.rungs]
    lines += [
        "rem",
        f'set /a "USABLE_MIB=%FREE_MIB% - {spec.reserve_mib}"',
        'set "CTX="',
    ]
    # Descending, each guarded, so the first rung that fits is the one that is taken and
    # the rest of the ladder is walked past rather than tested with a jump.
    lines += [
        f'if %USABLE_MIB% GEQ {rung.vram_required_mib} if not defined CTX set "CTX={rung.tokens}"'
        for rung in spec.rungs
    ]
    lines += [
        "if defined CTX goto have_context",
        "",
        *(cmd_echo(line) for line in message_block(nothing_fits(spec, free_memory="%FREE_MIB%"))),
        "exit /b 4",
        "",
        ":no_probe",
        *(cmd_echo(line) for line in message_block(cannot_measure(spec))),
        cmd_assignment("CTX", str(spec.planned_context)),
        "",
        ":have_context",
    ]
    return lines


def _context_override(spec: PresetSpec) -> list[str]:
    """The one hand-hold: an environment variable that overrules the ladder for a run."""
    return [
        "rem LLAMAFIT_CONTEXT overrules all of the above, for a deliberate experiment.",
        'if defined LLAMAFIT_CONTEXT set "CTX=%LLAMAFIT_CONTEXT%"',
        "",
    ]


def _launch(spec: PresetSpec) -> list[str]:
    """The command itself, one flag to a line, and what to say if it comes straight back."""
    lines = [
        section(_RULE, "go"),
        cmd_echo(f"[llamafit] {spec.name} ({spec.quant}) at %CTX% tokens on {spec.endpoint}"),
        cmd_echo("[llamafit] Press Ctrl+C to stop it."),
        "",
        "rem Anything you pass to this script is appended to the command below, so you",
        "rem always have the last word:  " + spec.script_stem + ".cmd --verbose",
        "rem",
        "rem The carets are line continuations. A space after one ends the command there,",
        "rem which is the one edit to this block that fails quietly, so watch for it.",
    ]
    lines += _command_lines(spec)
    lines += [
        'set "RC=%ERRORLEVEL%"',
        'if "%RC%"=="0" goto stopped',
        cmd_echo("[llamafit] llama-server exited with code %RC%."),
        *_upgrade_hint(spec),
        ":stopped",
        "exit /b %RC%",
    ]
    return lines


def _upgrade_hint(spec: PresetSpec) -> list[str]:
    """What a non-zero exit most often means on a machine whose llama.cpp has moved on."""
    build = (
        f" since build {spec.llamacpp_build}" if spec.llamacpp_build is not None else " since then"
    )
    return [
        cmd_echo(f"[llamafit] If llama.cpp has been upgraded{build}, a flag above may no"),
        cmd_echo("[llamafit] longer be one it accepts. Re-render them with:"),
        cmd_echo(f"[llamafit]   llamafit preset {spec.model_id} --force"),
    ]


def _command_lines(spec: PresetSpec) -> list[str]:
    """``llama-server`` with one flag and its value to a line, continued with carets."""
    words = ['"%LLAMA_SERVER%"']
    words += [f"  {group}" for group in _grouped_arguments(spec)]
    words.append("  %*")
    return [word + " ^" for word in words[:-1]] + [words[-1]]


def _grouped_arguments(spec: PresetSpec) -> list[str]:
    """The flags, each with the value that belongs to it, ready for one line each."""
    return [" ".join(group) for group in group_flags(_batch_words(spec))]


def _batch_words(spec: PresetSpec) -> list[str]:
    """The rendered flags with the three long values replaced by the variables above."""
    out: list[str] = []
    for word in spec.flags:
        if word == CONTEXT_TOKEN:
            out.append("%CTX%")
        elif word == spec.model_path:
            out.append('"%MODEL%"')
        elif spec.projector_path is not None and word == spec.projector_path:
            out.append('"%MMPROJ%"')
        else:
            out.append(cmd_argument(word))
    return out


def _nvidia_query(spec: PresetSpec, field: str) -> str:
    r"""One ``nvidia-smi`` query, escaped for the inside of a ``for /f`` back-quote block.

    Two escapes, and the second one cost an afternoon to find.

    ``>`` is still cmd's redirection inside the back quotes, so the ``2>nul`` that hides
    the tool's own complaint has to arrive as ``2^>nul`` or the line redirects the whole
    ``for``.

    And ``--format=csv,noheader,nounits`` has to be quoted. ``for /f`` hands the command
    to a second ``cmd``, whose tokeniser treats a comma and an equals sign as argument
    separators, so unquoted the tool is handed three arguments and answers
    ``ERROR: Option noheader is not recognized`` -- on standard output, where it looks
    exactly like a memory figure until something tries to compare it with one. Quoting the
    value and escaping the equals signs is what was measured to work on the reference
    machine; the ``findstr`` guard at the call site is what catches it if it ever does not.
    """
    index = 0 if spec.gpu_index is None else spec.gpu_index
    return f'nvidia-smi -i {index} --query-gpu^={field} --format^="csv,noheader,nounits" 2^>nul'


def _is_ascii(spec: PresetSpec) -> bool:
    """Whether every path and name in the plan is ASCII, which decides the code page."""
    values: Iterable[str | None] = (
        spec.model_path,
        spec.projector_path,
        spec.server_path,
        spec.name,
        spec.gpu_name,
    )
    return all(value is None or value.isascii() for value in values)


def _c_quote(word: str) -> str:
    """Quote one argument by the Microsoft C runtime's rules, which is what reads it.

    A backslash is ordinary except in a run immediately before a quote, where the run
    doubles; the closing quote is a quote too, so a value ending in a backslash needs the
    same treatment or it escapes the delimiter.
    """
    out = ['"']
    backslashes = 0
    for char in word:
        if char == "\\":
            backslashes += 1
            out.append(char)
            continue
        if char == '"':
            out.append("\\" * (backslashes + 1))
            out.append('"')
        else:
            out.append(char)
        backslashes = 0
    out.append("\\" * backslashes)
    out.append('"')
    return "".join(out)


def _double_percent(text: str) -> str:
    """``%`` doubled, because a batch file expands it before it parses anything else."""
    return text.replace("%", "%%")


def _escape_echo(text: str) -> str:
    """Caret-escape what ``echo`` would otherwise read as syntax.

    The caret goes first: escaping it after ``&`` would put a caret in front of the caret
    that was added to escape something else. The per cent sign is deliberately left alone,
    because every message that carries one is naming a variable it wants expanded.
    """
    out = text.replace("^", "^^")
    for char in _ECHO_SPECIAL[1:]:
        out = out.replace(char, "^" + char)
    return out
