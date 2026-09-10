# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
r"""Turn a specification into files on disk, without ever losing an edit somebody made.

Three things happen here and each of them is a decision rather than plumbing.

**Which files.** A launch script for this operating system, a router section, and a README
naming the endpoints -- section 15.3's list. The script is written for the machine that
asked for it, not both: a ``.sh`` full of ``C:\\`` paths is not a portable convenience, it
is a file that will be found later and believed.

**Line endings and encoding, per file.** A ``.sh`` written with carriage returns fails at
its own shebang, because the kernel looks for an interpreter called ``/bin/sh\\r``. A
``.cmd`` written with a byte-order mark prints the mark instead of skipping it, and read in
the console's code page rather than UTF-8. So the renderers produce ``\\n`` throughout, and
this module decides -- once, in :func:`_unstamped` -- what each file becomes on the way to
the disk. Every golden file in the test suite is therefore plain text with plain newlines, and
the translation is tested on its own.

**Whether to write at all.** These files exist to be edited; that is the whole reason for
writing a file rather than printing a command. So each one carries a stamp, which is a
checksum of the rest of it, and a file whose stamp no longer matches its contents is a file
somebody has changed. LlamaFit keeps it and says so. ``--force`` overwrites and says that
too, because silently discarding an evening of tuning is worse than any error message.

The check is deliberately conservative in one direction: a file with no stamp at all counts
as edited. A file that has lost its stamp was either hand-written or hand-cut, and both are
somebody's work.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from llamafit.presets.models_ini import render_ini_section
from llamafit.presets.posix_sh import render_sh
from llamafit.presets.spec import CONTEXT_TOKEN, PresetSpec, Rung
from llamafit.presets.windows_cmd import STAMP_MARKER, render_cmd

PLACEHOLDER_STAMP = "0" * 64
"""What stands where the stamp goes while the stamp is being computed.

A checksum cannot cover itself, so the text is hashed with this in place of the value and
the value is substituted afterwards. Sixty-four zeros because that is the width of the
hexadecimal digest, which keeps the line the same length before and after and so keeps the
hash independent of how wide the number happened to be.
"""

_STAMP_RE = re.compile(rf"{re.escape(STAMP_MARKER)} ([0-9a-f]{{64}})")
"""How a stamp is found again in a file somebody may have moved lines around in."""

Outcome = Literal["created", "updated", "unchanged", "kept-your-edits", "overwritten"]
"""What happened to one file, in the words the interface reports.

``kept-your-edits`` is the interesting one and the reason the others are named at all: a
command that wrote three files and left one alone has to be able to say which, and a
boolean cannot.
"""


@dataclass(frozen=True)
class PresetFile:
    r"""One rendered file, before it has touched the disk.

    Attributes:
        name: The file name, without a directory.
        text: Its contents, with ``\\n`` line endings whatever it will get on disk.
        newline: What ``\\n`` becomes when it is written.
        executable: Whether the owner execute bit should be set, where there is one.
    """

    name: str
    text: str
    newline: str
    executable: bool


class FileOutcome(BaseModel):
    """One file and what became of it, for ``--json``.

    Attributes:
        path: Where it is.
        outcome: What happened, in the words :data:`Outcome` defines.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    outcome: Outcome


class PresetReport(BaseModel):
    """What ``llamafit preset`` produced, in the shape a script can read.

    Attributes:
        model_id: The catalog id.
        name: The model's display name.
        quant: The quantisation planned.
        directory: Where the files went.
        endpoint: Where the server will answer.
        planned_context: The top of the ladder, which is what the planner chose.
        reserve_mib: What the script holds back from free card memory.
        probe_free_memory: Whether the script chooses its context at start time. False is
            worth a script reading this: it means the context is a number rather than a
            decision, and the header of the script says which of the two reasons applies.
        rungs: The ladder, largest first.
        files: What became of each file.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    name: str
    quant: str
    directory: str
    endpoint: str
    planned_context: int
    reserve_mib: int
    probe_free_memory: bool
    rungs: list[Rung]
    files: list[FileOutcome]


def report_of(spec: PresetSpec, directory: Path, results: Sequence[WriteResult]) -> PresetReport:
    """Gather what was written into the one object both renderings are drawn from."""
    return PresetReport(
        model_id=spec.model_id,
        name=spec.name,
        quant=spec.quant,
        directory=str(directory),
        endpoint=spec.endpoint,
        planned_context=spec.planned_context,
        reserve_mib=spec.reserve_mib,
        probe_free_memory=spec.probe_free_memory,
        rungs=list(spec.rungs),
        files=[FileOutcome(path=str(r.path), outcome=r.outcome) for r in results],
    )


@dataclass(frozen=True)
class WriteResult:
    """What became of one file.

    Attributes:
        path: Where it is, or where it would have been.
        outcome: What happened, in the words :data:`Outcome` defines.
    """

    path: Path
    outcome: Outcome

    @property
    def kept(self) -> bool:
        """Whether somebody's edits stopped this file from being written."""
        return self.outcome == "kept-your-edits"


def stamp_of(text: str) -> str:
    """The checksum a file carries, computed over the text with the stamp blanked out.

    Args:
        text: The file's contents, with :data:`PLACEHOLDER_STAMP` where the stamp goes.

    Returns:
        The hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_unedited(text: str) -> bool:
    r"""Whether a file on disk is still exactly what LlamaFit wrote.

    Args:
        text: The file as it was read, with its line endings already normalised to ``\\n``.

    Returns:
        True only when the file carries a stamp and that stamp matches everything else in
        it. A file with no stamp is reported as edited, which is the safe direction: the
        cost of being wrong is one message asking for ``--force``, and the cost of being
        wrong the other way is somebody's work.
    """
    match = _STAMP_RE.search(text)
    if match is None:
        return False
    blanked = text[: match.start(1)] + PLACEHOLDER_STAMP + text[match.end(1) :]
    return stamp_of(blanked) == match.group(1)


def render_files(spec: PresetSpec) -> tuple[PresetFile, ...]:
    """Every file a preset consists of, rendered and stamped.

    Args:
        spec: The plan to render.

    Returns:
        The launch script for this machine, the router section and the README, in the order
        a person should meet them.
    """
    return tuple(_stamped(file) for file in _unstamped(spec))


def write_files(
    files: Sequence[PresetFile], directory: Path, *, force: bool = False
) -> list[WriteResult]:
    """Write the rendered files, leaving alone any the user has changed.

    Args:
        files: What to write.
        directory: Where to write it; created if it is not there.
        force: Overwrite a file whose stamp says it was edited.

    Returns:
        One :class:`WriteResult` per file, in the order they were given.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return [_write_one(file, directory / file.name, force=force) for file in files]


def _write_one(file: PresetFile, path: Path, *, force: bool) -> WriteResult:
    """Write one file, deciding first whether it is allowed to be written."""
    existing = _read(path)
    if existing is None:
        _put(file, path)
        return WriteResult(path, "created")
    if is_unedited(existing):
        if existing == file.text:
            return WriteResult(path, "unchanged")
        _put(file, path)
        return WriteResult(path, "updated")
    if not force:
        return WriteResult(path, "kept-your-edits")
    _put(file, path)
    return WriteResult(path, "overwritten")


def _read(path: Path) -> str | None:
    r"""The file as text with ``\\n`` endings, or ``None`` when it is not there.

    Undecodable bytes are replaced rather than raising. A file that is not UTF-8 was
    written by something other than LlamaFit, its stamp will not match whatever it holds,
    and it will therefore be kept -- which is the right answer for a file nobody here
    understands.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def _put(file: PresetFile, path: Path) -> None:
    r"""Write one file with its own line endings, encoding and permissions.

    ``newline=""`` turns Python's own newline translation off, so the ``\\r\\n`` a batch
    file wants is the only one it gets and the ``\\n`` a shell script needs survives being
    written on Windows.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(file.text.replace("\n", file.newline))
    if file.executable:
        _make_executable(path)


def _make_executable(path: Path) -> None:
    """Set the execute bits, where the platform has any.

    Windows has no such bit and ``os.chmod`` there accepts only the read-only flag, so
    asking for more is not an error but is not anything either. A failure is swallowed on
    purpose: a script that is not executable can still be run with ``sh script.sh``, and a
    preset that refused to be written because of a permission bit would be a worse outcome
    than one that is written and needs ``chmod +x``.
    """
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:  # a file system with no execute bit is not a failure to report
        pass


def _stamped(file: PresetFile) -> PresetFile:
    """The same file with its placeholder stamp replaced by the real one."""
    stamp = stamp_of(file.text)
    return PresetFile(
        name=file.name,
        text=file.text.replace(PLACEHOLDER_STAMP, stamp),
        newline=file.newline,
        executable=file.executable,
    )


def _unstamped(spec: PresetSpec) -> list[PresetFile]:
    """The three files, rendered with the placeholder where each stamp will go."""
    stem = spec.script_stem
    windows = spec.os_name == "windows"
    script = (
        PresetFile(
            name=f"{stem}.cmd",
            text=render_cmd(spec, stamp=PLACEHOLDER_STAMP),
            newline="\r\n",
            executable=False,
        )
        if windows
        else PresetFile(
            name=f"{stem}.sh",
            text=render_sh(spec, stamp=PLACEHOLDER_STAMP),
            newline="\n",
            executable=True,
        )
    )
    return [
        script,
        PresetFile(
            name=f"models-{spec.model_id}.ini",
            text=render_ini_section(spec, stamp=PLACEHOLDER_STAMP),
            newline="\n",
            executable=False,
        ),
        PresetFile(
            name=f"README-{spec.model_id}.md",
            text=render_readme(spec, stamp=PLACEHOLDER_STAMP, script_name=script.name),
            newline="\n",
            executable=False,
        ),
    ]


def render_readme(spec: PresetSpec, *, stamp: str, script_name: str) -> str:
    """The snippet that says what was written, what the endpoints are, and what will change.

    Args:
        spec: The plan.
        stamp: The value for the stamp line, so this file is protected like the others.
        script_name: The launch script that was written beside it.

    Returns:
        Markdown, ending in a newline.

    Its job is the one the scripts cannot do: a table of the rungs. A person deciding
    whether to close a browser before starting a model wants to see what the next rung down
    costs, and a comment block inside a batch file is not where anybody looks that up.
    """
    lines = [
        f"# {spec.name} on this machine",
        "",
        f"Written by LlamaFit {spec.version} on {spec.generated.isoformat()}, "
        f"by `llamafit preset {spec.model_id}`.",
        "",
        f"- **Model** {spec.name}, quantised {spec.quant}",
        f"- **File** `{spec.model_path}`",
        f"- **Placement** {spec.mode}, planned at {spec.planned_context} tokens{spec.build_phrase}",
        f"- **Machine** {spec.card_sentence}",
        "",
        "## Start it",
        "",
        "```",
        _invocation(spec, script_name),
        "```",
        "",
        _start_sentence(spec),
        "",
        "## Endpoints",
        "",
        "| What | Where |",
        "| --- | --- |",
        f"| Web interface | {spec.endpoint}/ |",
        f"| Chat completions, OpenAI shape | POST {spec.endpoint}/v1/chat/completions |",
        f"| Completions | POST {spec.endpoint}/v1/completions |",
        f"| Models | {spec.endpoint}/v1/models |",
        f"| Health | {spec.endpoint}/health |",
        "",
        f"The alias to ask for is `{spec.model_id}`. The server binds loopback only, "
        "because llama-server has no authentication of any kind.",
        "",
        "## The context it will choose",
        "",
        *_ladder_table(spec),
        "",
        *_ladder_note(spec),
        "",
        "## When this machine changes",
        "",
        "| What changed | What the script does |",
        "| --- | --- |",
        "| Something else is holding the card | takes a lower rung, and says which |",
        "| The model file has moved | stops, with the path it looked in, and exits 3 |",
        "| llama.cpp has moved | uses the one on `PATH` if there is one, else exits 3 |",
        "| A different card | starts, and says the layer split was chosen for another |",
        "| llama.cpp has been upgraded and rejects a flag | reports the exit code and "
        "says to re-render |",
        "",
        f"Re-render after any of them with `llamafit preset {spec.model_id}`. "
        "It will not overwrite a file you have edited: each one carries a checksum of "
        "itself, and `--force` is what says to discard your changes anyway.",
        "",
        "## The command it runs",
        "",
        "```",
        *_flag_lines(spec),
        "```",
        "",
        f"<!-- {STAMP_MARKER} {stamp} -->",
    ]
    return "\n".join(lines) + "\n"


def _invocation(spec: PresetSpec, script_name: str) -> str:
    """How the script is run, in the shell of the machine it was written for."""
    return script_name if spec.os_name == "windows" else f"./{script_name}"


def _start_sentence(spec: PresetSpec) -> str:
    """One sentence on what happens at start, which differs when there is no ladder."""
    if spec.probe_free_memory:
        return (
            "Double-clicking it works too. It reads how much card memory is free at that "
            f"moment, keeps {spec.reserve_mib} MiB back, and takes the largest context in "
            "the table below that still fits."
        )
    return (
        "Double-clicking it works too. It starts at the planned "
        f"{spec.planned_context} tokens; see the note under the table for why it does not "
        "choose one for itself here."
    )


def _ladder_table(spec: PresetSpec) -> list[str]:
    """The rungs, with the free card memory each one needs before the reserve."""
    rows = [
        "| Context | The card needs | Free card memory to reach it |",
        "| ---: | ---: | ---: |",
    ]
    for rung in spec.rungs:
        needed = rung.vram_required_mib
        rows.append(f"| {rung.tokens} | {needed} MiB | {needed + spec.reserve_mib} MiB |")
    return rows


def _ladder_note(spec: PresetSpec) -> list[str]:
    """Why the table is a ladder and not a single number, or why it is not used here."""
    if spec.probe_free_memory:
        return [
            "The plan behind this table was made when the machine was quieter than it may "
            "be when you start it. A configuration that asks for more card memory than is "
            "free does not fail on an NVIDIA driver: it starts, pages the overflow into "
            "system memory, and runs at a fraction of its speed while the log looks "
            "healthy. Stepping down a rung is how that is avoided.",
            "",
            "Nothing above the planned "
            f"{spec.planned_context} is offered, even when a longer context would fit the "
            "card, because system memory and what you asked for went into that choice and "
            "the script can measure only the card.",
        ]
    return [
        "This ladder is not walked at start time on this machine: see the header of "
        f"`{spec.script_stem}` for which of the two reasons applies. The table is still "
        "worth having, because it says what a shorter context would buy you, and "
        "`LLAMAFIT_CONTEXT` sets one for a single run.",
    ]


def _flag_lines(spec: PresetSpec) -> list[str]:
    """The command as it will be run, with the chosen context shown as a variable."""
    words = ["llama-server"]
    words += [word.replace(CONTEXT_TOKEN, "<context>") for word in spec.flags]
    return [" ".join(words)]
