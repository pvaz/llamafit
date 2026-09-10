# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Launch presets: a plan turned into a file somebody can double-click.

Section 15.3 of the design. A plan is a command line a person has to paste; a preset is a
script they can run, and it is what makes a recommendation survive contact with a real
desktop -- because a plan is computed when the machine is idle and it runs when a browser
has taken a gigabyte of the card.

That is the whole idea, and everything here serves it. A configuration that asks for more
card memory than is free does not fail on an NVIDIA driver: it starts, pages silently into
system memory, and runs at a fraction of its speed for weeks before anyone thinks to
measure it. So the script does not carry a context; it carries the ladder the planner
produced -- each rung with what it costs the card -- reads how much is free at the moment
it runs, and takes the largest rung that fits. :mod:`llamafit.presets.spec` is that rule as
arithmetic, and the two renderers unroll the same arithmetic into their own dialect.

Five modules: :mod:`~llamafit.presets.spec` (what a script is told, and the one decision it
makes), :mod:`~llamafit.presets.windows_cmd` and :mod:`~llamafit.presets.posix_sh` (the two
dialects), :mod:`~llamafit.presets.models_ini` (a section for a router, which is the one
artifact that cannot run the ladder and says so), :mod:`~llamafit.presets.render` (files on
disk, and never over an edit somebody made) and :mod:`~llamafit.presets.launch` (running
one, waiting for ``/health``, stopping it again).

Two decisions about the generated files themselves are worth stating here, because they run
against habits the rest of this project holds firmly.

**The generated files are in English, and nothing in them goes through the translation
layer.** Everything LlamaFit *prints* about a preset is translated as usual: what was
written, what was kept, where the server is. The artifact is different in kind. A ``.cmd``
has no way to declare its encoding -- ``cmd`` reads it in the console's code page, and a
byte-order mark at the top is printed rather than skipped -- so a batch file whose comments
and messages were Japanese or Portuguese would be mojibake on the machine it was written
for. Beyond the encoding, these files outlive the session that made them: they are shared,
pasted into bug reports and read by whoever inherits the machine, and a script whose
language depends on what ``--language`` said one afternoon is a script nobody else can
read. What is not LlamaFit's choice is the user's own directory names, and a path that is
not ASCII makes :mod:`~llamafit.presets.windows_cmd` emit a ``chcp`` line for that reason
alone.

**The generated files carry a provenance header, not the copyright notice every source file
here carries.** A source module is part of LlamaFit and the AGPL notice belongs on it. A
launch script is output: it is the user's configuration, written to be edited, and stamping
somebody's own start script with this project's licence terms would be a claim over their
file that the licence never made. What the header carries instead is what a licence notice
is actually for here -- who wrote it, when, from what, and how to regenerate it -- plus the
checksum that stops LlamaFit overwriting it once they have made it theirs.
"""

from llamafit.presets.launch import (
    FakeLauncher,
    Launcher,
    LaunchOutcome,
    RunningPreset,
    Started,
    SubprocessLauncher,
    clear_record,
    is_answering,
    launch_preset,
    load_record,
    port_of_script,
    record_path,
    records_dir,
    save_record,
    script_command,
    stop_preset,
    wait_for_health,
)
from llamafit.presets.models_ini import render_ini_section
from llamafit.presets.posix_sh import render_sh, sh_quote
from llamafit.presets.render import (
    PresetFile,
    WriteResult,
    is_unedited,
    render_files,
    render_readme,
    stamp_of,
    write_files,
)
from llamafit.presets.spec import (
    CONTEXT_TOKEN,
    PresetSpec,
    Rung,
    build_spec,
    can_probe_free_memory,
    choose_context,
    group_flags,
    mib_ceiling,
)
from llamafit.presets.windows_cmd import cmd_argument, cmd_assignment, cmd_echo, render_cmd

__all__ = [
    "CONTEXT_TOKEN",
    "FakeLauncher",
    "LaunchOutcome",
    "Launcher",
    "PresetFile",
    "PresetSpec",
    "Rung",
    "RunningPreset",
    "Started",
    "SubprocessLauncher",
    "WriteResult",
    "build_spec",
    "can_probe_free_memory",
    "choose_context",
    "clear_record",
    "cmd_argument",
    "cmd_assignment",
    "cmd_echo",
    "group_flags",
    "is_answering",
    "is_unedited",
    "launch_preset",
    "load_record",
    "mib_ceiling",
    "port_of_script",
    "record_path",
    "records_dir",
    "render_cmd",
    "render_files",
    "render_ini_section",
    "render_readme",
    "render_sh",
    "save_record",
    "script_command",
    "sh_quote",
    "stamp_of",
    "stop_preset",
    "wait_for_health",
    "write_files",
]
