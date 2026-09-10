# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Render a plan as one section of a ``models.ini``, for llama.cpp's router mode.

A router keeps several models behind one port and starts each one when a request first
asks for it. Section 15.3 asks for a section per model, and this writes it: the id as the
section name, the file, and the arguments the planner chose.

**This is the one artifact that cannot run the ladder, and it says so in its own comments.**
A router starts a model from a configuration file, in the background, with nobody watching;
there is no shell in between to read free memory first. So the context here is a number
fixed on the day it was written -- exactly the thing ``start-<id>`` exists to avoid -- and
the honest thing to do is put that in the file next to it rather than let a reader assume
the two behave alike. A person running a router on a card that is also driving a desktop
should use the smaller rung and know why; the comment names the ladder's bottom so the
choice can be made without going back to ``llamafit plan``.

The section is written on its own rather than a whole file, because a ``models.ini`` holds
every model a router serves and LlamaFit knows about one. Appending is the caller's
decision; :func:`llamafit.presets.render.render_files` writes it to
``models-<id>.ini`` for that reason.
"""

from __future__ import annotations

from llamafit.presets.spec import CONTEXT_TOKEN, PresetSpec
from llamafit.presets.windows_cmd import STAMP_MARKER


def render_ini_section(spec: PresetSpec, *, stamp: str) -> str:
    """One ``[model]`` section for a router's ``models.ini``.

    Args:
        spec: The plan to render.
        stamp: The value to write on the stamp line, so the overwrite check works on this
            file exactly as it does on the scripts.

    Returns:
        The section's text, ending in a newline.
    """
    arguments = " ".join(_arguments(spec))
    lines = [
        f"; {spec.name} ({spec.quant}), planned by LlamaFit {spec.version} "
        f"on {spec.generated.isoformat()}.",
        f"; Placement: {spec.mode}{spec.build_phrase}.",
        ";",
        "; The context below is fixed at "
        f"{spec.planned_context}, and that is a compromise this file",
        "; cannot avoid. A router starts a model from a configuration, with no shell in",
        "; between, so nothing here can read free card memory first the way",
        f"; {spec.script_stem} does. This number was true on "
        f"{spec.generated.isoformat()}; if the card also",
        "; drives a desktop, something else may be holding part of it by the time the",
        "; router starts this model, and the driver will page rather than refuse.",
        ";",
        f"; The smallest configuration this plan has is {spec.smallest_rung.tokens} tokens at "
        f"{spec.smallest_rung.vram_required_mib} MiB;",
        f"; the planned one needs {spec.rungs[0].vram_required_mib} MiB. Use the smaller"
        " one here if this card is shared.",
        ";",
        f"; {STAMP_MARKER} {stamp}",
        f"[{spec.model_id}]",
        f"model = {spec.model_path}",
    ]
    if spec.projector_path is not None:
        lines.append(f"mmproj = {spec.projector_path}")
    lines.append(f"args = {arguments}")
    return "\n".join(lines) + "\n"


def _arguments(spec: PresetSpec) -> list[str]:
    """The planner's flags, with the paths and the context filled in.

    Nothing is quoted: a value in an INI file runs to the end of the line, so a path with a
    space in it needs no quotes and would be harmed by them -- the quotes would arrive as
    part of the value. The two path flags are dropped instead, because ``model`` and
    ``mmproj`` are keys of their own and repeating them in ``args`` would hand llama.cpp
    the same file twice.
    """
    out: list[str] = []
    skip_next = False
    for word in spec.flags:
        if skip_next:
            skip_next = False
            continue
        if word in {"-m", "--mmproj"}:
            skip_next = True
            continue
        out.append(str(spec.planned_context) if word == CONTEXT_TOKEN else word)
    return out
