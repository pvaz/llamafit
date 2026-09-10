"""Preset specifications the renderers and the golden files are built from.

The Windows one is the real join: the bundled catalog, the real planner and the recorded
reference machine, so a golden file is a rendering of a plan somebody could have. The rest
are that specification with one thing changed, which is what keeps the difference between
two golden files down to the thing being tested.
"""

from __future__ import annotations

from datetime import date

from llamafit.models.plan import Needs
from llamafit.placement import LaunchOptions, render_flags
from llamafit.presets import PresetSpec, build_spec
from llamafit.services.plan import plan_report
from tests.fixtures.board import model_and_quant, report

GOLDEN_DATE = date(2026, 9, 10)
GOLDEN_VERSION = "1.2.3"
"""Pinned so a release does not rewrite every golden file for no reason."""

SERVER_PATH = "D:\\llama.cpp\\bin\\llama-server.exe"
MODEL_PATH = "D:\\Models\\Qwen3 Coder Next\\Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

AWKWARD_WINDOWS_PATH = "D:\\Models\\100% Mine & Yours!\\Qwen3 (v2).gguf"
"""A path holding every character that breaks a careless batch file.

A space, a per cent sign that ``cmd`` expands before it parses anything, an ampersand that
would end the command, an exclamation mark that a script using delayed expansion would eat,
and brackets. All of them are legal in a Windows path and all of them are ordinary.
"""

AWKWARD_POSIX_PATH = "/home/pv/models/it's 100% mine & more/Qwen3 $HOME (v2).gguf"
"""The same idea for a shell: an apostrophe, a dollar, an ampersand and a space."""


def windows_spec(model_id: str = "qwen3-coder-next", **changes: object) -> PresetSpec:
    """The reference machine's plan for one bundled model, frozen for rendering."""
    model, quant = model_and_quant(model_id)
    system = report()
    planned = plan_report(model, quant, system.host, needs=Needs(use_case=model.use_cases[0]))
    model_path = MODEL_PATH if model_id == "qwen3-coder-next" else planned.model_path
    options = LaunchOptions(model_path=model_path, projector_path=planned.projector_path, port=8080)
    spec = build_spec(
        model=model,
        quant=quant.name,
        placement=planned.placement,
        flags=render_flags(planned.placement, model, options),
        model_path=model_path,
        projector_path=planned.projector_path,
        host=system.host,
        server_path=SERVER_PATH,
        llamacpp_build=system.llamacpp.build,
        port=8080,
        generated=GOLDEN_DATE,
        version=GOLDEN_VERSION,
    )
    return spec.model_copy(update=changes) if changes else spec


def posix_spec(**changes: object) -> PresetSpec:
    """The same plan as it would be written on a Linux box with the same card."""
    base = windows_spec()
    update: dict[str, object] = {
        "os_name": "linux",
        "server_program": "llama-server",
        "server_path": "/opt/llama.cpp/bin/llama-server",
        "model_path": "/home/pv/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
    }
    update.update(changes)
    spec = base.model_copy(update=update)
    return spec.model_copy(update={"flags": _repath(spec, base.model_path)})


def cpu_spec(**changes: object) -> PresetSpec:
    """A plan with nothing on the card, where there is no ladder to walk."""
    return windows_spec(mode="cpu", probe_free_memory=False, **changes)


def _repath(spec: PresetSpec, old: str) -> tuple[str, ...]:
    """The flags with the model path swapped, so a copied spec stays self-consistent."""
    return tuple(spec.model_path if word == old else word for word in spec.flags)


def vision_spec(**changes: object) -> PresetSpec:
    """A plan that keeps a vision projector, so the ``--mmproj`` path is rendered too."""
    return windows_spec("qwen3.8-flash-next", **changes)
