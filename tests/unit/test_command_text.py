"""A command copied from a plan preserves paths and literal llama.cpp patterns."""

from __future__ import annotations

import base64
import io
import json
import os
import shlex
import subprocess
import sys

import pytest
from rich.console import Console

from llamafit.cli.render_board import render_plan
from llamafit.services.plan import plan_report
from llamafit.tui.app import LlamaFitApp
from llamafit.tui.screens.plan import PlanPane
from tests.fixtures.board import model_and_quant
from tests.fixtures.budget_hosts import reference_host
from tests.fixtures.dashboard import ready

COMMAND = ["llama-server", "-m", "Models/O'Brien & $HOME [draft].gguf", "-ot", "ffn_.*=CPU"]
EXPECTED = (
    "& 'llama-server' '-m' 'Models/O''Brien & $HOME [draft].gguf' '-ot' 'ffn_.*=CPU'"
    if os.name == "nt"
    else shlex.join(COMMAND)
)


def test_the_rendered_plan_quotes_paths_and_literal_patterns() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host()).model_copy(update={"command": COMMAND})
    output = io.StringIO()
    Console(file=output, width=1000, color_system=None).print(render_plan(report))
    assert EXPECTED in output.getvalue()


@pytest.mark.asyncio
async def test_the_clipboard_preserves_paths_and_literal_patterns() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)):
        plan = app.query_one("#plan", PlanPane)
        plan.command = COMMAND.copy()
        plan.action_copy()
        assert app.clipboard == EXPECTED


def test_the_json_plan_keeps_argv_and_adds_the_copyable_command_and_shell() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host()).model_copy(update={"command": COMMAND})
    body = report.model_dump(mode="json")
    assert body["command"] == COMMAND
    assert body["command_text"] == EXPECTED
    assert body["command_shell"] == ("PowerShell" if os.name == "nt" else "sh")


def test_posix_quoting_round_trips_quotes_empty_arguments_and_metacharacters() -> None:
    from llamafit.command_text import render_command

    command = [*COMMAND, "", 'a "quoted" path', "$(echo unsafe);*.gguf", "line\nbreak"]
    assert shlex.split(render_command(command, shell="sh")) == command


def test_powershell_uses_literal_arguments_and_the_call_operator() -> None:
    from llamafit.command_text import render_command

    assert render_command(COMMAND, shell="PowerShell") == (
        "& 'llama-server' '-m' 'Models/O''Brien & $HOME [draft].gguf' '-ot' 'ffn_.*=CPU'"
    )


def test_an_absent_command_has_no_copyable_text() -> None:
    from llamafit.command_text import render_command

    assert render_command([]) == ""


def test_the_local_shell_passes_the_plan_arguments_to_a_native_program_unchanged() -> None:
    from llamafit.command_text import render_command

    # These are valid Windows file-name characters too. PowerShell 5.1 is sufficient;
    # no new shell or change to its native argument-passing preferences is required.
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host(), alias="My Model & Tools")
    arguments = [
        *report.command[1:],
        *COMMAND[1:],
        r"C:\Projectos Pessoais\Models\O'Brien; $(throw 'oops') & 100%!.gguf",
        "Model \u2018one\u2019 and \u2019two\u2019.gguf",
    ]
    command = [sys.executable, "-c", "import json,sys;print(json.dumps(sys.argv[1:]))", *arguments]
    script = render_command(command)
    if os.name == "nt":
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        invocation = ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    else:
        invocation = ["sh", "-c", script]
    result = subprocess.run(invocation, capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(result.stdout) == arguments
