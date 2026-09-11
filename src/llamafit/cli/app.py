# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Typer application: global options, error rendering, command registration.

The language is chosen here and nowhere else, and it is chosen **before the command is
built**. Typer turns this application into Click objects when it is called, and Click
renders every option's ``help`` as it builds the option, so a language chosen any later
-- in the eager ``--language`` callback, say -- reaches the arguments' help, which Typer
renders when the screen is drawn, and nothing else on the screen. ``main`` therefore
reads ``--language`` from the raw arguments the way colour is read, and asks the
environment and the operating system when it is not there, before ``app`` is called.
The eager callback is still the authority on what Click parsed, and it is how a caller
that never goes through ``main`` -- the test runner, on every invocation -- chooses at
all; for those callers :class:`_Group` puts every deferred message back where Click
froze its rendering, so the order of building and choosing stops mattering.

Where the words are going is settled before the language is. A file or a pipe on Windows
takes the system code page, which cannot write Japanese and cannot write the direction
marks an Arabic table carries, and a text stream raises on the first such character.
``main`` makes both streams forgiving first, and tells the choice what the stream can
write, so a language the stream cannot carry is refused with a reason and a character
it cannot carry is shown as ``?`` and counted. :mod:`llamafit.i18n.encoding` argues both
halves; this module only has to do them in the right order, which is that one.

Section 13.1's five substitution flags -- ``--profile``, ``--memory``, ``--ram``,
``--cpu-cores`` and ``--max-context`` -- are global options here because they are
questions about the *subject* of every answer rather than about any one command. The
first four replace the machine; the fifth caps the context. Nothing in this module acts
on them: they are stored on :class:`CliState` as typed and
:mod:`llamafit.cli.common` turns them into a :class:`~llamafit.models.host.Host` and a
:class:`~llamafit.models.plan.Needs`, so a command that never reads a machine cannot
half-honour them.

Every ``help=`` string on this page is deferred. A decorator runs while the module is
imported, long before any language has been chosen, so an eager ``_()`` here would freeze
the whole interface in English and nothing would fail. ``cast(str, ...)`` is what says out
loud that Typer's ``str`` annotation is being taken at its word: Typer only ever renders
these, and rendering is exactly what a ``LazyString`` does.
"""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import typer
from rich.console import Console
from rich.text import Text
from typer.core import TyperGroup
from typer.models import ParameterInfo
from typer.utils import get_params_from_function

from llamafit import __version__
from llamafit.errors import LlamaFitError, NotInstalledError, PackagedDataError, ProbeError
from llamafit.i18n import (
    LazyString,
    _,
    encoding_of,
    lazy_gettext,
    ngettext,
    replacements,
    set_language,
    tolerate,
)
from llamafit.logging import setup_logging
from llamafit.paths import get_paths


class _Group(TyperGroup):
    """The command tree, with every deferred help message put back where it was frozen.

    Click's ``Option`` cleans its ``help`` with ``inspect.cleandoc`` as it is built, which
    renders a ``LazyString`` and keeps the result: a ``str``, in whatever language was
    installed at that moment. Typer does the same to a command's and a group's ``help``.
    Only ``typer.Argument`` keeps what it was given and renders it when the screen is
    drawn, which is why a help screen asked for in Portuguese used to come out with one
    line in Portuguese and every other line in English.

    ``main`` chooses the language before the tree is built, so a real run never meets
    this. This class is for every other way in -- a caller that builds first and chooses
    afterwards, which is what the test runner does on every invocation. Typer hands the
    finished tree here, still under the translator it was rendered with, so each frozen
    text can be matched back to the deferred message it came from exactly, and the
    message is put back in its place. From then on Click and Typer hold the
    ``LazyString`` where they held a ``str``, and read it when they draw, as they already
    do for an argument. Text that matches nothing -- the completion options Typer adds, a
    docstring used as help -- was never deferred and is left alone.
    """

    def __init__(self, **attrs: Any) -> None:
        super().__init__(**attrs)
        _put_back(self, _deferred_help(app))


def _deferred_help(root: typer.Typer) -> dict[str, LazyString]:
    """Every deferred help message in the Typer tree, keyed by the text Click froze it as.

    The key is what ``inspect.cleandoc`` makes of the message under the translator
    installed now, which is what Click and Typer stored moments ago under the same one.
    """
    deferred: dict[str, LazyString] = {}

    def note(candidate: object) -> None:
        if isinstance(candidate, LazyString):
            deferred[inspect.cleandoc(str(candidate))] = candidate

    def note_parameters(function: object) -> None:
        if not callable(function):
            return
        for meta in get_params_from_function(function).values():
            if isinstance(meta.default, ParameterInfo):
                note(meta.default.help)

    def walk(instance: typer.Typer) -> None:
        note(instance.info.help)
        if instance.registered_callback is not None:
            note(instance.registered_callback.help)
            note_parameters(instance.registered_callback.callback)
        for command in instance.registered_commands:
            note(command.help)
            note_parameters(command.callback)
        for group in instance.registered_groups:
            note(group.help)
            if group.typer_instance is not None:
                walk(group.typer_instance)

    walk(root)
    return deferred


def _put_back(command: Any, deferred: Mapping[str, LazyString]) -> None:
    """Replace each frozen rendering in a Click tree with the message it was rendered from.

    The tree is walked by shape rather than by type: Typer ships its own copy of Click,
    so its classes are not the ones ``click`` exports, and what matters here is only that
    a node has ``help``, ``params`` and perhaps ``commands``.
    """
    frozen = command.help
    if isinstance(frozen, str) and frozen in deferred:
        command.help = deferred[frozen]
    for parameter in command.params:
        frozen = getattr(parameter, "help", None)
        if isinstance(frozen, str) and frozen in deferred:
            parameter.help = deferred[frozen]
    for child in getattr(command, "commands", {}).values():
        _put_back(child, deferred)


app = typer.Typer(
    name="llamafit",
    cls=_Group,
    help=cast(
        str,
        lazy_gettext(
            "Find, size, install and verify open-weight LLMs for llama.cpp on your own machine."
        ),
    ),
    # The bare command is not a command, so nothing in the command list can describe it,
    # and a stranger who reads help first never finds the dashboard without this line.
    epilog=cast(
        str,
        lazy_gettext(
            "With no command, llamafit opens the terminal dashboard; in a pipe, or with "
            "--json, it prints what `llamafit recommend` would."
        ),
    ),
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)


@dataclass
class CliState:
    """Options shared by every command, stored in ``ctx.obj``.

    The five after ``no_color`` are section 13.1's substitution flags. They are kept here
    exactly as typed -- strings for the two sizes, which nothing in this module is allowed
    to parse -- and :mod:`llamafit.cli.common` turns them into a machine and a request.
    ``app.py`` cannot import ``common``, which imports ``CliState`` from here, and putting
    the arithmetic behind the import would be a cycle for the sake of five fields.
    """

    json_output: bool = False
    verbose: bool = False
    no_color: bool = False
    profile: str | None = None
    memory: str | None = None
    ram: str | None = None
    cpu_cores: int | None = None
    max_context: int | None = None

    @property
    def console(self) -> Console:
        """A console for normal output, respecting ``--no-color``."""
        return Console(no_color=self.no_color, highlight=False)

    @property
    def substituting(self) -> bool:
        """Whether any machine other than this one was asked for.

        ``--max-context`` is not one of these. It changes the question, not the machine,
        so a command that reads this to refuse a stand-in machine must not refuse it too.
        """
        return any(
            value is not None for value in (self.profile, self.memory, self.ram, self.cpu_cores)
        )


def _colours_off() -> bool:
    """Whether colour is unwanted, read from the raw arguments and the environment.

    The language notice and the error report are both written before, or instead of, the
    parsed options, so neither can ask a :class:`CliState` that does not exist yet.
    """
    return "--no-color" in sys.argv or bool(os.environ.get("NO_COLOR"))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llamafit {__version__}")
        raise typer.Exit()


_UNCHOSEN: Final = object()
"""What ``_chosen_by_main`` holds while no run of ``main`` is in progress."""

_chosen_by_main: object = _UNCHOSEN
"""The ``--language`` value ``main`` read from the raw arguments and has already spoken.

``None`` when ``main`` found no option and chose from the environment and the operating
system instead; :data:`_UNCHOSEN` outside a run of ``main``, which is where the test
runner always is.
"""


def _language_argument(arguments: Sequence[str]) -> str | None:
    """The value of ``--language`` in the raw arguments, before Click has parsed anything.

    Read the way ``_colours_off`` reads ``--no-color``, and for the same reason: it has to
    be known before the command is built, and Click builds the command before it parses.
    Both spellings Click accepts are read, and the last one wins, as it does for Click.
    Anything odd -- a value missing, the option after a subcommand -- is left for Click to
    complain about; this only has to agree with Click where Click agrees with itself.
    """
    found: str | None = None
    for index, argument in enumerate(arguments):
        if argument == "--language" and index + 1 < len(arguments):
            found = arguments[index + 1]
        elif argument.startswith("--language="):
            found = argument.partition("=")[2]
    return found


def _speak(requested: str | None) -> None:
    """Install the chosen language and say out loud when the request was not met exactly.

    A notice arrives on a request served by another region's catalog as well as on one
    LlamaFit cannot honour at all, so it does not mean the request was refused. It is the
    only place a reader is told why some of the wording looks foreign, and losing it is
    the failure this whole layer exists to prevent. The stream's encoding goes along with
    the request, so a language the stream cannot write is refused here too, with a
    sentence the stream can write.

    It goes to stderr as ``Text``: as ``Text`` because it quotes a catalog's own
    ``Language-Team`` header, which is text read from a file and would otherwise be
    parsed as Rich markup, and to stderr because ``--json`` writes machine-readable
    output to stdout that a remark must not join.
    """
    choice = set_language(requested, encoding=encoding_of(sys.stdout))
    if choice.notice:
        console = Console(stderr=True, no_color=_colours_off(), highlight=False)
        console.print(Text(choice.notice, style="yellow"))
        if choice.hint:
            console.print(Text(choice.hint, style="dim"))


def _language_callback(value: str | None) -> str | None:
    """Speak the language Click parsed, unless ``main`` has already spoken it.

    In a real run ``main`` has read the same arguments and chosen before the command was
    built, and there is nothing left to do but agree; choosing again would read the
    catalog a second time for the same answer. When Click's reading differs from the raw
    one, Click's is right and wins. When nobody has spoken -- the test runner never goes
    through ``main`` -- this is where the language is chosen, after the command is built,
    which :class:`_Group` makes harmless.
    """
    if value != _chosen_by_main:
        _speak(value)
    return value


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help=cast(str, lazy_gettext("Print machine-readable JSON instead of tables.")),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=cast(
            str,
            lazy_gettext("Write details to llamafit.log in the log directory and show tracebacks."),
        ),
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help=cast(str, lazy_gettext("Disable colours."))
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        metavar="NAME|FILE",
        help=cast(
            str,
            lazy_gettext(
                "Answer for the machine this hardware profile describes instead of this one. "
                "Nothing about this machine is probed, and every answer is marked as "
                "simulated. `llamafit hardware list` names the profiles you have."
            ),
        ),
    ),
    memory: str | None = typer.Option(
        None,
        "--memory",
        metavar="SIZE",
        help=cast(
            str,
            lazy_gettext(
                "Pretend the graphics card has this much memory, for example 24G. Everything "
                "else stays as scanned, and every answer is marked as simulated."
            ),
        ),
    ),
    ram: str | None = typer.Option(
        None,
        "--ram",
        metavar="SIZE",
        help=cast(
            str,
            lazy_gettext(
                "Pretend the machine has this much system memory, for example 128GiB. "
                "Every answer is marked as simulated."
            ),
        ),
    ),
    cpu_cores: int | None = typer.Option(
        None,
        "--cpu-cores",
        metavar="N",
        min=1,
        help=cast(
            str,
            lazy_gettext(
                "Pretend the processor has this many physical cores. Every answer is "
                "marked as simulated."
            ),
        ),
    ),
    max_context: int | None = typer.Option(
        None,
        "--max-context",
        metavar="N",
        min=1,
        help=cast(
            str,
            lazy_gettext(
                "Plan, report and score no context longer than this. It caps the model's "
                "own length; it does not describe a machine."
            ),
        ),
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        metavar="TAG",
        callback=_language_callback,
        is_eager=True,
        help=cast(
            str,
            lazy_gettext(
                "Speak this language, for example pt_PT. A language LlamaFit does not have "
                "falls back to English, and it says which ones it does have."
            ),
        ),
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help=cast(str, lazy_gettext("Print the version and exit.")),
    ),
) -> None:
    """LlamaFit command-line interface."""
    ctx.obj = CliState(
        json_output=json_output,
        verbose=verbose,
        no_color=no_color,
        profile=profile,
        memory=memory,
        ram=ram,
        cpu_cores=cpu_cores,
        max_context=max_context,
    )
    if ctx.invoked_subcommand is None:
        open_dashboard(ctx)


def _report_replacements(console: Console) -> None:
    """Say once, at the end, how many characters the output could not carry.

    A ``?`` where a letter was is honest only if the reader is told it stands for one.
    The sentence is translated, unlike the notice that refuses a whole language: the
    language it is in has just been judged writable enough to show, holes and all, and a
    hole in this sentence is the same bargain as a hole anywhere else on the screen. The
    encoding is named because it is the thing to look up, and the setting because it is
    the fix.
    """
    count = replacements()
    if not count:
        return
    encoding = encoding_of(sys.stdout) or _("the output's encoding")
    console.print(
        Text(
            ngettext(
                "%(count)d character could not be written in %(encoding)s and is shown as ?.",
                "%(count)d characters could not be written in %(encoding)s and are shown as ?.",
                count,
            )
            % {"count": count, "encoding": encoding},
            style="yellow",
        )
    )
    console.print(
        Text(
            _(
                "Set PYTHONUTF8=1 in the environment and run again: Python then writes "
                "UTF-8, which carries every language."
            ),
            style="dim",
        )
    )


def main() -> None:
    """Run the app, turning known errors into messages and unexpected ones into a short report.

    The order of the first three things is the point. The streams are made forgiving
    before anything is written, so that no report -- least of all the one saying
    something could not be written -- can raise while it is being printed. The language
    is chosen next, from the raw arguments, before ``app`` builds the command tree that
    renders every option's help. Only then does Click get to parse.

    Error text can contain anything, including square brackets from a path, so it is printed
    as ``Text`` and never parsed as Rich markup.
    """
    global _chosen_by_main
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        setup_logging(get_paths().log_dir, verbose=True)
    tolerate(sys.stdout)
    tolerate(sys.stderr)
    console = Console(stderr=True, no_color=_colours_off(), highlight=False)
    try:
        requested = _language_argument(sys.argv[1:])
        _chosen_by_main = requested
        _speak(requested)
        app(standalone_mode=True)
    except LlamaFitError as exc:
        console.print(Text(exc.render(), style="red"))
        # 1 is a user or configuration error, 2 an environment problem: llama.cpp or a
        # tool missing, a probe that could not run, or an installation missing the data
        # that shipped inside it. A script has to be able to tell "fix your file" from
        # "fix your machine"; docs/cli.md documents both.
        environment = (NotInstalledError, PackagedDataError, ProbeError)
        sys.exit(2 if isinstance(exc, environment) else 1)
    except Exception as exc:  # an unexpected failure is a bug, not a user error
        if verbose:
            raise
        console.print(Text(_("Unexpected error: %(error)s") % {"error": exc}, style="red"))
        console.print(Text(_("Run again with --verbose for the full traceback."), style="dim"))
        sys.exit(1)
    finally:
        _chosen_by_main = _UNCHOSEN
        _report_replacements(console)


from llamafit.cli import (  # noqa: E402  (registers commands on import)
    bench_cmd,
    board_cmd,
    catalog_cmd,
    doctor_cmd,
    hardware_cmd,
    install_cmd,
    llamacpp_cmd,
    plan_cmd,
    preset_cmd,
    serve_cmd,
    system_cmd,
)
from llamafit.tui.entry import open_dashboard  # noqa: E402  (`llamafit` with no arguments)

__all__ = [
    "CliState",
    "app",
    "bench_cmd",
    "board_cmd",
    "catalog_cmd",
    "doctor_cmd",
    "hardware_cmd",
    "install_cmd",
    "llamacpp_cmd",
    "main",
    "open_dashboard",
    "plan_cmd",
    "preset_cmd",
    "serve_cmd",
    "system_cmd",
]
