# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit hardware list|show|validate|path``: the hardware profile commands.

Section 13.1's row, four subcommands. ``list`` says what profiles LlamaFit can reach,
``show`` prints one, ``validate`` checks files the way ``catalog validate`` checks
catalog files, and ``path`` says where a profile of your own goes.

``show --as-host`` is the one addition, and it is the one that matters most. A profile
is only ever useful because it becomes a :class:`~llamafit.models.host.Host`, and this
prints that host: the same table ``llamafit system`` prints, from a file instead of from
the probes, opening with the line that says so. It is where a reader can see for
themselves that a simulated machine is marked as one before any figure is computed from
it, and where ``--json`` shows the ``simulated`` flag a script reads.

Every ``help=`` here is deferred for the reason ``app.py`` gives: a decorator runs while
the module is imported, before any language has been chosen.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.text import Text

from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_host, render_profile, render_profiles
from llamafit.hwprofile import (
    PROFILE_SUFFIX,
    bundled_profiles_dir,
    host_from_profile,
    load_profiles,
    profile_files,
    resolve_profile,
    user_profiles_dir,
    validate_files,
)
from llamafit.i18n import _, lazy_gettext, ngettext

hardware_app = typer.Typer(
    name="hardware",
    help=cast(
        str,
        lazy_gettext(
            "Hardware profiles: describe a machine in a file and score models against it."
        ),
    ),
)
app.add_typer(hardware_app, name="hardware")

# Module-level singletons rather than inline calls in a default position, for the reason
# `catalog_cmd` gives: ruff's B008 does not recognise every annotation shape as safe to
# call there, and a singleton sidesteps the warning without arguing with it.
_VALIDATE_FILE_ARGUMENT: Path | None = typer.Argument(
    None,
    help=cast(
        str,
        lazy_gettext("Validate only this file instead of every bundled and user profile."),
    ),
)
_PROFILE_ARGUMENT: str = typer.Argument(
    ...,
    metavar="NAME",
    help=cast(str, lazy_gettext("A profile name, or a path to a profile file.")),
)


def _warn_about(state: CliState, count: int) -> None:
    """Say once, on stderr, that some profile file has a problem worth looking at.

    The same shape ``catalog`` uses: browsing keeps working with whatever loaded, one
    line says there is something wrong, and ``hardware validate`` is where the detail
    lives. stderr because ``--json`` writes to stdout and a remark must not join it.
    """
    if not count:
        return
    stderr = Console(stderr=True, no_color=state.no_color, highlight=False)
    stderr.print(
        Text(
            ngettext(
                "%(count)d hardware profile problem found; run `llamafit hardware validate`"
                " for details.",
                "%(count)d hardware profile problems found; run `llamafit hardware validate`"
                " for details.",
                count,
            )
            % {"count": count},
            style="yellow",
        )
    )


@hardware_app.command(
    "list",
    help=cast(
        str,
        lazy_gettext("List the hardware profiles LlamaFit has: the bundled ones and your own."),
    ),
)
def list_command(ctx: typer.Context) -> None:
    """List the hardware profiles LlamaFit has: the bundled ones and your own."""
    state: CliState = ctx.obj
    profiles, problems = load_profiles()
    _warn_about(state, len(problems))
    if state.json_output:
        payload = [
            {
                "name": loaded.name,
                "bundled": loaded.bundled,
                "path": str(loaded.path),
                "profile": loaded.profile.model_dump(mode="json"),
            }
            for loaded in profiles
        ]
        typer.echo(json.dumps(payload, indent=2))
        return
    console = state.console
    if not profiles:
        console.print(
            _("No hardware profiles. Run `llamafit hardware path` to see where yours would go.")
        )
        return
    console.print(render_profiles(profiles))


@hardware_app.command(
    "show",
    help=cast(
        str,
        lazy_gettext(
            "Show one profile; with --as-host, the machine it stands in for.\n\n"
            "A host built from a profile is marked as simulated, in the table and in the "
            "JSON, so a what-if is never mistaken for a scan of this machine."
        ),
    ),
)
def show_command(
    ctx: typer.Context,
    name: str = _PROFILE_ARGUMENT,
    as_host: bool = typer.Option(
        False,
        "--as-host",
        help=cast(
            str,
            lazy_gettext("Show the host this profile substitutes, as `llamafit system` would."),
        ),
    ),
) -> None:
    """Show one profile; with --as-host, the machine it stands in for."""
    state: CliState = ctx.obj
    loaded, problems = resolve_profile(name)
    _warn_about(state, len(problems))
    if as_host:
        host = host_from_profile(loaded)
        if state.json_output:
            typer.echo(host.model_dump_json(indent=2))
        else:
            state.console.print(render_host(host))
        return
    if state.json_output:
        typer.echo(loaded.profile.model_dump_json(indent=2))
        return
    state.console.print(render_profile(loaded))


def _default_profile_paths() -> list[Path]:
    """Every bundled profile file, then every one of the user's."""
    return [*profile_files(bundled_profiles_dir()), *profile_files(user_profiles_dir())]


@hardware_app.command(
    "validate",
    help=cast(
        str,
        lazy_gettext(
            "Check hardware profile files against the schema, printing one line per problem."
        ),
    ),
)
def validate_command(
    ctx: typer.Context,
    file: Path | None = _VALIDATE_FILE_ARGUMENT,
) -> None:
    """Check hardware profile files against the schema, printing one line per problem."""
    state: CliState = ctx.obj
    paths = [file] if file is not None else _default_profile_paths()
    problems = validate_files(paths)
    if state.json_output:
        typer.echo(json.dumps([dataclasses.asdict(p) for p in problems], indent=2))
    else:
        console = state.console
        if not problems:
            console.print(
                ngettext(
                    "%(count)d file checked, no problems found.",
                    "%(count)d files checked, no problems found.",
                    len(paths),
                )
                % {"count": len(paths)}
            )
        for problem in problems:
            where = f" ({problem.profile})" if problem.profile else ""
            console.print(Text(f"{problem.file}: {problem.location}{where}: {problem.message}"))
    if problems:
        raise typer.Exit(code=1)


@hardware_app.command(
    "path",
    help=cast(str, lazy_gettext("Print the directory your own hardware profiles go in.")),
)
def path_command(ctx: typer.Context) -> None:
    """Print the directory your own hardware profiles go in."""
    state: CliState = ctx.obj
    user = user_profiles_dir()
    if state.json_output:
        typer.echo(
            json.dumps(
                {
                    "user": str(user),
                    "exists": user.is_dir(),
                    "bundled": str(bundled_profiles_dir()),
                    "suffix": PROFILE_SUFFIX,
                },
                indent=2,
            )
        )
        return
    console = state.console
    console.print(Text(str(user)))
    if not user.is_dir():
        console.print(
            Text(
                _("It does not exist yet; create it and put a %(suffix)s file in it.")
                % {"suffix": PROFILE_SUFFIX},
                style="dim",
            )
        )


__all__ = ["hardware_app", "list_command", "path_command", "show_command", "validate_command"]
