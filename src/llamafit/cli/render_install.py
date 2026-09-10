# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Everything a person reads while something large is being fetched and written.

The ``install`` group's rule is that nobody starts something large and irreversible
without being told what it costs first, so most of what these commands do is *say* things:
a plan before a byte is written, a line saying what happened afterwards, and — when a run
was stopped — a sentence promising that what arrived is still there. Those sentences and
tables are here rather than in the commands for the reason :mod:`llamafit.cli.render` is
its own page: what a command decides and what a reader sees are two jobs, and only one of
them has to be right about a person's language.

Every function returns a renderable and prints nothing. The command chooses the console —
and, for ``--json``, chooses not to have one at all — which is what keeps a machine-
readable run from being the one that draws a table nobody is watching. The machine's own
side of the same facts is in :mod:`llamafit.cli.install_json`, deliberately not here: a
document a script parses must not go through the bidirectional marks and the localised
figures a person needs, and keeping the two in one module is how they would.

Every figure goes through :mod:`llamafit.units`, every identifier inside a sentence is
isolated, and every finished cell goes through :func:`~llamafit.i18n.for_display`, so a
model id, a repository name or a path reads correctly in a language written right to left.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from llamafit.download.history import DownloadRecord
from llamafit.download.install import InstallOutcome
from llamafit.download.plan import DiskCheck, DownloadPlan
from llamafit.i18n import _, for_display, isolate, ngettext
from llamafit.llamacpp.install import InstallPlan, InstallResult, uninstall_note
from llamafit.units import format_bytes

# --- llama.cpp itself ---------------------------------------------------------------------


def render_release_plan(plan: InstallPlan) -> Table:
    """The plan as the table a person reads before agreeing to it."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    build = plan.release.build
    table.add_row(
        _("Release"),
        Text(
            _("%(tag)s (build %(build)s)") % {"tag": plan.release.tag, "build": build}
            if build is not None
            else plan.release.tag
        ),
    )
    table.add_row(_("Backend"), Text(plan.backend))
    table.add_row(_("Archive"), Text(isolate(plan.asset.name)))
    for extra in plan.companions:
        table.add_row(_("Also"), Text(isolate(extra.name)))
    table.add_row(
        _("Download"),
        Text(
            _("%(size)s (%(done)s already on disk)")
            % {
                "size": format_bytes(plan.download_bytes),
                "done": format_bytes(plan.resume_bytes),
            }
            if plan.resume_bytes
            else format_bytes(plan.download_bytes)
        ),
    )
    table.add_row(
        _("Checksum"),
        Text(
            _("published, checked before anything is unpacked")
            if plan.verified
            else _("none published"),
            style="" if plan.verified else "yellow",
        ),
    )
    table.add_row(_("Install into"), Text(isolate(str(plan.bin_dir))))
    table.add_row(_("Directory"), Text(_ownership_line(plan), style=_ownership_style(plan)))
    for check in plan.space:
        table.add_row(
            _("Free space") if check.purpose == "install" else _("Free space (cache)"),
            Text(
                _("%(free)s on %(path)s, %(needed)s needed")
                % {
                    "free": format_bytes(check.free),
                    "path": isolate(str(check.path)),
                    "needed": format_bytes(check.needed),
                },
                style="" if check.enough else "red",
            ),
        )
    return table


def _ownership_line(plan: InstallPlan) -> str:
    """One sentence about what is already in the install directory."""
    target = plan.target
    if target.ownership == "absent":
        return _("will be created")
    if target.ownership == "empty":
        return _("exists and is empty")
    if target.ownership == "ours":
        marker = target.marker
        if marker is not None and marker.tag:
            return _("holds %(tag)s, installed by LlamaFit; it will be replaced") % {
                "tag": marker.tag
            }
        return _("installed by LlamaFit; it will be replaced")
    return target.detail or _("was not installed by LlamaFit")


def _ownership_style(plan: InstallPlan) -> str:
    return "red" if plan.target.ownership == "foreign" else ""


def render_release_result(result: InstallResult) -> Group:
    """What was installed, what was detected in it, and how to undo it.

    The undo is part of the report rather than something to go and look up, for the same
    reason the PATH offer prints its own: the change outlives the process that made it.
    """
    lines: list[RenderableType] = [
        Text(
            _("Installed llama.cpp %(tag)s into %(path)s.")
            % {"tag": result.marker.tag, "path": isolate(str(result.bin_dir))},
            style="green",
        )
    ]
    if result.replaced is not None and result.replaced.tag:
        lines.append(Text(_("Replaced %(tag)s.") % {"tag": result.replaced.tag}, style="dim"))
    lines.append(
        Text(
            _("Backends found in it: %(backends)s")
            % {"backends": ", ".join(result.backends) or _("none")}
        )
    )
    lines.extend(Text(warning, style="yellow") for warning in result.warnings)
    lines.append(Text(_("Check it with `llamafit doctor`."), style="dim"))
    lines.append(Text(uninstall_note(result.root), style="dim"))
    return Group(*lines)


# --- a model's weights --------------------------------------------------------------------


def _status_text(*, present: bool) -> str:
    """What a file's Status cell says: already here, or still to fetch."""
    return _("already here") if present else _("to fetch")


def render_download_files(plan: DownloadPlan) -> Table:
    """One row per file, with its size where the catalog knows it.

    The columns are few on purpose: this is read once, in a hurry, by somebody deciding
    whether to spend the next two hours on it. A shard's own size is not among the things
    the catalog knows, so that cell reads *unknown* and the honest total is on the line
    below the table.
    """
    table = Table(title=_("About to download"), title_justify="left", expand=False)
    table.add_column(_("File"), overflow="fold")
    table.add_column(_("Role"))
    table.add_column(_("Size"), justify="right")
    table.add_column(_("Status"))
    for file in plan.files:
        present = file.already_present()
        table.add_row(
            Text(for_display(isolate(file.name))),
            Text(for_display(isolate(file.role))),
            Text(for_display(isolate(format_bytes(file.expected_size)))),
            Text(for_display(_status_text(present=present)), style="green" if present else ""),
        )
    return table


def render_download_plan(plan: DownloadPlan, check: DiskCheck) -> Group:
    """What is about to happen and how big it is, for printing before a socket is opened."""
    pieces: list[RenderableType] = [
        Text(
            for_display(
                _("%(name)s %(quant)s from %(repo)s")
                % {
                    "name": isolate(plan.model_name),
                    "quant": isolate(plan.quant),
                    "repo": isolate(plan.repo),
                }
            ),
            style="bold",
        ),
        render_download_files(plan),
        Text(
            for_display(
                ngettext(
                    "%(count)d file, %(total)s in all, %(needed)s still to fetch.",
                    "%(count)d files, %(total)s in all, %(needed)s still to fetch.",
                    len(plan.files),
                )
                % {
                    "count": len(plan.files),
                    "total": isolate(format_bytes(plan.total_bytes)),
                    "needed": isolate(format_bytes(check.needed)),
                }
            )
        ),
        Text(
            for_display(
                _("Into %(path)s, which has %(free)s free; %(after)s would be left.")
                % {
                    "path": isolate(str(plan.directory)),
                    "free": isolate(format_bytes(check.free)),
                    "after": isolate(format_bytes(max(0, check.free - check.needed))),
                }
            )
        ),
    ]
    if plan.unverifiable:
        pieces.append(
            Text(
                for_display(
                    ngettext(
                        "The catalog holds no checksum for %(files)s.",
                        "The catalog holds no checksums for %(files)s.",
                        len(plan.unverifiable),
                    )
                    % {"files": isolate(", ".join(plan.unverifiable))}
                ),
                style="yellow",
            )
        )
    return Group(*pieces)


def render_already_here(plan: DownloadPlan) -> Text:
    """The line for a model whose files are all on disk already."""
    return Text(
        for_display(
            _("%(model)s %(quant)s is already here; nothing to do.")
            % {"model": isolate(plan.model_id), "quant": isolate(plan.quant)}
        )
    )


def render_download_stopped(plan: DownloadPlan) -> Text:
    """That a stopped run kept what it had, which is the whole point of saying it."""
    return Text(
        for_display(
            _(
                "Stopped. What arrived is kept; run `llamafit install model %(model)s` "
                "again to carry on from there."
            )
            % {"model": isolate(plan.model_id)}
        ),
        style="yellow",
    )


def render_download_done(outcome: InstallOutcome) -> Group:
    """The last two lines: what is on disk now, and where."""
    return Group(
        Text(
            for_display(
                ngettext(
                    "%(count)d file, %(size)s, checked against the catalog.",
                    "%(count)d files, %(size)s, checked against the catalog.",
                    len(outcome.files),
                )
                % {
                    "count": len(outcome.files),
                    "size": isolate(format_bytes(outcome.plan.total_bytes)),
                }
            ),
            style="green",
        ),
        Text(
            for_display(
                _("In %(path)s. Run `llamafit plan %(model)s` for the command line that runs it.")
                % {
                    "path": isolate(str(outcome.plan.directory)),
                    "model": isolate(outcome.plan.model_id),
                }
            )
        ),
    )


# --- what has already been fetched ---------------------------------------------------------


def render_history(records: Sequence[DownloadRecord]) -> Table:
    """Every download the other two commands recorded, newest first."""
    table = Table(title=_("Downloads"), title_justify="left")
    table.add_column(_("When"))
    table.add_column(_("Model"), overflow="fold")
    table.add_column(_("Quant"))
    table.add_column(_("Size"), justify="right")
    table.add_column(_("Outcome"))
    for record in records:
        table.add_row(
            Text(record.at.strftime("%Y-%m-%d %H:%M")),
            Text(for_display(isolate(record.model_id))),
            Text(for_display(isolate(record.quant))),
            Text(for_display(isolate(format_bytes(record.total_bytes)))),
            Text(
                for_display(_("finished") if record.finished else _("stopped")),
                style="green" if record.finished else "yellow",
            ),
        )
    return table


__all__ = [
    "render_already_here",
    "render_download_done",
    "render_download_files",
    "render_download_plan",
    "render_download_stopped",
    "render_history",
    "render_release_plan",
    "render_release_result",
]
